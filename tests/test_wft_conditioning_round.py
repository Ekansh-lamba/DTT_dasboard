"""WFT-scope conditioning fixes: sanitization, legacy filter, dead channels, Gxy.

Scope is WFT force/moment channels only. Nothing here touches Latacc/FiltLP,
the IMU channels, or GPS -- those are a later round.

Each test pins a behaviour that was wrong in a way that produced plausible
output rather than an error, which is the only reason these were survivable:

  * outliers were counted in the report but left in the data
  * NaN gaps were forward-filled, planting a flat plateau under smo(0.1)
  * the legacy Butterworth ran forward-backward, moving every WFT peak
  * a WFT channel dead end to end was reported as if it had been conditioned
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from dtt.analysis.severity import g_severity
from dtt.config import RunConfig
from dtt.preprocessing import (apply_famos_recipe, count_conditioned,
                               famos_recipe, is_wft_channel)
from dtt.processing.signal_processor import _butterworth_lpf
from dtt.sanitization.sanitizer import _fill_nan_gaps, _flag_outliers, sanitize

FS = 1000.0


# ------------------------------------------------- outlier blanking (defect 1)

def test_outliers_are_blanked_not_only_counted():
    """The report and the frame must agree: counted means removed."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"Time": np.arange(100) / 100.0,
                       "FL_Fz": rng.normal(6000, 300, 100)})
    df.loc[50, "FL_Fz"] = 999_999.0

    out, report = sanitize(df, RunConfig())

    assert np.isnan(out["FL_Fz"].iloc[50]), "the artifact is still in the data"
    assert report.outlier_flags_per_channel.get("FL_Fz", 0) >= 1


def test_flag_outliers_returns_the_frame_it_modified():
    """Returning counts alone let the caller drop the blanking on the floor."""
    df = pd.DataFrame({"FL_Fz": np.r_[np.full(98, 6000.0), 1e9, -1e9]})
    out, counts = _flag_outliers(df, ["FL_Fz"])
    assert isinstance(out, pd.DataFrame)
    assert counts["FL_Fz"] == int(out["FL_Fz"].isna().sum())
    assert df["FL_Fz"].notna().all(), "input frame must not be mutated in place"


def test_outlier_blanking_reaches_every_wft_force_channel():
    cols = {f"{p}_{c}": np.full(200, 5000.0)
            for p in ("FL", "FR", "RL", "RR") for c in ("Fx", "Fy", "Fz")}
    for v in cols.values():
        v[10], v[20] = 9e5, -9e5
    df = pd.DataFrame({"Time": np.arange(200) / FS, **cols})
    out, report = sanitize(df, RunConfig())
    for ch in cols:
        assert np.isnan(out[ch].iloc[10]) and np.isnan(out[ch].iloc[20]), ch
        assert ch in report.outlier_flags_per_channel


# --------------------------------------------------- NaN gap fill (defect 3)

def test_short_gap_is_interpolated_linearly_not_held_flat():
    """ffill plants a plateau that smo(0.1) then smears into real signal."""
    truth = np.linspace(100.0, 200.0, 50)
    vals = truth.copy()
    vals[20:25] = np.nan
    out = _fill_nan_gaps(pd.DataFrame({"FL_Fz": vals}), ["FL_Fz"], max_gap=10)

    assert out["FL_Fz"].iloc[20:25].notna().all()
    # linear interpolation recovers a linear signal exactly; ffill would hold
    # vals[19] flat across the whole gap
    np.testing.assert_allclose(out["FL_Fz"].iloc[20:25], truth[20:25])
    assert abs(out["FL_Fz"].iloc[22] - vals[19]) > 0.1


@pytest.mark.parametrize("gap_len,filled", [(9, True), (10, True),
                                            (11, False), (20, False)])
def test_max_gap_is_a_total_run_length_not_a_per_direction_limit(gap_len, filled):
    """A limit counted per direction closes gaps of up to 2x max_gap.

    pandas applies `limit` separately to each fill direction, so a 15-sample
    run with limit=10 and limit_direction="both" gets 10 filled from the left
    and 10 from the right and vanishes -- which is not what "leave gaps longer
    than max_gap alone" means.
    """
    vals = np.linspace(0.0, 100.0, 60)
    vals[25:25 + gap_len] = np.nan
    out = _fill_nan_gaps(pd.DataFrame({"FL_Fz": vals}), ["FL_Fz"], max_gap=10)
    gap = out["FL_Fz"].iloc[25:25 + gap_len]
    if filled:
        assert bool(gap.notna().all()), f"gap of {gap_len} should be filled"
    else:
        assert bool(gap.isna().all()), f"gap of {gap_len} should stay NaN"


def test_leading_and_trailing_nan_are_left_alone():
    """There is nothing to interpolate between at an edge -- filling there
    would extrapolate, which is the flat-hold problem again."""
    vals = np.linspace(0.0, 100.0, 40)
    vals[:5] = np.nan
    vals[-5:] = np.nan
    out = _fill_nan_gaps(pd.DataFrame({"FL_Fz": vals}), ["FL_Fz"], max_gap=10)
    assert out["FL_Fz"].iloc[:5].isna().all()
    assert out["FL_Fz"].iloc[-5:].isna().all()


# ------------------------------------------------ legacy Butterworth (defect 2)

def test_legacy_butterworth_is_not_forward_backward():
    src = pathlib.Path("dtt/processing/signal_processor.py").read_text(
        encoding="utf-8")
    assert "filtfilt" not in src


def test_legacy_butterworth_starts_settled():
    """Step-response init: y[0] == x[0], no startup ramp from zero."""
    x = np.full(1000, 3.7)
    y = _butterworth_lpf(x, order=4, cutoff=10.0, sr=FS)
    assert abs(y[0] - x[0]) < 1e-9
    np.testing.assert_allclose(y, x, atol=1e-9)


def test_legacy_butterworth_is_causal():
    """No response before the impulse. A zero-phase pass would ring ahead of it,
    which is what moved WFT peaks off the events they belong to."""
    x = np.zeros(600)
    x[300] = 1.0
    y = _butterworth_lpf(x, order=4, cutoff=10.0, sr=FS)
    assert np.allclose(y[:300], 0.0, atol=1e-12), "filter responded before the input"
    assert np.abs(y[300:]).max() > 0.0


# ------------------------------------------------- dead WFT channel (defect 4)

def _frozen_channel(dead_from_s: float, total_s: float = 180.0) -> pd.DataFrame:
    n = int(total_s * FS)
    rng = np.random.default_rng(1)
    y = rng.normal(7000, 800, n)
    y[int(dead_from_s * FS):] = 0.0
    return pd.DataFrame({"Time": np.arange(n) / FS, "RR_Fz": y})


def test_blank_dropout_is_counted_as_conditioning():
    df = _frozen_channel(174.0)
    _, _, applied = apply_famos_recipe(df, FS, decimate_factor=10)
    assert "blank dropout(" in applied["RR_Fz"]
    assert count_conditioned(applied) == 1


def test_fully_dead_channel_is_named_and_skipped():
    """All-NaN through smo/red is silent; the report must say DEAD instead."""
    n = int(180 * FS)
    df = pd.DataFrame({"Time": np.arange(n) / FS, "RR_Fz": np.zeros(n)})
    out, _, applied = apply_famos_recipe(df, FS, decimate_factor=10)

    assert "DEAD" in applied["RR_Fz"]
    assert count_conditioned(applied) == 1, "a dead channel was changed, so it counts"
    assert len(out["RR_Fz"]) == int(np.ceil(n / 10)), "must stay aligned"
    assert not np.isfinite(out["RR_Fz"]).any()


# --------------------------------------------------- channel dispatch (defect 5)

WFT_NAMES = (
    [f"{p}_{c}_{s}" for p, s in (("FR", "2"), ("RR", "1"))
     for c in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")]
    + [f"WFT_{c}_{corner}" for corner in ("fl", "fr", "rl", "rr")
       for c in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")]
)
NON_WFT_NAMES = (
    [f"WFT_RPM_{c}" for c in ("fl", "fr", "rl", "rr")]
    + [f"WFT_Angle_{c}" for c in ("fl", "fr", "rl", "rr")]
    + ["FR_Angle_2", "FR_Anglespeed_2", "RR_Angle_1", "RR_Anglespeed_1",
       "WFT_rot_fx_fl", "WFT_rot_fx_fr", "WFT_rot_fx_rl",
       "WFT_MxCorr_fl", "WFT_MxCorr_fr"]
)


@pytest.mark.parametrize("name", WFT_NAMES)
def test_wft_names_get_the_force_recipe(name):
    assert is_wft_channel(name)
    assert famos_recipe(name).smooth_width_s == 0.1
    assert famos_recipe(name).apply_filter is False, "WFT channels get no FiltLP"


@pytest.mark.parametrize("name", NON_WFT_NAMES)
def test_non_force_channels_are_not_smoothed(name):
    """RPM, angle, rotated-Fx and corrected-Mx are passthrough.

    WFT_MxCorr_* is a derived, already-corrected quantity rather than a raw
    load channel, so the recipe's smo(0.1) does not apply to it; it rides
    red(10) with everything else.
    """
    assert not is_wft_channel(name)
    assert famos_recipe(name).smooth_width_s == 0.0


# ------------------------------------------------------------ Gxy (defect 6)

def test_g_severity_known_answer():
    gx, gy, gxy = g_severity(np.full(1000, 100.0), np.full(1000, 200.0),
                             np.full(1000, 5000.0))
    assert gx == pytest.approx(100.0 / 5000.0, abs=1e-12)
    assert gy == pytest.approx(200.0 / 5000.0, abs=1e-12)
    assert gxy == pytest.approx(np.hypot(100.0, 200.0) / 5000.0, abs=1e-12)


def test_g_severity_matches_the_imc_block_literally():
    """Transcription check against imc lines 105-119 on a varying signal."""
    rng = np.random.default_rng(7)
    n = 200_000
    fx = rng.normal(300, 900, n)
    fy = rng.normal(-100, 700, n)
    fz = np.abs(rng.normal(6000, 1200, n)) + 50.0

    gx, gy, gxy = g_severity(fx, fy, fz)

    assert gx == pytest.approx(np.sqrt(np.mean((fx / fz) ** 2)), rel=1e-12)
    assert gy == pytest.approx(np.sqrt(np.mean((fy / fz) ** 2)), rel=1e-12)
    assert gxy == pytest.approx(
        np.sqrt(np.mean(fx ** 2 + fy ** 2) / np.mean(fz) ** 2), rel=1e-12)


def test_gxy_averages_fz_before_squaring_it():
    """Moving Fz inside the mean is a different number, not a rounding detail."""
    rng = np.random.default_rng(7)
    n = 200_000
    fx = rng.normal(300, 900, n)
    fy = rng.normal(-100, 700, n)
    fz = np.abs(rng.normal(6000, 1200, n)) + 50.0

    _, _, gxy = g_severity(fx, fy, fz)
    per_sample = np.sqrt(np.mean((fx ** 2 + fy ** 2) / fz ** 2))
    assert abs(per_sample - gxy) / gxy > 0.01
