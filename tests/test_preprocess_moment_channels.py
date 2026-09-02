"""Regression tests for the moment-channel (``f*_m*``) preprocessing path.

The bug these pin down: WFT **moment** channels (``FR_Mx_2``, ``RR_Mz_1``, …)
were conditioned correctly and written to ``processed_data.csv``, but no
unconditioned copy of them was ever kept. ``raw_data.csv`` held the Fx/Fy/Fz
forces alone, so the Preprocess screen had nothing to compare a moment against
and drew a single trace -- in the raw colour -- which reads exactly like a
channel that was never preprocessed at all.

So the tests below deliberately do not stop at "``smo`` runs". They check that
the raw *reference* survives to the place the GUI reads it from, because that
is the link that was broken:

  * the names in the real dataset parse as moment channels
  * the recipe conditions moments, not just forces
  * the raw snapshot keeps every conditioned channel
  * ``famos_stages`` agrees with ``apply_famos_recipe`` sample-for-sample
  * ``red`` actually reduces the sample count
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dtt.channels import parse_channel
from dtt.preprocessing import (STAGE_FILTLP, STAGE_FINAL, STAGE_RAW, STAGE_SMO,
                               apply_famos_recipe, conditions_channel,
                               famos_recipe, famos_stages, is_wft_channel,
                               log_preprocess_trace)

FS = 1000.0
RED = 10

# Exactly as they appear in "raw data in .dat format/2006-08-25 09-10-46 (1)".
MOMENTS = ["FR_Mx_2", "FR_My_2", "FR_Mz_2", "RR_Mx_1", "RR_My_1", "RR_Mz_1"]
FORCES = ["FR_Fx_2", "FR_Fy_2", "FR_Fz_2", "RR_Fx_1", "RR_Fy_1", "RR_Fz_1"]
PASSTHROUGH = ["Latitude", "Longitude", "Yawrate", "Distance", "FR_Angle_2"]


def signal(n: int = 20_000, seed: int = 7) -> np.ndarray:
    """Broadband load-like trace: something ``smo(0.1)`` visibly changes."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    return (500.0 * np.sin(2 * np.pi * 1.7 * t)
            + 60.0 * np.sin(2 * np.pi * 47.0 * t)
            + 25.0 * rng.standard_normal(n))


# ----------------------------------------------------- 1. discovery / mapping

@pytest.mark.parametrize("name", MOMENTS)
def test_moment_names_parse_to_a_position_and_component(name):
    """``f*_m*`` files must resolve; returning None is what skips a channel."""
    parsed = parse_channel(name)
    assert parsed is not None, f"{name} did not parse -- it would be skipped"
    pos, comp = parsed
    assert comp in ("Mx", "My", "Mz")
    assert pos.id in ("A1R", "A2R")


@pytest.mark.parametrize("name", MOMENTS + FORCES)
def test_moments_and_forces_are_both_wft_channels(name):
    assert is_wft_channel(name)


# ------------------------------------------------- 2. the recipe applies smo

@pytest.mark.parametrize("name", MOMENTS)
def test_recipe_smooths_moments_at_the_force_width(name):
    r = famos_recipe(name)
    assert r.smooth_width_s == pytest.approx(0.1)
    assert not r.apply_filter, "moments get smo, never FiltLP"


@pytest.mark.parametrize("name", MOMENTS + FORCES + ["Latacc", "Vehicle_Speed"])
def test_conditioned_channels_are_flagged_for_raw_capture(name):
    assert conditions_channel(name)


@pytest.mark.parametrize("name", PASSTHROUGH)
def test_passthrough_channels_are_not_flagged(name):
    """A channel the recipe does not touch has no before/after to keep."""
    assert not conditions_channel(name)


# ------------------------------- 3. preprocessing changes the data, non-empty

@pytest.mark.parametrize("name", MOMENTS)
def test_moment_channel_is_actually_changed_and_decimated(name):
    x = signal()
    df = pd.DataFrame({"Time": np.arange(x.size) / FS, name: x})
    out, new_fs, applied = apply_famos_recipe(df, FS, decimate_factor=RED)

    assert f"smo(0.1s)" in applied[name] and f"red({RED})" in applied[name]
    y = out[name].to_numpy(float)
    assert y.size > 0, "final output is empty"
    assert np.isfinite(y).any(), "final output is all-NaN"
    assert new_fs == pytest.approx(FS / RED)
    # red() reduces the sample count by the factor
    assert y.size == pytest.approx(x.size / RED, rel=0.01)
    # smo genuinely removed the 47 Hz content rather than passing data through
    assert np.std(y) < np.std(x[::RED])


def test_moment_and_force_are_conditioned_the_same_way():
    """Nothing in the recipe treats a moment differently from a force."""
    x = signal()
    df = pd.DataFrame({"Time": np.arange(x.size) / FS,
                       "FR_Fx_2": x, "FR_Mx_2": x})
    out, _, applied = apply_famos_recipe(df, FS, decimate_factor=RED)
    assert applied["FR_Fx_2"] == applied["FR_Mx_2"]
    np.testing.assert_allclose(out["FR_Fx_2"], out["FR_Mx_2"])


# ------------------------------------------------- 4. raw reference survives

def _config_with_channels(cols):
    from dtt.run_channels import build_run_channels

    class _Cfg:
        pass
    cfg = _Cfg()
    cfg.run_channels = build_run_channels(cols)
    return cfg


def test_raw_reference_frame_keeps_moments():
    """The regression: raw_data.csv must carry the moments, not forces only.

    Without this the GUI reads ``raw_data.csv``, finds no ``FR_Mx_2`` column,
    and falls back to drawing the processed trace alone.
    """
    from dtt.pipeline import _raw_reference_frame

    cols = ["Time"] + FORCES + MOMENTS + PASSTHROUGH
    df = pd.DataFrame({c: np.zeros(50) for c in cols})
    frame = _raw_reference_frame(df, _config_with_channels(cols))

    assert frame is not None
    for m in MOMENTS:
        assert m in frame.columns, f"{m} missing from the raw reference frame"
    for f in FORCES:
        assert f in frame.columns
    assert "Time" in frame.columns
    # passthrough channels have no before/after and are not stored
    for p in PASSTHROUGH:
        assert p not in frame.columns


def test_raw_reference_frame_falls_back_when_nothing_matches():
    cols = ["Time", "mystery_a", "mystery_b"]
    df = pd.DataFrame({c: np.zeros(10) for c in cols})
    assert _raw_reference_frame_or_none(df, cols) is None


def _raw_reference_frame_or_none(df, cols):
    from dtt.pipeline import _raw_reference_frame
    return _raw_reference_frame(df, _config_with_channels(cols))


# --------------------------------------------- 5. stages match the real thing

@pytest.mark.parametrize("name", ["FR_Mx_2", "FR_Fx_2", "Latacc",
                                  "Vehicle_Speed", "Latitude"])
def test_famos_stages_final_equals_apply_famos_recipe(name):
    """The debugging view must not be a second, drifting implementation."""
    x = signal()
    df = pd.DataFrame({"Time": np.arange(x.size) / FS, name: x})
    out, new_fs, _ = apply_famos_recipe(df, FS, decimate_factor=RED,
                                        emit_lpf_columns=False)
    stages = famos_stages(x, FS, name, decimate_factor=RED)

    final = stages[-1]
    assert final.name == STAGE_FINAL
    assert final.fs == pytest.approx(new_fs)
    n = min(final.n_samples, len(out))
    np.testing.assert_allclose(final.y[:n], out[name].to_numpy(float)[:n],
                               rtol=0, atol=0)


def test_stages_expose_the_intermediates_for_a_moment_channel():
    stages = famos_stages(signal(), FS, "FR_Mx_2", decimate_factor=RED)
    names = [s.name for s in stages]
    assert names[0] == STAGE_RAW
    assert STAGE_SMO in names
    assert names[-1] == STAGE_FINAL
    # the SMO stage is still at the native rate; only the final one is decimated
    smo = stages[names.index(STAGE_SMO)]
    assert smo.fs == pytest.approx(FS)
    assert stages[-1].fs == pytest.approx(FS / RED)
    # and the raw stage is genuinely different from the final one
    assert not np.allclose(stages[0].y[::RED][:10], stages[-1].y[:10])


def test_latacc_stages_include_filtlp_before_smo():
    stages = famos_stages(signal(), FS, "Latacc", decimate_factor=RED)
    names = [s.name for s in stages]
    assert names.index(STAGE_FILTLP) < names.index(STAGE_SMO)


def test_passthrough_channel_reports_no_conditioning():
    stages = famos_stages(signal(), FS, "Latitude", decimate_factor=RED)
    assert [s.name for s in stages] == [STAGE_RAW, STAGE_FINAL]
    assert "red(10)" in stages[-1].detail


# ------------------------------------------------------ 6. the trace is loud

def test_trace_names_the_channel_and_says_smo_ran():
    stages = famos_stages(signal(), FS, "FR_Mx_2", decimate_factor=RED)
    text = log_preprocess_trace("FR_Mx_2", stages, source="FR_Mx_2.raw")
    assert "[PREPROCESS]" in text
    assert "FR_Mx_2" in text
    assert "SMO applied : YES" in text
    assert "FiltLP      : NO" in text
    assert "Final output: VALID" in text


def test_trace_reports_empty_rather_than_staying_silent():
    stages = famos_stages(np.array([]), FS, "FR_Mx_2", decimate_factor=RED)
    text = log_preprocess_trace("FR_Mx_2", stages)
    assert "Final output: EMPTY" in text


def test_trace_reports_all_nan_as_error():
    stages = famos_stages(np.full(5000, np.nan), FS, "FR_Mx_2",
                          decimate_factor=RED)
    text = log_preprocess_trace("FR_Mx_2", stages)
    assert "ERROR (all-NaN)" in text


# ------------------------------------- 7. widening the frame must not rescale

def test_raw_csv_scales_forces_only(tmp_path):
    """The decade/daN divisor applies to Fx/Fy/Fz and nothing else.

    ``_write_raw_csv`` used to divide every column it was handed, which was
    correct only because it was handed forces alone. Once the frame widened to
    every conditioned channel, a blanket divide would have written the moments
    and ``Latacc``/``Vehicle_Speed`` 10-100x low and then drawn them against
    their correctly-scaled processed traces -- a far more convincing error than
    the missing channel it replaced.
    """
    from dtt.pipeline import _write_raw_csv

    n = 20
    frame = pd.DataFrame({
        "Time": np.arange(n) / 100.0,
        "FR_Fx_2": np.full(n, 1000.0),      # force  -> divided
        "FR_Mx_2": np.full(n, 1000.0),      # moment -> untouched
        "Latacc": np.full(n, 1000.0),       # aux    -> untouched
        "Vehicle_Speed": np.full(n, 1000.0),
    })

    class _Cfg:
        run_output_dir = tmp_path
    _write_raw_csv(frame, _Cfg(), {"n_to_dan_applied": True},
                   {"force_scale_divisor": 10.0})

    out = pd.read_csv(tmp_path / "raw_data.csv")
    assert out["FR_Fx_2"].iloc[0] == pytest.approx(10.0)     # 1000 / (10 * 10)
    assert out["FR_Mx_2"].iloc[0] == pytest.approx(1000.0)
    assert out["Latacc"].iloc[0] == pytest.approx(1000.0)
    assert out["Vehicle_Speed"].iloc[0] == pytest.approx(1000.0)
    assert out["Time"].iloc[1] == pytest.approx(0.01)
