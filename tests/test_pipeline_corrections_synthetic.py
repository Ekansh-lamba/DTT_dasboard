"""Synthetic arithmetic checks for three pipeline corrections that
``tests/test_wft_conditioning_round.py`` does not cover: the N->daN unit
heuristic, the preset-anchored decade correction, and stop removal.

These are **not FAMOS-parity evidence**. FAMOS does not perform any of these
three corrections -- MODULE1_STATUS.md is explicit that ``processed_data.csv``
is the FAMOS chain *plus* corrections FAMOS itself never makes, and diffing
against a real FAMOS export is expected to show large differences here, not
agreement. What these tests verify is only that each correction's own
documented arithmetic does what its docstring claims, using small in-memory
fixtures -- no real recording, no golden corpus, no licensed FAMOS involved.

Two despike-rule boundary tests are included at the end because
``detect_rail``/``detect_dropout`` had no direct unit test anywhere in the
repo (confirmed by grep before writing this file).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dtt.config import RunConfig
from dtt.ingestion.loader import _apply_n_to_dan, normalise_force_units
from dtt.preprocessing import detect_dropout, detect_rail, remove_stops_frame
from dtt.run_channels import build_run_channels
from dtt.sanitization.sanitizer import sanitize
from dtt.vehicles import TyreParams


# ------------------------------------------------------- 1. _apply_n_to_dan

def test_apply_n_to_dan_below_threshold_leaves_values_untouched():
    df = pd.DataFrame({"FR_Fz_2": np.full(10, 500.0)})   # median < N_TO_DAN_THRESHOLD (1000)
    out, scaled = _apply_n_to_dan(df, ["FR_Fz_2"])
    assert scaled is False
    np.testing.assert_array_equal(out["FR_Fz_2"].to_numpy(), df["FR_Fz_2"].to_numpy())


def test_apply_n_to_dan_above_threshold_divides_by_ten():
    df = pd.DataFrame({"FR_Fz_2": np.full(10, 5000.0)})  # median > threshold
    out, scaled = _apply_n_to_dan(df, ["FR_Fz_2"])
    assert scaled is True
    assert np.allclose(out["FR_Fz_2"].to_numpy(), 500.0)


def test_apply_n_to_dan_only_touches_listed_channels():
    """A channel not named in ``channels`` must survive even if it is itself
    above the threshold -- the function scales exactly the list it is given."""
    df = pd.DataFrame({
        "FR_Fz_2": np.full(10, 5000.0),
        "FR_Fx_2": np.full(10, 8000.0),   # above threshold too, deliberately not passed
    })
    out, scaled = _apply_n_to_dan(df, ["FR_Fz_2"])
    assert scaled is True
    assert np.allclose(out["FR_Fz_2"].to_numpy(), 500.0)
    assert np.allclose(out["FR_Fx_2"].to_numpy(), 8000.0)


# --------------------------------------------------- 2. normalise_force_units

def _two_wheel_force_config(tmp_path, fz_range_dan=(0.0, 2000.0)):
    """A minimal RunConfig wired the way dtt.pipeline builds one: real
    production channel names (FR_Fz_2 etc., as used in golden_corpus/
    VALIDATION_GUIDE.md) run through the real discovery path, not a stub.

    output_dir=tmp_path keeps RunConfig.__post_init__'s mkdir() confined to
    pytest's own scratch directory -- a bare RunConfig() defaults to the
    real dtt/outputs/ and litters it with an empty timestamped folder per
    test run."""
    columns = ["FR_Fx_2", "FR_Fz_2"]
    cfg = RunConfig(output_dir=tmp_path, study_name="synthetic_test")
    cfg.run_channels = build_run_channels(columns)
    cfg.tyre = TyreParams(tyre_type="test", fz_range_dan=fz_range_dan)
    return cfg, columns


def test_normalise_force_units_corrects_a_clean_decade(tmp_path):
    cfg, _ = _two_wheel_force_config(tmp_path, fz_range_dan=(0.0, 2000.0))
    # ratio = med / (0.5*hi) = 10000 / 1000 = 10 -> a clean decade (decades=1)
    df = pd.DataFrame({"FR_Fx_2": np.full(50, 3000.0),
                       "FR_Fz_2": np.full(50, 10000.0)})

    out, meta = normalise_force_units(df, cfg)

    assert meta["force_scale_decades"] == 1
    assert meta["force_scale_divisor"] == 10.0
    assert np.allclose(out["FR_Fz_2"].to_numpy(), 1000.0)
    assert np.allclose(out["FR_Fx_2"].to_numpy(), 300.0)


def test_normalise_force_units_leaves_a_non_decade_mismatch_alone(tmp_path):
    cfg, _ = _two_wheel_force_config(tmp_path, fz_range_dan=(0.0, 2000.0))
    # ratio = 2000 / 1000 = 2 -> log10(2) = 0.301, rounds to 0 decades: no-op
    df = pd.DataFrame({"FR_Fx_2": np.full(50, 300.0),
                       "FR_Fz_2": np.full(50, 2000.0)})

    out, meta = normalise_force_units(df, cfg)

    assert meta == {}
    np.testing.assert_array_equal(out["FR_Fz_2"].to_numpy(), df["FR_Fz_2"].to_numpy())


def test_normalise_force_units_leaves_moment_channels_alone(tmp_path):
    """Docstring: 'Moments are deliberately left alone: they carry their own
    CR factor and their own unit (Nm)'. Fx/Fy/Fz should scale, Mx must not."""
    columns = ["FR_Fx_2", "FR_Fz_2", "FR_Mx_2"]
    cfg = RunConfig(output_dir=tmp_path, study_name="synthetic_test_moments")
    cfg.run_channels = build_run_channels(columns)
    cfg.tyre = TyreParams(tyre_type="test", fz_range_dan=(0.0, 2000.0))
    df = pd.DataFrame({"FR_Fx_2": np.full(50, 3000.0),
                       "FR_Fz_2": np.full(50, 10000.0),
                       "FR_Mx_2": np.full(50, 4321.0)})

    out, meta = normalise_force_units(df, cfg)

    assert meta["force_scale_decades"] == 1
    assert np.allclose(out["FR_Fz_2"].to_numpy(), 1000.0)
    assert np.allclose(out["FR_Fx_2"].to_numpy(), 300.0)
    np.testing.assert_array_equal(out["FR_Mx_2"].to_numpy(), df["FR_Mx_2"].to_numpy())


# ------------------------------------------------------ 3. remove_stops_frame

def _stop_window_frame(n=40, fs=10.0, stop_start=10, stop_len=15,
                       moving_kph=20.0, stopped_kph=0.0):
    """40 samples @ 10 Hz: moving, a 1.5 s stop, moving again. FR_Fz_2 carries
    a distinct value per sample so exclusion/preservation can be checked by
    value, not just by count."""
    speed = np.full(n, moving_kph)
    speed[stop_start:stop_start + stop_len] = stopped_kph
    df = pd.DataFrame({
        "Time": np.arange(n) / fs,
        "Speed": speed,
        "FR_Fz_2": np.arange(n, dtype=float) * 100.0,
    })
    return df


def test_remove_stops_frame_excludes_the_stop_window():
    df = _stop_window_frame()
    out, meta = remove_stops_frame(
        df, fs=10.0, speed_column="Speed", min_stop_s=1.0,
        moving_kph=1.5, seam_search_s=0.0, seam_blend_s=0.0)

    assert meta["stops_removed"] == 1
    assert meta["stop_seconds"] == 1.5
    assert len(out) == 40 - 15


def test_remove_stops_frame_moving_speed_boundary():
    """detect_stops_from_speed uses strict '<': exactly at moving_kph must
    stay moving, just below it must be treated as stopped.

    The plateau either side is genuinely moving (20 kph), not merely at the
    boundary, because detect_stops_from_speed treats a channel whose max
    never exceeds moving_kph as unusable ('never moves -> carries no
    information') and returns None for it -- a channel sitting at the
    boundary throughout would trip that guard rather than exercise the '<'
    comparison this test targets."""
    n = 40
    speed = np.full(n, 20.0)
    speed[10:20] = 1.5               # exactly at the boundary: must stay "moving"
    speed[20:30] = 1.4               # just below: must be treated as stopped
    df = pd.DataFrame({
        "Time": np.arange(n) / 10.0,
        "Speed": speed,
        "FR_Fz_2": np.arange(n, dtype=float) * 100.0,
    })

    out, meta = remove_stops_frame(
        df, fs=10.0, speed_column="Speed", min_stop_s=1.0,
        moving_kph=1.5, seam_search_s=0.0, seam_blend_s=0.0)

    assert meta["stops_removed"] == 1
    assert meta["stop_seconds"] == 1.0
    assert len(out) == n - 10
    # the boundary segment [10:20] must have survived, unremoved
    np.testing.assert_array_equal(
        out["FR_Fz_2"].to_numpy()[10:20], np.arange(10, 20, dtype=float) * 100.0)


def test_remove_stops_frame_keeps_non_stop_values_unchanged():
    """With seam_search_s=0 and seam_blend_s=0 (no boundary nudge, no seam
    blend), the surviving samples must be byte-identical to the source --
    not interpolated, not smoothed."""
    df = _stop_window_frame()
    out, _ = remove_stops_frame(
        df, fs=10.0, speed_column="Speed", min_stop_s=1.0,
        moving_kph=1.5, seam_search_s=0.0, seam_blend_s=0.0)

    expected = np.concatenate([df["FR_Fz_2"].to_numpy()[:10],
                               df["FR_Fz_2"].to_numpy()[25:]])
    np.testing.assert_array_equal(out["FR_Fz_2"].to_numpy(), expected)


def test_remove_stops_frame_does_not_mutate_the_input_frame():
    df = _stop_window_frame()
    before = df.copy(deep=True)

    remove_stops_frame(df, fs=10.0, speed_column="Speed", min_stop_s=1.0,
                       moving_kph=1.5, seam_search_s=0.0, seam_blend_s=0.0)

    pd.testing.assert_frame_equal(df, before)


# ---------------------------------------------------- 4. order of operations

def test_scale_correction_runs_before_percentile_blanking(tmp_path):
    """dtt/pipeline.py runs normalise_force_units (before stage 2 validation)
    strictly before sanitize() / _flag_outliers (stage 3) -- read directly
    from dtt/pipeline.py lines 452-466. Composing the two real functions in
    that same order and checking the result lands in the tyre preset's own
    daN band is what depends on the order; it would not if sanitize() ran on
    the raw, decade-high magnitude."""
    cfg, columns = _two_wheel_force_config(tmp_path, fz_range_dan=(0.0, 2000.0))
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "FR_Fx_2": rng.normal(3000.0, 20.0, 200),
        "FR_Fz_2": rng.normal(10000.0, 50.0, 200),   # -> clean decade, as above
    })

    scaled_df, scale_meta = normalise_force_units(df, cfg)
    assert scale_meta["force_scale_decades"] == 1, "fixture must trip the decade correction"

    sanitized_df, _report = sanitize(scaled_df, cfg)

    # Scaled Fz sits near 1000 daN; if sanitize() had instead run on the
    # un-scaled frame it would be operating on values near 10000.
    assert sanitized_df["FR_Fz_2"].max() < cfg.tyre.fz_range_dan[1] * 5


# ------------------------------------------- 5. structural despike boundaries

def test_detect_rail_run_length_boundary():
    # channel max is 100.0; a 2-sample run at it must not trip min_run=3,
    # a 3-sample run must.
    arr = np.array([0.0, 50.0, 30.0, 100.0, 100.0, 40.0,
                    10.0, 100.0, 100.0, 100.0, 20.0])
    flag = detect_rail(arr, min_run=3)
    assert not flag[3:5].any(), "a 2-sample rail run must not trip min_run=3"
    assert flag[7:10].all(), "a 3-sample rail run must trip min_run=3"


def test_detect_dropout_run_length_boundary():
    # a frozen (non-zero, non-rail) value; 4-sample run must not trip
    # max_run=5, 5-sample run must.
    arr = np.concatenate([
        np.array([10.0, 20.0]),
        np.full(4, 42.0),
        np.array([15.0]),
        np.full(5, 42.0),
        np.array([25.0]),
    ])
    flag = detect_dropout(arr, max_run=5)
    assert not flag[2:6].any(), "a 4-sample frozen run must not trip max_run=5"
    assert flag[7:12].all(), "a 5-sample frozen run must trip max_run=5"
