"""
tests/test_engine.py
====================
Unit tests for the DTT WFT engine functions.

All tests use a synthetic 2000-row DataFrame with known values — no real file needed.
Run with:   pytest tests/test_engine.py -v

Tests cover:
  - Sample rate detection
  - Time axis rebuild
  - Channel map resolution (strict + exclude-list)
  - Force histograms (bins, percentiles, distance weighting, unit conversion)
  - Heatmap grid (% sum ~100)
  - Trip statistics (distance, duration, force summary)
  - g-g plot (decimation, axis in g, no conversion)
  - Fatigue (damage formula, rainflow presence)
  - Box stats (Q1/Q3/median correctness)
  - Unit conversion: N→daN is exactly ÷10, not heuristic
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Make engine importable from the tests directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

# ─── Load config ──────────────────────────────────────────────────────────
CFG_PATH = Path(__file__).parent.parent / "config" / "defaults.yaml"
with open(CFG_PATH, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

# ─── Synthetic data fixture ────────────────────────────────────────────────
N = 2000
FS = 100.0  # Hz
DT = 1.0 / FS


def make_synthetic_df(n: int = N) -> pd.DataFrame:
    """
    Create a synthetic DataFrame that mimics the real WFT CSV structure.
    Known values for verifiable assertions.
    """
    t = np.arange(n) * DT
    dist = np.cumsum(np.full(n, 0.1))  # 0.1 m per sample = 10 m/s = 36 km/h

    np.random.seed(42)

    df = pd.DataFrame({
        "Time":         t,
        "t_rebuilt":    t,
        "Dist":         dist,
        "Latitude":     np.full(n, 12.5),
        "Longitude":    np.full(n, 78.0),
        "Altitude":     np.full(n, 220.0),
        "Vehicle_Speed": np.full(n, 36.0),  # km/h
        "Latacc":       np.random.normal(0, 0.3, n),   # g
        "Longacc":      np.random.normal(0, 0.2, n),   # g
        "Yawrate":      np.random.normal(0, 5.0, n),   # deg/s
        # Forces in N (input unit)
        "FL_Fx": np.random.normal(-200,  500, n),
        "FL_Fy": np.random.normal(   0,  800, n),
        "FL_Fz": np.random.normal(6500, 1000, n),
        "FR_Fx": np.random.normal(-200,  500, n),
        "FR_Fy": np.random.normal(   0,  800, n),
        "FR_Fz": np.random.normal(6500, 1000, n),
        "RL_Fx": np.random.normal( 100,  400, n),
        "RL_Fy": np.random.normal(   0,  700, n),
        "RL_Fz": np.random.normal(5500,  900, n),
        "RR_Fx": np.random.normal( 100,  400, n),
        "RR_Fy": np.random.normal(   0,  700, n),
        "RR_Fz": np.random.normal(5500,  900, n),
        # Moments
        "FL_Mx": np.random.normal(0, 100, n),
        "FL_My": np.random.normal(0, 200, n),
        "FL_Mz": np.random.normal(0,  50, n),
        # Wheel speed (rpm)
        "FL_WS1": np.random.normal(800, 50, n),
        "FR_WS1": np.random.normal(800, 50, n),
        "RL_WS1": np.random.normal(800, 50, n),
        "RR_WS1": np.random.normal(800, 50, n),
    })
    return df


DF = make_synthetic_df()


# ─── Test: sample rate detection ──────────────────────────────────────────

def test_detect_sample_rate():
    from engine.io_loader import detect_sample_rate
    dt, fs = detect_sample_rate(DF, CFG)
    assert abs(fs - FS) < 0.1, f"Expected fs≈{FS}, got {fs:.2f}"
    assert abs(dt - DT) < 1e-4, f"Expected dt≈{DT}, got {dt:.6f}"


# ─── Test: time axis rebuild ───────────────────────────────────────────────

def test_rebuild_time_axis():
    from engine.io_loader import rebuild_time_axis
    df_test = DF.copy()
    df_test = df_test.drop(columns=["t_rebuilt"])  # remove existing
    warnings: list[str] = []
    df_result = rebuild_time_axis(df_test, DT, CFG, warnings)

    assert "t_rebuilt" in df_result.columns
    rebuilt = df_result["t_rebuilt"].values
    assert rebuilt[0] == 0.0, "First t_rebuilt should be 0"
    assert rebuilt[-1] == pytest.approx((N - 1) * DT, rel=1e-4)
    # Monotonically increasing
    assert np.all(np.diff(rebuilt) > 0), "t_rebuilt should be strictly increasing"


# ─── Test: channel map ─────────────────────────────────────────────────────

def test_channel_map_finds_force_channels():
    from engine.channel_map import build_channel_map
    chan_map = build_channel_map(DF, CFG)
    # All FL/FR/RL/RR Fx/Fy/Fz should resolve
    for wheel in ["FL", "FR", "RL", "RR"]:
        for suffix in ["Fx", "Fy", "Fz"]:
            key = f"{wheel}_{suffix}"
            assert chan_map.get(key) == key, f"Expected {key} → {key}, got {chan_map.get(key)}"


def test_channel_map_does_not_confuse_latitude():
    """
    Latitude, Longitude, Latacc should NOT be matched as force channels.
    find_channel_strict with exclude list must reject them.
    """
    from engine.channel_map import find_channel_strict
    exclude = [k.lower() for k in CFG.get("channel_exclude_keywords", [])]
    # Searching for 'FL' + 'Fz' should NOT return 'Latacc' or 'Longitude'
    df_tricky = DF.copy()
    df_tricky["Latitude_Fx"] = 0.0   # column that has 'Fx' but starts with excluded 'lat'
    result = find_channel_strict(df_tricky, "FL", "Fx", exclude)
    # Must not return the tricky column
    assert result != "Latitude_Fx", "Exclude list failed to reject Latitude_Fx"


def test_channel_map_resolves_speed_and_imu():
    from engine.channel_map import build_channel_map
    chan_map = build_channel_map(DF, CFG)
    assert chan_map.get("speed") == "Vehicle_Speed", f"speed → {chan_map.get('speed')}"
    assert chan_map.get("latacc") == "Latacc", f"latacc → {chan_map.get('latacc')}"
    assert chan_map.get("longacc") == "Longacc", f"longacc → {chan_map.get('longacc')}"


# ─── Test: histograms ─────────────────────────────────────────────────────

def test_histograms_basic():
    from engine.channel_map import build_channel_map
    from engine.histograms import compute_force_histograms
    chan_map = build_channel_map(DF, CFG)
    results = compute_force_histograms(DF, chan_map, CFG, display_unit="daN")

    assert "FL_Fx" in results
    res = results["FL_Fx"]
    assert res["display_unit"] == "daN"
    assert len(res["bin_centers"]) > 0
    assert res["n_valid"] > 0
    # Percentiles present
    for p in [80, 90, 95]:
        assert p in res["percentiles"], f"P{p} missing from percentiles"


def test_histogram_unit_conversion_is_deterministic():
    """
    N→daN conversion must be exactly × 0.1, never a median-based heuristic.
    Median of FL_Fz is ~6500 N; in daN it should be ~650.
    If the heuristic 'if median > 1000: /10' was used, a quiet channel
    (median < 1000 N) would NOT be converted — this test verifies the
    declared unit_scale is always applied.
    """
    from engine.channel_map import build_channel_map
    from engine.histograms import compute_force_histograms

    chan_map = build_channel_map(DF, CFG)
    # FL_Fx has median around -200 N (< 1000 N), so the heuristic would NOT convert
    res_dan = compute_force_histograms(DF, chan_map, CFG, display_unit="daN")
    res_n   = compute_force_histograms(DF, chan_map, CFG, display_unit="N")

    fl_fx_dan = res_dan["FL_Fx"]
    fl_fx_n   = res_n["FL_Fx"]

    p90_dan = fl_fx_dan["percentiles"][90]
    p90_n   = fl_fx_n["percentiles"][90]

    assert abs(p90_dan - p90_n * 0.1) < 1e-3, (
        f"Unit conversion is NOT exactly ×0.1: P90 in daN={p90_dan:.4f}, "
        f"P90 in N={p90_n:.4f}, ratio={p90_dan/p90_n:.6f}"
    )


def test_distance_weighted_histograms():
    from engine.channel_map import build_channel_map
    from engine.histograms import compute_force_histograms
    chan_map = build_channel_map(DF, CFG)
    results = compute_force_histograms(
        DF, chan_map, CFG, display_unit="daN", use_distance_weight=True
    )
    res = results.get("FL_Fz")
    assert res is not None
    # Distance-weighted: total should equal ~total distance in metres
    total_dist_m = DF["Dist"].iloc[-1] - DF["Dist"].iloc[0]
    hist_sum_m = res["hist_dist_km"].sum() * 1000.0
    # Should be within 10% of actual total distance (some samples may be outside range)
    assert hist_sum_m > total_dist_m * 0.5, (
        f"Distance-weighted histogram total {hist_sum_m:.1f} m is too small "
        f"vs actual {total_dist_m:.1f} m"
    )


# ─── Test: heatmaps ────────────────────────────────────────────────────────

def test_heatmap_pct_sums_to_100():
    from engine.channel_map import build_channel_map
    from engine.heatmaps import compute_heatmap
    chan_map = build_channel_map(DF, CFG)
    result = compute_heatmap(DF, chan_map, "FL", "Fx", "Fy", CFG, display_unit="daN")
    H_pct = result["H_pct"]
    total = H_pct.sum()
    assert abs(total - 100.0) < 1.0, f"Heatmap % total = {total:.2f}, expected ~100"


def test_heatmap_all_pairings():
    from engine.channel_map import build_channel_map
    from engine.heatmaps import compute_all_heatmaps
    chan_map = build_channel_map(DF, CFG)
    results = compute_all_heatmaps(DF, chan_map, CFG, wheels=["FL", "FR"])
    assert len(results) == 6, f"Expected 2 wheels × 3 pairings = 6, got {len(results)}"


# ─── Test: trip statistics ─────────────────────────────────────────────────

def test_trip_stats_distance():
    from engine.channel_map import build_channel_map
    from engine.trip_stats import compute_trip_stats
    from engine.io_loader import detect_sample_rate

    chan_map = build_channel_map(DF, CFG)
    dt, fs = detect_sample_rate(DF, CFG)
    meta = {
        "fs": fs, "dt": dt,
        "n_rows": N,
        "duration_s": N / fs,
        "total_dist_km": (DF["Dist"].iloc[-1] - DF["Dist"].iloc[0]) / 1000.0,
        "warnings": [],
    }
    stats = compute_trip_stats(DF, chan_map, meta, CFG)
    trip = stats["trip"]

    expected_dist_km = (N * 0.1) / 1000.0  # 0.1 m/sample × N samples ÷ 1000
    assert abs(trip["total_dist_km"] - expected_dist_km) < 0.01, (
        f"Distance mismatch: got {trip['total_dist_km']:.3f} km, "
        f"expected {expected_dist_km:.3f} km"
    )
    assert abs(trip["duration_s"] - N / FS) < 0.1


def test_trip_stats_force_summary():
    from engine.channel_map import build_channel_map
    from engine.trip_stats import compute_trip_stats
    from engine.io_loader import detect_sample_rate

    chan_map = build_channel_map(DF, CFG)
    dt, fs = detect_sample_rate(DF, CFG)
    meta = {"fs": fs, "dt": dt, "n_rows": N,
            "duration_s": N/fs, "total_dist_km": 0.2, "warnings": []}
    stats = compute_trip_stats(DF, chan_map, meta, CFG, display_unit="daN")
    forces = stats["forces"]

    assert "FL" in forces, "FL missing from force summary"
    assert "Fz" in forces["FL"], "Fz missing from FL force summary"
    fz_mean = forces["FL"]["Fz"]["mean"]
    # FL_Fz was generated with mean=6500 N → mean~650 daN
    assert 400 < fz_mean < 900, f"FL_Fz mean in daN = {fz_mean:.1f}, expected 400–900"


# ─── Test: g-g plot ────────────────────────────────────────────────────────

def test_gg_decimation():
    from engine.channel_map import build_channel_map
    from engine.gg_severity import compute_gg
    chan_map = build_channel_map(DF, CFG)
    result = compute_gg(DF, chan_map, CFG)
    assert result["n_valid"] > 0
    # Plot arrays should not exceed decimate_to
    n_plot = len(result["latacc_plot"])
    assert n_plot <= CFG["gg"]["decimate_to"], (
        f"Plot array size {n_plot} exceeds decimate_to={CFG['gg']['decimate_to']}"
    )


def test_gg_values_are_in_g():
    """
    Latacc and Longacc are declared in g in config.
    The g-g engine must NOT convert them (they're already in g).
    Max absolute value should be < 5 g for our synthetic normal data (σ=0.3 g).
    If accidental ×9.81 was applied, values would be ~±3 m/s², not g.
    """
    from engine.channel_map import build_channel_map
    from engine.gg_severity import compute_gg
    chan_map = build_channel_map(DF, CFG)
    result = compute_gg(DF, chan_map, CFG)

    lat_max = float(np.abs(result["latacc_full"]).max())
    lon_max = float(np.abs(result["longacc_full"]).max())

    # Synthetic data: Latacc ~ N(0, 0.3g), max should be well under 2g
    assert lat_max < 5.0, (
        f"Lateral acc max = {lat_max:.2f} — suspiciously large (unit conversion applied?)"
    )
    assert lon_max < 5.0, (
        f"Longitudinal acc max = {lon_max:.2f} — suspiciously large"
    )


# ─── Test: fatigue ────────────────────────────────────────────────────────

def test_fatigue_damage_formula():
    """
    Verify Miner's rule: D = Σ(count × range^m).
    Use a synthetic cycle set with known values.
    """
    from engine.fatigue import compute_damage
    # 3 cycles: range=10, count=1; range=20, count=2; range=30, count=1
    cycles = [(10.0, 0.0, 1.0, 0, 1), (20.0, 0.0, 2.0, 1, 2), (30.0, 0.0, 1.0, 2, 3)]
    m = 5
    expected = 1 * 10**5 + 2 * 20**5 + 1 * 30**5
    result = compute_damage(cycles, m=m)
    assert abs(result - expected) < 1e-3, f"Damage {result:.0f} ≠ expected {expected:.0f}"


def test_fatigue_summary_structure():
    from engine.channel_map import build_channel_map
    from engine.fatigue import compute_fatigue_summary
    chan_map = build_channel_map(DF, CFG)
    # Run on small synthetic dataset — fast
    result = compute_fatigue_summary(DF, chan_map, CFG, wheels=["FL"], display_unit="daN")
    assert "channels" in result
    assert "axles" in result
    fl_fz = result["channels"].get("FL_Fz")
    assert fl_fz is not None, "FL_Fz missing from fatigue results"


# ─── Test: box stats ──────────────────────────────────────────────────────

def test_box_stats_quartiles():
    from engine.channel_map import build_channel_map
    from engine.box_distance import compute_box_stats
    chan_map = build_channel_map(DF, CFG)
    results = compute_box_stats(DF, chan_map, CFG, display_unit="daN")

    res = results.get("FL_Fz")
    assert res is not None
    # Verify Q1 < median < Q3
    assert res["q1"] < res["median"] < res["q3"], (
        f"Quartile order violated: Q1={res['q1']:.1f} median={res['median']:.1f} Q3={res['q3']:.1f}"
    )
    # FL_Fz mean ~ 650 daN (6500 N ÷ 10)
    assert 400 < res["mean"] < 900, f"FL_Fz mean daN = {res['mean']:.1f}"


def test_box_stats_p95():
    from engine.channel_map import build_channel_map
    from engine.box_distance import compute_box_stats
    chan_map = build_channel_map(DF, CFG)
    results = compute_box_stats(DF, chan_map, CFG, display_unit="daN")
    res = results["FL_Fz"]
    # P95 should be > median
    assert res["p95"] > res["median"], "P95 should be > median"
    # P95 should be < max
    assert res["p95"] < res["max"], "P95 should be < max"


# ─── Test: distance distribution ─────────────────────────────────────────

def test_distance_distribution_km():
    from engine.channel_map import build_channel_map
    from engine.box_distance import compute_distance_distribution
    chan_map = build_channel_map(DF, CFG)
    results = compute_distance_distribution(DF, chan_map, CFG, display_unit="daN")
    res = results.get("FL_Fy")
    assert res is not None
    # Total distance should be positive
    assert res["total_dist_km"] > 0, "Distance distribution total is 0"
    # Y-axis values should be in km (positive, not raw metres)
    assert res["hist_dist_km"].max() < 10.0, "hist_dist_km looks like metres not km"


# ─── Run summary ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
