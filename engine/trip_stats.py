"""
engine/trip_stats.py
====================
Trip summary statistics engine for the DTT WFT analysis.

Implements §6.3 of the build spec:
  - Total distance (km), total duration (s / h)
  - Vehicle_Speed percentiles (P50, P90), mean, max
  - Latacc and Longacc standard deviation and max (abs)
  - Yawrate max
  - Per-wheel force summary: min / mean / max / P90 / P95 for Fx, Fy, Fz
  - Per-wheel WS1 (wheel speed) min / mean / max (§Q4 — minor stat, not a full output)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_trip_stats(
    df: pd.DataFrame,
    chan_map: dict,
    meta: dict,
    cfg: dict,
    display_unit: str = "daN",
) -> dict:
    """
    Compute full trip statistics.

    Parameters
    ----------
    df           : Loaded DataFrame
    chan_map     : Channel map from build_channel_map()
    meta         : Loader meta dict (fs, dt, n_rows, duration_s, total_dist_km, warnings)
    cfg          : Parsed defaults.yaml dict
    display_unit : 'N' or 'daN' for force summary

    Returns
    -------
    dict with sub-dicts:
        'trip'   : overall trip metrics
        'accel'  : IMU acceleration/yaw stats
        'forces' : per-wheel force summary dict
        'wheels' : per-wheel WS1 (wheel speed rpm) summary
    """
    unit_scale = _get_unit_scale(cfg, display_unit)
    ts_cfg = cfg.get("trip_stats", {})
    speed_pcts = ts_cfg.get("speed_percentiles", [50, 90])
    force_pcts = ts_cfg.get("force_percentiles", [90, 95])

    trip = _compute_trip(df, chan_map, meta)
    accel = _compute_accel_stats(df, chan_map)
    forces = _compute_force_summary(df, chan_map, cfg, unit_scale, force_pcts, display_unit)
    wheels = _compute_wheel_speed_stats(df, chan_map)

    return {
        "trip":         trip,
        "accel":        accel,
        "forces":       forces,
        "wheels":       wheels,
        "display_unit": display_unit,
    }


# ─── Sub-computations ─────────────────────────────────────────────────────

def _compute_trip(df: pd.DataFrame, chan_map: dict, meta: dict) -> dict:
    """Overall trip metrics."""
    duration_s = meta.get("duration_s", 0.0)
    total_dist_km = meta.get("total_dist_km", 0.0)
    n_rows = meta.get("n_rows", len(df))
    fs = meta.get("fs", 100.0)

    # Vehicle speed stats
    speed_col = chan_map.get("speed")
    speed_stats: dict = {}
    if speed_col and speed_col in df.columns:
        spd = pd.to_numeric(df[speed_col], errors="coerce").dropna()
        if len(spd) > 0:
            speed_stats = {
                "mean_kmh":   float(spd.mean()),
                "max_kmh":    float(spd.max()),
                "p50_kmh":    float(np.percentile(spd, 50)),
                "p90_kmh":    float(np.percentile(spd, 90)),
            }

    return {
        "total_dist_km":  total_dist_km,
        "duration_s":     duration_s,
        "duration_h":     duration_s / 3600.0,
        "n_rows":         n_rows,
        "fs_hz":          fs,
        **speed_stats,
    }


def _compute_accel_stats(df: pd.DataFrame, chan_map: dict) -> dict:
    """
    IMU acceleration and yaw-rate statistics.
    Channels are already in g (latacc / longacc) and deg/s (yawrate)
    per the declared input units — no conversion needed.
    """
    result: dict = {}

    def _series_stats(col_name: Optional[str], label: str) -> None:
        if col_name is None or col_name not in df.columns:
            return
        vals = pd.to_numeric(df[col_name], errors="coerce").dropna()
        if len(vals) == 0:
            return
        result[f"{label}_std"]    = float(vals.std())
        result[f"{label}_mean"]   = float(vals.mean())
        result[f"{label}_max_abs"] = float(vals.abs().max())
        result[f"{label}_p95_abs"] = float(np.percentile(vals.abs(), 95))

    _series_stats(chan_map.get("latacc"),  "latacc")
    _series_stats(chan_map.get("longacc"), "longacc")
    _series_stats(chan_map.get("yawrate"), "yawrate")

    return result


def _compute_force_summary(
    df: pd.DataFrame,
    chan_map: dict,
    cfg: dict,
    unit_scale: float,
    force_pcts: list[int],
    display_unit: str,
) -> dict:
    """
    Per-wheel, per-channel force summary: min / mean / max / P90 / P95.
    Values converted to display_unit.
    """
    from .channel_map import WHEELS, FORCE_SUFFIXES

    summary: dict = {}
    for wheel in WHEELS:
        wheel_stats: dict = {}
        for suffix in FORCE_SUFFIXES:
            key = f"{wheel}_{suffix}"
            col = chan_map.get(key)
            if col is None or col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").dropna().values * unit_scale
            if len(vals) == 0:
                continue
            ch_stats = {
                "min":  float(vals.min()),
                "mean": float(vals.mean()),
                "max":  float(vals.max()),
                "std":  float(vals.std()),
            }
            for p in force_pcts:
                ch_stats[f"p{p}"] = float(np.percentile(vals, p))
            wheel_stats[suffix] = ch_stats
        if wheel_stats:
            summary[wheel] = wheel_stats

    return summary


def _compute_wheel_speed_stats(df: pd.DataFrame, chan_map: dict) -> dict:
    """
    Per-wheel rotational speed (WS1, rpm) statistics.
    Minor stat for Trip Statistics tab and data-sanity checks.
    """
    from .channel_map import WHEELS

    result: dict = {}
    for wheel in WHEELS:
        key = f"{wheel}_WS1"
        col = chan_map.get(key)
        if col is None or col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        result[wheel] = {
            "min_rpm":    float(vals.min()),
            "mean_rpm":   float(vals.mean()),
            "max_rpm":    float(vals.max()),
            "p90_rpm":    float(np.percentile(vals, 90)),
        }

    return result


# ─── Utility ──────────────────────────────────────────────────────────────

def _get_unit_scale(cfg: dict, display_unit: str) -> float:
    scales = cfg.get("display", {}).get("unit_scales", {"N": 1.0, "daN": 0.1})
    return float(scales.get(display_unit, 1.0))


def stats_to_dataframe(trip_stats: dict) -> dict[str, pd.DataFrame]:
    """
    Convert the trip_stats result dict into a set of display-ready DataFrames
    suitable for the Streamlit table widget and CSV export.

    Returns
    -------
    dict with keys: 'trip', 'accel', 'forces', 'wheels'
    Each value is a pd.DataFrame.
    """
    dfs = {}

    # Trip overview
    trip = trip_stats.get("trip", {})
    dfs["trip"] = pd.DataFrame([{
        "Metric": k, "Value": v
    } for k, v in trip.items()])

    # Accel / IMU
    accel = trip_stats.get("accel", {})
    dfs["accel"] = pd.DataFrame([{
        "Metric": k, "Value": v
    } for k, v in accel.items()])

    # Force summary — flatten per-wheel per-channel
    forces = trip_stats.get("forces", {})
    rows = []
    for wheel, ch_stats in forces.items():
        for suffix, stats in ch_stats.items():
            row = {"Wheel": wheel, "Channel": suffix}
            row.update(stats)
            rows.append(row)
    dfs["forces"] = pd.DataFrame(rows)

    # Wheel speed
    wheels = trip_stats.get("wheels", {})
    rows = []
    for wheel, stats in wheels.items():
        row = {"Wheel": wheel}
        row.update(stats)
        rows.append(row)
    dfs["wheels"] = pd.DataFrame(rows) if rows else pd.DataFrame()

    return dfs
