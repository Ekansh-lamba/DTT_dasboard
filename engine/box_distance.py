"""
engine/box_distance.py
======================
Box plot statistics and distance-distribution histogram engine.

Implements §6.5 of the build spec:
  - Box plot statistics: min, Q1, median, mean, Q3, P95, max, IQR per channel
  - Distance-distribution histograms: force on X-axis, distance in km on Y-axis
    (matches the Apollo reference 'FL_Fy — Distance distribution' chart style)

Ported / inspired by:
  M&M_PV_WFT_Analyser: distance-weighted histogram with Y-axis in km
  WFT_Analyzer_v5: box plot with P5/Mean/P95 reference lines
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_box_stats(
    df: pd.DataFrame,
    chan_map: dict,
    cfg: dict,
    wheels: Optional[list[str]] = None,
    display_unit: str = "daN",
) -> dict:
    """
    Compute box plot statistics for all force channels.

    Parameters
    ----------
    df           : Loaded DataFrame
    chan_map     : Channel map
    cfg          : Config dict
    wheels       : Wheel positions (None = all)
    display_unit : 'N' or 'daN'

    Returns
    -------
    dict keyed by internal channel key (e.g. 'FL_Fx'), each value:
    {
        'min', 'q1', 'median', 'mean', 'q3', 'p5', 'p95', 'max', 'iqr',
        'display_unit', 'wheel', 'suffix', 'channel',
        'values': np.ndarray (for Plotly go.Box raw data — downsampled to 10k)
    }
    """
    from .channel_map import WHEELS as ALL_WHEELS, FORCE_SUFFIXES

    if wheels is None:
        wheels = ALL_WHEELS

    unit_scale = _get_unit_scale(cfg, display_unit)
    box_cfg = cfg.get("box_distance", {})
    ylim_cfg = box_cfg.get("box_ylim_dan", {
        "Fx": [-300, 300], "Fy": [-300, 300], "Fz": [100, 1000]
    })

    results: dict = {}
    for wheel in wheels:
        for suffix in FORCE_SUFFIXES:
            key = f"{wheel}_{suffix}"
            col = chan_map.get(key)
            if col is None or col not in df.columns:
                continue

            raw = pd.to_numeric(df[col], errors="coerce").dropna().values * unit_scale
            if len(raw) == 0:
                continue

            # Downsample for Plotly box (raw mode) — capped at 10k points
            plot_vals = _downsample(raw, max_pts=10_000)

            results[key] = {
                "min":      float(raw.min()),
                "q1":       float(np.percentile(raw, 25)),
                "median":   float(np.median(raw)),
                "mean":     float(raw.mean()),
                "q3":       float(np.percentile(raw, 75)),
                "p5":       float(np.percentile(raw, 5)),
                "p95":      float(np.percentile(raw, 95)),
                "max":      float(raw.max()),
                "iqr":      float(np.percentile(raw, 75) - np.percentile(raw, 25)),
                "ylim":     ylim_cfg.get(suffix),
                "display_unit": display_unit,
                "wheel":    wheel,
                "suffix":   suffix,
                "channel":  col,
                "values":   plot_vals,   # downsampled raw values for Plotly go.Box
            }
            logger.debug(
                "Box stats: %s | median=%.1f | P95=%.1f %s",
                key, results[key]["median"], results[key]["p95"], display_unit,
            )

    return results


def compute_distance_distribution(
    df: pd.DataFrame,
    chan_map: dict,
    cfg: dict,
    wheels: Optional[list[str]] = None,
    display_unit: str = "daN",
) -> dict:
    """
    Compute distance-distribution histograms.

    For each force channel: bin by force value (daN), weight by distance increment.
    Y-axis = distance in km spent at each force level.
    Matches the Apollo reference 'FL_Fy — Distance distribution' chart style.

    Parameters
    ----------
    df           : Loaded DataFrame
    chan_map     : Channel map
    cfg          : Config dict
    wheels       : Wheel positions (None = all)
    display_unit : 'N' or 'daN'

    Returns
    -------
    dict keyed by internal channel key, each value:
    {
        'bin_centers'  : np.ndarray (force in display_unit)
        'bin_edges'    : np.ndarray
        'hist_dist_km' : np.ndarray (distance in km per bin)
        'total_dist_km': float
        'display_unit' : str
        'wheel'        : str
        'suffix'       : str
        'channel'      : str
    }
    """
    from .channel_map import WHEELS as ALL_WHEELS, FORCE_SUFFIXES
    from .histograms import _get_distance_weights

    if wheels is None:
        wheels = ALL_WHEELS

    unit_scale = _get_unit_scale(cfg, display_unit)
    hist_cfg = cfg.get("histogram", {})
    n_bins = hist_cfg.get("bins", 60)
    ranges_dan = hist_cfg.get("force_ranges_dan", {
        "Fx": [-300, 300], "Fy": [-300, 300], "Fz": [100, 1000]
    })

    # Distance increment weights
    dist_weights_m = _get_distance_weights(df, chan_map)

    results: dict = {}
    for wheel in wheels:
        for suffix in FORCE_SUFFIXES:
            key = f"{wheel}_{suffix}"
            col = chan_map.get(key)
            if col is None or col not in df.columns:
                continue

            raw = pd.to_numeric(df[col], errors="coerce").values * unit_scale
            mask = ~np.isnan(raw)
            if mask.sum() == 0:
                continue

            vals = raw[mask]
            weights = dist_weights_m[mask]

            r = ranges_dan.get(suffix, [None, None])
            hist_range = (r[0], r[1]) if r[0] is not None else None

            hist_dist, bin_edges = np.histogram(
                vals,
                bins=n_bins,
                weights=weights,
                range=hist_range,
            )
            hist_dist_km = hist_dist / 1000.0  # m → km
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            total_dist_km = float(hist_dist_km.sum())

            # Auto-zoom view range (P1–P99)
            if len(vals) > 0:
                p1 = float(np.percentile(vals, 1))
                p99 = float(np.percentile(vals, 99))
                # Add a 5% margin to the range
                margin = max(abs(p99 - p1) * 0.05, 1.0)
                view_range = [p1 - margin, p99 + margin]
            else:
                view_range = None

            results[key] = {
                "bin_centers":   bin_centers,
                "bin_edges":     bin_edges,
                "hist_dist_km":  hist_dist_km,
                "total_dist_km": total_dist_km,
                "view_range":    view_range,
                "display_unit":  display_unit,
                "wheel":         wheel,
                "suffix":        suffix,
                "channel":       col,
            }
            logger.debug(
                "Distance dist: %s | total=%.1f km", key, total_dist_km
            )

    return results


# ─── Helpers ───────────────────────────────────────────────────────────────

def _get_unit_scale(cfg: dict, display_unit: str) -> float:
    scales = cfg.get("display", {}).get("unit_scales", {"N": 1.0, "daN": 0.1})
    return float(scales.get(display_unit, 1.0))


def _downsample(arr: np.ndarray, max_pts: int = 10_000) -> np.ndarray:
    """Return a uniformly-decimated version of arr capped at max_pts."""
    if len(arr) <= max_pts:
        return arr
    step = max(1, len(arr) // max_pts)
    return arr[::step]
