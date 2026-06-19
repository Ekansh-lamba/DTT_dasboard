"""
engine/histograms.py
====================
Force distribution histogram engine for the DTT WFT analysis.

Implements §6.1 of the build spec:
  - Distributions of Fx, Fy, Fz for each wheel position
  - Distance-weighted option (weight each sample by distance increment from Dist col)
  - P80 / P90 / P95 percentile markers
  - Configurable bins / ranges from config/defaults.yaml
  - Deterministic N→daN conversion from declared input unit (never inferred)

Anti-patterns avoided:
  - No hard-coded bin counts or axis ranges
  - No unit inference from data magnitude
  - Warning emitted (not silent clipping) when samples fall outside histogram range
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_force_histograms(
    df: pd.DataFrame,
    chan_map: dict,
    cfg: dict,
    wheels: Optional[list[str]] = None,
    display_unit: str = "daN",
    use_distance_weight: bool = False,
) -> dict:
    """
    Compute force histograms for specified wheel positions and Fx/Fy/Fz channels.

    Parameters
    ----------
    df            : Loaded DataFrame (full or range-filtered)
    chan_map      : Channel map from build_channel_map()
    cfg           : Parsed defaults.yaml dict
    wheels        : List of wheel positions to compute (None = all available)
    display_unit  : 'N' or 'daN' — controls axis scaling
    use_distance_weight : If True, weight each sample by distance increment (m)
                          giving a distance-based histogram (Y = km travelled at that force)

    Returns
    -------
    dict keyed by internal channel name (e.g. 'FL_Fx'), each value is:
    {
        'bin_centers': np.ndarray,      # in display_unit
        'bin_edges':   np.ndarray,      # in display_unit
        'hist_count':  np.ndarray,      # sample counts (unweighted)
        'hist_dist_km':np.ndarray,      # distance in km per bin (distance-weighted)
        'hist_active': np.ndarray,      # the histogram to display (count or dist based on flag)
        'percentiles': {P: value},      # in display_unit
        'n_clipped':   int,             # samples outside histogram range (warn if high)
        'n_valid':     int,             # valid (non-NaN) samples
        'display_unit': str,
        'channel':     str,             # CSV column name
        'wheel':       str,
        'suffix':      str,             # Fx / Fy / Fz
    }
    """
    from .channel_map import WHEELS, FORCE_SUFFIXES

    if wheels is None:
        wheels = WHEELS

    # ── Config ────────────────────────────────────────────────────────────
    hist_cfg = cfg.get("histogram", {})
    n_bins = hist_cfg.get("bins", 60)
    pct_list = hist_cfg.get("percentiles", [80, 90, 95])
    clip_warn = hist_cfg.get("clip_warn_fraction", 0.01)
    autozoom = hist_cfg.get("autozoom_view", True)

    unit_scale = _get_unit_scale(cfg, display_unit)
    ranges_dan = hist_cfg.get("force_ranges_dan", {
        "Fx": [-300, 300], "Fy": [-300, 300], "Fz": [100, 1000]
    })

    # ── Distance weights ──────────────────────────────────────────────────
    dist_weights_m = _get_distance_weights(df, chan_map) if use_distance_weight else None

    results: dict = {}

    for wheel in wheels:
        for suffix in FORCE_SUFFIXES:
            key = f"{wheel}_{suffix}"
            col = chan_map.get(key)
            if col is None or col not in df.columns:
                logger.debug("Skipping histogram for '%s' — column not available.", key)
                continue

            # Raw values in N (input unit)
            raw = pd.to_numeric(df[col], errors="coerce").values
            n_valid = int(np.sum(~np.isnan(raw)))

            # Convert to display unit deterministically
            vals_display = raw * unit_scale  # N × 0.1 → daN, or N × 1.0 → N

            # Range for this force type in display unit
            r = ranges_dan.get(suffix, [None, None])
            hist_range = (r[0], r[1]) if r[0] is not None else None

            # Distance weights for valid samples only
            if dist_weights_m is not None:
                w = dist_weights_m.copy()
            else:
                w = np.ones(len(vals_display))

            mask = ~np.isnan(vals_display)

            # ── Clipping warning ──────────────────────────────────────────
            n_clipped = 0
            if hist_range is not None and mask.sum() > 0:
                in_range = (vals_display[mask] >= hist_range[0]) & \
                           (vals_display[mask] <= hist_range[1])
                n_clipped = int(np.sum(~in_range))
                clip_frac = n_clipped / max(mask.sum(), 1)
                if clip_frac > clip_warn:
                    logger.warning(
                        "%s: %.1f%% of samples (n=%d) fall outside histogram range "
                        "[%.0f, %.0f] %s — peaks may be silently dropped. "
                        "Consider widening the range in config/defaults.yaml.",
                        key, clip_frac * 100, n_clipped,
                        hist_range[0], hist_range[1], display_unit,
                    )

            # ── Unweighted count histogram ────────────────────────────────
            hist_count, bin_edges = np.histogram(
                vals_display[mask],
                bins=n_bins,
                range=hist_range,
            )

            # ── Distance-weighted histogram ───────────────────────────────
            w_masked = w[mask]
            hist_dist, _ = np.histogram(
                vals_display[mask],
                bins=bin_edges,
                weights=w_masked,
            )
            hist_dist_km = hist_dist / 1000.0  # m → km

            # ── Percentiles ───────────────────────────────────────────────
            valid_vals = vals_display[mask]
            percentiles = {}
            for p in pct_list:
                if len(valid_vals) > 0:
                    percentiles[p] = float(np.percentile(valid_vals, p))
                else:
                    percentiles[p] = float("nan")

            # ── Auto-zoom view range (P1–P99) ─────────────────────────────
            # view_range is the axis range the UI should display (auto-zoomed).
            # bin_edges / percentiles are computed on the full hist_range — unchanged.
            if autozoom and len(valid_vals) >= 2:
                p1  = float(np.percentile(valid_vals, 1))
                p99 = float(np.percentile(valid_vals, 99))
                # Pad by 10% of the P1–P99 span so bars are not flush with axis
                span = max(p99 - p1, 1.0)
                view_range = [p1 - 0.05 * span, p99 + 0.05 * span]
            else:
                r = ranges_dan.get(suffix, [None, None])
                view_range = r if r[0] is not None else [None, None]

            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            results[key] = {
                "bin_centers":  bin_centers,
                "bin_edges":    bin_edges,
                "hist_count":   hist_count,
                "hist_dist_km": hist_dist_km,
                "hist_active":  hist_dist_km if use_distance_weight else hist_count,
                "percentiles":  percentiles,
                "view_range":   view_range,   # P1–P99 span for axis auto-zoom
                "n_clipped":    n_clipped,
                "n_valid":      n_valid,
                "display_unit": display_unit,
                "channel":      col,
                "wheel":        wheel,
                "suffix":       suffix,
            }
            logger.debug(
                "Histogram computed: %s | %d valid samples | P90=%.1f %s",
                key, n_valid, percentiles.get(90, float("nan")), display_unit,
            )

    return results


# ─── Helpers ───────────────────────────────────────────────────────────────

def _get_unit_scale(cfg: dict, display_unit: str) -> float:
    """
    Return the multiplicative scale factor to convert N → display_unit.
    Source of truth is cfg.display.unit_scales; fallback is 1.0 for N, 0.1 for daN.
    Never inferred from data magnitude.
    """
    scales = cfg.get("display", {}).get("unit_scales", {"N": 1.0, "daN": 0.1})
    scale = scales.get(display_unit, None)
    if scale is None:
        logger.warning("Unknown display unit '%s'; defaulting to N (scale=1.0).", display_unit)
        scale = 1.0
    return float(scale)


def _get_distance_weights(df: pd.DataFrame, chan_map: dict) -> np.ndarray:
    """
    Compute per-sample distance increment weights from the cumulative Dist column.

    Dist is a cumulative odometer in metres.
    np.diff gives the increment per sample (metres moved per time step).
    The weight array is the same length as the DataFrame (first sample gets 0 weight).

    Returns
    -------
    np.ndarray of float32, shape (n_rows,), values in metres.
    """
    dist_col = chan_map.get("dist")
    if dist_col is None:
        for candidate in ("Dist", "Distance"):
            if candidate in df.columns:
                dist_col = candidate
                break

    if dist_col is None or dist_col not in df.columns:
        logger.warning(
            "Distance column not found; distance-weighted histogram will use uniform weights."
        )
        return np.ones(len(df), dtype=np.float32)

    dist_vals = pd.to_numeric(df[dist_col], errors="coerce").ffill().values
    # Per-sample increment (metres)
    increments = np.empty(len(dist_vals), dtype=np.float32)
    increments[0] = 0.0
    increments[1:] = np.diff(dist_vals).clip(min=0.0).astype(np.float32)

    # Sanity: if all increments are zero (constant dist col), fall back to uniform
    if increments.sum() < 1.0:
        logger.warning(
            "Dist column increments sum to < 1 m — may not be a cumulative odometer. "
            "Falling back to uniform weights."
        )
        return np.ones(len(df), dtype=np.float32)

    return increments
