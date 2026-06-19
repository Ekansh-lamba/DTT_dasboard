"""
engine/fatigue.py
=================
Rainflow cycle counting + Miner's rule damage ratio engine.

Implements §6.4 (fatigue part) of the build spec.

Ported from: WFT_Analyzer_v5._rainflow_cycles, _run_rf_compare
  - Rainflow cycle extraction via the `rainflow` library
  - Miner's-rule damage ratio:  D = Σ(count × range^m)   (m = configurable slope)
  - Per-wheel and per-axle (front FL+FR / rear RL+RR) comparison
  - Load-range histogram + cumulative distribution per channel
  - Configurable Siemens-style colour steps for from-to matrix

Key design:
  - Signal is DECIMATED before rainflow extraction (for speed on 2.4M rows)
    but the damage ratio is computed on the FULL extracted cycle set, not
    the truncated signal. Decimation affects which local extrema are found,
    not the math once cycles are extracted.
  - Channels are processed independently (pure function per channel).
  - Returns data, not figures — UI layer renders everything.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import rainflow as rf_lib
    _RAINFLOW_AVAILABLE = True
except ImportError:
    logger.error("'rainflow' package not installed. Run: pip install rainflow")
    _RAINFLOW_AVAILABLE = False


# ─── Public API ────────────────────────────────────────────────────────────

def extract_rainflow_cycles(
    signal: np.ndarray,
    max_pts: int = 100_000,
) -> list[tuple]:
    """
    Extract rainflow cycles from a 1-D signal.

    Signal is decimated to max_pts if longer (for speed on large datasets).
    Returns a list of (range, mean, count, i_start, i_end) tuples
    (as returned by rainflow.extract_cycles).

    Parameters
    ----------
    signal  : 1-D array of force values (any unit — caller applies unit scale)
    max_pts : Max length of decimated signal

    Returns
    -------
    List of cycle tuples; empty list if rainflow library unavailable or signal
    is too short.
    """
    if not _RAINFLOW_AVAILABLE:
        return []

    data = np.asarray(signal, dtype=np.float64)
    # Remove NaN
    data = data[~np.isnan(data)]

    if len(data) < 3:
        return []

    # Decimate if necessary
    if len(data) > max_pts:
        step = max(1, len(data) // max_pts)
        data = data[::step]
        logger.debug("Rainflow: decimated signal from %d to %d points", len(signal), len(data))

    try:
        cycles = list(rf_lib.extract_cycles(data))
        return cycles
    except Exception as exc:
        logger.error("Rainflow extraction failed: %s", exc)
        return []


def compute_damage(cycles: list[tuple], m: float = 5.0) -> float:
    """
    Compute Miner's rule damage index.

    D = Σ(count_i × range_i ^ m)

    Parameters
    ----------
    cycles : List of (range, mean, count, ...) tuples from extract_rainflow_cycles
    m      : Wöhler slope exponent (default 5 per Apollo/spec)

    Returns
    -------
    float damage index.  Divide two channels' D values for the damage ratio.
    """
    if not cycles:
        return 0.0
    rng = np.array([c[0] for c in cycles], dtype=np.float64)
    cnt = np.array([c[2] for c in cycles], dtype=np.float64)
    return float(np.sum(cnt * (rng ** m)))


def compute_fatigue_summary(
    df: pd.DataFrame,
    chan_map: dict,
    cfg: dict,
    wheels: Optional[list[str]] = None,
    display_unit: str = "daN",
) -> dict:
    """
    Compute complete fatigue summary for all force channels.

    For each channel:
      - Extracts rainflow cycles (from decimated signal)
      - Computes damage index via Miner's rule
      - Builds range histogram and cumulative exceedance distribution
      - Reports P95 range and total cycle count

    For each force suffix (Fx, Fy, Fz):
      - Compares front axle (FL+FR) damage vs rear axle (RL+RR) damage

    Parameters
    ----------
    df           : Loaded DataFrame
    chan_map     : Channel map
    cfg          : Config dict
    wheels       : Wheel positions (None = all)
    display_unit : 'N' or 'daN'

    Returns
    -------
    dict with:
        'channels': { 'FL_Fx': { ... }, ... }   per-channel results
        'axles':    { 'Fx': { 'front': {...}, 'rear': {...} }, ... }  axle comparison
    """
    from .channel_map import WHEELS as ALL_WHEELS, FORCE_SUFFIXES

    if wheels is None:
        wheels = ALL_WHEELS

    fat_cfg = cfg.get("fatigue", {})
    m = float(fat_cfg.get("miner_slope_m", 5))
    n_rf_bins = int(fat_cfg.get("rf_bins", 35))
    max_pts = int(fat_cfg.get("max_pts_per_channel", 100_000))

    # ── FIXED internal unit (display_unit toggle must NOT affect damage absolute) ──
    # Damage = Σ(n × range^m).  N vs daN differs by 10^m = 100,000×.
    # We always extract cycles in damage_internal_unit; the display_unit is only
    # used for axis labels on the range histogram (converted at plot time).
    internal_unit = fat_cfg.get("damage_internal_unit", "daN")
    scales = cfg.get("display", {}).get("unit_scales", {"N": 1.0, "daN": 0.1})
    internal_scale = float(scales.get(internal_unit, 0.1))  # N × scale → internal_unit
    # Scale to convert internal_unit ranges to display_unit for histogram axes
    # e.g. internal=daN, display=N → display_scale = 10.0
    display_scale_over_internal = float(scales.get(display_unit, 0.1)) / internal_scale

    channel_results: dict = {}

    for wheel in wheels:
        for suffix in FORCE_SUFFIXES:
            key = f"{wheel}_{suffix}"
            col = chan_map.get(key)
            if col is None or col not in df.columns:
                continue

            raw = pd.to_numeric(df[col], errors="coerce").values
            # Apply FIXED internal scale (never display_unit) for rainflow input.
            # This ensures damage absolute values are stable across N⇔daN toggle.
            signal_internal = raw * internal_scale

            logger.info("Extracting rainflow cycles for %s (internal unit=%s) …", key, internal_unit)
            cycles = extract_rainflow_cycles(signal_internal, max_pts=max_pts)

            if not cycles:
                logger.warning("%s: no cycles extracted (too few extrema or library missing).", key)
                channel_results[key] = _empty_channel_result(key, wheel, suffix, display_unit)
                continue

            rng = np.array([c[0] for c in cycles], dtype=np.float64)
            cnt = np.array([c[2] for c in cycles], dtype=np.float64)

            # Repeat cycles by count for percentile / histogram
            rng_rep = np.repeat(rng, cnt.astype(int).clip(1))

            damage = compute_damage(cycles, m=m)  # in internal_unit^m

            # Convert rng to display unit for histogram / percentile labels
            rng_display = rng * display_scale_over_internal
            rng_rep_display = rng_rep * display_scale_over_internal

            # Range histogram (display unit axis)
            p99_rng = float(np.percentile(rng_rep_display, 99))
            bin_edges = np.linspace(0, p99_rng, n_rf_bins + 1)
            hist_counts, _ = np.histogram(rng_rep_display, bins=bin_edges)

            # Cumulative exceedance (display unit axis)
            rng_sorted_display = np.sort(rng_rep_display)[::-1]
            exceedances = np.arange(1, len(rng_sorted_display) + 1)

            channel_results[key] = {
                "cycles":          cycles,
                "rng":             rng_display,        # in display_unit
                "cnt":             cnt,
                "rng_rep":         rng_rep_display,    # in display_unit
                "damage":          damage,             # in internal_unit^m (FIXED)
                "total_cycles":    int(cnt.sum()),
                "p95_range":       float(np.percentile(rng_rep_display, 95)),
                "p90_range":       float(np.percentile(rng_rep_display, 90)),
                "max_range":       float(rng_display.max()),
                "mean_range":      float(rng_display.mean()),
                "range_hist_counts": hist_counts,
                "range_hist_edges":  bin_edges,
                "cumexceed_range":   rng_sorted_display,
                "cumexceed_count":   exceedances,
                "wheel":           wheel,
                "suffix":          suffix,
                "key":             key,
                "display_unit":    display_unit,
                "internal_unit":   internal_unit,
                "miner_m":         m,
            }
            logger.info(
                "%s: %d cycles | damage=%.3e | P95 range=%.1f %s",
                key, int(cnt.sum()), damage,
                float(np.percentile(rng_rep, 95)), display_unit,
            )

    # ── Axle comparisons ──────────────────────────────────────────────────
    axle_results: dict = {}
    for suffix in FORCE_SUFFIXES:
        front_keys = [f"{w}_{suffix}" for w in ("FL", "FR") if f"{w}_{suffix}" in channel_results]
        rear_keys  = [f"{w}_{suffix}" for w in ("RL", "RR") if f"{w}_{suffix}" in channel_results]

        def _axle_combined(keys: list[str]) -> dict:
            """Combine cycles across multiple channels for axle-level stats."""
            all_cycles = []
            for k in keys:
                all_cycles.extend(channel_results[k]["cycles"])
            if not all_cycles:
                return {"damage": 0.0, "total_cycles": 0, "p95_range": 0.0}
            dmg = compute_damage(all_cycles, m=m)
            rng_all = np.array([c[0] for c in all_cycles])
            cnt_all = np.array([c[2] for c in all_cycles])
            rng_rep = np.repeat(rng_all, cnt_all.astype(int).clip(1))
            return {
                "damage": dmg,
                "total_cycles": int(cnt_all.sum()),
                "p95_range": float(np.percentile(rng_rep, 95)) if len(rng_rep) else 0.0,
                "keys": keys,
            }

        if front_keys or rear_keys:
            front_dmg = _axle_combined(front_keys) if front_keys else None
            rear_dmg  = _axle_combined(rear_keys)  if rear_keys  else None
            damage_ratio = None
            damage_norm_front = None
            damage_norm_rear  = None
            p95_range_ratio   = None

            if front_dmg and rear_dmg and front_dmg["damage"] > 0:
                damage_ratio = rear_dmg["damage"] / front_dmg["damage"]
                # Normalised: front = 1.0, rear = ratio
                damage_norm_front = 1.0
                damage_norm_rear  = damage_ratio
            elif rear_dmg and rear_dmg["damage"] > 0 and front_dmg and front_dmg["damage"] == 0:
                damage_norm_rear  = None  # can't normalise to zero

            # P95 range ratio (amplitude story: how much larger are rear ranges?)
            if (front_dmg and front_dmg["p95_range"] > 0
                    and rear_dmg and rear_dmg["p95_range"] > 0):
                p95_range_ratio = rear_dmg["p95_range"] / front_dmg["p95_range"]

            axle_results[suffix] = {
                "front": front_dmg,
                "rear":  rear_dmg,
                "damage_ratio_rear_over_front": damage_ratio,
                "damage_norm_front":            damage_norm_front,
                "damage_norm_rear":             damage_norm_rear,
                "p95_range_ratio":              p95_range_ratio,
                "miner_m":      m,
                "internal_unit": internal_unit,
                "display_unit": display_unit,
            }

    return {
        "channels": channel_results,
        "axles":    axle_results,
    }


# ─── Helpers ───────────────────────────────────────────────────────────────

def _get_unit_scale(cfg: dict, display_unit: str) -> float:
    scales = cfg.get("display", {}).get("unit_scales", {"N": 1.0, "daN": 0.1})
    return float(scales.get(display_unit, 1.0))


def _empty_channel_result(key: str, wheel: str, suffix: str, display_unit: str) -> dict:
    return {
        "cycles":            [],
        "rng":               np.array([]),
        "cnt":               np.array([]),
        "rng_rep":           np.array([]),
        "damage":            0.0,
        "total_cycles":      0,
        "p95_range":         0.0,
        "p90_range":         0.0,
        "max_range":         0.0,
        "mean_range":        0.0,
        "range_hist_counts": np.array([]),
        "range_hist_edges":  np.array([]),
        "cumexceed_range":   np.array([]),
        "cumexceed_count":   np.array([]),
        "wheel":             wheel,
        "suffix":            suffix,
        "key":               key,
        "display_unit":      display_unit,
        "miner_m":           5.0,
    }
