"""
engine/heatmaps.py
==================
2D correlation density heatmap engine for the DTT WFT analysis.

Implements §6.2 of the build spec:
  - Three pairings per wheel: Fx×Fy, Fz×Fy, Fz×Fx
  - 2D binned density grid with per-cell percentage of total samples
  - Configurable bin resolution (from config/defaults.yaml)
  - Deterministic N→daN conversion

Ported / inspired by:
  PV_WFT_Analyser_Final.show_heatmap_fx_fy / show_heatmap_fz_fy / show_heatmap_fz_fx
  — the % annotation style and bin logic is reproduced; hexbin replaced by
  proper 2D histogram so we get exact bin boundaries and labelled % cells
  that match the Apollo reference chart style.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard heatmap pairings: (x_suffix, y_suffix, label)
HEATMAP_PAIRINGS = [
    ("Fx", "Fy", "Fx×Fy"),
    ("Fz", "Fy", "Fz×Fy"),
    ("Fz", "Fx", "Fz×Fx"),
]


def compute_heatmap(
    df: pd.DataFrame,
    chan_map: dict,
    wheel: str,
    x_suffix: str,
    y_suffix: str,
    cfg: dict,
    display_unit: str = "daN",
) -> dict:
    """
    Compute a single 2D correlation density heatmap for one wheel and one pairing.

    Parameters
    ----------
    df           : Loaded DataFrame
    chan_map     : Channel map from build_channel_map()
    wheel        : 'FL', 'FR', 'RL', or 'RR'
    x_suffix     : e.g. 'Fx', 'Fz'  — the X-axis force channel suffix
    y_suffix     : e.g. 'Fy', 'Fx'  — the Y-axis force channel suffix
    cfg          : Parsed defaults.yaml dict
    display_unit : 'N' or 'daN'

    Returns
    -------
    dict with:
        H_pct    : 2D np.ndarray (n_x_bins, n_y_bins) — % of total samples
        xedges   : bin edges on X axis (in display_unit)
        yedges   : bin edges on Y axis (in display_unit)
        x_label  : str  (e.g. 'Fx (daN)')
        y_label  : str
        title    : str
        wheel    : str
        pairing  : str  (e.g. 'Fx×Fy')
        n_valid  : int
        display_unit: str
    """
    hm_cfg = cfg.get("heatmap", {})
    n_bins = hm_cfg.get("bins", 20)
    ranges_dan = hm_cfg.get("ranges_dan", {
        "Fx": [-300, 300], "Fy": [-300, 300], "Fz": [100, 1000]
    })
    unit_scale = _get_unit_scale(cfg, display_unit)

    # Resolve column names
    x_key = f"{wheel}_{x_suffix}"
    y_key = f"{wheel}_{y_suffix}"
    x_col = chan_map.get(x_key)
    y_col = chan_map.get(y_key)

    if x_col is None or y_col is None:
        logger.warning("Heatmap %s-%s×%s: one or both columns missing.", wheel, x_suffix, y_suffix)
        return _empty_heatmap(wheel, x_suffix, y_suffix, display_unit)

    if x_col not in df.columns or y_col not in df.columns:
        logger.warning("Heatmap %s-%s×%s: columns not in DataFrame.", wheel, x_suffix, y_suffix)
        return _empty_heatmap(wheel, x_suffix, y_suffix, display_unit)

    # Extract and scale values
    x_raw = pd.to_numeric(df[x_col], errors="coerce").values * unit_scale
    y_raw = pd.to_numeric(df[y_col], errors="coerce").values * unit_scale
    mask = ~(np.isnan(x_raw) | np.isnan(y_raw))
    x_vals = x_raw[mask]
    y_vals = y_raw[mask]
    n_valid = int(mask.sum())

    if n_valid == 0:
        logger.warning("Heatmap %s-%s×%s: no valid data.", wheel, x_suffix, y_suffix)
        return _empty_heatmap(wheel, x_suffix, y_suffix, display_unit)

    # Build bin edges from config ranges
    x_range = ranges_dan.get(x_suffix, [None, None])
    y_range = ranges_dan.get(y_suffix, [None, None])

    if x_range[0] is not None:
        x_edges = np.linspace(x_range[0], x_range[1], n_bins + 1)
    else:
        x_edges = np.linspace(x_vals.min(), x_vals.max(), n_bins + 1)

    if y_range[0] is not None:
        y_edges = np.linspace(y_range[0], y_range[1], n_bins + 1)
    else:
        y_edges = np.linspace(y_vals.min(), y_vals.max(), n_bins + 1)

    # 2D histogram
    H, xedges, yedges = np.histogram2d(x_vals, y_vals, bins=[x_edges, y_edges])
    total = H.sum()
    H_pct = (H / total * 100.0) if total > 0 else H

    unit_str = display_unit
    pairing = f"{x_suffix}×{y_suffix}"

    return {
        "H_pct":        H_pct,
        "xedges":       xedges,
        "yedges":       yedges,
        "x_label":      f"{x_suffix} ({unit_str})",
        "y_label":      f"{y_suffix} ({unit_str})",
        "title":        f"{wheel} — {pairing}",
        "wheel":        wheel,
        "pairing":      pairing,
        "x_suffix":     x_suffix,
        "y_suffix":     y_suffix,
        "n_valid":      n_valid,
        "display_unit": display_unit,
    }


def compute_all_heatmaps(
    df: pd.DataFrame,
    chan_map: dict,
    cfg: dict,
    wheels: Optional[list[str]] = None,
    display_unit: str = "daN",
) -> dict:
    """
    Compute all three heatmap pairings (Fx×Fy, Fz×Fy, Fz×Fx) for each wheel.

    Parameters
    ----------
    df           : Loaded DataFrame
    chan_map     : Channel map
    cfg          : Config dict
    wheels       : List of wheel positions (None = all available)
    display_unit : 'N' or 'daN'

    Returns
    -------
    dict keyed by (wheel, pairing) tuples:
        { ('FL', 'Fx×Fy'): {...}, ('FL', 'Fz×Fy'): {...}, ... }
    """
    from .channel_map import WHEELS as ALL_WHEELS
    if wheels is None:
        wheels = ALL_WHEELS

    results = {}
    for wheel in wheels:
        for x_suffix, y_suffix, pairing in HEATMAP_PAIRINGS:
            result = compute_heatmap(
                df, chan_map, wheel, x_suffix, y_suffix, cfg, display_unit
            )
            results[(wheel, pairing)] = result
            logger.debug(
                "Heatmap %s-%s: %d valid samples", wheel, pairing, result.get("n_valid", 0)
            )

    return results


# ─── Helpers ───────────────────────────────────────────────────────────────

def _get_unit_scale(cfg: dict, display_unit: str) -> float:
    scales = cfg.get("display", {}).get("unit_scales", {"N": 1.0, "daN": 0.1})
    return float(scales.get(display_unit, 1.0))


def _empty_heatmap(wheel: str, x_suffix: str, y_suffix: str, display_unit: str) -> dict:
    return {
        "H_pct":        np.zeros((1, 1)),
        "xedges":       np.array([0.0, 1.0]),
        "yedges":       np.array([0.0, 1.0]),
        "x_label":      f"{x_suffix} ({display_unit})",
        "y_label":      f"{y_suffix} ({display_unit})",
        "title":        f"{wheel} — {x_suffix}×{y_suffix} (no data)",
        "wheel":        wheel,
        "pairing":      f"{x_suffix}×{y_suffix}",
        "x_suffix":     x_suffix,
        "y_suffix":     y_suffix,
        "n_valid":      0,
        "display_unit": display_unit,
    }
