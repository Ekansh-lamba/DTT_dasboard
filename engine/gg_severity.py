"""
engine/gg_severity.py
=====================
g-g plot and force severity engine for the DTT WFT analysis.

Implements §6.4 (g-g part) of the build spec:
  - g-g diagram: Longacc (X) vs Latacc (Y), both in g (NOT m/s²)
  - Reference circles at configurable g levels (0.2, 0.4, 0.6 g …)
  - Decimated scatter for display performance (max 50k points by default)
  - Force severity banding: P80 / P90 / P95 of |Fz| per wheel

Critical correctness note:
  The PV_WFT_Analyser_Final script mislabels the g-g axes as "m/s²" with
  ±6 limits.  The Latacc / Longacc channels are declared as g in config —
  no conversion is performed.  Axes are labelled in g with ±1.2 g limits.

Sign convention:
  Applied as a per-channel multiplier (from config sign_conventions).
  Default is SAE J2047 (all multipliers = +1).  ISO 8855 flips Fy and Fz.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_gg(
    df: pd.DataFrame,
    chan_map: dict,
    cfg: dict,
    sign_convention: str = "SAE",
) -> dict:
    """
    Compute the g-g diagram data.

    Parameters
    ----------
    df               : Loaded DataFrame
    chan_map         : Channel map
    cfg              : Parsed defaults.yaml dict
    sign_convention  : 'SAE' or 'ISO'

    Returns
    -------
    dict with:
        longacc_full  : np.ndarray (g) — full resolution
        latacc_full   : np.ndarray (g) — full resolution
        longacc_plot  : np.ndarray (g) — decimated for scatter display
        latacc_plot   : np.ndarray (g) — decimated for scatter display
        decimate_to   : int — max points used for scatter
        n_valid       : int — total valid samples
        axis_limit_g  : float
        ref_circles_g : list[float]
        latacc_col    : str — actual CSV column used
        longacc_col   : str
    """
    gg_cfg = cfg.get("gg", {})
    axis_limit_g = gg_cfg.get("axis_limit_g", 1.2)
    ref_circles = gg_cfg.get("ref_circles_g", [0.2, 0.4, 0.6, 0.8, 1.0])
    decimate_to = gg_cfg.get("decimate_to", 50_000)

    # ── Resolve channels ──────────────────────────────────────────────────
    latacc_col = chan_map.get("latacc")
    longacc_col = chan_map.get("longacc")

    if latacc_col is None or longacc_col is None:
        missing = []
        if latacc_col is None:
            missing.append("latacc")
        if longacc_col is None:
            missing.append("longacc")
        logger.warning("g-g plot: missing channels %s", missing)
        return _empty_gg(axis_limit_g, ref_circles)

    if latacc_col not in df.columns or longacc_col not in df.columns:
        logger.warning("g-g plot: columns not in DataFrame.")
        return _empty_gg(axis_limit_g, ref_circles)

    # ── Extract values ─────────────────────────────────────────────────────
    # Channels are declared as g in config — no unit conversion
    lat = pd.to_numeric(df[latacc_col], errors="coerce").values.astype(np.float32)
    lon = pd.to_numeric(df[longacc_col], errors="coerce").values.astype(np.float32)

    # Apply sign convention multipliers
    sign_mults = cfg.get("sign_conventions", {}).get(sign_convention, {})
    fy_sign = sign_mults.get("Fy", 1)  # Latacc follows Fy convention
    fx_sign = sign_mults.get("Fx", 1)  # Longacc follows Fx convention
    lat = lat * fy_sign
    lon = lon * fx_sign

    mask = ~(np.isnan(lat) | np.isnan(lon))
    lat_valid = lat[mask]
    lon_valid = lon[mask]
    n_valid = int(mask.sum())

    if n_valid == 0:
        logger.warning("g-g plot: no valid data after NaN removal.")
        return _empty_gg(axis_limit_g, ref_circles)

    # ── Decimate for scatter display ──────────────────────────────────────
    if n_valid > decimate_to:
        step = n_valid // decimate_to
        lat_plot = lat_valid[::step]
        lon_plot = lon_valid[::step]
    else:
        lat_plot = lat_valid
        lon_plot = lon_valid

    logger.info(
        "g-g: %d valid samples, %d plotted (step=%d)",
        n_valid, len(lat_plot), max(1, n_valid // decimate_to),
    )

    return {
        "latacc_full":   lat_valid,
        "longacc_full":  lon_valid,
        "latacc_plot":   lat_plot,
        "longacc_plot":  lon_plot,
        "decimate_to":   decimate_to,
        "n_valid":       n_valid,
        "axis_limit_g":  axis_limit_g,
        "ref_circles_g": ref_circles,
        "latacc_col":    latacc_col,
        "longacc_col":   longacc_col,
    }


def compute_force_severity(
    df: pd.DataFrame,
    chan_map: dict,
    cfg: dict,
    display_unit: str = "daN",
) -> dict:
    """
    Compute per-wheel force severity bands from Fz (vertical load).

    Severity is derived from |Fz| percentile banding:
      Low:    |Fz| < P80
      Medium: P80 ≤ |Fz| < P90
      High:   P90 ≤ |Fz| < P95
      Extreme:|Fz| ≥ P95

    Also includes Fx / Fy severity as a percentage of P95 threshold.

    Parameters
    ----------
    df           : Loaded DataFrame
    chan_map     : Channel map
    cfg          : Config dict
    display_unit : 'N' or 'daN'

    Returns
    -------
    dict keyed by wheel, each containing:
      {
        'fz_bands': {  'P80': float, 'P90': float, 'P95': float  },   # in display_unit
        'fx_stats': {  'p90': ..., 'p95': ..., 'max_abs': ...    },
        'fy_stats': {  'p90': ..., 'p95': ..., 'max_abs': ...    },
      }
    """
    from .channel_map import WHEELS

    unit_scale = _get_unit_scale(cfg, display_unit)
    pct_bands = cfg.get("gg", {}).get("severity_percentile_bands", [80, 90, 95])

    result: dict = {}
    for wheel in WHEELS:
        wheel_result: dict = {}

        # Fz severity bands
        fz_key = f"{wheel}_Fz"
        fz_col = chan_map.get(fz_key)
        if fz_col and fz_col in df.columns:
            fz_vals = pd.to_numeric(df[fz_col], errors="coerce").dropna().abs().values
            fz_vals_d = fz_vals * unit_scale
            if len(fz_vals_d) > 0:
                wheel_result["fz_bands"] = {
                    f"P{p}": float(np.percentile(fz_vals_d, p))
                    for p in pct_bands
                }
                # Fraction in each band
                p80 = wheel_result["fz_bands"][f"P{pct_bands[0]}"]
                p90 = wheel_result["fz_bands"][f"P{pct_bands[1]}"]
                p95 = wheel_result["fz_bands"][f"P{pct_bands[2]}"]
                total = len(fz_vals_d)
                wheel_result["fz_band_fractions"] = {
                    "low":     float(np.mean(fz_vals_d < p80)),
                    "medium":  float(np.mean((fz_vals_d >= p80) & (fz_vals_d < p90))),
                    "high":    float(np.mean((fz_vals_d >= p90) & (fz_vals_d < p95))),
                    "extreme": float(np.mean(fz_vals_d >= p95)),
                }

        # Fx / Fy severity stats
        for suffix in ("Fx", "Fy"):
            key = f"{wheel}_{suffix}"
            col = chan_map.get(key)
            if col and col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").dropna().values * unit_scale
                if len(vals) > 0:
                    wheel_result[f"{suffix.lower()}_stats"] = {
                        "p90":     float(np.percentile(np.abs(vals), 90)),
                        "p95":     float(np.percentile(np.abs(vals), 95)),
                        "max_abs": float(np.abs(vals).max()),
                    }

        if wheel_result:
            result[wheel] = wheel_result

    return result


# ─── Helpers ───────────────────────────────────────────────────────────────

def _get_unit_scale(cfg: dict, display_unit: str) -> float:
    scales = cfg.get("display", {}).get("unit_scales", {"N": 1.0, "daN": 0.1})
    return float(scales.get(display_unit, 1.0))


def _empty_gg(axis_limit_g: float, ref_circles: list) -> dict:
    return {
        "latacc_full":   np.array([]),
        "longacc_full":  np.array([]),
        "latacc_plot":   np.array([]),
        "longacc_plot":  np.array([]),
        "decimate_to":   50_000,
        "n_valid":       0,
        "axis_limit_g":  axis_limit_g,
        "ref_circles_g": ref_circles,
        "latacc_col":    None,
        "longacc_col":   None,
    }
