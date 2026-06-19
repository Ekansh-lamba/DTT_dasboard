"""
engine/channel_map.py
=====================
Name-based channel resolver for the DTT WFT engine.

Design principles:
  - Channels are NEVER identified by column index/position.
  - `find_channel_strict` uses a regex pattern with an exclude-list so that
    columns like 'Latitude' or 'Latacc' are never confused with a lateral
    force channel when searching for 'FL_Fy' etc.
  - The exclude-list is config-driven (channel_exclude_keywords in defaults.yaml).
  - `build_channel_map` returns a flat dict:
      internal_name -> actual_csv_column_name
    for every channel the engine needs.  Missing channels are None (not errors).

Ported from: M&M_PV_WFT_Analyser.find_channel_strict (strictly keyword-regex match
with exclude-list).  Extended to cover all channel types (not just force channels).
"""

from __future__ import annotations

import re
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Wheels and force/moment/speed channel suffixes
WHEELS = ["FL", "FR", "RL", "RR"]
FORCE_SUFFIXES = ["Fx", "Fy", "Fz"]
MOMENT_SUFFIXES = ["Mx", "My", "Mz"]
WS_SUFFIX = "WS1"


def find_channel_strict(
    df: pd.DataFrame,
    wheel: str,
    signal: str,
    exclude_keywords: list[str],
) -> Optional[str]:
    """
    Find a per-wheel channel column by name.

    Matches columns of the form:
        {wheel}[_ -]?{signal}
    (case-insensitive) while rejecting any column whose name contains a
    keyword from `exclude_keywords`.

    Parameters
    ----------
    df : DataFrame whose columns to search
    wheel : e.g. 'FL', 'FR', 'RL', 'RR'
    signal : e.g. 'Fx', 'Fy', 'Fz', 'WS1'
    exclude_keywords : list of lowercase substrings that disqualify a column

    Returns
    -------
    Matched column name, or None if not found.
    """
    pat = re.compile(
        rf"^{re.escape(wheel)}[_\s\-]*{re.escape(signal)}$",
        re.IGNORECASE,
    )
    for col in df.columns:
        name = str(col).strip()
        lname = name.lower()
        # Exclude columns whose names contain disqualifying keywords
        if any(k in lname for k in exclude_keywords):
            continue
        if pat.match(name):
            return name
    return None


def _resolve_scalar_channel(
    df: pd.DataFrame,
    patterns: list[str],
) -> Optional[str]:
    """
    Resolve a non-wheel-specific channel (e.g. Vehicle_Speed, Latacc) by
    trying each pattern in order.  Patterns are full regex strings (compiled
    with IGNORECASE).

    Returns the first matching column name, or None.
    """
    for pat_str in patterns:
        pat = re.compile(pat_str, re.IGNORECASE)
        for col in df.columns:
            if pat.match(str(col).strip()):
                return col
    return None


def build_channel_map(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Build the complete channel map for a loaded DataFrame.

    Returns a dict with keys:
      Scalar channels: 'time', 'dist', 'lat', 'lon', 'alt',
                       'speed', 'latacc', 'longacc', 'yawrate'
      Per-wheel force : '{POS}_{suffix}'  for POS in WHEELS, suffix in FORCE_SUFFIXES
      Per-wheel moment: '{POS}_{suffix}'  for POS in WHEELS, suffix in MOMENT_SUFFIXES
      Per-wheel speed : '{POS}_WS1'       for POS in WHEELS

    All values are the actual CSV column name, or None if the channel is absent.

    Parameters
    ----------
    df  : The loaded DataFrame (with stripped column names)
    cfg : Parsed defaults.yaml dict
    """
    exclude = [k.lower() for k in cfg.get("channel_exclude_keywords", [])]
    patterns = cfg.get("channel_patterns", {})

    chan_map: dict[str, Optional[str]] = {}

    # ── Scalar channels ──────────────────────────────────────────────────────
    for key, pat_list in patterns.items():
        chan_map[key] = _resolve_scalar_channel(df, pat_list)
        if chan_map[key] is None:
            logger.debug("Channel '%s' not found in CSV.", key)

    # ── Per-wheel channels ────────────────────────────────────────────────────
    for wheel in WHEELS:
        # Force channels (Fx, Fy, Fz)
        for suffix in FORCE_SUFFIXES:
            key = f"{wheel}_{suffix}"
            chan_map[key] = find_channel_strict(df, wheel, suffix, exclude)
            if chan_map[key] is None:
                logger.debug("Force channel '%s' not found.", key)

        # Moment channels (Mx, My, Mz)
        for suffix in MOMENT_SUFFIXES:
            key = f"{wheel}_{suffix}"
            chan_map[key] = find_channel_strict(df, wheel, suffix, exclude)
            if chan_map[key] is None:
                logger.debug("Moment channel '%s' not found.", key)

        # Wheel rotational speed (WS1)
        key = f"{wheel}_{WS_SUFFIX}"
        # WS1 is NOT in the exclude list (it's what we're looking for),
        # but we use a custom exclude that omits 'ws1' from the list for this call
        ws_exclude = [k for k in exclude if k not in ("ws1", "rpm")]
        chan_map[key] = find_channel_strict(df, wheel, WS_SUFFIX, ws_exclude)
        if chan_map[key] is None:
            logger.debug("Wheel speed channel '%s' not found.", key)

    return chan_map


def get_available_force_channels(chan_map: dict) -> dict[str, list[str]]:
    """
    Return a dict of wheel → list of available force channel internal keys.
    Only includes channels that have a non-None mapping.

    Example:
        {'FL': ['FL_Fx', 'FL_Fy', 'FL_Fz'], 'FR': [...], ...}
    """
    result = {}
    for wheel in WHEELS:
        available = []
        for suffix in FORCE_SUFFIXES:
            key = f"{wheel}_{suffix}"
            if chan_map.get(key) is not None:
                available.append(key)
        if available:
            result[wheel] = available
    return result


def get_force_channels_flat(chan_map: dict) -> list[str]:
    """
    Return a flat list of all internal force channel keys that have a real column.
    Order: FL_Fx, FL_Fy, FL_Fz, FR_Fx, ... RR_Fz
    """
    out = []
    for wheel in WHEELS:
        for suffix in FORCE_SUFFIXES:
            key = f"{wheel}_{suffix}"
            if chan_map.get(key) is not None:
                out.append(key)
    return out


def log_channel_map(chan_map: dict) -> None:
    """Log the resolved channel map at INFO level."""
    logger.info("=== Resolved channel map ===")
    for k, v in chan_map.items():
        status = v if v is not None else "NOT FOUND"
        logger.info("  %-20s → %s", k, status)
