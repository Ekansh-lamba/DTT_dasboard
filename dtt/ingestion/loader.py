import re
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from dtt.config import (
    MANDATORY_CHANNELS,
    N_TO_DAN_THRESHOLD,
    N_TO_DAN_FACTOR,
    EXCLUDE_KEYWORDS,
    TIME_COLUMN,
    RunConfig,
)

logger = logging.getLogger(__name__)


def _looks_like_time_series(arr) -> bool:
    try:
        vals = pd.to_numeric(arr, errors="coerce").dropna().values
        if len(vals) < 3:
            return False
        diffs = np.diff(vals)
        if np.nanmedian(np.abs(diffs)) == 0:
            return False
        pos_frac   = np.sum(diffs >= -1e-8) / len(diffs)
        med_abs    = np.nanmedian(np.abs(diffs))
        return pos_frac > 0.85 and med_abs < 1.0
    except Exception:
        return False


def _try_load_csv(path: str, skiprows, nrows: Optional[int]) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(
            path,
            encoding="latin1",
            nrows=nrows,
            skiprows=skiprows,
            skip_blank_lines=True,
            skipinitialspace=True,
            engine="c",
            low_memory=False,
        )
        return df
    except Exception:
        try:
            df = pd.read_csv(
                path,
                encoding="latin1",
                nrows=nrows,
                skiprows=skiprows,
                skip_blank_lines=True,
                skipinitialspace=True,
                engine="python",
            )
            return df
        except Exception:
            return None


def _rename_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    for i, c in enumerate(df.columns):
        s = str(c).strip()
        if s == "" or s.lower().startswith("unnamed"):
            try:
                if _looks_like_time_series(df.iloc[:, i]):
                    new_cols.append(TIME_COLUMN)
                else:
                    new_cols.append(f"Unnamed_{i}")
            except Exception:
                new_cols.append(f"Unnamed_{i}")
        else:
            new_cols.append(s)
    df.columns = new_cols
    return df


def _sanitize_header(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if len(df) == 0:
        return df
    first = df.iloc[0]
    non_num = 0
    total   = len(first)
    for v in first:
        if v is None:
            non_num += 1
            continue
        s = str(v).strip()
        try:
            float(s)
        except Exception:
            if not any(ch.isdigit() for ch in s):
                non_num += 1
    if non_num >= 0.3 * total:
        df = df.drop(df.index[0]).reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_channel_strict(df: pd.DataFrame, wheel: str, force: str) -> Optional[str]:
    if df is None:
        return None
    pat = re.compile(
        rf"^{re.escape(wheel)}[_\s\-]*{re.escape(force)}$",
        re.IGNORECASE,
    )
    for c in df.columns:
        name  = str(c).strip()
        lname = name.lower()
        if any(k in lname for k in EXCLUDE_KEYWORDS):
            continue
        if pat.match(name):
            return name
    return None


def _detect_force_channels(df: pd.DataFrame) -> list:
    found = []
    for ch in MANDATORY_CHANNELS:
        if ch in df.columns:
            found.append(ch)
    if not found:
        wheel_parts = ["FL", "FR", "RL", "RR"]
        force_parts = ["Fx", "Fy", "Fz"]
        for w in wheel_parts:
            for f in force_parts:
                strict = find_channel_strict(df, w, f)
                if strict:
                    found.append(strict)
    return found


def _apply_n_to_dan(df: pd.DataFrame, channels: list) -> Tuple[pd.DataFrame, bool]:
    if not channels:
        return df, False
    medians = [df[ch].abs().median() for ch in channels if ch in df.columns]
    if not medians:
        return df, False
    max_med = max(m for m in medians if not np.isnan(m))
    if max_med > N_TO_DAN_THRESHOLD:
        df = df.copy()
        for ch in channels:
            if ch in df.columns:
                df[ch] = df[ch] / N_TO_DAN_FACTOR
        logger.info("Applied N->daN conversion (div 10) to %d channels", len(channels))
        return df, True
    return df, False


def load_csv(path: Path, config: RunConfig) -> Tuple[pd.DataFrame, dict]:
    path_str = str(path)
    logger.info("Loading CSV: %s", path_str)

    df_raw = None
    for skip in (2, 1, 0, 3):
        candidate = _try_load_csv(path_str, skip, None)
        if candidate is not None:
            candidate = _rename_unnamed_columns(candidate)
            candidate = _sanitize_header(candidate)
            candidate.columns = [str(c).strip() for c in candidate.columns]
            candidate = candidate.apply(pd.to_numeric, errors="coerce")
            if _detect_force_channels(candidate):
                df_raw = candidate
                logger.info("CSV loaded with skiprows=%d", skip)
                break

    if df_raw is None:
        raise ValueError(
            f"Could not parse CSV: {path.name}. No force channels detected."
        )

    force_channels = _detect_force_channels(df_raw)
    df_raw         = df_raw.dropna(subset=force_channels, how="all")
    df_raw, dan_applied = _apply_n_to_dan(df_raw, force_channels)

    sr = config.sampling_rate
    if TIME_COLUMN in df_raw.columns:
        time_col = pd.to_numeric(df_raw[TIME_COLUMN], errors="coerce").dropna()
        if len(time_col) > 1:
            dt = float(np.nanmedian(np.diff(time_col.values)))
            if dt > 0:
                sr = round(1.0 / dt, 2)
                logger.info("Sampling rate detected from Time column: %.2f Hz", sr)

    duration_s = len(df_raw) / sr

    metadata = {
        "file_name":       path.name,
        "rows":            len(df_raw),
        "columns":         len(df_raw.columns),
        "force_channels":  force_channels,
        "sampling_rate_hz": sr,
        "duration_s":      round(duration_s, 2),
        "n_to_dan_applied": dan_applied,
    }

    logger.info(
        "Loaded: %d rows, %d columns, %.1f s, SR=%.0f Hz",
        len(df_raw), len(df_raw.columns), duration_s, sr,
    )
    return df_raw, metadata
