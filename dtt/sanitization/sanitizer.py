import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from dtt.config import (
    MANDATORY_CHANNELS,
    OUTLIER_LOW_PCTILE,
    OUTLIER_HIGH_PCTILE,
    TIME_COLUMN,
    RunConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class SanitizationReport:
    rows_input:              int
    rows_output:             int
    nan_rows_dropped:        int                = 0
    duplicate_ts_removed:    int                = 0
    nan_filled_per_channel:  Dict[str, int]     = field(default_factory=dict)
    outlier_flags_per_channel: Dict[str, int]   = field(default_factory=dict)
    warnings:                List[str]          = field(default_factory=list)


def _fill_nan_gaps(df: pd.DataFrame, channels: List[str], max_gap: int = 10) -> pd.DataFrame:
    df = df.copy()
    for ch in channels:
        if ch not in df.columns:
            continue
        series = df[ch].copy()
        filled     = series.ffill(limit=max_gap)
        df[ch]     = filled
    return df


def _remove_all_nan_rows(df: pd.DataFrame, channels: List[str]) -> tuple:
    present = [ch for ch in channels if ch in df.columns]
    if not present:
        return df, 0
    mask   = df[present].isna().all(axis=1)
    n_drop = int(mask.sum())
    df     = df[~mask].reset_index(drop=True)
    return df, n_drop


def _remove_duplicate_timestamps(df: pd.DataFrame) -> tuple:
    if TIME_COLUMN not in df.columns:
        return df, 0
    before = len(df)
    df     = df.drop_duplicates(subset=[TIME_COLUMN], keep="first").reset_index(drop=True)
    n_removed = before - len(df)
    return df, n_removed


def _flag_outliers(df: pd.DataFrame, channels: List[str]) -> Dict[str, int]:
    flag_counts = {}
    for ch in channels:
        if ch not in df.columns:
            continue
        col  = pd.to_numeric(df[ch], errors="coerce")
        lo   = np.nanpercentile(col.dropna(), OUTLIER_LOW_PCTILE)
        hi   = np.nanpercentile(col.dropna(), OUTLIER_HIGH_PCTILE)
        mask = (col < lo) | (col > hi)
        n    = int(mask.sum())
        if n > 0:
            flag_counts[ch] = n
    return flag_counts


def sanitize(df: pd.DataFrame, config: RunConfig) -> tuple:
    report = SanitizationReport(rows_input=len(df), rows_output=0)
    rc = getattr(config, "run_channels", None)
    _mandatory = rc.mandatory_channels if rc is not None and rc.mandatory_channels \
        else MANDATORY_CHANNELS
    force_channels = [ch for ch in _mandatory if ch in df.columns]

    df, n_ts_dup = _remove_duplicate_timestamps(df)
    report.duplicate_ts_removed = n_ts_dup
    if n_ts_dup:
        logger.info("Removed %d duplicate timestamps", n_ts_dup)

    df, n_dropped = _remove_all_nan_rows(df, force_channels)
    report.nan_rows_dropped = n_dropped
    if n_dropped:
        logger.info("Dropped %d all-NaN rows", n_dropped)

    nan_before = {ch: int(df[ch].isna().sum()) for ch in force_channels if ch in df.columns}
    df = _fill_nan_gaps(df, force_channels, max_gap=10)
    nan_after  = {ch: int(df[ch].isna().sum()) for ch in force_channels if ch in df.columns}
    for ch in force_channels:
        n_filled = nan_before.get(ch, 0) - nan_after.get(ch, 0)
        if n_filled > 0:
            report.nan_filled_per_channel[ch] = n_filled

    report.outlier_flags_per_channel = _flag_outliers(df, force_channels)
    if report.outlier_flags_per_channel:
        total_flags = sum(report.outlier_flags_per_channel.values())
        logger.info("Outlier flags (P1/P99): %d samples across %d channels",
                    total_flags, len(report.outlier_flags_per_channel))

    report.rows_output = len(df)

    out_path = config.run_output_dir / "processed_data.csv"
    df.to_csv(out_path, index=False)
    logger.info("Processed data saved: %s  (%d rows)", out_path, len(df))

    return df, report
