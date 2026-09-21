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
    """Bridge short NaN gaps in the WFT force channels by linear interpolation.

    ``max_gap`` is in samples; a run of NaN longer than that is left alone,
    because a long dropout is missing data and inventing a ramp across it would
    be fabrication.

    Linear, not forward-fill. ``ffill`` holds the last known load flat across
    the gap, and a WFT force channel is almost never constant -- so the fill
    itself becomes a plateau with a step at each end. ``smo(0.1)`` then spreads
    that step across its 99-tap kernel into the surrounding real signal, the
    plateau shows up in rainflow counting as a load excursion that never
    happened, and the mean/sigma either side of the gap shift. Interpolation
    matches what the operators in :mod:`dtt.preprocessing` already do when they
    bridge gaps for FiltLP and smo.
    """
    df = df.copy()
    for ch in channels:
        if ch not in df.columns:
            continue
        series = pd.to_numeric(df[ch], errors="coerce")
        gap = series.isna().to_numpy()
        if not gap.any():
            df[ch] = series
            continue
        # `limit=max_gap` alone does NOT express "leave long gaps alone":
        # pandas counts the limit per direction, so with limit_direction="both"
        # a 15-sample gap gets 10 filled from the left and 10 from the right and
        # closes completely. Interpolate the interior, then put back every run
        # that was longer than max_gap.
        filled = series.interpolate(method="linear", limit_area="inside")
        edges = np.flatnonzero(np.diff(np.r_[0, gap.view(np.int8), 0]))
        for start, stop in zip(edges[::2], edges[1::2]):
            if stop - start > max_gap:
                filled.iloc[start:stop] = np.nan
        df[ch] = filled
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


# A percentile band removes a fixed fraction of every channel whether or not
# anything is actually wrong with it: P1/P99 blanks ~2 % of samples by
# construction. Warn when the blanked fraction sits at that floor, because then
# the band is clipping the load distribution rather than catching artifacts.
_EXPECTED_BAND_TOLERANCE = 0.15      # relative; 2.00 % vs an expected 2.00 %


def _flag_outliers(df: pd.DataFrame,
                   channels: List[str]) -> tuple[pd.DataFrame, Dict[str, int]]:
    """Blank samples outside the [P_low, P_high] band and count them.

    This used to count without blanking, so the report said an artifact had been
    detected while the frame handed downstream still carried it: a spike five
    times the physical maximum went into ``smo(0.1)``, spread across the 99-tap
    kernel, and landed in the fatigue analysis. Counting and removing are now
    the same pass, so the report and the data cannot disagree.

    Blanked samples are left as NaN rather than refilled here. The FAMOS
    operators downstream bridge gaps for their own pass and restore them
    afterwards, which is the project's established NaN policy; filling them in
    this stage would hide the gap from that machinery.

    Returns the modified frame alongside the counts -- the caller must take the
    frame, or the blanking is silently discarded.
    """
    flag_counts: Dict[str, int] = {}
    df = df.copy()
    expected_frac = (OUTLIER_LOW_PCTILE + (100.0 - OUTLIER_HIGH_PCTILE)) / 100.0
    for ch in channels:
        if ch not in df.columns:
            continue
        col = pd.to_numeric(df[ch], errors="coerce")
        valid = col.dropna()
        if valid.size < 2:
            continue
        lo = np.nanpercentile(valid, OUTLIER_LOW_PCTILE)
        hi = np.nanpercentile(valid, OUTLIER_HIGH_PCTILE)
        mask = (col < lo) | (col > hi)
        n = int(mask.sum())
        if n > 0:
            df.loc[mask, ch] = np.nan
            flag_counts[ch] = n
            frac = n / max(1, int(col.notna().sum()))
            if abs(frac - expected_frac) <= _EXPECTED_BAND_TOLERANCE * expected_frac:
                logger.warning(
                    "%s: blanked %d samples (%.2f %%), which is the %.2f %% the "
                    "P%g/P%g band removes from any distribution -- this is "
                    "clipping the load range (kept %.6g..%.6g), not removing "
                    "artifacts. A physical limit would target real faults.",
                    ch, n, 100 * frac, 100 * expected_frac,
                    OUTLIER_LOW_PCTILE, OUTLIER_HIGH_PCTILE, lo, hi)
    return df, flag_counts


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

    # Take the frame back, not just the counts -- the blanking lives in it.
    df, report.outlier_flags_per_channel = _flag_outliers(df, force_channels)
    if report.outlier_flags_per_channel:
        total_flags = sum(report.outlier_flags_per_channel.values())
        logger.info("Outlier flags (P1/P99): %d samples across %d channels",
                    total_flags, len(report.outlier_flags_per_channel))

    report.rows_output = len(df)
    # processed_data.csv is written by the pipeline *after* signal conditioning
    # (stage 4). Writing it here published the un-filtered frame while every
    # statistic, histogram and plot was computed from the filtered one.
    return df, report
