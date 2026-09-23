import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from dtt.config import (
    MANDATORY_CHANNELS,
    OUTLIER_LOW_PCTILE,
    OUTLIER_HIGH_PCTILE,
    OUTLIER_FENCE_WIDTHS,
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


def _flag_outliers(df: pd.DataFrame,
                   channels: List[str]) -> tuple[pd.DataFrame, Dict[str, int]]:
    """Blank gross artefacts in the force channels and count them.

    A sample is an artefact when it sits more than ``OUTLIER_FENCE_WIDTHS``
    band-widths outside the channel's [P1, P99] band. The band alone is not the
    fence: blanking everything outside P1..P99 removes exactly 2 % of every
    channel whatever the data -- its highest and lowest real loads. On the
    reference studies that flat-topped every Fx/Fy/Fz trace at its P1/P99 (the
    "over-filtered" look), left 8-11 k NaN holes per channel in
    processed_data.csv, and took the peaks that dominate fatigue damage out of
    rainflow, histograms and box plots. One band-width out, the same recordings
    lose 0-93 samples per channel: the DAQ glitches, not the road.

    This used to count without blanking, so the report said an artifact had been
    detected while the frame handed downstream still carried it. Counting and
    removing are the same pass, so the report and the data cannot disagree.

    Blanked samples are left as NaN rather than refilled here. The FAMOS
    operators downstream bridge gaps for their own pass and restore them
    afterwards, which is the project's established NaN policy.

    Returns the modified frame alongside the counts -- the caller must take the
    frame, or the blanking is silently discarded.
    """
    flag_counts: Dict[str, int] = {}
    df = df.copy()
    for ch in channels:
        if ch not in df.columns:
            continue
        col = pd.to_numeric(df[ch], errors="coerce")
        valid = col.dropna()
        if valid.size < 2:
            continue
        lo = np.nanpercentile(valid, OUTLIER_LOW_PCTILE)
        hi = np.nanpercentile(valid, OUTLIER_HIGH_PCTILE)
        reach = OUTLIER_FENCE_WIDTHS * (hi - lo)
        mask = (col < lo - reach) | (col > hi + reach)
        n = int(mask.sum())
        if n > 0:
            df.loc[mask, ch] = np.nan
            flag_counts[ch] = n
            logger.info("%s: blanked %d artefact samples outside %.6g..%.6g "
                        "(P%g/P%g band +/- %g widths)", ch, n, lo - reach,
                        hi + reach, OUTLIER_LOW_PCTILE, OUTLIER_HIGH_PCTILE,
                        OUTLIER_FENCE_WIDTHS)
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
        logger.info("Outlier flags (P1/P99 fence): %d samples across %d channels",
                    total_flags, len(report.outlier_flags_per_channel))

    report.rows_output = len(df)
    # processed_data.csv is written by the pipeline *after* signal conditioning
    # (stage 4). Writing it here published the un-filtered frame while every
    # statistic, histogram and plot was computed from the filtered one.
    return df, report
