import logging
from typing import List

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from dtt.config import RunConfig, MANDATORY_CHANNELS

logger = logging.getLogger(__name__)


def _butterworth_lpf(signal: np.ndarray, order: int, cutoff: float, sr: float) -> np.ndarray:
    nyq = 0.5 * sr
    if cutoff >= nyq:
        cutoff = nyq * 0.9
        logger.warning("Cutoff adjusted to %.2f Hz (below Nyquist)", cutoff)
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, signal)


def apply_filter(df: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    if not config.apply_filter:
        logger.info("Filtering skipped (apply_filter=False)")
        return df

    channels = [ch for ch in MANDATORY_CHANNELS if ch in df.columns]
    sr       = config.sampling_rate

    df = df.copy()
    for ch in channels:
        series = pd.to_numeric(df[ch], errors="coerce")
        valid  = series.dropna()
        if len(valid) < 4 * config.filter_order:
            logger.warning("Channel %s too short for filtering, skipping", ch)
            continue
        arr              = series.values.copy()
        finite_mask      = np.isfinite(arr)
        if finite_mask.sum() < 4 * config.filter_order:
            continue
        arr[~finite_mask] = np.nanmean(arr[finite_mask])
        filtered          = _butterworth_lpf(arr, config.filter_order, config.filter_cutoff, sr)
        df[ch]            = filtered

    logger.info(
        "Butterworth LPF applied: order=%d  cutoff=%.1f Hz  SR=%.0f Hz  channels=%d",
        config.filter_order, config.filter_cutoff, sr, len(channels),
    )
    return df
