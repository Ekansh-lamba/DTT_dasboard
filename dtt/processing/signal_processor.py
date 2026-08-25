"""Signal conditioning stage.

By default this reproduces the imc/FAMOS recipe rather than a generic
Butterworth pass: WFT forces and moments are smoothed with ``smo(x, 0.1)``,
``Latacc`` gets ``FiltLP(4, 5 Hz)`` then ``smo(x, 0.5)``, and speed/longitudinal
accel get ``smo(x, 0.5)``. See :mod:`dtt.preprocessing` for the operators and
how they were validated against a real FAMOS export.

A 4th-order 10 Hz Butterworth on the force channels — the previous behaviour —
is materially more aggressive than FAMOS's 0.1 s smoothing and shifts every
downstream statistic, histogram and rainflow count. Set ``famos_mode=False``
to get it back.
"""

import logging

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from dtt.config import RunConfig, MANDATORY_CHANNELS
from dtt.preprocessing import apply_famos_recipe, count_conditioned

logger = logging.getLogger(__name__)


def _butterworth_lpf(signal: np.ndarray, order: int, cutoff: float, sr: float) -> np.ndarray:
    nyq = 0.5 * sr
    if cutoff >= nyq:
        cutoff = nyq * 0.9
        logger.warning("Cutoff adjusted to %.2f Hz (below Nyquist)", cutoff)
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, signal)


def _apply_legacy_butterworth(df: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    rc = getattr(config, "run_channels", None)
    _mandatory = rc.mandatory_channels if rc is not None and rc.mandatory_channels \
        else MANDATORY_CHANNELS
    channels = [ch for ch in _mandatory if ch in df.columns]
    sr = config.sampling_rate

    df = df.copy()
    for ch in channels:
        series = pd.to_numeric(df[ch], errors="coerce")
        valid = series.dropna()
        if len(valid) < 4 * config.filter_order:
            logger.warning("Channel %s too short for filtering, skipping", ch)
            continue
        arr = series.values.copy()
        finite_mask = np.isfinite(arr)
        if finite_mask.sum() < 4 * config.filter_order:
            continue
        arr[~finite_mask] = np.nanmean(arr[finite_mask])
        df[ch] = _butterworth_lpf(arr, config.filter_order, config.filter_cutoff, sr)

    logger.info(
        "Butterworth LPF applied: order=%d  cutoff=%.1f Hz  SR=%.0f Hz  channels=%d",
        config.filter_order, config.filter_cutoff, sr, len(channels),
    )
    return df


def apply_filter(df: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    """Condition every channel per the FAMOS recipe (or the legacy Butterworth)."""
    if not config.apply_filter:
        logger.info("Filtering skipped (apply_filter=False)")
        return df

    if not getattr(config, "famos_mode", True):
        return _apply_legacy_butterworth(df, config)

    if getattr(config, "famos_applied", False):
        # imc raw ingestion already ran smo/FiltLP at the native rate, which is
        # the only place it can be done correctly (before red()).
        logger.info("FAMOS recipe already applied during ingestion — not repeating")
        return df

    out, _, applied = apply_famos_recipe(
        df, config.sampling_rate,
        decimate_factor=1,                     # rate is fixed by ingestion
        deglitch=getattr(config, "deglitch", False),
        deglitch_nsigma=getattr(config, "deglitch_nsigma", 6.0),
        despike_enabled=getattr(config, "despike", False),
        despike_rail_min_run=getattr(config, "despike_rail_min_run", 3),
        despike_dropout_max_run=getattr(config, "despike_dropout_max_run", 5),
        despike_hw_cutoff_hz=getattr(config, "despike_hw_cutoff_hz", 200.0),
        despike_net=getattr(config, "despike_net", False),
        despike_net_nsigma=getattr(config, "despike_net_nsigma", 6.0),
        despike_net_window_s=getattr(config, "despike_net_window_s", 0.011),
        transient_enabled=getattr(config, "transient_despike", False),
        transient_pct=getattr(config, "transient_despike_pct", 20.0),
        transient_window_s=getattr(config, "transient_despike_window_s", 1.0),
        transient_noise_floor_mult=getattr(config, "transient_despike_noise_floor_mult", 10.0),
        transient_max_spike_frac=getattr(config, "transient_despike_max_spike_frac", 0.4),
        emit_lpf_columns=False,
    )
    treated = count_conditioned(applied)
    logger.info("FAMOS recipe applied at %.0f Hz: %d/%d channels conditioned",
                config.sampling_rate, treated, len(applied))
    if not treated:
        logger.warning("No channel matched the FAMOS recipe — check the channel names")
    for ch, op in applied.items():
        logger.debug("  %-16s %s", ch, op)
    return out
