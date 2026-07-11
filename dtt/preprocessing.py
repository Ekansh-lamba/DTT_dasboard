

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.signal import butter, filtfilt


@dataclass
class PreprocessSettings:
    lower_threshold: float = 0.0        # |x| below this -> removed (aim 5)
    remove_outliers: bool = False       # aim 6
    outlier_low_pct: float = 1.0
    outlier_high_pct: float = 99.0
    apply_filter: bool = True
    filter_cutoff: float = 10.0
    filter_order: int = 4
    resample_factor: int = 1            # decimation factor (1 = none)


def apply_lower_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """Blank samples whose magnitude is below ``threshold`` (noise floor)."""
    if threshold <= 0:
        return x
    out = x.astype(float).copy()
    out[np.abs(out) < threshold] = np.nan
    return out


def remove_outliers(x: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    """Blank samples outside the [low_pct, high_pct] percentile band."""
    finite = x[np.isfinite(x)]
    if finite.size < 2:
        return x
    lo = np.percentile(finite, low_pct)
    hi = np.percentile(finite, high_pct)
    out = x.astype(float).copy()
    out[(out < lo) | (out > hi)] = np.nan
    return out


def butterworth_lpf(x: np.ndarray, cutoff: float, order: int, fs: float) -> np.ndarray:
    """Zero-phase Butterworth low-pass, NaN-safe."""
    nyq = 0.5 * fs
    if cutoff >= nyq:
        cutoff = nyq * 0.9
    arr = x.astype(float).copy()
    mask = np.isfinite(arr)
    if mask.sum() < 4 * order:
        return arr
    arr[~mask] = np.nanmean(arr[mask])
    b, a = butter(order, cutoff / nyq, btype="low")
    y = filtfilt(b, a, arr)
    y[~mask] = np.nan
    return y


def decimate_factor(x: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return x
    return x[::factor]


def apply_pipeline(x: np.ndarray, fs: float, s: PreprocessSettings
                   ) -> Tuple[np.ndarray, float]:
    """Apply threshold -> outliers -> filter -> resample. Returns (y, new_fs)."""
    y = np.asarray(x, dtype=float)
    if s.lower_threshold > 0:
        y = apply_lower_threshold(y, s.lower_threshold)
    if s.remove_outliers:
        y = remove_outliers(y, s.outlier_low_pct, s.outlier_high_pct)
    if s.apply_filter:
        y = butterworth_lpf(y, s.filter_cutoff, s.filter_order, fs)
    new_fs = fs
    if s.resample_factor > 1:
        y = decimate_factor(y, s.resample_factor)
        new_fs = fs / s.resample_factor
    return y, new_fs


def summary_stats(x: np.ndarray) -> dict:
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"), "removed": len(x)}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "removed": int(len(x) - finite.size),
    }