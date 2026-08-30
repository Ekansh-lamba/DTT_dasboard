"""
Frequency analysis + time-synchronisation helpers (FAMOS-style).

* ``amplitude_spectrum_db`` — single-sided FFT amplitude in dB vs frequency,
  the plot FAMOS shows (|dB| against a log-frequency axis).
* ``welch_psd`` — Welch's averaged-periodogram power spectral density. A
  distinct, complementary view to ``amplitude_spectrum_db``: segment
  averaging trades exact spectral lines for a smoothed, statistically
  stable noise floor, which is what a durability/fatigue read of "where is
  the energy" usually wants rather than a single windowed FFT's line detail.
* ``estimate_lag`` — cross-correlation lag (in samples) that best aligns one
  channel to a reference, for time-synchronising channels on a shared axis.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def amplitude_spectrum_db(x: np.ndarray, fs: float,
                          detrend: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum in dB.

    Returns (frequencies_hz, magnitude_db). NaNs are dropped first.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return np.array([1.0]), np.array([-120.0])
    if detrend:
        x = x - np.mean(x)
    n = x.size
    # Hann window reduces spectral leakage (as FAMOS does)
    w = np.hanning(n)
    xw = x * w
    scale = 2.0 / np.sum(w)
    mag = np.abs(np.fft.rfft(xw)) * scale
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    db = 20.0 * np.log10(mag + 1e-12)
    return freq, db


def welch_psd(x: np.ndarray, fs: float,
             nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density: ``(frequencies_hz, psd)``.

    ``nperseg`` defaults to a batch-pipeline-appropriate segment length
    derived from ``fs`` (``min(n, max(256, fs*4))``) — a longer segment than
    a live-GUI tool would use, trading update latency (irrelevant here, this
    runs once per study) for a smoother, more resolved PSD estimate. This is
    an explicit accuracy choice, not a default inherited from anywhere else.

    NaNs are dropped first. Returns a flat near-zero spectrum for input too
    short to window meaningfully.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8 or fs <= 0:
        return np.array([0.0]), np.array([0.0])
    from scipy.signal import welch
    if nperseg is None:
        nperseg = int(min(x.size, max(256, fs * 4)))
    nperseg = max(8, min(nperseg, x.size))
    freq, psd = welch(x, fs=fs, nperseg=nperseg)
    return freq, psd


def log_downsample(f: np.ndarray, db: np.ndarray, n: int = 2000
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce a spectrum to ~n log-spaced points (envelope) for fast plotting."""
    pos = f > 0
    f, db = f[pos], db[pos]
    if f.size <= n:
        return f, db
    edges = np.logspace(np.log10(f[0]), np.log10(f[-1]), n + 1)
    idx = np.searchsorted(f, edges)
    out_f, out_db = [], []
    for i in range(n):
        a, b = idx[i], idx[i + 1]
        if b > a:
            out_f.append(f[a:b].mean())
            out_db.append(db[a:b].max())
    return np.array(out_f), np.array(out_db)


def estimate_lag(ref: np.ndarray, sig: np.ndarray, max_lag: int = 5000) -> int:
    """Integer sample lag that best aligns ``sig`` to ``ref`` (positive = sig
    is delayed and should shift left). Uses FFT cross-correlation on a
    downsampled, mean-removed copy for speed."""
    a = np.asarray(ref, dtype=float)
    b = np.asarray(sig, dtype=float)
    n = min(a.size, b.size)
    if n < 16:
        return 0
    a = a[:n]; b = b[:n]
    a = np.nan_to_num(a - np.nanmean(a))
    b = np.nan_to_num(b - np.nanmean(b))
    # downsample for a fast, robust estimate
    step = max(1, n // 20000)
    a_d, b_d = a[::step], b[::step]
    from scipy.signal import correlate
    corr = correlate(b_d, a_d, mode="full", method="fft")
    lags = np.arange(-len(a_d) + 1, len(b_d))
    lim = max_lag // step
    mid = len(corr) // 2
    lo = max(0, mid - lim)
    hi = min(len(corr), mid + lim + 1)
    best = lags[lo + int(np.argmax(corr[lo:hi]))]
    return int(best * step)