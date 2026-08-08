"""imc/FAMOS-equivalent signal preprocessing.

The operators here reproduce the imc FAMOS sequence in
``imc coding_filtering_Channel mapping.txt`` so the platform's output can be
used in place of a FAMOS export:

    FL_Fx      = smo(FL_Fx1, 0.1)              # every WFT force / moment
    Latacc_LPF = FiltLP(Lat_acc, 0, 0, 4, 5)   # accel only
    Latacc     = smo(Latacc_LPF, 0.5)
    Vehicle_Speed = smo(Speed_kmph, 0.5)
    cutdata    = red(cutd, 10)                 # decimate x10

Both operators were reverse-engineered from a genuine 100 Hz FAMOS export
(``csv/RLDA WFT PV data sample.csv``), which ships ``Latacc`` alongside its
pre-smoothing twin ``Latacc_LPF`` — an exact, sample-wise ground truth:

* ``smo(x, w)`` is a **triangular** kernel, not a single moving average and
  not a boxcar-of-boxcars: the exact kernel has half-width
  ``a = (round(w·fs) - 1) / 2``, i.e. ``h[k] = max(0, 1 - |k|/a)``. Scored
  against a matched raw-vs-processed FAMOS export (``data/Fx_raw_cut.csv``)
  this reproduces FAMOS to max abs error 5.6e-3 N, r = 1.000000000 — a
  boxcar-of-boxcars kernel is a coarser triangle and only reaches r = 0.996.
* ``FiltLP`` is a **causal, single-pass** Butterworth filter (``lfilter``), not
  zero-phase. Scored against a matched raw-vs-processed FAMOS export
  (``data/Fx_raw_cut.csv``), a single ``lfilter`` pass reproduces FAMOS to
  5e-6 max abs error, r = 1.0; ``filtfilt`` misses it by 0.66 and drops r to
  0.93. The filtered channel therefore carries a real lag relative to an
  unfiltered reference — that is correct, not a bug.
* ``red(x, n)`` is a plain every-n-th-sample reduction, applied *after* the
  smoothing has already band-limited the signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, convolve1d
from scipy.signal import butter, filtfilt, lfilter, decimate

# ---------------------------------------------------------------- FAMOS recipe

FAMOS_FORCE_SMOOTH_S = 0.1      # smo(FL_Fx1, 0.1)  — WFT forces & moments
FAMOS_AUX_SMOOTH_S = 0.5        # smo(Latacc_LPF, 0.5), smo(Speed_kmph, 0.5)
FAMOS_LPF_ORDER = 4             # FiltLP(..., 4, 5)
FAMOS_LPF_CUTOFF = 5.0
FAMOS_DECIMATE = 10             # red(cutd, 10)

_WFT_CHANNEL_RE = re.compile(r"^(FL|FR|RL|RR)_[FM][xyz]$", re.I)
_ACCEL_LPF_CHANNELS = {"latacc", "lat_acc"}                  # FiltLP then smo
_ACCEL_SMO_CHANNELS = {"longacc", "long_acc", "vehicle_speed", "speed_kmph"}
_PASSTHROUGH_CHANNELS = {"time", "dist", "latitude", "longitude", "altitude",
                         "lat", "long", "yawrate", "speed2d"}


def is_wft_channel(name: str) -> bool:
    """True for a WFT force/moment channel (``FL_Fx`` … ``RR_Mz``)."""
    return bool(_WFT_CHANNEL_RE.match(name.strip()))


@dataclass
class PreprocessSettings:
    lower_threshold: float = 0.0        # |x| below this -> removed (aim 5)
    remove_outliers: bool = False       # aim 6
    outlier_low_pct: float = 1.0
    outlier_high_pct: float = 99.0
    moderate_spikes: bool = False
    spike_method: str = "hampel"        # "hampel" (rolling median) | "global"
    spike_window_s: float = 0.011       # rolling-median window (s)
    spike_nsigma: float = 6.0           # robust threshold (in σ)
    spike_strength: float = 1.0         # 0 = off, 1 = full de-glitch (interp)
    spike_max_frac: float = 0.02        # bail out if more than this fraction flags
    bridge_gaps: bool = False
    min_gap_s: float = 10.0
    smooth_width_s: float = 0.0         # FAMOS smo(x, width_s); 0 = off
    apply_filter: bool = False          # FAMOS FiltLP (accel only); forces use smo
    filter_cutoff: float = FAMOS_LPF_CUTOFF
    filter_order: int = FAMOS_LPF_ORDER
    resample_factor: int = 1            # decimation factor (FAMOS red(x, 10))
    resample_mode: str = "famos"        # "famos" = red() stride | "antialias"


def famos_recipe(channel: str, decimate_factor: int = 1) -> PreprocessSettings:
    """The FAMOS settings for ``channel``, straight from the imc config file.

    WFT forces/moments get ``smo(x, 0.1)``; ``Latacc`` gets ``FiltLP(4, 5)``
    followed by ``smo(x, 0.5)``; ``Longacc``/``Vehicle_Speed`` get ``smo(x, 0.5)``;
    GPS/yaw/time channels are passed through untouched, exactly as the recipe
    leaves them.
    """
    key = channel.strip().lower()
    s = PreprocessSettings(resample_factor=decimate_factor)
    if is_wft_channel(channel):
        s.smooth_width_s = FAMOS_FORCE_SMOOTH_S
    elif key in _ACCEL_LPF_CHANNELS:
        s.apply_filter = True
        s.filter_order = FAMOS_LPF_ORDER
        s.filter_cutoff = FAMOS_LPF_CUTOFF
        s.smooth_width_s = FAMOS_AUX_SMOOTH_S
    elif key in _ACCEL_SMO_CHANNELS:
        s.smooth_width_s = FAMOS_AUX_SMOOTH_S
    return s


# ------------------------------------------------------------------ FAMOS smo

def famos_smooth_window(fs: float, width_s: float) -> np.ndarray:
    """Triangular (Bartlett) kernel for one FAMOS ``smo`` pass.

    ``W = round(width_s * fs)`` samples span the window; half-width
    ``a = (W - 1) / 2``. The kernel is the exact triangle::

        h[k] = max(0, 1 - |k| / a),   integer lags k with |k| < a
        h    = h / sum(h)             (unit area)

    For ``width_s = 0.1`` s at ``fs = 1000`` Hz this is ``W = 100``,
    ``a = 49.5``, 99 taps, peak weight 0.020200 — matches the least-squares
    kernel identified from a matched raw-vs-processed FAMOS export
    (``data/Fx_raw_cut.csv``, peak 0.020201) and reproduces FAMOS ``Fx_smo``
    to 0.0056 N (the CSV export's own rounding floor).
    """
    w = int(round(width_s * fs))
    a = (w - 1) / 2.0
    if a <= 0:
        return np.array([1.0])
    kmax = int(np.ceil(a)) - 1
    k = np.arange(-kmax, kmax + 1)
    h = np.maximum(0.0, 1.0 - np.abs(k) / a)
    return h / h.sum()


def famos_smooth(x: np.ndarray, fs: float, width_s: float) -> np.ndarray:
    """FAMOS ``smo(x, width_s)`` — exact triangular weighted moving average.

    A single pass with the triangular kernel from :func:`famos_smooth_window`
    (not the previous two cascaded box-cars, which reached only 99.995 % match
    against a real FAMOS export because a boxcar-of-boxcars kernel is a coarser
    triangle than the true one). Scored against ``data/Fx_raw_cut.csv``, this
    single-kernel form matches FAMOS to max abs error 5.6e-3 N, r = 1.000000000.
    Zero-phase, so no lag is introduced.

    NaN-safe and edge-shrinking: both gaps and the true start/end of the
    array are treated as zero-weight regions and the kernel is renormalised
    by the fraction of it that overlapped real data, rather than padding
    with the edge value repeated. This keeps the ~0.25 s at each channel end
    unbiased, at a progressively shorter effective window; the interior is
    unaffected either way.
    """
    if width_s <= 0:
        return x
    arr = np.asarray(x, dtype=float)
    h = famos_smooth_window(fs, width_s)
    if h.size <= 1 or arr.size < 2:
        return arr.copy()

    mask = np.isfinite(arr)
    if not mask.any():
        return arr.copy()

    filled = np.where(mask, arr, 0.0)
    w = mask.astype(float)
    filled = convolve1d(filled, h, mode="constant", cval=0.0)
    w = convolve1d(w, h, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = filled / w
    out[~np.isfinite(out)] = np.nan
    out[~mask] = np.nan          # keep original gaps as gaps
    return out


# ------------------------------------------------------------------ FAMOS red

def famos_red(x: np.ndarray, factor: int) -> np.ndarray:
    """FAMOS ``red(x, n)`` — keep every ``n``-th sample, no extra filtering.

    Correct only once the signal is already band-limited, which is why the imc
    recipe puts ``red(cutd, 10)`` *after* all the ``smo``/``FiltLP`` steps.
    """
    if factor <= 1:
        return x
    return np.asarray(x)[::factor]


def decimate_factor(x: np.ndarray, factor: int) -> np.ndarray:
    """Naive stride decimation (kept for callers that pre-filter)."""
    return famos_red(x, factor)


def resample_antialias(x: np.ndarray, factor: int, fs: float) -> Tuple[np.ndarray, float]:
    """Anti-aliased decimation (zero-phase). Low-passes before downsampling so
    high-frequency content does not alias — unlike naive slicing. NaN-safe:
    gaps are linearly interpolated for the filter, then re-blanked.

    This is *stricter* than FAMOS ``red``; use it when the input has not been
    smoothed first.
    """
    if factor <= 1:
        return x, fs
    arr = x.astype(float).copy()
    mask = np.isfinite(arr)
    if mask.sum() < 8:
        return arr[::factor], fs / factor
    if not mask.all():
        idx = np.arange(arr.size)
        arr = np.interp(idx, idx[mask], arr[mask])
    # scipy.decimate is stable for q<=13; cascade for larger factors.
    q = factor
    y = arr
    while q > 1:
        step = min(10, q)
        # find a clean divisor to avoid a tiny final stage
        while q % step != 0 and step > 2:
            step -= 1
        if q % step != 0:
            step = q  # fall back to a single (larger) stage
        y = decimate(y, step, ftype="fir", zero_phase=True)
        q //= step
    # propagate NaN gaps to the decimated grid
    if not mask.all():
        m = mask.astype(float)
        mm = m[:len(m) - (len(m) % factor)].reshape(-1, factor).min(axis=1)
        mm = np.pad(mm, (0, max(0, y.size - mm.size)), constant_values=1)[:y.size]
        y[mm < 1] = np.nan
    return y, fs / factor


def resample(x: np.ndarray, factor: int, fs: float, mode: str = "famos"
             ) -> Tuple[np.ndarray, float]:
    """Decimate by ``factor``; ``mode='famos'`` reproduces ``red()`` exactly."""
    if factor <= 1:
        return x, fs
    if mode == "famos":
        return famos_red(x, factor), fs / factor
    return resample_antialias(x, factor, fs)


# --------------------------------------------------------------- FAMOS FiltLP

def butterworth_lpf(x: np.ndarray, cutoff: float, order: int, fs: float) -> np.ndarray:
    """FAMOS ``FiltLP(x, 0, 0, order, cutoff)`` — causal Butterworth low-pass.

    Causal, single-pass (``lfilter``) is the verified FAMOS convention: scored
    against a matched raw-vs-processed FAMOS export (``data/Fx_raw_cut.csv``),
    a single ``lfilter`` pass matches the FAMOS ``Lat_lpf`` column to 5e-6 max
    abs error with r = 1.000000000. ``filtfilt`` (zero-phase, double filtering)
    misses it by 0.66 and drops correlation to 0.93 — do not substitute it back
    in. NaN-safe.
    """
    nyq = 0.5 * fs
    if cutoff >= nyq:
        cutoff = nyq * 0.9
    arr = x.astype(float).copy()
    mask = np.isfinite(arr)
    if mask.sum() < 4 * order:
        return arr
    arr[~mask] = np.nanmean(arr[mask])
    b, a = butter(order, cutoff / nyq, btype="low")
    y = lfilter(b, a, arr)
    y[~mask] = np.nan
    return y


famos_filtlp = butterworth_lpf


# ------------------------------------------------------------- sanitisation

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


def _interpolate_over(arr: np.ndarray, spike: np.ndarray, finite: np.ndarray,
                      strength: float) -> np.ndarray:
    """Bridge each flagged run by interpolating from its clean neighbours."""
    if not spike.any():
        return arr
    # dilate flagged runs by 1 sample so the spike shoulders go too
    spike = (np.convolve(spike.astype(int), np.ones(3), mode="same") > 0) & finite
    good = finite & ~spike
    if good.sum() < 2:
        return arr
    idx = np.arange(arr.size)
    repl = np.interp(idx, idx[good], arr[good])
    k = float(np.clip(strength, 0.0, 1.0))
    out = arr.copy()
    out[spike] = arr[spike] + k * (repl[spike] - arr[spike])
    return out


_MAX_MEDIAN_WINDOW = 51        # cap: median_filter cost grows with the window


def hampel_deglitch(x: np.ndarray, fs: float, window_s: float = 0.011,
                    n_sigmas: float = 6.0, strength: float = 1.0,
                    max_frac: float = 0.02) -> np.ndarray:
    """De-glitch against a **rolling median** — the median-residual de-spiker.

    A short rolling median gives the local trend the signal *should* be
    following; ``dev = x - median`` is what the sample does on top of it. A DAQ
    artifact produces a ``dev`` far outside the family of every other sample's,
    so the threshold is ``n_sigmas`` × a **global robust σ of the residual**.

    The scale has to be global. Scoring ``dev`` against a *local* σ is
    self-defeating: a spike inflates the very window used to judge it, so the
    detector goes blind exactly where it matters, while quiet stretches get
    their own microscopic ripple flagged. A global residual scale asks the right
    question — "is this excursion unlike anything else in the recording?" — and
    is what keeps genuine road-load peaks (which the rolling median tracks, so
    their ``dev`` stays small) out of the flagged set.

    Flagged samples are bridged by interpolation from clean neighbours, never
    flattened onto the median.

    ``max_frac`` is a bail-out, not a target: if that much of the signal still
    flags even after backing the threshold off, the excursions are the signal
    rather than artifacts, and the input is returned **untouched**.
    """
    if strength <= 0:
        return x
    arr = np.asarray(x, dtype=float).copy()
    finite = np.isfinite(arr)
    if finite.sum() < 8:
        return arr

    win = max(3, min(_MAX_MEDIAN_WINDOW, int(round(window_s * fs))))
    if win % 2 == 0:
        win += 1
    # median_filter cannot see NaN, so detect on a gap-filled copy
    probe = arr
    if not finite.all():
        idx = np.arange(arr.size)
        probe = np.interp(idx, idx[finite], arr[finite])

    med = median_filter(probe, size=win, mode="nearest")
    dev = np.abs(probe - med)
    sigma = 1.4826 * np.median(dev[finite])
    if not np.isfinite(sigma) or sigma <= 0:
        return arr                      # median tracks the signal exactly

    thr = float(n_sigmas)
    spike = finite & (dev > thr * sigma)
    while spike.mean() > max_frac and thr < 200.0:
        thr *= 2.0
        spike = finite & (dev > thr * sigma)
    if spike.mean() > max_frac:
        return arr                      # nothing here is separable from signal
    return _interpolate_over(arr, spike, finite, strength)


def global_deglitch(x: np.ndarray, n_sigmas: float = 6.0,
                    strength: float = 1.0) -> np.ndarray:
    """De-glitch against a single global robust threshold.

    Cruder than :func:`hampel_deglitch` but useful when the artifact is a wide
    cluster far outside the signal's whole range (a local median rises inside
    such a cluster and the spike hides from a rolling test).
    """
    if strength <= 0:
        return x
    arr = np.asarray(x, dtype=float).copy()
    finite = np.isfinite(arr)
    if finite.sum() < 8:
        return arr
    med = np.median(arr[finite])
    sigma = 1.4826 * np.median(np.abs(arr[finite] - med))
    if sigma <= 0:
        return arr
    spike = finite & (np.abs(arr - med) > n_sigmas * sigma)
    return _interpolate_over(arr, spike, finite, strength)


def moderate_spikes(x: np.ndarray, fs: float = 1.0, window_s: float = 0.011,
                    n_sigmas: float = 6.0, strength: float = 1.0,
                    method: str = "hampel", max_frac: float = 0.02) -> np.ndarray:
    """De-glitch artifact spikes, preserving signal shape.

    ``method='hampel'`` (default) uses the rolling-median test; ``'global'``
    falls back to a single whole-signal threshold.
    """
    if method == "global":
        return global_deglitch(x, n_sigmas, strength)
    return hampel_deglitch(x, fs, window_s, n_sigmas, strength, max_frac)


def bridge_gaps(t: np.ndarray, x: np.ndarray, fs: float, min_gap_s: float = 10.0
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Remove runs of NaN or (near-)flat/zero signal lasting >= ``min_gap_s``.

    Long dropouts (sensor loss, vehicle stationary) are excised and the time
    axis re-stitched to stay continuous, so the remaining signal has no large
    empty stretches. Short gaps are left alone.
    """
    n = x.size
    if n == 0 or min_gap_s <= 0:
        return t, x
    min_len = max(1, int(round(min_gap_s * fs)))

    finite = x[np.isfinite(x)]
    if finite.size:
        scale = np.nanstd(finite)
        eps = max(1e-9, 1e-3 * scale)
    else:
        eps = 1e-9
    # A sample is "empty" if NaN, or part of a near-constant/near-zero run.
    is_nan = ~np.isfinite(x)
    d = np.abs(np.diff(x, prepend=x[:1]))
    near_flat = np.isfinite(x) & (np.abs(x) < eps) & (d < eps)
    empty = is_nan | near_flat

    # Find contiguous "empty" runs and mark those >= min_len for removal.
    remove = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if empty[i]:
            j = i
            while j < n and empty[j]:
                j += 1
            if (j - i) >= min_len:
                remove[i:j] = True
            i = j
        else:
            i += 1

    if not remove.any():
        return t, x
    keep = ~remove
    x_out = x[keep]
    # Re-stitch time: preserve sample spacing, drop the excised durations.
    t_out = np.arange(x_out.size) / fs
    if t.size:
        t_out = t_out + float(t[0])
    return t_out, x_out


# -------------------------------------------------------------------- pipeline

def apply_pipeline(t: np.ndarray, x: np.ndarray, fs: float, s: PreprocessSettings
                   ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Threshold -> outliers -> de-glitch -> gap-bridge -> FiltLP -> smo -> red.

    The FiltLP-then-smo order is the FAMOS one (``Latacc = smo(FiltLP(...), 0.5)``),
    and ``red`` comes last, after the signal is band-limited. Returns
    ``(time, signal, new_fs)``; the time axis is kept in step with the signal
    even when gap-bridging or decimation changes its length.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(x, dtype=float)
    if t.size != y.size:
        t = np.arange(y.size) / fs
    if s.lower_threshold > 0:
        y = apply_lower_threshold(y, s.lower_threshold)
    if s.remove_outliers:
        y = remove_outliers(y, s.outlier_low_pct, s.outlier_high_pct)
    if s.moderate_spikes:
        y = moderate_spikes(y, fs, s.spike_window_s, s.spike_nsigma,
                            s.spike_strength, s.spike_method, s.spike_max_frac)
    if s.bridge_gaps:
        t, y = bridge_gaps(t, y, fs, s.min_gap_s)
    if s.apply_filter:
        y = butterworth_lpf(y, s.filter_cutoff, s.filter_order, fs)   # FiltLP
    if s.smooth_width_s > 0:
        y = famos_smooth(y, fs, s.smooth_width_s)                     # smo
    new_fs = fs
    if s.resample_factor > 1:
        y, new_fs = resample(y, s.resample_factor, fs, s.resample_mode)
        t = (np.arange(y.size) / new_fs) + (float(t[0]) if t.size else 0.0)
    return t, y, new_fs


def apply_famos_recipe(df: pd.DataFrame, fs: float,
                       decimate_factor: int = 1,
                       deglitch: bool = False,
                       deglitch_nsigma: float = 6.0,
                       time_column: str = "Time",
                       emit_lpf_columns: bool = True,
                       ) -> Tuple[pd.DataFrame, float, Dict[str, str]]:
    """Apply the full imc/FAMOS recipe to every column of ``df``.

    Each channel gets the treatment the imc config file specifies for it —
    ``smo(0.1)`` for WFT forces/moments, ``FiltLP(4,5)`` + ``smo(0.5)`` for
    ``Latacc``, ``smo(0.5)`` for ``Longacc``/``Vehicle_Speed``, passthrough for
    GPS/yaw — then ``red(n)`` across the board so all channels stay aligned.

    ``deglitch`` inserts a rolling-median de-glitch ahead of the FAMOS steps.
    It is off by default because a correctly-read imc file is already clean;
    turn it on for recordings with DAQ artifact spikes.

    With ``emit_lpf_columns``, the pre-smoothing ``Latacc_LPF`` intermediate is
    exported too, matching the column set of a FAMOS CSV.

    Returns ``(processed_df, new_fs, applied)`` where ``applied`` maps each
    column to the operation performed, for logging and reports.
    """
    out: Dict[str, np.ndarray] = {}
    applied: Dict[str, str] = {}
    new_fs = fs / decimate_factor if decimate_factor > 1 else fs

    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if col == time_column:
            n = int(np.ceil(series.size / max(1, decimate_factor)))
            start = float(series[0]) if series.size and np.isfinite(series[0]) else 0.0
            out[col] = np.arange(n) / new_fs + start
            applied[col] = "rebuilt time base"
            continue

        s = famos_recipe(col, decimate_factor=1)      # decimate once, below
        y = series
        steps: List[str] = []
        if deglitch:
            y = hampel_deglitch(y, fs, n_sigmas=deglitch_nsigma)
            steps.append("de-glitch")
        if s.apply_filter:
            y = butterworth_lpf(y, s.filter_cutoff, s.filter_order, fs)
            steps.append(f"FiltLP({s.filter_order},{s.filter_cutoff:g}Hz)")
            if emit_lpf_columns:
                out[f"{col}_LPF"] = famos_red(y, decimate_factor)
                applied[f"{col}_LPF"] = "FAMOS FiltLP intermediate"
        if s.smooth_width_s > 0:
            y = famos_smooth(y, fs, s.smooth_width_s)
            steps.append(f"smo({s.smooth_width_s:g}s)")
        if decimate_factor > 1:
            y = famos_red(y, decimate_factor)
            steps.append(f"red({decimate_factor})")
        out[col] = y
        applied[col] = " -> ".join(steps) if steps else "passthrough"

    # red() can leave a one-sample length difference between channels
    n = min(len(v) for v in out.values()) if out else 0
    result = pd.DataFrame({k: v[:n] for k, v in out.items()})
    # keep the original column order, with each *_LPF right after its source
    ordered: List[str] = []
    for col in df.columns:
        ordered.append(col)
        if f"{col}_LPF" in result.columns:
            ordered.append(f"{col}_LPF")
    result = result[[c for c in ordered if c in result.columns]]
    return result, new_fs, applied


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
