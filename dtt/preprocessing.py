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
from scipy.ndimage import median_filter, convolve1d, uniform_filter1d
from scipy.signal import butter, filtfilt, lfilter, decimate, sosfilt, sosfilt_zi

from dtt.channels import COMPONENTS, parse_channel

# ---------------------------------------------------------------- FAMOS recipe

FILTER_INIT_FAMOS = "famos"                # step-response: zi = sosfilt_zi * x[0]
FILTER_INIT_LEGACY = "legacy_zero_state"   # pre-2026-08 behaviour, reproduction only
FILTER_INIT_DEFAULT = FILTER_INIT_FAMOS

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
    """True for a WFT force/moment channel, in **any** of the naming schemes.

    The imc config file names its channels ``FL_Fx`` … ``RR_Mz``, but a real
    export carries the axle-dynamic form the recording used — ``FR_Fx_2``,
    ``RR_Mz_1``, ``A3RO_My``, ``WFT_Fx_fl``. Matching only the two-axle literal
    silently leaves every one of those unconditioned: ``smo`` never runs, and
    ``red(10)`` then folds the whole 50–500 Hz band back over the signal as
    phantom spikes. :func:`dtt.channels.parse_channel` is the project's single
    authority on which names are wheel force/moment channels, so defer to it and
    keep the literal only as a fast path.
    """
    n = name.strip()
    if _WFT_CHANNEL_RE.match(n):
        return True
    parsed = parse_channel(n)
    return parsed is not None and parsed[1] in COMPONENTS


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
    # FAMOS FiltLP initial conditions. "famos" = step-response (correct);
    # "legacy_zero_state" reproduces pre-fix outputs and nothing else.
    filter_init: str = FILTER_INIT_DEFAULT


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
    """FAMOS ``smo(x, width_s)`` -- delegates to :func:`famos.ops.smo`.

    The maths lives in one place only. This module used to carry its own copy,
    and a third lived in ``famos_repro.py``; they drifted apart on the edge
    convention, which is exactly how a signal-processing codebase corrupts data
    without anyone noticing -- two callers, two answers, both plausible.

    This wrapper keeps the lenient boundary behaviour existing callers (and the
    GUI) rely on: a width below one sample, or a window longer than the record,
    returns the input untouched rather than raising, because the GUI previews
    arbitrary channels at arbitrary zoom and must not throw on a short one.
    """
    from famos import ops as _ops

    arr = np.asarray(x, dtype=float)
    if width_s <= 0:
        return arr
    h = _ops.smo_kernel(width_s, fs)
    if h.size <= 1 or h.size > arr.size or arr.size < 2:
        return arr
    return _ops.smo(arr, width_s, fs)


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

def butterworth_lpf(x: np.ndarray, cutoff: float, order: int, fs: float,
                    init: str = FILTER_INIT_DEFAULT) -> np.ndarray:
    """FAMOS ``FiltLP(x, 0, 0, order, cutoff)`` -- delegates to
    :func:`famos.ops.filtlp`.

    Confirmed against the imc FAMOS Function Reference: SvCharacter 0 is
    Butterworth, coefficients come from a bilinear transformation, and
    ``FiltLpZ`` is a *separate* "without phase shift" function -- so plain
    ``FiltLP`` is single-pass causal by design, not by our choice.

    ``init="famos"`` (default) is step-response initialisation, ``y[0] == x[0]``.
    ``init="legacy_zero_state"`` exists solely to reproduce study outputs
    published before that was fixed, and must never become the default.

    As with :func:`famos_smooth`, the lenient boundary behaviour is kept for the
    GUI's sake: a cutoff at or above Nyquist is clamped and a too-short record
    is returned untouched, where the library core would raise.
    """
    from famos import ops as _ops

    if init not in (FILTER_INIT_FAMOS, FILTER_INIT_LEGACY):
        raise ValueError(f"unknown filter init {init!r}; expected "
                         f"{FILTER_INIT_FAMOS!r} or {FILTER_INIT_LEGACY!r}")
    arr = np.asarray(x, dtype=float)
    nyq = 0.5 * fs
    if cutoff >= nyq:
        cutoff = nyq * 0.9
    if np.isfinite(arr).sum() < 4 * order:
        return arr
    return _ops.filtlp(arr, cutoff, order, fs, init=init)


famos_filtlp = butterworth_lpf


# ------------------------------------------------------------- sanitisation

def apply_lower_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """Blank samples whose magnitude is below ``threshold`` (noise floor)."""
    if threshold <= 0:
        return x
    out = x.astype(float).copy()
    out[np.abs(out) < threshold] = np.nan
    return out


def blank_dead_runs(x: np.ndarray, fs: float, min_s: float = 1.0) -> np.ndarray:
    """Blank stretches where the signal is *exactly* constant for ``min_s`` or more.

    When a WFT stops streaming mid-recording the DAQ keeps writing samples at a
    frozen value — almost always a hard 0. Those samples are not a measurement of
    zero load, they are the absence of a measurement, and carrying them as data
    is what puts a phantom step at the end of a trace and drags the channel's
    mean and σ with it (in the reference recording ``RR_Fz_1`` dies 174 s early:
    mean 7104 vs median 7336, σ 1526 against 803 for the live front channel).

    The test is *bit-exact* constancy, which is the one thing a live sensor never
    does — even a stationary vehicle dithers by an ADC count — so a genuine
    quiet stretch is never mistaken for a dropout. Blanked samples become NaN,
    the gap marker every operator downstream already honours.
    """
    arr = np.asarray(x, dtype=float).copy()
    if arr.size < 2 or min_s <= 0 or not np.isfinite(fs) or fs <= 0:
        return arr
    min_len = max(2, int(round(min_s * fs)))
    if arr.size < min_len:
        return arr

    # Runs of bit-exact repeats: same[i] means sample i+1 equals sample i.
    same = np.zeros(arr.size, dtype=bool)
    np.equal(arr[1:], arr[:-1], out=same[1:])
    same &= np.isfinite(arr)
    if not same.any():
        return arr

    edges = np.flatnonzero(np.diff(np.r_[False, same, False]))
    for start, stop in zip(edges[0::2], edges[1::2]):
        if (stop - start) + 1 >= min_len:       # +1: the run's first sample
            arr[start - 1:stop] = np.nan
    return arr


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


def _flag_value_runs(arr: np.ndarray, min_len: int) -> np.ndarray:
    """Flag every sample in a run of ``>= min_len`` consecutive identical
    finite values. A run never crosses a NaN (each NaN breaks it)."""
    n = arr.size
    flag = np.zeros(n, dtype=bool)
    if n == 0 or min_len <= 1:
        return flag
    finite = np.isfinite(arr)
    same_as_prev = np.zeros(n, dtype=bool)
    same_as_prev[1:] = finite[1:] & finite[:-1] & (arr[1:] == arr[:-1])
    starts = np.where(~same_as_prev)[0]
    ends = np.append(starts[1:], n)
    for s, e in zip(starts, ends):
        if finite[s] and (e - s) >= min_len:
            flag[s:e] = True
    return flag


def detect_dropout(x: np.ndarray, max_run: int = 5) -> np.ndarray:
    """Despike rule 2 — dropout: exact zeros or nulls, and a run of any
    identical value repeated ``>= max_run`` samples (a frozen sensor),
    whatever that value is. Nearly the same test as :func:`detect_rail`,
    just without the "at the channel's own extreme" restriction.
    """
    arr = np.asarray(x, dtype=float)
    finite = np.isfinite(arr)
    flag = ~finite | (finite & (arr == 0.0))
    flag |= _flag_value_runs(arr, max_run)
    return flag


def detect_rail(x: np.ndarray, min_run: int = 3) -> np.ndarray:
    """Despike rule 1 — saturation/rail: a run of ``>= min_run`` samples
    pinned at the channel's own min or max value. Detected by value, not raw
    ADC counts, so it is scale-independent and needs no factor plumbing for
    either the raw or the CSV source.
    """
    arr = np.asarray(x, dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() < min_run:
        return np.zeros(arr.size, dtype=bool)
    lo, hi = np.min(arr[finite]), np.max(arr[finite])
    if lo == hi:
        return np.zeros(arr.size, dtype=bool)     # constant channel, no rail to hit
    at_rail = finite & ((arr == lo) | (arr == hi))
    masked = np.where(at_rail, arr, np.nan)
    return _flag_value_runs(masked, min_run) & at_rail


def detect_narrow_spikes(x: np.ndarray, fs: float, hw_cutoff_hz: float = 200.0
                        ) -> np.ndarray:
    """Despike rule 3 — sub-hardware-width spike: the raw channel was
    hardware low-pass filtered at ``hw_cutoff_hz`` before being digitised at
    ``fs``, so no genuine feature can be narrower than roughly half a cycle
    at that cutoff (``fs / (2 * hw_cutoff_hz)`` samples) — a real peak is
    always several samples wide.

    The primary criterion is **width**: a same-signed excursion off the
    local trend that spans fewer than that physical minimum of samples is
    non-physical regardless of its size, which is what lets a genuine
    (wider) road-load peak of any size survive untouched — a real peak is
    never removed for being big, only a real *glitch* is removed for being
    too narrow to exist.

    A structural width test alone cannot tell a genuine single-sample
    artifact from the ordinary sample-to-sample chatter every noisy signal
    has relative to its own local median — that chatter is *always* exactly
    1 sample wide by construction, so a bare width test flags a few percent
    of any channel. A light noise floor (``2x`` the local high-frequency
    residual's robust scale) excludes that chatter; it is far below any real
    spike or peak amplitude, so it still does not discriminate on size.
    """
    arr = np.asarray(x, dtype=float)
    n = arr.size
    finite = np.isfinite(arr)
    min_width = max(1, int(round(fs / (2.0 * hw_cutoff_hz))))
    if n < 5 or finite.sum() < 5:
        return np.zeros(n, dtype=bool)

    # The trend window has to be well wider than min_width, or the "trend"
    # just tracks the raw signal and every ordinary sample-to-sample wiggle
    # reads as a narrow excursion. 8x gives the median room to represent the
    # slow-moving signal a hw_cutoff_hz-limited channel should show locally,
    # while a real road-load peak (many min_width's wide) still stands clear
    # of it for long enough not to trip the width test below.
    win = max(3, 8 * min_width + 1)
    if win % 2 == 0:
        win += 1
    probe = arr
    if not finite.all():
        idx = np.arange(n)
        probe = np.interp(idx, idx[finite], arr[finite])
    trend = median_filter(probe, size=win, mode="nearest")   # what a hw_cutoff_hz-limited signal should look like locally
    resid = probe - trend
    noise_floor = 2.0 * 1.4826 * np.median(np.abs(resid[finite]))
    sign = np.sign(resid)

    change = np.ones(n, dtype=bool)
    change[1:] = sign[1:] != sign[:-1]
    starts = np.where(change)[0]
    ends = np.append(starts[1:], n)
    flag = np.zeros(n, dtype=bool)
    for s, e in zip(starts, ends):
        if sign[s] != 0 and (e - s) < min_width and np.max(np.abs(resid[s:e])) >= noise_floor:
            flag[s:e] = True
    return flag & finite


def detect_transient_spikes(x: np.ndarray, fs: float, window_s: float = 1.0,
                            pct_threshold: float = 20.0,
                            noise_floor_mult: float = 10.0,
                            max_spike_frac: float = 0.4) -> np.ndarray:
    """Despike rule 4 — sudden transient: encodes the manual step "a sudden
    spike of more than 20% within a 1 second time frame should be adjusted
    or eliminated."

    Runs on the **already-conditioned** (smo/FiltLP'd, typically decimated)
    signal, not the raw one — see :func:`despike` and the pipeline notes for
    why. The manual's "20% within 1 second" describes what an operator saw
    on a conditioned FAMOS trace; measured against genuinely raw 1kHz
    samples the same test flags 25-90% of the channel, because raw
    sample-to-sample jitter routinely exceeds 20% of a local median on its
    own — that is ordinary raw noise, not a transient, and running this rule
    post-smooth is what keeps it a transient detector instead of a second,
    unwanted smoothing pass.

    A plain "changed more than 20%" amplitude rule is still dangerous even on
    conditioned data: real WFT road load genuinely swings far more than 20%
    in a second when the tyre hits an event, and that swing is the real
    signal the fatigue analysis exists to see. And "20% of what" is
    undefined for a force channel that crosses zero. Both are handled the
    way rule 3 handles its own version of this: measure deviation against
    the **local trend** (a `window_s`-wide rolling median, robust to the
    spike itself), require the excursion to be **narrow**, and gate on the
    *larger* of 20% of the local level and `noise_floor_mult` times the
    channel's own robust local noise scale, so the effective threshold can
    never sit below the channel's own noise (the same fix that kept rule 3
    from over-flagging ordinary chatter) while still reading as "20%"
    everywhere the local level is not itself close to the noise floor.

    "Narrow" is *not* "under `window_s`" — validated on a real recording, a
    genuine ~0.7 s braking/cornering swing (smooth, monotonic, real load) is
    still well under a 1 s cap and would pass a bare "< window_s" test. A
    real spike-and-return is a small fraction of the window it's measured
    against; `max_spike_frac` caps the flagged run at that fraction of
    `window_s` (0.3 s of a 1 s window by default), which is what actually
    tells a sudden manual-spec transient apart from an ordinary, wider real
    load excursion that merely happens to resolve inside a second too.

    Complements :func:`detect_narrow_spikes` rather than replacing it: that
    rule catches sub-2-sample glitches of any size; this one catches wider
    (but still short) transients that also clear the threshold.
    """
    arr = np.asarray(x, dtype=float)
    n = arr.size
    finite = np.isfinite(arr)
    win = max(3, int(round(window_s * fs)))
    if win % 2 == 0:
        win += 1
    max_run = max(1, int(round(max_spike_frac * window_s * fs)))
    if n < win or finite.sum() < 8:
        return np.zeros(n, dtype=bool)

    probe = arr
    if not finite.all():
        idx = np.arange(n)
        probe = np.interp(idx, idx[finite], arr[finite])
    trend = median_filter(probe, size=win, mode="nearest")
    resid = probe - trend
    sign = np.sign(resid)

    pct_level = (pct_threshold / 100.0) * np.abs(trend)     # "20% of local level", absolute units
    noise_scale = 1.4826 * np.median(np.abs(resid[finite])) if finite.any() else 0.0
    eff_threshold = np.maximum(pct_level, noise_floor_mult * noise_scale)

    change = np.ones(n, dtype=bool)
    change[1:] = sign[1:] != sign[:-1]
    starts = np.where(change)[0]
    ends = np.append(starts[1:], n)
    flag = np.zeros(n, dtype=bool)
    for s, e in zip(starts, ends):
        if sign[s] != 0 and (e - s) < max_run and \
           np.max(np.abs(resid[s:e])) >= np.max(eff_threshold[s:e]):
            flag[s:e] = True
    return flag & finite


def despike(x: np.ndarray, fs: float,
           rail_min_run: int = 3, dropout_max_run: int = 5,
           hw_cutoff_hz: float = 200.0,
           net: bool = False, net_nsigma: float = 6.0, net_window_s: float = 0.011,
           strength: float = 1.0) -> Tuple[np.ndarray, float]:
    """Physical-rule despike ahead of FiltLP/smo: rail, dropout, and
    sub-hardware-width spikes, replaced by interpolation from good
    neighbours — never dropped.

    Deliberately not an amplitude threshold. This is spiky WFT road load
    where a genuine peak can be as large as any artifact, so every primary
    rule is structural (pinned at the rail, a frozen run, or narrower than
    the hardware could produce) rather than "how big is it" — which is what
    keeps real peaks intact for rainflow counting.

    ``net``, off by default, optionally runs a very loose adaptive
    :func:`hampel_deglitch` pass afterwards for gross leftovers the physical
    rules miss.

    The manual's "20% within 1 second" transient rule (:func:`detect_transient_spikes`)
    is deliberately **not** included here — it runs post-smooth, on the
    already-conditioned signal, wired separately into
    :func:`apply_famos_recipe`. See that function's docstring for why.

    Returns ``(despiked, pct_flagged)``.
    """
    arr = np.asarray(x, dtype=float)
    finite = np.isfinite(arr)
    flag = detect_dropout(arr, dropout_max_run)
    flag |= detect_rail(arr, rail_min_run)
    flag |= detect_narrow_spikes(arr, fs, hw_cutoff_hz)
    out = _interpolate_over(arr, flag, finite, strength)
    if net:
        out = hampel_deglitch(out, fs, window_s=net_window_s,
                              n_sigmas=net_nsigma, strength=strength)
    pct_flagged = 100.0 * flag.sum() / max(1, arr.size)
    return out, pct_flagged


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
        y = butterworth_lpf(y, s.filter_cutoff, s.filter_order, fs,
                            init=s.filter_init)                # FiltLP
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
                       despike_enabled: bool = False,
                       despike_rail_min_run: int = 3,
                       despike_dropout_max_run: int = 5,
                       despike_hw_cutoff_hz: float = 200.0,
                       despike_net: bool = False,
                       despike_net_nsigma: float = 6.0,
                       despike_net_window_s: float = 0.011,
                       time_column: str = "Time",
                       emit_lpf_columns: bool = True,
                       blank_dead_s: float = 1.0,
                       transient_enabled: bool = False,
                       transient_pct: float = 20.0,
                       transient_window_s: float = 1.0,
                       transient_noise_floor_mult: float = 10.0,
                       transient_max_spike_frac: float = 0.4,
                       ) -> Tuple[pd.DataFrame, float, Dict[str, str]]:
    """Apply the full imc/FAMOS recipe to every column of ``df``.

    Each channel gets the treatment the imc config file specifies for it —
    ``smo(0.1)`` for WFT forces/moments, ``FiltLP(4,5)`` + ``smo(0.5)`` for
    ``Latacc``, ``smo(0.5)`` for ``Longacc``/``Vehicle_Speed``, passthrough for
    GPS/yaw — then ``red(n)`` across the board so all channels stay aligned.

    ``blank_dead_s`` first marks frozen-value sensor dropouts on the force and
    moment channels as gaps (see :func:`blank_dead_runs`) so they are neither
    smoothed into the live signal nor counted as measurements downstream.

    ``deglitch`` inserts a rolling-median de-glitch ahead of the FAMOS steps.
    It is off by default because a correctly-read imc file is already clean;
    turn it on for recordings with DAQ artifact spikes.

    ``despike_enabled`` runs :func:`despike` (rail/dropout/sub-hardware-width
    rules, see its docstring) ahead of ``deglitch`` and the FAMOS steps. Off
    by default so existing callers are unaffected; the ``despike_*`` kwargs
    tune its thresholds and its optional loose adaptive net.

    ``transient_enabled`` runs :func:`detect_transient_spikes` (the manual's
    "20% within 1 second" rule) **after** FiltLP/smo/red, on the
    already-conditioned, decimated signal — not with the other despike rules
    on the raw one. Measured against literal raw 1kHz samples, the same 20%
    test flags 25-90% of the channel (ordinary raw jitter routinely exceeds
    20% of a very local median); the manual's language describes what an
    operator saw on a conditioned trace, so this rule runs where that
    reading is actually true. Off by default; only channels the recipe
    actually smoothed/filtered are eligible (a passthrough channel is
    untouched by this too).

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
        if blank_dead_s > 0 and is_wft_channel(col):
            # Ahead of everything else, despike included. A long frozen run is
            # the absence of a measurement, and `despike` would interpolate
            # straight across it — fabricating the 174 s a dead WFT leaves in
            # the reference recording. Marking it NaN first is what stops that:
            # `_interpolate_over` masks its flags with `& finite`, so the gap
            # survives the despike pass instead of being filled in.
            blanked = blank_dead_runs(y, fs, blank_dead_s)
            n_dead = int(np.isfinite(y).sum() - np.isfinite(blanked).sum())
            if n_dead and np.isfinite(blanked).any():
                y = blanked
                steps.append(f"blank dropout({n_dead / fs:.1f}s)")
        if despike_enabled:
            y, pct_flagged = despike(
                y, fs,
                rail_min_run=despike_rail_min_run,
                dropout_max_run=despike_dropout_max_run,
                hw_cutoff_hz=despike_hw_cutoff_hz,
                net=despike_net, net_nsigma=despike_net_nsigma,
                net_window_s=despike_net_window_s)
            steps.append(f"despike({pct_flagged:.3g}%)")
        if deglitch:
            y = hampel_deglitch(y, fs, n_sigmas=deglitch_nsigma)
            steps.append("de-glitch")
        if s.apply_filter:
            y = butterworth_lpf(y, s.filter_cutoff, s.filter_order, fs,
                                init=s.filter_init)
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
        if transient_enabled and (s.apply_filter or s.smooth_width_s > 0):
            # Post-conditioning, on whatever this channel ended up at (the
            # decimated rate, once red() has run) -- see the docstring for
            # why this rule specifically does not run on the raw signal.
            t_flag = detect_transient_spikes(
                y, new_fs, window_s=transient_window_s,
                pct_threshold=transient_pct,
                noise_floor_mult=transient_noise_floor_mult,
                max_spike_frac=transient_max_spike_frac)
            if t_flag.any():
                y = _interpolate_over(y, t_flag, np.isfinite(y), 1.0)
                steps.append(f"transient({100.0 * t_flag.sum() / max(1, t_flag.size):.3g}%)")
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


def count_conditioned(applied: Dict[str, str]) -> int:
    """How many channels in an :func:`apply_famos_recipe` report were *filtered*.

    ``red(n)`` is applied to every column so the channels stay aligned, so its
    presence says nothing about whether a channel was recognised — counting any
    non-passthrough entry reports "37/38 conditioned" for a run in which only 15
    channels actually matched the recipe, which is exactly the kind of reassuring
    number that hides an unmatched-channel bug. Only ``smo`` and ``FiltLP`` count.
    """
    return sum(1 for v in applied.values() if "smo(" in v or "FiltLP(" in v)


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


# ------------------------------------------------------- stationary / pause removal

STOP_MIN_S = 5.0            # shorter than this is traffic, not a stop worth cutting
STOP_WINDOW_S = 1.0         # rolling window the stationarity test looks through
STOP_QUIET_FRAC = 0.10      # "quiet" = this fraction of the channel's own robust sigma
STOP_MAX_FRAC = 0.60        # bail out rather than gut the recording


def _rolling_std(a: np.ndarray, win: int) -> np.ndarray:
    """Rolling standard deviation, O(n) via the mean of squares."""
    arr = np.asarray(a, dtype=float)
    fill = float(np.nanmedian(arr)) if np.isfinite(arr).any() else 0.0
    arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
    m = uniform_filter1d(arr, win, mode="nearest")
    m2 = uniform_filter1d(arr * arr, win, mode="nearest")
    # m2 - m^2 is a difference of close numbers and can land just below zero
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def _runs_at_least(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Keep only the runs of True in ``mask`` lasting ``min_len`` samples or more."""
    out = np.zeros(mask.size, dtype=bool)
    if not mask.any():
        return out
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    for start, stop in zip(edges[0::2], edges[1::2]):
        if (stop - start) >= min_len:
            out[start:stop] = True
    return out


def detect_stops(x: np.ndarray, fs: float, min_stop_s: float = STOP_MIN_S,
                 window_s: float = STOP_WINDOW_S,
                 quiet_frac: float = STOP_QUIET_FRAC) -> np.ndarray:
    """Boolean mask of samples where the channel carries no road input.

    A parked vehicle still reads its static corner weight, so the test cannot be
    on the *level* -- :func:`bridge_gaps` looks for a near-zero run and therefore
    never sees a stop at all. What actually vanishes when the wheels stop turning
    is the road input, so the measure is the **rolling standard deviation**: on
    the reference recording it runs at 57 daN while driving and under 7 daN at a
    standstill, an eight-fold separation that makes the threshold choice almost
    free -- 0.05 sigma and 0.20 sigma pick out the same three stops.

    Scoring against the channel's own robust sigma keeps it scale-free: it works
    on newtons or decanewtons, a hatchback or a loaded truck, with no tuning.

    Samples already blanked to NaN count as stopped. A dead sensor is not road
    input either, and leaving them out would split one stop into two.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0 or min_stop_s <= 0 or fs <= 0:
        return np.zeros(x.size, dtype=bool)

    finite = np.isfinite(x)
    if finite.sum() < 8:
        return np.zeros(x.size, dtype=bool)
    med = np.median(x[finite])
    sigma = 1.4826 * np.median(np.abs(x[finite] - med))
    if not np.isfinite(sigma) or sigma <= 0:
        return np.zeros(x.size, dtype=bool)      # a constant channel says nothing

    win = max(3, int(round(window_s * fs)))
    quiet = (_rolling_std(x, win) < quiet_frac * sigma) | ~finite
    return _runs_at_least(quiet, max(1, int(round(min_stop_s * fs))))


def detect_stops_from_speed(speed, fs: float, min_stop_s: float = STOP_MIN_S,
                            moving_kph: float = 1.0):
    """Stops straight from a speed channel, or ``None`` if it is unusable.

    Preferred whenever the recording actually has speed, because it states
    directly what the force channels only imply. Returns ``None`` for an absent
    or all-zero channel -- the reference recording's ``Vehicle_Speed`` is exactly
    that, which is why the force-dynamics route is not merely a fallback.
    """
    if speed is None:
        return None
    v = np.asarray(speed, dtype=float)
    finite = np.isfinite(v)
    if finite.sum() < 8 or float(np.max(np.abs(v[finite]))) <= moving_kph:
        return None                          # never moves -> carries no information
    stopped = (np.abs(np.nan_to_num(v)) < moving_kph) | ~finite
    return _runs_at_least(stopped, max(1, int(round(min_stop_s * fs))))


def remove_stops(t: np.ndarray, x: np.ndarray, fs: float,
                 min_stop_s: float = STOP_MIN_S,
                 window_s: float = STOP_WINDOW_S,
                 quiet_frac: float = STOP_QUIET_FRAC,
                 mask=None) -> Tuple[np.ndarray, np.ndarray, float]:
    """Excise stationary stretches and re-stitch time. Returns ``(t, x, seconds)``.

    Pass ``mask`` to excise a decision made elsewhere -- that is how every channel
    in a frame gets cut identically. Cutting each channel on its own evidence
    would remove different samples from each and slide them out of sync, which is
    considerably worse than leaving the stops in.
    """
    x = np.asarray(x, dtype=float)
    if mask is None:
        mask = detect_stops(x, fs, min_stop_s, window_s, quiet_frac)
    if mask is None or not mask.any():
        return t, x, 0.0
    if mask.mean() > STOP_MAX_FRAC:
        # Most of the recording reads as stationary: either the vehicle never
        # really moved or the channel is flat. Cutting is not the right response
        # to either, so hand back what came in.
        return t, x, 0.0

    keep = ~mask
    removed_s = float(mask.sum()) / fs
    x_out = x[keep]
    start = float(t[0]) if t is not None and len(t) else 0.0
    t_out = np.arange(x_out.size) / fs + start
    return t_out, x_out, removed_s


def remove_stops_frame(df: pd.DataFrame, fs: float,
                       speed_column=None, time_column: str = "Time",
                       min_stop_s: float = STOP_MIN_S,
                       window_s: float = STOP_WINDOW_S,
                       quiet_frac: float = STOP_QUIET_FRAC,
                       ) -> Tuple[pd.DataFrame, dict]:
    """Drop stationary stretches from every channel at once.

    One mask decides for the whole frame -- taken from the speed channel when it
    says anything, otherwise from a per-force-channel majority vote, so a single
    noisy wheel cannot cut the recording on its own. Time is rebuilt at the
    original spacing, leaving a continuous record with the stops closed up.
    """
    if df.empty or fs <= 0:
        return df, {}

    mask = None
    basis = ""
    if speed_column and speed_column in df.columns:
        mask = detect_stops_from_speed(
            pd.to_numeric(df[speed_column], errors="coerce").to_numpy(dtype=float),
            fs, min_stop_s)
        if mask is not None:
            basis = "speed (%s)" % speed_column

    if mask is None:
        votes = [detect_stops(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float),
                              fs, min_stop_s, window_s, quiet_frac)
                 for c in df.columns if c != time_column and is_wft_channel(c)]
        if not votes:
            return df, {}
        # >= 0.5, so with an even channel count a half-and-half split still
        # cuts. That is the wanted behaviour: when three of six wheels read as
        # stationary it is usually because their sensor died, and half a frame
        # of dead channel is not data worth keeping either.
        mask = np.median(np.vstack(votes).astype(float), axis=0) >= 0.5
        # the vote is per-sample; re-impose the duration rule on the consensus
        mask = _runs_at_least(mask, max(1, int(round(min_stop_s * fs))))
        basis = "force dynamics (%d channels)" % len(votes)

    if not mask.any() or mask.mean() > STOP_MAX_FRAC:
        return df, {"stops_removed": 0, "stop_seconds": 0.0, "stop_basis": basis}

    n_stops = int(np.count_nonzero(np.diff(np.r_[False, mask].astype(np.int8)) == 1))
    out = df.loc[~mask].reset_index(drop=True)
    if time_column in out.columns:
        start = float(df[time_column].iloc[0]) if len(df) else 0.0
        out[time_column] = np.arange(len(out)) / fs + start
    return out, {"stops_removed": n_stops,
                 "stop_seconds": round(float(mask.sum()) / fs, 2),
                 "stop_basis": basis}
