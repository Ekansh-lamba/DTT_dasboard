"""Canonical imc FAMOS operators. One implementation, no second copy anywhere.

Every function here is pure: same input, same output, no globals, no I/O, no
hidden state. Each docstring names the FAMOS function it replaces and the
evidence that fixes its semantics.

Provenance of the semantics
---------------------------
imc FAMOS Function Reference (public, 1300 pp), plus measurements against a real
FAMOS export (`csv/RLDA WFT PV data sample.csv`, which ships `Latacc_LPF` beside
`Latacc = smo(Latacc_LPF, 0.5)` -- a matched operator input/output pair).

Confirmed from the manual:
  FiltLP(Data, SvCharacter, SvParameter, SvOrder, SvCutOffFreq)
      SvCharacter 0=Butterworth 1=Bessel 2=Chebychev 3=Critical damping
      "coefficients are calculated ... using bilinear transformation"
      FiltLpZ is a SEPARATE function ("without phase shift"), so plain FiltLP
      is phase-shifting, i.e. single-pass causal.
  Smo(Data, SvWidth)
      triangular weighting; "can only be an odd number of points";
      "non-causal filter with a phase of zero"; "preserves the mean value";
      "For filtering near the edges, it is assumed that the data sets are
       extended with the same values as the edges."

Measured, not documented:
  FiltLP initial conditions are step-response (`y[0] == x[0]`). The export's
  Latacc_LPF begins at 0.180000, exactly the input level; zero state would
  begin at b0*x[0] = 7.5e-05. Still to be confirmed against a golden corpus.

Open (needs the FAMOS golden-corpus run):
  whether SvCutOffFreq is the -3 dB point, and red()'s decimation phase.

Numeric policy
--------------
float64 everywhere. Every entry point coerces via `as_f64`, which REJECTS
float32 rather than silently upcasting -- a float32 input means precision was
already lost upstream, and quietly widening it hides that.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

__all__ = [
    "as_f64", "smo_kernel", "smo", "filtlp_sos", "filtlp", "red",
    "INIT_FAMOS", "INIT_LEGACY_ZERO_STATE",
    "CHARACTER_BUTTERWORTH", "CHARACTER_BESSEL", "CHARACTER_CHEBYCHEV",
    "CHARACTER_CRITICAL_DAMPING",
]

# FiltLP SvCharacter codes, from the Function Reference.
CHARACTER_BUTTERWORTH = 0
CHARACTER_BESSEL = 1
CHARACTER_CHEBYCHEV = 2
CHARACTER_CRITICAL_DAMPING = 3

INIT_FAMOS = "famos"                       # step response: zi = sosfilt_zi * x[0]
INIT_LEGACY_ZERO_STATE = "legacy_zero_state"   # reproduction of pre-fix output only


def as_f64(x, name: str = "signal") -> np.ndarray:
    """Coerce to a contiguous float64 array, refusing lossy inputs.

    float32 raises rather than upcasting. Silently widening it would produce an
    array that *looks* like float64 while carrying only 24 bits of mantissa, and
    every downstream tolerance would then be meeting a precision the data never
    had. Integers are exact under widening, so they are accepted.
    """
    arr = np.asarray(x)
    if arr.dtype == np.float32 or arr.dtype == np.float16:
        raise TypeError(
            f"{name}: got {arr.dtype}, which has already lost precision. "
            "Read the source as float64; this library will not upcast silently.")
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{name}: expected a numeric array, got {arr.dtype}")
    return np.ascontiguousarray(arr, dtype=np.float64)


# --------------------------------------------------------------- FAMOS Smo

def _pad_edges(a: np.ndarray, hw: int) -> np.ndarray:
    """Extend by ``hw`` samples each side, replicating the edge value.

    The imc FAMOS Function Reference on Smo: "For filtering near the edges, it
    is assumed that the data sets are extended with the same values as the
    edges."
    """
    if hw <= 0:
        return a
    return np.concatenate([np.full(hw, a[0]), a, np.full(hw, a[-1])])


def smo_kernel(width_s: float, fs_hz: float) -> np.ndarray:
    """Kernel for FAMOS ``Smo(x, width_s)`` -- triangular, odd, unit area.

    Manual: "The weighting function is triangular if the interval width is
    greater than five points"; "can only be an odd number of points, otherwise
    the specified width will be rounded to the next possible value"; "preserves
    the mean value".

    W = round(width_s * fs) samples; half-width a = (W-1)/2; taps at integer
    lags |k| < a. That yields an odd count, symmetric about a sample, hence the
    zero phase the manual specifies. A width sweep against the FAMOS export
    confirms the seconds->samples convention: RMS error minimises sharply at
    exactly the specified 0.5 s (0.45 s and 0.55 s are 6.6x and 4.4x worse).
    """
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError(f"smo: sample rate must be finite and positive, got {fs_hz}")
    if width_s < 0:
        raise ValueError(f"smo: width must be >= 0, got {width_s}")
    w = int(round(width_s * fs_hz))
    a = (w - 1) / 2.0
    if a <= 0:
        return np.array([1.0], dtype=np.float64)
    kmax = int(np.ceil(a)) - 1
    k = np.arange(-kmax, kmax + 1, dtype=np.float64)
    h = np.maximum(0.0, 1.0 - np.abs(k) / a)
    return h / h.sum()


def smo(x, width_s: float, fs_hz: float) -> np.ndarray:
    """FAMOS ``Smo(x, width_s)`` -- zero-phase triangular moving average.

    **Edges** replicate the edge value outward, per the manual: "the data sets
    are extended with the same values as the edges". Measured against the
    export's Latacc_LPF -> Latacc pair, replication beats kernel renormalisation
    by 4.5x over the leading half-window (rms 1.04e-04 vs 4.66e-04) and 5.4x
    over the trailing one (1.05e-03 vs 5.72e-03); the interior is bit-identical
    under either, since the convention reaches only one half-window in.

    **Interior NaN gaps** keep the renormalising treatment. A gap is a missing
    measurement, not an array end: replicating across it would invent data, and
    letting it into the convolution would smear one dropout across a whole
    window. The gap is restored as NaN afterwards, so it survives as a gap.
    """
    arr = as_f64(x, "smo input")
    if width_s <= 0:
        return arr.copy()
    h = smo_kernel(width_s, fs_hz)
    if h.size <= 1:
        return arr.copy()
    if h.size > arr.size:
        raise ValueError(
            f"smo: window is {h.size} samples but the record is only "
            f"{arr.size}. The manual requires the interval width not exceed "
            "the length of the data set.")

    mask = np.isfinite(arr)
    if not mask.any():
        return np.full_like(arr, np.nan)

    # Explicit edge padding + np.convolve(mode="valid"), NOT ndimage's
    # convolve1d(mode="nearest"). The two are mathematically identical but sum
    # in different orders, so they disagree at ~24 ulp -- which meant the
    # chunked path could never be bit-identical to the whole-record path. One
    # code path for both makes chunked == unchunked exactly, which is what lets
    # a long record be processed in pieces with no risk of a silent difference.
    hw = (h.size - 1) // 2
    filled = np.where(mask, arr, 0.0)
    w = mask.astype(np.float64)
    num = np.convolve(_pad_edges(filled, hw), h, mode="valid")
    den = np.convolve(_pad_edges(w, hw), h, mode="valid")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~np.isfinite(out)] = np.nan
    out[~mask] = np.nan
    return out


# ------------------------------------------------------------ FAMOS FiltLP

def filtlp_sos(order: int, cutoff_hz: float, fs_hz: float,
               character: int = CHARACTER_BUTTERWORTH,
               parameter: float = 0.0) -> np.ndarray:
    """Design the ``FiltLP`` filter as second-order sections.

    Second-order sections rather than a transfer-function pair: FAMOS realises
    the filter as cascaded biquads, and tf form loses precision as the order
    climbs. At order 4 the two agree to ~1e-12, so this is robustness, not a
    numerical change.

    Only Butterworth is implemented, because that is what the production recipe
    calls (`FiltLP(Lat_acc, 0, 0, 4, 5)`, SvCharacter=0). The other three
    characteristics raise rather than quietly substituting Butterworth, which
    would produce plausible output from the wrong filter.
    """
    if character != CHARACTER_BUTTERWORTH:
        raise NotImplementedError(
            f"FiltLP SvCharacter={character} is not implemented "
            "(0=Butterworth is; 1=Bessel, 2=Chebychev, 3=Critical damping are "
            "not). Refusing to substitute Butterworth for a filter you asked "
            "to be something else.")
    if parameter != 0.0:
        raise ValueError(
            "FiltLP SvParameter is the Chebychev ripple and must be 0 for "
            f"Butterworth; got {parameter}")
    if order < 1:
        raise ValueError(f"FiltLP: order must be >= 1, got {order}")
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError(f"FiltLP: sample rate must be positive, got {fs_hz}")
    nyq = 0.5 * fs_hz
    if not 0.0 < cutoff_hz < nyq:
        raise ValueError(
            f"FiltLP: cutoff {cutoff_hz} Hz must lie in (0, Nyquist={nyq}). "
            "Refusing to clamp -- a clamped cutoff yields plausible but wrong "
            "output. The manual warns the amplitude response degrades as the "
            "cutoff approaches the sampling frequency.")
    # bilinear transformation, as the manual specifies
    return butter(order, cutoff_hz / nyq, btype="low", output="sos")


def filtlp(x, cutoff_hz: float, order: int, fs_hz: float,
           init: str = INIT_FAMOS,
           character: int = CHARACTER_BUTTERWORTH,
           parameter: float = 0.0) -> np.ndarray:
    """FAMOS ``FiltLP(x, character, parameter, order, cutoff)``.

    **Phase.** Single-pass causal. `FiltLpZ` is the manual's separate
    "without phase shift" function, so plain FiltLP shifts phase by design. The
    group delay in the output is FAMOS's own and must be preserved, not removed.

    **Initial conditions.** ``init="famos"`` uses step-response initialisation
    so ``y[0] == x[0]``; ``init="legacy_zero_state"`` reproduces pre-fix output
    and exists only for that. See the module docstring for the evidence.

    **Gaps.** NaN cannot pass through a recursive filter -- one NaN poisons
    every later sample. Gaps are bridged for the pass, then restored, so a
    dropout stays exactly as long as it was.
    """
    if init not in (INIT_FAMOS, INIT_LEGACY_ZERO_STATE):
        raise ValueError(f"FiltLP: unknown init {init!r}")
    arr = as_f64(x, "filtlp input")
    sos = filtlp_sos(order, cutoff_hz, fs_hz, character, parameter)
    mask = np.isfinite(arr)
    if mask.sum() < 4 * order:
        raise ValueError(
            f"FiltLP: only {int(mask.sum())} finite samples for an order-{order} "
            "filter; too few to filter meaningfully.")
    work = arr.copy()
    if not mask.all():
        idx = np.arange(arr.size)
        work[~mask] = np.interp(idx[~mask], idx[mask], arr[mask])

    if init == INIT_FAMOS:
        y, _ = sosfilt(sos, work, zi=sosfilt_zi(sos) * work[0])
    else:
        y = sosfilt(sos, work)
    y[~mask] = np.nan
    return y


# --------------------------------------------------------------- FAMOS red

def red(x, factor: int) -> np.ndarray:
    """FAMOS ``red(x, n)`` -- keep every n-th sample. No anti-alias filter.

    This is deliberately NOT `scipy.signal.decimate`. FAMOS `red` is a plain
    stride, which is only safe because the recipe applies it last, once the
    signal is already band-limited by FiltLP/smo. Substituting an anti-aliased
    decimation gives a smoother, different answer that is not FAMOS.

    Phase: the first sample of each block is kept (0, n, 2n, ...), so the
    decimated grid still starts at x0. NOT yet confirmed against FAMOS -- the
    golden corpus settles it, and a wrong phase would show as a fixed sample
    lag. `famos/GOLDEN_CORPUS.md` records this as open.
    """
    arr = as_f64(x, "red input")
    if factor < 1:
        raise ValueError(f"red: factor must be >= 1, got {factor}")
    if factor == 1:
        return arr.copy()
    return np.ascontiguousarray(arr[::factor])


def red_time(n_out: int, x0: float, dx: float, factor: int) -> Tuple[np.ndarray, float]:
    """Time axis after ``red``: dx scales by the factor, x0 is unchanged."""
    new_dx = dx * factor
    return x0 + np.arange(n_out, dtype=np.float64) * new_dx, new_dx
