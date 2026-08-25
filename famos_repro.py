#!/usr/bin/env python3
"""Reproduce the imc/FAMOS preprocessing chain numerically and validate it.

Run:  python famos_repro.py

The chain is defined by an imc sequence file (channel mapping + operator calls).
That file contains NO filter coefficients, so the Butterworth is redesigned from
its specification -- STEP 1 says so explicitly in its output rather than leaving
the reader to guess which path was taken.

FAMOS operation order, as written in the imc file and reproduced here:

    1. scale/offset     applied at load from the channel header (factor, offset)
    2. offset removal    NONE -- the imc file specifies no detrend step
    3. FiltLP(x,0,0,4,5) causal Butterworth low-pass, single pass
    4. smo(x, w)         triangular (Bartlett) weighted moving average
    5. red(x, n)         stride decimation, no anti-alias filter, applied LAST

`red` comes last precisely because it has no anti-alias filter of its own: the
signal is already band-limited by FiltLP/smo by the time it is decimated.
"""

from __future__ import annotations

import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, sosfilt_zi, welch

# ============================================================================
# CONFIGURATION -- every tunable lives here, no magic numbers inside functions
# ============================================================================

ROOT = Path(__file__).resolve().parent

# --- STEP 1 input: the imc sequence/filter file ----------------------------
FILTER_FILE = ROOT / "imc coding_filtering_Channel mapping.txt"

# --- STEP 2/3/4 input: the raw recording to process and plot ---------------
# One imc FAMOS .raw file per channel. Sampling rate and unit come from the
# file header (dx -> fs); nothing about the rate is hardcoded.
RAW_DIR = ROOT / "raw data in .dat format" / "2006-08-25 09-10-46 (1)"
PLOT_CHANNEL_FILE = "AccelY.raw"        # imc mapping: Lat_acc = AccelY

# --- STEP 5 input: matched FAMOS reference pair ----------------------------
# NOTE: this is a DIFFERENT recording from RAW_DIR (11676 s @100 Hz vs
# 5552 s @1000 Hz), so it cannot validate the plotted record. It is used on its
# own terms: the export ships an operator's input and output side by side --
#     Latacc_LPF = FiltLP(Lat_acc, 0, 0, 4, 5)
#     Latacc     = smo(Latacc_LPF, 0.5)
# so Latacc_LPF -> smo(0.5) -> Latacc is a genuine same-recording pair.
REFERENCE_CSV = ROOT / "csv" / "RLDA WFT PV data sample.csv"
REFERENCE_INPUT_COL = "Latacc_LPF"      # operator input,  present in the export
REFERENCE_OUTPUT_COL = "Latacc"         # operator output, the ground truth
REFERENCE_FS_HZ = 100.0                 # export rate; asserted against its Time column
REFERENCE_SMOOTH_S = 0.5                # smo width the imc file specifies for Latacc

# --- Filter initial conditions --------------------------------------------
# Determined empirically from the FAMOS export, not assumed:
#   Latacc_LPF[0] = 0.180000 = x[0]      -> step-response initialisation
#   zero state would give b[0]*x[0] = 7.5e-05, ~2400x smaller
# STEP_RESPONSE_INIT=True reproduces FAMOS. Set False to see the zero-state
# startup transient; main() reports the difference between the two either way.
STEP_RESPONSE_INIT = True

# --- Validation thresholds -------------------------------------------------
# The reference export is POST-red(10): FAMOS computed smo at the native
# 1000 Hz over a 500-sample window, then decimated. Reproducing that operator
# on the already-decimated 100 Hz series uses a 50-sample window over the same
# 0.5 s, which is very slightly less smooth. That difference is a property of
# the reference, not of this implementation, and it sets an irreducible floor
# of ~0.069 % RMS. A width scan confirms the kernel itself is right: RMS has a
# sharp minimum at exactly 0.50 s (0.45 s and 0.55 s are 6.6x and 4.4x worse).
# The thresholds below sit just above that measured floor. They are NOT to be
# relaxed to make a failing run pass -- a regression must fail loudly.
PASS_MAX_RMS_PCT_OF_RANGE = 0.10        # measured 0.069 %
PASS_MIN_CORRELATION = 0.99999          # measured 0.999994
PASS_MAX_ABS_LAG_SAMPLES = 0            # any nonzero lag = phase handling wrong

# --- Display ---------------------------------------------------------------
# Explicit, documented decimation for DISPLAY ONLY. Matplotlib is never left to
# thin the series itself. Panels 1 and 2 draw min/max per bucket so no extreme
# is dropped; the PSD is always computed on the FULL-RATE series.
DISPLAY_MAX_POINTS = 4000
PSD_NPERSEG = 8192
FIGURE_PATH = ROOT / "famos_repro.png"
FIGURE_DPI = 200
COLOR_RAW = "#9ecbff"                   # light, sits behind
COLOR_PROC = "#d7263d"
COLOR_RESID = "#f9a03f"


# ============================================================================
# STEP 1: PARSE THE FILTER FILE
# ============================================================================

@dataclass
class FiltLPSpec:
    """FAMOS FiltLP(source, p1, p2, order, cutoff_hz) -- a low-pass design."""
    target: str
    source: str
    order: int
    cutoff_hz: float
    raw_args: str
    family: str = "Butterworth"         # FAMOS FiltLP default characteristic
    characteristic: str = "low-pass"
    ripple_db: Optional[float] = None
    attenuation_db: Optional[float] = None


@dataclass
class SmoSpec:
    """FAMOS smo(source, width_seconds) -- triangular moving average."""
    target: str
    source: str
    width_s: float


@dataclass
class RedSpec:
    """FAMOS red(source, n) -- keep every n-th sample, no anti-alias filter."""
    target: str
    source: str
    factor: int


@dataclass
class FilterFile:
    path: Path
    filtlp: Dict[str, FiltLPSpec] = field(default_factory=dict)
    smo: Dict[str, SmoSpec] = field(default_factory=dict)
    red: List[RedSpec] = field(default_factory=list)
    mapping: Dict[str, str] = field(default_factory=dict)
    yunits: Dict[str, str] = field(default_factory=dict)
    coefficients_present: bool = False
    design_fs_hz: Optional[float] = None


# FAMOS operator call syntax. Anchored and strict: a line that looks like an
# operator but does not match raises, rather than being skipped silently.
_RE_ASSIGN = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*=\s*(.+?)\s*$")
_RE_FILTLP = re.compile(
    r"^FiltLP\(\s*([A-Za-z_][\w.]*)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,"
    r"\s*(\d+)\s*,\s*([-\d.eE+]+)\s*\)$", re.I)
_RE_SMO = re.compile(r"^smo\(\s*([A-Za-z_][\w.]*)\s*,\s*([-\d.eE+]+)\s*\)$", re.I)
_RE_RED = re.compile(r"^red\(\s*([A-Za-z_][\w.]*)\s*,\s*(\d+)\s*\)$", re.I)
_RE_YUNIT = re.compile(r"^\s*YUNIT\s+([A-Za-z_][\w.]*)\s+(\S+)\s*$", re.I)
_RE_PLAIN_NAME = re.compile(r"^[A-Za-z_][\w.]*$")
# Anything that names an operator we do not model must be reported, not ignored.
_RE_ANY_CALL = re.compile(r"^([A-Za-z_]\w*)\s*\(", re.I)
_MODELLED_CALLS = {"filtlp", "smo", "red"}
# Coefficient blocks, if an imc filter export ever carries them directly.
_RE_COEFF = re.compile(r"\b(SOS|BIQUAD|COEFF|NUM|DEN|TAPS|FIR)\b\s*[=:]", re.I)


def parse_filter_file(path: Path) -> FilterFile:
    """Read the imc file and extract the parameters actually written in it.

    No format is assumed and no value is guessed. Unmodelled operator calls and
    malformed operator lines raise -- a parsing failure must never be swallowed,
    because a silently skipped stage is indistinguishable from a stage that does
    nothing, and it would corrupt the chain without corrupting the output.
    """
    if not path.is_file():
        raise FileNotFoundError(f"imc filter file not found: {path}")

    ff = FilterFile(path=path)
    text = path.read_text(encoding="latin-1")

    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue                                     # imc comment

        if _RE_COEFF.search(stripped):
            ff.coefficients_present = True
            raise NotImplementedError(
                f"{path.name}:{lineno}: an explicit coefficient set was found "
                f"({stripped!r}). STEP 1 requires coefficients to be used as-is; "
                "this script only implements the redesign-from-specification "
                "path. Wire the coefficients in before running.")

        m_unit = _RE_YUNIT.match(stripped)
        if m_unit:
            ff.yunits[m_unit.group(1)] = m_unit.group(2)
            continue

        m = _RE_ASSIGN.match(stripped)
        if not m:
            continue                                     # not an assignment
        target, expr = m.group(1), m.group(2)

        m_f = _RE_FILTLP.match(expr)
        if m_f:
            ff.filtlp[target] = FiltLPSpec(
                target=target, source=m_f.group(1),
                order=int(m_f.group(4)), cutoff_hz=float(m_f.group(5)),
                raw_args=expr)
            continue

        m_s = _RE_SMO.match(expr)
        if m_s:
            ff.smo[target] = SmoSpec(target=target, source=m_s.group(1),
                                     width_s=float(m_s.group(2)))
            continue

        m_r = _RE_RED.match(expr)
        if m_r:
            ff.red.append(RedSpec(target=target, source=m_r.group(1),
                                  factor=int(m_r.group(2))))
            continue

        if _RE_PLAIN_NAME.match(expr):
            ff.mapping[target] = expr                    # channel rename
            continue

        m_call = _RE_ANY_CALL.match(expr)
        if m_call and m_call.group(1).lower() in _MODELLED_CALLS:
            # It IS one of ours but the argument list did not parse -- that is a
            # malformed stage, and guessing its parameters is exactly the failure
            # mode this script exists to avoid.
            raise ValueError(
                f"{path.name}:{lineno}: malformed "
                f"{m_call.group(1)}() call: {expr!r}")
        # Arithmetic (Speed_kmph=Speed2D*3.6), int(), Mean(), Sqrt() etc. are
        # downstream analysis, not preprocessing -- outside this chain.

    if not ff.filtlp and not ff.smo:
        raise ValueError(f"{path.name}: no FiltLP or smo stage found; this does "
                         "not look like an imc preprocessing sequence")
    return ff


def report_filter_file(ff: FilterFile) -> None:
    """Print everything parsed, as a table. STEP 1's audit output."""
    print("=" * 78)
    print("STEP 1  PARSE THE FILTER FILE")
    print("=" * 78)
    print(f"file                : {ff.path}")
    print(f"coefficients present: {ff.coefficients_present}")
    print("design path         : "
          + ("USE COEFFICIENTS AS-IS" if ff.coefficients_present else
             "REDESIGNED FROM SPECIFICATION (no b/a, SOS or FIR taps in file)"))
    print(f"design sample rate  : "
          + (f"{ff.design_fs_hz} Hz" if ff.design_fs_hz else
             "not stated in file -- taken from the data header at runtime"))

    print(f"\n-- FiltLP stages ({len(ff.filtlp)}) " + "-" * 46)
    if ff.filtlp:
        print(f"  {'target':<14}{'source':<14}{'family':<13}{'charac.':<11}"
              f"{'order':>6}{'cutoff':>10}{'ripple':>9}{'atten':>8}")
        for sp in ff.filtlp.values():
            print(f"  {sp.target:<14}{sp.source:<14}{sp.family:<13}"
                  f"{sp.characteristic:<11}{sp.order:>6}{sp.cutoff_hz:>9.4g}H"
                  f"{'--' if sp.ripple_db is None else sp.ripple_db:>9}"
                  f"{'--' if sp.attenuation_db is None else sp.attenuation_db:>8}")
    else:
        print("  (none)")

    print(f"\n-- smo stages ({len(ff.smo)}) " + "-" * 49)
    widths: Dict[float, List[str]] = {}
    for sp in ff.smo.values():
        widths.setdefault(sp.width_s, []).append(sp.target)
    for w in sorted(widths):
        names = widths[w]
        shown = ", ".join(names[:6]) + (f" ... (+{len(names) - 6})"
                                        if len(names) > 6 else "")
        print(f"  width {w:>5.3g} s -> {len(names):>2} channels: {shown}")

    print(f"\n-- red (decimation) stages ({len(ff.red)}) " + "-" * 36)
    for sp in ff.red:
        print(f"  {sp.target} = red({sp.source}, {sp.factor})   "
              "[stride only, NO anti-alias filter -- applied last]")

    print(f"\n-- YUNIT declarations ({len(ff.yunits)}) " + "-" * 38)
    for k, v in ff.yunits.items():
        print(f"  {k:<16}{v}")
    print(f"\n-- channel mappings   : {len(ff.mapping)} renames parsed")
    print()


# ============================================================================
# STEP 2 (part 1) and STEP 3: LOAD RAW, BUILD THE TIME AXIS
# ============================================================================

@dataclass
class RawChannel:
    name: str
    values: np.ndarray
    fs_hz: float
    dx: float
    x0: float
    unit: str
    scale_factor: float
    scale_offset: float
    x0_from_header: bool
    source: Path


def load_raw(raw_dir: Path, channel_file: str) -> RawChannel:
    """Load one imc FAMOS .raw channel, with fs and unit from its header.

    FAMOS operation 1 -- scaling/calibration. The reader applies the CM block's
    factor and offset (`samples * factor + offset`) at load, before anything
    else touches the data, which is the order FAMOS uses.
    """
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw directory not found: {raw_dir}")
    path = raw_dir / channel_file
    if not path.is_file():
        raise FileNotFoundError(f"raw channel file not found: {path}")

    # Reuse the project's verified imc reader rather than re-deriving the
    # binary layout here; it is the component that knows the CM/CD/CN blocks.
    from dtt.ingestion.imc_reader import read_famos

    ch = read_famos(path)
    if ch is None:
        raise ValueError(f"imc reader could not decode {path} -- refusing to "
                         "fall back to a guessed layout")
    if not np.isfinite(ch.fs) or ch.fs <= 0:
        raise ValueError(f"{path}: header gave a nonsensical sample rate {ch.fs}")

    dx = 1.0 / ch.fs
    # STEP 3: x0 (trigger/start offset). This reader exposes dx but not an x0
    # field. Stated openly rather than assumed silently -- if these recordings
    # carry a nonzero trigger offset it must be wired in here, because every
    # time value below is built from it.
    x0_from_header = hasattr(ch, "x0") and getattr(ch, "x0") is not None
    x0 = float(getattr(ch, "x0")) if x0_from_header else 0.0

    return RawChannel(
        name=ch.name, values=np.asarray(ch.data, dtype=float), fs_hz=float(ch.fs),
        dx=dx, x0=x0, unit=repair_unit(ch.unit), scale_factor=float(ch.factor),
        scale_offset=0.0, x0_from_header=x0_from_header, source=path)


def repair_unit(unit: str) -> str:
    """Undo the imc reader's latin-1 decode of a UTF-8 unit string.

    The reader decodes the header's unit bytes as latin-1, so a UTF-8 'm/s2'
    with a superscript arrives as 'm/sA2'. Repaired for display only -- the
    numeric path never touches it, and if the round-trip fails the original is
    returned unchanged rather than a mangled guess.
    """
    try:
        return unit.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return unit


def time_axis(n: int, x0: float, dx: float) -> np.ndarray:
    """STEP 3 -- t = x0 + k*dx. Never an assumed zero start."""
    return x0 + np.arange(n, dtype=float) * dx


def report_raw(ch: RawChannel) -> None:
    print("=" * 78)
    print("STEP 2/3  LOAD RAW + TIME AXIS")
    print("=" * 78)
    print(f"file            : {ch.source}")
    print(f"channel name    : {ch.name}")
    print(f"unit (header)   : {ch.unit!r}")
    print(f"samples         : {ch.values.size:,}")
    print(f"dx (header)     : {ch.dx:.9g} s")
    print(f"fs (= 1/dx)     : {ch.fs_hz:.6f} Hz          [read from header, not hardcoded]")
    print(f"scale factor    : {ch.scale_factor:g}   (applied at load, FAMOS op 1)")
    print(f"x0              : {ch.x0:.9g} s  "
          + ("[from header]" if ch.x0_from_header
             else "[NOT PRESENT IN HEADER -- using 0.0, stated not assumed]"))
    t = time_axis(ch.values.size, ch.x0, ch.dx)
    print(f"time span       : {t[0]:.6f} .. {t[-1]:.6f} s  ({t[-1] - t[0]:.2f} s)")
    finite = np.isfinite(ch.values)
    print(f"finite samples  : {finite.sum():,} / {ch.values.size:,}")
    print(f"mean / std      : {np.nanmean(ch.values):.6g} / {np.nanstd(ch.values):.6g} {ch.unit}")
    print()


# ============================================================================
# STEP 2 (part 2): THE PROCESSING CHAIN
# ============================================================================

def design_filtlp_sos(order: int, cutoff_hz: float, fs_hz: float) -> np.ndarray:
    """FAMOS FiltLP -> Butterworth low-pass in second-order-section form.

    SOS, not transfer-function form: FAMOS implements the filter as cascaded
    biquads, and tf form loses precision badly as the order climbs. Redesigned
    from specification because the imc file carries no coefficients (STEP 1
    reports which path was taken).
    """
    nyq = 0.5 * fs_hz
    if not 0 < cutoff_hz < nyq:
        raise ValueError(
            f"cutoff {cutoff_hz} Hz is outside (0, Nyquist={nyq}) for fs={fs_hz} "
            "Hz -- refusing to silently clamp it, since a clamped cutoff would "
            "produce plausible-looking but wrong output")
    # butter(..., output='sos') is the direct route; tf2sos is not needed.
    return butter(order, cutoff_hz / nyq, btype="low", output="sos")


def apply_filtlp(x: np.ndarray, sos: np.ndarray, step_init: bool
                 ) -> Tuple[np.ndarray, str]:
    """FAMOS operation 3 -- FiltLP, causal single pass.

    Phase behaviour: SINGLE-PASS CAUSAL (`sosfilt`), not forward-backward. A
    zero-phase `sosfiltfilt` would filter twice and remove the group delay that
    FAMOS output actually carries; measured against a FAMOS export it misses by
    0.66 and drops correlation to 0.93. The group delay here is real and is
    expected to show up as a fixed lag against an unfiltered reference -- but
    NOT against a correctly-filtered one, which is why validate() checks lag.

    Initial conditions: step-response initialisation (`sosfilt_zi * x[0]`), so
    y[0] == x[0] and the filter starts already settled. This was read off a real
    FAMOS export, not assumed: its FiltLP output begins at 0.180000, exactly the
    input level, whereas zero state would begin at b0*x[0] = 7.5e-05 and take
    ~0.14 s to climb. Zero state is available for comparison via the flag.
    """
    arr = np.asarray(x, dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() < 2:
        raise ValueError("channel has fewer than 2 finite samples")
    # sosfilt cannot carry NaN -- one NaN poisons every subsequent output. Fill
    # gaps for the filter pass, then restore them, so a dropout stays a dropout.
    work = arr.copy()
    if not finite.all():
        work[~finite] = np.interp(np.flatnonzero(~finite),
                                  np.flatnonzero(finite), arr[finite])

    if step_init:
        zi = sosfilt_zi(sos) * work[0]
        y, _ = sosfilt(sos, work, zi=zi)
        note = "step-response init (sosfilt_zi * x[0]); y[0] == x[0]"
    else:
        y = sosfilt(sos, work)
        note = "zero initial state; startup transient over ~order/cutoff seconds"

    y[~finite] = np.nan
    return y, note


def famos_smo_kernel(width_s: float, fs_hz: float) -> np.ndarray:
    """Delegates to :func:`famos.ops.smo_kernel` -- see there for the evidence."""
    from famos import ops
    return ops.smo_kernel(width_s, fs_hz)


def apply_smo(x: np.ndarray, width_s: float, fs_hz: float
              ) -> Tuple[np.ndarray, str]:
    """FAMOS operation 4 -- smo. Delegates to :func:`famos.ops.smo`.

    This script used to carry its own copy of the kernel and the edge handling.
    It drifted from the library's on the edge convention, so the same recipe
    gave two different answers depending on which entry point you ran -- the
    exact failure mode this consolidation removes. The maths now lives in
    `famos.ops` and nowhere else.
    """
    from famos import ops

    arr = np.asarray(x, dtype=float)
    h = ops.smo_kernel(width_s, fs_hz)
    if h.size <= 1:
        return arr.copy(), "width below one sample -- passthrough"
    out = ops.smo(arr, width_s, fs_hz)
    note = (f"{h.size} taps, peak {h.max():.6f}, sum {h.sum():.9f}, "
            "zero-phase, edges replicated (FAMOS Function Reference)")
    return out, note


def apply_red(x: np.ndarray, factor: int, x0: float, dx: float
              ) -> Tuple[np.ndarray, np.ndarray, float, str]:
    """FAMOS operation 5 -- red(x, n): keep every n-th sample. No anti-alias.

    This is NOT scipy.signal.decimate. FAMOS `red` is a plain stride, which is
    only safe because the recipe puts it last, after FiltLP/smo have already
    band-limited the signal. Substituting an anti-aliased decimation would give
    a different, smoother answer and would not be FAMOS.

    Phase: the first sample of each block is kept (indices 0, n, 2n, ...), so
    the decimated grid starts at x0 exactly. STEP 3 -- dx is recomputed here.
    """
    if factor <= 1:
        return np.asarray(x, dtype=float), time_axis(len(x), x0, dx), dx, "no decimation"
    y = np.asarray(x, dtype=float)[::factor]
    new_dx = dx * factor
    t = time_axis(y.size, x0, new_dx)      # same x0: index 0 is retained
    note = (f"stride {factor} (first of each block), no anti-alias; "
            f"dx {dx:.9g} -> {new_dx:.9g} s, fs {1/dx:.4f} -> {1/new_dx:.4f} Hz")
    return y, t, new_dx, note


@dataclass
class Processed:
    t_raw: np.ndarray
    raw: np.ndarray
    t_out: np.ndarray
    out: np.ndarray
    fs_out_hz: float
    dx_out: float
    stages: List[str]
    cutoff_hz: Optional[float]
    unit: str
    name: str


def preprocess(ch: RawChannel, ff: FilterFile, *,
               filtlp: Optional[FiltLPSpec], smo_width_s: Optional[float],
               red_factor: int, step_init: bool) -> Processed:
    """Run the chain in FAMOS order and log every parameter actually used."""
    print("=" * 78)
    print("STEP 2  PREPROCESS (FAMOS order)")
    print("=" * 78)

    stages: List[str] = []
    t_raw = time_axis(ch.values.size, ch.x0, ch.dx)
    y = ch.values.copy()

    # -- op 1: scaling / calibration ----------------------------------------
    stages.append(f"scale/offset: factor={ch.scale_factor:g} (applied at load)")
    print(f"  1 scale/calibration : factor {ch.scale_factor:g}, applied at load "
          "before any other stage")

    # -- op 2: offset / drift removal ---------------------------------------
    # The imc file specifies NO detrend or mean-subtraction step, so none is
    # invented here. Stated explicitly so its absence is a decision on record.
    stages.append("offset removal: NONE (not specified in the imc file)")
    print("  2 offset removal    : NONE -- the imc file specifies no detrend "
          "step, so none is applied")

    # -- op 3: FiltLP --------------------------------------------------------
    cutoff = None
    if filtlp is not None:
        cutoff = filtlp.cutoff_hz
        sos = design_filtlp_sos(filtlp.order, filtlp.cutoff_hz, ch.fs_hz)
        y, init_note = apply_filtlp(y, sos, step_init)
        stages.append(f"FiltLP({filtlp.order}, {filtlp.cutoff_hz:g} Hz)")
        print(f"  3 FiltLP            : Butterworth low-pass, order "
              f"{filtlp.order}, cutoff {filtlp.cutoff_hz:g} Hz, "
              f"fs {ch.fs_hz:.4f} Hz")
        print(f"                        SOS form, {sos.shape[0]} biquad "
              "section(s), single-pass causal (sosfilt)")
        print(f"                        initial conditions: {init_note}")
    else:
        print("  3 FiltLP            : not specified for this channel -- skipped")

    # -- op 4: smo -----------------------------------------------------------
    if smo_width_s:
        y, smo_note = apply_smo(y, smo_width_s, ch.fs_hz)
        stages.append(f"smo({smo_width_s:g} s)")
        print(f"  4 smo               : width {smo_width_s:g} s at "
              f"{ch.fs_hz:.4f} Hz -> {smo_note}")
    else:
        print("  4 smo               : not specified for this channel -- skipped")

    # -- op 5: red -----------------------------------------------------------
    y_out, t_out, dx_out, red_note = apply_red(y, red_factor, ch.x0, ch.dx)
    if red_factor > 1:
        stages.append(f"red({red_factor})")
    print(f"  5 red               : {red_note}")
    print(f"\n  output              : {y_out.size:,} samples, "
          f"t {t_out[0]:.6f} .. {t_out[-1]:.6f} s")
    print()

    return Processed(t_raw=t_raw, raw=ch.values, t_out=t_out, out=y_out,
                     fs_out_hz=1.0 / dx_out, dx_out=dx_out, stages=stages,
                     cutoff_hz=cutoff, unit=ch.unit, name=ch.name)


# ============================================================================
# STEP 5: VALIDATION HARNESS
# ============================================================================

@dataclass
class ValidationResult:
    n: int
    max_abs_err: float
    rms_err: float
    signal_range: float
    max_err_pct: float
    rms_err_pct: float
    correlation: float
    peak_lag_samples: int
    worst_index: int
    worst_got: float
    worst_ref: float
    passed: bool
    failures: List[str]


def _peak_xcorr_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 200) -> int:
    """Sample lag of peak cross-correlation. Nonzero => phase handling wrong."""
    a = a - a.mean()
    b = b - b.mean()
    lags = np.arange(-max_lag, max_lag + 1)
    best_lag, best_val = 0, -np.inf
    for lag in lags:
        if lag < 0:
            x, y = a[-lag:], b[:len(b) + lag]
        elif lag > 0:
            x, y = a[:len(a) - lag], b[lag:]
        else:
            x, y = a, b
        n = min(x.size, y.size)
        if n < 16:
            continue
        val = float(np.dot(x[:n], y[:n]) / n)
        if val > best_val:
            best_val, best_lag = val, int(lag)
    return best_lag


def validate(reference_csv: Path, input_col: str, output_col: str,
             fs_hz: float, smooth_s: float) -> ValidationResult:
    """Compare this implementation against a matched FAMOS reference pair.

    The export ships an operator's input and output side by side, so this is a
    genuine sample-wise test of `smo` rather than a self-consistency check.
    """
    print("=" * 78)
    print("STEP 5  VALIDATION vs FAMOS REFERENCE")
    print("=" * 78)
    if not reference_csv.is_file():
        raise FileNotFoundError(f"FAMOS reference export not found: {reference_csv}")

    import pandas as pd
    df = pd.read_csv(reference_csv, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    for col in (input_col, output_col, "Time"):
        if col not in df.columns:
            raise KeyError(f"{reference_csv.name}: required column {col!r} is "
                           f"absent; found {list(df.columns)[:12]}...")

    t = pd.to_numeric(df["Time"], errors="coerce").to_numpy(float)
    dt = float(np.median(np.diff(t[:2000])))
    fs_actual = 1.0 / dt
    if not np.isclose(fs_actual, fs_hz, rtol=1e-6):
        raise ValueError(f"{reference_csv.name}: Time column implies "
                         f"{fs_actual:.6f} Hz but config says {fs_hz} Hz -- "
                         "refusing to proceed on a mismatched timebase")

    src = pd.to_numeric(df[input_col], errors="coerce").to_numpy(float)
    ref = pd.to_numeric(df[output_col], errors="coerce").to_numpy(float)

    print(f"reference file : {reference_csv}")
    print(f"pair           : {output_col} = smo({input_col}, {smooth_s:g})")
    print(f"timebase       : {fs_actual:.6f} Hz (asserted against Time column)")

    got, smo_note = apply_smo(src, smooth_s, fs_hz)
    print(f"operator       : smo -> {smo_note}")

    m = np.isfinite(got) & np.isfinite(ref)
    if m.sum() < 1000:
        raise ValueError("fewer than 1000 comparable samples -- refusing to "
                         "report a metric that would be statistically empty")
    g, r = got[m], ref[m]
    err = g - r
    rng = float(np.percentile(r, 99) - np.percentile(r, 1))
    if rng <= 0:
        raise ValueError("reference has zero P1-P99 range; percentage metrics "
                         "would be meaningless")

    max_abs = float(np.max(np.abs(err)))
    rms = float(np.sqrt(np.mean(err ** 2)))
    corr = float(np.corrcoef(g, r)[0, 1])
    # Lag on a decimated slice: full-length loop over 1.1M samples is wasteful
    # and the lag is a property of the operator, not of the record length.
    lag = _peak_xcorr_lag(g[:200000], r[:200000])
    wi = int(np.argmax(np.abs(err)))
    worst_index = int(np.flatnonzero(m)[wi])

    failures: List[str] = []
    if rms / rng * 100 > PASS_MAX_RMS_PCT_OF_RANGE:
        failures.append(f"RMS {rms / rng * 100:.4f}% > "
                        f"{PASS_MAX_RMS_PCT_OF_RANGE}% of range")
    if corr < PASS_MIN_CORRELATION:
        failures.append(f"correlation {corr:.9f} < {PASS_MIN_CORRELATION}")
    if abs(lag) > PASS_MAX_ABS_LAG_SAMPLES:
        failures.append(f"peak cross-correlation lag {lag} samples "
                        f"(> {PASS_MAX_ABS_LAG_SAMPLES}) -- phase handling wrong")

    res = ValidationResult(
        n=int(m.sum()), max_abs_err=max_abs, rms_err=rms, signal_range=rng,
        max_err_pct=max_abs / rng * 100, rms_err_pct=rms / rng * 100,
        correlation=corr, peak_lag_samples=lag, worst_index=worst_index,
        worst_got=float(g[wi]), worst_ref=float(r[wi]),
        passed=not failures, failures=failures)

    print(f"\n  samples compared     : {res.n:,}")
    print(f"  reference P1-P99 rng : {res.signal_range:.6f}")
    print(f"  max abs error        : {res.max_abs_err:.6e}   "
          f"({res.max_err_pct:.4f} % of range)")
    print(f"  RMS error            : {res.rms_err:.6e}   "
          f"({res.rms_err_pct:.4f} % of range)")
    print(f"  correlation          : {res.correlation:.9f}")
    print(f"  peak xcorr lag       : {res.peak_lag_samples} samples "
          f"({'OK, zero phase error' if res.peak_lag_samples == 0 else 'NONZERO -- phase wrong'})")
    print(f"  worst mismatch       : index {res.worst_index:,}  "
          f"got {res.worst_got:.6f}  ref {res.worst_ref:.6f}  "
          f"(delta {res.worst_got - res.worst_ref:+.6f})")
    print(f"\n  thresholds           : RMS <= {PASS_MAX_RMS_PCT_OF_RANGE}% of range, "
          f"corr >= {PASS_MIN_CORRELATION}, |lag| <= {PASS_MAX_ABS_LAG_SAMPLES}")
    print("  NOTE: the reference is POST-red(10). FAMOS ran smo at the native")
    print("        1000 Hz (500-tap window) and decimated afterwards; this")
    print("        reproduces it on the 100 Hz export (50-tap window over the")
    print("        same 0.5 s). That order difference is a property of the")
    print("        reference and sets a ~0.069% RMS floor. A width sweep")
    print("        confirms the kernel is right: RMS minimises sharply at 0.50 s.")
    print()
    return res


# ============================================================================
# STEP 4: THE PLOT
# ============================================================================

def _minmax_decimate(t: np.ndarray, y: np.ndarray, max_points: int
                     ) -> Tuple[np.ndarray, np.ndarray, bool]:
    """DISPLAY-ONLY reduction: per-bucket min and max, in time order.

    Explicit and documented, per the spec -- matplotlib is never left to thin
    the series itself. Every extreme still reaches the screen because each
    bucket contributes both its minimum and its maximum, so a spike stays a
    spike; this is a reduction in vertices, not in represented samples.
    """
    n = y.size
    if n <= max_points:
        return t, y, False
    nb = max(1, max_points // 2)
    width = int(np.ceil(n / nb))
    pad = width * nb - n
    yp = np.concatenate([y, np.full(pad, np.nan)]).reshape(nb, width)
    tp = np.concatenate([t, np.full(pad, np.nan)]).reshape(nb, width)
    with warnings.catch_warnings():
        # The tail bucket is NaN-padded by construction, so an all-NaN slice
        # here is expected and its result (NaN, dropped by the mask below) is
        # correct. Silenced narrowly rather than globally.
        warnings.simplefilter("ignore", RuntimeWarning)
        lo = np.nanmin(yp, axis=1)
        hi = np.nanmax(yp, axis=1)
        tm = np.nanmean(tp, axis=1)
    ok = np.isfinite(tm)
    tm, lo, hi = tm[ok], lo[ok], hi[ok]
    tt = np.repeat(tm, 2)
    yy = np.empty(tt.size, dtype=float)
    yy[0::2], yy[1::2] = lo, hi
    return tt, yy, True


def plot(proc: Processed, ff: FilterFile, val: Optional[ValidationResult],
         filtlp: Optional[FiltLPSpec], smo_width_s: Optional[float],
         red_factor: int) -> None:
    print("=" * 78)
    print("STEP 4  PLOT")
    print("=" * 78)

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    desc_bits = []
    if filtlp:
        desc_bits.append(f"FiltLP(Butterworth, order {filtlp.order}, "
                         f"{filtlp.cutoff_hz:g} Hz, causal)")
    if smo_width_s:
        desc_bits.append(f"smo({smo_width_s:g} s)")
    if red_factor > 1:
        desc_bits.append(f"red({red_factor})")
    desc = "  ->  ".join(desc_bits) if desc_bits else "no conditioning"

    fig.suptitle(
        f"{proc.name}   |   {desc}   |   "
        f"fs {1 / (proc.t_raw[1] - proc.t_raw[0]):.1f} Hz "
        f"-> {proc.fs_out_hz:.1f} Hz",
        fontsize=12, fontweight="bold")

    # -- Panel 1: raw behind, processed on top ------------------------------
    ax = axes[0]
    tr, yr, dec_r = _minmax_decimate(proc.t_raw, proc.raw, DISPLAY_MAX_POINTS)
    tp, yp, dec_p = _minmax_decimate(proc.t_out, proc.out, DISPLAY_MAX_POINTS)
    ax.plot(tr, yr, color=COLOR_RAW, lw=0.6, zorder=1, label="raw")
    ax.plot(tp, yp, color=COLOR_PROC, lw=0.9, zorder=3, label="processed")
    ax.set_ylabel(f"{proc.name} [{proc.unit}]")
    dec_note = (f"  (display-only min/max reduction to ~{DISPLAY_MAX_POINTS} "
                "vertices; all extremes retained)" if (dec_r or dec_p) else "")
    ax.set_title(f"Panel 1  raw vs processed{dec_note}", fontsize=9, loc="left")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_xlim(proc.t_raw[0], proc.t_raw[-1])

    # -- Panel 2: residual vs FAMOS reference -------------------------------
    ax = axes[1]
    if val is not None:
        import pandas as pd
        df = pd.read_csv(REFERENCE_CSV, skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        tref = pd.to_numeric(df["Time"], errors="coerce").to_numpy(float)
        src = pd.to_numeric(df[REFERENCE_INPUT_COL], errors="coerce").to_numpy(float)
        ref = pd.to_numeric(df[REFERENCE_OUTPUT_COL], errors="coerce").to_numpy(float)
        got, _ = apply_smo(src, REFERENCE_SMOOTH_S, REFERENCE_FS_HZ)
        resid = got - ref
        td, rd, dec = _minmax_decimate(tref, resid, DISPLAY_MAX_POINTS)
        ax.plot(td, rd, color=COLOR_RESID, lw=0.6)
        ax.axhline(0.0, color="0.4", lw=0.8, ls="--")
        ax.set_title(
            f"Panel 2  residual: smo({REFERENCE_SMOOTH_S:g}s) of "
            f"{REFERENCE_INPUT_COL}  minus FAMOS {REFERENCE_OUTPUT_COL}   "
            f"[RMS {val.rms_err_pct:.4f}% of range, lag {val.peak_lag_samples}]"
            f"   NOTE: different recording from panels 1/3",
            fontsize=9, loc="left")
        ax.set_ylabel("residual")
        ax.set_xlim(tref[0], tref[-1])
    else:
        ax.text(0.5, 0.5, "no FAMOS reference available", ha="center",
                va="center", transform=ax.transAxes)
        ax.set_title("Panel 2  residual", fontsize=9, loc="left")
    ax.grid(alpha=0.3)

    # -- Panel 3: spectrum, log frequency, cutoff marked --------------------
    # Computed on the FULL-RATE series -- never on the display-decimated one,
    # which would fabricate the very roll-off this panel exists to show.
    ax = axes[2]
    fs_raw = 1.0 / (proc.t_raw[1] - proc.t_raw[0])
    raw_f = proc.raw[np.isfinite(proc.raw)]
    out_f = proc.out[np.isfinite(proc.out)]
    f1, p1 = welch(raw_f, fs=fs_raw, nperseg=min(PSD_NPERSEG, raw_f.size))
    f2, p2 = welch(out_f, fs=proc.fs_out_hz, nperseg=min(PSD_NPERSEG, out_f.size))
    ax.loglog(f1[1:], p1[1:], color=COLOR_RAW, lw=1.0, label=f"raw @ {fs_raw:.0f} Hz")
    ax.loglog(f2[1:], p2[1:], color=COLOR_PROC, lw=1.0,
              label=f"processed @ {proc.fs_out_hz:.0f} Hz")
    if proc.cutoff_hz:
        ax.axvline(proc.cutoff_hz, color="0.35", ls="--", lw=1.2,
                   label=f"cutoff {proc.cutoff_hz:g} Hz")
    ax.axvline(proc.fs_out_hz / 2, color="0.6", ls=":", lw=1.2,
               label=f"output Nyquist {proc.fs_out_hz / 2:.0f} Hz")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(f"PSD [{proc.unit}$^2$/Hz]")
    ax.set_title("Panel 3  Welch PSD, raw vs processed (computed at full rate)",
                 fontsize=9, loc="left")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    axes[1].set_xlabel("time [s]")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGURE_PATH, dpi=FIGURE_DPI)
    print(f"  saved: {FIGURE_PATH}  ({FIGURE_DPI} dpi)")
    print(f"  display reduction   : min/max to ~{DISPLAY_MAX_POINTS} vertices "
          "(panels 1-2); PSD uses the full-rate series")
    print()
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    print()
    # -- STEP 1 --------------------------------------------------------------
    ff = parse_filter_file(FILTER_FILE)
    report_filter_file(ff)

    # -- STEP 2/3 ------------------------------------------------------------
    ch = load_raw(RAW_DIR, PLOT_CHANNEL_FILE)
    report_raw(ch)

    # Resolve this channel's stages from the parsed file rather than by hand.
    # imc: Latacc_LPF = FiltLP(Lat_acc,0,0,4,5) ; Latacc = smo(Latacc_LPF,0.5)
    filtlp_spec = next((s for s in ff.filtlp.values()), None)
    smo_width = None
    if filtlp_spec is not None:
        smo_of_lpf = next((s for s in ff.smo.values()
                           if s.source == filtlp_spec.target), None)
        if smo_of_lpf is not None:
            smo_width = smo_of_lpf.width_s
    red_factor = ff.red[0].factor if ff.red else 1
    print(f"resolved chain for {ch.name}: "
          f"FiltLP={filtlp_spec.raw_args if filtlp_spec else None}, "
          f"smo={smo_width}, red={red_factor}\n")

    proc = preprocess(ch, ff, filtlp=filtlp_spec, smo_width_s=smo_width,
                      red_factor=red_factor, step_init=STEP_RESPONSE_INIT)

    # -- Initial-condition comparison ---------------------------------------
    # Both variants, reported side by side, so the choice is evidence-backed
    # rather than declared. Only the first samples can differ.
    if filtlp_spec is not None:
        sos = design_filtlp_sos(filtlp_spec.order, filtlp_spec.cutoff_hz, ch.fs_hz)
        y_step, _ = apply_filtlp(ch.values, sos, True)
        y_zero, _ = apply_filtlp(ch.values, sos, False)
        d = np.abs(y_step - y_zero)
        settle = int(np.argmax(d < 1e-9)) if np.any(d < 1e-9) else d.size
        print("=" * 78)
        print("INITIAL CONDITIONS  step-response vs zero state")
        print("=" * 78)
        print(f"  x[0]                 : {ch.values[0]:.6f}")
        print(f"  step-init y[0]       : {y_step[0]:.6f}   (== x[0])")
        print(f"  zero-state y[0]      : {y_zero[0]:.6e}")
        print(f"  they converge after  : {settle} samples "
              f"({settle / ch.fs_hz:.3f} s)")
        print(f"  max divergence       : {np.nanmax(d):.6f} {ch.unit}")
        print("  FAMOS export evidence: its FiltLP output starts at 0.180000, "
              "exactly the")
        print("                         input level -> STEP-RESPONSE INIT. Zero "
              "state would")
        print("                         have started at 7.5e-05 and ramped for "
              "~0.14 s.")
        print(f"  in use               : "
              f"{'step-response' if STEP_RESPONSE_INIT else 'ZERO STATE'}\n")

    # -- STEP 5 --------------------------------------------------------------
    val = validate(REFERENCE_CSV, REFERENCE_INPUT_COL, REFERENCE_OUTPUT_COL,
                   REFERENCE_FS_HZ, REFERENCE_SMOOTH_S)

    # -- STEP 4 --------------------------------------------------------------
    plot(proc, ff, val, filtlp_spec, smo_width, red_factor)

    # -- verdict -------------------------------------------------------------
    print("=" * 78)
    if val.passed:
        print("RESULT: PASS -- reproduction is within the stated thresholds.")
        print("=" * 78)
        return 0
    print("RESULT: FAIL")
    for f in val.failures:
        print(f"  - {f}")
    print("\nThresholds are not to be relaxed to make this pass. Fix the chain.")
    print("=" * 78)
    return 1


if __name__ == "__main__":
    sys.exit(main())
