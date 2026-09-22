"""FAMOS cross-validation.

Confirms the platform's signal processing reproduces imc **FAMOS** output.

A FAMOS CSV export ships intermediate channels next to their final form —
``Latacc_LPF`` beside ``Latacc`` — which is sample-wise ground truth for the
operators in :mod:`dtt.preprocessing`.

The direction of that pair matters, and it is the opposite of what a
``<name>_LPF`` suffix suggests. The imc recipe reads::

    Latacc_LPF = FiltLP(Lat_acc, 0, 0, 4, 5)
    Latacc     = smo(Latacc_LPF, 0.5)

so ``Latacc`` is the *output* of smoothing ``Latacc_LPF`` — not a raw channel
that ``Latacc_LPF`` was derived from. Scoring it the other way round (running a
low-pass on ``Latacc`` and comparing to ``Latacc_LPF``) measures nothing real;
it fits a filter to an inverted relationship and lands near 96–97 % no matter
how good the code is.

So each pair is tested **both ways** and the relationship that actually holds is
reported:

``smo``     ``smo(suffixed, w)`` vs ``base``      — the imc recipe's own order
``FiltLP``  ``butterworth(base, fc, n)`` vs ``suffixed`` — a genuine raw/filtered pair

Metrics
    match_pct   100·(1 − NRMSE), NRMSE normalised by the reference P1..P99 range
    corr_pct    100·Pearson correlation
    within_pct  % of samples within 2% of the reference range
"""

from __future__ import annotations

import csv as _csv
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from dtt.preprocessing import butterworth_lpf, famos_smooth, FAMOS_AUX_SMOOTH_S

PASS_THRESHOLD = 95.0          # % match accepted as FAMOS-grade
_LPF_SUFFIXES = ("_LPF", "_lpf", "_Filt", "_filt", "_filtered")
_SMOOTH_CANDIDATES = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)


@dataclass
class ChannelMatch:
    channel: str
    reference: str
    match_pct: float
    corr_pct: float
    within_pct: float
    best_cutoff: float
    best_order: int
    n: int
    operation: str = "FiltLP"      # which relationship was found to hold
    best_smooth_s: float = 0.0     # smo width, when operation == "smo"

    @property
    def passed(self) -> bool:
        return self.match_pct >= PASS_THRESHOLD

    def describe(self) -> str:
        if self.operation == "smo":
            return f"smo({self.reference}, {self.best_smooth_s:g}s) -> {self.channel}"
        return (f"FiltLP({self.channel}, {self.best_cutoff:g}Hz, "
                f"order {self.best_order}) -> {self.reference}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passed"] = self.passed
        d["description"] = self.describe()
        return d


def match_metrics(ours: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    """Sample-wise agreement of ``ours`` against ``reference``."""
    m = np.isfinite(ours) & np.isfinite(reference)
    a, b = ours[m], reference[m]
    if a.size < 8:
        return {"match_pct": float("nan"), "corr_pct": float("nan"),
                "within_pct": float("nan"), "n": int(a.size)}
    rng = np.percentile(b, 99) - np.percentile(b, 1)
    rng = rng if rng > 0 else (np.max(b) - np.min(b) or 1.0)
    nrmse = np.sqrt(np.mean((a - b) ** 2)) / rng
    corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 0.0
    within = 100.0 * np.mean(np.abs(a - b) <= 0.02 * rng)
    return {"match_pct": 100.0 * (1.0 - nrmse), "corr_pct": 100.0 * corr,
            "within_pct": float(within), "n": int(a.size)}


def _trim(*arrays, edge: int = 0):
    """Drop ``edge`` samples from both ends so filter start-up is not scored."""
    if edge <= 0:
        return arrays
    return tuple(a[edge:-edge] for a in arrays)


def _fit_filter(raw: np.ndarray, ref: np.ndarray, fs: float
                ) -> tuple[float, int, Dict[str, float]]:
    """Find the (cutoff, order) whose filtered ``raw`` best matches ``ref``.

    Doubles as reverse-engineering FAMOS's filter setting for the channel.
    """
    best = (10.0, 4, {"match_pct": -1e9, "corr_pct": 0.0, "within_pct": 0.0,
                      "n": 0})
    edge = int(fs)
    for order in (2, 3, 4):
        for fc in (0.5, 1, 2, 3, 5, 8, 10, 15, 20, 30, 40):
            if fc >= 0.5 * fs:
                continue
            our = butterworth_lpf(raw, fc, order, fs)
            a, b = _trim(our, ref, edge=edge)
            mm = match_metrics(a, b)
            if mm["match_pct"] > best[2]["match_pct"]:
                best = (fc, order, mm)
    return best


def _fit_smooth(src: np.ndarray, ref: np.ndarray, fs: float
                ) -> tuple[float, Dict[str, float]]:
    """Find the ``smo`` width whose smoothed ``src`` best matches ``ref``."""
    best = (FAMOS_AUX_SMOOTH_S, {"match_pct": -1e9, "corr_pct": 0.0,
                                 "within_pct": 0.0, "n": 0})
    edge = int(fs)
    for width in _SMOOTH_CANDIDATES:
        our = famos_smooth(src, fs, width)
        a, b = _trim(our, ref, edge=edge)
        mm = match_metrics(a, b)
        if mm["match_pct"] > best[1]["match_pct"]:
            best = (width, mm)
    return best


def _find_header_row(path: Path, max_scan: int = 6) -> int:
    """FAMOS CSVs have blank lead lines; return the index of the header row."""
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for i in range(max_scan):
            line = fh.readline()
            cols = [c.strip() for c in next(_csv.reader([line]), [])]
            if any(c == "Time" for c in cols):
                return i
    return 0


def find_reference_pairs(columns: List[str]) -> List[tuple[str, str]]:
    """Return (base_col, suffixed_col) pairs where both members exist."""
    cols = {c.strip(): c.strip() for c in columns}
    pairs: List[tuple[str, str]] = []
    for c in cols:
        for suf in _LPF_SUFFIXES:
            if c.endswith(suf):
                base = c[: -len(suf)]
                if base in cols:
                    pairs.append((base, c))
                break
    return pairs


def crosscheck_csv(csv_path: Path, max_rows: int = 300_000
                   ) -> tuple[List[ChannelMatch], Optional[str]]:
    """Validate our operators against every intermediate/final pair in a CSV.

    Returns (matches, note). ``note`` is set when no reference pair is found.
    """
    csv_path = Path(csv_path)
    hdr = _find_header_row(csv_path)
    header = pd.read_csv(csv_path, skiprows=hdr, nrows=0)
    header.columns = [c.strip() for c in header.columns]
    pairs = find_reference_pairs(list(header.columns))
    if not pairs:
        return [], ("No FAMOS intermediate columns (e.g. '<name>_LPF' beside "
                    "'<name>') found in this export — cannot compute a "
                    "sample-wise match.")

    need = ["Time"] + sorted({c for p in pairs for c in p})
    df = pd.read_csv(csv_path, skiprows=hdr, nrows=max_rows,
                     usecols=lambda c: c.strip() in need, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    t = df["Time"].to_numpy(float)
    dt = np.nanmedian(np.diff(t[:2000]))
    fs = 1.0 / dt if dt and dt > 0 else 100.0

    results: List[ChannelMatch] = []
    for base_col, suf_col in pairs:
        base = df[base_col].to_numpy(float)
        suf = df[suf_col].to_numpy(float)

        # Relationship 1 (the imc recipe's own): base = smo(suffixed, w)
        width, sm = _fit_smooth(suf, base, fs)
        # Relationship 2: suffixed = FiltLP(base, fc, order)
        fc, order, fm = _fit_filter(base, suf, fs)

        if sm["match_pct"] >= fm["match_pct"]:
            results.append(ChannelMatch(
                channel=base_col, reference=suf_col,
                match_pct=round(sm["match_pct"], 3), corr_pct=round(sm["corr_pct"], 3),
                within_pct=round(sm["within_pct"], 2), best_cutoff=0.0, best_order=0,
                n=sm["n"], operation="smo", best_smooth_s=width))
        else:
            results.append(ChannelMatch(
                channel=base_col, reference=suf_col,
                match_pct=round(fm["match_pct"], 3), corr_pct=round(fm["corr_pct"], 3),
                within_pct=round(fm["within_pct"], 2), best_cutoff=fc, best_order=order,
                n=fm["n"], operation="FiltLP"))
    results.sort(key=lambda r: r.match_pct, reverse=True)
    return results, None


# ==========================================================================
# All-channel cross-check against a FAMOS export
#
# `crosscheck_csv` above answers a different question: it compares two columns
# *inside one export* and fits the operator that relates them. That is the
# right tool for the Latacc/Latacc_LPF direction and the wrong one for "does
# our output match FAMOS", because (a) no force or moment channel has an
# intermediate-column pair in any export, so it structurally cannot reach
# them, and (b) fitting the best of 33 cutoff/order combinations reports the
# best achievable agreement rather than the agreement of the code that ships.
#
# What follows scores our own re-derivation, from the same `.raw` files FAMOS
# read, against every column of the export -- per channel, per recipe stage,
# against the per-sample export quantum. It was `tools/score_golden.py`; it
# lives here because the GUI's "Validate vs FAMOS..." action needs it and the
# packaged app bundles `dtt/` but not `tools/`. The CLI still owns the
# printing and the exit status.
# ==========================================================================


# `parse_channel` and `famos_recipe` are imported inside the functions that
# need them, not here: this module is imported by the GUI at start-up and the
# deferred import keeps that path from pulling in the channel model and the
# whole preprocessing module before anything asks for them.
from famos import ops                                               # noqa: E402
from dtt.ingestion.imc_reader import read_famos_all                 # noqa: E402

# --------------------------------------------------------------------------
# Stage A: every channel in the recording, not a hand-picked few.
#
# `_register_forces` above covers six force channels by name, which is what
# the 2026-08-29 capture happened to contain. That left two gaps recorded in
# MODULE1_STATUS.md as *inferred rather than measured*: the moment channels
# (Mx/My/Mz) and Latacc on a real recording. Neither is closed by adding six
# more literals -- the next recording will have different names again -- so
# the channel list comes from the folder, and what each channel should have
# had done to it comes from `famos_recipe`, which is the transcription of
# "imc coding_filtering_Channel mapping.txt". One recipe table, no second
# copy of "forces get smo(0.1)" living in a validation script.
# --------------------------------------------------------------------------

# The imc config file's Step 1 channel map, for the three aux channels whose
# recorded name is not the name the recipe gives the result:
#
#     Long_acc = AccelX            Lat_acc  = AccelY
#     Speed_kmph = Speed2D * 3.6
#
# Read off that file, not invented here; the forces and moments need no entry
# because the recipe keeps their names.
_AUX_SOURCES = {
    "AccelY": ("Latacc", 1.0),
    "AccelX": ("Longacc", 1.0),
    "Speed2D": ("Vehicle_Speed", 3.6),
}

# Channels the recipe *derives* from another channel in the same folder. Some
# recorders also write the derived signal out as its own .raw, and scoring both
# paths would report one channel twice under two names -- the second of which
# the sequence has no reason to export, so it would sit in the report as a
# permanent "absent from the export" that is neither a gap nor a failure.
_DERIVED_DUPLICATES = {
    "Speed_kmph": "the recipe derives this as Speed2D * 3.6; it is scored "
                  "through Vehicle_Speed, from Speed2D.raw",
}

# A FAMOS output column is its input channel plus the stage that made it.
# Longest first: "_rawred" also ends in "_red".
_STAGE_SUFFIXES = ("_rawred", "_lpf_red", "_red", "_smo", "_lpf")

_DECIMATE = 10


def canonical_key(name: str) -> str:
    """A name a FAMOS export and our own channel list can be matched on.

    Matching on the raw string does not survive contact with a real export:
    the same wheel is ``WFT_Fz_fr`` in one recording, ``FR_Fz_2`` in another
    and ``A1R_Fz`` in our own frame. :func:`dtt.channels.parse_channel` already
    resolves all three to one position+component, so a force or moment channel
    is keyed on that.

    Everything else falls back to a normalised name. That is not laziness: the
    channel model deliberately parses *wheel* channels only, and its
    ``_DERIVED_RE`` rejects anything matching speed/accel/angle/rpm outright —
    so ``Latacc`` and ``Vehicle_Speed`` return None from it by design, and a
    single-tier canonical match would silently drop exactly the two channels
    this round exists to measure.
    """
    from dtt.channels import parse_channel

    parsed = parse_channel(name)
    if parsed is not None:
        pos, comp = parsed
        return f"{pos.id}_{comp}"
    return name.strip().lower().replace(" ", "_")


def split_stage(column: str) -> Tuple[str, str]:
    """``(base_channel, stage)`` for a FAMOS output column name."""
    c = column.strip()
    low = c.lower()
    for suf in _STAGE_SUFFIXES:
        if low.endswith(suf):
            return c[: -len(suf)], suf[1:]
    return c, ""


def _recipe_chain(name: str, fs: float, scale: float = 1.0):
    """``{stage: fn}`` — what the imc recipe does to ``name``, stage by stage.

    Built from :func:`dtt.preprocessing.famos_recipe`, so this script cannot
    disagree with the pipeline about what a channel receives. ``scale`` carries
    the one Step-1 map entry that is arithmetic rather than a rename
    (``Speed_kmph = Speed2D * 3.6``).
    """
    from dtt.preprocessing import famos_recipe

    r = famos_recipe(name)
    if not (r.apply_filter or r.smooth_width_s > 0):
        return {}

    def pre(v):
        return v * scale if scale != 1.0 else v

    w = r.smooth_width_s
    chain = {}
    # The raw channel as FAMOS itself read it. Nothing is computed on either
    # side, so a difference here is the file reader alone -- byte offsets,
    # |RC5 event-record stripping, the leading/trailing trim -- and no amount
    # of operator work would fix it. Perfect operators on misread samples
    # still give wrong answers.
    chain["raw"] = pre
    if r.apply_filter:
        cut, order = r.filter_cutoff, r.filter_order
        chain["lpf"] = lambda v: ops.filtlp(pre(v), cut, order, fs)
        chain["smo"] = lambda v: ops.smo(ops.filtlp(pre(v), cut, order, fs), w, fs)
        # The FiltLP intermediate after decimation. Without it a Latacc export
        # taken at the production rate can only score FiltLP and smo together,
        # and a disagreement could not be attributed to either -- which is the
        # same attribution problem the force sequence solved by exporting smo
        # separately from red.
        chain["lpf_red"] = lambda v: ops.red(
            ops.filtlp(pre(v), cut, order, fs), _DECIMATE)
    else:
        chain["smo"] = lambda v: ops.smo(pre(v), w, fs)
    final = chain["smo"]
    chain["red"] = lambda v: ops.red(final(v), _DECIMATE)
    chain["rawred"] = lambda v: ops.red(pre(v), _DECIMATE)
    return chain


def _expected_name(source: str, recipe_name: str, stage: str) -> str:
    """The column name the companion sequence gives this stage.

    A label, not the thing matching is done on -- see :func:`match_export`.
    The convention is the imc config file's own: a channel the recipe renames
    carries its new name on the smoothed result (``Latacc``), one it does not
    carries a stage suffix (``FR_Fz_2_smo``).
    """
    if stage == "raw":
        return source
    renamed = recipe_name != source
    if stage == "smo":
        return recipe_name if renamed else f"{source}_smo"
    if stage == "lpf":
        return f"{recipe_name}_LPF"
    if stage == "lpf_red":
        return f"{recipe_name}_LPF_red"
    return f"{recipe_name}_{stage}"


class Case:
    """One scoreable (channel, stage): how we make it, and how to find it."""

    __slots__ = ("canon", "stage", "expected", "source", "recipe_name",
                 "data", "fn", "fs", "note")

    def __init__(self, canon, stage, expected, source, recipe_name,
                 data, fn, fs, note=""):
        self.canon, self.stage, self.expected = canon, stage, expected
        self.source, self.recipe_name = source, recipe_name
        self.data, self.fn, self.fs, self.note = data, fn, fs, note

    @property
    def key(self):
        return (self.canon, self.stage)

    @property
    def decimated(self) -> bool:
        return self.stage in ("red", "rawred", "lpf_red")


def register_all_channels(raw_dir: Path, cut: Optional[Tuple[int, int]] = None
                          ) -> Tuple[List[Case], Dict[str, str]]:
    """Every conditionable channel in ``raw_dir``, at every recipe stage.

    Returns ``(cases, skipped)``; ``skipped`` maps a raw file to the reason it
    was not registered, because the report has to say why a channel was not
    compared rather than quietly drop it.

    ``cut`` mirrors the sequence's own ``Cut(ch, a, b)``. FAMOS conditions the
    cut region, so we cut before conditioning too -- otherwise ``FiltLP``'s
    step-response initialisation and ``smo``'s edge padding start from
    different samples on the two sides and the first half-window disagrees for
    a reason that has nothing to do with the operators.
    """
    from dtt.ingestion.imc_reader import read_famos

    cases: List[Case] = []
    skipped: Dict[str, str] = {}

    for f in sorted(raw_dir.glob("*.raw")):
        stem = f.stem
        if "copy" in stem.lower():
            # imc exports routinely carry "<name> - Copy" duplicates; scoring
            # one is scoring the same channel twice under a name that cannot
            # be matched back to a single wheel.
            skipped[stem] = "duplicate '- Copy' export of another channel"
            continue
        if stem in _DERIVED_DUPLICATES:
            skipped[stem] = _DERIVED_DUPLICATES[stem]
            continue

        recipe_name, scale = _AUX_SOURCES.get(stem, (stem, 1.0))
        try:
            ch = read_famos(f)
        except Exception as exc:                                 # noqa: BLE001
            skipped[stem] = f"reader raised {type(exc).__name__}: {exc}"
            continue
        if ch is None or not ch.data.size:
            skipped[stem] = ("read_famos returned nothing -- it parses the "
                             "classic |CF layout only, and imc3 files are "
                             "readable through the folder path instead "
                             "(PROJECT_HISTORY §10)")
            continue

        x, fs = ch.data, ch.fs
        if cut is not None:
            a, b = cut
            x = x[a:b]
            if x.size < 64:
                skipped[stem] = f"cut {a}:{b} leaves only {x.size} samples"
                continue

        chain = _recipe_chain(recipe_name, fs, scale)
        if not chain:
            skipped[stem] = ("the imc recipe passes this channel through "
                             "unchanged -- there is no FAMOS operation here "
                             "to check")
            continue

        note = "" if scale == 1.0 else f"x{scale:g} per the Step-1 channel map"
        for stage, fn in chain.items():
            # The raw stage keys on the *recorded* name, every conditioned
            # stage on the name the recipe gives the result. For a force they
            # are the same canonical channel and the stage tells them apart;
            # for Latacc they are genuinely two different names for two
            # different signals, and keying both on "AccelY" would make the
            # unfiltered input and the filtered output collide.
            canon = canonical_key(stem if stage == "raw" else recipe_name)
            cases.append(Case(canon, stage,
                              _expected_name(stem, recipe_name, stage),
                              stem, recipe_name, x, fn, fs, note))
    return cases, skipped


def sniff_export(path: Path) -> tuple:
    """``(sep, header_row, skiprows, column_names)`` for a FAMOS ASCII export.

    A FAMOS ASCII export may carry metadata lines above the channel names, and
    the separator follows the machine's locale, so neither is assumed. The
    layout is settled from the first few lines only: a real force export runs
    to 1.2 GB, and probing that by re-parsing the whole file (let alone with
    the python engine) is not a thing that finishes.

    Split out from :func:`load_outputs` so the column *names* can be read
    without reading the data — the all-channel report has to name every column
    in the export it did **not** compare, and loading a 2 GB file to discover
    what is in it is not an option.
    """
    head = []
    with open(path, "r", encoding="latin1", errors="replace") as fh:
        for _ in range(6):
            line = fh.readline()
            if not line:
                break
            head.append(line)

    sep, hdr_row = ",", 0
    for i, line in enumerate(head):
        for cand in (",", ";", "\t"):
            names = [c.strip() for c in line.split(cand)]
            if len(names) > 2 and sum(bool(n) for n in names) > 2:
                sep, hdr_row = cand, i
                break
        else:
            continue
        break

    # FAMOS writes a units row directly under the channel names. Left in, every
    # channel shifts by one sample and the comparison silently misaligns.
    skip = []
    if len(head) > hdr_row + 1:
        cells = [c.strip() for c in head[hdr_row + 1].split(sep)]
        numeric = 0
        for c in cells:
            try:
                float(c)
                numeric += 1
            except ValueError:
                pass
        if numeric == 0:
            skip = [hdr_row + 1]

    cols = [c.strip() for c in head[hdr_row].split(sep)] if head else []
    return sep, hdr_row, skip, [c for c in cols if c]


def export_columns(path: Path) -> List[str]:
    """Every channel name in a FAMOS export, without reading its data."""
    path = Path(path)
    if path.is_dir():
        names: List[str] = []
        for f in sorted(path.glob("*")):
            if f.suffix.lower() in (".csv",):
                names += export_columns(f)
        return names
    return sniff_export(path)[3]


def load_outputs(path: Path, wanted=None) -> Dict[str, np.ndarray]:
    """FAMOS results, from a directory of .dat files or a single CSV."""
    out: Dict[str, np.ndarray] = {}
    if path.is_dir():
        for f in sorted(path.glob("*")):
            if f.suffix.lower() not in (".dat", ".raw"):
                continue
            try:
                chans = read_famos_all(f)
            except Exception as exc:                             # noqa: BLE001
                print(f"  ! could not read {f.name}: {exc}")
                continue
            # FAMOS writes every selected variable into one file, so a single
            # .dat routinely holds the whole export.
            for ch in chans:
                if ch.data.size:
                    # the name inside the file wins; the filename is a fallback
                    out[(ch.raw_name or f.stem).strip()] = ch.data
        return out

    sep, hdr_row, skip, names = sniff_export(path)
    del names
    # FAMOS pads the header names out to a fixed width, and usecols matches the
    # raw text, so the comparison has to strip before it decides.
    use = (lambda c: str(c).strip() in wanted) if wanted else None
    df = pd.read_csv(path, sep=sep, header=hdr_row, skiprows=skip,
                     usecols=use, engine="c", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    for c in df.columns:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        # One wide export, one shared x-axis. A channel FAMOS decimated is
        # written *sparsely* onto that axis -- for red(x,10), one value every
        # tenth row with blanks between -- so the column has to be compacted
        # back to the channel's own rate before it means anything. Comparing
        # across the blanks reads as a total mismatch (r ~ 0) while the data is
        # in fact identical, which is a very convincing way to be wrong.
        finite = np.flatnonzero(np.isfinite(v))
        if finite.size > 2:
            step = np.diff(finite)
            if step[0] > 1 and np.all(step == step[0]):
                v = v[finite]                       # regular stride: compact
            else:
                v = v[:finite[-1] + 1]              # merely padded at the end
        out[str(c).strip()] = v
    return out


def quantum_ratio(err: np.ndarray, ref: np.ndarray, sig_figs: int = 6) -> float:
    """Worst error as a multiple of *that sample's* own export quantum.

    Six significant figures is a different absolute precision at every
    magnitude: 0.1 at 79,000 but 1.0 the moment a peak crosses 100,000. Judging
    a channel by one quantum taken from its typical value therefore mis-scores
    exactly the samples where the largest errors live -- the biggest ones. So
    each sample is compared against the quantum at its own magnitude, and the
    verdict is the worst of those ratios. At or below 0.5 means every sample
    agrees to within half a stored digit, which is as close as the file can
    record.
    """
    m = np.isfinite(err) & np.isfinite(ref) & (np.abs(ref) > 0)
    if not m.any():
        return 0.0
    q = 10.0 ** (np.floor(np.log10(np.abs(ref[m]))) - (sig_figs - 1))
    return float(np.max(np.abs(err[m]) / q))


def score(ours: np.ndarray, theirs: np.ndarray, skip: int = 0) -> dict:
    """Max/mean absolute error and correlation over the comparable region."""
    n = min(ours.size, theirs.size)
    a, b = ours[:n], theirs[:n]
    if skip:
        # A causal filter's startup transient is real output, not error, but it
        # swamps the comparison; report with and without so neither hides.
        a, b = a[skip:n - skip or None], b[skip:n - skip or None]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return {"n": 0, "max": float("nan"), "mean": float("nan"), "r": float("nan")}
    a, b = a[m], b[m]
    d = np.abs(a - b)
    r = (float(np.corrcoef(a, b)[0, 1])
         if np.std(a) > 0 and np.std(b) > 0 else float("nan"))
    return {"n": int(a.size), "max": float(d.max()),
            "mean": float(d.mean()), "r": r,
            "qratio": quantum_ratio(a - b, b)}


def quantum_within_pct(err: np.ndarray, ref: np.ndarray,
                       sig_figs: int = 6, tol: float = 0.5) -> float:
    """% of samples agreeing to within ``tol`` of their own export quantum.

    The companion to :func:`quantum_ratio`, which reports the single worst
    sample. A channel can be perfect everywhere and carry one bad sample, or be
    uniformly half a digit out; one number cannot tell those apart, and the
    per-channel report wants the share of samples inside tolerance alongside
    the worst case. ``tol = 0.5`` is the rounding floor -- half a stored digit
    is the most a correctly-rounded value can differ by.
    """
    m = np.isfinite(err) & np.isfinite(ref) & (np.abs(ref) > 0)
    if not m.any():
        return float("nan")
    q = 10.0 ** (np.floor(np.log10(np.abs(ref[m]))) - (sig_figs - 1))
    return float(100.0 * np.mean(np.abs(err[m]) / q <= tol))


def match_export(cases: List[Case], columns: List[str]) -> Tuple[Dict, List[str]]:
    """Match export columns to registered cases **canonically**.

    ``(matched, unmatched)``: ``matched`` is ``{case key: column name}``,
    ``unmatched`` lists the export columns nothing claimed -- which the report
    prints rather than discards.

    A bare column with no stage suffix is ambiguous by construction: for a
    force, ``FR_Fz_2`` is the unconditioned input FAMOS read; for an aux
    channel, ``Latacc`` is the *output* of ``smo(Latacc_LPF, 0.5)``. Both
    readings are tried, raw first, and whichever corresponds to a registered
    channel wins.
    """
    index = {c.key: c for c in cases}
    matched: Dict[tuple, str] = {}
    unmatched: List[str] = []
    for col in columns:
        base, stage = split_stage(col)
        canon = canonical_key(base)
        cands = ((canon, stage),) if stage else ((canon, "raw"), (canon, "smo"))
        for cand in cands:
            if cand in index and cand not in matched:
                matched[cand] = col
                break
        else:
            unmatched.append(col)
    return matched, unmatched


# What `processed_data.csv` additionally does that FAMOS does not. Scoring runs
# raw -> famos.ops -> export, in the file's own units, so none of these is in
# the comparison -- but anyone diffing our CSV against a FAMOS export will see
# every one of them, so each channel states which apply to it. They are
# corrections, not errors, and reporting them as failures was the trap.
_CORRECTION_FORCE = "N->daN + CR decade fix (processed_data.csv only)"
_CORRECTION_BLANK = "dropout blanking"
_CORRECTION_STOPS = "stop removal, when enabled (analysis stages only)"


def _corrections(recipe_name: str) -> str:
    from dtt.channels import parse_channel

    notes = [_CORRECTION_BLANK, _CORRECTION_STOPS]
    parsed = parse_channel(recipe_name)
    if parsed is not None and parsed[1] in ("Fx", "Fy", "Fz"):
        notes.insert(0, _CORRECTION_FORCE)
    return "; ".join(notes)


def _latacc_direction(export_paths: List[Path]) -> List[str]:
    """The Latacc/Latacc_LPF relationship, tested in both directions.

    Delegated to :func:`dtt.validation.famos_validation.crosscheck_csv`, which
    already tries ``smo(suffixed) -> base`` against ``FiltLP(base) -> suffixed``
    and reports whichever actually holds. Assuming the (raw, filtered) reading
    the suffix suggests scores ~96.8% no matter how good the code is, because
    it fits a filter to an inverted relationship -- so the direction is
    measured here, never assumed.
    """
    from dtt.validation.famos_validation import crosscheck_csv

    lines: List[str] = []
    for path in export_paths:
        if path.is_dir():
            continue
        try:
            matches, note = crosscheck_csv(path)
        except Exception as exc:                                 # noqa: BLE001
            lines.append(f"{path.name}: could not run "
                         f"({type(exc).__name__}: {exc})")
            continue
        if note:
            lines.append(f"{path.name}: {note}")
            continue
        for m in matches:
            lines.append(f"{path.name}: {m.describe()} -- "
                         f"{m.match_pct:.3f}% match, {m.corr_pct:.3f}% corr, "
                         f"{m.within_pct:.1f}% of samples within 2% (n={m.n:,})")
    return lines


_REPORT_PREAMBLE = (
    "Scored per sample against **that sample's own export quantum**. FAMOS "
    "writes six significant figures, which is 1e-6 at order 1 but 1.0 once an "
    "Fz peak crosses 1e5, so a fixed absolute tolerance reports healthy "
    "channels as failing while r = 1.000000000. `err/q <= 0.5` is the rounding "
    "floor -- as close as the file is capable of recording."
)
_REPORT_UNITS = (
    "The comparison runs `.raw` -> `famos.ops` -> FAMOS export, in the file's "
    "own units. The corrections listed per channel are what "
    "`processed_data.csv` additionally applies; **none of them is in this "
    "comparison, by design**, so a correction of ours can never be scored as a "
    "disagreement with FAMOS."
)


def _group_absent(not_found: List[tuple]) -> List[tuple]:
    """``[(stage, [expected names])]`` — absent stages grouped, not enumerated.

    A sequence that exports only the decimated stages leaves one absent entry
    per channel per stage. Listed one per line that is forty lines of noise
    around the two entries that are genuinely a gap.
    """
    out: Dict[str, List[str]] = {}
    for stage, name in not_found:
        out.setdefault(stage or "raw", []).append(name)
    return sorted(out.items())


def _write_report(path: Path, rows: List[dict], skipped: Dict[str, str],
                  unmatched: List[str], not_found: List[str],
                  direction: List[str], meta: dict) -> None:
    """Per-channel table as CSV, and the whole run as markdown beside it.

    Two formats because they answer different questions: the CSV is the number
    for every channel, sortable and diffable; the markdown carries the reasons,
    which a column of figures cannot.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)

    lines = ["# FAMOS all-channel cross-check", ""]
    for k, v in meta.items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", _REPORT_PREAMBLE, "", _REPORT_UNITS, ""]
    if rows:
        cols = ["channel", "stage", "export_column", "n", "max_err",
                "mean_err", "r", "err_q", "within_pct", "verdict"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for r in rows:
            lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
        lines.append("")
    lines += ["## Not compared, and why", ""]
    if not (skipped or unmatched or not_found):
        lines.append("Every channel in the recording and every column in the "
                     "export was compared.")
    for name, why in sorted(skipped.items()):
        lines.append(f"- **{name}** (in the recording) -- {why}")
    for stage, names in _group_absent(not_found):
        lines.append(f"- **`{stage}` stage** -- registered from the recording "
                     f"but absent from the export for {len(names)} channel(s) "
                     f"({', '.join(names)}). Run that stage, or ignore it if "
                     f"the sequence deliberately exported only the decimated "
                     f"production output.")
    for col in unmatched:
        lines.append(f"- **{col}** (in the export) -- no channel in the "
                     f"recording matches it canonically; a time axis, a "
                     f"derived channel, or a name the channel model does not "
                     f"parse")
    if direction:
        lines += ["", "## Latacc / Latacc_LPF direction", "",
                  "Per the imc recipe `Latacc = smo(Latacc_LPF, 0.5)`: the "
                  "bare column is the output, `_LPF` the intermediate. Both "
                  "directions are tested and the one that actually holds is "
                  "what is reported.", ""]
        lines += [f"- {d}" for d in direction]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass
class AllChannelResult:
    """One all-channel cross-check, ready for a CLI table or a GUI dialog."""
    rows: List[dict] = field(default_factory=list)
    skipped: Dict[str, str] = field(default_factory=dict)
    unmatched: List[str] = field(default_factory=list)
    not_found: List[tuple] = field(default_factory=list)   # (stage, expected)
    direction: List[str] = field(default_factory=list)
    report_path: Optional[Path] = None

    @property
    def n_ok(self) -> int:
        return sum(1 for r in self.rows if r.get("verdict") == "ok")

    @property
    def n_miss(self) -> int:
        return len(self.rows) - self.n_ok

    @property
    def passed(self) -> bool:
        return bool(self.rows) and self.n_miss == 0

    def headline(self) -> str:
        if not self.rows:
            return ("Nothing could be compared — no column in the export "
                    "matched a channel in the recording.")
        return (f"{'PASS' if self.passed else 'BELOW THE FLOOR'} — "
                f"{self.n_ok}/{len(self.rows)} channel-stages at or below the "
                f"export's own rounding (err/q <= 0.5 is the floor)")

    def not_compared_lines(self) -> List[str]:
        out = [f"{name} — {why}" for name, why in sorted(self.skipped.items())]
        out += [f"{stage} stage — not in the export for {len(names)} "
                f"channel(s): {', '.join(names)}"
                for stage, names in _group_absent(self.not_found)]
        out += [f"{col} — in the export, no canonical match in the recording"
                for col in self.unmatched]
        return out


def crosscheck_all_channels(export_paths: Sequence[Path], raw_dir: Path,
                            cut: Optional[Tuple[int, int]] = None,
                            skip: int = 3000, tol: float = 1e-4,
                            report_path: Optional[Path] = None
                            ) -> AllChannelResult:
    """Score every channel in ``raw_dir`` against ``export_paths``.

    The one entry point both the CLI and the GUI call, so the button and the
    command cannot drift apart in what they measure or what they call a pass.

    ``cut`` mirrors the sequence's own ``Cut(a, b)`` and is applied before
    conditioning, so ``FiltLP``'s start-up and ``smo``'s edge padding begin on
    the same sample on both sides. ``skip`` drops that start-up from the score
    at both ends; it is real output, not error, but it swamps the comparison.
    """
    raw_dir = Path(raw_dir)
    paths = [Path(p) for p in export_paths]
    cases, skipped = register_all_channels(raw_dir, cut)
    result = AllChannelResult(skipped=skipped)
    if not cases:
        return result

    columns: List[str] = []
    for p in paths:
        columns += export_columns(p)
    matched, result.unmatched = match_export(cases, columns)
    wanted = set(matched.values())

    got: Dict[str, np.ndarray] = {}
    for p in paths:
        got.update(load_outputs(p, wanted=wanted))

    for case in sorted(cases, key=lambda c: (c.canon, c.stage)):
        col = matched.get(case.key)
        if col is None or col not in got:
            result.not_found.append((case.stage, case.expected))
            continue
        try:
            ours = np.asarray(case.fn(case.data), dtype=float)
        except Exception as exc:                                 # noqa: BLE001
            result.rows.append({"channel": case.recipe_name, "stage": case.stage,
                                "export_column": col,
                                "verdict": f"ERROR {exc}"})
            continue

        edge = max(0, skip // (_DECIMATE if case.decimated else 1))
        st = score(ours, got[col], skip=edge)
        n = min(ours.size, got[col].size)
        a, b = ours[:n], got[col][:n]
        if edge:
            a, b = a[edge:n - edge or None], b[edge:n - edge or None]
        ratio = st["qratio"]
        ok = bool(np.isfinite(st["max"])) and (st["max"] <= tol or ratio <= 0.75)
        result.rows.append({
            "channel": case.recipe_name, "stage": case.stage,
            "export_column": col, "canonical": case.canon,
            "source_file": f"{case.source}.raw", "fs_hz": case.fs,
            "n": st["n"], "max_err": f"{st['max']:.3e}",
            "mean_err": f"{st['mean']:.3e}", "r": f"{st['r']:.9f}",
            "err_q": f"{ratio:.2f}",
            "within_pct": f"{quantum_within_pct(a - b, b):.2f}",
            "verdict": "ok" if ok else "MISS",
            "corrections_not_scored": _corrections(case.recipe_name),
            "note": case.note,
        })

    result.direction = _latacc_direction(paths)
    if report_path is not None:
        meta = {
            "recording": str(raw_dir),
            "export": ", ".join(str(p) for p in paths),
            "cut": f"{cut[0]}:{cut[1]}" if cut else "whole recording",
            "edge samples dropped per end": skip,
            "channels compared": len(result.rows),
            "at or below the export precision": result.n_ok,
        }
        _write_report(Path(report_path), result.rows, result.skipped,
                      result.unmatched, result.not_found, result.direction, meta)
        result.report_path = Path(report_path)
    return result
