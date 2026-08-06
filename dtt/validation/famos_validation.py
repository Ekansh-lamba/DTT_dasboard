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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

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
