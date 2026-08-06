"""Compare two WFT recordings — a previous session against a current one.

Answers "what changed between these two runs?" for road-load data, where the
two folders are independent physical recordings rather than two versions of one
table. That distinction drives the whole design:

* **Channels are matched canonically, not by name.** The same wheel is called
  ``WFT_Fx_fr`` in one export and ``FR_Fx_2`` in another; both resolve to
  position A1R, component Fx via :mod:`dtt.channels`. Matching on raw strings
  would report every channel as removed-and-added.
* **Alignment is on elapsed time from the start of each run, not wall clock.**
  Recorder clocks are frequently unset (one of the sample folders is stamped
  2006), and two drives never start at the same instant anyway.
* **"Modified" is statistical, not row-by-row.** Sample *n* of one drive has no
  counterpart in another — the vehicle was somewhere else. Comparing them
  element-wise would produce a number that looks precise and means nothing. So a
  matched channel is compared on its distribution and load statistics, which is
  what actually tells an engineer whether the wheel saw a harsher session.

The result is deliberately explicit about what could not be compared, so an
absent rear axle shows up as a removed channel rather than silently vanishing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from dtt.channels import parse_channel

# A matched channel counts as "changed" once any headline metric moves by more
# than this fraction. 5% is below run-to-run scatter for a repeated route but
# well under a genuine load change.
DEFAULT_CHANGE_TOLERANCE = 0.05

# Metrics compared per channel. RMS and P95 carry the fatigue-relevant content;
# mean and range catch calibration or loading differences.
METRICS = ("mean", "std", "rms", "min", "max", "p95", "range")


def _canonical(name: str) -> Optional[str]:
    """Canonical ``A1R_Fx``-style id for a channel name, or None if not a WFT channel."""
    parsed = parse_channel(name)
    if parsed is None:
        return None
    pos, comp = parsed
    return f"{pos.id}_{comp}"


def _prefer(a: str, b: str) -> str:
    """Pick the better of two source names that map to the same channel.

    imc exports routinely contain ``<name> - Copy`` duplicates; the plain name
    is the real channel.
    """
    for name in (a, b):
        other = b if name is a else a
        if "copy" in other.lower() and "copy" not in name.lower():
            return name
    return a if len(a) <= len(b) else b


def canonical_map(columns: Sequence[str]) -> Dict[str, str]:
    """``{canonical_id: source_column}`` for every WFT channel in ``columns``."""
    out: Dict[str, str] = {}
    for col in columns:
        cid = _canonical(str(col))
        if cid is None:
            continue
        out[cid] = _prefer(out[cid], str(col)) if cid in out else str(col)
    return out


@dataclass
class ChannelDelta:
    """One matched channel's before/after metrics."""
    channel: str                      # canonical id, e.g. A1R_Fz
    label: str                        # friendly label, e.g. FR_Fz
    previous_column: str
    current_column: str
    previous: Dict[str, float]
    current: Dict[str, float]
    delta: Dict[str, float]           # current - previous
    pct: Dict[str, float]             # 100 * delta / |previous|
    changed: bool
    max_pct_metric: str = ""
    max_pct: float = 0.0

    def to_row(self) -> Dict[str, object]:
        row: Dict[str, object] = {
            "Channel": self.label,
            "Canonical": self.channel,
            "Status": "CHANGED" if self.changed else "same",
        }
        for m in METRICS:
            row[f"prev_{m}"] = self.previous.get(m, float("nan"))
            row[f"curr_{m}"] = self.current.get(m, float("nan"))
            row[f"delta_{m}"] = self.delta.get(m, float("nan"))
            row[f"pct_{m}"] = self.pct.get(m, float("nan"))
        return row


@dataclass
class ComparisonResult:
    previous_label: str
    current_label: str
    matched: List[ChannelDelta] = field(default_factory=list)
    only_previous: List[str] = field(default_factory=list)   # removed
    only_current: List[str] = field(default_factory=list)    # added
    previous_rows: int = 0
    current_rows: int = 0
    previous_fs: float = 0.0
    current_fs: float = 0.0
    previous_duration_s: float = 0.0
    current_duration_s: float = 0.0
    tolerance: float = DEFAULT_CHANGE_TOLERANCE
    notes: List[str] = field(default_factory=list)

    @property
    def changed(self) -> List[ChannelDelta]:
        return [d for d in self.matched if d.changed]

    @property
    def unchanged(self) -> List[ChannelDelta]:
        return [d for d in self.matched if not d.changed]

    def table(self) -> pd.DataFrame:
        """Per-channel comparison as a DataFrame (the export/table payload)."""
        if not self.matched:
            return pd.DataFrame()
        return pd.DataFrame([d.to_row() for d in self.matched])

    def summary(self) -> Dict[str, object]:
        return {
            "previous": self.previous_label,
            "current": self.current_label,
            "channels_matched": len(self.matched),
            "channels_changed": len(self.changed),
            "channels_added": len(self.only_current),
            "channels_removed": len(self.only_previous),
            "added": list(self.only_current),
            "removed": list(self.only_previous),
            "previous_rows": self.previous_rows,
            "current_rows": self.current_rows,
            "previous_duration_s": round(self.previous_duration_s, 1),
            "current_duration_s": round(self.current_duration_s, 1),
            "previous_fs_hz": round(self.previous_fs, 3),
            "current_fs_hz": round(self.current_fs, 3),
            "tolerance_pct": 100 * self.tolerance,
            "notes": list(self.notes),
        }

    def summary_lines(self) -> List[str]:
        s = self.summary()
        lines = [
            f"Previous : {s['previous']}  "
            f"({s['previous_rows']:,} rows, {s['previous_duration_s']:.0f}s @ {s['previous_fs_hz']:g} Hz)",
            f"Current  : {s['current']}  "
            f"({s['current_rows']:,} rows, {s['current_duration_s']:.0f}s @ {s['current_fs_hz']:g} Hz)",
            "",
            f"Channels matched : {s['channels_matched']}",
            f"  changed        : {s['channels_changed']}  (> {s['tolerance_pct']:.0f}% on any metric)",
            f"  unchanged      : {s['channels_matched'] - s['channels_changed']}",
            f"Channels added   : {s['channels_added']}"
            + (f"  {', '.join(s['added'])}" if s["added"] else ""),
            f"Channels removed : {s['channels_removed']}"
            + (f"  {', '.join(s['removed'])}" if s["removed"] else ""),
        ]
        if s["notes"]:
            lines += ["", "Notes:"] + [f"  - {n}" for n in s["notes"]]
        return lines


def _metrics(x: np.ndarray) -> Dict[str, float]:
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return {m: float("nan") for m in METRICS} | {"n": 0}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "rms": float(np.sqrt(np.mean(finite ** 2))),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "p95": float(np.percentile(finite, 95)),
        "range": float(np.max(finite) - np.min(finite)),
        "n": int(finite.size),
    }


def _label_for(canonical: str, prev_col: str, curr_col: str) -> str:
    """Prefer a legacy FL/FR/RL/RR label so the table reads the way users speak."""
    parsed = parse_channel(prev_col) or parse_channel(curr_col)
    if parsed:
        pos, comp = parsed
        return f"{pos.legacy_label or pos.id}_{comp}"
    return canonical


def compare_frames(prev: pd.DataFrame, curr: pd.DataFrame,
                   previous_label: str = "previous",
                   current_label: str = "current",
                   fs_prev: float = 0.0, fs_curr: float = 0.0,
                   tolerance: float = DEFAULT_CHANGE_TOLERANCE,
                   ) -> ComparisonResult:
    """Compare two loaded recordings channel by channel."""
    pmap = canonical_map(prev.columns)
    cmap = canonical_map(curr.columns)

    result = ComparisonResult(
        previous_label=previous_label, current_label=current_label,
        previous_rows=len(prev), current_rows=len(curr),
        previous_fs=fs_prev, current_fs=fs_curr,
        previous_duration_s=len(prev) / fs_prev if fs_prev else 0.0,
        current_duration_s=len(curr) / fs_curr if fs_curr else 0.0,
        tolerance=tolerance,
    )

    for cid in sorted(set(pmap) - set(cmap)):
        result.only_previous.append(_label_for(cid, pmap[cid], pmap[cid]))
    for cid in sorted(set(cmap) - set(pmap)):
        result.only_current.append(_label_for(cid, cmap[cid], cmap[cid]))

    for cid in sorted(set(pmap) & set(cmap)):
        pcol, ccol = pmap[cid], cmap[cid]
        a = pd.to_numeric(prev[pcol], errors="coerce").to_numpy(float)
        b = pd.to_numeric(curr[ccol], errors="coerce").to_numpy(float)
        pm, cm = _metrics(a), _metrics(b)

        delta, pct = {}, {}
        for m in METRICS:
            d = cm[m] - pm[m]
            delta[m] = d
            base = abs(pm[m])
            pct[m] = (100.0 * d / base) if base > 1e-12 else float("nan")

        finite_pct = {m: abs(v) for m, v in pct.items() if np.isfinite(v)}
        worst_metric, worst = ("", 0.0)
        if finite_pct:
            worst_metric = max(finite_pct, key=finite_pct.get)
            worst = finite_pct[worst_metric]
        result.matched.append(ChannelDelta(
            channel=cid, label=_label_for(cid, pcol, ccol),
            previous_column=pcol, current_column=ccol,
            previous=pm, current=cm, delta=delta, pct=pct,
            changed=worst > tolerance * 100.0,
            max_pct_metric=worst_metric, max_pct=worst,
        ))

    if result.only_previous:
        result.notes.append(
            f"{len(result.only_previous)} channel(s) present in the previous "
            f"recording only — that instrumentation is missing from the current "
            f"session, so those wheels cannot be compared.")
    if result.only_current:
        result.notes.append(
            f"{len(result.only_current)} channel(s) present in the current "
            f"recording only.")
    if fs_prev and fs_curr and abs(fs_prev - fs_curr) / max(fs_prev, fs_curr) > 0.01:
        result.notes.append(
            f"Sample rates differ ({fs_prev:g} Hz vs {fs_curr:g} Hz); signals are "
            f"resampled onto a common time base for the overlay plots.")
    dur_p, dur_c = result.previous_duration_s, result.current_duration_s
    if dur_p and dur_c and abs(dur_p - dur_c) / max(dur_p, dur_c) > 0.05:
        result.notes.append(
            f"Run lengths differ ({dur_p:.0f}s vs {dur_c:.0f}s). Statistics are "
            f"computed over each full run; the overlay covers the shorter one.")
    _add_scale_note(result)
    result.notes.append(
        "Recordings are independent drives, so channels are compared on their "
        "distributions and load statistics, not sample by sample.")
    return result


# A vertical load ratio beyond this between two sessions is not a session-to-
# session change — it is a different vehicle class or a calibration mismatch.
IMPLAUSIBLE_FZ_RATIO = 3.0


def _add_scale_note(result: ComparisonResult) -> None:
    """Warn when the two runs are not on the same footing at all.

    Vertical load is roughly the corner weight of the vehicle, so it barely
    moves between sessions on the same car. If it moves by more than a factor of
    a few, the two folders are not two sessions of one vehicle, and every
    per-channel percentage below is measuring that mismatch rather than
    anything that happened on the road. Saying so is the difference between a
    report that misleads and one that is merely unhelpful.
    """
    fz = [d for d in result.matched
          if d.channel.endswith("_Fz") and np.isfinite(d.previous.get("mean", np.nan))
          and np.isfinite(d.current.get("mean", np.nan))]
    if not fz:
        return
    prev_mean = float(np.mean([abs(d.previous["mean"]) for d in fz]))
    curr_mean = float(np.mean([abs(d.current["mean"]) for d in fz]))
    if prev_mean <= 1e-9 or curr_mean <= 1e-9:
        return
    ratio = max(prev_mean, curr_mean) / min(prev_mean, curr_mean)
    if ratio < IMPLAUSIBLE_FZ_RATIO:
        return
    result.notes.append(
        f"WARNING: mean vertical load differs by {ratio:.1f}x "
        f"({prev_mean:.0f} vs {curr_mean:.0f} daN per wheel). Corner weight barely "
        f"changes between sessions on one vehicle, so these two folders are most "
        f"likely different vehicles or different WFT calibrations. Treat the "
        f"per-channel percentages below as a scale mismatch, not a load change.")


def align_channel(prev: pd.DataFrame, curr: pd.DataFrame, delta: ChannelDelta,
                  fs_prev: float, fs_curr: float, max_points: int = 4000
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Put one matched channel from both runs on a shared elapsed-time axis.

    Returns ``(t, previous, current)``. Both runs are interpolated onto the same
    grid spanning the *shorter* of the two, because past that point there is
    nothing to compare against.
    """
    a = pd.to_numeric(prev[delta.previous_column], errors="coerce").to_numpy(float)
    b = pd.to_numeric(curr[delta.current_column], errors="coerce").to_numpy(float)
    ta = np.arange(a.size) / (fs_prev or 1.0)
    tb = np.arange(b.size) / (fs_curr or 1.0)
    if a.size == 0 or b.size == 0:
        return np.array([]), np.array([]), np.array([])
    t_end = min(ta[-1], tb[-1])
    n = int(min(max_points, max(2, min(a.size, b.size))))
    t = np.linspace(0.0, t_end, n)

    def _interp(src_t, src_y):
        ok = np.isfinite(src_y)
        if ok.sum() < 2:
            return np.full(t.shape, np.nan)
        return np.interp(t, src_t[ok], src_y[ok])

    return t, _interp(ta, a), _interp(tb, b)


def merge_channel_frame(prev: pd.DataFrame, curr: pd.DataFrame,
                        result: ComparisonResult,
                        fs_prev: float, fs_curr: float,
                        max_points: int = 20000) -> pd.DataFrame:
    """Merged wide frame: elapsed time plus ``<label>_prev`` / ``<label>_curr``.

    This is the "joined" dataset — both sessions on one shared time axis, ready
    to export or plot.
    """
    if not result.matched:
        return pd.DataFrame()
    frames: Dict[str, np.ndarray] = {}
    time_axis: Optional[np.ndarray] = None
    for d in result.matched:
        t, a, b = align_channel(prev, curr, d, fs_prev, fs_curr, max_points)
        if t.size == 0:
            continue
        if time_axis is None:
            time_axis = t
            frames["Time_s"] = t
        frames[f"{d.label}_prev"] = a
        frames[f"{d.label}_curr"] = b
        frames[f"{d.label}_diff"] = b - a
    return pd.DataFrame(frames) if frames else pd.DataFrame()
