"""Join several imc recording sessions into one continuous record.

A durability route is rarely captured in one sitting: 1-50 km one day, 50-100 km
the next, each landing in its own ``.raw`` folder with its own time base
starting at zero. This module stitches those back into the single drive they
represent, so statistics, rainflow counts and the preprocess graph see the whole
route rather than one leg of it.

Four decisions here have a plausible-looking wrong answer, so they are spelled
out rather than left to the reader:

**Condition first, then join** -- not the other way round. Each session is
filtered inside :func:`~dtt.ingestion.imc_reader.read_folder` before it gets
here. ``FiltLP`` is causal, so joining raw signals and filtering afterwards
would let the tail of one session ring into the head of the next across a seam
where the vehicle may have been switched off for a day. Filtering per session
keeps every transient inside the recording that produced it.

**Scale once, after joining** -- the N-to-daN conversion and the decade check
are magnitude heuristics run against the whole record. Applied per session they
can reach different verdicts on two legs of the same drive, and the joined trace
then steps by a factor of ten at the seam. Callers must hand raw-scaled frames
in and scale the result.

**Elapsed time is continuous** -- the gap between sessions is discarded, so leg
two begins the sample after leg one ends. The alternative preserves the
overnight wall-clock gap and leaves hours of emptiness in the middle of the
record, which is not a thing the vehicle did. Road load cares about exposure,
not calendar time.

**Order by when it was recorded**, from the imc timestamps, not by the order the
folders were picked or their names -- "Run 10" sorts before "Run 9". Sessions
without a timestamp keep their given order, after the timestamped ones.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TIME_COLUMN = "Time"
# Channels that count distance rather than measure it: they restart at zero
# every session and have to be made cumulative instead of concatenated as-is.
_CUMULATIVE = ("dist", "distance", "odo", "odometer")


def _is_cumulative(name: str) -> bool:
    return name.strip().lower() in _CUMULATIVE


def order_sessions(metas: Sequence[dict]) -> List[int]:
    """Indices of ``metas`` in recording order.

    Sorts on the imc start timestamp where there is one. Sessions without a
    usable timestamp keep the order they were given and follow the dated ones,
    because a guess at their position is worse than the operator's own choice.
    """
    dated, undated = [], []
    for i, m in enumerate(metas):
        epoch = m.get("start_epoch")
        if epoch is not None and np.isfinite(epoch) and epoch > 0:
            dated.append((float(epoch), i))
        else:
            undated.append(i)
    return [i for _, i in sorted(dated)] + undated


def common_channels(frames: Sequence[pd.DataFrame]) -> List[str]:
    """Channels present in every session, in the first session's order.

    An intersection rather than a union: a channel missing from one leg would
    otherwise arrive as a column that is real for 40 km and NaN for the next 50,
    and every statistic computed from it would silently describe only part of
    the drive.
    """
    if not frames:
        return []
    shared = set(frames[0].columns)
    for f in frames[1:]:
        shared &= set(f.columns)
    return [c for c in frames[0].columns if c in shared]


def concat_sessions(frames: Sequence[pd.DataFrame], fs: float,
                    metas: Optional[Sequence[dict]] = None,
                    raw_frames: Optional[Sequence[Optional[pd.DataFrame]]] = None,
                    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], dict]:
    """Join conditioned sessions end to end. Returns ``(df, raw_df, report)``.

    ``raw_frames`` are the matching unconditioned copies, joined the same way so
    the before/after view still lines up after stitching.
    """
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame(), None, {}
    if len(frames) == 1:
        raw = raw_frames[0] if raw_frames else None
        return frames[0].reset_index(drop=True), raw, {
            "sessions": 1, "seam_times_s": [], "session_durations_s":
                [round(len(frames[0]) / fs, 2)] if fs else []}

    cols = common_channels(frames)
    if TIME_COLUMN in cols:
        cols = [c for c in cols if c != TIME_COLUMN]
    dropped = sorted(set().union(*(set(f.columns) for f in frames)) - set(cols)
                     - {TIME_COLUMN})

    pieces: List[pd.DataFrame] = []
    seams: List[float] = []
    durations: List[float] = []
    offsets: Dict[str, float] = {c: 0.0 for c in cols if _is_cumulative(c)}
    n_total = 0

    for f in frames:
        part = f[cols].reset_index(drop=True).copy()
        for c in list(offsets):
            v = pd.to_numeric(part[c], errors="coerce")
            # Distance restarts at zero each session; carry the running total so
            # the joined channel keeps counting the route instead of resetting.
            part[c] = v + offsets[c]
            last = v.dropna()
            offsets[c] += float(last.iloc[-1]) if len(last) else 0.0
        durations.append(round(len(part) / fs, 2) if fs else 0.0)
        if pieces:
            seams.append(round(n_total / fs, 3) if fs else float(n_total))
        n_total += len(part)
        pieces.append(part)

    out = pd.concat(pieces, ignore_index=True)
    # One clock for the joined record, at the original spacing.
    out.insert(0, TIME_COLUMN, np.arange(len(out)) / fs if fs else
               np.arange(len(out), dtype=float))

    raw_out = None
    if raw_frames and any(r is not None and len(r) for r in raw_frames):
        raw_out = _concat_raw(raw_frames, fs)

    report = {
        "sessions": len(frames),
        "seam_times_s": seams,
        "session_durations_s": durations,
        "total_duration_s": round(n_total / fs, 2) if fs else 0.0,
        "channels_joined": len(cols),
        "channels_dropped": dropped,
    }
    if dropped:
        logger.warning("Joined on %d channels common to all %d sessions; "
                       "dropped %d present in only some: %s",
                       len(cols), len(frames), len(dropped), ", ".join(dropped))
    logger.info("Joined %d sessions -> %.1f s total, seams at %s s",
                len(frames), report["total_duration_s"],
                ", ".join(f"{s:.0f}" for s in seams) or "-")
    return out, raw_out, report


def _concat_raw(raw_frames, fs: float) -> Optional[pd.DataFrame]:
    """Join the unconditioned copies on the channels they all share."""
    present = [r for r in raw_frames if r is not None and len(r)]
    if not present:
        return None
    cols = common_channels(present)
    cols = [c for c in cols if c != TIME_COLUMN]
    if not cols:
        return None
    out = pd.concat([r[cols].reset_index(drop=True) for r in present],
                    ignore_index=True)
    out.insert(0, TIME_COLUMN, np.arange(len(out)) / fs if fs else
               np.arange(len(out), dtype=float))
    return out


def merge_metadata(metas: Sequence[dict], report: dict) -> dict:
    """One metadata dict for the joined record.

    Values that must agree across sessions (sample rate, decimation) are taken
    from the first and checked against the rest, because a mismatch means the
    sessions were not recorded the same way and joining them would produce a
    record whose time base is wrong in the middle.
    """
    if not metas:
        return {}
    base = dict(metas[0])
    for key in ("raw_fs_hz", "output_fs_hz", "famos_decimate"):
        values = {m.get(key) for m in metas if m.get(key) is not None}
        if len(values) > 1:
            logger.warning("Sessions disagree on %s: %s - using %s from the first",
                           key, sorted(values), base.get(key))
    base["sessions"] = report.get("sessions", len(metas))
    base["seam_times_s"] = report.get("seam_times_s", [])
    base["session_durations_s"] = report.get("session_durations_s", [])
    base["session_sources"] = [m.get("source", "") for m in metas]
    base["rows"] = report.get("rows", base.get("rows"))
    base["duration_s"] = report.get("total_duration_s", base.get("duration_s"))
    base.pop("raw_frame", None)
    return base
