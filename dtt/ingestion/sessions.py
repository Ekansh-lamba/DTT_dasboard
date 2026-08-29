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


def _canonical(name: str):
    """``A1R_Fx`` for any spelling of a wheel channel, else ``None``."""
    from dtt.channels import parse_channel
    parsed = parse_channel(name)
    if parsed is None:
        return None
    pos, comp = parsed
    return pos.channel(comp)


def common_channels(frames: Sequence[pd.DataFrame]) -> Tuple[List[str], List[Dict[str, str]]]:
    """``(names, renames)`` -- the channels every session has, and how to get them.

    Wheel channels are matched **canonically**, not by spelling. The same wheel
    arrives as ``FR_Fx_2`` from one export and ``FR_Fx`` from another depending
    on how the recorder was configured, and a literal intersection of those two
    sessions finds no force channels at all -- which is a confusing way to
    discover that two legs of one route were named differently.
    ``dtt.channels.parse_channel`` resolves both to position A1R component Fx,
    which is the identity that actually matters.

    Everything else -- speed, GPS, angles -- is matched literally, having no
    canonical form to appeal to.

    An intersection rather than a union: a channel missing from one leg would
    otherwise be real for 40 km and NaN for the next 50, and every statistic
    computed from it would silently describe only part of the drive.

    ``renames`` maps each session's own column names to the shared names, which
    are the first session's, so the ordinary case of identical naming is a no-op.
    """
    if not frames:
        return [], []

    # canonical id -> that session's column name, per session
    canon = [{c: _canonical(c) for c in f.columns} for f in frames]
    per_session = [{v: k for k, v in m.items() if v} for m in canon]
    shared_canon = set(per_session[0])
    for m in per_session[1:]:
        shared_canon &= set(m)

    plain = [set(c for c in f.columns if canon[i][c] is None)
             for i, f in enumerate(frames)]
    shared_plain = set(plain[0])
    for pl in plain[1:]:
        shared_plain &= pl

    names: List[str] = []
    renames: List[Dict[str, str]] = [{} for _ in frames]
    for c in frames[0].columns:
        cid = canon[0][c]
        if cid is not None and cid in shared_canon:
            names.append(c)
            for i, m in enumerate(per_session):
                renames[i][m[cid]] = c          # each session's spelling -> ours
        elif cid is None and c in shared_plain:
            names.append(c)
            for i in range(len(frames)):
                renames[i][c] = c
    return names, renames



def _static_fz(df: pd.DataFrame) -> float:
    """Median |Fz| across a session's vertical channels, or NaN."""
    from dtt.channels import parse_channel
    mags = []
    for c in df.columns:
        parsed = parse_channel(c)
        if parsed is None or parsed[1] != "Fz":
            continue
        v = pd.to_numeric(df[c], errors="coerce").abs()
        v = v[np.isfinite(v) & (v > 0)]
        if len(v):
            mags.append(float(v.median()))
    return float(np.median(mags)) if mags else float("nan")


def align_session_scales(frames: Sequence[pd.DataFrame],
                         renames: Sequence[Dict[str, str]]) -> List[float]:
    """Put every session's forces on the first session's scale. Returns the factors.

    Sessions of one route can still be recorded through different calibrations:
    in the reference pair, one file carries a ``CR`` factor of 1.37 and stores a
    correct 5,856 N static wheel load, the other a factor of 10 and stores
    79,120 N for the same physical quantity. Joining those and applying a single
    correction afterwards -- which is right when the legs agree -- divides both
    by the same number and leaves one of them a decade wrong, at 60 kg on a
    wheel.

    Static Fz is the anchor again: it is the corner weight, so two legs of one
    route must agree on it. Only clean decades are corrected, because that is
    what a calibration difference looks like; anything else is a different
    vehicle, and quietly rescaling one to match the other would manufacture
    agreement rather than find it.
    """
    ref = _static_fz(frames[0].rename(columns=renames[0]))
    factors = [1.0]
    for f, rn in zip(frames[1:], renames[1:]):
        mag = _static_fz(f.rename(columns=rn))
        if not (np.isfinite(ref) and np.isfinite(mag)) or ref <= 0 or mag <= 0:
            factors.append(1.0)
            continue
        decades = int(round(np.log10(mag / ref)))
        if decades == 0:
            factors.append(1.0)
            continue
        factor = 10.0 ** decades
        # A clean decade is a calibration difference; anything else is not.
        if abs(np.log10(mag / ref) - decades) > 0.25:
            logger.warning(
                "Session static |Fz| is %.0f against %.0f in the first session, "
                "a factor of %.2f that is not a clean decade. These are probably "
                "different vehicles or instrumentation; joining them anyway, "
                "unscaled, but the result will step at the seam.",
                mag, ref, mag / ref)
            factors.append(1.0)
            continue
        logger.warning(
            "Session static |Fz| is %.0f against %.0f in the first session - "
            "a 10^%d calibration difference; scaling its forces to match so the "
            "joined record sits on one scale.", mag, ref, decades)
        factors.append(factor)
    return factors


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

    cols, renames = common_channels(frames)
    if TIME_COLUMN in cols:
        cols = [c for c in cols if c != TIME_COLUMN]
    dropped = sorted(set().union(*(set(f.columns) for f in frames)) - set(cols)
                     - set().union(*(set(r) for r in renames)) - {TIME_COLUMN})

    # Bring every leg onto the first leg's scale before appending.
    from dtt.channels import parse_channel
    scale_factors = align_session_scales(frames, renames)

    pieces: List[pd.DataFrame] = []
    seams: List[float] = []
    durations: List[float] = []
    offsets: Dict[str, float] = {c: 0.0 for c in cols if _is_cumulative(c)}
    n_total = 0

    for f, rename, sf in zip(frames, renames, scale_factors):
        # Each session is relabelled to the shared names before it is appended,
        # so a wheel keeps one column whatever the recorder called it.
        own = {src: dst for src, dst in rename.items() if dst in cols}
        part = f[list(own)].rename(columns=own).reset_index(drop=True)
        part = part[[c for c in cols if c in part.columns]].copy()
        if sf != 1.0:
            # Forces and moments only: the aux channels carry their own units
            # and are unaffected by a force-transducer calibration.
            for c in part.columns:
                if parse_channel(c) is not None:
                    part[c] = pd.to_numeric(part[c], errors="coerce") / sf
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
        "session_scale_factors": scale_factors,
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
    cols, renames = common_channels(present)
    cols = [c for c in cols if c != TIME_COLUMN]
    if not cols:
        return None
    parts = []
    for r, rename in zip(present, renames):
        own = {src: dst for src, dst in rename.items() if dst in cols}
        q = r[list(own)].rename(columns=own).reset_index(drop=True)
        parts.append(q[[c for c in cols if c in q.columns]])
    out = pd.concat(parts, ignore_index=True)
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
