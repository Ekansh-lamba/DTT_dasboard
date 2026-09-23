"""Make a multi-session join visible.

A study built from several recording sessions (``--raw-folders``, "+ Session"
on New Study) is one continuous record: the gap between sessions is dropped
and elapsed time runs on. That is right for the analysis and invisible to the
reader -- nothing in the figures or the report said the 90-minute record was
three drives on two days. This draws the join:

* top: one bar per session, with its source folder and length;
* bottom: a reference trace (speed if there is one, else the first Fz) with
  every seam marked, so a step or a stop at the join can be seen.

``session_info`` is the single reader of the join facts, from the run's
metadata or the study's ``run_provenance.json``, for the pipeline, the GUI and
the report alike.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dtt.config import FIGURE_DPI
from dtt.channel_names import channel_unit, display_name

logger = logging.getLogger(__name__)

SESSIONS_FIGURE = "sessions_timeline.png"
_SESSION_COLORS = ["#2E86DE", "#E67E22", "#27AE60", "#8E44AD", "#C0392B",
                   "#16A085", "#D4AC0D", "#7F8C8D"]


def session_info(metadata: Optional[Dict] = None,
                 provenance: Optional[Dict] = None) -> Optional[Dict]:
    """``{"sessions", "seam_times_s", "session_durations_s", "session_sources"}``
    for a joined study, or None for a single-session one."""
    for src in (metadata, provenance):
        if src and int(src.get("sessions") or 1) > 1:
            n = int(src["sessions"])
            durations = [float(d) for d in (src.get("session_durations_s") or [])]
            sources = [str(s) for s in (src.get("session_sources") or [])]
            return {
                "sessions": n,
                "seam_times_s": [float(t) for t in (src.get("seam_times_s") or [])],
                "session_durations_s": durations,
                "session_sources": sources + [""] * (n - len(sources)),
            }
    return None


def session_spans(info: Dict) -> List[tuple]:
    """``[(start_s, end_s), ...]`` per session on the joined time axis."""
    seams = list(info["seam_times_s"])
    durs = info["session_durations_s"]
    starts = [0.0] + seams
    out = []
    for i in range(info["sessions"]):
        s = starts[i] if i < len(starts) else (out[-1][1] if out else 0.0)
        if i < len(seams):
            e = seams[i]
        elif i < len(durs):
            e = s + durs[i]
        else:
            e = s
        out.append((s, e))
    return out


def _short(source: str) -> str:
    return Path(source).name or source if source else ""


def _reference_trace(df: pd.DataFrame):
    from dtt.config import SPEED_CANDIDATES
    lower = {str(c).lower(): c for c in df.columns}
    for name in SPEED_CANDIDATES + ["GPS.speed"]:
        col = name if name in df.columns else lower.get(name.lower())
        if col is not None:
            v = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
            if np.nanstd(v) > 0:
                return col, v
    for c in df.columns:
        if str(c).endswith("Fz") or "_Fz" in str(c):
            return c, pd.to_numeric(df[c], errors="coerce").to_numpy(float)
    return None, None


def generate_sessions_figure(df: pd.DataFrame, info: Optional[Dict],
                             figures_dir: Path) -> Optional[Path]:
    """Write ``sessions_timeline.png`` for a joined study; None otherwise."""
    if not info:
        return None
    spans = session_spans(info)
    t = (pd.to_numeric(df["Time"], errors="coerce").to_numpy(float)
         if "Time" in df.columns else None)
    col, y = _reference_trace(df)

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(13, 6.2), gridspec_kw={"height_ratios": [1, 2.4]},
        sharex=True, facecolor="white")
    for i, ((s, e), src) in enumerate(zip(spans, info["session_sources"])):
        c = _SESSION_COLORS[i % len(_SESSION_COLORS)]
        ax0.barh(0, e - s, left=s, height=0.6, color=c, alpha=0.85,
                 edgecolor="black", linewidth=0.6)
        label = f"Session {i + 1}\n{_short(src)}\n{(e - s) / 60:.1f} min"
        ax0.text((s + e) / 2, 0, label, ha="center", va="center", fontsize=8,
                 color="white", fontweight="bold")
        ax1.axvspan(s, e, color=c, alpha=0.07, lw=0)
    ax0.set_yticks([])
    ax0.set_ylim(-0.5, 0.5)
    ax0.set_title(f"Joined record: {info['sessions']} recording sessions, "
                  f"{spans[-1][1] / 60:.1f} min in total — gaps between sessions "
                  f"removed, elapsed time continuous", fontsize=11, fontweight="bold")

    if t is not None and y is not None:
        n = min(t.size, y.size)
        step = max(1, n // 20000)
        ax1.plot(t[:n:step], y[:n:step], color="#34495E", linewidth=0.5)
        unit = channel_unit(col)
        ax1.set_ylabel(f"{display_name(col)}" + (f" ({unit})" if unit else ""),
                       fontsize=10)
    for k, seam in enumerate(info["seam_times_s"], start=1):
        for ax in (ax0, ax1):
            ax.axvline(seam, color="#C0392B", linestyle="--", linewidth=1.4)
        ax1.annotate(f"seam {k}\n{seam:.0f} s", xy=(seam, 1.0),
                     xycoords=("data", "axes fraction"), xytext=(3, -4),
                     textcoords="offset points", va="top", fontsize=8,
                     color="#C0392B", fontweight="bold")
    ax1.set_xlabel("Elapsed time in the joined record (s)", fontsize=10)
    ax1.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    path = Path(figures_dir) / SESSIONS_FIGURE
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved session join figure: %s (%d sessions)", path.name,
                info["sessions"])
    return path
