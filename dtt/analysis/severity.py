"""Load-severity metrics: G-severity (Gx, Gy, Gxy) and the Dynamic Load Coefficient.

Both answer "how hard did this road work the wheel?" as a single number per
position, which is what makes two routes — or two vehicles on one route —
directly comparable when their raw traces are not.

* **G-severity** normalises the horizontal forces by the vertical load the tyre
  was carrying at that instant, so it is a friction-demand figure rather than a
  force: ``Gx = RMS(Fx/Fz)`` and ``Gy = RMS(Fy/Fz)``. Dividing sample-by-sample
  (not by a static mean) is the point — a wheel unloaded over a crest is not
  generating the same grip it does mid-corner, and only the per-sample ratio
  sees that.

* **Gxy** is the combined horizontal demand, ``sqrt(mean(Fx² + Fy²) / mean(Fz)²)``.
  Note the asymmetry with Gx/Gy: it takes the mean of Fz *first*. That is
  deliberate and matches the reference implementation — it is a whole-run
  severity index scaled by the nominal corner load, not an instantaneous ratio,
  so it stays finite over the wheel-lift samples that would otherwise dominate.

* **DLC** is ``σ(Fz) / mean(Fz)`` — the standard Dynamic Load Coefficient, the
  spread of the vertical load about its static value. 0 is a perfectly smooth
  road; 0.3+ is severe pavement.

Samples with ``Fz == 0`` are dropped from the G-severity ratios rather than
guarded to infinity: a zero vertical load is either a genuine wheel-lift or a
dropout, and in neither case is Fx/Fz a meaningful friction demand.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dtt.config import RunConfig

logger = logging.getLogger(__name__)


def _clean(df: pd.DataFrame, ch: Optional[str]) -> Optional[np.ndarray]:
    if not ch or ch not in df.columns:
        return None
    return pd.to_numeric(df[ch], errors="coerce").to_numpy(dtype=float)


def g_severity(fx: np.ndarray, fy: np.ndarray, fz: np.ndarray
               ) -> Tuple[float, float, float]:
    """Return ``(Gx, Gy, Gxy)`` for one wheel. NaN when nothing is usable."""
    fx = np.asarray(fx, dtype=float)
    fy = np.asarray(fy, dtype=float)
    fz = np.asarray(fz, dtype=float)
    n = min(fx.size, fy.size, fz.size)
    fx, fy, fz = fx[:n], fy[:n], fz[:n]

    ok = np.isfinite(fx) & np.isfinite(fy) & np.isfinite(fz) & (fz != 0)
    fx, fy, fz = fx[ok], fy[ok], fz[ok]
    if fx.size == 0:
        return float("nan"), float("nan"), float("nan")

    gx = float(np.sqrt(np.mean((fx / fz) ** 2)))
    gy = float(np.sqrt(np.mean((fy / fz) ** 2)))

    mean_fz = float(np.mean(fz))
    gxy = (float(np.sqrt(np.mean(fx ** 2 + fy ** 2) / mean_fz ** 2))
           if mean_fz else float("nan"))
    return gx, gy, gxy


def dynamic_load_coefficient(fz: np.ndarray) -> Tuple[float, float, float]:
    """Return ``(static_mean, std, DLC)`` for one wheel's vertical load."""
    fz = np.asarray(fz, dtype=float)
    fz = fz[np.isfinite(fz)]
    if fz.size < 2:
        return float("nan"), float("nan"), float("nan")
    static = float(np.mean(fz))
    sd = float(np.std(fz))
    return static, sd, (sd / static if static else float("nan"))


def compute_severity(df: pd.DataFrame, config: RunConfig) -> Dict[str, dict]:
    """G-severity and DLC per wheel position, saved as JSON beside the stats."""
    rc = getattr(config, "run_channels", None)
    if rc is None:
        return {}

    results: Dict[str, dict] = {}
    for label in rc.labels:
        fx = _clean(df, rc.channel_for(label, "Fx"))
        fy = _clean(df, rc.channel_for(label, "Fy"))
        fz = _clean(df, rc.channel_for(label, "Fz"))
        if fz is None:
            continue
        entry: Dict[str, float] = {}
        if fx is not None and fy is not None:
            gx, gy, gxy = g_severity(fx, fy, fz)
            entry.update(Gx=_r(gx), Gy=_r(gy), Gxy=_r(gxy))
        static, sd, dlc = dynamic_load_coefficient(fz)
        entry.update(static_Fz_daN=_r(static, 2), std_Fz_daN=_r(sd, 2), DLC=_r(dlc))
        results[label] = entry

    if not results:
        return {}

    out_path = config.run_output_dir / "severity_summary.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Severity saved: %s", out_path)
    for label, e in results.items():
        logger.info("  %-5s Gx=%s Gy=%s Gxy=%s | static Fz=%s daN  DLC=%s",
                    label, e.get("Gx"), e.get("Gy"), e.get("Gxy"),
                    e.get("static_Fz_daN"), e.get("DLC"))
    return results


def _r(v: float, nd: int = 4) -> Optional[float]:
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


# ------------------------------------------------------------------ rendering

_COLS_G = ("Gx", "Gy", "Gxy")
_COLS_DLC = ("static_Fz_daN", "std_Fz_daN", "DLC")


def severity_table_rows(results: Dict[str, dict]) -> Tuple[List[str], List[List[str]]]:
    """``(header, rows)`` for display — one row per wheel, blanks where absent."""
    header = ["Wheel", "Gx", "Gy", "Gxy", "Static Fz (daN)", "Std Fz (daN)", "DLC"]
    rows: List[List[str]] = []
    for label, e in results.items():
        rows.append([
            label,
            *[("—" if e.get(c) is None else f"{e[c]:.4f}") for c in _COLS_G],
            "—" if e.get("static_Fz_daN") is None else f"{e['static_Fz_daN']:.1f}",
            "—" if e.get("std_Fz_daN") is None else f"{e['std_Fz_daN']:.1f}",
            "—" if e.get("DLC") is None else f"{e['DLC']:.4f}",
        ])
    return header, rows


def generate_severity_figure(results: Dict[str, dict], config: RunConfig) -> None:
    """Render the severity/DLC table as a figure for the PowerPoint report."""
    if not results:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dtt.config import FIGURE_DPI

    rc = getattr(config, "run_channels", None)
    header, rows = severity_table_rows(results)

    bg, entry, accent, muted = "#0D1B2A", "#0A2540", "#00B4D8", "#90E0EF"
    fig, ax = plt.subplots(figsize=(min(15, 1.9 * len(header) + 1.5),
                                    max(2.4, 0.5 * len(rows) + 1.8)),
                           facecolor=bg)
    ax.axis("off")
    ax.set_title("Load Severity  —  G-severity and Dynamic Load Coefficient\n"
                 "Gx=RMS(Fx/Fz)   Gy=RMS(Fy/Fz)   "
                 "Gxy=√(mean(Fx²+Fy²)/mean(Fz)²)   DLC=σ(Fz)/mean(Fz)",
                 color="white", fontsize=11, fontweight="bold", pad=16)

    tbl = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#1B3A5C")
        if r == 0:
            cell.set_facecolor(entry)
            cell.get_text().set_color(accent)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(entry if r % 2 == 0 else "#0F2A45")
            cell.get_text().set_color("white")
            if c == 0 and rc is not None:
                col = rc.wheel_colors.get(rows[r - 1][0], {}).get("pri", "white")
                cell.get_text().set_color(col)
                cell.get_text().set_fontweight("bold")

    fig.text(0.5, 0.015,
             "Higher Gx/Gy/Gxy = more friction demand on the tyre.  "
             "DLC 0 = perfectly smooth road, 0.3+ = severe pavement.",
             ha="center", va="bottom", fontsize=8, color=muted)
    fig.tight_layout()
    path = config.figures_dir / "severity_table.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    logger.info("Saved severity table: %s", path.name)
