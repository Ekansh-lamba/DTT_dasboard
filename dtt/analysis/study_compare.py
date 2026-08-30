"""Compare two finished, already-processed studies — RMS, DLC, and G-severity
deltas per channel/wheel.

This is a different question from ``dtt/comparison.py`` (which compares two
**raw** imc folders, bypassing the FAMOS/despike/stop-removal pipeline
entirely — see ``gui/pages/comparison_page.py``). Here both sides have
already been through the full pipeline; the "before/after" is between two
completed runs (e.g. an EV vs. an IC variant on the same route), not between
a raw recording and its conditioned version. Reuses ``dtt/comparison.py``'s
``ChannelDelta``-style shape (previous/current/delta/pct) for consistency,
and reuses the platform's existing severity formulas
(``dtt.analysis.severity.g_severity`` / ``dynamic_load_coefficient``) and the
new ``dtt.analysis.statistics.rms`` rather than reimplementing any of them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from dtt.channels import FORCE_COMPONENTS
from dtt.run_channels import build_run_channels, RunChannels
from dtt.analysis.statistics import rms
from dtt.analysis.severity import g_severity, dynamic_load_coefficient
from dtt.provenance import load_provenance, compare_provenance
from dtt.config import PLOT_COLORS, FIGURE_DPI, RAINFLOW_MINER

logger = logging.getLogger(__name__)


@dataclass
class MetricDelta:
    """One (label, metric) pair's before/after value — ``label`` is a
    channel for RMS rows, a wheel for DLC/G-severity rows."""
    label: str
    metric: str
    a: float
    b: float
    delta: float
    pct: float

    @classmethod
    def make(cls, label: str, metric: str, a: float, b: float) -> "MetricDelta":
        a, b = float(a), float(b)
        d = b - a
        pct = (100.0 * d / abs(a)) if np.isfinite(a) and abs(a) > 1e-12 else float("nan")
        return cls(label=label, metric=metric, a=a, b=b, delta=d, pct=pct)


@dataclass
class StudyComparisonResult:
    label_a: str
    label_b: str
    rms: List[MetricDelta] = field(default_factory=list)
    dlc: List[MetricDelta] = field(default_factory=list)
    gseverity: List[MetricDelta] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "label_a": self.label_a, "label_b": self.label_b,
            "rms": [asdict(d) for d in self.rms],
            "dlc": [asdict(d) for d in self.dlc],
            "gseverity": [asdict(d) for d in self.gseverity],
            "warnings": self.warnings,
        }


def load_study(study_dir: Path):
    """Load a finished study's processed frame and re-derive its channel
    configuration, the same way ``pipeline.py`` builds it at run time."""
    study_dir = Path(study_dir)
    csv_path = study_dir / "processed_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No processed_data.csv in {study_dir}")
    df = pd.read_csv(csv_path)
    rc = build_run_channels(list(df.columns))
    return df, rc, study_dir.name


def _clean(df: pd.DataFrame, ch: Optional[str]) -> Optional[np.ndarray]:
    if not ch or ch not in df.columns:
        return None
    return pd.to_numeric(df[ch], errors="coerce").to_numpy(dtype=float)


def _common_labels(rc_a: RunChannels, rc_b: RunChannels) -> List[str]:
    labels = [l for l in rc_a.labels if l in rc_b.labels]
    if not labels:
        logger.warning("No wheel labels in common between the two studies "
                       "(%s vs %s) — comparison will be empty",
                       rc_a.labels, rc_b.labels)
    missing_a = [l for l in rc_b.labels if l not in rc_a.labels]
    missing_b = [l for l in rc_a.labels if l not in rc_b.labels]
    if missing_a or missing_b:
        logger.warning("Wheel labels not shared by both studies — a=%s only, "
                       "b=%s only (skipped)", missing_b, missing_a)
    return labels


def _rms_deltas(df_a, rc_a, df_b, rc_b, labels: List[str]) -> List[MetricDelta]:
    rows: List[MetricDelta] = []
    for label in labels:
        for comp in FORCE_COMPONENTS:
            ch_a = rc_a.channel_for(label, comp)
            ch_b = rc_b.channel_for(label, comp)
            va, vb = _clean(df_a, ch_a), _clean(df_b, ch_b)
            if va is None or vb is None:
                continue
            rows.append(MetricDelta.make(f"{label}_{comp}", "RMS", rms(va), rms(vb)))
    return rows


def _dlc_deltas(df_a, rc_a, df_b, rc_b, labels: List[str]) -> List[MetricDelta]:
    rows: List[MetricDelta] = []
    for label in labels:
        fz_a = _clean(df_a, rc_a.channel_for(label, "Fz"))
        fz_b = _clean(df_b, rc_b.channel_for(label, "Fz"))
        if fz_a is None or fz_b is None:
            continue
        _, _, dlc_a = dynamic_load_coefficient(fz_a)
        _, _, dlc_b = dynamic_load_coefficient(fz_b)
        if np.isfinite(dlc_a) and np.isfinite(dlc_b):
            rows.append(MetricDelta.make(label, "DLC", dlc_a, dlc_b))
    return rows


def _gseverity_deltas(df_a, rc_a, df_b, rc_b, labels: List[str]) -> List[MetricDelta]:
    rows: List[MetricDelta] = []
    for label in labels:
        fx_a = _clean(df_a, rc_a.channel_for(label, "Fx"))
        fy_a = _clean(df_a, rc_a.channel_for(label, "Fy"))
        fz_a = _clean(df_a, rc_a.channel_for(label, "Fz"))
        fx_b = _clean(df_b, rc_b.channel_for(label, "Fx"))
        fy_b = _clean(df_b, rc_b.channel_for(label, "Fy"))
        fz_b = _clean(df_b, rc_b.channel_for(label, "Fz"))
        if any(v is None for v in (fx_a, fy_a, fz_a, fx_b, fy_b, fz_b)):
            continue
        gx_a, gy_a, gxy_a = g_severity(fx_a, fy_a, fz_a)
        gx_b, gy_b, gxy_b = g_severity(fx_b, fy_b, fz_b)
        for metric, va, vb in (("Gx", gx_a, gx_b), ("Gy", gy_a, gy_b), ("Gxy", gxy_a, gxy_b)):
            if np.isfinite(va) and np.isfinite(vb):
                rows.append(MetricDelta.make(label, metric, va, vb))
    return rows


def compare_studies(study_a_dir: Path, study_b_dir: Path) -> StudyComparisonResult:
    """Compare two finished studies' RMS, DLC and G-severity, reusing the
    platform's existing formulas — this function only orchestrates."""
    df_a, rc_a, name_a = load_study(study_a_dir)
    df_b, rc_b, name_b = load_study(study_b_dir)
    labels = _common_labels(rc_a, rc_b)

    result = StudyComparisonResult(
        label_a=name_a, label_b=name_b,
        rms=_rms_deltas(df_a, rc_a, df_b, rc_b, labels),
        dlc=_dlc_deltas(df_a, rc_a, df_b, rc_b, labels),
        gseverity=_gseverity_deltas(df_a, rc_a, df_b, rc_b, labels),
    )

    prov_a = load_provenance(Path(study_a_dir))
    prov_b = load_provenance(Path(study_b_dir))
    result.warnings = compare_provenance(prov_a, prov_b, name_a, name_b)
    for w in result.warnings:
        logger.warning("Study comparison: %s", w)

    return result


def save_comparison(result: StudyComparisonResult, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"study_comparison_{result.label_a}_vs_{result.label_b}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2)
    logger.info("Study comparison saved: %s", path)
    return path


def _rows_for_table(deltas: List[MetricDelta], label_a: str, label_b: str) -> tuple:
    header = ["Label", "Metric", label_a, label_b, "Δ", "Δ%"]
    rows = [[d.label, d.metric, f"{d.a:.4g}", f"{d.b:.4g}", f"{d.delta:+.4g}",
             ("—" if not np.isfinite(d.pct) else f"{d.pct:+.1f}%")]
            for d in deltas]
    return header, rows


def generate_comparison_figure(result: StudyComparisonResult, out_dir: Path) -> Optional[Path]:
    """Render the three delta tables as one figure, in the same table-visual
    style as ``severity.py::generate_severity_figure``, generalized to three
    tables instead of one."""
    if not (result.rms or result.dlc or result.gseverity):
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bg, entry, accent = PLOT_COLORS["bg"], PLOT_COLORS["entry_bg"], PLOT_COLORS["btn_bg"]
    sections = [("RMS  (daN)", result.rms), ("DLC", result.dlc),
                ("G-Severity", result.gseverity)]
    sections = [(t, d) for t, d in sections if d]

    fig, axes = plt.subplots(len(sections), 1,
                             figsize=(9, 1.6 + 2.2 * len(sections)), facecolor=bg)
    axes = np.atleast_1d(axes)
    fig.suptitle(f"Study Comparison — {result.label_a}  vs  {result.label_b}",
                 color="white", fontsize=13, fontweight="bold")
    if result.warnings:
        fig.text(0.5, 0.965, "  |  ".join(result.warnings)[:180],
                 ha="center", va="top", fontsize=7.5, color=PLOT_COLORS["danger"])

    for ax, (title, deltas) in zip(axes, sections):
        ax.axis("off")
        ax.set_title(title, color=accent, fontsize=10, fontweight="bold")
        header, rows = _rows_for_table(deltas, result.label_a, result.label_b)
        tbl = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.scale(1, 1.3)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor(PLOT_COLORS["panel"])
            if r == 0:
                cell.set_facecolor(entry)
                cell.get_text().set_color(accent)
                cell.get_text().set_fontweight("bold")
            else:
                cell.set_facecolor(entry if r % 2 == 0 else "#0F2A45")
                cell.get_text().set_color("white")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"study_comparison_{result.label_a}_vs_{result.label_b}.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    logger.info("Study comparison figure saved: %s", path)
    return path


# --------------------------------------------------------------------------
# A4 — RF Compare: rainflow front/rear axle overlay between two studies.
# Reuses rainflow.py's cycle extraction and Miner-damage primitives unchanged
# -- this is new orchestration (axle grouping across two studies, KDE
# overlay), not new rainflow math.
# --------------------------------------------------------------------------

@dataclass
class RfCompareResult:
    label_a: str
    label_b: str
    axle: str
    component: str
    wheels_a: List[str]
    wheels_b: List[str]
    damage_a: float
    damage_b: float
    damage_ratio: float          # damage_b / damage_a
    n_cycles_a: int
    n_cycles_b: int
    miner_exponent: float

    def to_dict(self) -> Dict:
        return asdict(self)


def _axle_labels(rc: RunChannels, axle: str) -> List[str]:
    """Front = first half of the run's wheel labels (in the order
    ``RunChannels`` reports them), rear = the rest -- generalizes "front
    axle (FL+FR)" beyond a hardcoded 2-axle car to whatever axle count a
    given study actually has."""
    labels = rc.labels
    half = (len(labels) + 1) // 2
    return labels[:half] if axle == "front" else labels[half:]


def _combined_cycles(df: pd.DataFrame, rc: RunChannels, labels: List[str], component: str) -> list:
    from dtt.analysis.rainflow import _extract_cycles
    cycles: list = []
    for label in labels:
        ch = rc.channel_for(label, component)
        if ch and ch in df.columns:
            cycles.extend(_extract_cycles(df, ch))
    return cycles


def compare_rainflow(study_a_dir: Path, study_b_dir: Path,
                     component: str = "Fx", axle: str = "front",
                     miner_exponent: float = RAINFLOW_MINER
                     ) -> tuple:
    """Combine the given axle's wheels' rainflow cycles within each study
    (e.g. FL+FR for "front"), compute Miner damage per study via the
    existing ``rainflow.py::_miner_damage``, and return
    ``(RfCompareResult, cycles_a, cycles_b)`` — cycles are returned alongside
    so a caller can plot the range-distribution overlay without re-extracting.
    """
    from dtt.analysis.rainflow import _miner_damage

    df_a, rc_a, name_a = load_study(study_a_dir)
    df_b, rc_b, name_b = load_study(study_b_dir)
    labels_a = _axle_labels(rc_a, axle)
    labels_b = _axle_labels(rc_b, axle)

    cyc_a = _combined_cycles(df_a, rc_a, labels_a, component)
    cyc_b = _combined_cycles(df_b, rc_b, labels_b, component)

    dmg_a = _miner_damage(cyc_a, miner_exponent)
    dmg_b = _miner_damage(cyc_b, miner_exponent)
    ratio = (dmg_b / dmg_a) if dmg_a else float("nan")

    n_a = int(sum(c[2] for c in cyc_a)) if cyc_a else 0
    n_b = int(sum(c[2] for c in cyc_b)) if cyc_b else 0

    result = RfCompareResult(
        label_a=name_a, label_b=name_b, axle=axle, component=component,
        wheels_a=labels_a, wheels_b=labels_b,
        damage_a=dmg_a, damage_b=dmg_b, damage_ratio=ratio,
        n_cycles_a=n_a, n_cycles_b=n_b, miner_exponent=miner_exponent,
    )
    return result, cyc_a, cyc_b


def _range_repeated(cycles: list) -> np.ndarray:
    if not cycles:
        return np.array([])
    rng = np.array([c[0] for c in cycles])
    cnt = np.array([c[2] for c in cycles])
    return np.repeat(rng, cnt.astype(int).clip(1))


def generate_rf_compare(study_a_dir: Path, study_b_dir: Path, out_dir: Path,
                        component: str = "Fx", axle: str = "front",
                        miner_exponent: float = RAINFLOW_MINER) -> tuple:
    """Compute and plot one RF Compare panel: range-distribution KDE overlay
    (via :func:`dtt.analysis.plot_style.draw_comparison_kde`/
    ``draw_stats_strip``) plus the Miner damage ratio between two studies'
    combined axle cycles. Returns ``(RfCompareResult, png_path)``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dtt.analysis.plot_style import draw_comparison_kde, draw_stats_strip

    result, cyc_a, cyc_b = compare_rainflow(study_a_dir, study_b_dir, component, axle, miner_exponent)
    rng_a, rng_b = _range_repeated(cyc_a), _range_repeated(cyc_b)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"rf_compare_{axle}_{component}_{result.label_a}_vs_{result.label_b}.png"

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=PLOT_COLORS["bg"])
    ax.set_facecolor(PLOT_COLORS["panel"])
    ax.tick_params(colors=PLOT_COLORS["text_sec"], labelsize=8)

    if rng_a.size >= 2 and rng_b.size >= 2:
        color_a, color_b = "#27AE60", "#E74C3C"          # DS1 green, DS2 red -- same
        draw_comparison_kde(ax, rng_a, rng_b, result.label_a, result.label_b, color_a, color_b)
        draw_stats_strip(ax, [(result.label_a, rng_a), (result.label_b, rng_b)],
                         colors=[color_a, color_b])
    else:
        ax.text(0.5, 0.5, "insufficient rainflow cycles", ha="center", va="center",
               transform=ax.transAxes, color="white")

    ax.set_xlabel(f"{component} range (daN)", color=PLOT_COLORS["text_sec"], fontsize=9)
    ax.set_ylabel("Density", color=PLOT_COLORS["text_sec"], fontsize=9)
    ax.set_title(
        f"RF Compare — {axle} axle {component}  ({'+'.join(result.wheels_a)})\n"
        f"Miner damage ratio ({result.label_b}/{result.label_a}, m={miner_exponent:g}): "
        f"{result.damage_ratio:.3f}   ({result.damage_a:.3e} -> {result.damage_b:.3e})",
        color="white", fontsize=10, fontweight="bold")

    fig.tight_layout(rect=[0, 0.18, 1, 1])
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=PLOT_COLORS["bg"])
    plt.close(fig)
    logger.info("RF Compare figure saved: %s", path)
    return result, path
