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


# --------------------------------------------------------------------------
# Two-dataset AUC comparison -- the reference analyzer's look, on the
# platform's channel model, units and KDE.
#
# Ported from `WFT_Analyzer_All_AUC_Gxy_Heatmap_boxplot_PV_July26 (1).py`,
# `_auc_generate` (~line 1287). Three things are deliberately NOT copied:
#
#   - its `KDE_MAX_SAMPLES = 20000` subsampling. Scott's bandwidth scales as
#     n^(-1/5), so fitting on 20k points instead of 600k widens the kernel by
#     ~1.7x and smears genuinely separate peaks into one. `auc.kde_curve` is
#     already O(n) via binned convolution and agrees with the exact full-data
#     KDE to ~1e-6, so the subsampling buys nothing and costs correctness.
#   - its legend, which sets `labelcolor="white"` on a `#F5F5F5` panel and is
#     therefore invisible. The light-theme colours from `histograms.py` are
#     used instead -- the same ones the Histogram and single-study AUC
#     sections already agreed on.
#   - its fixed twelve FL/FR/RL/RR channel list and its N-to-daN heuristic.
#     Channels come from `RunChannels`, whatever the study recorded, forces
#     and moments alike.
#
# Which dataset is the reference is the platform's convention, not the
# reference script's: `compare_distributions(a, b)` treats **a** as the
# baseline, so dataset 1 is the reference here, where the script made dataset
# 2 the reference. Same arithmetic, opposite argument order; mixing the two
# would silently change every exceedance figure.
# --------------------------------------------------------------------------

AUC_DS1_COLOR = "#27AE60"      # green -- dataset 1, solid, the reference
AUC_DS2_COLOR = "#E74C3C"      # red   -- dataset 2, dashed, judged against it

# Exactly two colours. Each dataset's KDE line, fill, histogram bars and both
# percentile rules are drawn in its own one colour, and P5 vs P95 is told apart
# by line style, never by a third colour.
_P5_STYLE = ":"
_P95_STYLE = "--"

_HIST_BINS_OVERLAY = 55        # as the reference draws it


# ---------------------------------------------------------------- channels

def _channel_index(rc: RunChannels) -> Dict[tuple, str]:
    """``{(wheel label, component): column}`` for **every** component.

    Not ``rc.channel_for``: ``RunChannels`` is built for the severity analysis
    and keeps Fx/Fy/Fz only (``build_run_channels`` filters on
    FORCE_COMPONENTS), so asking it for ``Mx`` returns None. Relying on it is
    how the comparison screen quietly offered six force channels and no
    moments at all. The channel set underneath has every component; the wheel
    label is taken from the same position the forces already resolved to, so
    moments line up under the same "FR" / "A2RO" names.
    """
    pos_label: Dict[object, str] = {}
    for label, cols in rc.wheel_groups.items():
        for col in cols:
            parsed = rc.channel_set.source_map.get(col)
            if parsed is not None:
                pos_label[parsed[0]] = label
    out: Dict[tuple, str] = {}
    for col, (pos, comp) in rc.channel_set.source_map.items():
        label = pos_label.get(pos, pos.id)
        key = (label, comp)
        # imc exports carry "<name> - Copy" duplicates; the plain name wins.
        if key in out and "copy" in col.lower():
            continue
        out[key] = col
    return out


def _positions(rc: RunChannels) -> Dict[str, object]:
    """``{wheel label: Position}``, for pairing wheels by axle and side."""
    out: Dict[str, object] = {}
    for label, cols in rc.wheel_groups.items():
        for col in cols:
            parsed = rc.channel_set.source_map.get(col)
            if parsed is not None:
                out[label] = parsed[0]
                break
    return out


def _component_order(comp: str) -> int:
    from dtt.channels import COMPONENTS
    return COMPONENTS.index(comp) if comp in COMPONENTS else 99


def auc_channels(rc_a: RunChannels, rc_b: RunChannels,
                 df_a: pd.DataFrame, df_b: pd.DataFrame) -> List[tuple]:
    """``[(display, column in A, column in B)]`` — the same wheel and
    component in both studies, forces **and** moments, position order.

    No fixed twelve-channel list: a two-WFT recording has six force channels,
    a three-axle truck has more, and the reference script's hardcoded
    FL/FR/RL/RR would offer entries that can never load.
    """
    ia, ib = _channel_index(rc_a), _channel_index(rc_b)
    out: List[tuple] = []
    for label in _common_labels(rc_a, rc_b):
        comps = sorted({c for (l, c) in ia if l == label}
                       & {c for (l, c) in ib if l == label}, key=_component_order)
        for comp in comps:
            ca, cb = ia[(label, comp)], ib[(label, comp)]
            if ca in df_a.columns and cb in df_b.columns:
                out.append((f"{label}_{comp}", ca, cb))
    return out


def auc_cross_channels(rc_a: RunChannels, rc_b: RunChannels,
                       df_a: pd.DataFrame, df_b: pd.DataFrame) -> List[tuple]:
    """Cross-position pairs: a front wheel in one run against the rear wheel
    on the same side in the other, **both ways round**.

    The reference script's `_run_auc_cross`: ``FL(DS1) vs RL(DS2)`` and
    ``RL(DS1) vs FL(DS2)``, forces and moments. It answers a different
    question from the same-channel view -- whether a load one axle sees in
    one vehicle is what the other axle sees in the other, the case when a
    powertrain change moves weight fore and aft.

    Pairs are formed from positions, not names: for each side (and inner /
    outer, on a dual), the front-most axle is paired with the rear-most. A
    single-axle side has no partner and is skipped rather than guessed.
    """
    ia, ib = _channel_index(rc_a), _channel_index(rc_b)
    pa, pb = _positions(rc_a), _positions(rc_b)

    def ends(positions: Dict[str, object]) -> Dict[tuple, tuple]:
        by_side: Dict[tuple, List[tuple]] = {}
        for label, pos in positions.items():
            by_side.setdefault((pos.side, pos.sub), []).append((pos.axle, label))
        out = {}
        for side, items in by_side.items():
            items.sort()
            if len(items) >= 2 and items[0][0] != items[-1][0]:
                out[side] = (items[0][1], items[-1][1])      # (front, rear)
        return out

    ea, eb = ends(pa), ends(pb)
    out: List[tuple] = []
    for side in sorted(set(ea) & set(eb), key=lambda s: (s[0], s[1] or "")):
        (front_a, rear_a), (front_b, rear_b) = ea[side], eb[side]
        for la, lb in ((front_a, rear_b), (rear_a, front_b)):
            comps = sorted({c for (l, c) in ia if l == la}
                           & {c for (l, c) in ib if l == lb}, key=_component_order)
            for comp in comps:
                ca, cb = ia[(la, comp)], ib[(lb, comp)]
                if ca in df_a.columns and cb in df_b.columns:
                    out.append((f"{la}_{comp} vs {lb}_{comp}", ca, cb))
    return out


def channel_unit(display: str) -> str:
    """daN for a force, daN·m for a moment — the units the pipeline writes.

    Read off the component in the channel's own name, never inferred from the
    magnitude: guessing units from the numbers is what produced a decade error
    once already.
    """
    comp = display.split(" vs ")[0].rsplit("_", 1)[-1]
    return "daN·m" if comp.startswith("M") else "daN"


# --------------------------------------------------------------- weighting

def study_duration_s(df: pd.DataFrame) -> Optional[float]:
    """Recording length from the frame's own Time column.

    Not row count over a sample rate: a stop-removed or sanitised study has
    fewer rows than its clock says, and the clock is the honest answer for
    "how much driving is this percentile built from".
    """
    if df is None or "Time" not in df.columns:
        return None
    t = pd.to_numeric(df["Time"], errors="coerce").to_numpy(float)
    t = t[np.isfinite(t)]
    return float(t[-1] - t[0]) if t.size > 1 else None


def distance_weights(df: pd.DataFrame) -> tuple:
    """``(weights, total_m)`` for a study, or ``(None, None)``.

    Reuses ``histograms._get_speed_weights`` -- the same speed-channel search,
    dead-channel rejection and unit detection the Histogram and single-study
    AUC sections already rely on -- so a comparison is distance-weighted by
    exactly the rule the rest of the app uses. The sample rate comes from the
    Time column, not from a config default.
    """
    from dtt.analysis.histograms import _get_speed_weights

    if df is None or "Time" not in df.columns:
        return None, None
    t = pd.to_numeric(df["Time"], errors="coerce").to_numpy(float)
    dt = np.nanmedian(np.diff(t[: min(t.size, 5000)])) if t.size > 2 else np.nan
    if not np.isfinite(dt) or dt <= 0:
        return None, None
    weights, total_m, _unit = _get_speed_weights(df, 1.0 / dt)
    if weights is None:
        return None, None
    return np.nan_to_num(np.asarray(weights, dtype=float), nan=0.0), total_m


# ------------------------------------------------------------------- stats

def auc_comparison_stats(values_a: np.ndarray, values_b: np.ndarray,
                         label_a: str, label_b: str, channel: str,
                         unit: str = "daN",
                         weights_a: Optional[np.ndarray] = None,
                         weights_b: Optional[np.ndarray] = None,
                         ) -> Optional[Dict]:
    """The footer/export row for one channel, with the labels in the keys.

    The reference builds ``f"P95_{lbl2}"``-style keys so a saved summary says
    which run each number belongs to; that is kept, driven by the editable
    labels. The numbers themselves all come from
    :func:`dtt.analysis.auc.compare_distributions` -- this only names them.
    """
    from dtt.analysis.auc import compare_distributions

    cmp = compare_distributions(values_a, values_b, weights_a, weights_b)
    if cmp is None:
        return None
    return {
        "Channel": channel, "Unit": unit,
        "Weighting": "distance" if weights_a is not None else "sample count",
        "Label1": label_a, "Label2": label_b,
        f"P5_{label_a}": round(cmp.p5_ref, 2),
        f"P95_{label_a}": round(cmp.p95_ref, 2),
        f"P5_{label_b}": round(cmp.p5_cur, 2),
        f"P95_{label_b}": round(cmp.p95_cur, 2),
        f"Delta_P95_{unit}": round(cmp.delta_p95, 2),
        f"Pct_exceed_{label_a}_P95": round(cmp.pct_exceed_ref_p95, 2),
        f"Pct_normal_{label_b}": round(cmp.pct_normal_cur, 2),
        f"Pct_normal_{label_a}": round(cmp.pct_normal_ref, 2),
        f"n_{label_a}": cmp.n_ref,
        f"n_{label_b}": cmp.n_cur,
    }


def auc_footer_text(cmp, label_a: str, label_b: str, unit: str = "daN") -> str:
    """The stats line under the two panels, in the reference's wording.

    Dataset 1 is the reference, so it is the one whose P5/P95 define the
    normal zone and whose P95 the exceedance is counted against. Both runs'
    P5 appear, not only the reference's: the reference script quotes one and
    the other is then only readable off the legend.
    """
    return (
        f"{label_a} (reference)  →  P5: {cmp.p5_ref:.0f} {unit}   "
        f"P95: {cmp.p95_ref:.0f} {unit}   "
        f"Normal zone covers {cmp.pct_normal_ref:.1f}% of data"
        f"        "
        f"{label_b}  →  P5: {cmp.p5_cur:.0f} {unit}   "
        f"P95: {cmp.p95_cur:.0f} {unit} "
        f"({cmp.delta_p95:+.0f})   "
        f"{cmp.pct_exceed_ref_p95:.1f}% of cycles exceed {label_a} P95   "
        f"{cmp.pct_normal_cur:.1f}% within {label_a} normal zone")


def auc_coverage_text(cmp, label_a: str, label_b: str,
                      duration_a: Optional[float] = None,
                      duration_b: Optional[float] = None,
                      distance_a_m: Optional[float] = None,
                      distance_b_m: Optional[float] = None,
                      weighting: str = "sample count") -> str:
    """How much data each percentile was computed from, and how it was counted.

    ``AucComparison`` carried the two sample counts all along and nothing
    displayed them. Two runs of very different length are an ordinary thing
    to compare, and every figure above is a percentile, which says nothing
    about how much was measured: a P95 from 8 minutes and one from 90 are not
    equally load bearing.
    """
    def _one(label, n, dur, dist):
        bits = [f"{n:,} samples"]
        if dur and dur > 0:
            bits.append(f"{dur:,.0f} s")
        if dist and dist > 0:
            bits.append(f"{dist / 1000.0:,.1f} km")
        return f"{label}: " + " · ".join(bits)

    text = (f"{_one(label_a, cmp.n_ref, duration_a, distance_a_m)}   |   "
            f"{_one(label_b, cmp.n_cur, duration_b, distance_b_m)}   |   "
            f"weighted by {weighting}")

    # Flag a real mismatch rather than leave the reader dividing two
    # six-figure numbers in their head.
    basis = ((distance_a_m, distance_b_m, "farther")
             if distance_a_m and distance_b_m else
             (duration_a, duration_b, "longer"))
    x, y, word = basis
    if x and y and min(x, y) > 0:
        ratio = max(x, y) / min(x, y)
        if ratio >= 1.25:
            which = label_a if x > y else label_b
            text += f"   —  {which} is {ratio:.1f}x {word}"
    return text


def _legend_labels(label_a: str, label_b: str, cmp) -> List[str]:
    """Legend text in reading order: the reference dataset, then the other."""
    return [label_a, f"{label_a} P5: {cmp.p5_ref:.0f}",
            f"{label_a} P95: {cmp.p95_ref:.0f}",
            label_b, f"{label_b} P5: {cmp.p5_cur:.0f}",
            f"{label_b} P95: {cmp.p95_cur:.0f}"]


def _paired(values, weights):
    """Finite values and their weights, masked together."""
    v = np.asarray(values, dtype=float)
    if weights is None:
        return v[np.isfinite(v)], None
    w = np.asarray(weights, dtype=float)
    n = min(v.size, w.size)
    v, w = v[:n], w[:n]
    ok = np.isfinite(v) & np.isfinite(w) & (w >= 0)
    return v[ok], w[ok]


# --------------------------------------------------------------- the pair

def draw_auc_comparison(ax_kde, ax_hist, values_a: np.ndarray,
                        values_b: np.ndarray, label_a: str, label_b: str,
                        channel: str, unit: str = "daN",
                        xlim: Optional[tuple] = None,
                        range_mode: str = "autoscale",
                        ax_stats=None,
                        weights_a: Optional[np.ndarray] = None,
                        weights_b: Optional[np.ndarray] = None):
    """Draw the two-panel AUC comparison onto a caller's axes.

    Takes axes rather than building a figure, so the embedded GUI panel and a
    headless PNG export are the same drawing rather than two that have to be
    kept in step.

    ``weights_a`` / ``weights_b`` (per-sample distance, from
    :func:`distance_weights`) distance-weight **everything** -- both curves,
    both histograms, all four percentile rules and the stats strip -- so no
    element of the plot describes a different distribution from the others.

    Returns the :class:`~dtt.analysis.auc.AucComparison`, or ``None`` when
    either dataset is too small to have a distribution.
    """
    from dtt.analysis.auc import auc_grid, compare_distributions
    from dtt.analysis.plot_style import draw_stats_strip
    from dtt.analysis.histograms import (
        BG, LABEL_FONTSIZE, TEXT_PRI, TITLE_FONTSIZE, _axis_range,
        _get_force_type, _style_ax)

    a, wa = _paired(values_a, weights_a)
    b, wb = _paired(values_b, weights_b)
    for ax in (ax_kde, ax_hist):
        _style_ax(ax)
    if ax_stats is not None:
        ax_stats.axis("off")
    if a.size < 2 or b.size < 2:
        ax_kde.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                    transform=ax_kde.transAxes, color=TEXT_PRI)
        return None

    if xlim is None:
        # The Histogram and single-study AUC sections' own range logic, on the
        # two datasets pooled, so both panels and both datasets share one
        # x-range and agree with the rest of the app. "autoscale" (P0.5-P99.5)
        # is the default because a two-run comparison is about where the mass
        # sits. "full" is the configured sensor range while >= 98% of samples
        # fit, else the data's own range -- a fixed axis for laying studies
        # side by side. Neither removes a rail holding more than ~2% of the
        # samples: on the reference studies FR_Fx is frozen at -451 daN for
        # 1.6-2.6% of the run, and both modes keep it on screen.
        both = np.concatenate([a, b])
        rng = _axis_range(both, _get_force_type(channel), channel,
                          range_mode=range_mode)
        if rng and rng[1] > rng[0]:
            lo, hi = rng
        else:
            lo, hi = float(both.min()), float(both.max()) or 1.0
    else:
        lo, hi = xlim

    cmp = compare_distributions(a, b, wa, wb)
    x, kde_a, kde_b = auc_grid(a, b, xlim=(lo, hi), weights_ref=wa, weights_cur=wb)

    def _rules(ax):
        """P5 and P95 for both datasets, each in its own single colour.

        Returns the handles in reading order -- reference first -- because
        matplotlib orders a legend by draw order, and the datasets have to be
        drawn with the reference last so it sits on top.
        """
        out = {}
        for key, p5, p95, colour, label in (
                ("a", cmp.p5_ref, cmp.p95_ref, AUC_DS1_COLOR, label_a),
                ("b", cmp.p5_cur, cmp.p95_cur, AUC_DS2_COLOR, label_b)):
            out[key] = [
                ax.axvline(p5, color=colour, linewidth=1.4,
                           linestyle=_P5_STYLE, label=f"{label} P5: {p5:.0f}"),
                ax.axvline(p95, color=colour, linewidth=1.8,
                           linestyle=_P95_STYLE,
                           label=f"{label} P95: {p95:.0f}"),
            ]
        return out

    def _legend(ax, handles, labels):
        # Explicit handles in reading order, reference first. The reference
        # script's own legend is unreadable -- labelcolor="white" on a #F5F5F5
        # panel -- so these are the light-theme colours the Histogram and
        # single-study AUC sections already use.
        ax.legend(handles=handles, labels=labels,
                  fontsize=8, facecolor=BG, edgecolor="#CCCCCC",
                  labelcolor=TEXT_PRI, framealpha=0.9, loc="upper right")

    # Left: shaded KDE curves. Dataset 2 drawn first so the reference sits on
    # top of it.
    ax_kde.fill_between(x, kde_a, alpha=0.25, color=AUC_DS1_COLOR)
    ax_kde.fill_between(x, kde_b, alpha=0.25, color=AUC_DS2_COLOR)
    line_b, = ax_kde.plot(x, kde_b, color=AUC_DS2_COLOR, linewidth=2,
                          linestyle="--", label=label_b)
    line_a, = ax_kde.plot(x, kde_a, color=AUC_DS1_COLOR, linewidth=2,
                          linestyle="-", label=label_a)
    rules = _rules(ax_kde)
    ax_kde.set_title("Shaded Area — KDE Curves", fontsize=TITLE_FONTSIZE,
                     fontweight="bold")
    _legend(ax_kde, [line_a, *rules["a"], line_b, *rules["b"]],
            _legend_labels(label_a, label_b, cmp))

    # Right: the same data binned, same x-range, same two colours, and the
    # same weights -- a count histogram under a distance-weighted curve would
    # be two different distributions on one axis.
    bins = np.linspace(lo, hi, _HIST_BINS_OVERLAY)
    _, _, bars_b = ax_hist.hist(b, bins=bins, weights=wb, density=True,
                                alpha=0.45, color=AUC_DS2_COLOR,
                                edgecolor="none", label=label_b)
    _, _, bars_a = ax_hist.hist(a, bins=bins, weights=wa, density=True,
                                alpha=0.45, color=AUC_DS1_COLOR,
                                edgecolor="none", label=label_a)
    rules = _rules(ax_hist)
    ax_hist.set_title("Density Histogram Overlay", fontsize=TITLE_FONTSIZE,
                      fontweight="bold")
    # A histogram's handle is a BarContainer, whose own label is an internal
    # "_containerN" -- so labels are passed explicitly.
    _legend(ax_hist, [bars_a, *rules["a"], bars_b, *rules["b"]],
            _legend_labels(label_a, label_b, cmp))

    ylabel = ("Distance-weighted density" if wa is not None
              else "Normalised density")
    for ax in (ax_kde, ax_hist):
        ax.set_xlim(lo, hi)
        ax.set_xlabel(f"{channel} ({unit})" if unit else channel,
                      fontsize=LABEL_FONTSIZE, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE, fontweight="bold")

    if ax_stats is not None:
        # Min/P5/Q1/Median/Mean/Q3/P95/Max per run -- the strip the
        # single-study AUC and RF Compare already carry, reused. It has its
        # own axes spanning both panels: a table hung under one axes with a
        # negative bbox is sized in *axes* fractions, collides with the
        # x-label whenever the panels are short, and drifts on resize.
        draw_stats_strip(ax_stats, [(label_a, a), (label_b, b)],
                         colors=[AUC_DS1_COLOR, AUC_DS2_COLOR],
                         theme="light", fontsize=8, bbox=[0.0, 0.0, 1.0, 1.0],
                         weights=[wa, wb] if wa is not None else None)
    return cmp


def auc_figure_layout(fig):
    """``(ax_kde, ax_hist, ax_stats)`` on ``fig``: two panels over a strip row.

    One layout for both the embedded canvas and the PNG export, so the two
    cannot drift apart. ``height_ratios`` keeps the strip a caption rather
    than a third panel.
    """
    gs = fig.add_gridspec(2, 2, height_ratios=[4.2, 1.0], hspace=0.45,
                          wspace=0.16, left=0.06, right=0.98,
                          top=0.87, bottom=0.05)
    return (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, :]))


def generate_auc_comparison(values_a: np.ndarray, values_b: np.ndarray,
                            label_a: str, label_b: str, channel: str,
                            out_path: Path, unit: str = "daN",
                            range_mode: str = "autoscale",
                            weights_a: Optional[np.ndarray] = None,
                            weights_b: Optional[np.ndarray] = None,
                            footer_extra: str = "") -> Optional[Path]:
    """The same two panels and strip as a PNG, for a report or a batch export."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dtt.analysis.histograms import BG, SUPTITLE_FONTSIZE, TEXT_PRI, TEXT_SEC

    fig = plt.figure(figsize=(14, 7.0), facecolor=BG)
    ax_kde, ax_hist, ax_stats = auc_figure_layout(fig)
    # Leave room under the strip for the two footer lines.
    fig.subplots_adjust(bottom=0.13)
    cmp = draw_auc_comparison(ax_kde, ax_hist, values_a, values_b,
                              label_a, label_b, channel, unit,
                              range_mode=range_mode, ax_stats=ax_stats,
                              weights_a=weights_a, weights_b=weights_b)
    fig.suptitle(f"AUC — {channel}  ({label_a}  vs  {label_b}  reference)",
                 color=TEXT_PRI, fontsize=SUPTITLE_FONTSIZE, fontweight="bold")
    if cmp is not None:
        footer = auc_footer_text(cmp, label_a, label_b, unit)
        if footer_extra:
            footer += "\n" + footer_extra
        fig.text(0.5, 0.012, footer, ha="center", va="bottom", fontsize=8,
                 color=TEXT_SEC, fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F0F0",
                           edgecolor="#CCCCCC", alpha=0.95))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info("AUC comparison figure saved: %s", out_path.name)
    return out_path


# ----------------------------------------------------- every channel at once

@dataclass
class AucSummaryRow:
    """One channel's comparison, for the all-channels view and batch export."""
    display: str
    unit: str
    cmp: object


def auc_summary(channels: List[tuple], df_a: pd.DataFrame, df_b: pd.DataFrame,
                weights_a: Optional[np.ndarray] = None,
                weights_b: Optional[np.ndarray] = None) -> List[AucSummaryRow]:
    """Run :func:`compare_distributions` for every channel in ``channels``.

    Percentiles only, no KDE -- this is what the all-channels chart needs, and
    it is cheap enough (a sort per channel) to run on a label change without
    anyone noticing.
    """
    from dtt.analysis.auc import compare_distributions

    rows: List[AucSummaryRow] = []
    for display, ca, cb in channels:
        va = pd.to_numeric(df_a[ca], errors="coerce").to_numpy(float)
        vb = pd.to_numeric(df_b[cb], errors="coerce").to_numpy(float)
        a, wa = _paired(va, weights_a)
        b, wb = _paired(vb, weights_b)
        cmp = compare_distributions(a, b, wa, wb)
        if cmp is not None:
            rows.append(AucSummaryRow(display, channel_unit(display), cmp))
    return rows


# The share of dataset 2 beyond dataset 1's P95 when the two distributions are
# identical -- 5% by the definition of a 95th percentile. It is the no-change
# line every bar is read against, which is what makes this metric comparable
# across channels whose units and magnitudes are not.
_EXCEED_BASELINE = 5.0


def _fmt_delta(delta: float, scale: float) -> str:
    """A ΔP95 with as many decimals as the channel's size warrants.

    Integers throughout printed "+0 daN·m" for every moment on the reference
    studies, which hides a real shift behind rounding on a channel whose P95
    is tens rather than hundreds. Decimals follow the magnitude of the
    reference P95 instead.
    """
    mag = abs(scale) if np.isfinite(scale) else 0.0
    places = 0 if mag >= 100 else 1 if mag >= 10 else 2
    return f"{delta:+.{places}f}"


def draw_auc_summary(ax, rows: List[AucSummaryRow], label_a: str,
                     label_b: str) -> None:
    """Every channel on one chart: how much of dataset 2 sits beyond dataset
    1's P95.

    Not ΔP95. A raw ΔP95 in daN puts Fz (~800) and Fy (~20) on one axis, and
    moments in daN·m on the same axis again; the chart would be read by
    magnitude and say nothing. Exceedance share is unitless and has a built-in
    zero: 5% means the top of the distribution did not move, more means
    dataset 2 reaches loads dataset 1 rarely did. ΔP95 is written at the end
    of each bar in the channel's own unit, so nothing is lost.

    Two colours, same scheme as the panels: the bars describe dataset 2, so
    they are its red; the baseline is dataset 1's P95, so it is its green.
    """
    from dtt.analysis.histograms import (
        LABEL_FONTSIZE, TEXT_PRI, TEXT_SEC, TICK_FONTSIZE, _style_ax)

    _style_ax(ax)
    if not rows:
        ax.text(0.5, 0.5, "no channels in common", ha="center", va="center",
                transform=ax.transAxes, color=TEXT_PRI)
        return
    names = [r.display for r in rows]
    vals = [r.cmp.pct_exceed_ref_p95 for r in rows]
    y = np.arange(len(rows))
    ax.barh(y, vals, color=AUC_DS2_COLOR, alpha=0.75, height=0.62)
    ax.axvline(_EXCEED_BASELINE, color=AUC_DS1_COLOR, linewidth=1.8,
               linestyle="--", label=f"no change ({_EXCEED_BASELINE:.0f}%)")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=TICK_FONTSIZE)
    ax.invert_yaxis()
    right = max(max(vals), _EXCEED_BASELINE) * 1.30 + 1.0
    ax.set_xlim(0, right)
    for yi, r, v in zip(y, rows, vals):
        ax.text(v + right * 0.01, yi,
                f"ΔP95 {_fmt_delta(r.cmp.delta_p95, r.cmp.p95_ref)} {r.unit}",
                va="center", fontsize=8, color=TEXT_SEC)
    ax.set_xlabel(f"% of {label_b} beyond {label_a} P95",
                  fontsize=LABEL_FONTSIZE, fontweight="bold")
    ax.set_title(f"All channels — {label_b} against {label_a}",
                 fontsize=LABEL_FONTSIZE, fontweight="bold", color=TEXT_PRI)
    ax.legend(fontsize=8, loc="lower right", facecolor="#FFFFFF",
              edgecolor="#CCCCCC", labelcolor=TEXT_PRI)


def export_auc_batch(channels: List[tuple], df_a: pd.DataFrame,
                     df_b: pd.DataFrame, label_a: str, label_b: str,
                     out_dir: Path, range_mode: str = "autoscale",
                     weights_a: Optional[np.ndarray] = None,
                     weights_b: Optional[np.ndarray] = None,
                     progress=None) -> tuple:
    """A PNG per channel plus one summary CSV and the all-channels chart.

    The reference generates every channel in one go; one-at-a-time "Save
    PNG" does not scale to twelve wheels times six components. ``progress``,
    if given, is called as ``progress(i, n, display)`` so a GUI can keep its
    window responsive.

    Returns ``(pngs, csv_path, summary_png)``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dtt.analysis.histograms import BG

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = lambda s: "".join(c if c.isalnum() or c in "-_." else "_"   # noqa: E731
                             for c in s)
    weighting = "distance" if weights_a is not None else "sample count"
    pngs: List[Path] = []
    rows: List[Dict] = []
    for i, (display, ca, cb) in enumerate(channels):
        if progress is not None:
            progress(i, len(channels), display)
        va = pd.to_numeric(df_a[ca], errors="coerce").to_numpy(float)
        vb = pd.to_numeric(df_b[cb], errors="coerce").to_numpy(float)
        unit = channel_unit(display)
        png = out_dir / safe(f"auc_{display}_{label_a}_vs_{label_b}.png")
        generate_auc_comparison(va, vb, label_a, label_b, display, png, unit,
                                range_mode=range_mode, weights_a=weights_a,
                                weights_b=weights_b,
                                footer_extra=f"weighted by {weighting}")
        pngs.append(png)
        row = auc_comparison_stats(va, vb, label_a, label_b, display, unit,
                                   weights_a, weights_b)
        if row:
            rows.append(row)

    csv_path = out_dir / safe(f"auc_summary_{label_a}_vs_{label_b}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    summary = auc_summary(channels, df_a, df_b, weights_a, weights_b)
    fig, ax = plt.subplots(figsize=(11, 1.2 + 0.42 * max(1, len(summary))),
                           facecolor=BG)
    draw_auc_summary(ax, summary, label_a, label_b)
    fig.tight_layout()
    summary_png = out_dir / safe(f"auc_all_channels_{label_a}_vs_{label_b}.png")
    fig.savefig(summary_png, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    if progress is not None:
        progress(len(channels), len(channels), "done")
    return pngs, csv_path, summary_png
