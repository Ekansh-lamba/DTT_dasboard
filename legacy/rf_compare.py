"""DISABLED 2026-10-28 -- RF Compare, formerly the "A4 -- RF Compare" block of
dtt/analysis/study_compare.py: the front/rear rainflow Miner damage-ratio
overlay between two finished studies (Round 4). Nothing in the app imports
this; see legacy/README.md.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from dtt.config import PLOT_COLORS, FIGURE_DPI
from dtt.run_channels import RunChannels
from dtt.analysis.study_compare import load_study
from legacy.rainflow import RAINFLOW_MINER

logger = logging.getLogger(__name__)


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
    from legacy.rainflow import _extract_cycles
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
    from legacy.rainflow import _miner_damage

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
