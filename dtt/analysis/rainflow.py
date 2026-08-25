import csv
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rainflow

from dtt.config import (
    WHEEL_GROUPS,
    WHEEL_COLORS,
    CHAN_COLORS,
    PLOT_COLORS,
    RAINFLOW_BINS,
    RAINFLOW_MAXPTS,
    RF_STEPS,
    RF_COLORS,
    FIGURE_DPI,
    RunConfig,
)
from dtt.analysis.plot_style import draw_histogram

logger = logging.getLogger(__name__)

BG       = PLOT_COLORS["bg"]
PANEL    = PLOT_COLORS["panel"]
TEXT_PRI = PLOT_COLORS["text_pri"]
TEXT_SEC = PLOT_COLORS["text_sec"]
ENTRY_BG = PLOT_COLORS["entry_bg"]
DANGER   = PLOT_COLORS["danger"]
SUCCESS  = PLOT_COLORS["success"]


def _style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT_SEC, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(PLOT_COLORS["accent"])
        sp.set_linewidth(0.5)


def _rf_color(count: float) -> str:
    for i, thr in enumerate(RF_STEPS[1:], 1):
        if count <= thr:
            return RF_COLORS[i]
    return RF_COLORS[-1]


def _extract_cycles(df: pd.DataFrame, ch: str) -> list:
    data = pd.to_numeric(df[ch], errors="coerce").dropna().values
    if len(data) > RAINFLOW_MAXPTS:
        step = len(data) // RAINFLOW_MAXPTS
        data = data[::step]
    return list(rainflow.extract_cycles(data))


def _build_fromto_matrix(cycles: list, n_bins: int = RAINFLOW_BINS, shared_edges=None):
    if not cycles:
        return np.zeros((n_bins, n_bins)), np.linspace(-1, 1, n_bins + 1)
    rng  = np.array([c[0] for c in cycles])
    mean = np.array([c[1] for c in cycles])
    cnt  = np.array([c[2] for c in cycles])
    frm  = mean - rng / 2
    to_  = mean + rng / 2
    if shared_edges is not None:
        edges = shared_edges
    else:
        lo    = min(frm.min(), to_.min())
        hi    = max(frm.max(), to_.max())
        edges = np.linspace(lo, hi, n_bins + 1)
    n   = len(edges) - 1
    mat = np.zeros((n, n))
    for f, t, c in zip(frm, to_, cnt):
        fi = int(np.clip(np.searchsorted(edges, f, "right") - 1, 0, n - 1))
        ti = int(np.clip(np.searchsorted(edges, t, "right") - 1, 0, n - 1))
        mat[ti, fi] += c
    return mat, edges


def _draw_fromto_ax(ax, mat: np.ndarray, edges: np.ndarray, title: str, col: str):
    _style_ax(ax)
    extent = [edges[0], edges[-1], edges[0], edges[-1]]
    log_mat = np.log10(mat + 1)
    im = ax.imshow(
        log_mat,
        origin="lower",
        extent=extent,
        cmap="turbo",
        aspect="auto",
    )
    ax.set_title(title, color=col, fontsize=8, fontweight="bold")
    ax.set_xlabel("From (daN)", color=TEXT_SEC, fontsize=7)
    ax.set_ylabel("To (daN)", color=TEXT_SEC, fontsize=7)
    return im


def _miner_damage(cycles: list, m: float) -> float:
    if not cycles:
        return 0.0
    rng = np.array([c[0] for c in cycles])
    cnt = np.array([c[2] for c in cycles])
    return float(np.sum(cnt * (rng ** m)))


def _save_cycles_csv(cycles: list, ch: str, out_dir: Path) -> None:
    fname = out_dir / f"rainflow_cycles_{ch}.csv"
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["range_daN", "mean_daN", "count", "from_daN", "to_daN"])
        for rng, mean, cnt, *_ in cycles:
            writer.writerow([
                round(rng,  4),
                round(mean, 4),
                round(cnt,  4),
                round(mean - rng / 2, 4),
                round(mean + rng / 2, 4),
            ])


def generate_rainflow(df: pd.DataFrame, config: RunConfig) -> None:
    out    = config.figures_dir
    m      = config.miner_exponent
    rc     = config.run_channels
    wheel_groups = rc.wheel_groups if rc is not None else WHEEL_GROUPS
    wheel_colors = rc.wheel_colors if rc is not None else WHEEL_COLORS
    chan_colors  = rc.chan_colors if rc is not None else CHAN_COLORS
    wheels = list(wheel_groups.keys())

    for wheel in wheels:
        channels = [ch for ch in wheel_groups[wheel] if ch in df.columns]
        if not channels:
            continue

        col_pri = wheel_colors[wheel]["pri"]
        all_cycles = {}
        for ch in channels:
            cycles = _extract_cycles(df, ch)
            all_cycles[ch] = cycles
            _save_cycles_csv(cycles, ch, out)

        all_ranges = []
        for ch, cycs in all_cycles.items():
            if cycs:
                rng = np.array([c[0] for c in cycs])
                cnt = np.array([c[2] for c in cycs])
                all_ranges.extend(np.repeat(rng, cnt.astype(int).clip(1)))
        shared_edges = None
        if all_ranges:
            lo = 0.0
            hi = float(np.percentile(all_ranges, 99))
            if hi > lo:
                shared_edges = np.linspace(lo, hi, RAINFLOW_BINS + 1)

        fig, axes = plt.subplots(2, len(channels), figsize=(5.5 * len(channels), 9), facecolor=BG)
        if len(channels) == 1:
            axes = [[axes[0]], [axes[1]]]
        fig.suptitle(
            f"Rainflow  –  {wheel}  (Siemens From-To Matrix)\n"
            f"Miner damage exponent m={m}",
            color=TEXT_PRI, fontsize=11, fontweight="bold",
        )

        for col_idx, ch in enumerate(channels):
            col    = chan_colors.get(ch, col_pri)
            cycles = all_cycles[ch]
            mat, edges = _build_fromto_matrix(cycles, RAINFLOW_BINS, shared_edges)

            ax1 = axes[0][col_idx]
            _draw_fromto_ax(ax1, mat, edges, f"{ch}  –  From-To Matrix", col)

            rng_arr = np.array([c[0] for c in cycles]) if cycles else np.array([0.0])
            cnt_arr = np.array([c[2] for c in cycles]) if cycles else np.array([1.0])
            rng_rep = np.repeat(rng_arr, cnt_arr.astype(int).clip(1))
            p95_val = float(np.percentile(rng_rep, 95)) if len(rng_rep) > 0 else 0.0
            dmg     = _miner_damage(cycles, m)

            ax2 = axes[1][col_idx]
            _style_ax(ax2)
            if len(rng_rep) > 0:
                bin_edges = np.linspace(0, float(np.percentile(rng_rep, 99)), 35)
                # Range is |from - to|, never negative by construction, so this
                # is always the one-sided case: no forced symmetric bell, a
                # boundary-reflected KDE anchored at 0.
                draw_histogram(ax2, rng_rep, bin_edges, color=col,
                               bar_is_density=True, boundary=0.0)
                ax2.axvline(p95_val, color="#FFFFFF", linewidth=1.8, linestyle="--",
                            label=f"P95: {p95_val:.0f} daN")
            ax2.set_title(f"{ch}  –  Range Distribution", color=col, fontsize=8, fontweight="bold")
            ax2.set_xlabel("Range (daN)", color=TEXT_SEC, fontsize=7)
            ax2.set_ylabel("Density", color=TEXT_SEC, fontsize=7)

            ax2.text(
                0.97, 0.95,
                f"Damage\n(m={m})\n{dmg:.2e}",
                transform=ax2.transAxes, fontsize=7, color=DANGER,
                ha="right", va="top", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=ENTRY_BG, edgecolor=DANGER, alpha=0.9),
            )
            ax2.legend(fontsize=6.5, facecolor=BG, labelcolor="white", framealpha=0.85)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fname = out / f"rainflow_{wheel}.png"
        fig.savefig(fname, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        logger.info("Saved rainflow: %s", fname.name)
