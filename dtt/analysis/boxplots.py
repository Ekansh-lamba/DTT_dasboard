import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from dtt.config import (
    WHEEL_GROUPS,
    CHAN_COLORS,
    BOX_YLIMS,
    PLOT_COLORS,
    FIGURE_DPI,
    RunConfig,
)

logger = logging.getLogger(__name__)

BG       = PLOT_COLORS["bg"]
PANEL    = PLOT_COLORS["panel"]
TEXT_PRI = PLOT_COLORS["text_pri"]
TEXT_SEC = PLOT_COLORS["text_sec"]
DANGER   = PLOT_COLORS["danger"]
SUCCESS  = PLOT_COLORS["success"]


def _style_ax(ax):
    ax.set_facecolor("#F5F5F5")
    ax.tick_params(colors="#000000", labelsize=10, width=1.2)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")
    ax.grid(True, axis="y", color="#CCCCCC", linewidth=0.6, alpha=0.8)


def generate_boxplots(df: pd.DataFrame, config: RunConfig) -> None:
    out    = config.figures_dir
    rc     = config.run_channels
    wheels = list(rc.wheel_groups.keys()) if rc is not None else list(WHEEL_GROUPS.keys())

    for force_type in ["Fx", "Fy", "Fz"]:
        if rc is not None:
            channels = [rc.channel_for(w, force_type) for w in wheels]
        else:
            channels = [f"{w}_{force_type}" for w in wheels]
        present   = [ch for ch in channels if ch and ch in df.columns]
        if not present:
            continue

        data_per_ch  = []
        labels       = []
        colors       = []

        for ch in present:
            series = pd.to_numeric(df[ch], errors="coerce").dropna()
            if len(series) < 4:
                continue
            vals = series.values
            data_per_ch.append(vals)
            labels.append(ch)
            _chan_colors = rc.chan_colors if rc is not None else CHAN_COLORS
            colors.append(_chan_colors.get(ch, "#00B4D8"))

        if not data_per_ch:
            continue

        fig, ax = plt.subplots(figsize=(max(7, 2.5 * len(data_per_ch)), 8), facecolor=BG)
        fig.patch.set_facecolor(BG)
        _style_ax(ax)

        bp = ax.boxplot(
            data_per_ch,
            patch_artist=True,
            notch=False,
            widths=0.38,
            medianprops=dict(color="#000000", linewidth=2.5),
            whiskerprops=dict(color="#000000", linewidth=1.4, linestyle="--"),
            capprops=dict(color="#000000", linewidth=2),
            flierprops=dict(
                marker="o",
                markersize=2.5,
                markerfacecolor="#E67E22",
                markeredgewidth=0,
                alpha=0.35,
            ),
            boxprops=dict(linewidth=1.6),
        )

        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.70)

        # Each wheel's P5/Mean/P95 line is scoped to just its own box (xmin/xmax),
        # not drawn full-width across the other wheels' boxes.
        n = len(data_per_ch)
        for i, vals in enumerate(data_per_ch, start=1):
            p5  = np.percentile(vals, 5)
            mu  = np.mean(vals)
            p95 = np.percentile(vals, 95)
            x_lo = (i - 1 + 0.31) / n
            x_hi = (i - 1 + 0.69) / n
            ax.axhline(p5,  xmin=x_lo, xmax=x_hi, color="#1E8449", linewidth=2.5, linestyle=":")
            ax.axhline(mu,  xmin=x_lo, xmax=x_hi, color="#B7950B", linewidth=2.5, linestyle="--")
            ax.axhline(p95, xmin=x_lo, xmax=x_hi, color="#C0392B", linewidth=2.6, linestyle="-.")

        ylim = BOX_YLIMS.get(force_type)
        if ylim:
            ax.set_ylim(ylim)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=11, color="#000000", fontweight="bold")
        ax.set_ylabel("Force (daN)", color="#000000", fontsize=11, fontweight="bold")
        ax.set_title(f"Box Plot — {force_type} — All Wheels", color=TEXT_PRI, fontsize=12, fontweight="bold")

        legend_handles = [
            Line2D([0], [0], color="white",   linewidth=2.5,                label="Median"),
            Line2D([0], [0], color=SUCCESS,   linewidth=1.5, linestyle=":", label="P5"),
            Line2D([0], [0], color="#F1C40F", linewidth=1.5, linestyle="--", label="Mean"),
            Line2D([0], [0], color=DANGER,    linewidth=1.6, linestyle="-.", label="P95"),
            Line2D([0], [0], color=TEXT_SEC,  linewidth=1.2, linestyle="--", label="Whiskers (1.5×IQR)"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#E67E22", markersize=5,                  label="Outliers"),
        ]
        ax.legend(handles=legend_handles, fontsize=8, facecolor=BG, labelcolor="white",
                  framealpha=0.95, edgecolor="#AAAAAA", loc="upper right")

        stats_lines = []
        for ch, vals in zip(labels, data_per_ch):
            p5  = np.percentile(vals, 5)
            q1  = np.percentile(vals, 25)
            med = np.median(vals)
            mu  = np.mean(vals)
            q3  = np.percentile(vals, 75)
            p95 = np.percentile(vals, 95)
            stats_lines.append(
                f"{ch:>12s} │ Min:{vals.min():>7.0f} │ P5:{p5:>7.0f} │ "
                f"Q1:{q1:>7.0f} │ Median:{med:>7.0f} │ Mean:{mu:>7.0f} │ "
                f"Q3:{q3:>7.0f} │ P95:{p95:>7.0f} │ Max:{vals.max():>7.0f}"
            )
        fig.text(
            0.5, 0.01, "\n".join(stats_lines),
            ha="center", va="bottom", fontsize=6.5, color="#000000",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F0F0", edgecolor="#AAAAAA", alpha=0.95),
        )
        fig.tight_layout(rect=[0, 0.14, 1, 0.97])

        fname = out / f"boxplot_{force_type}.png"
        fig.savefig(fname, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        logger.info("Saved boxplot: %s", fname.name)
