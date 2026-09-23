"""Box plots for every parameter.

* ``boxplot_<comp>.png`` -- one per wheel component present (Fx, Fy, Fz, Mx,
  My, Mz), every wheel side by side. It used to be Fx/Fy/Fz only:
  ``RunChannels`` keeps forces alone, so moments were never asked for.
* ``boxplot_parameters.png`` -- every other measured parameter (speed,
  accelerations, yaw rate, wheel speeds...), one panel each on its own unit.

Channel names on the plots come from :mod:`dtt.channel_names`.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

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
from dtt.channel_names import channel_unit, display_name
from dtt.channels import COMPONENTS

logger = logging.getLogger(__name__)

BG       = PLOT_COLORS["bg"]
PANEL    = PLOT_COLORS["panel"]
TEXT_PRI = PLOT_COLORS["text_pri"]
TEXT_SEC = PLOT_COLORS["text_sec"]
DANGER   = PLOT_COLORS["danger"]
SUCCESS  = PLOT_COLORS["success"]

_BOX_STYLE = dict(
    patch_artist=True,
    notch=False,
    medianprops=dict(color="#000000", linewidth=2.5),
    whiskerprops=dict(color="#000000", linewidth=1.4, linestyle="--"),
    capprops=dict(color="#000000", linewidth=2),
    flierprops=dict(marker="o", markersize=2.5, markerfacecolor="#E67E22",
                    markeredgewidth=0, alpha=0.35),
    boxprops=dict(linewidth=1.6),
)

# Not load parameters: the clock, where the vehicle is, which way it points,
# and running totals. A box plot of latitude or of a cumulative distance
# describes the route, not the loading.
_NOT_A_PARAMETER = re.compile(
    r"^(time|dist|distance|latitude|longitude|altitude|lat|lon|"
    r"angleheading|angletrack)$|_angle(_\d+)?$", re.I)


def _style_ax(ax):
    ax.set_facecolor("#F5F5F5")
    ax.tick_params(colors="#000000", labelsize=10, width=1.2)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")
    ax.grid(True, axis="y", color="#CCCCCC", linewidth=0.6, alpha=0.8)


def _finite(df: pd.DataFrame, ch: str) -> np.ndarray:
    v = pd.to_numeric(df[ch], errors="coerce").to_numpy(float)
    return v[np.isfinite(v)]


def wheel_channels(df: pd.DataFrame, rc) -> Dict[str, List[Tuple[str, str]]]:
    """``{component: [(wheel label, column), ...]}`` for every component.

    Resolved from the channel set, not ``rc.channel_for`` -- that keeps
    Fx/Fy/Fz only, which is why moments never had box plots.
    """
    out: Dict[str, List[Tuple[str, str]]] = {}
    if rc is None:
        for w, chans in WHEEL_GROUPS.items():
            for comp in COMPONENTS:
                ch = f"{w}_{comp}"
                if ch in df.columns:
                    out.setdefault(comp, []).append((w, ch))
        return out
    from dtt.analysis.study_compare import _channel_index
    index = _channel_index(rc)
    order = list(rc.wheel_groups.keys())
    for (label, comp), ch in index.items():
        if ch in df.columns:
            out.setdefault(comp, []).append((label, ch))
    for comp in out:
        out[comp].sort(key=lambda lc: order.index(lc[0]) if lc[0] in order else 99)
    return out


def other_parameters(df: pd.DataFrame, wheel_cols) -> List[str]:
    """Every numeric, varying column that is not a wheel channel or excluded."""
    taken = set(wheel_cols)
    out = []
    for ch in df.columns:
        if ch in taken or _NOT_A_PARAMETER.search(str(ch)):
            continue
        v = _finite(df, ch)
        if v.size < 4 or float(np.nanstd(v)) == 0.0:
            continue          # empty, or a channel that never moves (all-zero GPS)
        out.append(ch)
    return out


def _robust_ylim(arrays) -> Optional[Tuple[float, float]]:
    """Frame on P0.1..P99.9 of all boxes, so one acquisition glitch cannot
    shrink every box to a line. Fliers beyond it are clipped, and the stats
    strip under the plot still reports the true min/max."""
    both = np.concatenate([a for a in arrays if a.size])
    if not both.size:
        return None
    lo, hi = np.percentile(both, [0.1, 99.9])
    if hi <= lo:
        return None
    pad = 0.08 * (hi - lo)
    return float(lo - pad), float(hi + pad)


def _mark_stats(ax, data):
    n = len(data)
    for i, vals in enumerate(data, start=1):
        x_lo, x_hi = (i - 1 + 0.31) / n, (i - 1 + 0.69) / n
        ax.axhline(np.percentile(vals, 5), xmin=x_lo, xmax=x_hi, color="#1E8449",
                   linewidth=2.5, linestyle=":")
        ax.axhline(np.mean(vals), xmin=x_lo, xmax=x_hi, color="#B7950B",
                   linewidth=2.5, linestyle="--")
        ax.axhline(np.percentile(vals, 95), xmin=x_lo, xmax=x_hi, color="#C0392B",
                   linewidth=2.6, linestyle="-.")


def _legend(ax, fontsize=8):
    handles = [
        Line2D([0], [0], color="white",   linewidth=2.5,                label="Median"),
        Line2D([0], [0], color=SUCCESS,   linewidth=1.5, linestyle=":", label="P5"),
        Line2D([0], [0], color="#F1C40F", linewidth=1.5, linestyle="--", label="Mean"),
        Line2D([0], [0], color=DANGER,    linewidth=1.6, linestyle="-.", label="P95"),
        Line2D([0], [0], color=TEXT_SEC,  linewidth=1.2, linestyle="--", label="Whiskers (1.5×IQR)"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#E67E22", markersize=5,                  label="Outliers"),
    ]
    ax.legend(handles=handles, fontsize=fontsize, facecolor=BG, labelcolor="white",
              framealpha=0.95, edgecolor="#AAAAAA", loc="upper right")


def _stats_line(name, vals):
    p5, q1, med = np.percentile(vals, 5), np.percentile(vals, 25), np.median(vals)
    mu, q3, p95 = np.mean(vals), np.percentile(vals, 75), np.percentile(vals, 95)
    return (f"{name:>14s} │ Min:{vals.min():>8.1f} │ P5:{p5:>8.1f} │ "
            f"Q1:{q1:>8.1f} │ Median:{med:>8.1f} │ Mean:{mu:>8.1f} │ "
            f"Q3:{q3:>8.1f} │ P95:{p95:>8.1f} │ Max:{vals.max():>8.1f}")


def _component_boxplot(df, comp, wheel_cols, colors, out) -> Optional[str]:
    data, labels, cols = [], [], []
    for label, ch in wheel_cols:
        vals = _finite(df, ch)
        if vals.size < 4:
            continue
        data.append(vals)
        labels.append(display_name(ch))
        cols.append(colors.get(ch) or colors.get(label, "#00B4D8"))
    if not data:
        return None

    unit = channel_unit(wheel_cols[0][1]) or ("daN·m" if comp.startswith("M") else "daN")
    kind = "Moment" if comp.startswith("M") else "Force"
    fig, ax = plt.subplots(figsize=(max(7, 2.5 * len(data)), 8), facecolor=BG)
    fig.patch.set_facecolor(BG)
    _style_ax(ax)
    bp = ax.boxplot(data, widths=0.38, **_BOX_STYLE)
    for patch, col in zip(bp["boxes"], cols):
        patch.set_facecolor(col)
        patch.set_alpha(0.70)
    _mark_stats(ax, data)

    ylim = BOX_YLIMS.get(comp)
    if ylim is not None:
        # the fixed force frame is the reference analyzer's; widen it rather
        # than cut a box when a vehicle's loads reach past it
        lo = min(ylim[0], min(float(np.percentile(v, 5)) for v in data))
        hi = max(ylim[1], max(float(np.percentile(v, 95)) for v in data))
        ax.set_ylim(lo, hi)
    else:
        rl = _robust_ylim(data)
        if rl:
            ax.set_ylim(rl)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=11, color="#000000", fontweight="bold")
    ax.set_ylabel(f"{kind} ({unit})", color="#000000", fontsize=11, fontweight="bold")
    ax.set_title(f"Box Plot — {comp} — All Wheels", color=TEXT_PRI, fontsize=12,
                 fontweight="bold")
    _legend(ax)
    fig.text(0.5, 0.01, "\n".join(_stats_line(l, v) for l, v in zip(labels, data)),
             ha="center", va="bottom", fontsize=6.5, color="#000000",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F0F0",
                       edgecolor="#AAAAAA", alpha=0.95))
    fig.tight_layout(rect=[0, 0.14, 1, 0.97])
    fname = out / f"boxplot_{comp}.png"
    fig.savefig(fname, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info("Saved boxplot: %s", fname.name)
    return fname.name


def _parameters_boxplot(df, params, out) -> Optional[str]:
    data = [(ch, _finite(df, ch)) for ch in params]
    data = [(ch, v) for ch, v in data if v.size >= 4]
    if not data:
        return None
    # wide rather than tall: the sheet goes on a 16:9 report slide
    ncol = min(7, max(1, len(data)))
    nrow = int(np.ceil(len(data) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.6 * nrow + 0.8),
                             facecolor=BG, squeeze=False)
    fig.patch.set_facecolor(BG)
    for ax, (ch, vals) in zip(axes.ravel(), data):
        _style_ax(ax)
        ax.tick_params(labelsize=8)
        bp = ax.boxplot([vals], widths=0.45, **_BOX_STYLE)
        bp["boxes"][0].set_facecolor("#5DADE2")
        bp["boxes"][0].set_alpha(0.70)
        _mark_stats(ax, [vals])
        rl = _robust_ylim([vals])
        if rl:
            ax.set_ylim(rl)
        unit = channel_unit(ch)
        ax.set_xticks([])
        ax.set_title(display_name(ch), fontsize=10, fontweight="bold", color=TEXT_PRI)
        ax.set_ylabel(unit, fontsize=9, color=TEXT_PRI)
        ax.text(0.5, -0.06,
                f"P5 {np.percentile(vals, 5):.3g} · median {np.median(vals):.3g}"
                f" · P95 {np.percentile(vals, 95):.3g}",
                transform=ax.transAxes, ha="center", va="top", fontsize=7,
                color=TEXT_SEC)
    for ax in axes.ravel()[len(data):]:
        ax.axis("off")
    _legend(axes.ravel()[0], fontsize=6)
    fig.suptitle("Box Plots — Vehicle & Other Parameters", color=TEXT_PRI,
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fname = out / "boxplot_parameters.png"
    fig.savefig(fname, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info("Saved boxplot: %s  (%d parameters)", fname.name, len(data))
    return fname.name


def generate_boxplots(df: pd.DataFrame, config: RunConfig) -> List[str]:
    """Write every box plot for ``df``; returns the file names written."""
    out = config.figures_dir
    rc = config.run_channels
    by_comp = wheel_channels(df, rc)
    colors = dict(CHAN_COLORS)
    if rc is not None:
        colors.update(rc.chan_colors)
        for label, wc in rc.wheel_colors.items():
            colors[label] = wc["pri"]

    written = []
    for comp in COMPONENTS:
        if comp in by_comp:
            name = _component_boxplot(df, comp, by_comp[comp], colors, out)
            if name:
                written.append(name)
    wheel_cols = [ch for pairs in by_comp.values() for _, ch in pairs]
    name = _parameters_boxplot(df, other_parameters(df, wheel_cols), out)
    if name:
        written.append(name)
    return written
