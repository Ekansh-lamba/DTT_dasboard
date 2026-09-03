"""Single-study Area Under Curve (AUC) figures — the smoothed density view.

Item A split: :mod:`dtt.analysis.histograms` now draws actual binned bars
(the *Histogram* section); this module draws the filled KDE curve on its own
(the *AUC* section) that histograms used to show instead of bars. Same
underlying data, weighting, axis-range logic and KDE math as the Histogram
section — reused via import, not reimplemented — so the two sections agree
on what they're showing and only differ in bars vs. curve.

Single-study only: one channel's own distribution, with its own P5/P95
markers and stats strip. The two-run overlay (reference vs. current) is a
separate, already-existing feature (``dtt.analysis.auc``'s
``compare_distributions``/``auc_grid``, used by ``gui/pages/comparison_page.py``)
and is out of scope here.
"""

import logging
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dtt.config import (
    WHEEL_GROUPS,
    WHEEL_COLORS,
    CHAN_COLORS,
    HIST_BINS,
    FIGURE_DPI,
    RunConfig,
)
from dtt.analysis.plot_style import draw_histogram, draw_stats_strip
from dtt.analysis.histograms import (
    BG, PANEL, TEXT_PRI, TEXT_SEC,
    TICK_FONTSIZE, LABEL_FONTSIZE, TITLE_FONTSIZE, SUPTITLE_FONTSIZE,
    RANGE_MODES,
    _style_ax,
    _get_speed_weights,
    _get_force_type,
    _axis_range,
    weighted_plot_series,
)

logger = logging.getLogger(__name__)

P5_COLOR  = "#4A4A4A"
P95_COLOR = "#C0392B"


def _plot_single_auc(ax, vals, weights, bins, xlim, ch, mode, total_m, col):
    _style_ax(ax)
    w_plot, ylabel, total_label = weighted_plot_series(vals, weights, total_m, mode)

    # Item A: this is the *AUC* section — the filled, smoothed density curve,
    # no bars (the binned view lives in the Histogram section above).
    draw_histogram(ax, vals, bins, weights=w_plot, color=col, style="soft", show_kde=True)
    if xlim:
        ax.set_xlim(xlim)

    finite = vals[np.isfinite(vals)]
    if finite.size:
        p5, p95 = float(np.percentile(finite, 5)), float(np.percentile(finite, 95))
        ax.axvline(p5, color=P5_COLOR, linewidth=1.3, linestyle=":", label=f"P5 {p5:.0f}")
        ax.axvline(p95, color=P95_COLOR, linewidth=1.6, linestyle="--", label=f"P95 {p95:.0f}")
        ax.legend(fontsize=9, facecolor=BG, edgecolor="#CCCCCC", labelcolor=TEXT_PRI,
                  framealpha=0.9, loc="upper right")

    ax.set_xlabel("Force (daN)", fontsize=LABEL_FONTSIZE, fontweight="bold", color=TEXT_SEC)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE, fontweight="bold", color=TEXT_SEC)
    ax.set_title(f"{ch}  [{mode}]  {total_label}", fontsize=TITLE_FONTSIZE,
                fontweight="bold", color=TEXT_PRI)

    if finite.size:
        draw_stats_strip(ax, [(ch, finite)], colors=[col], theme="light",
                         fontsize=9, bbox=[0.0, -0.42, 1.0, 0.24])


def generate_auc(df: pd.DataFrame, config: RunConfig, range_mode: str = "full") -> None:
    """``range_mode``: same meaning and values as
    :func:`dtt.analysis.histograms.generate_histograms` -- ``"full"``
    (default) uses the configured sensor range while it fits the data,
    ``"autoscale"`` always clips to where the data actually sits.
    """
    if range_mode not in RANGE_MODES:
        raise ValueError(f"range_mode must be one of {RANGE_MODES}, got {range_mode!r}")
    suffix = "" if range_mode == "full" else f"_{range_mode}"

    out = config.figures_dir
    sr  = config.sampling_rate
    weights_full, total_m, speed_unit = _get_speed_weights(df, sr)

    rc = config.run_channels
    wheel_groups = rc.wheel_groups if rc is not None else WHEEL_GROUPS
    wheel_colors = rc.wheel_colors if rc is not None else WHEEL_COLORS
    chan_colors  = rc.chan_colors if rc is not None else CHAN_COLORS
    all_force    = rc.mandatory_channels if rc is not None else \
                   [ch for g in WHEEL_GROUPS.values() for ch in g]

    for wheel, channels in wheel_groups.items():
        wc   = wheel_colors[wheel]
        present = [ch for ch in channels if ch in df.columns]
        if not present:
            continue

        for mode in ("distance", "percentage"):
            fig, axes = plt.subplots(1, len(present), figsize=(6.5 * len(present), 6.4), facecolor=BG)
            if len(present) == 1:
                axes = [axes]
            fig.suptitle(
                f"{wheel}  –  Area Under Curve  ({mode.title()})\n"
                + (f"Speed weighting: {speed_unit}" if weights_full is not None
                   else "Sample-count weighting — no usable speed channel"),
                color=TEXT_PRI, fontsize=SUPTITLE_FONTSIZE, fontweight="bold",
            )

            for ax, ch in zip(axes, present):
                col        = chan_colors.get(ch, wc["pri"])
                force_type = _get_force_type(ch)
                series     = pd.to_numeric(df[ch], errors="coerce").dropna()
                vals       = series.values
                if len(vals) == 0:
                    continue

                if weights_full is not None:
                    w = weights_full[:len(vals)] if len(weights_full) >= len(vals) else np.ones(len(vals)) * np.nanmean(weights_full)
                    w = np.nan_to_num(w)
                else:
                    w = np.ones(len(vals))

                xlim = _axis_range(vals, force_type, ch, range_mode)
                bins = np.linspace(xlim[0], xlim[1], HIST_BINS + 1)

                _plot_single_auc(ax, vals, w, bins, xlim, ch, mode, total_m, col)

            fig.tight_layout(rect=[0, 0.08, 1, 0.90])
            fig.subplots_adjust(wspace=0.32)
            fname = out / f"auc_{mode}_{wheel}{suffix}.png"
            fig.savefig(fname, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=BG)
            plt.close(fig)
            logger.info("Saved AUC plot: %s", fname.name)

    for ch in [ch for ch in all_force if ch in df.columns]:
        col        = chan_colors.get(ch, "#00B4D8")
        force_type = _get_force_type(ch)
        series     = pd.to_numeric(df[ch], errors="coerce").dropna()
        vals       = series.values
        if len(vals) == 0:
            continue
        if weights_full is not None:
            w = weights_full[:len(vals)] if len(weights_full) >= len(vals) else np.ones(len(vals)) * np.nanmean(weights_full)
            w = np.nan_to_num(w)
        else:
            w = np.ones(len(vals))
        xlim = _axis_range(vals, force_type, ch, range_mode)
        bins = np.linspace(xlim[0], xlim[1], HIST_BINS + 1)

        for mode in ("distance", "percentage"):
            fig, ax = plt.subplots(figsize=(7.5, 6.2), facecolor=BG)
            _plot_single_auc(ax, vals, w, bins, xlim, ch, mode, total_m, col)
            fig.tight_layout(rect=[0, 0.1, 1, 1])
            fname = out / f"auc_{mode}_{ch}{suffix}.png"
            fig.savefig(fname, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=BG)
            plt.close(fig)
            logger.info("Saved AUC plot: %s", fname.name)
