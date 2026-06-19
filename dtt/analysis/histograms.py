import logging
from pathlib import Path
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
    FORCE_RANGES_DAN,
    HIST_BINS,
    PLOT_COLORS,
    SPEED_CANDIDATES,
    FIGURE_DPI,
    RunConfig,
)

logger = logging.getLogger(__name__)

BG       = PLOT_COLORS["bg"]
TEXT_SEC = PLOT_COLORS["text_sec"]
TEXT_PRI = PLOT_COLORS["text_pri"]
PANEL    = PLOT_COLORS["panel"]


def _try_find_col(df: pd.DataFrame, names: list) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def _style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT_SEC, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(PLOT_COLORS["accent"])
        sp.set_linewidth(0.7)
    ax.xaxis.label.set_color(TEXT_SEC)
    ax.yaxis.label.set_color(TEXT_SEC)
    ax.title.set_color(TEXT_PRI)


def _get_speed_weights(df: pd.DataFrame, sr: float) -> tuple:
    col = _try_find_col(df, SPEED_CANDIDATES)
    if col is None:
        return None, None, "none"
    raw = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    med = float(np.nanmedian(raw)) if raw.size else 0.0
    if med > 10.0:
        sp_mps = raw / 3.6
        unit   = "km/h"
    else:
        sp_mps = raw
        unit   = "m/s"
    weights     = sp_mps.values / float(sr)
    total_m     = float(np.nansum(np.nan_to_num(weights)))
    return weights, total_m, unit


def _get_force_type(ch: str) -> Optional[str]:
    for ft in ["Fx", "Fy", "Fz"]:
        if ft.lower() in ch.lower():
            return ft
    return None


def _plot_single_histogram(ax, vals, weights, bins, force_type, ch, mode, total_m, col):
    _style_ax(ax)
    if mode == "distance":
        w_plot = weights / 1000.0 if weights is not None else np.ones(len(vals))
        ylabel = "Distance (km)"
        total_label = f"{total_m / 1000.0:.3f} km" if total_m else ""
    else:
        if weights is not None and total_m and total_m > 0:
            w_plot = (weights / total_m) * 100.0
        else:
            w_plot = np.ones(len(vals)) / len(vals) * 100.0
        ylabel = "% Distance"
        total_label = f"{np.nansum(w_plot):.1f} %" if weights is not None else ""

    ax.bar(
        (bins[:-1] + bins[1:]) / 2.0,
        np.histogram(vals, bins=bins, weights=w_plot)[0],
        width=(bins[1] - bins[0]),
        color=col,
        edgecolor="none",
        alpha=0.82,
    )
    xlim = FORCE_RANGES_DAN.get(force_type) if force_type else None
    if xlim:
        ax.set_xlim(xlim)
    ax.set_xlabel("Force (daN)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(f"{ch}  [{mode}]  {total_label}", fontsize=9, fontweight="bold", color=col)


def generate_histograms(df: pd.DataFrame, config: RunConfig) -> None:
    out = config.figures_dir
    sr  = config.sampling_rate
    weights_full, total_m, speed_unit = _get_speed_weights(df, sr)

    for wheel, channels in WHEEL_GROUPS.items():
        wc   = WHEEL_COLORS[wheel]
        present = [ch for ch in channels if ch in df.columns]
        if not present:
            continue

        for mode in ("distance", "percentage"):
            fig, axes = plt.subplots(1, len(present), figsize=(5.5 * len(present), 4.5), facecolor=BG)
            if len(present) == 1:
                axes = [axes]
            fig.suptitle(
                f"{wheel}  –  Force Distribution  ({mode.title()})\n"
                f"Speed weighting: {speed_unit}",
                color=TEXT_PRI, fontsize=11, fontweight="bold",
            )

            for ax, ch in zip(axes, present):
                col        = CHAN_COLORS.get(ch, wc["pri"])
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

                xlim = FORCE_RANGES_DAN.get(force_type) if force_type else None
                if xlim:
                    bins = np.linspace(xlim[0], xlim[1], HIST_BINS + 1)
                else:
                    bins = np.linspace(vals.min(), vals.max(), HIST_BINS + 1)

                _plot_single_histogram(ax, vals, w, bins, force_type, ch, mode, total_m, col)

            fig.tight_layout(rect=[0, 0, 1, 0.93])
            fname = out / f"hist_{mode}_{wheel}.png"
            fig.savefig(fname, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=BG)
            plt.close(fig)
            logger.info("Saved histogram: %s", fname.name)

    for ch in [ch for g in WHEEL_GROUPS.values() for ch in g if ch in df.columns]:
        col        = CHAN_COLORS.get(ch, "#00B4D8")
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
        xlim = FORCE_RANGES_DAN.get(force_type) if force_type else None
        bins = np.linspace(xlim[0], xlim[1], HIST_BINS + 1) if xlim else np.linspace(vals.min(), vals.max(), HIST_BINS + 1)

        for mode in ("distance", "percentage"):
            fig, ax = plt.subplots(figsize=(6, 4), facecolor=BG)
            _plot_single_histogram(ax, vals, w, bins, force_type, ch, mode, total_m, col)
            fig.tight_layout()
            fname = out / f"hist_{mode}_{ch}.png"
            fig.savefig(fname, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=BG)
            plt.close(fig)
