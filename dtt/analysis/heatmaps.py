import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dtt.config import (
    WHEEL_GROUPS,
    WHEEL_COLORS,
    CHAN_COLORS,
    PLOT_COLORS,
    FZ_HEATMAP_BINS_DAN,
    FXY_HEATMAP_BIN_STEP_DAN,
    FXY_HEATMAP_RANGE_DAN,
    HEXBIN_EXTENT_N,
    HEXBIN_GRIDSIZE,
    FIGURE_DPI,
    RunConfig,
)

logger = logging.getLogger(__name__)

BG       = PLOT_COLORS["bg"]
TEXT_PRI = PLOT_COLORS["text_pri"]
TEXT_SEC = PLOT_COLORS["text_sec"]
PANEL    = PLOT_COLORS["panel"]


def _style_ax(ax):
    ax.set_facecolor("white")
    ax.tick_params(colors="#000000", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#CCCCCC")
    ax.grid(color="gray", linestyle="--", linewidth=0.4, alpha=0.4)


def _get_channel(df: pd.DataFrame, wheel: str, force: str):
    col = f"{wheel}_{force}"
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return None


def _heatmap_pct_annotated(ax, x_dan, y_dan, x_bins, y_bins, title, xlabel, ylabel, col):
    mask = (~np.isnan(x_dan)) & (~np.isnan(y_dan))
    if mask.sum() < 2:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return None

    H, xedges, yedges = np.histogram2d(x_dan[mask], y_dan[mask], bins=[x_bins, y_bins])
    total = H.sum()
    if total == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return None
    H_pct = (H / total) * 100.0

    im = ax.imshow(
        H_pct.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="turbo",
        vmin=0,
        vmax=float(np.max(H_pct)) if H_pct.max() > 0 else 1,
        aspect="auto",
    )

    for i in range(len(xedges) - 1):
        for j in range(len(yedges) - 1):
            val = H_pct[i, j]
            if val > 0.2:
                ax.text(
                    (xedges[i] + xedges[i + 1]) / 2,
                    (yedges[j] + yedges[j + 1]) / 2,
                    f"{val:.1f}%",
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    fontweight="bold",
                )

    _style_ax(ax)
    ax.set_title(title, fontsize=10, fontweight="bold", color=col)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    return im


def generate_heatmaps(df: pd.DataFrame, config: RunConfig) -> None:
    out = config.figures_dir

    fxy_bins = np.arange(
        -FXY_HEATMAP_RANGE_DAN,
        FXY_HEATMAP_RANGE_DAN + FXY_HEATMAP_BIN_STEP_DAN,
        FXY_HEATMAP_BIN_STEP_DAN,
    )
    fz_bins  = np.array(FZ_HEATMAP_BINS_DAN)
    wheels   = list(WHEEL_GROUPS.keys())

    fig_fxfy, axes_fxfy = plt.subplots(2, 2, figsize=(11, 7), facecolor="white")
    axes_fxfy = axes_fxfy.ravel()
    last_im_fxfy = None

    for ax, wheel in zip(axes_fxfy, wheels):
        col     = WHEEL_COLORS[wheel]["pri"]
        fx_s    = _get_channel(df, wheel, "Fx")
        fy_s    = _get_channel(df, wheel, "Fy")
        if fx_s is None or fy_s is None:
            ax.text(0.5, 0.5, f"{wheel}: no data", ha="center", va="center")
            continue
        fx_dan = fx_s.values.copy()
        fy_dan = fy_s.values.copy()
        im = _heatmap_pct_annotated(
            ax, fy_dan, fx_dan, fxy_bins, fxy_bins,
            f"{wheel}  –  Fx vs Fy", "Fy (daN)", "Fx (daN)", col,
        )
        if im is not None:
            last_im_fxfy = im
        ax.set_xlim(FXY_HEATMAP_RANGE_DAN, -FXY_HEATMAP_RANGE_DAN)
        ax.set_ylim(FXY_HEATMAP_RANGE_DAN, -FXY_HEATMAP_RANGE_DAN)
        ax.set_xticks(np.arange(-FXY_HEATMAP_RANGE_DAN, FXY_HEATMAP_RANGE_DAN + 100, 100))
        ax.set_yticks(np.arange(-FXY_HEATMAP_RANGE_DAN, FXY_HEATMAP_RANGE_DAN + 100, 100))

    fig_fxfy.subplots_adjust(right=0.86, wspace=0.35, hspace=0.4)
    if last_im_fxfy is not None:
        cbar_ax = fig_fxfy.add_axes([0.88, 0.15, 0.02, 0.7])
        cbar    = fig_fxfy.colorbar(last_im_fxfy, cax=cbar_ax)
        cbar.set_label("% of Occurrences", fontsize=10, fontweight="bold")
        cbar.ax.tick_params(labelsize=8)
    fig_fxfy.suptitle("Fx vs Fy  –  % Occurrence Heatmap (All Wheels)", fontsize=13, fontweight="bold")
    p_fxfy = out / "heatmap_fx_fy_all.png"
    fig_fxfy.savefig(p_fxfy, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig_fxfy)
    logger.info("Saved heatmap: %s", p_fxfy.name)

    fig_fzfy, axes_fzfy = plt.subplots(2, 2, figsize=(11, 7), facecolor="white")
    axes_fzfy = axes_fzfy.ravel()
    last_im_fzfy = None

    for ax, wheel in zip(axes_fzfy, wheels):
        col   = WHEEL_COLORS[wheel]["pri"]
        fz_s  = _get_channel(df, wheel, "Fz")
        fy_s  = _get_channel(df, wheel, "Fy")
        if fz_s is None or fy_s is None:
            ax.text(0.5, 0.5, f"{wheel}: no data", ha="center", va="center")
            continue
        fz_dan = fz_s.values.copy()
        fy_dan = fy_s.values.copy()
        im = _heatmap_pct_annotated(
            ax, fy_dan, fz_dan, fxy_bins, fz_bins,
            f"{wheel}  –  Fz vs Fy", "Fy (daN)", "Fz (daN)", col,
        )
        if im is not None:
            last_im_fzfy = im

    fig_fzfy.subplots_adjust(right=0.86, wspace=0.35, hspace=0.4)
    if last_im_fzfy is not None:
        cbar_ax = fig_fzfy.add_axes([0.88, 0.15, 0.02, 0.7])
        cbar    = fig_fzfy.colorbar(last_im_fzfy, cax=cbar_ax)
        cbar.set_label("% of Occurrences", fontsize=10, fontweight="bold")
    fig_fzfy.suptitle("Fz vs Fy  –  % Occurrence Heatmap (All Wheels)", fontsize=13, fontweight="bold")
    p_fzfy = out / "heatmap_fz_fy_all.png"
    fig_fzfy.savefig(p_fzfy, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig_fzfy)
    logger.info("Saved heatmap: %s", p_fzfy.name)

    fig_fzfx, axes_fzfx = plt.subplots(2, 2, figsize=(11, 7), facecolor="white")
    axes_fzfx = axes_fzfx.ravel()
    last_im_fzfx = None

    for ax, wheel in zip(axes_fzfx, wheels):
        col   = WHEEL_COLORS[wheel]["pri"]
        fz_s  = _get_channel(df, wheel, "Fz")
        fx_s  = _get_channel(df, wheel, "Fx")
        if fz_s is None or fx_s is None:
            ax.text(0.5, 0.5, f"{wheel}: no data", ha="center", va="center")
            continue
        fz_dan = fz_s.values.copy()
        fx_dan = fx_s.values.copy()
        im = _heatmap_pct_annotated(
            ax, fx_dan, fz_dan, fxy_bins, fz_bins,
            f"{wheel}  –  Fz vs Fx", "Fx (daN)", "Fz (daN)", col,
        )
        if im is not None:
            last_im_fzfx = im

    fig_fzfx.subplots_adjust(right=0.86, wspace=0.35, hspace=0.4)
    if last_im_fzfx is not None:
        cbar_ax = fig_fzfx.add_axes([0.88, 0.15, 0.02, 0.7])
        cbar    = fig_fzfx.colorbar(last_im_fzfx, cax=cbar_ax)
        cbar.set_label("% of Occurrences", fontsize=10, fontweight="bold")
    fig_fzfx.suptitle("Fz vs Fx  –  % Occurrence Heatmap (All Wheels)", fontsize=13, fontweight="bold")
    p_fzfx = out / "heatmap_fz_fx_all.png"
    fig_fzfx.savefig(p_fzfx, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig_fzfx)
    logger.info("Saved heatmap: %s", p_fzfx.name)

    for wheel in wheels:
        col  = WHEEL_COLORS[wheel]["pri"]
        fx_s = _get_channel(df, wheel, "Fx")
        fy_s = _get_channel(df, wheel, "Fy")
        if fx_s is None or fy_s is None:
            continue
        fx_n = fx_s.values * 10.0
        fy_n = fy_s.values * 10.0
        mask = (~np.isnan(fx_n)) & (~np.isnan(fy_n))
        if mask.sum() < 2:
            continue
        fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
        hb = ax.hexbin(
            fx_n[mask], fy_n[mask],
            gridsize=HEXBIN_GRIDSIZE,
            cmap="turbo",
            bins="log",
            mincnt=1,
            extent=HEXBIN_EXTENT_N,
        )
        ax.set_facecolor("white")
        ax.set_title(f"{wheel}  –  Fy vs Fx  (Density Hexbin)", fontsize=11, fontweight="bold", color=col)
        ax.set_xlabel("Fx (N)", fontsize=10)
        ax.set_ylabel("Fy (N)", fontsize=10)
        ax.set_xlim(HEXBIN_EXTENT_N[:2])
        ax.set_ylim(HEXBIN_EXTENT_N[2:])
        ax.grid(linestyle="--", alpha=0.3)
        cbar = plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Density (log)", fontsize=9)
        fig.tight_layout()
        p_hex = out / f"heatmap_hexbin_{wheel}.png"
        fig.savefig(p_hex, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved hexbin: %s", p_hex.name)
