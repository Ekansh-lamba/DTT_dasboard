import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dtt.config import (
    WHEEL_GROUPS,
    WHEEL_COLORS,
    FZ_HEATMAP_BINS_DAN,
    FXY_HEATMAP_RANGE_DAN,
    HEXBIN_EXTENT_N,
    HEXBIN_GRIDSIZE,
    FIGURE_DPI,
    RunConfig,
)

logger = logging.getLogger(__name__)


def _style_ax(ax):
    ax.set_facecolor("white")
    ax.tick_params(colors="#000000", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#CCCCCC")
    ax.grid(color="gray", linestyle="--", linewidth=0.4, alpha=0.4)


def _comp_bins(tyre):
    if tyre is not None:
        fx, fy, fz = tyre.fx_range_dan, tyre.fy_range_dan, tyre.fz_range_dan
    else:
        r = FXY_HEATMAP_RANGE_DAN
        fx = fy = (-r, r)
        fz = (FZ_HEATMAP_BINS_DAN[0], FZ_HEATMAP_BINS_DAN[-1])
    return {
        "Fx": np.linspace(fx[0], fx[1], 7),
        "Fy": np.linspace(fy[0], fy[1], 7),
        "Fz": np.linspace(fz[0], fz[1], 7),
    }


def _series(df, rc, wheel, comp):
    name = rc.channel_for(wheel, comp) if rc is not None else f"{wheel}_{comp}"
    if name and name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").values
    return None


def _hist2d_pct(x, y, x_bins, y_bins):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    H, xe, ye = np.histogram2d(x[mask], y[mask], bins=[x_bins, y_bins])
    total = H.sum()
    if total == 0:
        return None
    return H / total * 100.0, xe, ye


def _draw_panel(ax, panel, title, xlabel, ylabel, col, vmax):
    _style_ax(ax)
    ax.set_title(title, fontsize=10, fontweight="bold", color=col)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if panel is None:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return None
    H, xe, ye = panel
    im = ax.imshow(
        H.T, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
        cmap="turbo", vmin=0, vmax=vmax, aspect="auto",
    )
    for i in range(len(xe) - 1):
        for j in range(len(ye) - 1):
            v = H[i, j]
            if v > 0.2:
                ax.text((xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2,
                        f"{v:.1f}%", color="white", ha="center", va="center",
                        fontsize=6.5, fontweight="bold")
    return im


def _pair_heatmap(df, rc, wheels, wheel_colors, x_comp, y_comp, x_bins, y_bins,
                  x_label, y_label, out, fname, suptitle, reverse_axes=False):
    n = len(wheels)
    ncol = 2 if n <= 4 else 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 3.5 * nrow), facecolor="white")
    axes = np.atleast_1d(axes).ravel()

    panels = []
    for wheel in wheels:
        x = _series(df, rc, wheel, x_comp)
        y = _series(df, rc, wheel, y_comp)
        panels.append(_hist2d_pct(x, y, x_bins, y_bins) if x is not None and y is not None else None)

    vmax = max((float(np.max(p[0])) for p in panels if p is not None), default=1.0)

    last_im = None
    for ax, wheel, panel in zip(axes, wheels, panels):
        col = wheel_colors[wheel]["pri"]
        im = _draw_panel(ax, panel, f"{wheel}  –  {y_comp} vs {x_comp}", x_label, y_label, col, vmax)
        if im is not None:
            last_im = im
        if reverse_axes:
            ax.set_xlim(x_bins[-1], x_bins[0])
            ax.set_ylim(y_bins[-1], y_bins[0])
    for ax in axes[n:]:
        ax.axis("off")

    fig.subplots_adjust(right=0.88, wspace=0.35, hspace=0.4)
    if last_im is not None:
        cbar = fig.colorbar(last_im, cax=fig.add_axes([0.90, 0.15, 0.02, 0.7]))
        cbar.set_label("% of Occurrences", fontsize=10, fontweight="bold")
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    path = out / fname
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved heatmap: %s", path.name)


def generate_heatmaps(df: pd.DataFrame, config: RunConfig) -> None:
    out = config.figures_dir
    rc = config.run_channels
    wheels = list(rc.wheel_groups.keys()) if rc is not None else list(WHEEL_GROUPS.keys())
    wheel_colors = rc.wheel_colors if rc is not None else WHEEL_COLORS
    bins = _comp_bins(config.tyre)

    _pair_heatmap(df, rc, wheels, wheel_colors, "Fy", "Fx", bins["Fy"], bins["Fx"],
                  "Fy (daN)", "Fx (daN)", out, "heatmap_fx_fy_all.png",
                  "Fx vs Fy  –  % Occurrence Heatmap", reverse_axes=True)
    _pair_heatmap(df, rc, wheels, wheel_colors, "Fy", "Fz", bins["Fy"], bins["Fz"],
                  "Fy (daN)", "Fz (daN)", out, "heatmap_fz_fy_all.png",
                  "Fz vs Fy  –  % Occurrence Heatmap")
    _pair_heatmap(df, rc, wheels, wheel_colors, "Fx", "Fz", bins["Fx"], bins["Fz"],
                  "Fx (daN)", "Fz (daN)", out, "heatmap_fz_fx_all.png",
                  "Fz vs Fx  –  % Occurrence Heatmap")

    for wheel in wheels:
        col = wheel_colors[wheel]["pri"]
        fx = _series(df, rc, wheel, "Fx")
        fy = _series(df, rc, wheel, "Fy")
        if fx is None or fy is None:
            continue
        fx_n, fy_n = fx * 10.0, fy * 10.0
        mask = np.isfinite(fx_n) & np.isfinite(fy_n)
        if mask.sum() < 2:
            continue
        fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
        hb = ax.hexbin(fx_n[mask], fy_n[mask], gridsize=HEXBIN_GRIDSIZE,
                       cmap="turbo", bins="log", mincnt=1, extent=HEXBIN_EXTENT_N)
        ax.set_facecolor("white")
        ax.set_title(f"{wheel}  –  Fy vs Fx  (Density Hexbin)", fontsize=11, fontweight="bold", color=col)
        ax.set_xlabel("Fx (N)", fontsize=10)
        ax.set_ylabel("Fy (N)", fontsize=10)
        ax.set_xlim(HEXBIN_EXTENT_N[:2])
        ax.set_ylim(HEXBIN_EXTENT_N[2:])
        ax.grid(linestyle="--", alpha=0.3)
        plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.04).set_label("Density (log)", fontsize=9)
        fig.tight_layout()
        path = out / f"heatmap_hexbin_{wheel}.png"
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved hexbin: %s", path.name)