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
from dtt.analysis.histograms import _get_speed_weights

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


def bins_from_ranges(bin_ranges):
    """Build bin edges from user-entered ``{comp: (min, max, step)}`` (daN).

    Mirrors the reference WFT analyzer's Min/Max/Step force-heatmap inputs.
    Raises ``ValueError`` with a message naming the offending component.
    """
    bins = {}
    for comp, (mn, mx, step) in bin_ranges.items():
        mn, mx, step = float(mn), float(mx), float(step)
        if step <= 0:
            raise ValueError(f"{comp}: step must be greater than 0.")
        if mx <= mn:
            raise ValueError(f"{comp}: max must be greater than min.")
        edges = np.arange(mn, mx + step / 2.0, step)
        if len(edges) < 2:
            raise ValueError(f"{comp}: range/step produces fewer than 2 bin edges.")
        bins[comp] = edges
    return bins


def _series(df, rc, wheel, comp):
    name = rc.channel_for(wheel, comp) if rc is not None else f"{wheel}_{comp}"
    if name and name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").values
    return None


def _hist2d_pct(x, y, x_bins, y_bins, weights=None):
    """2D percentage histogram. ``weights`` (per-sample distance, m) makes each
    cell's percentage a share of distance travelled rather than sample count."""
    mask = np.isfinite(x) & np.isfinite(y)
    if weights is not None:
        mask &= np.isfinite(weights)
    if mask.sum() < 2:
        return None
    w = weights[mask] if weights is not None else None
    H, xe, ye = np.histogram2d(x[mask], y[mask], bins=[x_bins, y_bins], weights=w)
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
                  x_label, y_label, out, fname, suptitle, reverse_axes=False,
                  weights=None, weighted=False):
    n = len(wheels)
    ncol = 2 if n <= 4 else 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 3.5 * nrow), facecolor="white")
    axes = np.atleast_1d(axes).ravel()

    panels = []
    for wheel in wheels:
        x = _series(df, rc, wheel, x_comp)
        y = _series(df, rc, wheel, y_comp)
        panels.append(_hist2d_pct(x, y, x_bins, y_bins, weights=weights)
                      if x is not None and y is not None else None)

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
        cbar.set_label("% of Distance" if weighted else "% of Occurrences",
                       fontsize=10, fontweight="bold")
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    path = out / fname
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved heatmap: %s", path.name)


def generate_correlation_heatmaps(df: pd.DataFrame, out, rc, wheels, wheel_colors,
                                   sampling_rate, tyre=None, bin_ranges=None) -> None:
    """(Re)generate the Fx-Fy / Fz-Fy / Fz-Fx correlation heatmaps.

    ``bin_ranges``, when given, is ``{"Fx": (min, max, step), "Fy": ..., "Fz": ...}``
    (daN) and overrides the tyre/default bin edges — this is what lets the GUI
    regenerate these three figures on demand with a user-chosen bin range,
    without re-running the whole pipeline.
    """
    bins = bins_from_ranges(bin_ranges) if bin_ranges else _comp_bins(tyre)

    weights_full, total_m, speed_unit = _get_speed_weights(df, sampling_rate)
    weighted = weights_full is not None
    metric = "% Distance" if weighted else "% Occurrence"
    weight_line = (f"Speed weighting: {speed_unit}" if weighted
                   else "Sample-count weighting — no usable speed channel")

    _pair_heatmap(df, rc, wheels, wheel_colors, "Fy", "Fx", bins["Fy"], bins["Fx"],
                  "Fy (daN)", "Fx (daN)", out, "heatmap_fx_fy_all.png",
                  f"Fx vs Fy  –  {metric} Heatmap\n{weight_line}", reverse_axes=True,
                  weights=weights_full, weighted=weighted)
    _pair_heatmap(df, rc, wheels, wheel_colors, "Fy", "Fz", bins["Fy"], bins["Fz"],
                  "Fy (daN)", "Fz (daN)", out, "heatmap_fz_fy_all.png",
                  f"Fz vs Fy  –  {metric} Heatmap\n{weight_line}",
                  weights=weights_full, weighted=weighted)
    _pair_heatmap(df, rc, wheels, wheel_colors, "Fx", "Fz", bins["Fx"], bins["Fz"],
                  "Fx (daN)", "Fz (daN)", out, "heatmap_fz_fx_all.png",
                  f"Fz vs Fx  –  {metric} Heatmap\n{weight_line}",
                  weights=weights_full, weighted=weighted)


# ------------------------------------------- Fx & Fy combination (report slide)
#
# The charter report's "Combination of Fx & Fy" heatmap: share of samples in
# each (Fy, Fx) cell, in N, for one dataset's wheels. Sample count, not
# distance -- the slide reads "% of Occurrences". Fy runs positive-left and Fx
# negative-top, as the reference draws it; empty cells take the colormap's
# darkest colour, so the background reads dark.

COMBO_CMAP = "turbo"


def fx_fy_combination_panel(df, rc, wheel, half_range_n: float, step_n: float):
    """``(H, fy_edges, fx_edges)`` percentages for one wheel, or None."""
    fx = _series(df, rc, wheel, "Fx")
    fy = _series(df, rc, wheel, "Fy")
    if fx is None or fy is None:
        return None
    edges = np.arange(-half_range_n, half_range_n + step_n / 2.0, step_n)
    return _hist2d_pct(fy * 10.0, fx * 10.0, edges, edges)      # daN -> N


def draw_fx_fy_combination(ax, panel, title: str, half_range_n: float, step_n: float,
                           vmax: float, min_label_pct: float = 0.1,
                           tags: bool = False, fontsize: float = 6.0,
                           tick_every: int = 1) -> None:
    import matplotlib as mpl
    cmap = mpl.colormaps[COMBO_CMAP]
    ax.set_facecolor(cmap(0.0))
    ax.set_title(title, fontsize=fontsize + 1, fontweight="bold")
    ax.set_xlabel("Fy (N)", fontsize=fontsize)
    ax.set_ylabel("Fx (N)", fontsize=fontsize)
    ticks = np.arange(-half_range_n, half_range_n + step_n / 2.0, step_n * tick_every)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=fontsize - 1, length=2)
    if panel is not None:
        H, xe, ye = panel
        ax.imshow(H.T, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
                  cmap=cmap, vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
        for i in range(len(xe) - 1):
            for j in range(len(ye) - 1):
                if H[i, j] >= min_label_pct:
                    ax.text((xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2,
                            f"{H[i, j]:.1f}%", color="white", ha="center", va="center",
                            fontsize=fontsize - 1, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "not recorded", ha="center", va="center", color="white",
                fontsize=fontsize + 1, transform=ax.transAxes)
    ax.set_xlim(half_range_n, -half_range_n)       # +Fy on the left
    ax.set_ylim(half_range_n, -half_range_n)       # -Fx at the top
    if tags:
        box = dict(boxstyle="square,pad=0.3", facecolor="#D9D9D9", edgecolor="none")
        kw = dict(transform=ax.transAxes, ha="center", va="center", color="#00A04A",
                  fontsize=fontsize + 1, fontweight="bold", bbox=box, clip_on=False)
        ax.text(-0.36, 0.75, "Acc", rotation=90, **kw)
        ax.text(-0.36, 0.25, "Brake", rotation=90, **kw)
        ax.text(0.16, -0.235, "Left turn", **kw)      # either side of the "Fy (N)" label
        ax.text(0.84, -0.235, "Right turn", **kw)


def fx_fy_combination_figure(panels: dict, slots: list, path, half_range_n: float,
                             step_n: float, vmax: float, size_in: tuple,
                             tags: bool = True, dpi: int = 250) -> None:
    """One dataset's wheels in a grid (2 x 2 for a car's FL/FR/RL/RR), all
    on the one shared ``vmax`` so two datasets' images compare colour for
    colour. ``slots`` lists the wheel labels in grid order; a label with no
    panel is drawn as "not recorded" rather than dropped."""
    n = len(slots)
    ncol = 2 if n <= 4 else 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=size_in, facecolor="white")
    axes = np.atleast_1d(axes).ravel()
    dense = ncol > 2                       # a truck's 3-wide grid: fewer, smaller labels
    for k, (ax, wheel) in enumerate(zip(axes, slots)):
        draw_fx_fy_combination(ax, panels.get(wheel), f"{wheel} - Fx vs Fy",
                               half_range_n, step_n, vmax, tags=tags and k == 0,
                               fontsize=5.0 if dense else 6.0,
                               tick_every=2 if dense else 1)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Fx vs Fy Heatmap - % of Occurrences", fontsize=9, fontweight="bold")
    fig.subplots_adjust(left=0.17 if tags else 0.12, right=0.98, top=0.88, bottom=0.14,
                        wspace=0.55 if dense else 0.32, hspace=0.42)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def fx_fy_combination_colorbar(path, vmax: float, size_in: tuple, dpi: int = 250) -> None:
    """The one colorbar both datasets' heatmaps share."""
    import matplotlib as mpl
    fig = plt.figure(figsize=size_in, facecolor="white")
    cax = fig.add_axes([0.18, 0.05, 0.28, 0.9])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, vmax),
                                            cmap=COMBO_CMAP), cax=cax)
    cb.set_label("Percentage of Occurrences (%)", fontsize=7, fontweight="bold")
    cb.ax.tick_params(labelsize=6)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def generate_heatmaps(df: pd.DataFrame, config: RunConfig) -> None:
    out = config.figures_dir
    rc = config.run_channels
    wheels = list(rc.wheel_groups.keys()) if rc is not None else list(WHEEL_GROUPS.keys())
    wheel_colors = rc.wheel_colors if rc is not None else WHEEL_COLORS

    generate_correlation_heatmaps(df, out, rc, wheels, wheel_colors,
                                   config.sampling_rate, config.tyre)

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