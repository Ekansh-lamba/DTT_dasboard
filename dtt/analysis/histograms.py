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
    FORCE_RANGES_DAN,
    HIST_BINS,
    SPEED_CANDIDATES,
    FIGURE_DPI,
    RunConfig,
)
from dtt.analysis.plot_style import draw_histogram

logger = logging.getLogger(__name__)

# Local light theme for the force-distribution histograms only — literal
# colours, deliberately *not* sourced from the shared dark `PLOT_COLORS`
# dict, so this module's look can differ from boxplots.py/rainflow.py/
# heatmaps.py/severity.py without touching any of them (each defines its
# own independent `_style_ax`; nothing here is imported elsewhere except
# `_get_speed_weights`, confirmed by grep).
BG       = "#FFFFFF"      # figure background
PANEL    = "#FAFAFA"      # plot area background (very pale grey)
GRID     = "#E3E3E3"      # quiet gridlines
SPINE    = "#CCCCCC"      # thin axis spines
TEXT_PRI = "#333333"      # titles — dark grey, not the old bright white-on-navy
TEXT_SEC = "#555555"      # axis labels/ticks — dark grey


def _try_find_col(df: pd.DataFrame, names: list) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


# Legibility sizes shared by the Histogram and AUC sections (Item B) — bumped
# from the original 8/8/9/11pt, which read as cramped/small next to the
# reference AUC slides' bold, breathing-room layout.
TICK_FONTSIZE  = 10
LABEL_FONTSIZE = 12
TITLE_FONTSIZE = 13
SUPTITLE_FONTSIZE = 13


def _style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT_SEC, labelsize=TICK_FONTSIZE)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE)
        sp.set_linewidth(0.7)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(TEXT_SEC)
    ax.yaxis.label.set_color(TEXT_SEC)
    ax.title.set_color(TEXT_PRI)


_MAX_PLAUSIBLE_MPS = 120.0        # 432 km/h — beyond any test vehicle


def _speed_as_mps(raw: pd.Series) -> tuple:
    """Convert a candidate speed channel to m/s, or return (None, reason).

    Rejects channels that cannot be a road speed. A dead (all-zero) channel is
    the important case: it yields all-zero weights, so every histogram bar is
    zero and the plot renders *blank* — which looks like a loading failure
    rather than the missing-data problem it actually is. Sustained-negative or
    full-scale values (a mis-scaled aux channel) are rejected for the same
    reason: silently weighting by them produces confidently wrong distances.
    """
    v = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size < 8:
        return None, "no finite samples"
    if not np.any(np.abs(finite) > 1e-9):
        return None, "all zero"
    med = float(np.median(finite))
    if med < 0:
        return None, f"median is negative ({med:.1f})"
    # km/h if the typical value is too big to be m/s for a road vehicle
    unit = "km/h" if med > 10.0 else "m/s"
    mps = v / 3.6 if unit == "km/h" else v
    hi = float(np.nanpercentile(np.abs(mps[np.isfinite(mps)]), 99.9))
    if hi > _MAX_PLAUSIBLE_MPS:
        return None, f"implausible peak ({hi * 3.6:.0f} km/h)"
    return (np.nan_to_num(mps), unit), ""


def _get_speed_weights(df: pd.DataFrame, sr: float) -> tuple:
    """Per-sample distance weights (m) from the first *usable* speed channel.

    Returns ``(weights, total_m, unit)``; ``weights`` is None when no channel is
    usable, which tells the caller to weight by sample count instead and say so.
    """
    rejected = []
    seen = set()
    for name in SPEED_CANDIDATES:
        col = _try_find_col(df, [name])
        if col is None or col in seen:
            continue          # candidates alias onto the same column case-insensitively
        seen.add(col)
        result, why = _speed_as_mps(df[col])
        if result is None:
            rejected.append(f"{col} ({why})")
            continue
        mps, unit = result
        weights = mps / float(sr)
        total_m = float(np.nansum(weights))
        if total_m <= 0:
            rejected.append(f"{col} (zero total distance)")
            continue
        logger.info("Distance weighting from '%s' (%s): %.3f km total",
                    col, unit, total_m / 1000.0)
        return weights, total_m, unit
    if rejected:
        logger.warning("No usable speed channel — rejected: %s", "; ".join(rejected))
    else:
        logger.warning("No speed channel found (looked for %s)", ", ".join(SPEED_CANDIDATES))
    logger.warning("Histograms will be weighted by sample count, not distance")
    return None, None, "none"


def _get_force_type(ch: str) -> Optional[str]:
    for ft in ["Fx", "Fy", "Fz"]:
        if ft.lower() in ch.lower():
            return ft
    return None


# Keep the configured axis while it still holds this much of the data.
_RANGE_FIT_FRACTION = 0.98

RANGE_MODES = ("full", "autoscale")


def _percentile_range(finite: np.ndarray) -> tuple:
    lo = float(np.percentile(finite, 0.5))
    hi = float(np.percentile(finite, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        hi = lo + 1.0
    pad = 0.02 * (hi - lo)
    return (lo - pad, hi + pad)


def _axis_range(vals: np.ndarray, force_type: Optional[str], ch: str = "",
                range_mode: str = "full") -> Optional[tuple]:
    """Pick the histogram x-range for ``vals``.

    ``range_mode="full"`` (default) uses ``FORCE_RANGES_DAN`` — tuned for a
    passenger car (Fz 100-1000 daN) — while it actually holds the data; a
    heavier vehicle, a different WFT calibration or a truck axle can sit an
    order of magnitude outside it, and a fixed axis then puts the *entire*
    distribution off-plot, so a robust percentile range is used instead once
    the configured range no longer fits. ``range_mode="autoscale"`` always
    uses that percentile range (P0.5-P99.5 with a 2% pad), even when the
    configured range technically contains the data — real WFT data often
    sits in a narrow band near zero well inside the sensor's full range, and
    the full-range axis then crushes the actual shape into the middle of the
    plot.
    """
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return FORCE_RANGES_DAN.get(force_type) if force_type else None

    if range_mode == "autoscale":
        return _percentile_range(finite)

    cfg = FORCE_RANGES_DAN.get(force_type) if force_type else None
    if cfg:
        inside = float(np.mean((finite >= cfg[0]) & (finite <= cfg[1])))
        if inside >= _RANGE_FIT_FRACTION:
            return cfg
        logger.warning(
            "%s: only %.1f%% of samples fall in the configured %s range %s - "
            "using the data's own range instead", ch or force_type,
            100.0 * inside, force_type, cfg)
    return _percentile_range(finite)


def weighted_plot_series(vals, weights, total_m, mode: str) -> tuple:
    """``(w_plot, ylabel, total_label)`` for either bar heights or a KDE
    curve's weights — shared by the Histogram section (this module) and the
    AUC section (:mod:`dtt.analysis.auc_plots`) so both agree on what
    "distance" vs "percentage" mode means instead of each computing it
    separately.

    Without a usable speed channel the values are sample counts, not
    distance — labelled as such rather than printing "Distance (km)" over a
    plain count.
    """
    weighted = weights is not None and bool(total_m) and total_m > 0
    if mode == "distance":
        w_plot = weights / 1000.0 if weighted else np.ones(len(vals))
        ylabel = "Distance (km)" if weighted else "Samples (no speed data)"
        total_label = f"{total_m / 1000.0:.3f} km" if weighted else f"{len(vals):,} pts"
    else:
        if weighted:
            w_plot = (weights / total_m) * 100.0
            ylabel = "% Distance"
        else:
            w_plot = np.ones(len(vals)) / len(vals) * 100.0
            ylabel = "% Samples (no speed data)"
        total_label = f"{np.nansum(w_plot):.1f} %"
    return w_plot, ylabel, total_label


def _plot_single_histogram(ax, vals, weights, bins, xlim, ch, mode, total_m, col):
    _style_ax(ax)
    w_plot, ylabel, total_label = weighted_plot_series(vals, weights, total_m, mode)

    # Item A: this is the *Histogram* section — actual binned bars, no KDE
    # curve (the curve lives on its own in the AUC section, generated by
    # dtt/analysis/auc_plots.py from the same weights/bins/xlim so the two
    # sections agree on what they're showing).
    draw_histogram(ax, vals, bins, weights=w_plot, color=col, style="banded", show_kde=False)
    if xlim:
        ax.set_xlim(xlim)
    ax.set_xlabel("Force (daN)", fontsize=LABEL_FONTSIZE, fontweight="bold", color=TEXT_SEC)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE, fontweight="bold", color=TEXT_SEC)
    ax.set_title(f"{ch}  [{mode}]  {total_label}", fontsize=TITLE_FONTSIZE,
                fontweight="bold", color=TEXT_PRI)


def generate_histograms(df: pd.DataFrame, config: RunConfig, range_mode: str = "full") -> None:
    """``range_mode``: ``"full"`` (default, matches prior output) uses the
    configured sensor range; ``"autoscale"`` clips to where the data actually
    sits (P0.5-P99.5 with a small pad). See :func:`_axis_range`.
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
            fig, axes = plt.subplots(1, len(present), figsize=(6 * len(present), 5), facecolor=BG)
            if len(present) == 1:
                axes = [axes]
            fig.suptitle(
                f"{wheel}  –  Force Distribution  ({mode.title()})\n"
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

                _plot_single_histogram(ax, vals, w, bins, xlim, ch, mode, total_m, col)

            fig.tight_layout(rect=[0, 0, 1, 0.92])
            fig.subplots_adjust(wspace=0.32)   # room for larger tick/axis-label text
            fname = out / f"hist_{mode}_{wheel}{suffix}.png"
            fig.savefig(fname, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=BG)
            plt.close(fig)
            logger.info("Saved histogram: %s", fname.name)

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
            fig, ax = plt.subplots(figsize=(7, 4.6), facecolor=BG)
            _plot_single_histogram(ax, vals, w, bins, xlim, ch, mode, total_m, col)
            fig.tight_layout()
            fname = out / f"hist_{mode}_{ch}{suffix}.png"
            fig.savefig(fname, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=BG)
            plt.close(fig)
