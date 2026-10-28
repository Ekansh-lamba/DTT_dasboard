"""Shared histogram styling: banded bars + a scaled KDE curve.

One look, one place, so every histogram in the app — force distributions,
rainflow range distributions — reads the same way: bars coloured in bands by
distance from the centre of the distribution, dark in the middle fading to
light at the edges, with a smooth curve traced over them.

The reference look is a symmetric normal-fit bell, but WFT load data is often
skewed and a rainflow range histogram is one-sided by construction (it starts
at 0). Forcing a symmetric bell onto either would draw a shape that does not
match the data, so :func:`draw_histogram` detects which kind it has and
adapts: a roughly-symmetric distribution gets sigma-bands, a mean line, and a
plain KDE; a skewed or hard-bounded one gets quantile-position bands, no mean
line, and a boundary-reflected KDE that does not droop toward zero right at
the bound.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.colors import to_rgb

from dtt.analysis.auc import kde_curve
from dtt.config import PLOT_COLORS

KDE_COLOR = PLOT_COLORS["danger"]      # the reference's red curve
N_BANDS = 5
SIGMA_EDGES = (0.5, 1.0, 1.5, 2.0)     # 5 bands: <0.5, 0.5-1, 1-1.5, 1.5-2, >=2 sigma

_ONE_SIDED_SKEW = 1.0                  # |skew| beyond this reads as one-sided
_BOUNDARY_FRAC = 0.10                  # low edge within this fraction of range -> hard-bounded


def _weighted_mean_std(v: np.ndarray, w: Optional[np.ndarray]) -> Tuple[float, float]:
    if w is None or w.sum() <= 0:
        m = float(np.mean(v))
        s = float(np.std(v))
    else:
        w = w / w.sum()
        m = float(np.sum(w * v))
        s = float(np.sqrt(max(0.0, np.sum(w * (v - m) ** 2))))
    return m, (s if s > 0 else 1.0)


def detect_shape(values: np.ndarray, weights: Optional[np.ndarray] = None) -> str:
    """``"symmetric"`` or ``"one_sided"`` for this distribution.

    Two independent signals, either one is enough: a hard lower bound (the low
    edge of the data sits close to its own range's start — the rainflow-range
    case, which starts at exactly 0) or a strong skew (the tail signature a
    forced normal bell would misdraw).
    """
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        finite &= np.isfinite(w)
    v = v[finite]
    w = w[finite] if w is not None else None
    if v.size < 8:
        return "symmetric"

    lo, hi = float(np.min(v)), float(np.max(v))
    rng = (hi - lo) or 1.0
    near_bound = lo >= 0.0 and (lo / rng) < _BOUNDARY_FRAC

    mean, std = _weighted_mean_std(v, w)
    if w is None:
        skew = float(np.mean(((v - mean) / std) ** 3))
    else:
        skew = float(np.sum((w / w.sum()) * ((v - mean) / std) ** 3))

    return "one_sided" if (near_bound or abs(skew) > _ONE_SIDED_SKEW) else "symmetric"


def _band_ramp(base_hex: str, n: int = N_BANDS) -> list:
    """``n`` colours from a dark version of ``base_hex`` (band 0) to a light
    one (band ``n-1``), so the banding stays in the channel's own colour
    family instead of a fixed unrelated palette.
    """
    r, g, b = to_rgb(base_hex)
    dark = (r * 0.55, g * 0.55, b * 0.55)
    light = (r + (1 - r) * 0.82, g + (1 - g) * 0.82, b + (1 - b) * 0.82)
    return [tuple(dark[i] + (light[i] - dark[i]) * (k / (n - 1)) for i in range(3))
            for k in range(n)]


def banded_colors(bin_centers: np.ndarray, values: np.ndarray,
                  weights: Optional[np.ndarray], shape: str, base_color: str
                  ) -> list:
    """One colour per bin: darkest at the centre, lightest at the edges.

    ``symmetric`` bands by |z| = |x - mean| / std against the fixed edges
    (0.5, 1, 1.5, 2 sigma). ``one_sided`` bands by quantile position instead —
    "distance from the centre" is not meaningful for a distribution that only
    goes one way, so position along the data's own empirical CDF stands in
    for it, darkest where the data is bounded/densest.
    """
    ramp = _band_ramp(base_color)
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    v = v[finite]
    if v.size == 0:
        return [ramp[-1]] * len(bin_centers)

    if shape == "symmetric":
        w = weights[finite] if weights is not None else None
        mean, std = _weighted_mean_std(v, w)
        z = np.abs((np.asarray(bin_centers, dtype=float) - mean) / std)
        band = np.searchsorted(SIGMA_EDGES, z)
    else:
        sv = np.sort(v)
        pct = np.searchsorted(sv, bin_centers, side="right") / sv.size
        # quantile distance from the boundary (low end), not from a centre
        band = np.clip((pct * N_BANDS).astype(int), 0, N_BANDS - 1)

    band = np.clip(band, 0, N_BANDS - 1)
    return [ramp[int(k)] for k in band]


_SOFT_MARKER_COLOR = "#4A4A4A"          # dark-grey mean/percentile line, visible on a light bg
_SOFT_FILL_ALPHA = 0.25
_SOFT_LINE_WIDTH = 1.3
_BANDED_LINE_WIDTH = 1.7


def draw_histogram(ax, values: np.ndarray, bins: np.ndarray,
                   weights: Optional[np.ndarray] = None,
                   color: str = "#00B4D8",
                   bar_is_density: bool = False,
                   boundary: Optional[float] = None,
                   show_kde: bool = True,
                   style: str = "banded") -> Tuple[np.ndarray, str]:
    """Draw one histogram with a scaled KDE curve onto ``ax``.

    ``values``/``weights`` are the raw (already-finite) samples and their
    per-sample weight (distance, say) — pass ``weights=None`` for a plain
    count histogram. ``bar_is_density`` says whether the bars should be a
    normalised density (rainflow's ``density=True`` convention) or the literal
    weighted sum per bin (the force histograms' distance-km / %-distance
    convention); the KDE curve is scaled to match whichever the bars are, so
    it sits directly over them rather than at an unrelated density scale.

    ``boundary`` fixes the one-sided reflection point (default: the first bin
    edge) when :func:`detect_shape` calls this distribution one-sided.

    ``style``: ``"banded"`` (default, unchanged from before) draws bars
    coloured in dark-centre-to-light-edge bands with a fixed red KDE line —
    this is what rainflow's range-distribution panel uses and keeps using.
    ``"soft"`` draws no bars at all: a single soft low-alpha fill under a
    thin line, both in ``color`` — the light force-distribution-histogram
    look. Only the drawing changes between the two; the histogram/KDE maths
    above this point are identical either way.

    Returns ``(bar_heights, shape)`` — ``bar_heights`` is still computed via
    ``np.histogram`` under ``style="soft"`` even though no bars are drawn, so
    callers that inspect the return value see the same shape of data.
    """
    if style not in ("banded", "soft"):
        raise ValueError(f'style must be "banded" or "soft", got {style!r}')

    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        finite &= np.isfinite(w)
    v = v[finite]
    w = w[finite] if w is not None else None

    centers = (bins[:-1] + bins[1:]) / 2.0
    widths = np.diff(bins)
    if v.size == 0:
        if style == "banded":
            ax.bar(centers, np.zeros_like(centers), width=widths, color=color)
        return np.zeros_like(centers), "symmetric"

    shape = detect_shape(v, w)
    heights, _ = np.histogram(v, bins=bins, weights=w, density=bar_is_density)
    if style == "banded":
        colors = banded_colors(centers, v, w, shape, color)
        ax.bar(centers, heights, width=widths, color=colors, edgecolor="none",
              alpha=0.92, zorder=2)

    if not show_kde:
        return heights, shape

    x_grid = np.linspace(bins[0], bins[-1], 400)
    marker_color = "#FFFFFF" if style == "banded" else _SOFT_MARKER_COLOR
    if shape == "symmetric":
        mean, _ = _weighted_mean_std(v, w)
        ax.axvline(mean, color=marker_color, linewidth=1.3, alpha=0.85, zorder=4)
        density = kde_curve(v, x_grid, weights=w)
    else:
        # Reflection is only correct against a *real* physical bound (the
        # caller passes `boundary` when it knows one, e.g. rainflow ranges
        # can't go below 0). A one-sided classification can also come purely
        # from skew with no such bound -- reflecting around the plotted
        # axis edge then would be reflecting around an arbitrary point (just
        # wherever percentile-trimming happened to start the axis) and
        # invents density that isn't there. Without a known boundary, fall
        # back to a plain (unreflected) KDE: it still follows the actual
        # skewed shape with no symmetry assumption, it just doesn't apply
        # the edge correction there is no real edge to correct.
        density = kde_curve(v, x_grid, weights=w, reflect_boundary=boundary)

    # Scale the density curve onto the bars' own scale. `density` integrates
    # to 1 against `w` (or against the sample count when w is None -- that's
    # what kde_curve normalises by either way), so multiplying by bin width
    # gives per-bin mass and by that same total gives the bars' own units
    # (km / % / density) -- `bar_is_density` bars are already unit-area, so
    # they need no further scaling.
    if bar_is_density:
        curve = density
    else:
        bin_width = float(widths[0]) if widths.size else 1.0
        total = float(w.sum()) if w is not None else float(v.size)
        curve = density * bin_width * total

    if style == "soft":
        ax.fill_between(x_grid, curve, alpha=_SOFT_FILL_ALPHA, color=color, zorder=3)
        ax.plot(x_grid, curve, color=color, linewidth=_SOFT_LINE_WIDTH, zorder=5)
    else:
        ax.plot(x_grid, curve, color=KDE_COLOR, linewidth=_BANDED_LINE_WIDTH, zorder=5)

    return heights, shape


# --------------------------------------------------------------------------
# Two-run comparison styling (study-to-study, RF Compare)
#
# Unlike auc.py's compare_distributions/auc_grid (built around a fixed
# "reference vs current" pair), these two are for comparing two independent,
# equally-weighted finished studies -- callers name each run explicitly
# rather than the plot assuming a temporal order. Cross-position overlays
# (e.g. RL of run A vs FL of run B) need no special handling here: the
# caller just extracts values from whatever channel it likes for each side.
# --------------------------------------------------------------------------

_STATS_COLS = ("Min", "P5", "Q1", "Median", "Mean", "Q3", "P95", "Max")


def _finite(values) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return v[np.isfinite(v)]


def _plot_span(*arrays: np.ndarray) -> Tuple[float, float]:
    both = np.concatenate(arrays)
    lo = float(np.percentile(both, 0.5))
    hi = float(np.percentile(both, 99.5))
    if not (hi > lo):
        lo, hi = float(both.min()), float(both.max()) or 1.0
    return lo, hi


def draw_comparison_kde(ax, values_a: np.ndarray, values_b: Optional[np.ndarray],
                        label_a: str, label_b: str,
                        color_a: str, color_b: str,
                        weights_a: Optional[np.ndarray] = None,
                        weights_b: Optional[np.ndarray] = None,
                        xlim: Optional[Tuple[float, float]] = None,
                        n: int = 500,
                        p5_in_legend: bool = False,
                        fill_alpha: float = 0.28,
                        line_width: float = 1.8,
                        legend_fontsize: float = 7) -> dict:
    """Filled KDE overlay of two runs on one axis, with dashed P5/P95 lines
    per run and a legend naming each run. Returns the P5/P95 values used, so
    a caller building a Miner-ratio-style summary line can reuse them.

    ``weights_a``/``weights_b`` (e.g. per-sample distance) make each curve a
    weighted density via :func:`dtt.analysis.auc.kde_curve` -- pass ``None``
    for a plain sample-count KDE.

    ``values_b=None`` draws run A alone, in run A's style -- a single-study
    report uses the same plot. ``p5_in_legend`` names every percentile line
    with its value (``"EV P5: -40"``), P5s before P95s, as the charter report
    does; off, the legend is exactly as before.
    """
    a = _finite(values_a)
    b = None if values_b is None else _finite(values_b)
    if a.size < 2 or (b is not None and b.size < 2):
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes)
        return {}

    lo, hi = xlim if xlim is not None else _plot_span(a, *([b] if b is not None else []))
    x = np.linspace(lo, hi, n)

    runs = [("a", a, weights_a, label_a, color_a, "-")]
    if b is not None:
        runs.append(("b", b, weights_b, label_b, color_b, "--"))
    curves = [kde_curve(v, x, weights=w) for _, v, w, *_ in runs]
    for k, (_, _, _, _, color, _) in zip(curves, runs):
        ax.fill_between(x, k, alpha=fill_alpha, color=color)
    for k, (_, _, _, label, color, style) in zip(curves, runs):
        ax.plot(x, k, color=color, linewidth=line_width, linestyle=style, label=label)

    out = {}
    for key, v, _, label, color, _ in runs:
        p5, p95 = float(np.percentile(v, 5)), float(np.percentile(v, 95))
        ax.axvline(p5, color=color, linewidth=1.1, linestyle=":",
                   label=f"{label} P5: {p5:.0f}" if p5_in_legend else None)
        ax.axvline(p95, color=color, linewidth=1.6, linestyle="--",
                   label=f"{label} P95: {p95:.0f}" if p5_in_legend else f"{label} P95 {p95:.0f}")
        out[f"p5_{key}"], out[f"p95_{key}"] = p5, p95

    if p5_in_legend:
        handles, labels = ax.get_legend_handles_labels()
        rank = lambda l: 2 if " P95: " in l else (1 if " P5: " in l else 0)   # noqa: E731
        order = sorted(range(len(labels)), key=lambda i: rank(labels[i]))
        ax.legend([handles[i] for i in order], [labels[i] for i in order],
                  fontsize=legend_fontsize, facecolor=_LEGEND_BG, edgecolor=_LEGEND_BG,
                  labelcolor="white", framealpha=0.95, loc="upper right",
                  handlelength=1.6, borderpad=0.4, labelspacing=0.25)
    else:
        ax.legend(fontsize=legend_fontsize, facecolor=PLOT_COLORS["bg"], labelcolor="white",
                  framealpha=0.85)

    return out


_LEGEND_BG = "#263445"            # the charter plots' dark slate legend box
_ZONE_GREEN = "#3AA76D"
_EXCEED_RED = "#D9534F"
_REFERENCE_GREY = "#A6A6A6"


def draw_reference_kde(ax, values_cand: np.ndarray, values_ref: Optional[np.ndarray],
                       label_cand: str, label_ref: str, color_cand: str,
                       xlim: Optional[Tuple[float, float]] = None, n: int = 500,
                       line_width: float = 1.4, legend_fontsize: float = 5) -> dict:
    """A candidate run against a reference baseline -- the CV charter slide.

    The reference is drawn as a grey dashed curve and the band between its
    own P5 and P95 is shaded green ("normal zone"); wherever the candidate
    runs past the reference P95 its area is shaded red. Vertical rules: the
    reference P5 (green, dotted) and P95 (red, dashed), and the candidate P95
    (dashed, in the candidate's colour). Plain sample-count KDEs, the same
    kernel as :func:`draw_comparison_kde`.

    ``values_ref=None`` draws the candidate alone with its own P5/P95.
    Returns the percentiles used.
    """
    from matplotlib.lines import Line2D

    c = _finite(values_cand)
    r = None if values_ref is None else _finite(values_ref)
    if c.size < 2 or (r is not None and r.size < 2):
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes)
        return {}
    lo, hi = xlim if xlim is not None else _plot_span(c, *([r] if r is not None else []))
    x = np.linspace(lo, hi, n)
    kc = kde_curve(c, x)
    p5c, p95c = float(np.percentile(c, 5)), float(np.percentile(c, 95))
    out = {"p5_cand": p5c, "p95_cand": p95c}

    if r is None:
        ax.fill_between(x, kc, color=color_cand, alpha=0.25, linewidth=0)
        ax.plot(x, kc, color=color_cand, linewidth=line_width)
        ax.axvline(p5c, color=color_cand, linewidth=1.0, linestyle=":")
        ax.axvline(p95c, color=color_cand, linewidth=1.2, linestyle="--")
        handles = [Line2D([], [], color=color_cand, linewidth=line_width),
                   Line2D([], [], color=color_cand, linestyle=":"),
                   Line2D([], [], color=color_cand, linestyle="--")]
        labels = [label_cand, f"{label_cand} P5: {p5c:.0f}", f"{label_cand} P95: {p95c:.0f}"]
    else:
        kr = kde_curve(r, x)
        p5r, p95r = float(np.percentile(r, 5)), float(np.percentile(r, 95))
        out.update(p5_ref=p5r, p95_ref=p95r)
        top = np.maximum(kc, kr)
        ax.fill_between(x, kr, color=_REFERENCE_GREY, alpha=0.18, linewidth=0)
        ax.fill_between(x, kc, where=x <= p5r, color=color_cand, alpha=0.15, linewidth=0)
        zone = ax.fill_between(x, top, where=(x >= p5r) & (x <= p95r),
                               color=_ZONE_GREEN, alpha=0.35, linewidth=0)
        exceed = ax.fill_between(x, kc, where=x >= p95r, color=_EXCEED_RED,
                                 alpha=0.6, linewidth=0)
        ref_line, = ax.plot(x, kr, color=_REFERENCE_GREY, linewidth=line_width, linestyle="--")
        cand_line, = ax.plot(x, kc, color=color_cand, linewidth=line_width)
        p5_line = ax.axvline(p5r, color=_ZONE_GREEN, linewidth=1.0, linestyle=":")
        p95_line = ax.axvline(p95r, color=_EXCEED_RED, linewidth=1.2, linestyle="--")
        cand95 = ax.axvline(p95c, color=color_cand, linewidth=1.2, linestyle="--")
        handles = [zone, exceed, ref_line, cand_line, p5_line, p95_line, cand95]
        labels = [f"{label_ref} normal zone (P5-P95)", f"{label_cand} exceeds {label_ref} P95",
                  f"{label_ref} (reference)", label_cand, f"{label_ref} P5: {p5r:.0f}",
                  f"{label_ref} P95: {p95r:.0f}", f"{label_cand} P95: {p95c:.0f}"]
    ax.legend(handles, labels, fontsize=legend_fontsize, facecolor=_LEGEND_BG,
              edgecolor=_LEGEND_BG, labelcolor="white", framealpha=0.95, loc="upper right",
              handlelength=1.6, borderpad=0.4, labelspacing=0.25)
    return out


def stats_strip_rows(series: Sequence[Tuple[str, np.ndarray]],
                     weights: Optional[Sequence[Optional[np.ndarray]]] = None,
                     decimals: int = 1,
                     ) -> Tuple[List[str], List[List[str]]]:
    """``(header, rows)`` for the Min/P5/Q1/Median/Mean/Q3/P95/Max strip, one
    row per ``(label, values)`` pair. Matplotlib-free so it is independently
    testable; :func:`draw_stats_strip` renders it under a plot.

    ``weights`` (one array per series, or None) makes the percentiles and mean
    distance-weighted, so the strip agrees with a weighted curve drawn above
    it. Min and Max are the same either way -- a weight changes how much a
    sample counts, not whether it happened. Omitted, the output is unchanged.

    ``decimals`` is display rounding only (the charter report prints whole
    numbers); the statistics are the same either way.
    """
    from dtt.analysis.auc import weighted_percentile

    header = ["Run", *_STATS_COLS]
    rows: List[List[str]] = []
    for i, (label, vals) in enumerate(series):
        w = weights[i] if weights is not None and i < len(weights) else None
        v = np.asarray(vals, dtype=float)
        if w is not None:
            w = np.asarray(w, dtype=float)[:v.size]
            v = v[:w.size]
            ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
            v, w = v[ok], w[ok]
        else:
            v = v[np.isfinite(v)]
        if v.size == 0:
            rows.append([label] + ["—"] * len(_STATS_COLS))
            continue
        if w is None:
            stats = [v.min(), np.percentile(v, 5), np.percentile(v, 25), np.median(v),
                     v.mean(), np.percentile(v, 75), np.percentile(v, 95), v.max()]
        else:
            q = lambda p: weighted_percentile(v, w, p)          # noqa: E731
            stats = [v.min(), q(5), q(25), q(50), float(np.average(v, weights=w)),
                     q(75), q(95), v.max()]
        rows.append([label] + [f"{s:.{decimals}f}" for s in stats])
    return header, rows


def stats_strip_lines(header: List[str], rows: List[List[str]]) -> List[str]:
    """``EV | Min:  -244 | P5:   -40 | ...`` -- one fixed-width line per row,
    columns aligned across rows, as the charter report's strips read."""
    lab_w = max(len(r[0]) for r in rows)
    widths = [max(len(r[i]) for r in rows) for i in range(1, len(header))]
    return [" | ".join([r[0].ljust(lab_w)] +
                       [f"{h}: {v.rjust(w)}" for h, v, w in zip(header[1:], r[1:], widths)])
            for r in rows]


_STATS_STRIP_THEMES = {
    # dark: the original navy/cyan theme, for the dark-background comparison
    # plots (rainflow, heatmaps, comparison_page.py's two-run overlay).
    "dark": dict(header_bg=PLOT_COLORS["entry_bg"], header_text=PLOT_COLORS["btn_bg"],
                edge=PLOT_COLORS["panel"],
                row_bg=(PLOT_COLORS["entry_bg"], PLOT_COLORS["panel"]),
                row_text="white", default_label_color="white"),
    # light: for the white-background force histograms/AUC plots -- a pale
    # grey table with dark text, matching that section's own light theme
    # rather than the app's dark chrome.
    "light": dict(header_bg="#EDEDED", header_text="#333333",
                 edge="#CCCCCC",
                 row_bg=("#FFFFFF", "#F5F5F5"), row_text="#333333",
                 default_label_color="#333333"),
}


def draw_stats_strip(ax, series: Sequence[Tuple[str, np.ndarray]],
                     colors: Optional[Sequence[str]] = None,
                     theme: str = "dark", fontsize: float = 7,
                     bbox: Optional[List[float]] = None,
                     weights: Optional[Sequence[Optional[np.ndarray]]] = None,
                     layout: str = "table", decimals: int = 1,
                     ) -> None:
    """Render the Min/P5/Q1/Median/Mean/Q3/P95/Max strip as a small table
    directly under ``ax`` — the "more informative" element the reference
    EV-vs-IC slides carry below their comparison plots, standardised here so
    every plot in the app that wants one gets the same layout/columns.

    ``theme``: ``"dark"`` (default, unchanged from before) matches the app's
    navy chrome; ``"light"`` matches the white-background force-histogram/AUC
    style instead of clashing with it.

    ``layout="text"`` is the charter report's strip instead: ``ax`` is an
    axes of its own (turned off here), filled by a grey rounded panel of
    monospaced ``label | Min: | P5: | ...`` lines. ``theme``/``colors``/
    ``bbox`` do not apply to it.
    """
    header, rows = stats_strip_rows(series, weights=weights, decimals=decimals)
    if not rows:
        return
    if layout == "text":
        ax.axis("off")
        ax.text(0.5, 0.5, "\n".join(stats_strip_lines(header, rows)),
                transform=ax.transAxes, ha="center", va="center",
                family="monospace", fontsize=fontsize, color="#262626",
                linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.45", facecolor="#E9E9E9",
                          edgecolor="#BFBFBF", linewidth=0.6))
        return
    th = _STATS_STRIP_THEMES[theme]
    tbl = ax.table(cellText=rows, colLabels=header, loc="bottom",
                   bbox=bbox or [0.0, -0.46, 1.0, 0.32], cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    row_bg_odd, row_bg_even = th["row_bg"]
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(th["edge"])
        if r == 0:
            cell.set_facecolor(th["header_bg"])
            cell.get_text().set_color(th["header_text"])
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(row_bg_odd if r % 2 else row_bg_even)
            row_color = colors[r - 1] if colors and (r - 1) < len(colors) else th["default_label_color"]
            cell.get_text().set_color(row_color if c == 0 else th["row_text"])
