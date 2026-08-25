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

from typing import Optional, Tuple

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


def draw_histogram(ax, values: np.ndarray, bins: np.ndarray,
                   weights: Optional[np.ndarray] = None,
                   color: str = "#00B4D8",
                   bar_is_density: bool = False,
                   boundary: Optional[float] = None,
                   show_kde: bool = True) -> Tuple[np.ndarray, str]:
    """Draw one banded histogram with a scaled KDE curve onto ``ax``.

    ``values``/``weights`` are the raw (already-finite) samples and their
    per-sample weight (distance, say) — pass ``weights=None`` for a plain
    count histogram. ``bar_is_density`` says whether the bars should be a
    normalised density (rainflow's ``density=True`` convention) or the literal
    weighted sum per bin (the force histograms' distance-km / %-distance
    convention); the KDE curve is scaled to match whichever the bars are, so
    it sits directly over them rather than at an unrelated density scale.

    ``boundary`` fixes the one-sided reflection point (default: the first bin
    edge) when :func:`detect_shape` calls this distribution one-sided.

    Returns ``(bar_heights, shape)``.
    """
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
        ax.bar(centers, np.zeros_like(centers), width=widths, color=color)
        return np.zeros_like(centers), "symmetric"

    shape = detect_shape(v, w)
    heights, _ = np.histogram(v, bins=bins, weights=w, density=bar_is_density)
    colors = banded_colors(centers, v, w, shape, color)
    ax.bar(centers, heights, width=widths, color=colors, edgecolor="none",
          alpha=0.92, zorder=2)

    if not show_kde:
        return heights, shape

    x_grid = np.linspace(bins[0], bins[-1], 400)
    if shape == "symmetric":
        mean, _ = _weighted_mean_std(v, w)
        ax.axvline(mean, color="#FFFFFF", linewidth=1.3, alpha=0.85, zorder=4)
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
    ax.plot(x_grid, curve, color=KDE_COLOR, linewidth=1.7, zorder=5)

    return heights, shape
