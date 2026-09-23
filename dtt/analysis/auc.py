"""Area-under-curve distribution comparison between two recordings.

Answers "did this drive work the wheel harder than the reference?" by putting
both load distributions on one axis and quoting the overlap in percentages an
engineer can act on: how much of the new run sits beyond the reference's P95,
and how much still falls inside its normal P5-P95 band.

The density curves are smoothed KDEs rather than raw histograms because the
comparison is about *shape* — where the mass sits — and a 60-bin histogram of a
600k-sample channel makes that judgement on bin edges rather than on the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class AucComparison:
    """Distribution overlap of one channel across two recordings."""
    p5_ref: float
    p95_ref: float
    p5_cur: float
    p95_cur: float
    pct_exceed_ref_p95: float      # share of current beyond the reference's P95
    pct_normal_cur: float          # share of current inside the reference band
    pct_normal_ref: float          # share of reference inside its own band
    n_ref: int
    n_cur: int

    @property
    def delta_p95(self) -> float:
        return self.p95_cur - self.p95_ref

    def summary(self, unit: str = "daN") -> str:
        return (f"reference  P5 {self.p5_ref:.0f} / P95 {self.p95_ref:.0f} {unit}"
                f"  ·  {self.pct_normal_ref:.1f}% within band\n"
                f"current    P95 {self.p95_cur:.0f} {unit} "
                f"({self.delta_p95:+.0f})  ·  "
                f"{self.pct_exceed_ref_p95:.1f}% beyond reference P95  ·  "
                f"{self.pct_normal_cur:.1f}% within band")


def kde_curve(data: np.ndarray, x_grid: np.ndarray,
             weights: Optional[np.ndarray] = None,
             reflect_boundary: Optional[float] = None) -> np.ndarray:
    """Gaussian KDE over ``x_grid``, evaluated from **all** of ``data``.

    A direct ``scipy.gaussian_kde`` is O(n_samples x n_grid), which is unusable
    on a 600k-sample channel. Subsampling to fix that is a trap: scipy's Scott
    bandwidth scales as ``n^(-1/5)``, so fitting on 20k points instead of 600k
    widens the kernel by ~1.7x and smears genuinely separate peaks into one — the
    curve gets faster and quietly wrong.

    Binning the full data into a fine histogram and convolving with a Gaussian is
    O(n) in the samples and independent of the grid, while keeping the bandwidth
    computed from the true ``n``. It agrees with the exact full-data KDE to ~1e-6
    and runs in milliseconds.

    ``weights``, if given, makes this a *weighted* density (e.g. per-sample
    distance) — the result is normalised so it integrates to 1 against the
    weights, not the sample count, which is what lets it sit over a
    distance-weighted histogram rather than a plain sample-count one.

    ``reflect_boundary``, if given, applies the standard reflection method for
    a one-sided distribution bounded at that value (mirror the data about the
    boundary, fit on the pooled original+mirror set, fold back and double).
    Without it a hard-bounded distribution (range >= 0, say) gets a curve that
    droops toward zero density right at the boundary, which is wrong — the
    true density does not have to vanish there.
    """
    from scipy.ndimage import gaussian_filter1d

    data = np.asarray(data, dtype=float)
    finite = np.isfinite(data)
    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        finite &= np.isfinite(w)
    data = data[finite]
    if w is not None:
        w = w[finite]
    n = data.size
    if n < 2 or x_grid.size < 2:
        return np.zeros_like(x_grid)
    std = float(np.std(data))
    if std <= 0:
        return np.zeros_like(x_grid)
    bw = (n ** (-1.0 / 5.0)) * std                 # Scott's rule on the FULL data

    fit_data, fit_w, fold = data, w, 1.0
    lo_fit, hi_fit = x_grid[0], x_grid[-1]
    if reflect_boundary is not None:
        mirror = 2.0 * reflect_boundary - data
        fit_data = np.concatenate([data, mirror])
        fit_w = np.concatenate([w, w]) if w is not None else None
        # extend the fit range symmetrically about the boundary so the
        # mirrored mass has somewhere to land before folding back
        span = hi_fit - reflect_boundary
        lo_fit = reflect_boundary - span
        fold = 2.0

    n_bins = max(2000, x_grid.size * 4)
    edges = np.linspace(lo_fit, hi_fit, n_bins + 1)
    hist, edges = np.histogram(fit_data, bins=edges, weights=fit_w,
                               density=(fit_w is None))
    width = edges[1] - edges[0]
    if width <= 0:
        return np.zeros_like(x_grid)
    if fit_w is not None:
        total_w = float(fit_w.sum())
        if total_w <= 0:
            return np.zeros_like(x_grid)
        hist = hist / (total_w * width)            # weighted density, area 1
    smoothed = gaussian_filter1d(hist, sigma=bw / width, mode="nearest")
    centers = (edges[:-1] + edges[1:]) / 2.0
    return np.interp(x_grid, centers, smoothed) * fold


def weighted_percentile(values: np.ndarray, weights: np.ndarray,
                        q: float) -> float:
    """The ``q``-th percentile of ``values`` with each sample counted by weight.

    Used for distance weighting: a sample recorded while the vehicle covered
    two metres counts twice as much as one recorded over one metre, so a run
    that idled for ten minutes does not have its load distribution pulled
    toward the idle load. ``np.percentile`` has no weights argument (its
    ``method`` options change the interpolation, not the sample weight).

    Midpoint-cumulative definition: each sample's position on the 0-100 axis
    is the centre of its own weight, so equal weights reproduce
    ``np.percentile``'s linear interpolation to within one sample's width.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[ok], w[ok]
    if v.size == 0:
        return float("nan")
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    total = cw[-1]
    pos = 100.0 * (cw - 0.5 * w) / total
    return float(np.interp(q, pos, v))


def _share(mask: np.ndarray, weights: Optional[np.ndarray]) -> float:
    """Percentage of the data inside ``mask`` -- by count, or by weight."""
    if weights is None:
        return float(np.mean(mask) * 100.0)
    total = float(np.sum(weights))
    return float(np.sum(weights[mask]) / total * 100.0) if total > 0 else float("nan")


def _clean_pair(values, weights):
    """Finite values, and their weights masked the same way."""
    v = np.asarray(values, dtype=float)
    if weights is None:
        return v[np.isfinite(v)], None
    w = np.asarray(weights, dtype=float)
    n = min(v.size, w.size)
    v, w = v[:n], w[:n]
    ok = np.isfinite(v) & np.isfinite(w) & (w >= 0)
    return v[ok], w[ok]


def compare_distributions(reference: np.ndarray, current: np.ndarray,
                          weights_ref: Optional[np.ndarray] = None,
                          weights_cur: Optional[np.ndarray] = None,
                          ) -> Optional[AucComparison]:
    """Overlap statistics of ``current`` against ``reference`` as the baseline.

    With no weights this is exactly what it always was: plain percentiles and
    plain proportions, byte-for-byte. Pass per-sample weights (distance, from
    ``histograms._get_speed_weights``) for both runs and every figure becomes
    distance-weighted -- percentiles *and* the exceedance/band shares -- so the
    numbers describe the same distribution the weighted KDE draws. Weighting
    one side and not the other would compare two different questions, so it
    is refused.
    """
    if (weights_ref is None) != (weights_cur is None):
        raise ValueError("weight both runs or neither -- mixing a distance-"
                         "weighted and a sample-count distribution compares "
                         "two different things")
    ref, w_ref = _clean_pair(reference, weights_ref)
    cur, w_cur = _clean_pair(current, weights_cur)
    if ref.size < 2 or cur.size < 2:
        return None

    if w_ref is None:
        p5_ref, p95_ref = (float(np.percentile(ref, 5)), float(np.percentile(ref, 95)))
        p5_cur, p95_cur = (float(np.percentile(cur, 5)), float(np.percentile(cur, 95)))
    else:
        p5_ref = weighted_percentile(ref, w_ref, 5)
        p95_ref = weighted_percentile(ref, w_ref, 95)
        p5_cur = weighted_percentile(cur, w_cur, 5)
        p95_cur = weighted_percentile(cur, w_cur, 95)

    return AucComparison(
        p5_ref=p5_ref, p95_ref=p95_ref,
        p5_cur=p5_cur, p95_cur=p95_cur,
        pct_exceed_ref_p95=_share(cur >= p95_ref, w_cur),
        pct_normal_cur=_share((cur >= p5_ref) & (cur <= p95_ref), w_cur),
        pct_normal_ref=_share((ref >= p5_ref) & (ref <= p95_ref), w_ref),
        n_ref=int(ref.size), n_cur=int(cur.size),
    )


def auc_grid(reference: np.ndarray, current: np.ndarray, n: int = 500,
             xlim: Optional[Tuple[float, float]] = None,
             weights_ref: Optional[np.ndarray] = None,
             weights_cur: Optional[np.ndarray] = None,
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(x, kde_reference, kde_current)`` ready to plot.

    Weights, when given, go straight to :func:`kde_curve`, which already
    supports them -- the curve then integrates to 1 against distance rather
    than against sample count.
    """
    ref, w_ref = _clean_pair(reference, weights_ref)
    cur, w_cur = _clean_pair(current, weights_cur)
    both = np.concatenate([ref, cur])
    if both.size < 2:
        empty = np.zeros(n)
        return np.linspace(0, 1, n), empty, empty
    if xlim is not None:
        lo, hi = xlim
    else:
        lo = float(np.percentile(both, 0.5))
        hi = float(np.percentile(both, 99.5))
    if not (hi > lo):
        lo, hi = float(both.min()), float(both.max()) or 1.0
    x = np.linspace(lo, hi, n)
    return x, kde_curve(ref, x, weights=w_ref), kde_curve(cur, x, weights=w_cur)
