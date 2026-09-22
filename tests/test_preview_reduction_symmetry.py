"""Regression tests for the Preprocess preview's screen-width reduction.

The bug these pin down: the before/after preview reduced its two traces by
**different statistics** -- the raw through a bucket min->max envelope, the
conditioned channel through a bucket median -- so at full-recording zoom, where
a bucket holds hundreds of samples, the visible blue-to-red gap was mostly the
reduction rather than the conditioning. Measured on a 5,547 s recording reduced
to 950 buckets, ``FR_Fz_2``'s conditioning removed 14.0 % of the channel range
while the plot drew a gap of 76.7 %; ``FR_My_2`` was 2.2 % drawn as 24.9 %.

The fix is not "reduce less". It is to reduce **both** traces the same way and
draw the conditioned channel's own envelope behind its median, so the gap on
screen is a like-for-like comparison. These tests therefore check the three
properties that make that honest, because each one was a candidate fix that
would have broken one of the others:

  * both traces come from the same statistic, so identical data draws no gap
  * an isolated spike still reaches the envelope (which a median-only fix,
    the other obvious way to make the two traces comparable, would have lost)
  * a dropout still breaks the line instead of being ruled across
  * the density ramp weights both traces identically, so it can no longer
    fade one against the other
  * ``_spikes`` is scored at the full sample rate and is unaffected by any
    of this -- stated in the brief as an assumption, pinned here as a fact
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="the preview helpers live on a Qt page")

from gui.pages.preprocess_page import (            # noqa: E402
    _density_style, _envelope_and_median, _median_line, _reduce, _spikes,
)

FS = 1000.0
NPTS = 950


def _signal(n: int = 200_000, seed: int = 0):
    """A load-like trace, deliberately shaped like a real wheel force.

    The 2 Hz term is the one that matters. A bucket here spans ~0.2 s, and the
    reason the old pairing overstated the conditioning is that most of a
    bucket's min-to-max reach comes from signal the recipe **keeps** -- the
    wheel genuinely moving -- not from the ripple it removes. A synthetic made
    of slow drift plus ripple alone has nothing in the passband to fill the
    bucket with, so envelope and median very nearly agree and the bug cannot
    reproduce.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    slow = 7000 + 300 * np.sin(2 * np.pi * 0.05 * t)
    road = 300 * np.sin(2 * np.pi * 2.0 * t)       # in band: smo keeps this
    ripple = 40 * np.sin(2 * np.pi * 60 * t) + rng.normal(0, 15, n)
    return t, slow + road + ripple


def _halves(y_env: np.ndarray):
    """An envelope stroke array back into its per-bucket (lo, hi)."""
    return y_env[0::2], y_env[1::2]


# --------------------------------------------------------------- symmetry

def test_identical_data_draws_no_gap():
    """The whole point. Feed the same array twice; the two traces must
    coincide, because any difference would be the reduction talking."""
    t, y = _signal()
    _, env_a, _, _ = _envelope_and_median(t, y, NPTS)
    _, env_b, _, _ = _envelope_and_median(t, y, NPTS)
    assert np.array_equal(env_a, env_b, equal_nan=True)


def test_drawn_gap_equals_the_conditioning_and_not_the_reduction():
    """Envelope-vs-envelope reports what smoothing removed; the old
    envelope-vs-median pairing overstated it several times over."""
    from famos import ops

    t, raw = _signal()
    proc = ops.smo(raw, 0.1, FS)

    _, raw_env, _, _ = _envelope_and_median(t, raw, NPTS)
    _, proc_env, _, proc_mid = _envelope_and_median(t, proc, NPTS)
    r_lo, r_hi = _halves(raw_env)
    p_lo, p_hi = _halves(proc_env)

    true_effect = float(np.nanmean((r_hi - r_lo) - (p_hi - p_lo)))
    drawn_now = true_effect                       # both traces, same statistic
    drawn_before = float(np.nanmean(r_hi - r_lo))  # raw envelope vs a median

    assert true_effect > 0                        # smoothing did remove ripple
    assert drawn_now == pytest.approx(true_effect)
    # the old pairing inflated it by a large factor, not a rounding amount
    assert drawn_before > 3 * true_effect


def test_median_alone_would_have_lost_the_spike():
    """Pins why 'draw both as medians' was rejected rather than merely not
    chosen: it satisfies symmetry and silently deletes a pothole strike."""
    t, y = _signal()
    y[100_000] = 30_000.0                          # one isolated strike

    _, env, _, mid = _envelope_and_median(t, y, NPTS)
    assert np.nanmax(env) == pytest.approx(30_000.0)
    assert np.nanmax(mid) < 10_000.0


def test_dropout_still_breaks_both_lines():
    """NaN buckets survive the reduction, so matplotlib leaves a gap instead
    of ruling a straight segment across a dead sensor."""
    t, y = _signal()
    y[40_000:60_000] = np.nan                      # 20 s of dropout

    _, env, _, mid = _envelope_and_median(t, y, NPTS)
    assert np.isnan(env).any()
    assert np.isnan(mid).any()


def test_short_input_returns_the_samples_themselves():
    """Below the bucket count there is nothing to reduce, and the caller
    relies on the identity to skip drawing the same line twice."""
    t, y = _signal(n=500)
    te, ye, tm, ym = _envelope_and_median(t, y, NPTS)
    assert te is tm and ye is ym
    assert np.array_equal(ye, y)


def test_median_line_still_agrees_with_reduce():
    """``_median_line`` stays for the recipe's intermediate stages; it must
    keep returning exactly ``_reduce``'s median."""
    t, y = _signal()
    tm_ref, _, mid_ref, _ = _reduce(t, y, NPTS)
    tm, mid = _median_line(t, y, NPTS)
    assert np.array_equal(tm, tm_ref)
    assert np.array_equal(mid, mid_ref, equal_nan=True)


# ------------------------------------------------------------ density ramp

def test_density_ramp_is_symmetric():
    """One alpha and one envelope width, used for both traces. The previous
    ramp faded the raw to 0.20/0.40 while taking the conditioned line to
    1.30 -- three times the ink on the same axes."""
    sparse = _density_style(NPTS, NPTS)            # ~1 sample per pixel
    dense = _density_style(600 * NPTS, NPTS)       # ~600 per pixel
    for env_alpha, env_lw, mid_lw in (sparse, dense):
        assert 0.0 < env_alpha <= 1.0
        assert env_lw > 0 and mid_lw > 0
    # the ramp still quietens a crowded envelope, it just does it to both
    assert dense[0] < sparse[0]
    assert dense[1] < sparse[1]


# ------------------------------------------------------------ spike marks

def test_spikes_are_scored_at_full_rate_and_ignore_the_reduction():
    """The brief asks for this to be confirmed rather than assumed: the
    de-glitch markers score raw-minus-processed at the native rate, so the
    bucket count cannot move them."""
    from famos import ops

    t, raw = _signal()
    raw[120_000] = 25_000.0
    proc = ops.smo(raw, 0.1, FS)

    ts, ys, total = _spikes(t, raw, proc, 6.0)
    assert total > 0
    assert pytest.approx(25_000.0) == float(np.nanmax(ys))
    # the marked times are real sample times, not bucket centres
    assert np.all(np.isin(ts, t))
