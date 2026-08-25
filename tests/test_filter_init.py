"""Regression tests for the FAMOS FiltLP initial-condition convention.

The bug these guard against: `butterworth_lpf` used zero-state `lfilter`, so
every FiltLP'd channel was wrong for roughly the first 1.1 s at 1000 Hz. FAMOS
uses step-response initialisation. Evidence, from a real FAMOS export
(`csv/RLDA WFT PV data sample.csv`): its FiltLP output column `Latacc_LPF`
begins at exactly 0.180000, the input level, whereas zero state would begin at
`b0*x[0]` = 7.5e-05.

The `legacy_zero_state` mode is kept for reproducing previously published study
outputs. `test_default_is_never_zero_state` is the tripwire that stops it ever
becoming the default again.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, sosfilt_zi

from dtt.preprocessing import (
    FILTER_INIT_DEFAULT, FILTER_INIT_FAMOS, FILTER_INIT_LEGACY,
    PreprocessSettings, butterworth_lpf,
)

FS = 100.0
CUTOFF = 5.0
ORDER = 4
# The value FAMOS actually exported as the first sample of Latacc_LPF.
FAMOS_FIRST_SAMPLE = 0.180000


def test_default_is_never_zero_state():
    """The tripwire. If this fails, the init bug has been reintroduced."""
    assert FILTER_INIT_DEFAULT == FILTER_INIT_FAMOS
    assert FILTER_INIT_DEFAULT != FILTER_INIT_LEGACY
    assert PreprocessSettings().filter_init == FILTER_INIT_FAMOS


def test_step_init_passes_first_sample_through():
    """y[0] == x[0]: the filter starts settled, as the FAMOS export shows."""
    x = np.full(1000, FAMOS_FIRST_SAMPLE)
    y = butterworth_lpf(x, CUTOFF, ORDER, FS)
    assert y[0] == pytest.approx(FAMOS_FIRST_SAMPLE, abs=1e-12)


def test_step_init_has_no_startup_transient_on_dc():
    """A constant in gives that constant out, everywhere -- not a ramp.

    This is the property the zero-state version violated: it climbed from ~0 to
    the input level over ~0.14 s at 100 Hz, fabricating a transient that is not
    in the measurement.
    """
    x = np.full(1000, FAMOS_FIRST_SAMPLE)
    y = butterworth_lpf(x, CUTOFF, ORDER, FS)
    assert np.allclose(y, FAMOS_FIRST_SAMPLE, atol=1e-12)


def test_legacy_zero_state_still_reproduces_the_old_startup():
    """The legacy path must keep behaving exactly as it used to.

    Its whole purpose is bit-comparable reproduction of published outputs, so a
    change here silently invalidates the archive it exists to serve.
    """
    x = np.full(1000, FAMOS_FIRST_SAMPLE)
    y = butterworth_lpf(x, CUTOFF, ORDER, FS, init=FILTER_INIT_LEGACY)
    b, _ = butter(ORDER, CUTOFF / (0.5 * FS), btype="low")
    assert y[0] == pytest.approx(b[0] * FAMOS_FIRST_SAMPLE, rel=1e-9)
    assert y[0] < 1e-3                       # nowhere near the input level


def test_the_two_inits_differ_only_at_the_start():
    """Divergence is confined to the settling window, then they agree exactly.

    This is what bounds the blast radius: past the transient the archive is
    unaffected, so any re-issue decision is about the first seconds only.
    """
    rng = np.random.default_rng(0)
    x = 0.2 + 0.05 * rng.standard_normal(20000)
    a = butterworth_lpf(x, CUTOFF, ORDER, FS, init=FILTER_INIT_FAMOS)
    b = butterworth_lpf(x, CUTOFF, ORDER, FS, init=FILTER_INIT_LEGACY)
    d = np.abs(a - b)
    assert d[0] > 0.1                                  # clearly different at t=0
    assert np.all(d[2000:] < 1e-9)                     # identical after 20 s


def test_dc_gain_is_exactly_unity():
    """FiltLP must not rescale the passband; y[0]==x[0] already implies it."""
    for level in (-3.5, 0.0, 1.0, 812.4):
        y = butterworth_lpf(np.full(500, level), CUTOFF, ORDER, FS)
        assert np.allclose(y, level, atol=1e-9)


def test_nan_gaps_survive_the_filter():
    """A dropout must stay a dropout, not poison every later sample."""
    x = np.full(1000, 0.18)
    x[400:410] = np.nan
    y = butterworth_lpf(x, CUTOFF, ORDER, FS)
    assert np.all(np.isnan(y[400:410]))
    assert np.isfinite(y[:400]).all() and np.isfinite(y[410:]).all()


def test_unknown_init_raises_rather_than_defaulting():
    """No silent fallback -- a typo'd mode must fail loudly, not pick one."""
    with pytest.raises(ValueError, match="unknown filter init"):
        butterworth_lpf(np.zeros(500), CUTOFF, ORDER, FS, init="zero")


def test_smo_is_unaffected_by_init():
    """smo is a convolution with no recursive state.

    This is what confines the bug to `Latacc`: every WFT force and moment
    channel gets smo(0.1) and no FiltLP, so none of them can be affected.
    """
    from dtt.preprocessing import famos_recipe, is_wft_channel
    for ch in ("FL_Fx", "FR_Fz", "RR_Mz", "FR_My_2"):
        assert is_wft_channel(ch)
        assert famos_recipe(ch).apply_filter is False
    assert famos_recipe("Latacc").apply_filter is True


# ---------------------------------------------------------------------------
# smo edge convention -- imc FAMOS Function Reference, "Smo"
# ---------------------------------------------------------------------------
# "The weighting function is triangular if the interval width is greater than
#  five points... can only be an odd number of points... The implemented filter
#  is a non-causal filter with a phase of zero... For filtering near the edges,
#  it is assumed that the data sets are extended with the same values as the
#  edges."
#
# The last sentence is the one the implementation originally got wrong: it
# renormalised the kernel at the array ends instead of replicating the edge
# value. Measured against the FAMOS export's Latacc_LPF -> Latacc pair,
# replication is 4.5x better over the leading half-window and 5.4x over the
# trailing one.

def test_smo_kernel_is_triangular_and_odd():
    from dtt.preprocessing import famos_smooth_window
    h = famos_smooth_window(100.0, 0.5)
    assert h.size % 2 == 1                       # "only an odd number of points"
    assert h.sum() == pytest.approx(1.0, abs=1e-12)   # "preserves the mean value"
    mid = h.size // 2
    assert h[mid] == h.max()                     # triangular: peak at centre
    assert np.allclose(h, h[::-1])               # symmetric -> zero phase
    rising = np.diff(h[:mid + 1])
    assert np.all(rising > 0)                    # straight sides, no plateau


def test_smo_is_zero_phase():
    """A symmetric kernel must not shift features in time."""
    from dtt.preprocessing import famos_smooth
    x = np.zeros(2000)
    x[1000] = 1.0
    y = famos_smooth(x, 100.0, 0.5)
    assert int(np.argmax(y)) == 1000


def test_smo_edges_replicate_not_renormalise():
    """A constant record must smooth to that same constant, edges included.

    Edge replication preserves the level all the way to the ends. Kernel
    renormalisation also happens to preserve a constant, so the discriminating
    case is the ramp below -- this one guards the simpler invariant.
    """
    from dtt.preprocessing import famos_smooth
    y = famos_smooth(np.full(5000, 3.25), 100.0, 0.5)
    assert np.allclose(y, 3.25, atol=1e-12)


def test_smo_edge_matches_explicit_edge_padding():
    """The implementation must equal an explicit edge-padded convolution.

    This is the direct statement of the FAMOS rule, so it pins the convention
    rather than just a consequence of it.
    """
    from dtt.preprocessing import famos_smooth, famos_smooth_window
    rng = np.random.default_rng(7)
    x = np.cumsum(rng.standard_normal(4000)) * 0.01 + 5.0
    fs, w = 100.0, 0.5
    h = famos_smooth_window(fs, w)
    hw = h.size // 2
    expected = np.convolve(
        np.r_[np.full(hw, x[0]), x, np.full(hw, x[-1])], h, mode="valid")
    got = famos_smooth(x, fs, w)
    assert np.allclose(got, expected, atol=1e-12)


def test_smo_interior_gaps_are_not_replicated_across():
    """A NaN gap is a missing measurement, not an array end.

    Edge replication must not leak across an interior dropout, and the gap
    itself must survive as a gap.
    """
    from dtt.preprocessing import famos_smooth
    x = np.full(4000, 2.0)
    x[2000:2050] = np.nan
    y = famos_smooth(x, 100.0, 0.5)
    assert np.all(np.isnan(y[2000:2050]))
    assert np.isfinite(y[:2000]).all() and np.isfinite(y[2050:]).all()
