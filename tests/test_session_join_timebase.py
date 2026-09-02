"""Multi-session joining: where a second recording lands on the timeline.

The operational question this answers: a route captured in legs arrives as one
folder per run, and every leg's own clock starts at zero. If leg 1 runs to
4219.12 s, leg 2's samples must map to 4219.13 s onward — not restart at zero,
and not leave a hole.

``concat_sessions`` joins **end to end**: it rebuilds one clock over the
appended record, so the joined timeline is continuous by construction and the
real-world gap between legs (an hour, a day) is deliberately not represented.
That is the right model for "one route driven in several runs", which is what
the feature is for, and the wrong model for "place each leg at its absolute
wall-clock time" — the seam times in the report are what let a caller tell where
one leg ended and the next began.

Also pinned here: the unconditioned reference copies are joined the same way, so
the Preprocess screen's before/after pair still lines up after a join. That now
includes the moment channels, which only entered the raw frame recently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dtt.ingestion.sessions import concat_sessions

FS = 100.0
CH = ["FL_Fx", "FL_Mx", "RR_Mz"]


def _leg(n: int, seed: int, start_value: float = 0.0) -> pd.DataFrame:
    """One session, with its own clock starting at zero, as a recorder writes it."""
    rng = np.random.default_rng(seed)
    d = {"Time": np.arange(n) / FS}
    for i, c in enumerate(CH):
        d[c] = start_value + rng.standard_normal(n) + i
    return pd.DataFrame(d)


def test_second_leg_continues_the_first_legs_clock():
    """The requirement: leg 2 maps onto the end of leg 1, not back to zero."""
    a, b = _leg(4000, 1), _leg(2500, 2)
    assert b["Time"].iloc[0] == 0.0, "precondition: leg 2 arrives with its own clock"

    joined, _, rep = concat_sessions([a, b], FS)
    t = joined["Time"].to_numpy()

    assert len(joined) == 6500
    assert t[0] == pytest.approx(0.0)
    # leg 1 occupies 0 .. 39.99 s; leg 2 starts at the next sample, not at 0
    assert t[len(a) - 1] == pytest.approx((len(a) - 1) / FS)
    assert t[len(a)] == pytest.approx(len(a) / FS)
    assert t[-1] == pytest.approx((len(joined) - 1) / FS)


def test_joined_clock_has_no_gap_and_no_overlap_at_the_seam():
    a, b = _leg(4000, 1), _leg(2500, 2)
    joined, _, _ = concat_sessions([a, b], FS)
    dt = np.diff(joined["Time"].to_numpy())
    assert np.allclose(dt, 1.0 / FS), "the seam introduced a gap or a repeat"


def test_seam_and_durations_are_reported():
    """Where one leg ended has to survive the join, or it cannot be found again."""
    a, b, c = _leg(4000, 1), _leg(2500, 2), _leg(1000, 3)
    _, _, rep = concat_sessions([a, b, c], FS)
    assert rep["sessions"] == 3
    assert rep["seam_times_s"] == [pytest.approx(40.0), pytest.approx(65.0)]
    assert rep["session_durations_s"] == [40.0, 25.0, 10.0]
    assert rep["total_duration_s"] == pytest.approx(75.0)


def test_sample_values_survive_the_join_unchanged():
    a, b = _leg(4000, 1), _leg(2500, 2)
    joined, _, _ = concat_sessions([a, b], FS)
    for c in CH:
        np.testing.assert_allclose(joined[c].to_numpy()[:len(a)], a[c].to_numpy())
        np.testing.assert_allclose(joined[c].to_numpy()[len(a):], b[c].to_numpy())


def test_raw_reference_is_joined_the_same_way_including_moments():
    """The before/after pair must still line up after a join.

    The raw frame only started carrying moments recently; if the join dropped
    them the Preprocess screen would show a paired trace for the forces and a
    lone one for the moments of the same study.
    """
    a, b = _leg(4000, 1), _leg(2500, 2)
    ra, rb = _leg(4000, 11), _leg(2500, 12)

    joined, raw_joined, _ = concat_sessions([a, b], FS, raw_frames=[ra, rb])

    assert raw_joined is not None
    assert len(raw_joined) == len(joined)
    for c in CH:
        assert c in raw_joined.columns, f"{c} lost from the raw reference"
    assert "FL_Mx" in raw_joined.columns and "RR_Mz" in raw_joined.columns


def test_a_single_session_is_passed_through_untouched():
    a = _leg(4000, 1)
    joined, _, rep = concat_sessions([a], FS)
    assert rep["sessions"] == 1
    assert rep["seam_times_s"] == []
    np.testing.assert_allclose(joined["Time"].to_numpy(), a["Time"].to_numpy())


def test_cumulative_distance_keeps_counting_across_the_seam():
    """Distance restarts each session; the joined route must not."""
    a = pd.DataFrame({"Time": np.arange(100) / FS,
                      "FL_Fx": np.zeros(100),
                      "Distance": np.arange(100, dtype=float)})
    b = pd.DataFrame({"Time": np.arange(50) / FS,
                      "FL_Fx": np.zeros(50),
                      "Distance": np.arange(50, dtype=float)})
    joined, _, _ = concat_sessions([a, b], FS)
    d = joined["Distance"].to_numpy()
    assert d[len(a) - 1] == pytest.approx(99.0)
    assert d[len(a)] == pytest.approx(99.0), "leg 2 restarted the odometer"
    assert d[-1] == pytest.approx(99.0 + 49.0)
    assert np.all(np.diff(d) >= 0), "distance went backwards at the seam"
