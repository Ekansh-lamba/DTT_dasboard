"""Tests for the all-channel FAMOS cross-check.

The gap this closes, from ``MODULE1_STATUS.md``: the moment channels and
``Latacc`` on a real recording were **inferred, not measured**, because the
validator covered six force channels named as literals. Adding six more
literals would not have fixed it -- the next recording has different names
again -- so the channel list comes from the folder and the operators come from
``famos_recipe``.

What is pinned here is the part that can be wrong silently:

  * every conditioned channel in the recording gets registered, moments and
    Latacc included, and nothing is dropped without a stated reason
  * matching is canonical, so an export that names the same wheel differently
    still lines up, and the raw/smo ambiguity of a bare column resolves the
    right way round
  * the recipe comes from ``famos_recipe`` rather than a second copy of it
  * scoring is against the per-sample quantum, not a fixed tolerance -- the
    trap that reported healthy channels as failures at 1e5
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dtt.validation.famos_validation import (
    canonical_key, match_export, quantum_ratio, quantum_within_pct,
    register_all_channels, split_stage,
)

RAW_DIR = Path("raw data in .dat format/2006-08-25 09-10-46 (1)")

MOMENTS = ["FR_Mx_2", "FR_My_2", "FR_Mz_2", "RR_Mx_1", "RR_My_1", "RR_Mz_1"]
FORCES = ["FR_Fx_2", "FR_Fy_2", "FR_Fz_2", "RR_Fx_1", "RR_Fy_1", "RR_Fz_1"]


# ----------------------------------------------------------- name handling

@pytest.mark.parametrize("a,b", [
    ("FR_Fz_2", "WFT_Fz_fr"),          # two recordings, one wheel
    ("FR_Fz_2", "A1R_Fz"),             # and our own canonical form
    ("RR_Mz_1", "WFT_Mz_rr"),          # moments too
])
def test_the_same_wheel_matches_across_naming_schemes(a, b):
    assert canonical_key(a) == canonical_key(b)


def test_different_wheels_do_not_collide():
    keys = {canonical_key(c) for c in FORCES + MOMENTS}
    assert len(keys) == len(FORCES + MOMENTS)


def test_aux_channels_key_on_their_name_not_the_channel_model():
    """``parse_channel`` rejects speed/accel names by design, so a single-tier
    canonical match would drop exactly the channels this round measures."""
    from dtt.channels import parse_channel

    for name in ("Latacc", "Latacc_LPF", "Vehicle_Speed", "Longacc"):
        assert parse_channel(name) is None          # the model says "not mine"
        assert canonical_key(name)                  # we still get a key


@pytest.mark.parametrize("column,base,stage", [
    ("FR_Fz_2", "FR_Fz_2", ""),
    ("FR_Fz_2_smo", "FR_Fz_2", "smo"),
    ("FR_Fz_2_red", "FR_Fz_2", "red"),
    ("FR_Fz_2_rawred", "FR_Fz_2", "rawred"),        # not "_red" with a prefix
    ("Latacc_LPF", "Latacc", "lpf"),
    ("Latacc_LPF_red", "Latacc", "lpf_red"),
])
def test_stage_suffixes_split_longest_first(column, base, stage):
    assert split_stage(column) == (base, stage)


# ------------------------------------------------------------- the quantum

def test_quantum_scoring_does_not_punish_large_channels():
    """The trap this exists for: six significant figures is 1e-6 at order 1
    and 1.0 at 1e5, so one fixed absolute tolerance cannot judge both."""
    ref = np.array([1.0, 1e5])
    err = np.array([4e-7, 0.04])       # both well inside half a stored digit
    assert quantum_ratio(err, ref) <= 0.5
    assert quantum_within_pct(err, ref) == 100.0

    # and a real disagreement at the large end is still caught
    assert quantum_ratio(np.array([0.0, 5.0]), ref) > 0.5


def test_within_pct_separates_one_bad_sample_from_a_bad_channel():
    ref = np.full(100, 1e5)
    one_bad = np.zeros(100)
    one_bad[7] = 5.0
    assert quantum_ratio(one_bad, ref) > 0.5        # worst case fails
    assert quantum_within_pct(one_bad, ref) == 99.0  # but 99% is clean


# --------------------------------------------------------- the registration

@pytest.fixture(scope="module")
def registered():
    # A short cut keeps the test quick; the registration logic does not depend
    # on the length.
    return register_all_channels(RAW_DIR, cut=(0, 20_000))


@pytest.mark.skipif(not RAW_DIR.is_dir(),
                    reason="the reference recording is not on this machine")
class TestAgainstTheRealRecording:

    def test_moment_channels_are_registered(self, registered):
        """The §10 gap. Every moment gets the same stages a force does."""
        cases, _ = registered
        by_channel = {c.recipe_name for c in cases}
        for m in MOMENTS:
            assert m in by_channel, f"{m} was not registered"

    def test_moments_and_forces_get_identical_stages(self, registered):
        cases, _ = registered
        stages = {}
        for c in cases:
            stages.setdefault(c.recipe_name, set()).add(c.stage)
        assert stages["FR_Mx_2"] == stages["FR_Fx_2"]

    def test_latacc_is_registered_with_its_filter_stage(self, registered):
        """Latacc is the only channel that uses FiltLP, and it must be
        separable from smo or a disagreement cannot be attributed."""
        cases, _ = registered
        latacc = {c.stage for c in cases if c.recipe_name == "Latacc"}
        assert {"lpf", "lpf_red", "smo", "red", "rawred"} <= latacc
        assert all(c.source == "AccelY" for c in cases
                   if c.recipe_name == "Latacc")

    def test_speed_carries_the_step_one_scale_factor(self, registered):
        """``Speed_kmph = Speed2D * 3.6`` is the one map entry that is
        arithmetic rather than a rename."""
        cases, _ = registered
        speed = [c for c in cases if c.recipe_name == "Vehicle_Speed"]
        assert speed and all("3.6" in c.note for c in speed)

    def test_nothing_is_dropped_without_a_reason(self, registered):
        cases, skipped = registered
        registered_sources = {c.source for c in cases}
        for f in RAW_DIR.glob("*.raw"):
            assert f.stem in registered_sources or f.stem in skipped, (
                f"{f.stem} was neither registered nor explained")
        assert all(why.strip() for why in skipped.values())

    def test_passthrough_channels_say_so(self, registered):
        """A GPS channel is not a failure to compare -- the recipe does
        nothing to it, so there is no FAMOS operation to check."""
        _, skipped = registered
        assert "no FAMOS operation" in skipped.get("PosLat", "")

    def test_copy_duplicates_are_skipped(self, registered):
        _, skipped = registered
        copies = [k for k in skipped if "copy" in k.lower()]
        assert copies
        assert all("duplicate" in skipped[k] for k in copies)

    def test_bare_column_resolves_to_raw_for_a_force_and_smo_for_latacc(
            self, registered):
        """The ambiguity that a string match cannot see: ``FR_Fz_2`` is the
        input FAMOS read, ``Latacc`` is the output of smoothing."""
        cases, _ = registered
        matched, unmatched = match_export(cases, ["FR_Fz_2", "Latacc", "x"])
        assert matched[(canonical_key("FR_Fz_2"), "raw")] == "FR_Fz_2"
        assert matched[(canonical_key("Latacc"), "smo")] == "Latacc"
        assert unmatched == ["x"]

    def test_export_names_are_matched_canonically_not_literally(self, registered):
        """An export naming the wheel differently still lines up."""
        cases, _ = registered
        matched, unmatched = match_export(cases, ["WFT_Fz_fr_smo"])
        assert matched.get((canonical_key("FR_Fz_2"), "smo")) == "WFT_Fz_fr_smo"
        assert not unmatched
