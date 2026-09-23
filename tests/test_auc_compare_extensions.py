"""Tests for the AUC comparison extensions: stats strip, coverage, range mode,
distance weighting, cross-position, batch export and the all-channels chart.

The ones that matter most are the ones that could be wrong silently:

  * **Moments are offered.** ``RunChannels`` keeps forces only, and the first
    version of the comparison asked it for moments and got nothing back -- the
    screen offered six force channels and no one noticed.
  * **The unweighted path is unchanged.** Distance weighting is opt-in; with no
    weights every number must be byte-identical to what it was.
  * **Weighting is both-or-neither.** A distance-weighted run against a
    sample-count run compares two different questions.
  * **Identical data lands on the no-change line** of the all-channels chart,
    or the chart's baseline means nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

from dtt.analysis.auc import compare_distributions, weighted_percentile  # noqa: E402
from dtt.analysis.plot_style import stats_strip_rows              # noqa: E402
from dtt.analysis.study_compare import (                          # noqa: E402
    auc_channels, auc_coverage_text, auc_cross_channels, auc_figure_layout,
    auc_footer_text, auc_summary, channel_unit, draw_auc_comparison,
    draw_auc_summary, export_auc_batch, _EXCEED_BASELINE, _fmt_delta,
)
from dtt.run_channels import build_run_channels                   # noqa: E402


def _study(seed: int, shift: float = 0.0, n: int = 20_000) -> pd.DataFrame:
    """A two-axle, four-wheel study with forces AND moments."""
    rng = np.random.default_rng(seed)
    cols = {"Time": np.arange(n) / 100.0}
    for pos in ("FL", "FR", "RL", "RR"):
        for comp, centre, spread in (("Fx", 0, 40), ("Fy", 0, 30),
                                     ("Fz", 800, 60), ("Mx", 0, 15),
                                     ("My", 0, 20), ("Mz", 0, 5)):
            cols[f"{pos}_{comp}"] = rng.normal(centre + shift, spread, n)
    return pd.DataFrame(cols)


@pytest.fixture(scope="module")
def studies():
    a, b = _study(1), _study(2, shift=5.0)
    return a, build_run_channels(list(a.columns)), b, build_run_channels(list(b.columns))


# ------------------------------------------------------------ channel model

def test_moments_are_offered_not_just_forces(studies):
    da, ra, db, rb = studies
    names = [c[0] for c in auc_channels(ra, rb, da, db)]
    assert len(names) == 24                     # 4 wheels x 6 components
    for comp in ("Mx", "My", "Mz"):
        assert any(n.endswith(f"_{comp}") for n in names), comp


def test_runchannels_alone_would_have_lost_the_moments(studies):
    """Pins why _channel_index exists: channel_for returns None for Mx."""
    _, ra, _, _ = studies
    assert ra.channel_for("FL", "Mx") is None
    assert ra.channel_for("FL", "Fx") is not None


def test_cross_position_pairs_front_with_rear_both_ways(studies):
    da, ra, db, rb = studies
    names = [c[0] for c in auc_cross_channels(ra, rb, da, db)]
    assert "FL_Fz vs RL_Fz" in names and "RL_Fz vs FL_Fz" in names
    assert "FR_Mx vs RR_Mx" in names and "RR_Mx vs FR_Mx" in names
    # never a wheel against itself, never across sides
    for n in names:
        left, right = n.split(" vs ")
        assert left != right
        assert left[1] == right[1]              # FL<->RL, FR<->RR


def test_single_axle_has_no_cross_partner():
    df = _study(3)[["Time", "FL_Fx", "FR_Fx"]]
    rc = build_run_channels(list(df.columns))
    assert auc_cross_channels(rc, rc, df, df) == []


def test_units_come_from_the_component_not_the_magnitude():
    assert channel_unit("FR_Fz") == "daN"
    assert channel_unit("FR_My") == "daN·m"
    assert channel_unit("FR_Mx vs RR_Mx") == "daN·m"


# ------------------------------------------------------------- weighting

def test_unweighted_results_are_unchanged():
    rng = np.random.default_rng(0)
    a, b = rng.normal(800, 60, 30_000), rng.normal(830, 70, 30_000)
    cmp = compare_distributions(a, b)
    assert cmp.p95_ref == float(np.percentile(a, 95))
    assert cmp.p5_cur == float(np.percentile(b, 5))
    assert cmp.pct_exceed_ref_p95 == float(np.mean(b >= np.percentile(a, 95)) * 100)


def test_equal_weights_reproduce_the_unweighted_percentiles():
    rng = np.random.default_rng(1)
    v = rng.normal(0, 1, 50_000)
    for q in (5, 25, 50, 75, 95):
        assert weighted_percentile(v, np.ones_like(v), q) == pytest.approx(
            np.percentile(v, q), abs=1e-3)


def test_weight_pulls_the_distribution_toward_the_heavy_samples():
    v = np.concatenate([np.full(900, 10.0), np.full(100, 20.0)])
    w = np.concatenate([np.full(900, 1.0), np.full(100, 50.0)])
    assert weighted_percentile(v, w, 50) == pytest.approx(20.0)
    assert np.median(v) == 10.0


def test_weighting_one_run_and_not_the_other_is_refused():
    a = np.arange(100.0)
    with pytest.raises(ValueError):
        compare_distributions(a, a, np.ones(100), None)


def test_stats_strip_weights_are_opt_in():
    v = np.arange(1000.0)
    plain = stats_strip_rows([("A", v)])
    again = stats_strip_rows([("A", v)], weights=None)
    assert plain == again
    heavy = stats_strip_rows([("A", v)], weights=[np.where(v > 900, 100.0, 1.0)])
    median_col = plain[0].index("Median")
    assert float(heavy[1][0][median_col]) > float(plain[1][0][median_col])


# ---------------------------------------------------------- the pair + strip

def test_the_strip_is_drawn_in_its_own_row():
    rng = np.random.default_rng(4)
    a, b = rng.normal(800, 60, 20_000), rng.normal(820, 60, 20_000)
    fig = plt.figure(figsize=(14, 7))
    try:
        ax_kde, ax_hist, ax_stats = auc_figure_layout(fig)
        draw_auc_comparison(ax_kde, ax_hist, a, b, "EV", "IC", "FR_Fz",
                            ax_stats=ax_stats)
        assert ax_stats.tables, "no stats strip"
        cells = [c.get_text().get_text() for c in ax_stats.tables[0].get_celld().values()]
        assert "EV" in cells and "IC" in cells and "Median" in cells
        # sits entirely below both panels
        assert ax_stats.get_position().y1 < ax_kde.get_position().y0
        assert ax_stats.get_position().y1 < ax_hist.get_position().y0
    finally:
        plt.close(fig)


def test_weighted_axes_say_so():
    rng = np.random.default_rng(5)
    a, b = rng.normal(0, 1, 5_000), rng.normal(0, 1, 5_000)
    w = np.ones(5_000)
    fig, (k, h) = plt.subplots(1, 2)
    try:
        draw_auc_comparison(k, h, a, b, "EV", "IC", "FR_Fx",
                            weights_a=w, weights_b=w)
        assert "Distance" in k.get_ylabel()
    finally:
        plt.close(fig)


def test_full_range_mode_follows_the_98_percent_rule():
    """Pins what "Full sensor range" actually does, because the first
    description of it was wrong: it was sold as the cure for a stuck-sensor
    rail. It is the configured range (Fx +/-300) while >= 98% of samples fit
    it, and the data's own range otherwise -- so a small rail is cropped and a
    large one is not, and neither mode is a substitute for cleaning the data.
    """
    rng = np.random.default_rng(6)
    body = rng.normal(20, 30, 20_000)

    small = np.concatenate([body, np.full(200, -451.0)])     # ~1%: fits 98%
    large = np.concatenate([body, np.full(1_000, -451.0)])   # ~5%: does not
    fig, axes = plt.subplots(2, 2)
    try:
        draw_auc_comparison(*axes[0], small, small, "EV", "IC", "FR_Fx",
                            range_mode="full")
        assert axes[0][0].get_xlim() == (-300.0, 300.0)      # rail cropped
        draw_auc_comparison(*axes[1], large, large, "EV", "IC", "FR_Fx",
                            range_mode="full")
        assert axes[1][0].get_xlim()[0] < -400               # rail kept
    finally:
        plt.close(fig)


def test_footer_quotes_both_p5s_and_coverage_counts_the_data():
    rng = np.random.default_rng(7)
    a, b = rng.normal(800, 60, 4_000), rng.normal(820, 60, 8_000)
    cmp = compare_distributions(a, b)
    footer = auc_footer_text(cmp, "EV", "IC")
    assert footer.count("P5:") == 2
    cov = auc_coverage_text(cmp, "EV", "IC", 400.0, 800.0)
    assert "4,000 samples" in cov and "8,000 samples" in cov
    assert "IC is 2.0x longer" in cov
    assert "sample count" in cov


def test_delta_precision_follows_the_channel_size():
    assert _fmt_delta(0.37, 25.0) == "+0.4"      # moment-sized: one decimal
    assert _fmt_delta(0.37, 900.0) == "+0"        # Fz-sized: integer
    assert _fmt_delta(0.037, 3.0) == "+0.04"


# ------------------------------------------------------ all channels at once

def test_identical_data_sits_on_the_no_change_line(studies):
    da, ra, _, _ = studies
    rows = auc_summary(auc_channels(ra, ra, da, da), da, da)
    for r in rows:
        assert r.cmp.pct_exceed_ref_p95 == pytest.approx(_EXCEED_BASELINE, abs=0.1)


def test_summary_chart_uses_the_two_colours_and_the_labels(studies):
    da, ra, db, rb = studies
    rows = auc_summary(auc_channels(ra, rb, da, db), da, db)
    fig, ax = plt.subplots()
    try:
        draw_auc_summary(ax, rows, "EV", "IC")
        assert len(ax.patches) == len(rows)
        assert "IC" in ax.get_xlabel() and "EV" in ax.get_xlabel()
        colours = {matplotlib.colors.to_hex(p.get_facecolor()).upper()
                   for p in ax.patches}
        assert colours == {"#E74C3C"}
    finally:
        plt.close(fig)


def test_batch_export_writes_every_channel(tmp_path: Path, studies):
    da, ra, db, rb = studies
    channels = auc_channels(ra, rb, da, db)[:3]
    seen = []
    pngs, csv_path, summary = export_auc_batch(
        channels, da, db, "EV", "IC", tmp_path,
        progress=lambda i, n, name: seen.append(i))
    assert len(pngs) == 3 and all(p.exists() for p in pngs)
    assert summary.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 3
    assert "P95_EV" in df.columns and "P95_IC" in df.columns
    assert seen[-1] == 3
