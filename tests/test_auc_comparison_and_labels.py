"""Tests for the two-dataset AUC comparison and for dataset labelling.

Two things are pinned here, and they pull in opposite directions:

**The labels must reach everything.** ``"previous"`` and ``"current"`` were
baked into legends, titles and exported column names, which is useless when
the two runs are an EV and an IC variant. So a rename has to show up in the
legend entries, the percentile entries, the figure title, the footer strip and
the CSV's own column headings.

**The labels must reach nothing that matters.** A name is cosmetic. Renaming
must not move a percentile, an exceedance figure, or which dataset is the
reference -- otherwise a typo in a text box silently changes the engineering.

Plus the two rules the port itself had to get right: exactly two colours, and
the platform's full-data KDE rather than the reference script's 20k subsample.
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

from dtt.analysis.auc import compare_distributions                # noqa: E402
from dtt.analysis.study_compare import (                          # noqa: E402
    AUC_DS1_COLOR, AUC_DS2_COLOR, auc_comparison_stats, auc_footer_text,
    draw_auc_comparison,
)


def _two_runs(seed: int = 3):
    rng = np.random.default_rng(seed)
    a = rng.normal(800, 60, 50_000)          # the reference run
    b = rng.normal(830, 70, 50_000)          # a slightly harsher one
    return a, b


def _axes():
    fig, axes = plt.subplots(1, 2)
    return fig, axes[0], axes[1]


def _hex(colour) -> str:
    return matplotlib.colors.to_hex(colour).upper()


# -------------------------------------------------------------- two colours

def test_only_two_colours_are_used():
    """Every element of a dataset -- KDE line, fill, bars, P5 and P95 -- is
    that dataset's one colour. P5 and P95 are told apart by line style."""
    a, b = _two_runs()
    fig, ax_kde, ax_hist = _axes()
    try:
        draw_auc_comparison(ax_kde, ax_hist, a, b, "EV", "IC", "FR_Fz")
        allowed = {_hex(AUC_DS1_COLOR), _hex(AUC_DS2_COLOR)}
        for ax in (ax_kde, ax_hist):
            for line in ax.lines:
                assert _hex(line.get_color()) in allowed
            for patch in ax.patches:
                assert _hex(patch.get_facecolor()) in allowed
    finally:
        plt.close(fig)


def test_p5_and_p95_differ_by_line_style_not_by_colour():
    a, b = _two_runs()
    fig, ax_kde, ax_hist = _axes()
    try:
        draw_auc_comparison(ax_kde, ax_hist, a, b, "EV", "IC", "FR_Fz")
        rules = {line.get_label(): line for line in ax_kde.lines
                 if "P5" in line.get_label() or "P95" in line.get_label()}
        p5 = [l for k, l in rules.items() if "P5:" in k]
        p95 = [l for k, l in rules.items() if "P95:" in k]
        assert p5 and p95
        assert all(l.get_linestyle() == ":" for l in p5)
        assert all(l.get_linestyle() == "--" for l in p95)
    finally:
        plt.close(fig)


def test_both_panels_share_one_x_range():
    a, b = _two_runs()
    fig, ax_kde, ax_hist = _axes()
    try:
        draw_auc_comparison(ax_kde, ax_hist, a, b, "EV", "IC", "FR_Fz")
        assert ax_kde.get_xlim() == ax_hist.get_xlim()
        assert ax_kde.get_ylabel() == ax_hist.get_ylabel() == "Normalised density"
        assert "KDE Curves" in ax_kde.get_title()
        assert "Density Histogram Overlay" in ax_hist.get_title()
    finally:
        plt.close(fig)


def test_legend_is_readable_on_the_light_panel():
    """The reference sets labelcolor="white" on a #F5F5F5 panel, which is the
    unreadable legend this port exists to fix."""
    a, b = _two_runs()
    fig, ax_kde, ax_hist = _axes()
    try:
        draw_auc_comparison(ax_kde, ax_hist, a, b, "EV", "IC", "FR_Fz")
        for ax in (ax_kde, ax_hist):
            legend = ax.get_legend()
            assert legend is not None
            for text in legend.get_texts():
                assert _hex(text.get_color()) != "#FFFFFF"
    finally:
        plt.close(fig)


# ------------------------------------------------------------- the labels

def test_labels_reach_every_legend_entry():
    a, b = _two_runs()
    fig, ax_kde, ax_hist = _axes()
    try:
        draw_auc_comparison(ax_kde, ax_hist, a, b, "EV", "IC", "FR_Fz")
        for ax in (ax_kde, ax_hist):
            entries = [t.get_text() for t in ax.get_legend().get_texts()]
            assert entries[0] == "EV"          # the reference reads first
            assert "EV" in entries and "IC" in entries
            assert any(e.startswith("EV P5") for e in entries)
            assert any(e.startswith("EV P95") for e in entries)
            assert any(e.startswith("IC P5") for e in entries)
            assert any(e.startswith("IC P95") for e in entries)
            assert not any("previous" in e or "current" in e for e in entries)
    finally:
        plt.close(fig)


def test_labels_reach_the_footer():
    a, b = _two_runs()
    cmp = compare_distributions(a, b)
    footer = auc_footer_text(cmp, "EV", "IC")
    assert "EV (reference)" in footer
    assert "exceed EV P95" in footer
    assert "within EV normal zone" in footer
    assert "IC" in footer
    assert "previous" not in footer and "current" not in footer


def test_labels_reach_the_exported_column_names():
    a, b = _two_runs()
    row = auc_comparison_stats(a, b, "EV", "IC", "FR_Fz")
    for key in ("P5_EV", "P95_EV", "P95_IC", "Pct_exceed_EV_P95",
                "Pct_normal_IC", "n_EV", "n_IC"):
        assert key in row, key


def test_comparison_result_table_columns_follow_the_labels():
    """``ChannelDelta.to_row`` used to hardcode prev_/curr_ prefixes."""
    import pandas as pd
    from dtt.comparison import compare_frames

    t = np.arange(2000) / 100.0
    prev = pd.DataFrame({"Time": t, "FR_Fz_2": np.cos(t) * 30 + 800})
    curr = pd.DataFrame({"Time": t, "WFT_Fz_fr": np.cos(t) * 33 + 820})
    result = compare_frames(prev, curr, "EV", "IC", 100.0, 100.0)

    cols = list(result.table().columns)
    assert "EV_mean" in cols and "IC_mean" in cols
    assert "prev_mean" not in cols and "curr_mean" not in cols

    # renaming re-columns without recomputing anything
    before = result.table()["EV_mean"].tolist()
    result.previous_label, result.current_label = "EV run 1", "IC run 2"
    after = result.table()["EV_run_1_mean"].tolist()
    assert before == after


# ------------------------------------- labels are cosmetic, and only that

def test_renaming_changes_no_number():
    a, b = _two_runs()
    one = auc_comparison_stats(a, b, "EV", "IC", "FR_Fz")
    two = auc_comparison_stats(a, b, "Alpha", "Beta", "FR_Fz")
    assert one["P95_EV"] == two["P95_Alpha"]
    assert one["Pct_exceed_EV_P95"] == two["Pct_exceed_Alpha_P95"]
    assert one["Pct_normal_IC"] == two["Pct_normal_Beta"]


def test_dataset_one_is_the_reference_whatever_it_is_called():
    """The reference script makes dataset 2 the baseline; the platform makes
    dataset 1 the baseline, matching `compare_distributions(a, b)`. Mixing the
    two would silently change every exceedance figure, so it is pinned."""
    a, b = _two_runs()
    cmp = compare_distributions(a, b)
    assert cmp.p95_ref == pytest.approx(float(np.percentile(a, 95)))
    assert cmp.p95_cur == pytest.approx(float(np.percentile(b, 95)))
    assert cmp.pct_exceed_ref_p95 == pytest.approx(
        float(np.mean(b >= np.percentile(a, 95)) * 100))


# -------------------------------------------------------------- the KDE

def test_kde_is_fitted_on_all_the_data_not_a_subsample():
    """The reference's KDE_MAX_SAMPLES = 20000 widens Scott's bandwidth by
    ~1.7x at 600k samples and smears separate peaks into one. Two well-parted
    modes must survive."""
    from dtt.analysis.auc import kde_curve

    rng = np.random.default_rng(0)
    bimodal = np.concatenate([rng.normal(700, 12, 300_000),
                              rng.normal(900, 12, 300_000)])
    x = np.linspace(600, 1000, 600)
    curve = kde_curve(bimodal, x)

    # a genuine dip between the two peaks, not one merged hump
    trough = curve[(x > 780) & (x < 820)].max()
    peaks = max(curve[(x > 680) & (x < 720)].max(),
                curve[(x > 880) & (x < 920)].max())
    assert trough < 0.25 * peaks


def test_insufficient_data_says_so_rather_than_drawing_a_lie():
    fig, ax_kde, ax_hist = _axes()
    try:
        cmp = draw_auc_comparison(ax_kde, ax_hist, np.array([1.0]),
                                  np.array([2.0]), "EV", "IC", "FR_Fz")
        assert cmp is None
        assert any("insufficient" in t.get_text() for t in ax_kde.texts)
    finally:
        plt.close(fig)
