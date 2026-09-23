"""The AUC Compare screen: processed CSV files compared directly.

What is pinned:

* **Same numbers as the study path.** Two files loaded through
  ``load_processed_csv`` give exactly the AUC statistics ``study_compare``
  computes when the same files sit in study folders -- the new screen is a
  different way in, not a different calculation.
* **The RLDA file's quirks.** FAMOS ASCII exports open with blank CRLF lines
  and pad every name to a fixed width; a loader that missed either would find
  no ``Time`` column, or a column called ``'            Time'``.
* **Raw data is refused, and nothing is converted.** ``.raw``/``.dat`` are
  rejected with the reason; values come back exactly as stored -- this does
  not go through ``load_csv``'s N->daN heuristic.
* **Non-wheel channels are compared too**, and channels missing from one side
  are reported rather than silently dropped.
* **The screen sits under Preprocess and needs no study.**
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dtt.analysis.study_compare import (
    auc_channels, auc_summary, channel_differences, csv_channels,
    load_processed_csv, load_study, unit_scale_warning,
)

CSV = Path("csv")
MIXED = CSV / "WFT_Synthetic_Mixed.csv"
ROUGH = CSV / "WFT_Synthetic_Rough.csv"
RLDA = CSV / "RLDA WFT PV data sample.csv"

needs_csv = pytest.mark.skipif(not (MIXED.exists() and ROUGH.exists()),
                               reason="the csv/ sample files are not here")


# ---------------------------------------------------------- same numbers

@needs_csv
def test_csv_path_matches_the_study_path(tmp_path: Path):
    """The brief's check: the AUC values from the CSV loader equal what
    study_compare computes for the same data loaded as studies."""
    studies = []
    for src in (MIXED, ROUGH):
        d = tmp_path / src.stem
        d.mkdir()
        shutil.copyfile(src, d / "processed_data.csv")
        studies.append(d)

    sa, ra, _ = load_study(studies[0])
    sb, rb, _ = load_study(studies[1])
    ca, rca, _ = load_processed_csv(MIXED)
    cb, rcb, _ = load_processed_csv(ROUGH)

    ch_study = auc_channels(ra, rb, sa, sb)
    ch_csv = auc_channels(rca, rcb, ca, cb)
    assert [c[0] for c in ch_csv] == [c[0] for c in ch_study]
    assert len(ch_csv) == 24                    # 4 wheels x Fx..Mz

    by_study = {r.display: r.cmp for r in auc_summary(ch_study, sa, sb)}
    by_csv = {r.display: r.cmp for r in auc_summary(ch_csv, ca, cb)}
    for name, cmp in by_study.items():
        other = by_csv[name]
        for field in ("p5_ref", "p95_ref", "p5_cur", "p95_cur",
                      "pct_exceed_ref_p95", "pct_normal_cur", "pct_normal_ref",
                      "n_ref", "n_cur"):
            assert getattr(other, field) == getattr(cmp, field), (name, field)


@needs_csv
def test_values_are_as_stored_with_no_unit_conversion():
    df, _, _ = load_processed_csv(MIXED)
    raw = pd.read_csv(MIXED, usecols=["FL_Fz"])["FL_Fz"].to_numpy(float)
    assert np.array_equal(df["FL_Fz"].to_numpy(float), raw)


# -------------------------------------------------------------- the quirks

@pytest.fixture
def rlda_head(tmp_path: Path) -> Path:
    """The real RLDA file's first 2,000 data rows, quirks and all."""
    if not RLDA.exists():
        pytest.skip("the RLDA sample file is not here")
    out = tmp_path / "rlda_head.csv"
    with open(RLDA, "rb") as src, open(out, "wb") as dst:
        for i, line in enumerate(src):
            dst.write(line)
            if i >= 2_002:
                break
    return out


def test_rlda_blank_lines_and_padded_names(rlda_head: Path):
    df, rc, name = load_processed_csv(rlda_head)
    assert df.columns[0] == "Time"
    assert all(c == c.strip() for c in df.columns)
    assert "FL_Fz" in df.columns and "Latacc_LPF" in df.columns
    assert len(df) == 2_000
    assert df["Time"].iloc[0] == 0.0
    assert df["FL_Fz"].iloc[0] == pytest.approx(6121.62)
    assert rc.labels == ["FL", "FR", "RL", "RR"]


def test_units_row_is_dropped(tmp_path: Path):
    f = tmp_path / "with_units.csv"
    f.write_text("\r\n\r\n  Time,  FR_Fz,  Latacc\r\n     s,      N,   m/s2\r\n"
                 + "".join(f"{i * 0.01},{7000 + i},{0.1 * i}\r\n" for i in range(50)),
                 encoding="latin1")
    df, _, _ = load_processed_csv(f)
    assert len(df) == 50
    assert df["FR_Fz"].iloc[0] == 7000.0


def test_time_column_is_found_case_insensitively(tmp_path: Path):
    f = tmp_path / "lower.csv"
    f.write_text("time,FR_Fz\n" + "".join(f"{i},{i}\n" for i in range(10)))
    df, _, _ = load_processed_csv(f)
    assert "Time" in df.columns


@pytest.mark.parametrize("name", ["FR_Fz_2.raw", "FR_Fz_2.dat", "notes.txt"])
def test_raw_and_non_csv_are_refused(tmp_path: Path, name: str):
    f = tmp_path / name
    f.write_bytes(b"|CF,2,1,1;")
    with pytest.raises(ValueError) as exc:
        load_processed_csv(f)
    if name.endswith((".raw", ".dat")):
        assert "raw imc data" in str(exc.value)
        assert "processed CSV files only" in str(exc.value)


# ------------------------------------------------------------ the channels

def _frame(cols: dict, n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {"Time": np.arange(n) / 100.0}
    for c, centre in cols.items():
        data[c] = rng.normal(centre, 10, n)
    return pd.DataFrame(data)


def test_non_wheel_channels_are_compared_by_name():
    from dtt.run_channels import build_run_channels
    a = _frame({"FR_Fz": 800, "Latacc": 0, "Vehicle_Speed": 60})
    b = _frame({"WFT_Fz_fr": 810, "Latacc": 0, "Vehicle_Speed": 55})
    ra, rb = build_run_channels(list(a.columns)), build_run_channels(list(b.columns))
    names = [c[0] for c in csv_channels(ra, rb, a, b)]
    assert "FR_Fz" in names                      # matched canonically
    assert "Latacc" in names and "Vehicle_Speed" in names


def test_missing_channels_are_reported_not_dropped():
    from dtt.run_channels import build_run_channels
    a = _frame({"FR_Fz": 800, "FR_Mx": 0, "Latacc": 0})
    b = _frame({"FR_Fz": 800, "Yawrate": 0})
    ra, rb = build_run_channels(list(a.columns)), build_run_channels(list(b.columns))
    only_a, only_b = channel_differences(ra, rb, a, b)
    assert "FR_Mx" in only_a and "Latacc" in only_a
    assert only_b == ["Yawrate"]


def test_a_decade_between_files_is_flagged():
    from dtt.run_channels import build_run_channels
    a = _frame({"FR_Fz": 800, "RR_Fz": 700})
    b = a.copy()
    b[["FR_Fz", "RR_Fz"]] *= 10
    ra = build_run_channels(list(a.columns))
    ch = csv_channels(ra, ra, a, b)
    assert "different units" in unit_scale_warning(ch, a, b, "A", "B")
    assert unit_scale_warning(ch, a, a, "A", "B") is None


# ------------------------------------------------------------- the screen

def test_nav_puts_auc_compare_under_preprocess_and_not_a_study_page():
    pytest.importorskip("PySide6")
    from gui.main_window import NAV_ITEMS, _STUDY_PAGES
    keys = [k for k, _, _ in NAV_ITEMS]
    assert keys[keys.index("preprocess") + 1] == "auc_compare"
    assert "auc_compare" not in _STUDY_PAGES


@needs_csv
def test_page_compares_two_files_with_no_study(tmp_path: Path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from gui.models.repository import StudyRepository
    from gui.pages.auc_compare_page import AucComparePage

    app = QApplication.instance() or QApplication([])
    page = AucComparePage(StudyRepository())
    assert page.study is None
    bad = tmp_path / "rec.raw"
    bad.write_bytes(b"x")
    page.load_paths([str(MIXED), str(ROUGH), str(bad)])
    assert len(page._files) == 2
    assert any("raw imc data" in r for r in page._rejected)

    assert page.label_a.text() == "WFT_Synthetic_Mixed"
    assert page.label_b.text() == "WFT_Synthetic_Rough"
    names = [page.panel.channel_combo.itemText(i)
             for i in range(page.panel.channel_combo.count())]
    assert "FL_Fz" in names and "Latacc" in names
    assert page.panel.auc.axes[2].tables              # the stats strip

    page.label_a.setText("EV")
    page.label_b.setText("IC")
    page._on_labels_changed()
    app.processEvents()
    title = page.panel.auc.canvas.fig._suptitle.get_text()
    assert "EV" in title and "IC" in title
    rows = page.panel.auc_rows()
    assert rows and "P95_EV" in rows[0] and "P95_IC" in rows[0]
    assert "Delta_P95" in rows[0]                     # no unit suffix: unknown
