"""Compare studies no longer carries the "All channels — at a glance" chart.

Round 9 removed it from that page (on screen and from its "Export all…"); the
AUC Compare page, which shares the widget, keeps it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _texts(widget):
    from PySide6.QtWidgets import QLabel
    return [lbl.text() for lbl in widget.findChildren(QLabel)]


def test_compare_studies_page_has_no_all_channels_chart():
    _app()
    from gui.models.repository import StudyRepository
    from gui.pages.study_compare_page import StudyComparePage

    page = StudyComparePage(StudyRepository())
    assert page.panel.summary_plot is None
    assert not any("at a glance" in t for t in _texts(page))


def test_auc_compare_page_keeps_it():
    _app()
    from gui.widgets.auc_compare_panel import AucComparePanel

    panel = AucComparePanel(labels=lambda: ("A", "B"), start_dir=lambda: ".")
    assert panel.summary_plot is not None
    assert any("at a glance" in t for t in _texts(panel))


def test_batch_export_without_the_chart(tmp_path):
    from dtt.analysis.study_compare import export_auc_batch

    rng = np.random.default_rng(0)
    df_a = pd.DataFrame({"FR_Fx": rng.normal(0, 50, 4000)})
    df_b = pd.DataFrame({"FR_Fx": rng.normal(5, 50, 4000)})
    pngs, csv_path, summary_png = export_auc_batch(
        [("FR_Fx", "FR_Fx", "FR_Fx")], df_a, df_b, "A", "B", tmp_path,
        summary_chart=False)
    assert summary_png is None
    assert len(pngs) == 1 and csv_path.exists()
    assert not list(tmp_path.glob("auc_all_channels_*"))
