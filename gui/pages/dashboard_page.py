"""Screen 1 — Dashboard: totals, recent studies, last report, quick start."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QGridLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView,
)

from gui import theme
from gui.pages.base_page import BasePage
from gui.widgets.common import KpiCard, SectionTitle, Card, ScrollPage, status_badge


class DashboardPage(BasePage):
    def __init__(self, repo, parent=None):
        super().__init__(repo, parent)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = ScrollPage()
        outer.addWidget(scroll)
        body = scroll.body()

        # Header row
        head = QHBoxLayout()
        head.addWidget(SectionTitle("Mission Control"))
        head.addStretch(1)
        start_btn = QPushButton("＋  Start New Analysis")
        start_btn.setObjectName("Primary")
        start_btn.setCursor(Qt.PointingHandCursor)
        start_btn.clicked.connect(lambda: self.navigate_requested.emit("new_study"))
        head.addWidget(start_btn)
        body.addLayout(head)

        # KPI cards
        kpi_row = QGridLayout()
        kpi_row.setSpacing(16)
        self.kpi_total   = KpiCard("Total Studies", "0", "Processed in outputs/")
        self.kpi_reports = KpiCard("Reports", "0", "PowerPoint deliverables")
        self.kpi_figures = KpiCard("Figures", "0", "Across all studies")
        self.kpi_latest  = KpiCard("Latest Study", "—", "Most recently updated")
        for i, c in enumerate(
            (self.kpi_total, self.kpi_reports, self.kpi_figures, self.kpi_latest)):
            kpi_row.addWidget(c, 0, i)
        body.addLayout(kpi_row)

        # Recent studies table
        body.addWidget(SectionTitle("Recent Studies"))
        table_card = Card()
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Study", "Vehicle", "Date", "Figures", "Status"])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.cellDoubleClicked.connect(self._open_selected)
        table_card.layout().addWidget(self.table)
        body.addWidget(table_card)

        # Last report
        body.addWidget(SectionTitle("Last Generated Report"))
        report_card = Card()
        rc = QHBoxLayout()
        self.report_label = QLabel("No reports generated yet.")
        self.report_label.setStyleSheet(f"color:{theme.TEXT_MUTED};")
        rc.addWidget(self.report_label)
        rc.addStretch(1)
        self.open_report_btn = QPushButton("Open Report")
        self.open_report_btn.setObjectName("Secondary")
        self.open_report_btn.clicked.connect(self._open_latest_report)
        self.open_report_btn.setEnabled(False)
        rc.addWidget(self.open_report_btn)
        report_card.layout().addLayout(rc)
        body.addWidget(report_card)
        body.addStretch(1)

        self._latest_report: Path | None = None

    # Data
    def refresh(self) -> None:
        studies = self.repo.list_studies()
        total_figs = sum(s.figure_count for s in studies)
        total_reports = sum(len(s.report_files) for s in studies)

        self.kpi_total.set_value(str(len(studies)))
        self.kpi_reports.set_value(str(total_reports))
        self.kpi_figures.set_value(str(total_figs))
        self.kpi_latest.set_value(
            studies[0].name if studies else "—",
            studies[0].modified.strftime("%Y-%m-%d %H:%M") if studies else "")

        self.table.setRowCount(0)
        for s in studies[:12]:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(s.name))
            self.table.setItem(r, 1, QTableWidgetItem(s.vehicle))
            self.table.setItem(r, 2, QTableWidgetItem(s.report_date))
            self.table.setItem(r, 3, QTableWidgetItem(str(s.figure_count)))
            self.table.setCellWidget(r, 4, status_badge(s.status))

        self._latest_report = self.repo.latest_report()
        if self._latest_report:
            self.report_label.setText(self._latest_report.name)
            self.report_label.setStyleSheet(f"color:{theme.TEXT};")
            self.open_report_btn.setEnabled(True)
        else:
            self.report_label.setText("No reports generated yet.")
            self.open_report_btn.setEnabled(False)

    # Actions
    def _open_selected(self, row: int, _col: int) -> None:
        item = self.table.item(row, 0)
        if item:
            self.navigate_requested.emit(f"open:{item.text()}")

    def _open_latest_report(self) -> None:
        if self._latest_report:
            _open_path(self._latest_report)


def _open_path(path: Path) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))            # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass
