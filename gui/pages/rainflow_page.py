"""Screen 9 — Rainflow: matrix images, cycle CSV summaries, export."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QPushButton, QFileDialog,
)

from gui import theme
from gui.pages.base_page import BasePage
from gui.widgets.common import SectionTitle, Card, ScrollPage, FigureGrid
from gui.widgets.image_viewer import ImageViewerDialog
from gui.models.repository import WHEELS


class RainflowPage(BasePage):
    def __init__(self, repo, parent=None):
        super().__init__(repo, parent)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = ScrollPage()
        outer.addWidget(scroll)
        body = scroll.body()

        head = QHBoxLayout()
        head.addWidget(SectionTitle("Rainflow Analysis"))
        head.addStretch(1)
        self.export_btn = QPushButton("Export all cycle CSVs…")
        self.export_btn.setObjectName("Secondary")
        self.export_btn.clicked.connect(self._export_all)
        head.addWidget(self.export_btn)
        body.addLayout(head)

        body.addWidget(_mini("Rainflow Matrices"))
        self.grid = FigureGrid(columns=2, on_open=self._open)
        body.addWidget(self.grid)

        body.addWidget(_mini("Cycle Summary (per channel)"))
        card = Card()
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["File", "Channel", "Cycle Rows", "Size (KB)"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.cellDoubleClicked.connect(self._export_row)
        card.layout().addWidget(self.table)
        body.addWidget(card)
        hint = QLabel("Double-click a row to export that channel's cycle CSV.")
        hint.setStyleSheet(f"color:{theme.TEXT_FAINT}; font-size:11px;")
        body.addWidget(hint)
        body.addStretch(1)

        self._csvs: list[Path] = []

    def refresh(self) -> None:
        if not self.study:
            self.grid.set_figures([])
            self.table.setRowCount(0)
            return
        self.grid.set_figures(
            [(f"Rainflow — {w}", self.study.rainflow_image(w)) for w in WHEELS])

        self._csvs = self.study.rainflow_cycle_csvs()
        self.table.setRowCount(0)
        for path in self._csvs:
            r = self.table.rowCount()
            self.table.insertRow(r)
            channel = path.stem.replace("rainflow_cycles_", "")
            self.table.setItem(r, 0, QTableWidgetItem(path.name))
            self.table.setItem(r, 1, QTableWidgetItem(channel))
            self.table.setItem(r, 2, QTableWidgetItem(str(_count_rows(path))))
            kb = QTableWidgetItem(f"{path.stat().st_size / 1024:.1f}")
            kb.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(r, 3, kb)

    def _open(self, path) -> None:
        ImageViewerDialog.show_for(path, self)

    def _export_row(self, row: int, _col: int) -> None:
        if 0 <= row < len(self._csvs):
            src = self._csvs[row]
            dst, _ = QFileDialog.getSaveFileName(
                self, "Export cycle CSV", src.name, "CSV files (*.csv)")
            if dst:
                shutil.copyfile(src, dst)

    def _export_all(self) -> None:
        if not self._csvs:
            return
        folder = QFileDialog.getExistingDirectory(self, "Choose export folder")
        if not folder:
            return
        for src in self._csvs:
            shutil.copyfile(src, Path(folder) / src.name)


def _count_rows(path: Path) -> int:
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            return max(sum(1 for _ in csv.reader(fh)) - 1, 0)
    except OSError:
        return 0


def _mini(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight:700; font-size:13px;")
    return lbl
