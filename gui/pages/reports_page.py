"""Screen 10 — Reports: list PPTX, open folder, export, regenerate."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QPushButton, QFileDialog,
)

from gui import theme
from gui.pages.base_page import BasePage
from gui.widgets.common import SectionTitle, Card, ScrollPage


class ReportsPage(BasePage):
    regenerate_requested = Signal(object)        # Study

    def __init__(self, repo, parent=None):
        super().__init__(repo, parent)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = ScrollPage()
        outer.addWidget(scroll)
        body = scroll.body()

        head = QHBoxLayout()
        head.addWidget(SectionTitle("Reports"))
        head.addStretch(1)
        self.folder_btn = QPushButton("Open Study Folder")
        self.folder_btn.setObjectName("Secondary")
        self.folder_btn.clicked.connect(self._open_folder)
        self.regen_btn = QPushButton("⟳ Regenerate Report")
        self.regen_btn.setObjectName("Primary")
        self.regen_btn.clicked.connect(self._regen)
        head.addWidget(self.folder_btn)
        head.addWidget(self.regen_btn)
        body.addLayout(head)

        card = Card()
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Report", "Size (KB)", "Modified", "Action"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        card.layout().addWidget(self.table)
        body.addWidget(card)

        self.empty = QLabel("No PowerPoint report for this study yet.")
        self.empty.setStyleSheet(f"color:{theme.TEXT_FAINT};")
        body.addWidget(self.empty)
        body.addStretch(1)

    def refresh(self) -> None:
        self.table.setRowCount(0)
        reports = self.study.report_files if self.study else []
        self.empty.setVisible(not reports)
        self.regen_btn.setEnabled(self.study is not None)
        self.folder_btn.setEnabled(self.study is not None)
        from datetime import datetime
        for rpt in reports:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(rpt.name))
            self.table.setItem(r, 1, QTableWidgetItem(f"{rpt.stat().st_size/1024:.0f}"))
            self.table.setItem(r, 2, QTableWidgetItem(
                datetime.fromtimestamp(rpt.stat().st_mtime).strftime("%Y-%m-%d %H:%M")))
            open_btn = QPushButton("Open")
            open_btn.setObjectName("Secondary")
            open_btn.clicked.connect(lambda _=False, p=rpt: _open_path(p))
            exp_btn = QPushButton("Export")
            exp_btn.setObjectName("Secondary")
            exp_btn.clicked.connect(lambda _=False, p=rpt: self._export(p))
            from PySide6.QtWidgets import QWidget
            holder = QWidget()
            hl = QHBoxLayout(holder)
            hl.setContentsMargins(4, 2, 4, 2)
            hl.addWidget(open_btn)
            hl.addWidget(exp_btn)
            self.table.setCellWidget(r, 3, holder)

    def _open_folder(self) -> None:
        if self.study:
            _open_path(self.study.path)

    def _export(self, rpt: Path) -> None:
        dst, _ = QFileDialog.getSaveFileName(
            self, "Export report", rpt.name, "PowerPoint (*.pptx)")
        if dst:
            shutil.copyfile(rpt, dst)

    def _regen(self) -> None:
        if self.study:
            self.regenerate_requested.emit(self.study)


def _open_path(path: Path) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))                # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass
