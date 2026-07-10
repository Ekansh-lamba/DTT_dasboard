"""Screen 11 — Study History: all studies with actions."""

from __future__ import annotations

import os
import subprocess
import sys

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QPushButton, QWidget, QMessageBox,
)

from gui import theme
from gui.pages.base_page import BasePage
from gui.widgets.common import SectionTitle, Card, status_badge


class HistoryPage(BasePage):
    open_study_requested = Signal(str)           # study name
    studies_changed = Signal()                   # emitted after a delete

    def __init__(self, repo, parent=None):
        super().__init__(repo, parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)

        head = QHBoxLayout()
        head.addWidget(SectionTitle("Study History"))
        head.addStretch(1)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search studies…")
        self.search.setMaximumWidth(260)
        self.search.textChanged.connect(self._apply_filter)
        refresh = QPushButton("⟳ Refresh")
        refresh.setObjectName("Secondary")
        refresh.clicked.connect(self.refresh)
        head.addWidget(self.search)
        head.addWidget(refresh)
        root.addLayout(head)

        card = Card()
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Study Name", "Vehicle", "Date", "Status", "Actions"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Stretch)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.cellDoubleClicked.connect(self._open_row)
        card.layout().addWidget(self.table)
        root.addWidget(card, 1)

    def refresh(self) -> None:
        studies = self.repo.list_studies()
        self.table.setRowCount(0)
        for s in studies:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(s.name))
            self.table.setItem(r, 1, QTableWidgetItem(s.vehicle))
            self.table.setItem(r, 2, QTableWidgetItem(s.report_date))
            self.table.setCellWidget(r, 3, status_badge(s.status))

            holder = QWidget()
            hl = QHBoxLayout(holder)
            hl.setContentsMargins(4, 2, 4, 2)
            hl.setSpacing(6)
            open_btn = QPushButton("Open")
            open_btn.setObjectName("Secondary")
            open_btn.clicked.connect(lambda _=False, n=s.name: self.open_study_requested.emit(n))
            folder_btn = QPushButton("Folder")
            folder_btn.setObjectName("Secondary")
            folder_btn.clicked.connect(lambda _=False, p=s.path: _open_path(p))
            delete_btn = QPushButton("Delete")
            delete_btn.setObjectName("Secondary")
            delete_btn.setStyleSheet(
                f"color:{theme.DANGER}; border-color:{theme.DANGER}55;")
            delete_btn.clicked.connect(lambda _=False, n=s.name: self._delete_study(n))
            hl.addWidget(open_btn)
            hl.addWidget(folder_btn)
            hl.addWidget(delete_btn)
            self.table.setCellWidget(r, 4, holder)
        self._apply_filter()

    def _delete_study(self, name: str) -> None:
        resp = QMessageBox.question(
            self, "Delete study",
            f"Permanently delete study '{name}' and all its outputs?\n"
            "This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if resp != QMessageBox.Yes:
            return
        if self.repo.delete_study(name):
            self.refresh()
            self.studies_changed.emit()
        else:
            QMessageBox.warning(self, "Delete failed",
                                f"Could not delete '{name}'. It may be open elsewhere.")

    def _apply_filter(self) -> None:
        text = self.search.text().strip().lower()
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            name = item.text().lower() if item else ""
            veh_item = self.table.item(r, 1)
            veh = veh_item.text().lower() if veh_item else ""
            self.table.setRowHidden(r, text not in name and text not in veh)

    def _open_row(self, row: int, _col: int) -> None:
        item = self.table.item(row, 0)
        if item:
            self.open_study_requested.emit(item.text())


def _open_path(path) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))                # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass
