"""Screen 5 — Statistics: searchable channel table from stats_summary.json."""

from __future__ import annotations

import csv

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QComboBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QPushButton, QFileDialog,
)

from gui import theme
from gui.pages.base_page import BasePage
from gui.widgets.common import SectionTitle, Card

_COLS = ["Channel", "Mean", "Median", "Std", "Min", "Max", "P80", "P90", "P95"]


class _NumericItem(QTableWidgetItem):
    """Sort numerically rather than lexically."""
    def __lt__(self, other):                       # noqa: D401
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return super().__lt__(other)


class StatisticsPage(BasePage):
    def __init__(self, repo, parent=None):
        super().__init__(repo, parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)

        head = QHBoxLayout()
        head.addWidget(SectionTitle("Statistical Summary"))
        head.addStretch(1)
        unit = QLabel("Force metrics")
        unit.setStyleSheet(f"color:{theme.TEXT_MUTED};")
        head.addWidget(unit)
        root.addLayout(head)

        # Filters
        filt = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search channel… (e.g. FL, Fz)")
        self.search.textChanged.connect(self._apply_filter)
        self.wheel_filter = QComboBox()
        self.wheel_filter.addItems(["All wheels", "FL", "FR", "RL", "RR"])
        self.wheel_filter.currentTextChanged.connect(self._apply_filter)
        self.export_btn = QPushButton("Export CSV")
        self.export_btn.setObjectName("Secondary")
        self.export_btn.clicked.connect(self._export)
        filt.addWidget(self.search, 1)
        filt.addWidget(self.wheel_filter)
        filt.addWidget(self.export_btn)
        root.addLayout(filt)

        card = Card()
        self.table = QTableWidget(0, len(_COLS))
        self.table.setHorizontalHeaderLabels(_COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for i in range(1, len(_COLS)):
            hdr.setSectionResizeMode(i, QHeaderView.Stretch)
        card.layout().addWidget(self.table)
        root.addWidget(card, 1)

        self.empty_label = QLabel("No stats_summary.json for this study.")
        self.empty_label.setStyleSheet(f"color:{theme.TEXT_FAINT};")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.hide()
        root.addWidget(self.empty_label)

    def refresh(self) -> None:
        stats = self.study.stats() if self.study else {}
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        if not stats:
            self.empty_label.show()
            self.table.hide()
            return
        self.empty_label.hide()
        self.table.show()
        for cs in stats.values():
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(cs.channel))
            values = [cs.mean, cs.median, cs.std, cs.min, cs.max, cs.p80, cs.p90, cs.p95]
            for c, val in enumerate(values, start=1):
                item = _NumericItem(f"{val:.3f}" if val == val else "—")
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                # Tint Fz vertical-load rows for quick scanning
                if cs.signal == "Fz":
                    item.setForeground(Qt.cyan)
                self.table.setItem(r, c, item)
        self.table.setSortingEnabled(True)
        self._apply_filter()

    def _apply_filter(self) -> None:
        text = self.search.text().strip().lower()
        wheel = self.wheel_filter.currentText()
        for r in range(self.table.rowCount()):
            ch_item = self.table.item(r, 0)
            ch = ch_item.text() if ch_item else ""
            show = text in ch.lower()
            if wheel != "All wheels":
                show = show and ch.startswith(wheel)
            self.table.setRowHidden(r, not show)

    def _export(self) -> None:
        if not self.study or not self.study.stats():
            return
        default = str(self.study.path / f"{self.study.name}_stats.csv")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export statistics", default, "CSV files (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(_COLS)
            for cs in self.study.stats().values():
                w.writerow(cs.as_row())
