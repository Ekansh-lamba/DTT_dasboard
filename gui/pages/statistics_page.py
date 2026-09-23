"""Screen 5 — Statistics: searchable channel table from stats_summary.json.

Channels are listed the way the FAMOS-style CSV export lays them out (see
:mod:`dtt.famos_csv`): standard names (FR_Fx, not FR_Fx_2), A-Z, every
channel of the processed data with its unit -- and Export CSV writes that same
layout.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QComboBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QPushButton, QFileDialog,
)

from gui import theme
from gui.pages.base_page import BasePage
from gui.widgets.common import SectionTitle, Card

_COLS = ["Channel", "Unit", "Mean", "Median", "Std", "Min", "Max", "P80", "P90", "P95"]


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
        unit = QLabel("All channels · forces in daN")
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
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        for i in range(2, len(_COLS)):
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
        wheels = ["All wheels"] + (self.study.wheel_labels() or [])
        if [self.wheel_filter.itemText(i) for i in range(self.wheel_filter.count())] != wheels:
            self.wheel_filter.blockSignals(True)
            self.wheel_filter.clear()
            self.wheel_filter.addItems(wheels)
            self.wheel_filter.blockSignals(False)
        for cs in stats.values():
            r = self.table.rowCount()
            self.table.insertRow(r)
            name = QTableWidgetItem(cs.label)
            name.setToolTip(f"Column in processed_data.csv: {cs.channel}")
            self.table.setItem(r, 0, name)
            self.table.setItem(r, 1, QTableWidgetItem(cs.unit))
            values = [cs.mean, cs.median, cs.std, cs.min, cs.max, cs.p80, cs.p90, cs.p95]
            for c, val in enumerate(values, start=2):
                item = _NumericItem(f"{val:.3f}" if val == val else "—")
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                # Tint Fz vertical-load rows for quick scanning
                if cs.signal == "Fz" and cs.wheel in wheels:
                    item.setForeground(Qt.cyan)
                self.table.setItem(r, c, item)
        # keep the CSV's order (Time aside, A-Z) until a header is clicked;
        # Qt otherwise re-sorts on its default indicator, Z-A
        self.table.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
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
        from dtt.famos_csv import write_stats_csv
        default = str(self.study.path / f"WFT_Statistics_{self.study.name}.csv")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export statistics", default, "CSV files (*.csv)")
        if not path:
            return
        write_stats_csv(self.study.stats_raw(), path)
