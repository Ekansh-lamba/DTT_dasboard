"""Screen 4 — Validation Results, sourced from validation_report.json."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QGridLayout, QLabel, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView,
)

from gui import theme
from gui.pages.base_page import BasePage
from gui.widgets.common import SectionTitle, Card, ScrollPage, KpiCard, Badge


class ValidationPage(BasePage):
    def __init__(self, repo, parent=None):
        super().__init__(repo, parent)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = ScrollPage()
        outer.addWidget(scroll)
        body = scroll.body()

        head = QHBoxLayout()
        head.addWidget(SectionTitle("Validation Report"))
        head.addStretch(1)
        self.valid_badge = Badge("—", theme.TEXT_MUTED)
        head.addWidget(self.valid_badge)
        body.addLayout(head)

        # Dataset summary KPIs
        self.kpi_grid = QGridLayout()
        self.kpi_grid.setSpacing(16)
        self.kpi_rows     = KpiCard("Rows", "—")
        self.kpi_cols     = KpiCard("Columns", "—")
        self.kpi_duration = KpiCard("Duration", "—")
        self.kpi_sr       = KpiCard("Sampling Rate", "—")
        self.kpi_nan      = KpiCard("Total NaN Rows", "—")
        self.kpi_dupes    = KpiCard("Duplicate Cols", "—")
        for i, c in enumerate((self.kpi_rows, self.kpi_cols, self.kpi_duration,
                               self.kpi_sr, self.kpi_nan, self.kpi_dupes)):
            self.kpi_grid.addWidget(c, i // 3, i % 3)
        body.addLayout(self.kpi_grid)

        # Source file
        self.source_card = Card()
        self.source_label = QLabel("—")
        self.source_label.setStyleSheet("font-family:monospace;")
        self.source_card.layout().addWidget(_mini("Source File"))
        self.source_card.layout().addWidget(self.source_label)
        body.addWidget(self.source_card)

        # Channel status table
        body.addWidget(SectionTitle("Required Channel Status"))
        chan_card = Card()
        self.chan_table = QTableWidget(0, 4)
        self.chan_table.setHorizontalHeaderLabels(["Channel", "Wheel", "Axis", "Status"])
        self.chan_table.verticalHeader().setVisible(False)
        self.chan_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.chan_table.setAlternatingRowColors(True)
        self.chan_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        chan_card.layout().addWidget(self.chan_table)
        body.addWidget(chan_card)

        # Missing channels / warnings
        body.addWidget(SectionTitle("Missing Channels & Warnings"))
        self.issues_card = Card()
        self.issues_label = QLabel("—")
        self.issues_label.setWordWrap(True)
        self.issues_card.layout().addWidget(self.issues_label)
        body.addWidget(self.issues_card)

        # Optional channels
        body.addWidget(SectionTitle("Optional Channels Present"))
        self.optional_card = Card()
        self.optional_label = QLabel("—")
        self.optional_label.setWordWrap(True)
        self.optional_label.setStyleSheet("font-family:monospace;")
        self.optional_card.layout().addWidget(self.optional_label)
        body.addWidget(self.optional_card)
        body.addStretch(1)

    def refresh(self) -> None:
        if not self.study or not self.study.has_validation:
            self._set_empty()
            return
        v = self.study.validation()

        self.kpi_rows.set_value(f"{v.get('rows', 0):,}")
        self.kpi_cols.set_value(str(v.get("columns", "—")))
        dur = v.get("duration_s")
        self.kpi_duration.set_value(f"{dur:,.0f} s" if isinstance(dur, (int, float)) else "—")
        sr = v.get("sampling_rate_hz")
        self.kpi_sr.set_value(f"{sr:g} Hz" if isinstance(sr, (int, float)) else "—")
        nan = v.get("total_nan_rows", 0)
        self.kpi_nan.set_value(f"{nan:,}")
        dupes = v.get("duplicate_columns", [])
        self.kpi_dupes.set_value(str(len(dupes)))

        self.source_label.setText(v.get("file_name", "—"))

        is_valid = bool(v.get("is_valid"))
        self.valid_badge.set_status(
            "VALID" if is_valid else "INVALID",
            theme.SUCCESS if is_valid else theme.DANGER)

        # Channel table
        status = v.get("channel_status", {})
        self.chan_table.setRowCount(0)
        for ch, st in status.items():
            r = self.chan_table.rowCount()
            self.chan_table.insertRow(r)
            parts = ch.split("_")
            wheel = parts[0] if parts else ch
            axis = parts[-1] if len(parts) > 1 else "—"
            self.chan_table.setItem(r, 0, QTableWidgetItem(ch))
            self.chan_table.setItem(r, 1, QTableWidgetItem(wheel))
            self.chan_table.setItem(r, 2, QTableWidgetItem(axis))
            item = QTableWidgetItem(str(st))
            ok = str(st).upper() == "PRESENT"
            item.setForeground(Qt.green if ok else Qt.red)
            self.chan_table.setItem(r, 3, item)

        # Issues
        missing = v.get("missing_channels", [])
        dup_ts = v.get("duplicate_timestamps")
        nan_counts = v.get("nan_counts", {})
        warnings = v.get("warnings", [])
        lines = []
        lines.append(
            f"Missing channels: {', '.join(missing) if missing else 'none'}")
        if dup_ts is not None:
            lines.append(f"Duplicate timestamps: {dup_ts}")
        if nan_counts:
            lines.append("NaN counts per channel: " +
                         ", ".join(f"{k}={val}" for k, val in nan_counts.items()))
        else:
            lines.append("NaN counts per channel: none")
        if warnings:
            lines.append("Warnings:\n  • " + "\n  • ".join(map(str, warnings)))
        else:
            lines.append("Warnings: none")
        self.issues_label.setText("\n".join(lines))

        optional = v.get("optional_present", [])
        self.optional_label.setText(
            "   ".join(optional) if optional else "None present")

    def _set_empty(self) -> None:
        for kpi in (self.kpi_rows, self.kpi_cols, self.kpi_duration,
                    self.kpi_sr, self.kpi_nan, self.kpi_dupes):
            kpi.set_value("—")
        self.source_label.setText("No validation_report.json for this study.")
        self.valid_badge.set_status("—", theme.TEXT_MUTED)
        self.chan_table.setRowCount(0)
        self.issues_label.setText("—")
        self.optional_label.setText("—")


def _mini(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight:700; font-size:12px; color:" + theme.TEXT_MUTED)
    return lbl
