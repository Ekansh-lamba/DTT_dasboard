"""Screen 4 — Validation Results, sourced from validation_report.json."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QGridLayout, QLabel, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QPushButton, QFileDialog,
)

from gui import theme
from gui.pages.base_page import BasePage
from gui.widgets.common import SectionTitle, Card, ScrollPage, KpiCard, Badge


class _FamosWorker(QThread):
    """Runs the FAMOS cross-check off the UI thread (CSV read + filter sweep)."""
    done = Signal(object, object)     # (matches, note)
    failed = Signal(str)

    def __init__(self, csv_path: str):
        super().__init__()
        self.csv_path = csv_path

    def run(self):
        try:
            from dtt.validation.famos_validation import crosscheck_csv
            matches, note = crosscheck_csv(self.csv_path)
            self.done.emit(matches, note)
        except Exception as exc:                      # surface, don't crash
            self.failed.emit(f"{type(exc).__name__}: {exc}")


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

        # FAMOS cross-check
        fam_head = QHBoxLayout()
        fam_head.addWidget(SectionTitle("FAMOS Cross-Check"))
        fam_head.addStretch(1)
        self.famos_badge = Badge("—", theme.TEXT_MUTED)
        fam_head.addWidget(self.famos_badge)
        body.addLayout(fam_head)

        self.famos_card = Card()
        intro = QLabel(
            "Runs our low-pass filter on a FAMOS export's raw channels and scores "
            "the result against FAMOS's own filtered columns (e.g. Latacc vs "
            "Latacc_LPF), sample-by-sample. ≥ 95% match = FAMOS-grade.")
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:12px;")
        self.famos_card.layout().addWidget(intro)

        row = QHBoxLayout()
        self.famos_btn = QPushButton("Validate against FAMOS CSV…")
        self.famos_btn.setObjectName("Primary")
        self.famos_btn.clicked.connect(self._run_famos)
        row.addWidget(self.famos_btn)
        self.famos_status = QLabel("")
        self.famos_status.setStyleSheet(f"color:{theme.TEXT_MUTED};")
        row.addWidget(self.famos_status, 1)
        self.famos_card.layout().addLayout(row)

        self.famos_table = QTableWidget(0, 5)
        self.famos_table.setHorizontalHeaderLabels(
            ["Channel", "Reference", "Match %", "Corr %", "Operation"])
        self.famos_table.verticalHeader().setVisible(False)
        self.famos_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.famos_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.famos_table.setMaximumHeight(200)
        self.famos_card.layout().addWidget(self.famos_table)
        body.addWidget(self.famos_card)
        body.addStretch(1)

        self._famos_worker: _FamosWorker | None = None

    # FAMOS cross-check
    def _run_famos(self) -> None:
        start_dir = str(self.repo.csv_dir) if hasattr(self.repo, "csv_dir") else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a FAMOS CSV export", start_dir, "CSV files (*.csv)")
        if not path:
            return
        self.famos_btn.setEnabled(False)
        self.famos_status.setText(f"Validating {Path(path).name} …")
        self.famos_badge.set_status("RUNNING", theme.ORANGE)
        self.famos_table.setRowCount(0)
        self._famos_worker = _FamosWorker(path)
        self._famos_worker.done.connect(self._on_famos_done)
        self._famos_worker.failed.connect(self._on_famos_failed)
        self._famos_worker.start()

    def _on_famos_failed(self, msg: str) -> None:
        self.famos_btn.setEnabled(True)
        self.famos_status.setText(f"Failed: {msg}")
        self.famos_badge.set_status("ERROR", theme.DANGER)

    def _on_famos_done(self, matches, note) -> None:
        self.famos_btn.setEnabled(True)
        if note:
            self.famos_status.setText(note)
            self.famos_badge.set_status("NO REF", theme.WARNING)
            return
        for m in matches:
            r = self.famos_table.rowCount()
            self.famos_table.insertRow(r)
            self.famos_table.setItem(r, 0, QTableWidgetItem(m.channel))
            self.famos_table.setItem(r, 1, QTableWidgetItem(m.reference))
            mi = QTableWidgetItem(f"{m.match_pct:.2f}%")
            mi.setForeground(Qt.green if m.passed else Qt.red)
            self.famos_table.setItem(r, 2, mi)
            self.famos_table.setItem(r, 3, QTableWidgetItem(f"{m.corr_pct:.2f}%"))
            # The pair may be related by smo() rather than FiltLP(); showing a
            # cutoff/order for a smoothing match would read as "0 Hz · order 0".
            if getattr(m, "operation", "FiltLP") == "smo":
                op = f"smo({m.best_smooth_s:g} s)"
            else:
                op = f"FiltLP {m.best_cutoff:g} Hz · order {m.best_order}"
            self.famos_table.setItem(r, 4, QTableWidgetItem(op))
        if matches:
            best = max(m.match_pct for m in matches)
            passed = all(m.passed for m in matches)
            self.famos_status.setText(
                f"{len(matches)} channel(s) checked · best match {best:.2f}% "
                f"vs FAMOS · threshold 95%.")
            self.famos_badge.set_status(
                "FAMOS-GRADE" if passed else "BELOW 95%",
                theme.SUCCESS if passed else theme.DANGER)

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
