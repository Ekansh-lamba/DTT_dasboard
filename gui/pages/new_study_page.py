"""Screen 2 — New Study: choose CSV, configure filtering/analysis, launch."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QFormLayout, QLineEdit, QComboBox, QPushButton,
    QDoubleSpinBox, QSpinBox, QCheckBox, QLabel, QFileDialog, QGridLayout,
    QFrame,
)

from gui import theme
from gui.pages.base_page import BasePage
from gui.widgets.common import SectionTitle, Card, ScrollPage
from gui.workers.pipeline_worker import RunRequest


class NewStudyPage(BasePage):
    # Emitted with a fully-built RunRequest when the user starts an analysis.
    start_requested = Signal(object)

    def __init__(self, repo, parent=None):
        super().__init__(repo, parent)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = ScrollPage()
        outer.addWidget(scroll)
        body = scroll.body()

        body.addWidget(SectionTitle("New Study"))
        sub = QLabel("Configure and launch the DTT WFT analysis pipeline.")
        sub.setStyleSheet(f"color:{theme.TEXT_MUTED};")
        body.addWidget(sub)

        columns = QHBoxLayout()
        columns.setSpacing(18)
        body.addLayout(columns)

        # ── Left: source + identity ──────────────────────────────────────────
        left = Card()
        left.layout().addWidget(_card_title("Data Source"))
        form = QFormLayout()
        form.setSpacing(12)
        form.setLabelAlignment(Qt.AlignLeft)

        csv_row = QHBoxLayout()
        self.csv_combo = QComboBox()
        self.csv_combo.setMinimumWidth(260)
        browse = QPushButton("Browse…")
        browse.setObjectName("Secondary")
        browse.clicked.connect(self._browse_csv)
        csv_row.addWidget(self.csv_combo, 1)
        csv_row.addWidget(browse)
        csv_wrap = QFrame(); csv_wrap.setLayout(csv_row)
        form.addRow("CSV file", csv_wrap)

        self.vehicle_edit = QLineEdit("RLDA_PV")
        form.addRow("Vehicle name", self.vehicle_edit)

        self.study_edit = QLineEdit()
        self.study_edit.setPlaceholderText("Leave blank → timestamp")
        form.addRow("Study name", self.study_edit)
        left.layout().addLayout(form)
        columns.addWidget(left, 1)

        # ── Right: filtering + modules ───────────────────────────────────────
        right = Card()
        right.layout().addWidget(_card_title("Filtering"))
        fform = QFormLayout()
        fform.setSpacing(12)

        self.filter_check = QCheckBox("Apply Butterworth low-pass filter")
        self.filter_check.setChecked(True)
        self.filter_check.toggled.connect(self._toggle_filter)
        right.layout().addWidget(self.filter_check)

        self.cutoff_spin = QDoubleSpinBox()
        self.cutoff_spin.setRange(0.1, 1000.0)
        self.cutoff_spin.setValue(10.0)
        self.cutoff_spin.setSuffix(" Hz")
        fform.addRow("Cutoff frequency", self.cutoff_spin)

        self.order_spin = QSpinBox()
        self.order_spin.setRange(1, 12)
        self.order_spin.setValue(4)
        fform.addRow("Filter order", self.order_spin)

        self.miner_spin = QDoubleSpinBox()
        self.miner_spin.setRange(1.0, 20.0)
        self.miner_spin.setSingleStep(0.5)
        self.miner_spin.setValue(8.0)
        fform.addRow("Miner's exponent", self.miner_spin)
        right.layout().addLayout(fform)

        right.layout().addWidget(_card_title("Analysis Modules"))
        mod_grid = QGridLayout()
        mod_grid.setSpacing(8)
        self.module_checks = {}
        modules = [
            "Statistics", "Histograms", "Heatmaps",
            "Boxplots", "Rainflow", "PowerPoint Report",
        ]
        for i, m in enumerate(modules):
            cb = QCheckBox(m)
            cb.setChecked(True)
            cb.setEnabled(False)            # backend runs full pipeline
            cb.setToolTip("The backend pipeline runs all modules in one pass.")
            self.module_checks[m] = cb
            mod_grid.addWidget(cb, i // 2, i % 2)
        right.layout().addLayout(mod_grid)
        note = QLabel("Modules reflect the backend's fixed 9-stage pipeline.")
        note.setStyleSheet(f"color:{theme.TEXT_FAINT}; font-size:11px;")
        right.layout().addWidget(note)
        columns.addWidget(right, 1)

        # ── Launch ───────────────────────────────────────────────────────────
        launch_row = QHBoxLayout()
        launch_row.addStretch(1)
        self.start_btn = QPushButton("▶  Start Analysis")
        self.start_btn.setObjectName("Primary")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.clicked.connect(self._on_start)
        launch_row.addWidget(self.start_btn)
        body.addLayout(launch_row)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet(f"color:{theme.DANGER};")
        body.addWidget(self.error_label)
        body.addStretch(1)

        self._browsed_path: Path | None = None
        self.reload_csvs()

    # ── Helpers ─────────────────────────────────────────────────────────────
    def reload_csvs(self) -> None:
        self.csv_combo.clear()
        files = self.repo.list_csv_files()
        if not files:
            self.csv_combo.addItem("No CSV files in csv/ — use Browse…", None)
        for f in files:
            self.csv_combo.addItem(f.name, str(f))

    def _toggle_filter(self, on: bool) -> None:
        self.cutoff_spin.setEnabled(on)
        self.order_spin.setEnabled(on)

    def _browse_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select WFT CSV", str(self.repo.csv_dir), "CSV files (*.csv)")
        if path:
            self._browsed_path = Path(path)
            self.csv_combo.insertItem(0, Path(path).name + "  (browsed)", path)
            self.csv_combo.setCurrentIndex(0)

    def _selected_csv(self) -> Path | None:
        data = self.csv_combo.currentData()
        return Path(data) if data else None

    def _on_start(self) -> None:
        self.error_label.setText("")
        csv_path = self._selected_csv()
        if not csv_path or not csv_path.exists():
            self.error_label.setText("Please select a valid CSV file.")
            return
        req = RunRequest(
            csv_path=csv_path,
            vehicle=self.vehicle_edit.text().strip() or "Vehicle",
            study=self.study_edit.text().strip(),
            cutoff=self.cutoff_spin.value() if self.filter_check.isChecked() else None,
            order=self.order_spin.value() if self.filter_check.isChecked() else None,
            miner=self.miner_spin.value(),
            no_filter=not self.filter_check.isChecked(),
        )
        self.start_requested.emit(req)


def _card_title(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight:700; font-size:13px; margin-top:4px;")
    return lbl
