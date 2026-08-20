"""Screen 2 — New Study: choose CSV *or* imc raw folder, pick vehicle type,
configure filtering, and launch. Shows a live auto-detected axle configuration.
"""

from __future__ import annotations

import csv as _csv
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QFormLayout, QLineEdit, QComboBox, QPushButton, QDoubleSpinBox,
    QSpinBox, QCheckBox, QLabel, QFileDialog, QGridLayout, QWidget, QScrollArea,
)

from gui import theme
from gui.pages.base_page import BasePage
from gui.widgets.common import SectionTitle, Card, ScrollPage
from gui.workers.pipeline_worker import RunRequest

from dtt.channels import discover
from dtt.vehicles import preset_names, match_preset
from dtt.ingestion.imc_reader import resolve_raw_folder, pipeline_channel_name


class NewStudyPage(BasePage):
    start_requested = Signal(object)

    def __init__(self, repo, parent=None):
        super().__init__(repo, parent)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = ScrollPage()
        outer.addWidget(scroll)
        body = scroll.body()

        body.addWidget(SectionTitle("New Study"))
        sub = QLabel("Load a WFT CSV or an imc raw folder, then launch the pipeline.")
        sub.setStyleSheet(f"color:{theme.TEXT_MUTED};")
        body.addWidget(sub)

        columns = QHBoxLayout()
        columns.setSpacing(18)
        body.addLayout(columns)

        # Left: source + identity
        left = Card()
        left.layout().addWidget(_card_title("Data Source"))
        form = QFormLayout()
        form.setSpacing(12)
        form.setLabelAlignment(Qt.AlignLeft)

        self.source_type = QComboBox()
        self.source_type.addItems(["CSV file", "imc raw folder"])
        self.source_type.currentIndexChanged.connect(self._on_source_type)
        form.addRow("Source type", self.source_type)

        # CSV picker (container)
        self.csv_container = QWidget()
        csv_row = QHBoxLayout(self.csv_container)
        csv_row.setContentsMargins(0, 0, 0, 0)
        self.csv_combo = QComboBox()
        self.csv_combo.setMinimumWidth(240)
        self.csv_combo.currentIndexChanged.connect(self._update_detection)
        browse_csv = QPushButton("Browse…")
        browse_csv.setObjectName("Secondary")
        browse_csv.clicked.connect(self._browse_csv)
        csv_row.addWidget(self.csv_combo, 1)
        csv_row.addWidget(browse_csv)
        form.addRow("CSV file", self.csv_container)

        # Raw picker (folder or specific files)
        self.raw_container = QWidget()
        raw_row = QHBoxLayout(self.raw_container)
        raw_row.setContentsMargins(0, 0, 0, 0)
        self.raw_edit = QLineEdit()
        self.raw_edit.setPlaceholderText("Select an imc .raw folder or files…")
        self.raw_edit.setReadOnly(True)
        browse_raw = QPushButton("Folder…")
        browse_raw.setObjectName("Secondary")
        browse_raw.clicked.connect(self._browse_raw)
        files_raw = QPushButton("Files…")
        files_raw.setObjectName("Secondary")
        files_raw.clicked.connect(self._browse_raw_files)
        raw_row.addWidget(self.raw_edit, 1)
        raw_row.addWidget(browse_raw)
        raw_row.addWidget(files_raw)
        form.addRow("Raw source", self.raw_container)

        # Detected configuration (auto)
        self.detect_label = QLabel("—")
        self.detect_label.setWordWrap(True)
        self.detect_label.setStyleSheet(
            f"color:{theme.ACCENT}; font-size:12px; font-weight:600;")
        form.addRow("Detected", self.detect_label)

        self.vehicle_edit = QLineEdit("RLDA_PV")
        form.addRow("Vehicle name", self.vehicle_edit)

        self.vtype_combo = QComboBox()
        self.vtype_combo.addItem("Auto-detect", "")
        for name in preset_names():
            self.vtype_combo.addItem(name, name)
        form.addRow("Vehicle type", self.vtype_combo)

        self.study_edit = QLineEdit()
        self.study_edit.setPlaceholderText("Leave blank → timestamp")
        form.addRow("Study name", self.study_edit)
        left.layout().addLayout(form)

        # Every channel in the source, under the names the pipeline will give
        # them — listed before the run so the source can be checked at a glance.
        # A raw folder carries 40-odd channels, so this gets real room and its
        # own scrollbar rather than being squeezed into a form row.
        self.channels_title = _card_title("Channels in source")
        left.layout().addWidget(self.channels_title)
        self.channels_label = QLabel("")
        self.channels_label.setWordWrap(True)
        self.channels_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.channels_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        # The panel background goes on the label, not the QScrollArea: the app
        # stylesheet paints every QScrollArea transparent, and styling the area
        # would not reach the viewport the label is drawn on anyway.
        self.channels_label.setStyleSheet(
            f"color:{theme.TEXT}; font-family:Consolas,monospace; font-size:13px;"
            f"background:{theme.ENTRY_BG}; border-radius:8px; padding:12px;")
        self.channels_scroll = QScrollArea()
        self.channels_scroll.setWidgetResizable(True)
        self.channels_scroll.setFrameShape(QScrollArea.NoFrame)
        self.channels_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.channels_scroll.setWidget(self.channels_label)
        self.channels_scroll.setMinimumHeight(280)
        left.layout().addWidget(self.channels_scroll, 1)
        columns.addWidget(left, 1)

        # Right: filtering + modules
        right = Card()
        right.layout().addWidget(_card_title("Filtering"))
        fform = QFormLayout()
        fform.setSpacing(12)

        self.filter_check = QCheckBox("Apply signal conditioning")
        self.filter_check.setChecked(True)
        self.filter_check.toggled.connect(self._toggle_filter)
        right.layout().addWidget(self.filter_check)

        self.famos_check = QCheckBox("Use imc/FAMOS recipe (recommended)")
        self.famos_check.setChecked(True)
        self.famos_check.setToolTip(
            "Reproduce the imc config file exactly: smo(x, 0.1) on WFT forces and "
            "moments, FiltLP(4, 5 Hz) + smo(x, 0.5) on Latacc, smo(x, 0.5) on "
            "speed, then red().\nUncheck to use the legacy Butterworth low-pass "
            "below, which is markedly more aggressive than FAMOS on the forces "
            "and shifts every downstream statistic.")
        self.famos_check.toggled.connect(self._toggle_filter)
        right.layout().addWidget(self.famos_check)

        self.deglitch_check = QCheckBox("Remove DAQ artifact spikes")
        self.deglitch_check.setChecked(False)
        self.deglitch_check.setToolTip(
            "Bridge samples that sit far outside a rolling median of their "
            "neighbours.\nLeave off for a cleanly-read imc recording; turn on if "
            "the raw traces show isolated out-of-family spikes.")
        right.layout().addWidget(self.deglitch_check)

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
        for i, m in enumerate([
                "Statistics", "Histograms", "Heatmaps",
                "Boxplots", "Rainflow", "PowerPoint Report"]):
            cb = QCheckBox(m)
            cb.setChecked(True)
            cb.setEnabled(False)
            cb.setToolTip("The backend pipeline runs all modules in one pass.")
            self.module_checks[m] = cb
            mod_grid.addWidget(cb, i // 2, i % 2)
        right.layout().addLayout(mod_grid)
        note = QLabel("Modules reflect the backend's fixed 9-stage pipeline.")
        note.setStyleSheet(f"color:{theme.TEXT_FAINT}; font-size:11px;")
        right.layout().addWidget(note)
        columns.addWidget(right, 1)

        # Launch
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

        self._raw_folder: Path | None = None
        self._raw_files: list[Path] = []
        self.reload_csvs()
        self._on_source_type()
        self._toggle_filter()

    # Source switching
    def _on_source_type(self, *_) -> None:
        is_raw = self.source_type.currentIndex() == 1
        self.raw_container.setVisible(is_raw)
        self.csv_container.setVisible(not is_raw)
        self._update_detection()

    def reload_csvs(self) -> None:
        self.csv_combo.clear()
        files = self.repo.list_csv_files()
        if not files:
            self.csv_combo.addItem("No CSV files in csv/ — use Browse…", None)
        for f in files:
            self.csv_combo.addItem(f.name, str(f))

    def _toggle_filter(self, *_) -> None:
        on = self.filter_check.isChecked()
        # Cutoff/order drive the legacy Butterworth only — the FAMOS recipe
        # fixes them per channel from the imc config file.
        legacy = on and not self.famos_check.isChecked()
        self.famos_check.setEnabled(on)
        self.deglitch_check.setEnabled(on)
        self.cutoff_spin.setEnabled(legacy)
        self.order_spin.setEnabled(legacy)

    def _browse_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select WFT CSV", str(self.repo.csv_dir), "CSV files (*.csv)")
        if path:
            self.csv_combo.insertItem(0, Path(path).name + "  (browsed)", path)
            self.csv_combo.setCurrentIndex(0)
            self._update_detection()

    def _browse_raw(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select imc raw channel folder", str(self.repo.csv_dir.parent))
        if folder:
            self._raw_folder = Path(folder)
            self._raw_files = []
            self.raw_edit.setText(folder)
            self._update_detection()

    def _browse_raw_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select imc .raw files", str(self.repo.csv_dir.parent),
            "imc raw files (*.raw *.dat)")
        if paths:
            self._raw_files = [Path(p) for p in paths]
            self._raw_folder = None
            self.raw_edit.setText(f"{len(self._raw_files)} files: "
                                  + ", ".join(p.name for p in self._raw_files[:4])
                                  + (" …" if len(self._raw_files) > 4 else ""))
            self._update_detection()

    # Live axle detection
    def _current_columns(self) -> list[str]:
        if self.source_type.currentIndex() == 1:      # raw source
            if self._raw_files:
                stems = [f.stem for f in self._raw_files]
            elif self._raw_folder and self._raw_folder.exists():
                found = resolve_raw_folder(self._raw_folder)
                stems = [f.stem for f in sorted(found.glob("*.raw"))]
            else:
                return []
            # Report the names ingestion will produce, not the filenames, and
            # collapse "… - Copy" duplicates the reader would merge anyway.
            seen: dict[str, None] = {}
            for s in stems:
                seen.setdefault(pipeline_channel_name(s), None)
            return list(seen)
        data = self.csv_combo.currentData()
        if not data or not Path(data).exists():
            return []
        # Instrument exports may have blank/offset header rows (like FAMOS CSVs),
        # so scan the first few lines for the one that holds wheel-force channels.
        try:
            with open(data, newline="", encoding="utf-8", errors="replace") as fh:
                lines = [fh.readline() for _ in range(6)]
        except OSError:
            return []
        first_nonempty: list[str] = []
        for ln in lines:
            cols = [c.strip() for c in next(_csv.reader([ln]), [])]
            if cols and not first_nonempty:
                first_nonempty = cols
            if discover(cols).n_positions > 0:
                return cols
        return first_nonempty

    def _update_detection(self, *_) -> None:
        cols = self._current_columns()
        if not cols:
            self.detect_label.setText("—")
            self.channels_label.setText("Select a source to list its channels.")
            return
        cs = discover(cols)
        if cs.n_positions == 0:
            self.detect_label.setText("No wheel-force channels detected")
            self.detect_label.setStyleSheet(f"color:{theme.WARNING}; font-size:12px;")
            # Still list what *is* there — seeing the names is how you work out
            # why nothing matched.
            self.channels_label.setText(self._channel_listing(cols, cs))
            return
        preset = match_preset(cs)
        self.detect_label.setStyleSheet(
            f"color:{theme.ACCENT}; font-size:12px; font-weight:600;")
        self.detect_label.setText(f"{cs.summary()}   →  suggests: {preset.name}")

        self.channels_label.setText(self._channel_listing(cols, cs))

    @staticmethod
    def _channel_listing(cols: list[str], cs) -> str:
        """Every channel in the source: the WFT ones by position, then the rest.

        The non-WFT channels are the point of listing at all — accel, speed, GPS
        and yaw decide whether histograms can be distance-weighted and whether
        Latacc gets its FiltLP pass, and they were the ones the old listing
        dropped entirely.
        """
        order = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
        by_pos: dict[str, list[tuple[int, str]]] = {}
        for orig, (pos, comp) in cs.source_map.items():
            rank = order.index(comp) if comp in order else 99
            by_pos.setdefault(pos.display, []).append((rank, orig))

        n_wft = sum(len(v) for v in by_pos.values())
        out: list[str] = []
        if n_wft:
            out.append(f"WFT force & moment — {n_wft} channels, "
                       f"{cs.n_positions} positions")
            for lbl in sorted(by_pos):
                out.append(f"  {lbl}")
                names = [n for _, n in sorted(by_pos[lbl])]
                for i in range(0, len(names), 3):
                    out.append("    " + "".join(f"{n:<14}" for n in names[i:i + 3]).rstrip())

        rest = [c for c in cols if c not in cs.source_map]
        if rest:
            if out:
                out.append("")
            out.append(f"Other channels — {len(rest)}")
            for i in range(0, len(rest), 3):
                out.append("  " + "".join(f"{c:<16}" for c in rest[i:i + 3]).rstrip())
        return "\n".join(out)

    # Launch
    def _on_start(self) -> None:
        self.error_label.setText("")
        is_raw = self.source_type.currentIndex() == 1
        vehicle_type = self.vtype_combo.currentData() or ""

        common = dict(
            vehicle=self.vehicle_edit.text().strip() or "Vehicle",
            vehicle_type=vehicle_type,
            study=self.study_edit.text().strip(),
            cutoff=self.cutoff_spin.value() if self.cutoff_spin.isEnabled() else None,
            order=self.order_spin.value() if self.order_spin.isEnabled() else None,
            miner=self.miner_spin.value(),
            no_filter=not self.filter_check.isChecked(),
            no_famos=not self.famos_check.isChecked(),
            deglitch=self.deglitch_check.isChecked(),
        )

        if is_raw:
            if self._raw_files:
                existing = [f for f in self._raw_files if f.exists()]
                if not existing:
                    self.error_label.setText("Selected .raw files not found.")
                    return
                req = RunRequest(raw_files=existing, **common)
            elif self._raw_folder and self._raw_folder.exists():
                resolved = resolve_raw_folder(self._raw_folder)
                if not list(resolved.glob("*.raw")):
                    self.error_label.setText("No .raw files found in the selected folder or its sub-folders.")
                    return
                req = RunRequest(raw_folder=resolved, **common)
            else:
                self.error_label.setText("Please select an imc raw folder or files.")
                return
        else:
            data = self.csv_combo.currentData()
            csv_path = Path(data) if data else None
            if not csv_path or not csv_path.exists():
                self.error_label.setText("Please select a valid CSV file.")
                return
            req = RunRequest(csv_path=csv_path, **common)

        self.start_requested.emit(req)


def _card_title(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight:700; font-size:13px; margin-top:4px;")
    return lbl