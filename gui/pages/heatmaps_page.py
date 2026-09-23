"""Screen 7 — Heatmaps: correlation maps (Fx×Fy, Fz×Fy, Fz×Fx), with a
user-configurable bin range (min/max/step per component), mirroring the
reference WFT analyzer's Force Heatmaps panel."""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QFormLayout, QLabel, QDoubleSpinBox, QPushButton, QMessageBox,
    QApplication,
)

from gui.pages.base_page import BasePage
from gui.widgets.common import SectionTitle, Card, ScrollPage, FigureGrid
from gui.widgets.image_viewer import ImageViewerDialog
from gui.models.repository import WHEELS
from dtt.config import FORCE_RANGES_DAN, DEFAULT_SAMPLING_RATE
from dtt.analysis.heatmaps import generate_correlation_heatmaps

# (display title, figure base name)
_CORRELATION = [
    ("Fx × Fy", "fx_fy_all"),
    ("Fz × Fy", "fz_fy_all"),
    ("Fz × Fx", "fz_fx_all"),
]

# Default Min / Max / Step (daN) for each component's bin-range spinboxes.
_DEFAULT_RANGE = {
    "Fx": (FORCE_RANGES_DAN["Fx"][0], FORCE_RANGES_DAN["Fx"][1], 100.0),
    "Fy": (FORCE_RANGES_DAN["Fy"][0], FORCE_RANGES_DAN["Fy"][1], 100.0),
    "Fz": (FORCE_RANGES_DAN["Fz"][0], FORCE_RANGES_DAN["Fz"][1], 100.0),
}


class HeatmapsPage(BasePage):
    def __init__(self, repo, parent=None):
        super().__init__(repo, parent)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = ScrollPage()
        outer.addWidget(scroll)
        body = scroll.body()

        head = QHBoxLayout()
        head.addWidget(SectionTitle("Force Heatmaps"))
        head.addStretch(1)
        body.addLayout(head)

        body.addWidget(_range_card(self))

        body.addWidget(_mini("Correlation Heatmaps"))
        self.corr_grid = FigureGrid(columns=3, on_open=self._open)
        body.addWidget(self.corr_grid)
        body.addStretch(1)

    def refresh(self) -> None:
        self.generate_btn.setEnabled(self.study is not None)
        if not self.study:
            self.corr_grid.set_figures([])
            return
        self.corr_grid.set_figures(
            [(title, self.study.heatmap(name)) for title, name in _CORRELATION])

    def _open(self, path) -> None:
        ImageViewerDialog.show_for(path, self)

    def _read_bin_range(self, comp: str):
        v = self._range_boxes[comp]
        return (v["min"].value(), v["max"].value(), v["step"].value())

    def _generate(self) -> None:
        if not self.study:
            return
        if not self.study.processed_csv.exists():
            QMessageBox.warning(
                self, "Generate Force Heatmaps",
                "This study has no processed_data.csv to regenerate from.")
            return

        bin_ranges = {comp: self._read_bin_range(comp) for comp in ("Fx", "Fy", "Fz")}
        try:
            for comp, (mn, mx, step) in bin_ranges.items():
                if step <= 0:
                    raise ValueError(f"{comp}: step must be greater than 0.")
                if mx <= mn:
                    raise ValueError(f"{comp}: max must be greater than min.")
        except ValueError as ex:
            QMessageBox.critical(self, "Invalid bin range", str(ex))
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            rc = self.study.run_channels()
            wheels = self.study.wheel_labels() or list(WHEELS)
            wheel_colors = rc.wheel_colors
            sampling_rate = self.study.validation().get("sampling_rate_hz") or DEFAULT_SAMPLING_RATE
            df = pd.read_csv(self.study.processed_csv)
            generate_correlation_heatmaps(
                df, self.study.figures_dir, rc, wheels, wheel_colors,
                sampling_rate, bin_ranges=bin_ranges)
        except Exception as ex:
            QMessageBox.critical(self, "Generate Force Heatmaps", f"Failed: {ex}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.refresh()


def _range_card(page: HeatmapsPage) -> Card:
    card = Card()
    card.layout().addWidget(_mini("Bin range (daN) — Min / Max / Step"))

    form = QFormLayout()
    form.setSpacing(8)
    page._range_boxes = {}
    for comp in ("Fx", "Fy", "Fz"):
        mn_default, mx_default, step_default = _DEFAULT_RANGE[comp]
        row = QHBoxLayout()
        boxes = {}
        for key, default in (("min", mn_default), ("max", mx_default), ("step", step_default)):
            box = QDoubleSpinBox()
            box.setRange(-100000, 100000)
            box.setDecimals(1)
            box.setValue(default)
            box.setMaximumWidth(100)
            boxes[key] = box
            row.addWidget(QLabel(key.capitalize() + ":"))
            row.addWidget(box)
        row.addStretch(1)
        page._range_boxes[comp] = boxes
        form.addRow(f"{comp}:", _wrap(row))
    card.layout().addLayout(form)

    page.generate_btn = QPushButton("Generate Force Heatmaps (Fy-Fx / Fx-Fz / Fy-Fz)")
    page.generate_btn.setObjectName("Primary")
    page.generate_btn.clicked.connect(page._generate)
    page.generate_btn.setEnabled(False)
    card.layout().addWidget(page.generate_btn)
    return card


def _wrap(layout):
    from PySide6.QtWidgets import QWidget
    w = QWidget()
    w.setLayout(layout)
    return w


def _mini(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight:700; font-size:13px;")
    return lbl
