"""Screen 7 — Heatmaps: correlation maps (Fx×Fy, Fz×Fy, Fz×Fx), with a
user-configurable bin range (min/max/step per component), mirroring the
reference WFT analyzer's Force Heatmaps panel.

The boxes start at the bins this study's heatmaps use -- the standard
``HEATMAP_BIN_RANGES_DAN``, the vehicle preset's, or a range set here before
(resolved by ``dtt.analysis.heatmaps``, the same code the pipeline and report
use). Generating saves the range for the study, so its next pipeline run and
report use it too; "Reset to standard" removes it."""

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
from dtt.config import DEFAULT_SAMPLING_RATE
from dtt.analysis.heatmaps import (
    clear_user_bin_ranges, generate_correlation_heatmaps, heatmap_bin_ranges,
    load_user_bin_ranges, save_user_bin_ranges,
)

# (display title, figure base name)
_CORRELATION = [
    ("Fx × Fy", "fx_fy_all"),
    ("Fz × Fy", "fz_fy_all"),
    ("Fz × Fx", "fz_fx_all"),
]


def _study_default_ranges(study) -> dict:
    """The bins the pipeline gives this study without a GUI choice: the
    auto-matched vehicle preset's, which for a PV are the standard ones."""
    try:
        from dtt.vehicles import match_preset
        return heatmap_bin_ranges(match_preset(study.run_channels().channel_set).tyre)
    except Exception:                                             # noqa: BLE001
        return heatmap_bin_ranges()


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
        self.reset_btn.setEnabled(self.study is not None)
        if not self.study:
            self.corr_grid.set_figures([])
            return
        user = load_user_bin_ranges(self.study.path)
        self._set_bin_ranges(user or _study_default_ranges(self.study))
        self.range_source.setText(
            "Set for this study — its next pipeline run and report use these bins."
            if user else "Default bins for this study.")
        self.corr_grid.set_figures(
            [(title, self.study.heatmap(name)) for title, name in _CORRELATION])

    def _set_bin_ranges(self, ranges: dict) -> None:
        for comp, values in ranges.items():
            for key, value in zip(("min", "max", "step"), values):
                self._range_boxes[comp][key].setValue(value)

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
            save_user_bin_ranges(self.study.path, bin_ranges)
        except Exception as ex:
            QMessageBox.critical(self, "Generate Force Heatmaps", f"Failed: {ex}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.refresh()

    def _reset(self) -> None:
        """Forget this study's own range; the next pipeline run (and report)
        goes back to the default bins, and so do the heatmaps shown here."""
        if not self.study:
            return
        clear_user_bin_ranges(self.study.path)
        self._set_bin_ranges(_study_default_ranges(self.study))
        self._generate()
        clear_user_bin_ranges(self.study.path)     # _generate saved the defaults
        self.refresh()


def _range_card(page: HeatmapsPage) -> Card:
    card = Card()
    card.layout().addWidget(_mini("Bin range (daN) — Min / Max / Step"))

    form = QFormLayout()
    form.setSpacing(8)
    page._range_boxes = {}
    defaults = heatmap_bin_ranges()
    for comp in ("Fx", "Fy", "Fz"):
        mn_default, mx_default, step_default = defaults[comp]
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
    page.reset_btn = QPushButton("Reset to standard")
    page.reset_btn.setObjectName("Secondary")
    page.reset_btn.setToolTip("Forget this study's own bin range and redraw with the default bins.")
    page.reset_btn.clicked.connect(page._reset)
    page.reset_btn.setEnabled(False)
    buttons = QHBoxLayout()
    buttons.addWidget(page.generate_btn)
    buttons.addWidget(page.reset_btn)
    buttons.addStretch(1)
    card.layout().addLayout(buttons)
    page.range_source = QLabel("")
    page.range_source.setStyleSheet("font-size:11px;")
    card.layout().addWidget(page.range_source)
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
