"""Screen 7 — Heatmaps: correlation maps (Fx×Fy, Fz×Fy, Fz×Fx) + hexbin."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QPushButton

from gui import theme
from gui.pages.base_page import BasePage
from gui.widgets.common import SectionTitle, Card, ScrollPage, FigureGrid
from gui.widgets.image_viewer import ImageViewerDialog
from gui.models.repository import WHEELS

# (display title, figure base name)
_CORRELATION = [
    ("Fx × Fy", "fx_fy_all"),
    ("Fz × Fy", "fz_fy_all"),
    ("Fz × Fx", "fz_fx_all"),
]


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

        body.addWidget(_mini("Correlation Heatmaps"))
        self.corr_grid = FigureGrid(columns=3, on_open=self._open)
        body.addWidget(self.corr_grid)

        body.addWidget(_mini("Per-Wheel Hexbin Density"))
        self.hex_grid = FigureGrid(columns=2, on_open=self._open)
        body.addWidget(self.hex_grid)
        body.addStretch(1)

    def refresh(self) -> None:
        if not self.study:
            self.corr_grid.set_figures([])
            self.hex_grid.set_figures([])
            return
        self.corr_grid.set_figures(
            [(title, self.study.heatmap(name)) for title, name in _CORRELATION])
        self.hex_grid.set_figures(
            [(f"Hexbin — {w}", self.study.hexbin(w)) for w in WHEELS])

    def _open(self, path) -> None:
        ImageViewerDialog.show_for(path, self)


def _mini(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight:700; font-size:13px;")
    return lbl
