"""Screen — PSD: Welch power spectral density, one figure per wheel."""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout

from gui.pages.base_page import BasePage
from gui.widgets.common import SectionTitle, ScrollPage, FigureGrid
from gui.widgets.image_viewer import ImageViewerDialog
from gui.models.repository import WHEELS


class PsdPage(BasePage):
    def __init__(self, repo, parent=None):
        super().__init__(repo, parent)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = ScrollPage()
        outer.addWidget(scroll)
        body = scroll.body()

        body.addWidget(SectionTitle("Welch Power Spectral Density"))
        self.grid = FigureGrid(columns=2, on_open=self._open)
        body.addWidget(self.grid)
        body.addStretch(1)

    def refresh(self) -> None:
        if not self.study:
            self.grid.set_figures([])
            return
        wheels = (self.study.wheel_labels() if self.study.has_stats else None) or list(WHEELS)
        self.grid.set_figures(
            [(f"PSD — {w}", self.study.psd(w)) for w in wheels])

    def _open(self, path) -> None:
        ImageViewerDialog.show_for(path, self)
