"""Screen 8 — Boxplots: one per signal (Fx, Fy, Fz) across all wheels."""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout

from gui.pages.base_page import BasePage
from gui.widgets.common import SectionTitle, ScrollPage, FigureGrid
from gui.widgets.image_viewer import ImageViewerDialog
from gui.models.repository import SIGNALS


class BoxplotsPage(BasePage):
    def __init__(self, repo, parent=None):
        super().__init__(repo, parent)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = ScrollPage()
        outer.addWidget(scroll)
        body = scroll.body()

        body.addWidget(SectionTitle("Load Distribution Boxplots"))
        self.grid = FigureGrid(columns=3, on_open=self._open)
        body.addWidget(self.grid)
        body.addStretch(1)

    def refresh(self) -> None:
        if not self.study:
            self.grid.set_figures([])
            return
        self.grid.set_figures(
            [(f"Boxplot — {s}", self.study.boxplot(s)) for s in SIGNALS])

    def _open(self, path) -> None:
        ImageViewerDialog.show_for(path, self)
