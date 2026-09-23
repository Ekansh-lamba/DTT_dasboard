"""Screen 8 — Boxplots: one per wheel component (Fx..Mz) across all wheels,
plus one sheet for every other measured parameter."""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout

from gui.pages.base_page import BasePage
from gui.widgets.common import SectionTitle, ScrollPage, FigureGrid
from gui.widgets.image_viewer import ImageViewerDialog
from gui.models.repository import SIGNALS
from dtt.channels import COMPONENTS as _COMPONENTS


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
        figs = [(f"Boxplot — {c}", self.study.boxplot(c)) for c in _COMPONENTS
                if self.study.boxplot(c) is not None]
        params = self.study.boxplot("parameters")
        if params is not None:
            figs.append(("Boxplots — vehicle & other parameters", params))
        if not figs:
            # a study run before moments/parameters were plotted
            figs = [(f"Boxplot — {s}", self.study.boxplot(s)) for s in SIGNALS]
        self.grid.set_figures(figs)

    def _open(self, path) -> None:
        ImageViewerDialog.show_for(path, self)
