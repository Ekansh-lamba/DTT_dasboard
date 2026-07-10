"""Dark-themed matplotlib canvas + interactive plot panel for Qt embedding."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import QWidget, QVBoxLayout

from gui import theme


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, height=3.2):
        self.fig = Figure(figsize=(6, height), facecolor=theme.SURFACE)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self._style()

    def _style(self):
        ax = self.ax
        ax.set_facecolor(theme.ENTRY_BG)
        ax.tick_params(colors=theme.TEXT_MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(theme.BORDER)
        ax.grid(True, color=theme.BORDER, linewidth=0.4, alpha=0.5)
        ax.xaxis.label.set_color(theme.TEXT_MUTED)
        ax.yaxis.label.set_color(theme.TEXT_MUTED)
        ax.title.set_color(theme.TEXT)

    def clear(self):
        self.ax.clear()
        self._style()


class PlotPanel(QWidget):
    """A canvas with a zoom/pan toolbar (scroll-wheel + rectangle zoom)."""

    def __init__(self, height=3.6, parent=None):
        super().__init__(parent)
        self.canvas = MplCanvas(height=height)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setStyleSheet(
            f"background:{theme.SURFACE_2}; border:none; color:{theme.TEXT};")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    @property
    def ax(self):
        return self.canvas.ax

    def clear(self):
        self.canvas.clear()

    def draw(self):
        self.canvas.draw()

    def _on_scroll(self, event):
        if event.inaxes is None:
            return
        ax = event.inaxes
        factor = 0.8 if event.button == "up" else 1.25
        x, y = event.xdata, event.ydata
        for lo_hi, setter, c in (
            (ax.get_xlim(), ax.set_xlim, x), (ax.get_ylim(), ax.set_ylim, y)):
            lo, hi = lo_hi
            setter([c - (c - lo) * factor, c + (hi - c) * factor])
        self.canvas.draw_idle()
