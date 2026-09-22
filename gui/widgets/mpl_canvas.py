"""Dark-themed matplotlib canvas + interactive plot panel for Qt embedding."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from gui import theme


class MplCanvas(FigureCanvasQTAgg):
    """One or more axes on a Qt canvas, in the app's dark chrome or light.

    ``light=True`` is for the sections that deliberately do not use the app's
    navy chrome -- the force histograms and the AUC views, which were restyled
    to a white reference look and whose own colour constants live in
    ``dtt/analysis/histograms.py``. Styling those axes here as well would give
    them two sources of truth, so the light mode sets only the figure
    background and leaves the axes to the drawing code.

    ``ncols`` puts several axes on one canvas, for a view whose panels have to
    share an x-range and be read side by side.
    """

    def __init__(self, parent=None, height=3.2, ncols=1, light=False):
        self.light = light
        face = "#FFFFFF" if light else theme.SURFACE
        self._ncols = max(1, int(ncols))
        # Width scales with the panel count. At a fixed 6 inches two panels
        # overlap their titles and axis labels before Qt ever resizes the
        # figure -- and a figure saved straight to PNG never gets that resize.
        self.fig = Figure(figsize=(6 * self._ncols, height), facecolor=face)
        super().__init__(self.fig)
        self.setParent(parent)
        self.axes = [self.fig.add_subplot(1, self._ncols, i + 1)
                     for i in range(self._ncols)]
        self.ax = self.axes[0]
        self._style()

    def _style(self):
        if self.light:
            return                      # the drawing code owns the light look
        for ax in self.axes:
            ax.set_facecolor(theme.ENTRY_BG)
            ax.tick_params(colors=theme.TEXT_MUTED, labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor(theme.BORDER)
            ax.grid(True, color=theme.BORDER, linewidth=0.4, alpha=0.5)
            ax.xaxis.label.set_color(theme.TEXT_MUTED)
            ax.yaxis.label.set_color(theme.TEXT_MUTED)
            ax.title.set_color(theme.TEXT)

    def clear(self):
        # The figure's own texts (a suptitle, a footer strip) are not owned by
        # any axes, so clearing the axes alone leaves the previous render's
        # captions stacked under the new one.
        #
        # Removed one at a time, not with `fig.texts.clear()`: that empties the
        # list the renderer walks while `fig._suptitle` still points at the old
        # object, so the next `fig.suptitle(...)` quietly retitles something
        # that is no longer attached to the figure and nothing appears.
        for ax in self.axes:
            ax.clear()
        for text in list(self.fig.texts):
            text.remove()
        self.fig._suptitle = None
        self._style()


class PlotPanel(QWidget):
    """A canvas with a zoom/pan toolbar (scroll-wheel + rectangle zoom)."""

    def __init__(self, height=3.6, parent=None, ncols=1, light=False):
        super().__init__(parent)
        self.canvas = MplCanvas(height=height, ncols=ncols, light=light)
        # A canvas reports its figure size as its preferred width, and the
        # pages sit in a ScrollPage with horizontal scrolling switched off --
        # so a canvas wider than the viewport is not scrolled to, it is simply
        # cut off, with no indication that anything is missing. Ignoring the
        # width hint lets the canvas shrink to whatever the page has; the
        # figure keeps its aspect through `height`, and the toolbar's zoom is
        # there for detail.
        self.canvas.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.canvas.setMinimumWidth(240)
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

    @property
    def axes(self):
        return self.canvas.axes

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
