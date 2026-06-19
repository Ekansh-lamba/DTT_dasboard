"""Reusable presentation widgets shared across screens."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap, QPainter, QFont
from PySide6.QtWidgets import (
    QFrame, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QScrollArea,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QSizePolicy,
    QGridLayout, QPushButton,
)

from gui import theme


# ════════════════════════════════════════════════════════════════════════════
# Simple building blocks
# ════════════════════════════════════════════════════════════════════════════

class Card(QFrame):
    """A rounded surface panel with a vertical layout."""

    def __init__(self, parent=None, object_name: str = "Card"):
        super().__init__(parent)
        self.setObjectName(object_name)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(18, 16, 18, 16)
        self._layout.setSpacing(10)

    def layout(self) -> QVBoxLayout:           # type: ignore[override]
        return self._layout


class SectionTitle(QLabel):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setObjectName("SectionTitle")


class KpiCard(QFrame):
    """Big-number metric tile for the dashboard."""

    def __init__(self, label: str, value: str = "—", sub: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("KpiCard")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(4)

        self._label = QLabel(label.upper()); self._label.setObjectName("KpiLabel")
        self._value = QLabel(value);         self._value.setObjectName("KpiValue")
        self._sub   = QLabel(sub);           self._sub.setObjectName("KpiSub")
        self._sub.setWordWrap(True)

        lay.addWidget(self._label)
        lay.addWidget(self._value)
        lay.addWidget(self._sub)
        lay.addStretch(1)

    def set_value(self, value: str, sub: Optional[str] = None) -> None:
        self._value.setText(value)
        if sub is not None:
            self._sub.setText(sub)


class Badge(QLabel):
    """Small status pill."""

    def __init__(self, text: str, color: str = theme.ACCENT, parent=None):
        super().__init__(text, parent)
        self.set_status(text, color)

    def set_status(self, text: str, color: str = theme.ACCENT) -> None:
        self.setText(text)
        self.setStyleSheet(
            f"background:{color}22; color:{color}; border:1px solid {color}55;"
            f"border-radius:10px; padding:3px 10px; font-size:11px; font-weight:600;"
        )


def status_badge(status: str) -> Badge:
    color = theme.STATUS_COLORS.get(status, theme.TEXT_MUTED)
    return Badge(status, color)


# ════════════════════════════════════════════════════════════════════════════
# Image viewing
# ════════════════════════════════════════════════════════════════════════════

class ZoomableImageView(QGraphicsView):
    """A pan + scroll-to-zoom image canvas used for figure inspection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item: Optional[QGraphicsPixmapItem] = None
        self._empty = True

        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(Qt.transparent)
        self.setFrameShape(QFrame.NoFrame)
        self.setMinimumHeight(300)

    def load(self, path: Optional[Path]) -> bool:
        self._scene.clear()
        self._item = None
        if path is None or not Path(path).exists():
            self._empty = True
            placeholder = self._scene.addText("No figure available")
            placeholder.setDefaultTextColor(Qt.gray)
            return False
        pix = QPixmap(str(path))
        if pix.isNull():
            self._empty = True
            return False
        self._item = self._scene.addPixmap(pix)
        self._scene.setSceneRect(self._item.boundingRect())
        self._empty = False
        self.reset_zoom()
        return True

    def reset_zoom(self) -> None:
        if self._item is not None:
            self.resetTransform()
            self.fitInView(self._item, Qt.KeepAspectRatio)

    def wheelEvent(self, event) -> None:               # noqa: N802
        if self._empty:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def resizeEvent(self, event):                      # noqa: N802
        super().resizeEvent(event)
        if self._item is not None and self.transform().m11() <= 1.01:
            self.reset_zoom()


class FigureCard(QFrame):
    """A titled thumbnail of a figure that opens a full viewer when clicked."""

    def __init__(self, title: str, path: Optional[Path], on_open=None, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self._path = path
        self._on_open = on_open

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        header = QHBoxLayout()
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-weight:600;")
        header.addWidget(title_lbl)
        header.addStretch(1)
        if path is not None:
            open_btn = QPushButton("⤢ View")
            open_btn.setObjectName("Secondary")
            open_btn.setCursor(Qt.PointingHandCursor)
            open_btn.clicked.connect(self._open)
            header.addWidget(open_btn)
        lay.addLayout(header)

        self._thumb = QLabel()
        self._thumb.setAlignment(Qt.AlignCenter)
        self._thumb.setMinimumHeight(220)
        self._thumb.setStyleSheet(f"background:{theme.ENTRY_BG}; border-radius:8px;")
        self._thumb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self._thumb)

        self._pixmap: Optional[QPixmap] = None
        if path is not None and Path(path).exists():
            pix = QPixmap(str(path))
            if not pix.isNull():
                self._pixmap = pix
        if self._pixmap is None:
            self._thumb.setText("Figure not generated")
            self._thumb.setStyleSheet(
                f"background:{theme.ENTRY_BG}; border-radius:8px; color:{theme.TEXT_FAINT};"
            )

    def _rescale(self) -> None:
        if self._pixmap is None:
            return
        self._thumb.setPixmap(self._pixmap.scaled(
            self._thumb.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):                      # noqa: N802
        super().resizeEvent(event)
        self._rescale()

    def _open(self) -> None:
        if self._on_open and self._path:
            self._on_open(self._path)


class ScrollPage(QScrollArea):
    """A vertically scrolling page with a content column."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._content = QWidget()
        self._content.setObjectName("RootWidget")
        self.setWidget(self._content)
        self._vbox = QVBoxLayout(self._content)
        self._vbox.setContentsMargins(28, 24, 28, 28)
        self._vbox.setSpacing(18)

    def body(self) -> QVBoxLayout:
        return self._vbox


class FigureGrid(QWidget):
    """A responsive grid of FigureCards."""

    def __init__(self, columns: int = 2, on_open=None, parent=None):
        super().__init__(parent)
        self._columns = columns
        self._on_open = on_open
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(16)

    def set_figures(self, items: list[tuple[str, Optional[Path]]]) -> None:
        while self._grid.count():
            w = self._grid.takeAt(0).widget()
            if w:
                w.deleteLater()
        if not items:
            empty = QLabel("No figures available for this study.")
            empty.setStyleSheet(f"color:{theme.TEXT_FAINT}; padding:24px;")
            empty.setAlignment(Qt.AlignCenter)
            self._grid.addWidget(empty, 0, 0)
            return
        for i, (title, path) in enumerate(items):
            r, c = divmod(i, self._columns)
            self._grid.addWidget(FigureCard(title, path, self._on_open), r, c)
