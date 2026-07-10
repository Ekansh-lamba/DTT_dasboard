"""Full-screen / dialog figure viewer with zoom controls."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
)

from gui import theme
from gui.widgets.common import ZoomableImageView


class ImageViewerDialog(QDialog):
    """Modal viewer used by 'View' / 'Fullscreen' actions across screens."""

    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(Path(path).name)
        self.setModal(True)
        self.resize(1100, 800)
        self.setStyleSheet(f"background:{theme.BG};")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        bar = QHBoxLayout()
        title = QLabel(Path(path).name)
        title.setStyleSheet("font-weight:700; font-size:14px;")
        bar.addWidget(title)
        bar.addStretch(1)

        self._view = ZoomableImageView()
        for label, fn in (("− Zoom out", lambda: self._view.scale(0.8, 0.8)),
                          ("+ Zoom in", lambda: self._view.scale(1.25, 1.25)),
                          ("Fit", self._view.reset_zoom)):
            b = QPushButton(label)
            b.setObjectName("Secondary")
            b.clicked.connect(fn)
            bar.addWidget(b)
        close = QPushButton("Close")
        close.setObjectName("Primary")
        close.clicked.connect(self.accept)
        bar.addWidget(close)

        lay.addLayout(bar)
        lay.addWidget(self._view, 1)
        self._view.load(path)

    @staticmethod
    def show_for(path: Path, parent=None) -> None:
        if path and Path(path).exists():
            ImageViewerDialog(path, parent).exec()
