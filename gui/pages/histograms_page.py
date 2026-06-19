"""Screen 6 — Histograms: wheel + signal selectors, zoomable display."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QComboBox, QPushButton

from gui import theme
from gui.pages.base_page import BasePage
from gui.widgets.common import SectionTitle, Card, ZoomableImageView
from gui.widgets.image_viewer import ImageViewerDialog
from gui.models.repository import WHEELS, SIGNALS


class HistogramsPage(BasePage):
    def __init__(self, repo, parent=None):
        super().__init__(repo, parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)

        head = QHBoxLayout()
        head.addWidget(SectionTitle("Force Histograms"))
        head.addStretch(1)

        self.wheel_combo = QComboBox()
        self.wheel_combo.addItems(["All (combined)", *WHEELS])
        self.signal_combo = QComboBox()
        self.signal_combo.addItems(["Combined", *SIGNALS])
        self.kind_combo = QComboBox()
        self.kind_combo.addItems(["Distance-weighted", "Percentage"])
        for w in (QLabel("Wheel:"), self.wheel_combo,
                  QLabel("Signal:"), self.signal_combo,
                  QLabel("Type:"), self.kind_combo):
            head.addWidget(w)
        fs = QPushButton("⤢ Fullscreen")
        fs.setObjectName("Secondary")
        fs.clicked.connect(self._fullscreen)
        head.addWidget(fs)
        root.addLayout(head)

        for c in (self.wheel_combo, self.signal_combo, self.kind_combo):
            c.currentIndexChanged.connect(self._update_view)

        card = Card()
        self.view = ZoomableImageView()
        card.layout().addWidget(self.view)
        self.caption = QLabel("")
        self.caption.setAlignment(Qt.AlignCenter)
        self.caption.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-family:monospace;")
        card.layout().addWidget(self.caption)
        root.addWidget(card, 1)

        self._current = None

    def refresh(self) -> None:
        self._update_view()

    def _resolve(self):
        if not self.study:
            return None, "No study selected"
        wheel = self.wheel_combo.currentText()
        signal = self.signal_combo.currentText()
        kind = "distance" if self.kind_combo.currentIndex() == 0 else "percentage"

        if wheel.startswith("All"):
            # combined view needs a specific wheel; fall back to FL
            wheel = WHEELS[0]
            self.wheel_combo.blockSignals(True)
            self.wheel_combo.setCurrentText(wheel)
            self.wheel_combo.blockSignals(False)

        if signal == "Combined":
            path = self.study.histogram_combined(wheel, kind)
        else:
            path = self.study.histogram(wheel, signal, kind)
        return path, f"hist_{kind}_{wheel}" + (f"_{signal}" if signal != "Combined" else "") + ".png"

    def _update_view(self) -> None:
        path, name = self._resolve()
        self._current = path
        self.view.load(path)
        self.caption.setText(name if path else f"{name}  (not generated)")

    def _fullscreen(self) -> None:
        if self._current:
            ImageViewerDialog.show_for(self._current, self)
