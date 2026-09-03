"""Screen — Area Under Curve (AUC): wheel + signal selectors, zoomable display.

Mirrors ``histograms_page.py`` exactly (same selectors, same navigation
pattern) -- this is the Item A split: histograms show binned bars, this page
shows the smoothed density curve for the same channel/wheel, as its own
section rather than mixed into the histogram view.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QComboBox, QPushButton

from gui import theme
from gui.pages.base_page import BasePage
from gui.widgets.common import SectionTitle, Card, ZoomableImageView
from gui.widgets.image_viewer import ImageViewerDialog
from gui.models.repository import WHEELS, SIGNALS


class AucPage(BasePage):
    def __init__(self, repo, parent=None):
        super().__init__(repo, parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)

        head = QHBoxLayout()
        head.addWidget(SectionTitle("Area Under Curve (AUC)"))
        head.addStretch(1)

        self.wheel_combo = QComboBox()
        self.signal_combo = QComboBox()
        self.kind_combo = QComboBox()
        self.kind_combo.addItems(["Distance-weighted", "Percentage"])
        self.range_combo = QComboBox()
        self.range_combo.addItem("Full range", "full")
        self.range_combo.addItem("Autoscale to data", "autoscale")
        self.range_combo.setToolTip(
            "Which x-axis mode's figure to show — depends on which mode(s) "
            "the run generated.")
        for w in (QLabel("Wheel:"), self.wheel_combo,
                  QLabel("Signal:"), self.signal_combo,
                  QLabel("Type:"), self.kind_combo,
                  QLabel("X-axis:"), self.range_combo):
            head.addWidget(w)
        fs = QPushButton("⤢ Fullscreen")
        fs.setObjectName("Secondary")
        fs.clicked.connect(self._fullscreen)
        head.addWidget(fs)
        root.addLayout(head)

        for c in (self.wheel_combo, self.signal_combo, self.kind_combo, self.range_combo):
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
        self._reload_selectors()
        self._update_view()

    def _reload_selectors(self) -> None:
        """Offer only the wheels and components this study actually recorded.

        A two-WFT or multi-axle recording has neither all four legacy wheels nor
        necessarily Fx/Fy/Fz, so a fixed list offers entries that can never load.
        """
        wheels = list(WHEELS)
        signals = list(SIGNALS)
        if self.study and self.study.has_stats:
            wheels = self.study.wheel_labels() or wheels
            signals = self.study.components() or signals

        for combo, items in ((self.wheel_combo, wheels),
                             (self.signal_combo, ["Combined", *signals])):
            prev = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            if prev in items:
                combo.setCurrentText(prev)
            combo.blockSignals(False)

    def _resolve(self):
        if not self.study:
            return None, "No study selected"
        wheel = self.wheel_combo.currentText()
        signal = self.signal_combo.currentText()
        kind = "distance" if self.kind_combo.currentIndex() == 0 else "percentage"
        range_mode = self.range_combo.currentData() or "full"
        if not wheel:
            return None, "No wheel channels in this study"

        suffix = "" if range_mode == "full" else f"_{range_mode}"
        if signal == "Combined":
            path = self.study.auc_combined(wheel, kind, range_mode)
            name = f"auc_{kind}_{wheel}{suffix}.png"
        else:
            path = self.study.auc(wheel, signal, kind, range_mode)
            channel = self.study.channel_for(wheel, signal) or f"{wheel}_{signal}"
            name = f"auc_{kind}_{channel}{suffix}.png"
        return path, name

    def _update_view(self) -> None:
        path, name = self._resolve()
        self._current = path
        self.view.load(path)
        self.caption.setText(name if path else f"{name}  (not generated)")

    def _fullscreen(self) -> None:
        if self._current:
            ImageViewerDialog.show_for(self._current, self)
