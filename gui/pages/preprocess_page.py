"""Screen — Preprocess: interactive sanitisation, filtering, resampling with a
live before/after preview on the active study's data (aims B, 5, 6)."""

from __future__ import annotations


import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QFormLayout, QComboBox, QDoubleSpinBox, QSpinBox,
    QCheckBox, QLabel, QSlider, QPushButton,
)

from gui import theme
from gui.pages.base_page import BasePage
from gui.widgets.common import SectionTitle, Card
from gui.widgets.mpl_canvas import PlotPanel
from dtt.preprocessing import PreprocessSettings, apply_pipeline, summary_stats

_WINDOW = 2000     # samples shown in the preview window


class PreprocessPage(BasePage):
    def __init__(self, repo, parent=None):
        super().__init__(repo, parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)

        head = QHBoxLayout()
        head.addWidget(SectionTitle("Preprocess — Sanitize · Filter · Resample"))
        head.addStretch(1)
        self.channel_combo = QComboBox()
        self.channel_combo.setMinimumWidth(140)
        self.channel_combo.currentTextChanged.connect(self._load_channel)
        head.addWidget(QLabel("Channel:"))
        head.addWidget(self.channel_combo)
        self.max_btn = QPushButton("⛶ Maximize"); self.max_btn.setObjectName("Secondary")
        self.max_btn.setCheckable(True); self.max_btn.toggled.connect(self._toggle_maximize)
        head.addWidget(self.max_btn)
        root.addLayout(head)

        body = QHBoxLayout()
        body.setSpacing(16)
        root.addLayout(body, 1)

        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.setInterval(90)
        self._timer.timeout.connect(self._update)

        # Controls
        ctrl = Card()
        self.ctrl = ctrl
        ctrl.setMaximumWidth(320)
        ctrl.layout().addWidget(_title("Sanitization"))
        form = QFormLayout(); form.setSpacing(10)

        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0, 100000); self.threshold.setSuffix(" daN")
        self.threshold.setToolTip("Remove samples with |force| below this (aim 5)")
        form.addRow("Lower threshold", self.threshold)

        self.outlier_check = QCheckBox("Remove outliers (percentile)")
        form.addRow(self.outlier_check)
        self.olo = QDoubleSpinBox(); self.olo.setRange(0, 50); self.olo.setValue(1.0); self.olo.setSuffix(" %")
        self.ohi = QDoubleSpinBox(); self.ohi.setRange(50, 100); self.ohi.setValue(99.0); self.ohi.setSuffix(" %")
        form.addRow("  low pct", self.olo)
        form.addRow("  high pct", self.ohi)
        ctrl.layout().addLayout(form)

        ctrl.layout().addWidget(_title("Filtering"))
        f2 = QFormLayout(); f2.setSpacing(10)
        self.filter_check = QCheckBox("Butterworth low-pass"); self.filter_check.setChecked(True)
        ctrl.layout().addWidget(self.filter_check)
        self.cutoff = QDoubleSpinBox(); self.cutoff.setRange(0.1, 1000); self.cutoff.setValue(10.0); self.cutoff.setSuffix(" Hz")
        self.order = QSpinBox(); self.order.setRange(1, 12); self.order.setValue(4)
        f2.addRow("Cutoff", self.cutoff)
        f2.addRow("Order", self.order)
        ctrl.layout().addLayout(f2)

        ctrl.layout().addWidget(_title("Resampling"))
        f3 = QFormLayout(); f3.setSpacing(10)
        self.resample = QSpinBox(); self.resample.setRange(1, 50); self.resample.setValue(1)
        self.resample.setToolTip("Decimation factor (FAMOS red). 10 = 1000→100 Hz")
        f3.addRow("Decimate ×", self.resample)
        ctrl.layout().addLayout(f3)

        self.stats_label = QLabel("—")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet(
            f"color:{theme.TEXT_MUTED}; font-family:monospace; font-size:11px;")
        ctrl.layout().addWidget(self.stats_label)
        ctrl.layout().addStretch(1)
        body.addWidget(ctrl)

        # Preview
        preview = Card()
        self.plot = PlotPanel(height=3.6)
        self.canvas = self.plot.canvas
        preview.layout().addWidget(self.plot, 1)
        self.window_slider = QSlider(Qt.Horizontal)
        self.window_slider.setMinimum(0); self.window_slider.setMaximum(0)
        self.window_slider.valueChanged.connect(self._schedule)
        preview.layout().addWidget(QLabel("Scroll window:"))
        preview.layout().addWidget(self.window_slider)
        body.addWidget(preview, 1)

        # live wiring (debounced)
        for wdg in (self.threshold, self.olo, self.ohi, self.cutoff):
            wdg.valueChanged.connect(self._schedule)
        self.order.valueChanged.connect(self._schedule)
        self.resample.valueChanged.connect(self._schedule)
        self.outlier_check.toggled.connect(self._schedule)
        self.filter_check.toggled.connect(self._schedule)

        self._time = np.array([])
        self._data = np.array([])
        self._fs = 100.0

    # Data
    def refresh(self) -> None:
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        if self.study and self.study.has_stats:
            self.channel_combo.addItems(list(self.study.stats().keys()))
        self.channel_combo.blockSignals(False)
        if self.channel_combo.count():
            self._load_channel(self.channel_combo.currentText())
        else:
            self.canvas.clear(); self.canvas.draw()
            self.stats_label.setText("No processed data for this study.")

    def _load_channel(self, ch: str) -> None:
        if not ch or not self.study or not self.study.processed_csv.exists():
            return
        try:
            cols = ["Time", ch]
            df = pd.read_csv(self.study.processed_csv, usecols=lambda c: c in cols)
        except (ValueError, OSError):
            return
        if ch not in df.columns:
            return
        self._data = pd.to_numeric(df[ch], errors="coerce").to_numpy(dtype=float)
        if "Time" in df.columns:
            t = pd.to_numeric(df["Time"], errors="coerce").to_numpy(dtype=float)
            self._time = t
            dt = np.nanmedian(np.diff(t[:1000])) if t.size > 2 else 0.01
            self._fs = 1.0 / dt if dt and dt > 0 else 100.0
        else:
            self._time = np.arange(self._data.size) / self._fs
        max_start = max(0, self._data.size - _WINDOW)
        self.window_slider.setMaximum(max_start)
        self.window_slider.setValue(0)
        self._update()

    # Live preview
    def _schedule(self, *_) -> None:
        self._timer.start()

    def _toggle_maximize(self, on: bool) -> None:
        self.ctrl.setVisible(not on)

    def _settings(self) -> PreprocessSettings:
        return PreprocessSettings(
            lower_threshold=self.threshold.value(),
            remove_outliers=self.outlier_check.isChecked(),
            outlier_low_pct=self.olo.value(),
            outlier_high_pct=self.ohi.value(),
            apply_filter=self.filter_check.isChecked(),
            filter_cutoff=self.cutoff.value(),
            filter_order=self.order.value(),
            resample_factor=self.resample.value(),
        )

    def _update(self, *_) -> None:
        if self._data.size == 0:
            return
        start = self.window_slider.value()
        sl = slice(start, start + _WINDOW)
        raw = self._data[sl]
        t = self._time[sl]
        s = self._settings()
        proc, new_fs = apply_pipeline(raw, self._fs, s)
        t_proc = t[::s.resample_factor] if s.resample_factor > 1 else t
        t_proc = t_proc[:proc.size]

        ax = self.canvas.ax
        self.canvas.clear()
        ax.plot(t, raw, color=theme.TEXT_FAINT, linewidth=0.7, alpha=0.7, label="raw")
        ax.plot(t_proc, proc, color=theme.ACCENT, linewidth=1.1, label="processed")
        ax.set_xlabel("Time (s)"); ax.set_ylabel(f"{self.channel_combo.currentText()} (daN)")
        ax.legend(fontsize=8, facecolor=theme.SURFACE, labelcolor=theme.TEXT, framealpha=0.9)
        self.canvas.draw()

        a, b = summary_stats(raw), summary_stats(proc)
        self.stats_label.setText(
            f"fs: {self._fs:.1f} → {new_fs:.1f} Hz\n"
            f"samples: {a['n']} → {b['n']}  (removed {b['removed']})\n"
            f"mean: {a['mean']:.1f} → {b['mean']:.1f} daN\n"
            f"std:  {a['std']:.1f} → {b['std']:.1f} daN\n"
            f"min/max: {b['min']:.0f} / {b['max']:.0f} daN")


def _title(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight:700; font-size:12px; margin-top:6px;")
    return lbl