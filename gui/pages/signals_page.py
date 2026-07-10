"""Screen — Signals: FAMOS-style graph validation (time overlay), FFT amplitude
spectrum (dB vs log-Hz), and channel time-synchronisation (aims 3, 8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QComboBox, QSlider, QButtonGroup,
)

from gui import theme
from gui.pages.base_page import BasePage
from gui.widgets.common import SectionTitle, Card
from gui.widgets.mpl_canvas import PlotPanel
from dtt.spectral import amplitude_spectrum_db, estimate_lag, log_downsample

_WINDOW = 3000            # samples shown in the time-domain view
_FFT_WINDOW = 16384       # samples analysed for the spectrum at the scroll position
_PALETTE = ["#00B4D8", "#E74C3C", "#27AE60", "#E8862A", "#8E44AD", "#F1C40F"]


class SignalsPage(BasePage):
    def __init__(self, repo, parent=None):
        super().__init__(repo, parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)

        head = QHBoxLayout()
        head.addWidget(SectionTitle("Signals — Graph Validation · FFT Spectrum · Time-Sync"))
        head.addStretch(1)
        self.time_btn = QPushButton("Time Domain"); self.time_btn.setObjectName("TabPill"); self.time_btn.setCheckable(True); self.time_btn.setChecked(True)
        self.fft_btn = QPushButton("FFT Spectrum"); self.fft_btn.setObjectName("TabPill"); self.fft_btn.setCheckable(True)
        grp = QButtonGroup(self); grp.setExclusive(True); grp.addButton(self.time_btn); grp.addButton(self.fft_btn)
        self.time_btn.clicked.connect(self._schedule); self.fft_btn.clicked.connect(self._schedule)
        head.addWidget(self.time_btn); head.addWidget(self.fft_btn)
        self.max_btn = QPushButton("⛶ Maximize"); self.max_btn.setObjectName("Secondary")
        self.max_btn.setCheckable(True); self.max_btn.toggled.connect(self._toggle_maximize)
        head.addWidget(self.max_btn)
        root.addLayout(head)

        body = QHBoxLayout(); body.setSpacing(16)
        root.addLayout(body, 1)

        # debounce so dragging sliders/spinboxes doesn't redraw on every tick
        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.setInterval(90)
        self._timer.timeout.connect(self._update)

        # Controls
        self.ctrl = Card(); self.ctrl.setMaximumWidth(300)
        self.ctrl.layout().addWidget(_title("Channels (overlay)"))
        self.channel_list = QListWidget()
        self.channel_list.setMaximumHeight(260)
        self.channel_list.itemChanged.connect(self._schedule)
        self.ctrl.layout().addWidget(self.channel_list)
        ctrl = self.ctrl

        ctrl.layout().addWidget(_title("Time Synchronisation"))
        self.ref_combo = QComboBox()
        ctrl.layout().addWidget(QLabel("Reference channel:"))
        ctrl.layout().addWidget(self.ref_combo)
        self.sync_btn = QPushButton("⟲ Auto-sync to reference")
        self.sync_btn.setObjectName("Secondary")
        self.sync_btn.clicked.connect(self._auto_sync)
        ctrl.layout().addWidget(self.sync_btn)
        self.reset_btn = QPushButton("Reset offsets")
        self.reset_btn.setObjectName("Secondary")
        self.reset_btn.clicked.connect(self._reset_sync)
        ctrl.layout().addWidget(self.reset_btn)
        self.sync_label = QLabel("Offsets: none")
        self.sync_label.setWordWrap(True)
        self.sync_label.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:11px; font-family:monospace;")
        ctrl.layout().addWidget(self.sync_label)
        ctrl.layout().addStretch(1)
        body.addWidget(ctrl)

        # Canvas
        canvas_card = Card()
        self.plot = PlotPanel(height=4.0)
        self.canvas = self.plot.canvas
        canvas_card.layout().addWidget(self.plot, 1)
        self.window_slider = QSlider(Qt.Horizontal)
        self.window_slider.valueChanged.connect(self._schedule)
        self.slider_label = QLabel("Scroll analysis window:")
        self.slider_label.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:11px;")
        canvas_card.layout().addWidget(self.slider_label)
        canvas_card.layout().addWidget(self.window_slider)
        body.addWidget(canvas_card, 1)

        self._cache: dict[str, np.ndarray] = {}
        self._offsets: dict[str, int] = {}
        self._time = np.array([])
        self._fs = 100.0

    # Data
    def refresh(self) -> None:
        self._cache.clear(); self._offsets.clear()
        self.channel_list.blockSignals(True)
        self.channel_list.clear()
        self.ref_combo.clear()
        channels = list(self.study.stats().keys()) if (self.study and self.study.has_stats) else []
        for i, ch in enumerate(channels):
            item = QListWidgetItem(ch)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if i < 2 else Qt.Unchecked)
            self.channel_list.addItem(item)
        self.ref_combo.addItems(channels)
        self.channel_list.blockSignals(False)
        self._load_time()
        self._update()

    def _load_time(self) -> None:
        self._time = np.array([]); self._fs = 100.0
        if not self.study or not self.study.processed_csv.exists():
            return
        try:
            df = pd.read_csv(self.study.processed_csv, usecols=lambda c: c == "Time")
            t = pd.to_numeric(df["Time"], errors="coerce").to_numpy(dtype=float)
            self._time = t
            dt = np.nanmedian(np.diff(t[:1000])) if t.size > 2 else 0.01
            self._fs = 1.0 / dt if dt and dt > 0 else 100.0
            self.window_slider.setMaximum(max(0, t.size - _WINDOW))
            self.window_slider.setValue(0)
        except (OSError, ValueError, KeyError):
            pass

    def _channel_data(self, ch: str) -> np.ndarray:
        if ch not in self._cache:
            try:
                df = pd.read_csv(self.study.processed_csv, usecols=lambda c: c == ch)
                self._cache[ch] = pd.to_numeric(df[ch], errors="coerce").to_numpy(dtype=float)
            except (OSError, ValueError, KeyError):
                self._cache[ch] = np.array([])
        return self._cache[ch]

    def _checked_channels(self) -> list[str]:
        out = []
        for i in range(self.channel_list.count()):
            it = self.channel_list.item(i)
            if it.checkState() == Qt.Checked:
                out.append(it.text())
        return out

    # Time sync
    def _auto_sync(self) -> None:
        ref = self.ref_combo.currentText()
        if not ref:
            return
        ref_data = self._channel_data(ref)
        for ch in self._checked_channels():
            if ch == ref:
                self._offsets[ch] = 0
                continue
            lag = estimate_lag(ref_data, self._channel_data(ch))
            self._offsets[ch] = lag
        self._update()

    def _reset_sync(self) -> None:
        self._offsets.clear()
        self._update()

    # Render
    def _schedule(self, *_) -> None:
        self._timer.start()

    def _toggle_maximize(self, on: bool) -> None:
        self.ctrl.setVisible(not on)

    def _update(self, *_) -> None:
        channels = self._checked_channels()
        ax = self.canvas.ax
        self.canvas.clear()
        if not channels:
            ax.text(0.5, 0.5, "Select one or more channels", ha="center", va="center",
                    transform=ax.transAxes, color=theme.TEXT_FAINT)
            self.canvas.draw(); return

        if self.fft_btn.isChecked():
            self._draw_fft(channels)
        else:
            self._draw_time(channels)
        self.canvas.draw()
        self._update_sync_label()

    def _draw_time(self, channels: list[str]) -> None:
        ax = self.canvas.ax
        self.window_slider.setVisible(True); self.slider_label.setVisible(True)
        start = self.window_slider.value()
        for i, ch in enumerate(channels):
            data = self._channel_data(ch)
            off = self._offsets.get(ch, 0)
            s0 = start + off
            seg = data[max(0, s0):max(0, s0) + _WINDOW]
            t = self._time[start:start + len(seg)]
            ax.plot(t[:len(seg)], seg, linewidth=0.9,
                    color=_PALETTE[i % len(_PALETTE)], label=ch)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Force (daN)")
        ax.set_title("Time Domain — synchronised overlay")
        ax.legend(fontsize=8, facecolor=theme.SURFACE, labelcolor=theme.TEXT, framealpha=0.9)

    def _draw_fft(self, channels: list[str]) -> None:
        ax = self.canvas.ax
        self.window_slider.setVisible(True); self.slider_label.setVisible(True)
        start = self.window_slider.value()
        span = 0.0
        for i, ch in enumerate(channels):
            data = self._channel_data(ch)
            seg = data[start:start + _FFT_WINDOW]
            span = max(span, len(seg) / self._fs)
            f, db = amplitude_spectrum_db(seg, self._fs)
            f, db = log_downsample(f, db)
            ax.semilogx(f, db, linewidth=0.9,
                        color=_PALETTE[i % len(_PALETTE)], label=ch)
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Amplitude (dB)")
        ax.set_title(f"FFT Spectrum  ({span:.0f}s window @ scroll pos · "
                     f"fs {self._fs:.0f} Hz, Nyquist {self._fs/2:.0f} Hz)")
        ax.legend(fontsize=8, facecolor=theme.SURFACE, labelcolor=theme.TEXT, framealpha=0.9)

    def _update_sync_label(self) -> None:
        active = {c: o for c, o in self._offsets.items()
                  if o and c in self._checked_channels()}
        if not active:
            self.sync_label.setText("Offsets: none (aligned)")
            return
        lines = [f"{c}: {o:+d} samp ({o / self._fs * 1000:+.0f} ms)" for c, o in active.items()]
        self.sync_label.setText("Offsets:\n" + "\n".join(lines))


def _title(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight:700; font-size:12px; margin-top:6px;")
    return lbl