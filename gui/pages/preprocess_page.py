"""Screen — Preprocess: interactive sanitisation, filtering, resampling with a
live before/after preview on the active study's data (aims B, 5, 6)."""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QObject, QEvent
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QFormLayout, QComboBox, QDoubleSpinBox, QSpinBox,
    QCheckBox, QLabel, QPushButton, QScrollArea, QFileDialog, QMessageBox,
    QApplication, QAbstractSpinBox,
)

from gui import theme
from gui.pages.base_page import BasePage
from gui.widgets.common import SectionTitle, Card
from gui.widgets.mpl_canvas import PlotPanel
from dtt.preprocessing import (
    PreprocessSettings, apply_pipeline, summary_stats, famos_recipe,
    is_wft_channel, blank_dead_runs,
)

_MAX_PLOT_POINTS = 1400        # fallback before the canvas has been laid out

# FAMOS convention, used across the screen: blue is the raw channel, red is the
# conditioned one. The Comparison screen uses the same two for previous/current.
_MOMENT_RE = re.compile(r"(?:^|[_\W])M[xyz](?:$|[_\W])", re.I)

_RAW_COLOR = "#4a9eff"
_PROC_COLOR = "#ff4d4f"


def _plot_points(canvas) -> int:
    """How many buckets to reduce to: about one per horizontal pixel.

    Reducing to a fixed 6000 was the reason the band read as a solid smear —
    the canvas is nearer 1000 px wide, so six buckets stacked on every pixel and
    a min→max envelope collapsed into featureless hash. One bucket per pixel is
    the most detail the screen can actually resolve; past that, extra buckets add
    ink, not information. Zooming re-draws, so detail comes back on demand.
    """
    try:
        px = int(canvas.width())
    except (AttributeError, TypeError):
        return _MAX_PLOT_POINTS
    return int(np.clip(px, 600, 4000)) if px > 0 else _MAX_PLOT_POINTS


def _buckets(t: np.ndarray, y: np.ndarray, max_points: int):
    """Reshape into ``max_points`` equal time buckets, NaN-padded at the end."""
    n = y.size
    nb = max(1, max_points)
    width = int(np.ceil(n / nb))
    pad = width * nb - n
    yp = np.concatenate([y, np.full(pad, np.nan)]).reshape(nb, width)
    tp = np.concatenate([t, np.full(pad, np.nan)]).reshape(nb, width)
    return tp, yp


def _reduce(t: np.ndarray, y: np.ndarray, max_points: int = _MAX_PLOT_POINTS):
    """Reduce to ``(t, lo, mid, hi)`` — each bucket's min, median and max.

    Every sample lands in exactly one bucket, so nothing is thrown away and no
    sample can alias into or out of view (which is what stride-decimating to
    screen width does: at 555k samples over 6000 pixels it keeps one in ~90 and
    lets isolated samples masquerade as spikes).

    Returning all three from one reduction is the point: the two traces want
    different halves of it. The raw is drawn from ``lo``/``hi`` so every
    excursion in the recording still reaches the screen, and the conditioned
    trace from ``mid`` so it reads as the single smooth line FAMOS shows. Read
    the pair the way FAMOS is read — the blue is the channel's full reach, the
    red is where the conditioned signal sits — and note that at full-run zoom
    part of the gap between them is the reduction itself, not only the filter.
    Zoom in past ``max_points`` samples and both become the true samples.

    Empty buckets keep a finite time and a NaN value, so matplotlib breaks the
    line over a dropout instead of ruling a straight segment across it.
    """
    if y.size <= max_points:
        return t, y, y, y
    tp, yp = _buckets(t, y, max_points)
    with warnings.catch_warnings():
        # all-NaN buckets are intentional (padding at the tail, dropouts in the
        # middle) and become NaN in the output — the empty-slice warning is not
        # news, and the gaps are what we want on screen
        warnings.simplefilter("ignore", RuntimeWarning)
        lo = np.nanmin(yp, axis=1)
        mid = np.nanmedian(yp, axis=1)
        hi = np.nanmax(yp, axis=1)
        tm = np.nanmean(tp, axis=1)
    ok = np.isfinite(tm)
    return tm[ok], lo[ok], mid[ok], hi[ok]


def _spikes(t: np.ndarray, raw: np.ndarray, proc: np.ndarray, n_sigmas: float,
            max_marks: int = 400):
    """Locate the raw samples the conditioning flattened, biggest first.

    The residual ``raw - processed`` is what the filtering removed, so scoring it
    against its own robust σ finds the excursions that stand out from the ripple
    the filter takes off everywhere. Marking these is the only way the operator
    can see *what* was removed — a smoothed trace on its own looks equally
    plausible whether it swallowed a genuine 900 daN pothole strike or nothing
    at all.

    Only the largest ``max_marks`` are returned: scattering ten thousand points
    over a 5,000 s trace paints a solid bar and hides the very outliers it is
    supposed to call out.
    """
    n = min(t.size, raw.size, proc.size)
    if n == 0:
        return np.empty(0), np.empty(0), 0
    t, raw, proc = t[:n], raw[:n], proc[:n]
    resid = raw - proc
    ok = np.isfinite(resid)
    if ok.sum() < 8:
        return np.empty(0), np.empty(0), 0
    sigma = 1.4826 * np.median(np.abs(resid[ok] - np.median(resid[ok])))
    if not np.isfinite(sigma) or sigma <= 0:
        return np.empty(0), np.empty(0), 0
    hit = ok & (np.abs(resid) > n_sigmas * sigma)
    idx = np.flatnonzero(hit)
    total = int(idx.size)
    if total > max_marks:
        idx = idx[np.argsort(np.abs(resid[idx]))[-max_marks:]]
    return t[idx], raw[idx], total


def _envelope_line(t: np.ndarray, y: np.ndarray, max_points: int = _MAX_PLOT_POINTS):
    """The raw channel as FAMOS draws it — min→max strokes, nothing dropped.

    Pixel-identical to plotting every sample, at a fraction of the cost: within
    one bucket a full-resolution line plot can only paint the column between
    that bucket's min and max, so emitting exactly those two points in time
    order reproduces the same ink. What it is *not* is a summary — no averaging,
    no shaded band; every extreme in the recording still reaches the screen,
    which is why a spike stays visible as a spike.

    Strokes, never a filled band: a fill reads as a solid block and buries the
    red line that has to be legible through it.
    """
    if y.size <= max_points:
        return t, y
    tm, lo, _, hi = _reduce(t, y, max_points)
    tt = np.repeat(tm, 2)
    yy = np.empty(tt.size, dtype=float)
    # min then max within each bucket; the join between buckets is the same
    # vertical stroke a dense line plot would draw anyway
    yy[0::2], yy[1::2] = lo, hi
    return tt, yy


def _median_line(t: np.ndarray, y: np.ndarray, max_points: int = _MAX_PLOT_POINTS):
    """The conditioned channel as FAMOS draws it — one clean centre line.

    This is the half of the FAMOS look a min/max envelope cannot give. Reducing
    the sanitized trace to *its* min/max paints it as a second band the same
    height as the raw, so the two colours simply overprint and smoothing that
    demonstrably removed the ripple arrives on screen as an identical smear.
    The bucket median is immune to the lone sample that sets a bucket's min or
    max, so the red comes through as the smooth line it actually is — running
    down the middle of the blue, with the raw's excursions standing outside it,
    which is exactly how FAMOS shows what the conditioning took off.
    """
    if y.size <= max_points:
        return t, y
    tm, _, mid, _ = _reduce(t, y, max_points)
    return tm, mid


def _channel_unit(channel: str) -> str:
    """Axis unit for ``channel``, following the imc config file's YUNIT lines.

    The screen used to hard-code "daN" because the channel list only ever held
    forces. It now lists every column in the processed CSV, so a hard-coded
    force unit would label Latacc in daN and Vehicle_Speed in daN — a plot that
    states the wrong unit is worse than one that states none.
    """
    key = channel.strip().lower()
    if key in ("latacc", "lat_acc", "longacc", "long_acc", "latacc_lpf",
               "acceleration", "accelz"):
        return "m/s²"
    if key in ("velforward", "vellateral"):
        return "m/s"                       # velocities, not accelerations
    if key in ("vehicle_speed", "speed_kmph", "speed2d", "gps.speed"):
        return "km/h"
    if key in ("yawrate", "angratex", "angratey") or "anglespeed" in key:
        return "°/s"
    if key.startswith("angle") or key.endswith("_angle") or "_angle_" in key:
        return "°"
    if key in ("distance", "dist", "altitude"):
        return "m"
    if key in ("latitude", "longitude"):
        return "°"
    if is_wft_channel(channel):
        # forces daN, moments daN·m — both smo(0.1) in the recipe
        return "daN·m" if _MOMENT_RE.search(channel) else "daN"
    return ""


def _famos_op(channel: str) -> str:
    """The operator the imc config file specifies for ``channel``.

    Read straight off ``famos_recipe`` — which is the transcription of
    "imc coding_filtering_Channel mapping.txt" — so the red trace's legend
    names the exact FAMOS step that produced it:

        WFT forces / moments    smo(x, 0.1)
        Latacc                  FiltLP(x, 0, 0, 4, 5) then smo(x, 0.5)
        Long_acc, Vehicle_Speed smo(x, 0.5)
        GPS / yaw               passthrough

    A red line the operator cannot trace back to a line in that file is just a
    smooth curve; naming it is what makes the plot checkable against FAMOS.
    """
    r = famos_recipe(channel)
    if r.apply_filter:
        return (f"FiltLP({r.filter_order}, {r.filter_cutoff:g} Hz) "
                f"→ smo({r.smooth_width_s:g} s)")
    if r.smooth_width_s > 0:
        return f"smo({r.smooth_width_s:g} s)"
    return "passthrough"


class _WheelGuard(QObject):
    """Stops the mouse wheel from silently rewriting the processing settings.

    The controls live in a scrollable panel, and every spin box and combo in it
    accepts wheel events by default. Rolling the wheel to reach a lower section
    therefore decrements whatever sits under the pointer on the way past — and
    each change fires ``_schedule`` and re-plots, so the trace quietly becomes
    a picture of settings nobody chose. Observed in the wild: ``smo`` width
    driven to its 0.01 s floor, min gap to 1 s, the high percentile to 100 %,
    all three pinned at a range end, with the panel itself never scrolling
    because the spin boxes consumed the events.

    A focused widget still takes the wheel, which is the one case where the
    user is deliberately dialling that control. Otherwise the event is handed
    to the scroll area so the wheel does the only thing it looked like it was
    doing: scroll.
    """

    def __init__(self, area: QScrollArea, parent=None):
        super().__init__(parent)
        self._area = area

    def eventFilter(self, obj, ev):                    # noqa: N802
        if ev.type() != QEvent.Type.Wheel or obj.hasFocus():
            return False
        if self._area is not None:
            QApplication.sendEvent(self._area.viewport(), ev)
        return True                                    # never reaches the widget


class _FamosCheckWorker(QThread):
    """Runs the FAMOS cross-check off the UI thread."""
    done = Signal(object, object, object)     # (matches, note, error)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        try:
            from dtt.validation.famos_validation import crosscheck_csv
            matches, note = crosscheck_csv(self.path)
            self.done.emit(matches, note, None)
        except Exception as exc:
            self.done.emit(None, None, f"{type(exc).__name__}: {exc}")


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
        self.famos_auto = QCheckBox("FAMOS recipe")
        self.famos_auto.setChecked(False)
        self.famos_auto.setToolTip(
            "Apply the imc/FAMOS recipe for the selected channel automatically:\n"
            "  WFT forces & moments   smo(x, 0.1)\n"
            "  Latacc                 FiltLP(4, 5 Hz) then smo(x, 0.5)\n"
            "  Longacc, Vehicle_Speed smo(x, 0.5)\n"
            "  GPS / yaw              passthrough\n\n"
            "A study ingested from imc raw already had this applied at its "
            "native rate (the only correct place — before red()), so leave this "
            "off to view that result. Turn it on to preview a further pass, or "
            "for study data that was never conditioned.")
        self.famos_auto.toggled.connect(self._on_famos_auto)
        head.addWidget(self.famos_auto)
        self.max_btn = QPushButton("⛶ Maximize"); self.max_btn.setObjectName("Secondary")
        self.max_btn.setCheckable(True); self.max_btn.toggled.connect(self._toggle_maximize)
        head.addWidget(self.max_btn)
        root.addLayout(head)

        body = QHBoxLayout()
        body.setSpacing(16)
        root.addLayout(body, 1)

        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.setInterval(90)
        self._timer.timeout.connect(self._update)

        # Controls (scrollable so all sections stay reachable on any window size)
        ctrl = Card()
        self.ctrl = ctrl
        self.ctrl_scroll = QScrollArea()
        self.ctrl_scroll.setWidgetResizable(True)
        self.ctrl_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ctrl_scroll.setFrameShape(QScrollArea.NoFrame)
        self.ctrl_scroll.setWidget(ctrl)
        self.ctrl_scroll.setMinimumWidth(300)
        self.ctrl_scroll.setMaximumWidth(340)
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

        ctrl.layout().addWidget(_title("Spike de-glitch"))
        self.spike_check = QCheckBox("Remove DAQ artifact spikes")
        self.spike_check.setChecked(False)
        self.spike_check.setToolTip(
            "Compare each sample against a rolling MEDIAN of its neighbours and "
            "bridge the ones that sit far outside it.\nThe median tracks genuine "
            "road-load peaks, so only true artifacts are flagged — and each one "
            "is interpolated across, never flattened onto the centre line.\n"
            "Leave off for a cleanly-read imc file; turn on if the raw trace "
            "shows isolated out-of-family spikes.")
        ctrl.layout().addWidget(self.spike_check)
        fs2 = QFormLayout(); fs2.setSpacing(10)
        self.spike_nsigma = QDoubleSpinBox(); self.spike_nsigma.setRange(2.0, 15.0)
        self.spike_nsigma.setSingleStep(0.5); self.spike_nsigma.setValue(6.0)
        self.spike_nsigma.setToolTip("Flag samples beyond this many robust σ from the "
                                     "rolling median. Higher = only the most extreme.")
        self.spike_strength = QDoubleSpinBox(); self.spike_strength.setRange(0.0, 1.0)
        self.spike_strength.setSingleStep(0.1); self.spike_strength.setValue(1.0)
        self.spike_strength.setToolTip("0 = leave as-is, 1 = fully bridge the spike "
                                       "(interpolate across it). Lower keeps some height.")
        fs2.addRow("Threshold (σ)", self.spike_nsigma)
        fs2.addRow("Strength", self.spike_strength)
        ctrl.layout().addLayout(fs2)
        ctrl.layout().addWidget(_title("Gap bridging"))
        self.gap_check = QCheckBox("Remove long gaps")
        self.gap_check.setToolTip("Excise NaN / flat-signal runs and re-stitch the timeline.")
        ctrl.layout().addWidget(self.gap_check)
        fg = QFormLayout(); fg.setSpacing(10)
        self.gap_seconds = QDoubleSpinBox(); self.gap_seconds.setRange(1.0, 120.0)
        self.gap_seconds.setValue(10.0); self.gap_seconds.setSuffix(" s")
        fg.addRow("Min gap", self.gap_seconds)
        ctrl.layout().addLayout(fg)
        ctrl.layout().addWidget(_title("Smoothing (FAMOS smo)"))
        self.smooth_check = QCheckBox("FAMOS smo smoothing"); self.smooth_check.setChecked(True)
        self.smooth_check.setToolTip(
            "FAMOS smo(x, width) — the exact smoothing the imc recipe applies to "
            "WFT forces, smo(FL_Fx1, 0.1).\nA triangular kernel (two cascaded "
            "box-cars of width/2), verified to 99.995% against a real FAMOS export.")
        ctrl.layout().addWidget(self.smooth_check)
        f1 = QFormLayout(); f1.setSpacing(10)
        self.smooth_width = QDoubleSpinBox(); self.smooth_width.setRange(0.01, 5.0)
        self.smooth_width.setSingleStep(0.05); self.smooth_width.setValue(0.10); self.smooth_width.setSuffix(" s")
        self.smooth_width.setToolTip("Smoothing width in seconds (FAMOS uses 0.1 s for forces).")
        f1.addRow("Width", self.smooth_width)
        ctrl.layout().addLayout(f1)

        ctrl.layout().addWidget(_title("FiltLP — Butterworth (accel)"))
        f2 = QFormLayout(); f2.setSpacing(10)
        self.filter_check = QCheckBox("Butterworth low-pass"); self.filter_check.setChecked(False)
        self.filter_check.setToolTip(
            "FAMOS FiltLP(...,4,5) — applied to accel/Latacc only, never to the "
            "forces (those get smo).\nCausal (single-pass), matching FAMOS: the "
            "filtered channel carries a real lag against an unfiltered reference.")
        ctrl.layout().addWidget(self.filter_check)
        self.cutoff = QDoubleSpinBox(); self.cutoff.setRange(0.1, 1000); self.cutoff.setValue(5.0); self.cutoff.setSuffix(" Hz")
        self.order = QSpinBox(); self.order.setRange(1, 12); self.order.setValue(4)
        f2.addRow("Cutoff", self.cutoff)
        f2.addRow("Order", self.order)
        ctrl.layout().addLayout(f2)
        ctrl.layout().addWidget(_title("Resampling"))
        f3 = QFormLayout(); f3.setSpacing(10)
        self.resample = QSpinBox(); self.resample.setRange(1, 50); self.resample.setValue(1)
        self.resample.setToolTip(
            "Decimation factor — FAMOS red(x, 10).\nLeave at 1 when the study data "
            "has already been reduced during ingestion; decimating twice throws "
            "away real bandwidth.")
        f3.addRow("Decimate ×", self.resample)
        ctrl.layout().addLayout(f3)
        self.stats_label = QLabel("—")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet(
            f"color:{theme.TEXT_MUTED}; font-family:monospace; font-size:11px;")
        ctrl.layout().addWidget(self.stats_label)
        ctrl.layout().addStretch(1)
        body.addWidget(self.ctrl_scroll)
        preview = Card()
        self.plot = PlotPanel(height=3.6)
        self.canvas = self.plot.canvas
        preview.layout().addWidget(self.plot, 1)
        foot = QHBoxLayout()
        self.range_label = QLabel("Full recording — use the toolbar to zoom/pan.")
        # Word wrap is what lets this shrink. A QLabel with wrapping off reports
        # its whole text width as its *minimum*, so this one status line was
        # pinning the footer row — and with it the window — to 1342 px, forcing
        # a minimum width no 1920 px display could satisfy once the sidebar and
        # the other footer controls were added on.
        self.range_label.setWordWrap(True)
        self.range_label.setMinimumWidth(140)
        self.range_label.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:11px;")
        foot.addWidget(self.range_label, 1)
        self.mark_spikes = QCheckBox("Mark spikes")
        self.mark_spikes.setChecked(False)
        self.mark_spikes.setToolTip(
            "Ring the raw samples the conditioning flattened — the excursions "
            "that stand out from the ripple the filter removes everywhere.\n"
            "Scored on the raw-minus-processed residual against its own robust σ, "
            "so it follows whatever stages are enabled.\n"
            "The legend reports the full count; only the largest 400 are drawn, "
            "since more than that paints a solid bar and hides the outliers.")
        self.mark_spikes.toggled.connect(self._schedule)
        foot.addWidget(self.mark_spikes)
        self.spike_mark_nsigma = QDoubleSpinBox()
        self.spike_mark_nsigma.setRange(1.0, 20.0)
        self.spike_mark_nsigma.setSingleStep(0.5)
        self.spike_mark_nsigma.setValue(4.0)
        self.spike_mark_nsigma.setPrefix("σ ")
        self.spike_mark_nsigma.setToolTip("How far outside the residual's robust σ "
                                          "a sample must sit to be ringed.")
        self.spike_mark_nsigma.setMaximumWidth(80)
        self.spike_mark_nsigma.valueChanged.connect(self._schedule)
        foot.addWidget(self.spike_mark_nsigma)

        self.show_raw = QCheckBox("Raw overlay")
        self.show_raw.setChecked(True)
        self.show_raw.setToolTip(
            "Draw the unprocessed channel behind the result, as its true "
            "min/max envelope — every sample is represented, so the red "
            "median line can be seen sitting through the middle of the raw "
            "band, the FAMOS way.\n"
            "Blue = raw reach, red = sanitized centre line. At full-run zoom "
            "some of the gap is the on-screen reduction; zoom in and both "
            "converge on the true samples.")
        self.show_raw.toggled.connect(self._schedule)
        foot.addWidget(self.show_raw)
        self.famos_btn = QPushButton("Validate vs FAMOS…")
        self.famos_btn.setObjectName("Secondary")
        self.famos_btn.clicked.connect(self._validate_famos)
        foot.addWidget(self.famos_btn)
        preview.layout().addLayout(foot)
        body.addWidget(preview, 1)
        for wdg in (self.threshold, self.olo, self.ohi, self.cutoff,
                    self.spike_nsigma, self.spike_strength,
                    self.gap_seconds, self.smooth_width):
            wdg.valueChanged.connect(self._schedule)
        self.order.valueChanged.connect(self._schedule)
        self.resample.valueChanged.connect(self._schedule)
        self.outlier_check.toggled.connect(self._schedule)
        self.filter_check.toggled.connect(self._schedule)
        self.spike_check.toggled.connect(self._schedule)
        self.gap_check.toggled.connect(self._schedule)
        self.smooth_check.toggled.connect(self._schedule)
        # Wheel-proof every value control on the screen, including the header's
        # channel combo and the footer's sigma box — the pointer crosses those
        # too. Focus policy comes down to StrongFocus so a control can only be
        # reached by click or Tab: with the default WheelFocus a spin box takes
        # focus *from* the wheel, which would let the very first notch through
        # and defeat the guard's hasFocus() exemption.
        self._wheel_guard = _WheelGuard(self.ctrl_scroll, self)
        for w in (self.findChildren(QAbstractSpinBox)
                  + self.findChildren(QComboBox)):
            w.setFocusPolicy(Qt.StrongFocus)
            w.installEventFilter(self._wheel_guard)

        self._time = np.array([])
        self._data = np.array([])
        self._raw = None                 # true unconditioned channel, if kept
        self._raw_time = None
        self._fs = 100.0
        self._syncing = False
        self._conditioned = False

    # ---- FAMOS recipe -> controls -------------------------------------------

    def _apply_recipe_to_controls(self, channel: str) -> None:
        """Drive the stage controls from the imc/FAMOS recipe for ``channel``."""
        if not channel:
            return
        r = famos_recipe(channel)
        self._syncing = True                    # don't re-plot on every setter
        try:
            self.smooth_check.setChecked(r.smooth_width_s > 0)
            if r.smooth_width_s > 0:
                self.smooth_width.setValue(r.smooth_width_s)
            self.filter_check.setChecked(r.apply_filter)
            self.cutoff.setValue(r.filter_cutoff)
            self.order.setValue(r.filter_order)
        finally:
            self._syncing = False

    def _study_is_conditioned(self) -> bool:
        """True if this study's ingestion already ran the FAMOS recipe.

        The pipeline log records it; that is the only place the fact survives,
        and it decides whether re-applying the recipe here would double-smooth.

        Both conditioning paths have to be recognised. imc raw ingestion logs
        "applied at ingestion"; a CSV study is conditioned at stage 4 instead and
        logs "applied at <rate> Hz". Matching only the first made every CSV study
        look unconditioned, so the page labelled already-smoothed data "raw" and
        ran a second smo(0.1) over it.
        """
        try:
            if not (self.study and self.study.has_log):
                return False
            log = self.study.log_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        if "FAMOS recipe applied at ingestion" in log:
            return True
        # stage 4: "FAMOS recipe applied at 100 Hz: 19/21 channels conditioned"
        return bool(re.search(r"FAMOS recipe applied at [\d.]+ Hz: (\d+)/", log)
                    and not re.search(r"FAMOS recipe applied at [\d.]+ Hz: 0/", log))

    def _recipe_note(self, s: PreprocessSettings, ch: str,
                     have_raw: bool, sanitizing: bool) -> str:
        """Title line — what the two colours are, stated from the same facts
        the legend uses so the two can never disagree.

        It also has to say when there is only one trace. Only the channels
        stored in raw_data.csv can be shown as a before/after pair; for the
        rest the study kept the conditioned result alone, and a title that
        still promised "blue = raw, red = sanitized" over a single line would
        be describing a plot that is not on screen.
        """
        proc = self._proc_label(s, ch, have_raw, sanitizing)
        if have_raw:
            return f"Raw vs sanitized   (blue = raw, red = {proc.split('— ', 1)[-1]})"
        if self._conditioned:
            return ("Study data — imc/FAMOS recipe applied at ingestion   "
                    "(no raw stored for this channel, so no before/after pair)")
        return f"Study data — {proc.split('— ', 1)[-1]}"

    def _on_famos_auto(self, on: bool) -> None:
        for w in (self.smooth_check, self.smooth_width, self.filter_check,
                  self.cutoff, self.order):
            w.setEnabled(not on)
        if on:
            self._apply_recipe_to_controls(self.channel_combo.currentText())
        self._schedule()

    def refresh(self) -> None:
        self._conditioned = self._study_is_conditioned()
        # Re-applying the recipe on top of an already-conditioned study would
        # smooth it twice, so the auto checkbox only defaults on when nothing
        # has run yet. The per-channel recipe controls are *not* cleared with
        # it any more: `_load_channel` sets them from `famos_recipe`, so the
        # panel states the conditioning the data actually carries, and any
        # sanitisation previewed on top of it keeps that conditioning instead
        # of silently reverting the red trace to unsmoothed raw.
        self._syncing = True
        self.famos_auto.setChecked(not self._conditioned)
        self._syncing = False
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        self.channel_combo.addItems(self._channel_names())
        self.channel_combo.blockSignals(False)
        if self.channel_combo.count():
            self._load_channel(self.channel_combo.currentText())
        else:
            self.canvas.clear(); self.canvas.draw()
            self.stats_label.setText("No processed data for this study.")
    def _channel_names(self) -> list:
        """Every channel this screen can preprocess — the processed CSV's own
        columns.

        This used to come from ``stats_summary.json``, which is built from the
        run's *mandatory* channels only — the Fx/Fy/Fz forces the severity
        analysis needs. On a two-position study that is six entries, so the
        moments, Latacc, Longacc, Vehicle_Speed and the GPS channels were all
        missing from the dropdown even though they sit in processed_data.csv and
        the imc recipe names every one of them. Preprocessing applies to the
        whole file, so the file is what the list has to come from.

        Falls back to the stats keys if the CSV cannot be read, so a study with
        a damaged export still offers whatever it can.
        """
        if not self.study:
            return []
        try:
            if self.study.processed_csv.exists():
                head = pd.read_csv(self.study.processed_csv, nrows=0)
                names = [c for c in head.columns if c != "Time"]
                if names:
                    return names
        except (ValueError, OSError):
            pass
        return list(self.study.stats().keys()) if self.study.has_stats else []

    @staticmethod
    def _read_channel(path, ch: str):
        """``(time, values)`` for one channel, or ``(None, None)``."""
        try:
            cols = ("Time", ch)
            df = pd.read_csv(path, usecols=lambda c: c in cols)
        except (ValueError, OSError):
            return None, None
        if ch not in df.columns:
            return None, None
        y = pd.to_numeric(df[ch], errors="coerce").to_numpy(dtype=float)
        t = (pd.to_numeric(df["Time"], errors="coerce").to_numpy(dtype=float)
             if "Time" in df.columns else None)
        return t, y

    def _load_channel(self, ch: str) -> None:
        if not ch or not self.study or not self.study.processed_csv.exists():
            return
        t, y = self._read_channel(self.study.processed_csv, ch)
        if y is None:
            return
        # Red is the pipeline's own sanitized output, not a second pass over it.
        self._data = y
        if t is not None:
            self._time = t
            dt = np.nanmedian(np.diff(t[:1000])) if t.size > 2 else 0.01
            self._fs = 1.0 / dt if dt and dt > 0 else 100.0
        else:
            self._time = np.arange(y.size) / self._fs

        # Blue is the genuine unconditioned channel when the study kept one.
        # Each trace is plotted against its own Time column: sanitisation can
        # drop rows, so the two frames are not guaranteed to be the same length
        # and index-aligning them would slide one against the other.
        self._raw = self._raw_time = None
        if getattr(self.study, "has_raw", False):
            tr, yr = self._read_channel(self.study.raw_csv, ch)
            if yr is not None:
                self._raw = yr
                self._raw_time = (tr if tr is not None
                                  else np.arange(yr.size) / self._fs)

        # The recipe is per-channel (smo 0.1 for forces, FiltLP+smo 0.5 for
        # Latacc, passthrough for GPS), so the controls have to follow the
        # selection whether the recipe is re-applied here or was baked in at
        # ingestion — otherwise the panel describes the previous channel.
        self._apply_recipe_to_controls(ch)
        self._update()
    def _schedule(self, *_) -> None:
        if self._syncing:
            return
        self._timer.start()
    def _toggle_maximize(self, on: bool) -> None:
        self.ctrl_scroll.setVisible(not on)

    def _validate_famos(self) -> None:
        start = str(self.repo.csv_dir) if hasattr(self.repo, "csv_dir") else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a FAMOS CSV export (raw + *_LPF columns)", start,
            "CSV files (*.csv)")
        if not path:
            return
        self.famos_btn.setEnabled(False)
        self.famos_btn.setText("Validating…")
        self._fam_worker = _FamosCheckWorker(path)
        self._fam_worker.done.connect(self._on_famos_result)
        self._fam_worker.start()

    def _on_famos_result(self, matches, note, err) -> None:
        self.famos_btn.setEnabled(True)
        self.famos_btn.setText("Validate vs FAMOS…")
        if err:
            QMessageBox.warning(self, "FAMOS validation", f"Failed:\n{err}")
            return
        if note:
            QMessageBox.information(self, "FAMOS validation", note)
            return
        best = max((m.match_pct for m in matches), default=0.0)
        ok = bool(matches) and all(m.passed for m in matches)
        head = (f"{'✓ FAMOS-GRADE' if ok else '✗ Below 95%'} — "
                f"best {best:.3f}% match (threshold 95%)\n\n")
        lines = [f"{'PASS' if m.passed else 'FAIL'}  {m.describe()}\n"
                 f"      {m.match_pct:.3f}% match · {m.corr_pct:.3f}% corr · "
                 f"{m.within_pct:.1f}% of samples within 2%  (n={m.n:,})"
                 for m in matches]
        QMessageBox.information(self, "FAMOS validation", head + "\n".join(lines))
    def _settings(self) -> PreprocessSettings:
        if self.famos_auto.isChecked():
            s = famos_recipe(self.channel_combo.currentText())
        else:
            s = PreprocessSettings(
                smooth_width_s=(self.smooth_width.value()
                                if self.smooth_check.isChecked() else 0.0),
                apply_filter=self.filter_check.isChecked(),
                filter_cutoff=self.cutoff.value(),
                filter_order=self.order.value(),
            )
        # Sanitisation stages are always the operator's call — the FAMOS recipe
        # covers filtering, not the artifact handling our raw reader may need.
        s.lower_threshold = self.threshold.value()
        s.remove_outliers = self.outlier_check.isChecked()
        s.outlier_low_pct = self.olo.value()
        s.outlier_high_pct = self.ohi.value()
        s.moderate_spikes = self.spike_check.isChecked()
        s.spike_nsigma = self.spike_nsigma.value()
        s.spike_strength = self.spike_strength.value()
        s.bridge_gaps = self.gap_check.isChecked()
        s.min_gap_s = self.gap_seconds.value()
        s.resample_factor = self.resample.value()
        return s

    @staticmethod
    def _stages_active(s: PreprocessSettings) -> bool:
        """True if ``s`` would change the signal at all."""
        return bool(s.smooth_width_s > 0 or s.apply_filter or s.lower_threshold > 0
                    or s.remove_outliers or s.moderate_spikes or s.bridge_gaps
                    or s.resample_factor > 1)

    @staticmethod
    def _sanitize_active(s: PreprocessSettings) -> bool:
        """True if any *sanitisation* stage runs — conditioning excluded.

        Sanitisation and conditioning are different jobs and the screen has to
        tell them apart. Ticking "remove DAQ artifact spikes" used to flip the
        one combined flag, which re-derived the red trace from raw — carrying
        whatever the smoothing controls happened to say, which for an
        already-conditioned study was *nothing*. So asking for de-glitching
        silently threw away smo(0.1) and the red trace fell back to unsmoothed
        raw, labelled "sanitized — no smoothing".
        """
        return bool(s.lower_threshold > 0 or s.remove_outliers
                    or s.moderate_spikes or s.bridge_gaps
                    or s.resample_factor > 1)

    def _recipe_matches(self, s: PreprocessSettings, channel: str) -> bool:
        """True if ``s``'s conditioning is exactly the channel's FAMOS recipe.

        When it is, and no sanitisation is asked for, the study's stored output
        already *is* the requested result — so show that rather than recomputing
        an approximation of it (the stored one also carries the pipeline's
        dead-run blanking and decimation).
        """
        r = famos_recipe(channel)
        if s.apply_filter != r.apply_filter:
            return False
        if abs(s.smooth_width_s - r.smooth_width_s) > 1e-9:
            return False
        return not r.apply_filter or (
            abs(s.filter_cutoff - r.filter_cutoff) < 1e-9
            and s.filter_order == r.filter_order)

    def _proc_label(self, s: PreprocessSettings, ch: str,
                    have_raw: bool, sanitizing: bool) -> str:
        """Legend text for the red trace — every step that actually produced it.

        Built from what ran, not from which checkbox is ticked, so the three
        cases stay distinguishable: the FAMOS recipe out of the imc config file,
        a manual override the operator dialled in, and sanitisation stages on
        top of either. Naming a manual smo "FAMOS", or dropping the de-glitch
        from the label, is how a plot starts lying about what it shows.
        """
        op = _famos_op(ch)
        if have_raw or not self._conditioned:
            # The conditioning on screen is whatever `s` specifies.
            if self._recipe_matches(s, ch):
                cond = "" if op == "passthrough" else f"FAMOS {op}"
            else:
                manual = []
                if s.apply_filter:
                    manual.append(f"FiltLP({s.filter_order}, {s.filter_cutoff:g} Hz)")
                if s.smooth_width_s > 0:
                    manual.append(f"smo({s.smooth_width_s:g} s)")
                cond = " → ".join(manual)
        else:
            # No stored raw, but ingestion ran the recipe: that is what the
            # trace carries, whatever the (neutralised) controls now say.
            cond = "" if op == "passthrough" else f"FAMOS {op}"

        steps = []
        if s.lower_threshold > 0:
            steps.append(f"threshold {s.lower_threshold:g}")
        if s.remove_outliers:
            steps.append(f"outliers {s.outlier_low_pct:g}–{s.outlier_high_pct:g}%")
        if s.moderate_spikes:
            steps.append("de-glitch")
        if s.bridge_gaps:
            steps.append("gap bridge")
        if s.resample_factor > 1:
            steps.append(f"red({s.resample_factor})")
        if cond:
            steps.append(cond)

        if not steps:
            return ("unchanged — FAMOS passthrough" if op == "passthrough"
                    else "sanitized — no conditioning")
        return "sanitized — " + " → ".join(steps)

    def _update(self, *_) -> None:
        if self._data.size == 0:
            return
        s = self._settings()
        ch = self.channel_combo.currentText()
        have_raw = self._raw is not None and self._raw.size > 0
        sanitizing = self._sanitize_active(s)

        if have_raw:
            # Blue is the real unconditioned channel. Red is the conditioned
            # one — the pipeline's stored output when that already is exactly
            # what the controls ask for, otherwise the same recipe re-run on
            # the raw with the extra sanitisation folded in. Either way the
            # conditioning survives, which is the whole point of the pair.
            raw, t = self._raw, self._raw_time
            if self._conditioned and not sanitizing and self._recipe_matches(s, ch):
                t_proc, proc, new_fs = self._time, self._data, self._fs
            else:
                base = raw
                if is_wft_channel(ch):
                    # The pipeline blanks frozen-value dropouts before it
                    # conditions anything (a dead WFT keeps streaming a hard
                    # constant, which is the absence of a measurement, not a
                    # reading of zero). A preview that skips that step smooths
                    # the dead run into the trace and disagrees with the
                    # study's own output — visible as the red line diving to
                    # the frozen value at t=0 while the stored result does not.
                    base = blank_dead_runs(raw, self._fs, 1.0)
                t_proc, proc, new_fs = apply_pipeline(t, base, self._fs, s)
        else:
            # No stored raw: `self._data` is all there is, and on a conditioned
            # study it has already been through the recipe. Running the recipe
            # again here would smooth twice and quietly narrow the trace, so
            # only the sanitisation the operator explicitly asked for may run.
            raw = self._data
            t = self._time
            if self._conditioned:
                s.smooth_width_s = 0.0
                s.apply_filter = False
            t_proc, proc, new_fs = (apply_pipeline(t, raw, self._fs, s)
                                    if self._stages_active(s) else (t, raw, self._fs))
        active = self._stages_active(s)

        ax = self.canvas.ax
        self.canvas.clear()
        npts = _plot_points(self.canvas)
        src = "raw" if have_raw else ("study data" if self._conditioned else "raw")

        # The FAMOS view: the raw as its full min→max reach in blue, and the
        # conditioned channel as a single median centre line in red over it.
        # The raw's spikes need no marker of their own — they *are* the blue
        # excursions standing outside the red, which is how FAMOS shows what
        # the conditioning took off.
        yr = ymark = None
        if not active and not have_raw:
            # Without a stored raw, "before" and "after" are the same array.
            # Drawing it twice in two colours is not an empty comparison, it is
            # a misleading one. One trace, named for what it is.
            tp, yp = _envelope_line(t_proc, proc, npts)
            ax.plot(tp, yp, color=_RAW_COLOR, linewidth=0.7, label=src)
        else:
            if self.show_raw.isChecked():
                tr, yr = _envelope_line(t, raw, npts)
                # Underneath and slightly held back, so the red stays readable
                # through the densest stretches of the band.
                ax.plot(tr, yr, color=_RAW_COLOR, linewidth=0.6, alpha=0.55,
                        zorder=1, label=src)
            tp, yp = _median_line(t_proc, proc, npts)
            # Red on top, opaque and a touch heavier: it is one thin line now,
            # not a second band, so it no longer needs transparency to keep the
            # blue visible — and transparency would only wash it out.
            ax.plot(tp, yp, color=_PROC_COLOR, linewidth=1.0, zorder=3,
                    solid_joinstyle="round", solid_capstyle="round",
                    label=self._proc_label(s, ch, have_raw, sanitizing))
            # Optional, off by default: ring the removed excursions. FAMOS does
            # not do this, so it stays opt-in for when the blue-vs-red reading is
            # too dense to pick them out by eye.
            if self.mark_spikes.isChecked() and proc.size == raw.size:
                ts, ys, n_spikes = _spikes(t, raw, proc, self.spike_mark_nsigma.value())
                if ts.size:
                    ymark = ys
                    ax.plot(ts, ys, linestyle="none", marker="o", markersize=3.0,
                            markerfacecolor="none", markeredgecolor="#ffd166",
                            markeredgewidth=0.8, alpha=0.9, zorder=4,
                            label=f"raw spikes removed ({n_spikes:,})")

        ax.set_title(self._recipe_note(s, ch, have_raw, sanitizing),
                     fontsize=9, color=theme.TEXT_MUTED, loc="left")
        unit = _channel_unit(ch)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{ch} ({unit})" if unit else ch)
        if t.size:
            ax.set_xlim(float(t[0]), float(t[-1]))     # show the entire timeline
        ax.margins(x=0)

        # Frame what is actually on screen: the red centre line, widened to the
        # raw envelope when the overlay is on. Framing on the full-resolution
        # processed array instead would leave the median line floating in a band
        # of empty space, since the extremes that set those percentiles are
        # precisely what the median drops. Robust percentiles either way, so a
        # rare artifact clips off-view rather than squashing the trace flat.
        fp = yp[np.isfinite(yp)]
        if fp.size:
            lo, hi = np.percentile(fp, 0.2), np.percentile(fp, 99.8)
            if yr is not None:
                fr = yr[np.isfinite(yr)]
                if fr.size:
                    lo = min(lo, float(np.percentile(fr, 0.5)))
                    hi = max(hi, float(np.percentile(fr, 99.5)))
            if ymark is not None:
                # With the overlay off the red line is the only thing setting the
                # scale, and the rings sit by definition outside it — every one
                # of them would clip off-view. Same robust percentiles, so one
                # freak artifact still cannot flatten the trace.
                fm = ymark[np.isfinite(ymark)]
                if fm.size:
                    lo = min(lo, float(np.percentile(fm, 1.0)))
                    hi = max(hi, float(np.percentile(fm, 99.0)))
            pad = 0.15 * (hi - lo) if hi > lo else 1.0
            ax.set_ylim(lo - pad, hi + pad)
        ax.legend(fontsize=8, facecolor=theme.SURFACE, labelcolor=theme.TEXT, framealpha=0.9)
        self.canvas.fig.tight_layout()
        self.canvas.draw()

        if t.size:
            note = ("  ·  already FAMOS-conditioned at ingestion"
                    if self._conditioned else "")
            self.range_label.setText(
                f"Full recording: 0 – {t[-1]:.0f} s  ·  {raw.size:,} samples"
                f"{note}  ·  use the toolbar to zoom/pan.")

        a, b = summary_stats(raw), summary_stats(proc)
        self.stats_label.setText(
            f"{self._recipe_note(s, ch, have_raw, sanitizing)}\n"
            f"fs: {self._fs:.1f} → {new_fs:.1f} Hz\n"
            f"samples: {a['n']} → {b['n']}  (removed {b['removed']})\n"
            f"mean: {a['mean']:.1f} → {b['mean']:.1f} {unit}\n"
            f"std:  {a['std']:.1f} → {b['std']:.1f} {unit}\n"
            f"min/max: {b['min']:.0f} / {b['max']:.0f} {unit}")


def _title(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight:700; font-size:12px; margin-top:6px;")
    return lbl
