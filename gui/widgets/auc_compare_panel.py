"""The two-dataset AUC comparison, as one widget.

Used by two screens that differ only in where their data comes from:

* **Compare studies** — two finished study folders (``processed_data.csv``
  plus provenance, RMS/DLC/G-severity deltas);
* **AUC Compare** — two processed CSV files picked directly, no study folder.

Everything between "here are two frames" and "here is the comparison" lives
here, once: the mode / channel / weighting / x-axis selectors, the KDE and
histogram panels with the stats strip, the footer and coverage lines, the
all-channels chart, and PNG / batch export. The drawing and statistics
underneath are ``dtt.analysis.study_compare`` and ``dtt.analysis.auc`` --
this widget only drives them. Two copies of this screen would drift the way
three copies of ``smo`` once did.

The owning page supplies the channel lists (it decides how to match
channels), a ``labels()`` callable (it owns the name fields), and optionally a
``unit_fn`` (a study's forces are daN by pipeline guarantee; a bare CSV's
units are unknown). It listens to ``drawn`` and ``channels_changed`` to keep
its own KPI cards current.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel,
    QMessageBox, QProgressDialog, QPushButton, QVBoxLayout, QWidget,
)

from gui import theme
from gui.widgets.common import Card
from gui.widgets.mpl_canvas import PlotPanel


def _mini(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight:700; font-size:13px;")
    return lbl


class AucComparePanel(QWidget):
    """Controls, panels, strip, footer, overview chart and export."""

    drawn = Signal(object, str)            # (AucComparison or None, unit)
    channels_changed = Signal(int, str)    # (channel count, mode)

    def __init__(self, labels: Callable[[], tuple],
                 start_dir: Callable[[], str],
                 unit_fn: Optional[Callable[[str], str]] = None,
                 noun: str = "study", parent=None):
        super().__init__(parent)
        from dtt.analysis.study_compare import channel_unit

        self._labels = labels
        self._start_dir = start_dir
        self._unit_fn = unit_fn or channel_unit
        self._noun = noun
        self._df_a = self._df_b = None
        self._same: List[tuple] = []
        self._cross: List[tuple] = []
        self._channels: List[tuple] = []
        self._dist = (None, None, None, None)      # (w_a, m_a, w_b, m_b)
        self._dist_reason = ""
        self._summary_cache: dict = {}

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(18)

        # ---- the AUC pair ---------------------------------------------------
        auc_card = Card()
        head = QHBoxLayout()
        head.addWidget(_mini("Area under curve — distribution comparison"))
        head.addStretch(1)
        self.save_btn = QPushButton("Save PNG…")
        self.save_btn.setObjectName("Secondary")
        self.save_btn.clicked.connect(self._save_png)
        head.addWidget(self.save_btn)
        self.batch_btn = QPushButton("Export all…")
        self.batch_btn.setObjectName("Secondary")
        self.batch_btn.setToolTip(
            "A PNG for every channel in the current mode, one summary CSV with "
            "the labels in the column names, and the all-channels chart.")
        self.batch_btn.clicked.connect(self._export_batch)
        head.addWidget(self.batch_btn)
        auc_card.layout().addLayout(head)

        # Selectors as a 2x2 grid, actions up beside the title. All six in one
        # row came to ~1,900 px of minimum width inside a ScrollPage with
        # horizontal scrolling off -- on a laptop the row was simply cut.
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Same channel", "same")
        self.mode_combo.addItem("Cross-axle", "cross")
        self.mode_combo.setToolTip(
            "Same channel: each channel against the same channel in the other "
            "dataset.\n\n"
            "Cross-axle: a front wheel in one dataset against the rear wheel on "
            "the same side in the other, both ways round -- whether one axle's "
            "load in one vehicle is what the other axle sees in the other, the "
            "case when a powertrain change shifts weight.")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self.channel_combo = QComboBox()
        self.channel_combo.setMinimumWidth(150)
        self.channel_combo.currentTextChanged.connect(lambda *_: self.draw())

        self.weight_combo = QComboBox()
        self.weight_combo.addItem("Samples", "count")
        self.weight_combo.addItem("Distance", "distance")
        self.weight_combo.setToolTip(
            "Samples: every sample counts once.\n\n"
            "Distance: every sample counts by how far the vehicle travelled "
            "while it was recorded, so a run that idled longer does not have "
            "its distribution pulled toward the idle load. Uses the same "
            "speed-channel search as the Histogram and AUC screens, and needs a "
            "usable speed channel in BOTH datasets -- weighting one and not the "
            "other would compare two different things.")
        self.weight_combo.currentIndexChanged.connect(self._on_weighting_changed)

        self.range_combo = QComboBox()
        self.range_combo.addItem("Autoscale", "autoscale")
        self.range_combo.addItem("Full range", "full")
        self.range_combo.setToolTip(
            "Autoscale fits the axis to where the data sits (P0.5-P99.5), so "
            "small differences between the two runs are visible.\n\n"
            "Full uses the configured sensor range for the component (Fx "
            "±300 daN, for example), so every dataset is drawn on the same axis "
            "and plots can be laid side by side. If fewer than 98% of samples "
            "fall inside it, it falls back to the data's own range.\n\n"
            "Neither mode removes a stuck-sensor rail holding more than about "
            "2% of samples -- that needs the data cleaned, not the axis moved.")
        self.range_combo.currentIndexChanged.connect(lambda *_: self.draw())

        ctl = QGridLayout()
        ctl.setHorizontalSpacing(10)
        ctl.setVerticalSpacing(8)
        for row, pairs in enumerate(((("Compare:", self.mode_combo),
                                      ("Channel:", self.channel_combo)),
                                     (("Weight by:", self.weight_combo),
                                      ("X-axis:", self.range_combo)))):
            for col, (text, combo) in enumerate(pairs):
                ctl.addWidget(QLabel(text), row, col * 2)
                ctl.addWidget(combo, row, col * 2 + 1)
        ctl.setColumnStretch(4, 1)
        auc_card.layout().addLayout(ctl)

        # Two panels over a stats strip spanning both, from the same
        # auc_figure_layout the PNG export uses -- screen and file cannot
        # drift apart.
        from dtt.analysis.study_compare import auc_figure_layout
        self.auc = PlotPanel(height=6.6, ncols=2, light=True)
        fig = self.auc.canvas.fig
        for ax in list(fig.axes):
            ax.remove()
        kde, hist, stats = auc_figure_layout(fig)
        self.auc.canvas.axes = [kde, hist, stats]
        self.auc.canvas.ax = kde
        auc_card.layout().addWidget(self.auc)

        self.footer = QLabel("")
        self.footer.setWordWrap(True)
        self.footer.setStyleSheet(
            "font-family:monospace; font-size:11px; padding:8px; "
            f"background:{theme.SURFACE_2}; color:{theme.TEXT};")
        auc_card.layout().addWidget(self.footer)
        # How much data each percentile came from, and how it was counted --
        # context, kept apart from the percentile line so it does not read as
        # another result.
        self.coverage = QLabel("")
        self.coverage.setWordWrap(True)
        self.coverage.setStyleSheet(
            "font-family:monospace; font-size:11px; padding:4px 8px; "
            f"color:{theme.TEXT_MUTED};")
        auc_card.layout().addWidget(self.coverage)
        lay.addWidget(auc_card)

        # ---- every channel at once -----------------------------------------
        sum_card = Card()
        sum_card.layout().addWidget(_mini("All channels — at a glance"))
        sum_note = QLabel(
            "Share of the second dataset beyond the reference's P95, per "
            "channel. 5% is the no-change line by definition of a 95th "
            "percentile, so this is comparable across channels whose units and "
            "sizes are not. ΔP95 is printed at each bar in the channel's own "
            "unit.")
        sum_note.setWordWrap(True)
        sum_note.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:11px;")
        sum_card.layout().addWidget(sum_note)
        self.summary_plot = PlotPanel(height=4.0, light=True)
        sum_card.layout().addWidget(self.summary_plot)
        lay.addWidget(sum_card)

        self._set_enabled(False)

    # ------------------------------------------------------------------ data

    def set_data(self, df_a: pd.DataFrame, df_b: pd.DataFrame,
                 same: List[tuple], cross: List[tuple]) -> str:
        """Load a new pair and draw it. Returns why distance weighting is
        unavailable, or "" when it is available -- for the page's notes bar.
        """
        from dtt.analysis.study_compare import distance_weights

        self._df_a, self._df_b = df_a, df_b
        self._same, self._cross = list(same), list(cross)
        self._summary_cache = {}

        # Once per pair, not per redraw: the speed search reads whole columns.
        # Kept whatever the selector says, so switching to Distance is free.
        wa, ma = distance_weights(df_a)
        wb, mb = distance_weights(df_b)
        self._dist = (wa, ma, wb, mb)
        a, b = self._labels()
        missing = [n for n, w in ((a, wa), (b, wb)) if w is None]
        self._dist_reason = (
            "" if not missing else
            f"no usable speed channel in {' or '.join(missing)} "
            f"(every candidate is missing, dead or all zero)")
        item = self.weight_combo.model().item(1)
        if item is not None:
            item.setEnabled(not missing)
            item.setToolTip(f"Unavailable: {self._dist_reason}" if missing else "")
        if missing and self.weight_combo.currentData() == "distance":
            self.weight_combo.blockSignals(True)
            self.weight_combo.setCurrentIndex(0)
            self.weight_combo.blockSignals(False)

        self._fill_channel_combo()
        self._set_enabled(bool(self._channels))
        self.draw()
        self.draw_summary()
        return self._dist_reason

    def clear(self) -> None:
        self._df_a = self._df_b = None
        self._same, self._cross, self._channels = [], [], []
        self._summary_cache = {}
        self.channel_combo.clear()
        self.auc.clear()
        self.auc.draw()
        self.summary_plot.clear()
        self.summary_plot.draw()
        self.footer.setText("")
        self.coverage.setText("")
        self._set_enabled(False)
        self.drawn.emit(None, "")

    def relabel(self) -> None:
        """Redraw under the current names. Recomputes nothing: the summary
        rows are cached independently of the labels, and a name cannot reach
        the percentile or exceedance arithmetic."""
        if self._df_a is None:
            return
        self.draw()
        self.draw_summary()

    @property
    def dist_reason(self) -> str:
        return self._dist_reason

    @property
    def channels(self) -> List[tuple]:
        return self._channels

    def _set_enabled(self, on: bool) -> None:
        self.save_btn.setEnabled(on)
        self.batch_btn.setEnabled(on)

    # ------------------------------------------------------ mode / weighting

    def mode(self) -> str:
        return self.mode_combo.currentData() or "same"

    def _fill_channel_combo(self) -> None:
        """Offer the channels for the current mode, keeping the selection when
        the same entry still exists."""
        self._channels = self._cross if self.mode() == "cross" else self._same
        prev = self.channel_combo.currentText()
        names = [c[0] for c in self._channels]
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        self.channel_combo.addItems(names)
        if prev in names:
            self.channel_combo.setCurrentText(prev)
        self.channel_combo.blockSignals(False)
        self.channels_changed.emit(len(self._channels), self.mode())
        if self.mode() == "cross" and not self._channels:
            self.footer.setText(
                "No cross-axle pairs: cross-axle needs a front and a rear wheel "
                f"on the same side in both {self._noun}s.")

    def _on_mode_changed(self, *_) -> None:
        if self._df_a is None:
            return
        self._fill_channel_combo()
        self._set_enabled(bool(self._channels))
        self.draw()
        self.draw_summary()

    def _on_weighting_changed(self, *_) -> None:
        if self._df_a is None:
            return
        self.draw()
        self.draw_summary()

    def weights(self) -> tuple:
        """``(weights_a, weights_b)`` for the current setting, or two Nones.

        Both or neither, always: ``compare_distributions`` refuses a mixed
        pair, and the selector disables Distance when either side has no speed.
        """
        wa, _, wb, _ = self._dist
        if (self.weight_combo.currentData() == "distance"
                and wa is not None and wb is not None):
            return wa, wb
        return None, None

    def weighting_name(self) -> str:
        return "distance" if self.weights()[0] is not None else "sample count"

    # -------------------------------------------------------------- drawing

    def selected(self):
        """``(display, values_a, values_b)`` for the channel on screen."""
        label = self.channel_combo.currentText()
        for disp, ca, cb in self._channels:
            if disp == label:
                va = pd.to_numeric(self._df_a[ca], errors="coerce").to_numpy(float)
                vb = pd.to_numeric(self._df_b[cb], errors="coerce").to_numpy(float)
                return disp, va, vb
        return None, None, None

    def draw(self) -> None:
        if self._df_a is None or not self._channels:
            return
        from dtt.analysis.study_compare import (
            auc_coverage_text, auc_footer_text, draw_auc_comparison,
            study_duration_s)
        from dtt.analysis.histograms import SUPTITLE_FONTSIZE, TEXT_PRI

        disp, va, vb = self.selected()
        if disp is None:
            return
        a, b = self._labels()
        unit = self._unit_fn(disp)
        wa, wb = self.weights()
        ax_kde, ax_hist, ax_stats = self.auc.axes
        self.auc.clear()
        cmp = draw_auc_comparison(
            ax_kde, ax_hist, va, vb, a, b, disp, unit,
            range_mode=self.range_combo.currentData() or "autoscale",
            ax_stats=ax_stats, weights_a=wa, weights_b=wb)
        self.auc.canvas.fig.suptitle(
            f"AUC — {disp}  ({a}  vs  {b}  reference)",
            color=TEXT_PRI, fontsize=SUPTITLE_FONTSIZE, fontweight="bold")
        # No tight_layout: auc_figure_layout's gridspec already places the
        # panels and the strip, and tight_layout would undo it.
        self.auc.draw()

        if cmp is None:
            self.footer.setText("Insufficient data on this channel.")
            self.coverage.setText("")
            self.drawn.emit(None, unit)
            return
        _, ma, _, mb = self._dist
        weighted = wa is not None
        self.footer.setText(auc_footer_text(cmp, a, b, unit))
        self.coverage.setText(auc_coverage_text(
            cmp, a, b,
            study_duration_s(self._df_a), study_duration_s(self._df_b),
            ma if weighted else None, mb if weighted else None,
            weighting=self.weighting_name()))
        self.drawn.emit(cmp, unit)

    def _summary_rows(self):
        """Per-channel comparisons for the current mode and weighting, cached
        on (mode, weighting) -- the rows do not depend on the labels."""
        from dtt.analysis.study_compare import auc_summary

        key = (self.mode(), self.weighting_name())
        if key not in self._summary_cache:
            wa, wb = self.weights()
            self._summary_cache[key] = auc_summary(
                self._channels, self._df_a, self._df_b, wa, wb,
                unit_fn=self._unit_fn)
        return self._summary_cache[key]

    def draw_summary(self) -> None:
        if self._df_a is None:
            return
        from dtt.analysis.study_compare import draw_auc_summary

        rows = self._summary_rows()
        a, b = self._labels()
        # Grows with the channel count so every bar keeps a readable label.
        self.summary_plot.setMinimumHeight(170 + 26 * max(1, len(rows)))
        self.summary_plot.clear()
        draw_auc_summary(self.summary_plot.ax, rows, a, b)
        self.summary_plot.canvas.fig.tight_layout()
        self.summary_plot.draw()

    # --------------------------------------------------------------- export

    def _save_png(self) -> None:
        from dtt.analysis.study_compare import generate_auc_comparison

        disp, va, vb = self.selected()
        if disp is None:
            return
        a, b = self._labels()
        suggested = "".join(c if c.isalnum() or c in "-_." else "_"
                            for c in f"auc_{disp}_{a}_vs_{b}.png")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the AUC comparison",
            str(Path(self._start_dir()) / suggested), "PNG image (*.png)")
        if not path:
            return
        wa, wb = self.weights()
        try:
            generate_auc_comparison(
                va, vb, a, b, disp, Path(path), self._unit_fn(disp),
                range_mode=self.range_combo.currentData() or "autoscale",
                weights_a=wa, weights_b=wb,
                footer_extra=f"weighted by {self.weighting_name()}")
            QMessageBox.information(self, "Saved", f"Written:\n{path}")
        except Exception as exc:                                   # noqa: BLE001
            QMessageBox.critical(self, "Save failed", f"{type(exc).__name__}: {exc}")

    def _export_batch(self) -> None:
        """Every channel in the current mode, in one go."""
        from dtt.analysis.study_compare import export_auc_batch

        if not self._channels:
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Choose a folder for the AUC export", self._start_dir())
        if not folder:
            return
        a, b = self._labels()
        wa, wb = self.weights()
        dlg = QProgressDialog("Rendering…", "Cancel", 0, len(self._channels), self)
        dlg.setWindowTitle("Export all channels")
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(True)

        class _Cancelled(Exception):
            pass

        def progress(i, n, name):
            dlg.setValue(i)
            dlg.setLabelText(f"{name}  ({min(i + 1, n)} of {n})")
            QApplication.processEvents()
            if dlg.wasCanceled():
                raise _Cancelled

        try:
            pngs, csv_path, summary_png = export_auc_batch(
                self._channels, self._df_a, self._df_b, a, b, Path(folder),
                range_mode=self.range_combo.currentData() or "autoscale",
                weights_a=wa, weights_b=wb, progress=progress,
                unit_fn=self._unit_fn)
        except _Cancelled:
            QMessageBox.information(self, "Export cancelled",
                                    "Stopped. Files already written were kept.")
            return
        except Exception as exc:                                   # noqa: BLE001
            QMessageBox.critical(self, "Export failed",
                                 f"{type(exc).__name__}: {exc}")
            return
        finally:
            dlg.close()
        QMessageBox.information(
            self, "Export complete",
            f"{len(pngs)} channel figures, the summary CSV and the all-channels "
            f"chart written to:\n{folder}\n\n{csv_path.name}\n{summary_png.name}")

    def auc_rows(self) -> list:
        """Export rows for every channel in the current mode and weighting --
        the same numbers the summary CSV of a batch export carries."""
        from dtt.analysis.study_compare import auc_comparison_stats

        a, b = self._labels()
        wa, wb = self.weights()
        rows = []
        for disp, ca, cb in self._channels:
            va = pd.to_numeric(self._df_a[ca], errors="coerce").to_numpy(float)
            vb = pd.to_numeric(self._df_b[cb], errors="coerce").to_numpy(float)
            row = auc_comparison_stats(va, vb, a, b, disp, self._unit_fn(disp),
                                       wa, wb)
            if row:
                rows.append(row)
        return rows
