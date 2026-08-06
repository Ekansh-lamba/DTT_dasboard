"""Screen — Compare: two raw recordings side by side.

Pick a previous and a current imc raw folder; the page matches their channels,
merges them onto a shared elapsed-time axis, and reports what changed. Loading
runs on a worker thread because a 70-minute recording takes ~20 s to read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
    QProgressBar, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from gui import theme
from gui.pages.base_page import BasePage
from gui.widgets.common import Card, KpiCard, ScrollPage, SectionTitle
from gui.widgets.mpl_canvas import PlotPanel

from dtt.comparison import ComparisonResult, align_channel

_PREV_COLOR = "#4a9eff"
_CURR_COLOR = "#ff4d4f"


class _CompareWorker(QThread):
    """Loads both recordings and compares them off the UI thread."""
    progress = Signal(str)
    done = Signal(object, object, object, object, object, object)
    # (result, prev_df, curr_df, fs_prev, fs_curr, error)

    def __init__(self, prev_folder: Path, curr_folder: Path, deglitch: bool = False):
        super().__init__()
        self.prev_folder = Path(prev_folder)
        self.curr_folder = Path(curr_folder)
        self.deglitch = deglitch

    def run(self) -> None:
        try:
            from dtt.comparison import compare_frames
            from dtt.comparison_io import load_recording

            self.progress.emit(f"Reading previous — {self.prev_folder.name} …")
            prev, fs_p, _ = load_recording(self.prev_folder, deglitch=self.deglitch)
            self.progress.emit(f"Reading current — {self.curr_folder.name} …")
            curr, fs_c, _ = load_recording(self.curr_folder, deglitch=self.deglitch)
            if prev.empty or curr.empty:
                raise ValueError("One of the folders contained no readable channels.")
            self.progress.emit("Matching channels and comparing …")
            result = compare_frames(prev, curr, self.prev_folder.name,
                                    self.curr_folder.name, fs_p, fs_c)
            self.done.emit(result, prev, curr, fs_p, fs_c, None)
        except Exception as exc:                                  # noqa: BLE001
            self.done.emit(None, None, None, None, None, f"{type(exc).__name__}: {exc}")


class ComparisonPage(BasePage):
    def __init__(self, repo, parent=None):
        super().__init__(repo, parent)

        self._result: Optional[ComparisonResult] = None
        self._prev: Optional[pd.DataFrame] = None
        self._curr: Optional[pd.DataFrame] = None
        self._fs_prev = 0.0
        self._fs_curr = 0.0
        self._prev_folder: Optional[Path] = None
        self._curr_folder: Optional[Path] = None
        self._worker: Optional[_CompareWorker] = None

        page = ScrollPage()
        body = page.body()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(page)

        body.addWidget(SectionTitle("Compare — Previous vs Current Recording"))
        intro = QLabel(
            "Select two imc raw folders from different sessions. Channels are "
            "matched by wheel position and component, so different naming "
            "conventions (WFT_Fx_fr vs FR_Fx_2) still line up.")
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:12px;")
        body.addWidget(intro)

        # ---- folder pickers -------------------------------------------------
        pick = Card()
        row = QHBoxLayout()
        self.prev_label = QLabel("No folder selected")
        self.curr_label = QLabel("No folder selected")
        for title, lbl, slot in (("Previous", self.prev_label, self._pick_prev),
                                 ("Current", self.curr_label, self._pick_curr)):
            col = QVBoxLayout()
            head = QLabel(title)
            head.setStyleSheet("font-weight:700; font-size:12px;")
            btn = QPushButton(f"Choose {title.lower()} raw folder…")
            btn.setObjectName("Secondary")
            btn.clicked.connect(slot)
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:11px;")
            col.addWidget(head)
            col.addWidget(btn)
            col.addWidget(lbl)
            row.addLayout(col, 1)
        pick.layout().addLayout(row)

        run_row = QHBoxLayout()
        self.compare_btn = QPushButton("⇄  Compare")
        self.compare_btn.setObjectName("Primary")
        self.compare_btn.setCursor(Qt.PointingHandCursor)
        self.compare_btn.setEnabled(False)
        self.compare_btn.clicked.connect(self._run_compare)
        run_row.addWidget(self.compare_btn)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        run_row.addWidget(self.progress, 1)
        self.status = QLabel("")
        self.status.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:11px;")
        run_row.addWidget(self.status, 2)
        pick.layout().addLayout(run_row)
        body.addWidget(pick)

        # ---- KPI row --------------------------------------------------------
        kpi = QHBoxLayout()
        self.kpi_matched = KpiCard("Matched", "—", "channels in both")
        self.kpi_changed = KpiCard("Changed", "—", "beyond 5% tolerance")
        self.kpi_added = KpiCard("Added", "—", "current only")
        self.kpi_removed = KpiCard("Removed", "—", "previous only")
        for k in (self.kpi_matched, self.kpi_changed, self.kpi_added, self.kpi_removed):
            kpi.addWidget(k, 1)
        body.addLayout(kpi)

        # ---- notes ----------------------------------------------------------
        self.notes = QLabel("")
        self.notes.setWordWrap(True)
        self.notes.setVisible(False)
        self.notes.setStyleSheet(
            f"color:{theme.TEXT}; background:{theme.SURFACE_2}; "
            f"border-left:3px solid {theme.ORANGE}; padding:10px; font-size:11.5px;")
        body.addWidget(self.notes)

        # ---- table ----------------------------------------------------------
        table_card = Card()
        table_card.layout().addWidget(_mini("Per-channel comparison"))
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            ["Channel", "Status", "Prev mean", "Curr mean", "Δ mean", "% mean",
             "Prev RMS", "Curr RMS", "% RMS"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(260)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        table_card.layout().addWidget(self.table)
        body.addWidget(table_card)

        # ---- plots ----------------------------------------------------------
        plot_card = Card()
        head = QHBoxLayout()
        head.addWidget(_mini("Channel detail"))
        head.addStretch(1)
        head.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        self.channel_combo.setMinimumWidth(140)
        self.channel_combo.currentTextChanged.connect(self._draw_channel)
        head.addWidget(self.channel_combo)
        plot_card.layout().addLayout(head)
        self.overlay = PlotPanel(height=3.0)
        plot_card.layout().addWidget(self.overlay)
        self.dist = PlotPanel(height=2.4)
        plot_card.layout().addWidget(self.dist)
        body.addWidget(plot_card)

        summary_card = Card()
        summary_card.layout().addWidget(_mini("Change by channel (RMS)"))
        self.delta_plot = PlotPanel(height=3.0)
        summary_card.layout().addWidget(self.delta_plot)
        body.addWidget(summary_card)

        # ---- exports --------------------------------------------------------
        exp = QHBoxLayout()
        exp.addWidget(QLabel("Export:"))
        for text, kind in (("CSV", "csv"), ("Excel", "excel"),
                           ("PDF report", "pdf"), ("Merged data (CSV)", "merged")):
            b = QPushButton(text)
            b.setObjectName("Secondary")
            b.clicked.connect(lambda _=False, k=kind: self._export(k))
            exp.addWidget(b)
        exp.addStretch(1)
        self.export_all_btn = QPushButton("Export all to folder…")
        self.export_all_btn.setObjectName("Primary")
        self.export_all_btn.clicked.connect(lambda: self._export("all"))
        exp.addWidget(self.export_all_btn)
        body.addLayout(exp)
        body.addStretch(1)
        self._set_exports_enabled(False)

    # ------------------------------------------------------------------ picking

    def _start_dir(self) -> str:
        for p in (self._prev_folder, self._curr_folder):
            if p:
                return str(p.parent)
        return str(getattr(self.repo, "project_dir", Path.cwd()))

    def _pick(self, title: str) -> Optional[Path]:
        path = QFileDialog.getExistingDirectory(self, title, self._start_dir())
        if not path:
            return None
        from dtt.ingestion.imc_reader import resolve_raw_folder
        folder = resolve_raw_folder(Path(path))
        if not list(folder.glob("*.raw")):
            QMessageBox.warning(
                self, "No raw channels",
                f"No .raw files found in:\n{path}\n\n"
                "Pick the imc export folder (or its parent).")
            return None
        return folder

    def _describe(self, folder: Path) -> str:
        n = len(list(folder.glob("*.raw")))
        return f"{folder.name}\n{n} channels · {folder.parent.name}"

    def _pick_prev(self) -> None:
        f = self._pick("Select the PREVIOUS recording folder")
        if f:
            self._prev_folder = f
            self.prev_label.setText(self._describe(f))
            self._refresh_ready()

    def _pick_curr(self) -> None:
        f = self._pick("Select the CURRENT recording folder")
        if f:
            self._curr_folder = f
            self.curr_label.setText(self._describe(f))
            self._refresh_ready()

    def _refresh_ready(self) -> None:
        ready = bool(self._prev_folder and self._curr_folder)
        self.compare_btn.setEnabled(ready)
        if ready and self._prev_folder == self._curr_folder:
            self.status.setText("Both folders are the same — pick two different sessions.")
            self.compare_btn.setEnabled(False)
        elif ready:
            self.status.setText("Ready to compare.")

    # ----------------------------------------------------------------- running

    def _run_compare(self) -> None:
        if not (self._prev_folder and self._curr_folder):
            return
        self.compare_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.status.setText("Loading…")
        self._worker = _CompareWorker(self._prev_folder, self._curr_folder)
        self._worker.progress.connect(self.status.setText)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _on_done(self, result, prev, curr, fs_p, fs_c, error) -> None:
        self.progress.setVisible(False)
        self.compare_btn.setEnabled(True)
        if error:
            self.status.setText("Failed.")
            QMessageBox.critical(self, "Comparison failed", error)
            return
        self._result, self._prev, self._curr = result, prev, curr
        self._fs_prev, self._fs_curr = fs_p, fs_c
        self.status.setText(
            f"Compared {len(result.matched)} channels · "
            f"{len(result.changed)} changed.")
        self._populate(result)
        self._set_exports_enabled(True)

    # --------------------------------------------------------------- rendering

    def _populate(self, r: ComparisonResult) -> None:
        self.kpi_matched.set_value(str(len(r.matched)))
        self.kpi_changed.set_value(str(len(r.changed)))
        self.kpi_added.set_value(str(len(r.only_current)),
                                 ", ".join(r.only_current[:6]) or "none")
        self.kpi_removed.set_value(str(len(r.only_previous)),
                                   ", ".join(r.only_previous[:6]) or "none")

        if r.notes:
            self.notes.setText("• " + "\n• ".join(r.notes))
            self.notes.setVisible(True)
        else:
            self.notes.setVisible(False)

        self.table.setRowCount(0)
        for d in sorted(r.matched, key=lambda x: -x.max_pct):
            row = self.table.rowCount()
            self.table.insertRow(row)
            cells = [
                d.label,
                "CHANGED" if d.changed else "same",
                f"{d.previous['mean']:.1f}", f"{d.current['mean']:.1f}",
                f"{d.delta['mean']:+.1f}", f"{d.pct['mean']:+.1f}%",
                f"{d.previous['rms']:.1f}", f"{d.current['rms']:.1f}",
                f"{d.pct['rms']:+.1f}%",
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col > 1:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if d.changed:
                    item.setForeground(QColor(_CURR_COLOR if col == 1 else theme.TEXT))
                self.table.setItem(row, col, item)

        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        self.channel_combo.addItems([d.label for d in
                                     sorted(r.matched, key=lambda x: -x.max_pct)])
        self.channel_combo.blockSignals(False)
        self._draw_delta_summary(r)
        if self.channel_combo.count():
            self._draw_channel(self.channel_combo.currentText())

    def _delta_for(self, label: str):
        if not self._result:
            return None
        return next((d for d in self._result.matched if d.label == label), None)

    def _on_row_selected(self) -> None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return
        item = self.table.item(rows[0].row(), 0)
        if item:
            self.channel_combo.setCurrentText(item.text())

    def _draw_channel(self, label: str) -> None:
        d = self._delta_for(label)
        if d is None or self._prev is None or self._curr is None:
            return
        t, a, b = align_channel(self._prev, self._curr, d,
                                self._fs_prev, self._fs_curr, max_points=3000)
        ax = self.overlay.ax
        self.overlay.clear()
        if t.size:
            ax.plot(t, a, lw=0.7, color=_PREV_COLOR, label="previous", alpha=0.85)
            ax.plot(t, b, lw=0.7, color=_CURR_COLOR, label="current", alpha=0.85)
            ax.legend(fontsize=8, facecolor=theme.SURFACE, labelcolor=theme.TEXT,
                      framealpha=0.9)
        ax.set_xlabel("Elapsed time (s)")
        ax.set_ylabel(f"{label} (daN)")
        ax.set_title(f"{label} — {'CHANGED' if d.changed else 'within tolerance'}"
                     f"   worst: {d.max_pct_metric} {d.max_pct:+.1f}%", fontsize=9)
        self.overlay.canvas.fig.tight_layout()
        self.overlay.draw()

        ax2 = self.dist.ax
        self.dist.clear()
        fa, fb = a[np.isfinite(a)], b[np.isfinite(b)]
        if fa.size and fb.size:
            lo = min(np.percentile(fa, 0.5), np.percentile(fb, 0.5))
            hi = max(np.percentile(fa, 99.5), np.percentile(fb, 99.5))
            if hi > lo:
                bins = np.linspace(lo, hi, 60)
                ax2.hist(fa, bins=bins, alpha=0.55, color=_PREV_COLOR,
                         label="previous", density=True)
                ax2.hist(fb, bins=bins, alpha=0.55, color=_CURR_COLOR,
                         label="current", density=True)
                ax2.legend(fontsize=8, facecolor=theme.SURFACE,
                           labelcolor=theme.TEXT, framealpha=0.9)
        ax2.set_xlabel(f"{label} (daN)")
        ax2.set_ylabel("density")
        ax2.set_title("Distribution", fontsize=9)
        self.dist.canvas.fig.tight_layout()
        self.dist.draw()

    def _draw_delta_summary(self, r: ComparisonResult) -> None:
        ax = self.delta_plot.ax
        self.delta_plot.clear()
        items = sorted(r.matched, key=lambda d: -d.max_pct)
        if items:
            labels = [d.label for d in items]
            vals = [d.pct.get("rms", 0.0) if np.isfinite(d.pct.get("rms", np.nan)) else 0.0
                    for d in items]
            colors = [_CURR_COLOR if d.changed else theme.TEXT_MUTED for d in items]
            ax.barh(labels, vals, color=colors)
            ax.axvline(0, color=theme.TEXT_MUTED, lw=0.8)
            ax.invert_yaxis()
        ax.set_xlabel("Change in RMS, current vs previous (%)")
        ax.set_title("Per-channel load change", fontsize=9)
        self.delta_plot.canvas.fig.tight_layout()
        self.delta_plot.draw()

    # ----------------------------------------------------------------- export

    def _set_exports_enabled(self, on: bool) -> None:
        self.export_all_btn.setEnabled(on)

    def _export(self, kind: str) -> None:
        if not self._result:
            QMessageBox.information(self, "Nothing to export",
                                    "Run a comparison first.")
            return
        from dtt.comparison_io import (export_all, export_csv, export_excel,
                                       export_pdf)
        from dtt.comparison import merge_channel_frame

        stem = f"comparison_{self._prev_folder.name}_vs_{self._curr_folder.name}"
        stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:80]
        try:
            if kind == "all":
                folder = QFileDialog.getExistingDirectory(
                    self, "Choose an export folder", self._start_dir())
                if not folder:
                    return
                written = export_all(self._result, Path(folder), self._prev,
                                     self._curr, self._fs_prev, self._fs_curr,
                                     stem=stem)
                QMessageBox.information(
                    self, "Export complete",
                    "Written:\n" + "\n".join(str(p) for p in written.values()))
                return

            filt = {"csv": ("CSV (*.csv)", ".csv"),
                    "excel": ("Excel workbook (*.xlsx)", ".xlsx"),
                    "pdf": ("PDF report (*.pdf)", ".pdf"),
                    "merged": ("CSV (*.csv)", ".csv")}[kind]
            suggested = stem + ("_merged" if kind == "merged" else "") + filt[1]
            path, _ = QFileDialog.getSaveFileName(
                self, "Export comparison",
                str(Path(self._start_dir()) / suggested), filt[0])
            if not path:
                return
            path = Path(path)
            if kind == "csv":
                export_csv(self._result, path)
            elif kind == "excel":
                merged = merge_channel_frame(self._prev, self._curr, self._result,
                                             self._fs_prev, self._fs_curr)
                export_excel(self._result, path, merged)
            elif kind == "pdf":
                export_pdf(self._result, path, self._prev, self._curr,
                           self._fs_prev, self._fs_curr)
            else:
                merged = merge_channel_frame(self._prev, self._curr, self._result,
                                             self._fs_prev, self._fs_curr)
                merged.to_csv(path, index=False)
            QMessageBox.information(self, "Export complete", f"Written:\n{path}")
        except Exception as exc:                                   # noqa: BLE001
            QMessageBox.critical(self, "Export failed",
                                 f"{type(exc).__name__}: {exc}")

    def refresh(self) -> None:
        """This screen works on folders, not the active study — nothing to reload."""


def _mini(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight:700; font-size:13px;")
    return lbl
