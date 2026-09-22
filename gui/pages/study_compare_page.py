"""Screen — Compare studies: two finished studies, side by side.

The sibling of ``comparison_page.py``, and deliberately not part of it. That
screen compares two **raw imc folders**: it globs ``*.raw``, reads them with
``load_recording`` and bypasses the FAMOS/despike/stop-removal pipeline
entirely. This one compares two studies that have already been through it, so
the "before/after" is between two completed runs — an EV against an IC variant
on the same route — rather than between a raw recording and its conditioned
version. Sharing one widget tree between the two would mean one page holding
two input layers, two workers and two result types.

The backend has existed since Round 4 (``dtt/analysis/study_compare.py``: RMS,
DLC and G-severity deltas, provenance guard, RF Compare) and was reachable
only from Python. This is the surface for it, plus the two-dataset AUC view.

Both datasets are **named**, and the names go everywhere: legends, percentile
entries, the figure title, the footer strip, the table headers and the
exported CSV's column names. "previous" and "current" are useless labels when
the two runs are EV and IC. Editing a name re-renders; it never re-runs the
comparison, because a label is cosmetic and must not be able to change which
dataset is the reference for the P5/P95 and exceedance arithmetic.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMessageBox, QProgressBar, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout,
)

from gui import theme
from gui.pages.base_page import BasePage
from gui.widgets.common import Card, KpiCard, ScrollPage, SectionTitle
from gui.widgets.mpl_canvas import PlotPanel

_DEFAULT_A = "Dataset 1"
_DEFAULT_B = "Dataset 2"


class _StudyCompareWorker(QThread):
    """Loads both studies and computes the deltas off the UI thread."""
    progress = Signal(str)
    done = Signal(object, object, object, object, object, object)
    # (result, df_a, rc_a, df_b, rc_b, error)

    def __init__(self, dir_a: Path, dir_b: Path):
        super().__init__()
        self.dir_a, self.dir_b = Path(dir_a), Path(dir_b)

    def run(self) -> None:
        try:
            from dtt.analysis.study_compare import compare_studies, load_study

            self.progress.emit(f"Reading {self.dir_a.name} …")
            df_a, rc_a, _ = load_study(self.dir_a)
            self.progress.emit(f"Reading {self.dir_b.name} …")
            df_b, rc_b, _ = load_study(self.dir_b)
            self.progress.emit("Comparing …")
            result = compare_studies(self.dir_a, self.dir_b)
            self.done.emit(result, df_a, rc_a, df_b, rc_b, None)
        except Exception as exc:                                  # noqa: BLE001
            self.done.emit(None, None, None, None, None,
                           f"{type(exc).__name__}: {exc}")


class StudyComparePage(BasePage):
    def __init__(self, repo, parent=None):
        super().__init__(repo, parent)

        self._result = None
        self._df_a = self._df_b = None
        self._rc_a = self._rc_b = None
        self._dir_a: Optional[Path] = None
        self._dir_b: Optional[Path] = None
        self._channels: List[tuple] = []
        self._worker: Optional[_StudyCompareWorker] = None

        # Re-rendering on every keystroke of a label would redraw a KDE per
        # character; a short debounce keeps typing responsive without needing
        # the user to press anything.
        self._relabel_timer = QTimer(self)
        self._relabel_timer.setSingleShot(True)
        self._relabel_timer.setInterval(250)
        self._relabel_timer.timeout.connect(self._on_labels_changed)

        page = ScrollPage()
        body = page.body()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(page)

        body.addWidget(SectionTitle("Compare studies — two processed runs"))
        intro = QLabel(
            "Pick two finished studies. Channels are matched by wheel position "
            "and component, so studies that name them differently still line "
            "up, and every force and moment channel both studies recorded is "
            "offered. Dataset 1 is the reference: its P5/P95 define the normal "
            "zone and the exceedance is counted against its P95.")
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:12px;")
        body.addWidget(intro)

        # ---- pickers + labels ----------------------------------------------
        pick = Card()
        row = QHBoxLayout()
        self.path_a = QLabel("No study selected")
        self.path_b = QLabel("No study selected")
        self.label_a = QLineEdit(_DEFAULT_A)
        self.label_b = QLineEdit(_DEFAULT_B)
        for title, lbl, edit, slot, colour in (
                ("Dataset 1  (reference)", self.path_a, self.label_a,
                 self._pick_a, "#27AE60"),
                ("Dataset 2", self.path_b, self.label_b,
                 self._pick_b, "#E74C3C")):
            col = QVBoxLayout()
            head = QLabel(title)
            head.setStyleSheet(f"font-weight:700; font-size:12px; color:{colour};")
            btn = QPushButton("Choose study folder…")
            btn.setObjectName("Secondary")
            btn.clicked.connect(slot)
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:11px;")
            name_row = QHBoxLayout()
            name_row.addWidget(QLabel("Label:"))
            edit.setPlaceholderText("e.g. EV")
            edit.setToolTip(
                "The name this dataset is called everywhere on this screen — "
                "plot legends, percentile entries, the title, the footer, the "
                "table headers and the exported CSV's columns.\n\n"
                "Cosmetic only: renaming never changes which dataset is the "
                "reference for the P5/P95 and exceedance figures.")
            edit.textChanged.connect(lambda *_: self._relabel_timer.start())
            name_row.addWidget(edit, 1)
            col.addWidget(head)
            col.addWidget(btn)
            col.addWidget(lbl)
            col.addLayout(name_row)
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

        # ---- provenance warnings -------------------------------------------
        self.notes = QLabel("")
        self.notes.setWordWrap(True)
        self.notes.setVisible(False)
        self.notes.setStyleSheet(
            f"color:{theme.TEXT}; background:{theme.SURFACE_2}; "
            f"border-left:3px solid {theme.ORANGE}; padding:10px; font-size:11.5px;")
        body.addWidget(self.notes)

        # ---- KPI row --------------------------------------------------------
        kpi = QHBoxLayout()
        self.kpi_channels = KpiCard("Channels", "—", "in both studies")
        self.kpi_wheels = KpiCard("Wheels", "—", "matched positions")
        self.kpi_p95 = KpiCard("ΔP95", "—", "on the shown channel")
        self.kpi_exceed = KpiCard("Exceedance", "—", "beyond reference P95")
        for k in (self.kpi_channels, self.kpi_wheels, self.kpi_p95,
                  self.kpi_exceed):
            kpi.addWidget(k, 1)
        body.addLayout(kpi)

        # ---- the AUC pair ---------------------------------------------------
        auc_card = Card()
        head = QHBoxLayout()
        head.addWidget(_mini("Area under curve — distribution comparison"))
        head.addStretch(1)
        head.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        self.channel_combo.setMinimumWidth(140)
        self.channel_combo.currentTextChanged.connect(lambda *_: self._draw_auc())
        head.addWidget(self.channel_combo)
        self.save_btn = QPushButton("Save PNG…")
        self.save_btn.setObjectName("Secondary")
        self.save_btn.clicked.connect(self._save_png)
        self.save_btn.setEnabled(False)
        head.addWidget(self.save_btn)
        auc_card.layout().addLayout(head)
        # Light, two panels sharing an x-range: this section keeps the white
        # reference look the Histogram and AUC screens were restyled to, not
        # the app's navy chrome.
        self.auc = PlotPanel(height=4.4, ncols=2, light=True)
        auc_card.layout().addWidget(self.auc)
        self.footer = QLabel("")
        self.footer.setWordWrap(True)
        self.footer.setStyleSheet(
            "font-family:monospace; font-size:11px; padding:8px; "
            f"background:{theme.SURFACE_2}; color:{theme.TEXT};")
        auc_card.layout().addWidget(self.footer)
        body.addWidget(auc_card)

        # ---- delta tables ---------------------------------------------------
        table_card = Card()
        table_card.layout().addWidget(_mini("RMS · DLC · G-severity"))
        self.table = QTableWidget(0, 6)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(280)
        table_card.layout().addWidget(self.table)
        body.addWidget(table_card)

        exp = QHBoxLayout()
        exp.addWidget(QLabel("Export:"))
        self.export_btn = QPushButton("AUC summary + deltas (CSV)…")
        self.export_btn.setObjectName("Primary")
        self.export_btn.clicked.connect(self._export_csv)
        self.export_btn.setEnabled(False)
        exp.addWidget(self.export_btn)
        exp.addStretch(1)
        body.addLayout(exp)
        body.addStretch(1)

        self._set_table_headers()

    # ------------------------------------------------------------------ labels

    def _labels(self) -> tuple:
        """The two names, never blank — a legend entry with no text is worse
        than a default one."""
        a = self.label_a.text().strip() or _DEFAULT_A
        b = self.label_b.text().strip() or _DEFAULT_B
        return a, b

    def _on_labels_changed(self) -> None:
        """Re-render only. The comparison is not recomputed, and cannot be:
        nothing downstream of a label feeds the percentile arithmetic."""
        if self._result is None:
            return
        a, b = self._labels()
        # The result objects already carry the labels; keep them the single
        # store rather than adding a parallel one.
        self._result.label_a, self._result.label_b = a, b
        self._set_table_headers()
        self._fill_table()
        self._draw_auc()

    def _set_table_headers(self) -> None:
        a, b = self._labels()
        self.table.setHorizontalHeaderLabels(
            ["Channel / wheel", "Metric", a, b, "Δ", "Δ%"])

    # ----------------------------------------------------------------- picking

    def _start_dir(self) -> str:
        for p in (self._dir_a, self._dir_b):
            if p:
                return str(p.parent)
        return str(getattr(self.repo, "outputs_dir", Path.cwd()))

    def _pick(self, title: str) -> Optional[Path]:
        path = QFileDialog.getExistingDirectory(self, title, self._start_dir())
        if not path:
            return None
        folder = Path(path)
        if not (folder / "processed_data.csv").exists():
            QMessageBox.warning(
                self, "Not a processed study",
                f"No processed_data.csv in:\n{path}\n\n"
                "Pick a study folder from outputs/ — this screen compares two "
                "runs that have already been through the pipeline. For two raw "
                "imc folders, use the Compare screen instead.")
            return None
        return folder

    def _pick_a(self) -> None:
        f = self._pick("Select DATASET 1 (the reference study)")
        if f:
            self._dir_a = f
            self.path_a.setText(f.name)
            if self.label_a.text().strip() in ("", _DEFAULT_A):
                self.label_a.setText(f.name)
            self._refresh_ready()

    def _pick_b(self) -> None:
        f = self._pick("Select DATASET 2")
        if f:
            self._dir_b = f
            self.path_b.setText(f.name)
            if self.label_b.text().strip() in ("", _DEFAULT_B):
                self.label_b.setText(f.name)
            self._refresh_ready()

    def _refresh_ready(self) -> None:
        ready = bool(self._dir_a and self._dir_b)
        self.compare_btn.setEnabled(ready)
        if ready and self._dir_a == self._dir_b:
            self.status.setText("Both are the same study — pick two different runs.")
            self.compare_btn.setEnabled(False)
        elif ready:
            self.status.setText("Ready to compare.")

    # ----------------------------------------------------------------- running

    def _run_compare(self) -> None:
        if not (self._dir_a and self._dir_b):
            return
        self.compare_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.status.setText("Loading…")
        self._worker = _StudyCompareWorker(self._dir_a, self._dir_b)
        self._worker.progress.connect(self.status.setText)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _on_done(self, result, df_a, rc_a, df_b, rc_b, error) -> None:
        self.progress.setVisible(False)
        self.compare_btn.setEnabled(True)
        if error:
            self.status.setText("Failed.")
            QMessageBox.critical(self, "Comparison failed", error)
            return
        from dtt.analysis.study_compare import auc_channels

        self._result = result
        self._df_a, self._rc_a = df_a, rc_a
        self._df_b, self._rc_b = df_b, rc_b
        a, b = self._labels()
        result.label_a, result.label_b = a, b
        self._channels = auc_channels(rc_a, rc_b, df_a, df_b)

        # A provenance mismatch is a warning, never a block: comparing two runs
        # made by different pipeline versions is sometimes exactly what is
        # wanted, and sometimes the reason the numbers look odd.
        if result.warnings:
            self.notes.setText("• " + "\n• ".join(result.warnings))
            self.notes.setVisible(True)
        else:
            self.notes.setVisible(False)

        self.kpi_channels.set_value(str(len(self._channels)))
        self.kpi_wheels.set_value(
            str(len({c[0].rsplit("_", 1)[0] for c in self._channels})))

        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        self.channel_combo.addItems([c[0] for c in self._channels])
        self.channel_combo.blockSignals(False)

        self._set_table_headers()
        self._fill_table()
        self._draw_auc()
        self.save_btn.setEnabled(bool(self._channels))
        self.export_btn.setEnabled(True)
        self.status.setText(
            f"Compared {len(self._channels)} channels · "
            f"{len(result.rms)} RMS, {len(result.dlc)} DLC, "
            f"{len(result.gseverity)} severity rows.")

    # --------------------------------------------------------------- rendering

    def _fill_table(self) -> None:
        if self._result is None:
            return
        self.table.setRowCount(0)
        rows = (list(self._result.rms) + list(self._result.dlc)
                + list(self._result.gseverity))
        for d in rows:
            r = self.table.rowCount()
            self.table.insertRow(r)
            cells = [d.label, d.metric, f"{d.a:.4g}", f"{d.b:.4g}",
                     f"{d.delta:+.4g}",
                     "—" if not np.isfinite(d.pct) else f"{d.pct:+.1f}%"]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col > 1:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(r, col, item)

    def _selected(self):
        """``(display, values_a, values_b)`` for the channel on screen."""
        label = self.channel_combo.currentText()
        for disp, ch_a, ch_b in self._channels:
            if disp == label:
                va = pd.to_numeric(self._df_a[ch_a], errors="coerce").to_numpy(float)
                vb = pd.to_numeric(self._df_b[ch_b], errors="coerce").to_numpy(float)
                return disp, va, vb
        return None, None, None

    @staticmethod
    def _unit(display: str) -> str:
        """daN for a force, daN·m for a moment — the units the pipeline writes.

        Not a magnitude heuristic: the component is in the channel's own name,
        and guessing from the numbers is what produced a decade error once
        already.
        """
        return "daN·m" if display.rsplit("_", 1)[-1].startswith("M") else "daN"

    def _draw_auc(self) -> None:
        if self._result is None or not self._channels:
            return
        from dtt.analysis.study_compare import (
            auc_footer_text, draw_auc_comparison)
        from dtt.analysis.histograms import SUPTITLE_FONTSIZE, TEXT_PRI

        disp, va, vb = self._selected()
        if disp is None:
            return
        a, b = self._labels()
        unit = self._unit(disp)
        self.auc.clear()
        cmp = draw_auc_comparison(self.auc.axes[0], self.auc.axes[1],
                                  va, vb, a, b, disp, unit)
        self.auc.canvas.fig.suptitle(
            f"AUC — {disp}  ({a}  vs  {b}  reference)",
            color=TEXT_PRI, fontsize=SUPTITLE_FONTSIZE, fontweight="bold")
        self.auc.canvas.fig.tight_layout(rect=[0, 0, 1, 0.92])
        self.auc.draw()

        if cmp is None:
            self.footer.setText("Insufficient data on this channel.")
            self.kpi_p95.set_value("—")
            self.kpi_exceed.set_value("—")
            return
        self.footer.setText(auc_footer_text(cmp, a, b, unit))
        self.kpi_p95.set_value(f"{cmp.delta_p95:+.0f}", f"{unit}, {b} vs {a}")
        self.kpi_exceed.set_value(f"{cmp.pct_exceed_ref_p95:.1f}%",
                                  f"of {b} beyond {a} P95")

    # ----------------------------------------------------------------- export

    def _save_png(self) -> None:
        from dtt.analysis.study_compare import generate_auc_comparison

        disp, va, vb = self._selected()
        if disp is None:
            return
        a, b = self._labels()
        suggested = f"auc_{disp}_{a}_vs_{b}.png"
        suggested = "".join(c if c.isalnum() or c in "-_." else "_"
                            for c in suggested)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the AUC comparison",
            str(Path(self._start_dir()) / suggested), "PNG image (*.png)")
        if not path:
            return
        try:
            generate_auc_comparison(va, vb, a, b, disp, Path(path),
                                    self._unit(disp))
            QMessageBox.information(self, "Saved", f"Written:\n{path}")
        except Exception as exc:                                   # noqa: BLE001
            QMessageBox.critical(self, "Save failed", f"{type(exc).__name__}: {exc}")

    def _export_csv(self) -> None:
        """One CSV per comparison, with the labels in the column names.

        The reference analyzer writes ``P95_<label>`` keys so a saved summary
        says which run each number belongs to; that is what is reproduced here,
        driven by whatever the two fields currently say.
        """
        from dtt.analysis.study_compare import auc_comparison_stats

        if self._result is None:
            return
        a, b = self._labels()
        stem = "".join(c if c.isalnum() or c in "-_" else "_"
                       for c in f"study_comparison_{a}_vs_{b}")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export the comparison",
            str(Path(self._start_dir()) / f"{stem}.csv"), "CSV (*.csv)")
        if not path:
            return
        try:
            rows = []
            for disp, ch_a, ch_b in self._channels:
                va = pd.to_numeric(self._df_a[ch_a], errors="coerce").to_numpy(float)
                vb = pd.to_numeric(self._df_b[ch_b], errors="coerce").to_numpy(float)
                row = auc_comparison_stats(va, vb, a, b, disp, self._unit(disp))
                if row:
                    rows.append(row)
            auc_path = Path(path)
            pd.DataFrame(rows).to_csv(auc_path, index=False)

            deltas = [{"Label": d.label, "Metric": d.metric, a: d.a, b: d.b,
                       "Delta": d.delta, "Delta_pct": d.pct}
                      for d in (list(self._result.rms) + list(self._result.dlc)
                                + list(self._result.gseverity))]
            delta_path = auc_path.with_name(auc_path.stem + "_deltas.csv")
            pd.DataFrame(deltas).to_csv(delta_path, index=False)
            QMessageBox.information(
                self, "Export complete", f"Written:\n{auc_path}\n{delta_path}")
        except Exception as exc:                                   # noqa: BLE001
            QMessageBox.critical(self, "Export failed",
                                 f"{type(exc).__name__}: {exc}")

    def refresh(self) -> None:
        """Works on two chosen studies, not the active one — nothing to reload."""


def _mini(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight:700; font-size:13px;")
    return lbl
