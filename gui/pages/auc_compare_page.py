"""Screen — AUC Compare: two processed CSV files, compared directly.

For when the question is only "how do these two distributions differ?" and
building a study folder around each file first is overhead. Any number of
processed CSVs can be loaded; any two of them are compared, one as the
reference.

Processed files only. ``.raw`` and ``.dat`` are refused with the reason --
comparing an unconditioned recording here would compare raw signals while the
screen called them processed. Nothing is run on the files that are accepted:
no recipe, no despike, and deliberately no unit conversion. Bare CSVs carry
no units record, so values are compared **as stored** and the axes say so
rather than asserting daN; when the two files look like the same load a clean
power of ten apart, that is flagged instead of silently "fixed".

Works with no study open -- it is not in the main window's study pages.

The comparison itself is the same widget the Compare-studies screen uses
(``gui.widgets.auc_compare_panel``), so the two screens cannot disagree about
what an AUC comparison shows or how it is exported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QVBoxLayout,
)

from gui import theme
from gui.pages.base_page import BasePage
from gui.widgets.auc_compare_panel import AucComparePanel
from gui.widgets.common import Card, KpiCard, ScrollPage, SectionTitle

_DEFAULT_A = "Dataset 1"
_DEFAULT_B = "Dataset 2"
_CSV_DIR = Path(__file__).resolve().parents[2] / "csv"


def _as_stored(_display: str) -> str:
    """No unit on a bare CSV's axes: the file does not say what it is in."""
    return ""


class _CsvLoadWorker(QThread):
    """Reads processed CSVs off the UI thread -- the RLDA sample is 755 MB and
    takes the better part of a minute to parse."""
    progress = Signal(str)
    loaded = Signal(str, object, object, str)     # path, frame, RunChannels, name
    failed = Signal(str, str)                     # path, reason
    finished_all = Signal()

    def __init__(self, paths: List[str]):
        super().__init__()
        self.paths = list(paths)

    def run(self) -> None:
        from dtt.analysis.study_compare import load_processed_csv

        for i, p in enumerate(self.paths, 1):
            self.progress.emit(f"Reading {Path(p).name} ({i} of {len(self.paths)}) …")
            try:
                df, rc, name = load_processed_csv(p)
                self.loaded.emit(p, df, rc, name)
            except Exception as exc:                              # noqa: BLE001
                self.failed.emit(p, str(exc) if isinstance(exc, ValueError)
                                 else f"{type(exc).__name__}: {exc}")
        self.finished_all.emit()


class AucComparePage(BasePage):
    def __init__(self, repo, parent=None):
        super().__init__(repo, parent)

        self._files: Dict[str, tuple] = {}         # path -> (frame, rc, name)
        self._worker: Optional[_CsvLoadWorker] = None
        self._rejected: List[str] = []

        self._relabel_timer = QTimer(self)
        self._relabel_timer.setSingleShot(True)
        self._relabel_timer.setInterval(250)
        self._relabel_timer.timeout.connect(self._on_labels_changed)

        page = ScrollPage()
        body = page.body()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(page)

        body.addWidget(SectionTitle("AUC Compare — processed CSV files"))
        intro = QLabel(
            "Load two or more processed CSV files and compare any two of them. "
            "Raw imc data (.raw, .dat) is not accepted — run it through New "
            "Study or Preprocess first. Nothing is re-processed and no units "
            "are converted: values are compared exactly as stored in each "
            "file. Dataset 1 is the reference: its P5/P95 define the normal "
            "zone and the exceedance is counted against its P95.")
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:12px;")
        body.addWidget(intro)

        # ---- files ----------------------------------------------------------
        files = Card()
        row = QHBoxLayout()
        row.addWidget(_mini("Loaded files"))
        row.addStretch(1)
        self.add_btn = QPushButton("Add CSV files…")
        self.add_btn.setObjectName("Primary")
        self.add_btn.setCursor(Qt.PointingHandCursor)
        self.add_btn.clicked.connect(self._add_files)
        row.addWidget(self.add_btn)
        self.remove_btn = QPushButton("Remove selected")
        self.remove_btn.setObjectName("Secondary")
        self.remove_btn.clicked.connect(self._remove_selected)
        row.addWidget(self.remove_btn)
        files.layout().addLayout(row)
        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(96)
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        files.layout().addWidget(self.file_list)
        prog = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        prog.addWidget(self.progress, 1)
        self.status = QLabel("No files loaded.")
        self.status.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:11px;")
        prog.addWidget(self.status, 2)
        files.layout().addLayout(prog)
        body.addWidget(files)

        # ---- which two, and what to call them -------------------------------
        pick = Card()
        cols = QHBoxLayout()
        self.combo_a = QComboBox()
        self.combo_b = QComboBox()
        self.label_a = QLineEdit(_DEFAULT_A)
        self.label_b = QLineEdit(_DEFAULT_B)
        for title, combo, edit, colour in (
                ("Dataset 1  (reference)", self.combo_a, self.label_a, "#27AE60"),
                ("Dataset 2", self.combo_b, self.label_b, "#E74C3C")):
            col = QVBoxLayout()
            head = QLabel(title)
            head.setStyleSheet(f"font-weight:700; font-size:12px; color:{colour};")
            col.addWidget(head)
            combo.currentIndexChanged.connect(self._on_selection_changed)
            col.addWidget(combo)
            name_row = QHBoxLayout()
            name_row.addWidget(QLabel("Label:"))
            edit.setPlaceholderText("e.g. EV")
            edit.setToolTip(
                "The name this dataset is called everywhere on this screen — "
                "legends, percentile entries, the title, the footer and the "
                "exported CSV's columns.\n\nCosmetic only: renaming never "
                "changes which dataset is the reference.")
            edit.textChanged.connect(lambda *_: self._relabel_timer.start())
            name_row.addWidget(edit, 1)
            col.addLayout(name_row)
            cols.addLayout(col, 1)
        pick.layout().addLayout(cols)
        body.addWidget(pick)

        self.notes = QLabel("")
        self.notes.setWordWrap(True)
        self.notes.setVisible(False)
        self.notes.setStyleSheet(
            f"color:{theme.TEXT}; background:{theme.SURFACE_2}; "
            f"border-left:3px solid {theme.ORANGE}; padding:10px; font-size:11.5px;")
        body.addWidget(self.notes)

        kpi = QHBoxLayout()
        self.kpi_channels = KpiCard("Channels", "—", "in both files")
        self.kpi_p95 = KpiCard("ΔP95", "—", "on the shown channel")
        self.kpi_exceed = KpiCard("Exceedance", "—", "beyond reference P95")
        for k in (self.kpi_channels, self.kpi_p95, self.kpi_exceed):
            kpi.addWidget(k, 1)
        body.addLayout(kpi)

        self.panel = AucComparePanel(labels=self._labels,
                                     start_dir=self._start_dir,
                                     unit_fn=_as_stored, noun="file")
        self.panel.drawn.connect(self._on_panel_drawn)
        self.panel.channels_changed.connect(self._on_panel_channels)
        body.addWidget(self.panel)
        body.addStretch(1)

    # ------------------------------------------------------------ labels/dirs

    def _labels(self) -> tuple:
        a = self.label_a.text().strip() or _DEFAULT_A
        b = self.label_b.text().strip() or _DEFAULT_B
        return a, b

    def _on_labels_changed(self) -> None:
        self.panel.relabel()

    def _start_dir(self) -> str:
        if self._files:
            return str(Path(next(iter(self._files))).parent)
        return str(_CSV_DIR if _CSV_DIR.is_dir() else Path.cwd())

    # ----------------------------------------------------------------- files

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add processed CSV files", self._start_dir(),
            "Processed CSV (*.csv)")
        paths = [p for p in paths if p not in self._files]
        if not paths:
            return
        self._rejected = []
        self.add_btn.setEnabled(False)
        self.progress.setVisible(True)
        self._worker = _CsvLoadWorker(paths)
        self._worker.progress.connect(self.status.setText)
        self._worker.loaded.connect(self._on_file_loaded)
        self._worker.failed.connect(
            lambda p, why: self._rejected.append(f"{Path(p).name}: {why}"))
        self._worker.finished_all.connect(self._on_load_finished)
        self._worker.start()

    def load_paths(self, paths: List[str]) -> None:
        """Load files without the dialogs -- synchronous, for scripting and
        tests. Refusals are collected in ``_rejected`` rather than shown in a
        modal box, which would block a headless caller forever."""
        from dtt.analysis.study_compare import load_processed_csv

        self._rejected = []
        for p in paths:
            try:
                df, rc, name = load_processed_csv(p)
                self._on_file_loaded(str(p), df, rc, name)
            except ValueError as exc:
                self._rejected.append(f"{Path(p).name}: {exc}")
        self._on_load_finished(interactive=False)

    def _on_file_loaded(self, path: str, df, rc, name: str) -> None:
        from dtt.analysis.study_compare import study_duration_s

        self._files[path] = (df, rc, name)
        dur = study_duration_s(df) or 0.0
        n_ch = len(df.columns) - 1
        item = QListWidgetItem(
            f"{name}   —   {len(df):,} rows · {dur:,.0f} s · {n_ch} channels")
        item.setData(Qt.UserRole, path)
        item.setToolTip(path)
        self.file_list.addItem(item)

    def _on_load_finished(self, interactive: bool = True) -> None:
        self.add_btn.setEnabled(True)
        self.progress.setVisible(False)
        self._refill_combos()
        n = len(self._files)
        self.status.setText(f"{n} file{'s' if n != 1 else ''} loaded."
                            + ("" if n >= 2 else " Load at least two to compare."))
        if self._rejected and interactive:
            QMessageBox.warning(
                self, "Some files were not loaded",
                "These were refused and nothing was loaded from them:\n\n"
                + "\n\n".join(self._rejected))

    def _remove_selected(self) -> None:
        for item in self.file_list.selectedItems():
            self._files.pop(item.data(Qt.UserRole), None)
            self.file_list.takeItem(self.file_list.row(item))
        self._refill_combos()
        self.status.setText(f"{len(self._files)} file(s) loaded.")

    def _refill_combos(self) -> None:
        """Both selectors offer every loaded file; default to the first two."""
        prev_a, prev_b = self.combo_a.currentData(), self.combo_b.currentData()
        paths = list(self._files)
        for combo in (self.combo_a, self.combo_b):
            combo.blockSignals(True)
            combo.clear()
            for p in paths:
                combo.addItem(self._files[p][2], p)
        ia = paths.index(prev_a) if prev_a in paths else 0
        ib = (paths.index(prev_b) if prev_b in paths
              else (1 if len(paths) > 1 else 0))
        if paths:
            self.combo_a.setCurrentIndex(ia)
            self.combo_b.setCurrentIndex(ib)
        for combo in (self.combo_a, self.combo_b):
            combo.blockSignals(False)
        self._on_selection_changed()

    # ------------------------------------------------------------- comparing

    def _default_label(self, edit: QLineEdit, default: str, old: str,
                       new: str) -> None:
        """Name a dataset after its file, unless someone has typed a name."""
        if edit.text().strip() in ("", default, old):
            edit.blockSignals(True)
            edit.setText(new)
            edit.blockSignals(False)

    def _on_selection_changed(self, *_) -> None:
        from dtt.analysis.study_compare import (
            auc_cross_channels, channel_differences, csv_channels,
            unit_scale_warning)

        pa, pb = self.combo_a.currentData(), self.combo_b.currentData()
        if not pa or not pb or pa == pb:
            self.panel.clear()
            self.notes.setVisible(bool(pa and pa == pb))
            self.notes.setText("Dataset 1 and Dataset 2 are the same file — "
                               "pick two different files.")
            self.kpi_channels.set_value("—")
            return
        (df_a, rc_a, name_a), (df_b, rc_b, name_b) = self._files[pa], self._files[pb]
        self._default_label(self.label_a, _DEFAULT_A,
                            getattr(self, "_name_a", ""), name_a)
        self._default_label(self.label_b, _DEFAULT_B,
                            getattr(self, "_name_b", ""), name_b)
        self._name_a, self._name_b = name_a, name_b
        a, b = self._labels()

        same = csv_channels(rc_a, rc_b, df_a, df_b)
        cross = auc_cross_channels(rc_a, rc_b, df_a, df_b)
        only_a, only_b = channel_differences(rc_a, rc_b, df_a, df_b)
        reason = self.panel.set_data(df_a, df_b, same, cross)

        notes = []
        for who, missing in ((b, only_a), (a, only_b)):
            if missing:
                shown = ", ".join(missing[:8]) + (" …" if len(missing) > 8 else "")
                notes.append(f"Not in {who}, so not compared: {shown} "
                             f"({len(missing)} channel{'s' if len(missing) != 1 else ''}).")
        warn = unit_scale_warning(same, df_a, df_b, a, b)
        if warn:
            notes.append(warn)
        if reason:
            notes.append(f"Distance weighting unavailable — {reason}. "
                         f"Comparing by sample count.")
        self.notes.setText("• " + "\n• ".join(notes) if notes else "")
        self.notes.setVisible(bool(notes))

    def _on_panel_drawn(self, cmp, _unit: str) -> None:
        if cmp is None:
            self.kpi_p95.set_value("—")
            self.kpi_exceed.set_value("—")
            return
        a, b = self._labels()
        self.kpi_p95.set_value(f"{cmp.delta_p95:+.4g}", f"as stored, {b} vs {a}")
        self.kpi_exceed.set_value(f"{cmp.pct_exceed_ref_p95:.1f}%",
                                  f"of {b} beyond {a} P95")

    def _on_panel_channels(self, n: int, mode: str) -> None:
        self.kpi_channels.set_value(
            str(n), "cross-axle pairs" if mode == "cross" else "in both files")

    def refresh(self) -> None:
        """Works on files, not the active study — nothing to reload."""


def _mini(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight:700; font-size:13px;")
    return lbl
