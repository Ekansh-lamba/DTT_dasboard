"""Screen 3 — Processing: live pipeline stages, progress, timings, logs."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QLabel, QProgressBar, QPlainTextEdit,
    QFrame, QPushButton, QSplitter,
)

from gui import theme
from gui.pages.base_page import BasePage
from gui.widgets.common import SectionTitle, Card
from gui.workers.pipeline_worker import STAGES


_DOT = {
    "pending": (theme.TEXT_FAINT, "○"),
    "running": (theme.ORANGE, "◐"),
    "done":    (theme.SUCCESS, "●"),
    "error":   (theme.DANGER, "✕"),
}


class _StageRow(QFrame):
    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.name = name
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 7, 10, 7)
        lay.setSpacing(10)

        self.dot = QLabel("○")
        self.dot.setFixedWidth(16)
        self.label = QLabel(name)
        self.status = QLabel("Pending")
        self.status.setStyleSheet(f"color:{theme.TEXT_FAINT};")
        self.time = QLabel("")
        self.time.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-family:monospace;")

        lay.addWidget(self.dot)
        lay.addWidget(self.label, 1)
        lay.addWidget(self.status)
        lay.addWidget(self.time)
        self.set_state("pending")

    def set_state(self, state: str, detail: str = "", seconds: float | None = None):
        color, glyph = _DOT[state]
        self.dot.setText(glyph)
        self.dot.setStyleSheet(f"color:{color}; font-size:15px;")
        text = {"pending": "Pending", "running": "Running…",
                "done": "Done", "error": "Failed"}[state]
        self.status.setText(detail or text)
        self.status.setStyleSheet(f"color:{color};")
        if seconds is not None:
            self.time.setText(f"{seconds:.1f}s")


class ProcessingPage(BasePage):
    def __init__(self, repo, parent=None):
        super().__init__(repo, parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)

        head = QHBoxLayout()
        head.addWidget(SectionTitle("Pipeline Processing"))
        head.addStretch(1)
        self.badge = QLabel("Idle")
        self.badge.setStyleSheet(
            f"background:{theme.SURFACE_2}; color:{theme.TEXT_MUTED};"
            f"border-radius:10px; padding:4px 12px; font-weight:600;")
        head.addWidget(self.badge)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("Secondary")
        self.cancel_btn.setEnabled(False)
        head.addWidget(self.cancel_btn)
        root.addLayout(head)

        self.progress = QProgressBar()
        self.progress.setRange(0, len(STAGES))
        self.progress.setValue(0)
        self.progress.setFormat("%v / %m stages")
        root.addWidget(self.progress)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        # Stage list
        stage_card = Card()
        stage_card.layout().addWidget(_mini_title("Stages"))
        self.rows: dict[str, _StageRow] = {}
        for name in STAGES:
            row = _StageRow(name)
            self.rows[name] = row
            stage_card.layout().addWidget(row)
        stage_card.layout().addStretch(1)
        splitter.addWidget(stage_card)

        # Logs
        log_card = Card()
        log_card.layout().addWidget(_mini_title("Live Log"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        log_card.layout().addWidget(self.log)
        splitter.addWidget(log_card)
        splitter.setSizes([420, 640])

    # Worker signal handlers
    def begin(self, study_name: str) -> None:
        for row in self.rows.values():
            row.set_state("pending")
            row.time.setText("")
        self.log.clear()
        self.progress.setValue(0)
        self.badge.setText("Running")
        self.badge.setStyleSheet(
            f"background:{theme.ORANGE}33; color:{theme.ORANGE};"
            f"border-radius:10px; padding:4px 12px; font-weight:600;")
        self.cancel_btn.setEnabled(True)
        self.append_log(f"Starting analysis: {study_name or '(timestamp study)'}")

    def append_log(self, line: str) -> None:
        self.log.appendPlainText(line)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def on_stage_started(self, name: str) -> None:
        if name in self.rows:
            self.rows[name].set_state("running")

    def on_stage_finished(self, name: str, seconds: float) -> None:
        if name in self.rows:
            self.rows[name].set_state("done", seconds=seconds)

    def on_progress(self, completed: int, total: int) -> None:
        self.progress.setMaximum(total)
        self.progress.setValue(completed)

    def on_finished(self, success: bool, info: str) -> None:
        self.cancel_btn.setEnabled(False)
        if success:
            for row in self.rows.values():
                if row.status.text() in ("Pending", "Running…"):
                    row.set_state("done")
            self.progress.setValue(self.progress.maximum())
            self.badge.setText("Complete")
            self.badge.setStyleSheet(
                f"background:{theme.SUCCESS}33; color:{theme.SUCCESS};"
                f"border-radius:10px; padding:4px 12px; font-weight:600;")
            self.append_log(f"✔ Finished — study '{info}'")
        else:
            for row in self.rows.values():
                if row.status.text() == "Running…":
                    row.set_state("error")
            self.badge.setText("Error")
            self.badge.setStyleSheet(
                f"background:{theme.DANGER}33; color:{theme.DANGER};"
                f"border-radius:10px; padding:4px 12px; font-weight:600;")
            self.append_log(f"✘ {info}")


def _mini_title(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight:700; font-size:13px;")
    return lbl
