"""
Backend integration: run the DTT pipeline in a worker thread.

The GUI does NOT import the analytical backend. It launches the exact same CLI
the project documents::

    python -m dtt.pipeline --csv <path> --vehicle <name> --study <name> ...

run from the project root, captures the merged stdout/stderr stream line by
line, and translates the pipeline's ``[n/9]  Stage`` log markers into
structured Qt signals the Processing screen consumes.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QObject, QThread, Signal

from gui.models.repository import PROJECT_DIR


@dataclass
class RunRequest:
    csv_path: Optional[Path] = None
    raw_folder: Optional[Path] = None
    raw_files: Optional[List[Path]] = None
    vehicle: str = "Vehicle"
    vehicle_type: str = ""
    study: str = ""
    cutoff: Optional[float] = None
    order: Optional[int] = None
    miner: Optional[float] = None
    no_filter: bool = False
    no_famos: bool = False        # fall back to the legacy Butterworth LPF
    deglitch: bool = False        # rolling-median removal of DAQ artifact spikes

    def to_cmd(self) -> List[str]:
        # A frozen .exe re-invokes itself in pipeline mode; from source we call
        # the module directly.
        if getattr(sys, "frozen", False):
            cmd = [sys.executable, "--run-pipeline"]
        else:
            cmd = [sys.executable, "-m", "dtt.pipeline"]
        if self.raw_files:
            cmd += ["--raw-files"] + [str(f) for f in self.raw_files]
        elif self.raw_folder:
            cmd += ["--raw", str(self.raw_folder)]
        else:
            cmd += ["--csv", str(self.csv_path)]
        cmd += ["--vehicle", self.vehicle or "Vehicle"]
        if self.vehicle_type:
            cmd += ["--vehicle-type", self.vehicle_type]
        if self.study:
            cmd += ["--study", self.study]
        if self.cutoff is not None:
            cmd += ["--cutoff", str(self.cutoff)]
        if self.order is not None:
            cmd += ["--order", str(self.order)]
        if self.miner is not None:
            cmd += ["--miner", str(self.miner)]
        if self.no_filter:
            cmd += ["--no-filter"]
        if self.no_famos:
            cmd += ["--no-famos"]
        if self.deglitch:
            cmd += ["--deglitch"]
        return cmd


# Ordered stage list shown in the Processing screen. The regex maps a log line
# to the stage index. The backend logs e.g. "[1/9]  Data Ingestion".
STAGES: List[str] = [
    "Data Ingestion",
    "Validation",
    "Sanitization",
    "Filtering",
    "Statistics",
    "Histograms",
    "Heatmaps",
    "Boxplots",
    "Rainflow",
    "Report Generation",
]

# Backend log phrase  ->  GUI stage name
_STAGE_MARKERS = [
    (re.compile(r"\[1/9\].*Ingest", re.I),        "Data Ingestion"),
    (re.compile(r"\[2/9\].*Valid", re.I),         "Validation"),
    (re.compile(r"\[3/9\].*Sanitiz", re.I),       "Sanitization"),
    (re.compile(r"\[4/9\].*(Signal|Filter|Butter)", re.I), "Filtering"),
    (re.compile(r"\[5/9\].*Statist", re.I),       "Statistics"),
    (re.compile(r"\[6/9\].*Histogram", re.I),     "Histograms"),
    (re.compile(r"\[7/9\].*Heatmap", re.I),       "Heatmaps"),
    (re.compile(r"\[8/9\].*Boxplot", re.I),       "Boxplots"),
    (re.compile(r"\[9/9\].*Rainflow", re.I),      "Rainflow"),
    (re.compile(r"\[Report\]", re.I),             "Report Generation"),
]

_DONE_RE = re.compile(r"Pipeline complete", re.I)


class PipelineWorker(QObject):
    """Runs the pipeline subprocess and emits progress signals.

    Signals are emitted from the worker thread; connect them with the default
    (queued) connection so slots run on the GUI thread.
    """

    log_line     = Signal(str)            # raw log text
    stage_started = Signal(str)           # stage name
    stage_finished = Signal(str, float)   # stage name, seconds
    progress     = Signal(int, int)       # completed stages, total
    finished     = Signal(bool, str)      # success, study_name / error message

    def __init__(self, request: RunRequest):
        super().__init__()
        self._request = request
        self._proc: Optional[subprocess.Popen] = None
        self._cancelled = False
        self._current_stage: Optional[str] = None
        self._stage_start = 0.0
        self._completed = 0

    # Public API
    def cancel(self) -> None:
        self._cancelled = True
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def run(self) -> None:
        """Entry point — call via QThread.started."""
        cmd = self._request.to_cmd()
        self.log_line.emit("$ " + " ".join(cmd))
        # We decode the pipe as UTF-8, so tell the child to encode it that way.
        # Without this it writes the Windows ANSI codepage and any non-ASCII in a
        # log line (the en-dash in the banner) arrives as a replacement char.
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
                env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception as exc:                     # noqa: BLE001
            self.finished.emit(False, f"Failed to launch pipeline: {exc}")
            return

        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._cancelled:
                break
            line = line.rstrip("\n")
            if line:
                self.log_line.emit(line)
                self._dispatch_stage(line)

        self._proc.wait()
        self._close_current_stage()

        if self._cancelled:
            self.finished.emit(False, "Cancelled by user")
            return

        if self._proc.returncode == 0:
            self.progress.emit(len(STAGES), len(STAGES))
            self.finished.emit(True, self._request.study or _infer_latest_study())
        else:
            self.finished.emit(False, f"Pipeline exited with code {self._proc.returncode}")

    # Stage tracking
    def _dispatch_stage(self, line: str) -> None:
        for pattern, stage in _STAGE_MARKERS:
            if pattern.search(line):
                self._close_current_stage()
                self._current_stage = stage
                self._stage_start = time.perf_counter()
                self.stage_started.emit(stage)
                self.progress.emit(self._completed, len(STAGES))
                return
        if _DONE_RE.search(line):
            self._close_current_stage()

    def _close_current_stage(self) -> None:
        if self._current_stage is not None:
            elapsed = time.perf_counter() - self._stage_start
            self._completed += 1
            self.stage_finished.emit(self._current_stage, elapsed)
            self.progress.emit(self._completed, len(STAGES))
            self._current_stage = None


def _infer_latest_study() -> str:
    """When study was auto-named by timestamp, find the newest folder."""
    from gui.models.repository import StudyRepository
    latest = StudyRepository().latest_study()
    return latest.name if latest else ""


class PipelineController(QObject):
    """Owns the QThread + worker lifecycle so views stay simple."""

    log_line       = Signal(str)
    stage_started  = Signal(str)
    stage_finished = Signal(str, float)
    progress       = Signal(int, int)
    finished       = Signal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread: Optional[QThread] = None
        self._worker: Optional[PipelineWorker] = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def start(self, request: RunRequest) -> None:
        if self.is_running:
            return
        self._thread = QThread()
        self._worker = PipelineWorker(request)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log_line.connect(self.log_line)
        self._worker.stage_started.connect(self.stage_started)
        self._worker.stage_finished.connect(self.stage_finished)
        self._worker.progress.connect(self.progress)
        self._worker.finished.connect(self._on_finished)

        self._thread.start()

    def cancel(self) -> None:
        if self._worker:
            self._worker.cancel()

    def _on_finished(self, success: bool, info: str) -> None:
        self.finished.emit(success, info)
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        self._worker = None
        self._thread = None
