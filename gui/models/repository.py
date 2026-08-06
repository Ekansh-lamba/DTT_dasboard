"""
Model layer for the DTT WFT Automation Platform GUI.

This module is the single source of truth for reading everything the backend
pipeline produces. The GUI never reads files directly — it goes through
``StudyRepository`` so that the on-disk output contract lives in exactly one
place.

Backend output contract (one folder per study under ``dtt/outputs/``)::

    outputs/{study_name}/
        validation_report.json
        stats_summary.json
        processed_data.csv
        figures/
            hist_distance_*.png      hist_percentage_*.png
            heatmap_*.png            heatmap_hexbin_*.png
            boxplot_*.png
            rainflow_*.png           rainflow_cycles_*.csv
        logs/pipeline.log
        WFT_Report_{vehicle}_{date}.pptx
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Canonical locations — shared with the backend so the GUI and the pipeline
# (incl. the frozen exe's persistent data folder) always agree.
from dtt.config import OUTPUTS_DIR, CSV_DIR, DATA_DIR

PROJECT_DIR = Path(__file__).resolve().parents[2]      # …/apollo tyres (source runs)

WHEELS  = ("FL", "FR", "RL", "RR")
SIGNALS = ("Fx", "Fy", "Fz")

# Maps a study status to the order pipeline stages are emitted in the log.
PIPELINE_STAGES: List[str] = [
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


# Value objects

@dataclass
class ChannelStat:
    """One row of stats_summary.json."""
    channel: str
    mean: float
    median: float
    std: float
    min: float
    max: float
    p80: float
    p90: float
    p95: float

    @property
    def wheel(self) -> str:
        return self.channel.split("_")[0]

    @property
    def signal(self) -> str:
        return self.channel.split("_")[-1]

    def as_row(self) -> List:
        return [
            self.channel, self.mean, self.median, self.std,
            self.min, self.max, self.p80, self.p90, self.p95,
        ]


@dataclass
class Study:
    """A single processed (or in-progress) study folder."""
    name: str
    path: Path

    # Lazily-populated cached payloads
    _validation: Optional[dict] = field(default=None, repr=False)
    _stats: Optional[Dict[str, ChannelStat]] = field(default=None, repr=False)
    _channels: object = field(default=None, repr=False)

    # Derived paths
    @property
    def figures_dir(self) -> Path:
        return self.path / "figures"

    @property
    def logs_dir(self) -> Path:
        return self.path / "logs"

    @property
    def log_file(self) -> Path:
        return self.logs_dir / "pipeline.log"

    @property
    def validation_file(self) -> Path:
        return self.path / "validation_report.json"

    @property
    def stats_file(self) -> Path:
        return self.path / "stats_summary.json"

    @property
    def processed_csv(self) -> Path:
        return self.path / "processed_data.csv"

    # Existence flags
    @property
    def has_validation(self) -> bool:
        return self.validation_file.exists()

    @property
    def has_stats(self) -> bool:
        return self.stats_file.exists()

    @property
    def has_log(self) -> bool:
        return self.log_file.exists()

    @property
    def report_files(self) -> List[Path]:
        return sorted(self.path.glob("*.pptx"))

    @property
    def has_report(self) -> bool:
        return bool(self.report_files)

    @property
    def figure_count(self) -> int:
        if not self.figures_dir.exists():
            return 0
        return len(list(self.figures_dir.glob("*.png")))

    @property
    def modified(self) -> datetime:
        try:
            return datetime.fromtimestamp(self.path.stat().st_mtime)
        except OSError:
            return datetime.min

    @property
    def status(self) -> str:
        """Coarse lifecycle status inferred from disk artifacts."""
        if self.has_report and self.has_stats:
            return "Complete"
        if self.has_stats or self.figure_count:
            return "Partial"
        if self.has_validation:
            return "Validated"
        return "Empty"

    # Validation payload
    def validation(self) -> dict:
        if self._validation is None:
            self._validation = _read_json(self.validation_file) or {}
        return self._validation

    @property
    def vehicle(self) -> str:
        """Best-effort vehicle name parsed from the report filename."""
        for rpt in self.report_files:
            m = re.match(r"WFT_Report_(.+)_(\d{8})", rpt.stem)
            if m:
                return m.group(1)
        return "—"

    @property
    def report_date(self) -> str:
        for rpt in self.report_files:
            m = re.match(r"WFT_Report_(.+)_(\d{8})", rpt.stem)
            if m:
                raw = m.group(2)
                try:
                    return datetime.strptime(raw, "%Y%m%d").strftime("%Y-%m-%d")
                except ValueError:
                    return raw
        return self.modified.strftime("%Y-%m-%d")

    # Stats payload
    def stats(self) -> Dict[str, ChannelStat]:
        if self._stats is None:
            raw = _read_json(self.stats_file) or {}
            parsed: Dict[str, ChannelStat] = {}
            for channel, vals in raw.items():
                if not isinstance(vals, dict):
                    continue
                parsed[channel] = ChannelStat(
                    channel=channel,
                    mean=_num(vals.get("mean")),
                    median=_num(vals.get("median")),
                    std=_num(vals.get("std")),
                    min=_num(vals.get("min")),
                    max=_num(vals.get("max")),
                    p80=_num(vals.get("P80")),
                    p90=_num(vals.get("P90")),
                    p95=_num(vals.get("P95")),
                )
            self._stats = parsed
        return self._stats

    # Channel configuration
    def run_channels(self):
        """The axle-dynamic channel config for THIS study, from its own stats.

        Wheel labels and component names cannot be assumed to be FL/FR/RL/RR x
        Fx/Fy/Fz: a recording may carry only two instrumented wheels, more than
        two axles, or channel names with an axle suffix (``FR_Fx_2``). The
        pipeline names its figures after the real channel, so the UI has to
        resolve them the same way the pipeline did.
        """
        if self._channels is None:
            from dtt.run_channels import build_run_channels
            self._channels = build_run_channels(list(self.stats().keys()))
        return self._channels

    def wheel_labels(self) -> List[str]:
        """Wheel/position labels actually present, in axle order."""
        return self.run_channels().labels

    def components(self) -> List[str]:
        """Force components actually present (usually Fx, Fy, Fz)."""
        rc = self.run_channels()
        seen: List[str] = []
        for comps in rc._label_comp.values():
            for c in comps:
                if c not in seen:
                    seen.append(c)
        return seen

    def channel_for(self, label: str, component: str) -> Optional[str]:
        """Real channel name for a wheel label + component, e.g. FR+Fx -> FR_Fx_2."""
        return self.run_channels().channel_for(label, component)

    # Figure lookups
    def figure(self, filename: str) -> Optional[Path]:
        p = self.figures_dir / filename
        return p if p.exists() else None

    def figures(self, pattern: str) -> List[Path]:
        if not self.figures_dir.exists():
            return []
        return sorted(self.figures_dir.glob(pattern))

    def histogram(self, wheel: str, signal: str, kind: str = "distance") -> Optional[Path]:
        """kind = 'distance' | 'percentage'. Per-channel histogram.

        The figure is named after the real channel, which is not always
        ``{wheel}_{signal}`` — resolve it before building the filename.
        """
        channel = self.channel_for(wheel, signal) or f"{wheel}_{signal}"
        return self.figure(f"hist_{kind}_{channel}.png")

    def histogram_combined(self, wheel: str, kind: str = "distance") -> Optional[Path]:
        return self.figure(f"hist_{kind}_{wheel}.png")

    def heatmap(self, name: str) -> Optional[Path]:
        return self.figure(f"heatmap_{name}.png")

    def hexbin(self, wheel: str) -> Optional[Path]:
        return self.figure(f"heatmap_hexbin_{wheel}.png")

    def boxplot(self, signal: str) -> Optional[Path]:
        return self.figure(f"boxplot_{signal}.png")

    def rainflow_image(self, wheel: str) -> Optional[Path]:
        return self.figure(f"rainflow_{wheel}.png")

    def rainflow_cycle_csvs(self) -> List[Path]:
        return self.figures("rainflow_cycles_*.csv")

    # Log
    def read_log(self) -> str:
        if not self.has_log:
            return ""
        try:
            return self.log_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""


# Repository

class StudyRepository:
    """Discovers and loads studies from the backend ``outputs`` directory."""

    def __init__(self, outputs_dir: Path = OUTPUTS_DIR, csv_dir: Path = CSV_DIR):
        self.outputs_dir = Path(outputs_dir)
        self.csv_dir = Path(csv_dir)

    # Discovery
    def list_studies(self) -> List[Study]:
        """All study folders, newest first."""
        if not self.outputs_dir.exists():
            return []
        studies = [
            Study(name=d.name, path=d)
            for d in self.outputs_dir.iterdir()
            if d.is_dir()
        ]
        studies.sort(key=lambda s: s.modified, reverse=True)
        return studies

    def delete_study(self, name: str) -> bool:
        """Permanently remove a study folder. Returns True on success."""
        import shutil
        d = self.outputs_dir / name
        if not d.is_dir():
            return False
        try:
            shutil.rmtree(d)
            return True
        except OSError:
            return False

    def get_study(self, name: str) -> Optional[Study]:
        d = self.outputs_dir / name
        return Study(name=name, path=d) if d.is_dir() else None

    def latest_study(self) -> Optional[Study]:
        studies = self.list_studies()
        return studies[0] if studies else None

    def list_csv_files(self) -> List[Path]:
        if not self.csv_dir.exists():
            return []
        return sorted(self.csv_dir.glob("*.csv"))

    # Aggregate dashboard metrics
    def latest_report(self) -> Optional[Path]:
        latest: Optional[Path] = None
        latest_mtime = -1.0
        for study in self.list_studies():
            for rpt in study.report_files:
                m = rpt.stat().st_mtime
                if m > latest_mtime:
                    latest_mtime = m
                    latest = rpt
        return latest


# Helpers

def _read_json(path: Path) -> Optional[dict]:
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_csv_head(path: Path, max_rows: int = 200) -> tuple[List[str], List[List[str]]]:
    """Read a CSV header + up to ``max_rows`` rows for preview tables."""
    header: List[str] = []
    rows: List[List[str]] = []
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh)
            header = next(reader, [])
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                rows.append(row)
    except OSError:
        pass
    return header, rows
