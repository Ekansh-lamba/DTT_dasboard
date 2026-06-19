import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from dtt.config import MANDATORY_CHANNELS, OPTIONAL_MOMENT_CHANNELS, RunConfig

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    file_name:          str
    rows:               int
    columns:            int
    duration_s:         float
    sampling_rate_hz:   float
    channel_status:     Dict[str, str]  = field(default_factory=dict)
    missing_channels:   List[str]       = field(default_factory=list)
    optional_present:   List[str]       = field(default_factory=list)
    duplicate_columns:  List[str]       = field(default_factory=list)
    nan_counts:         Dict[str, int]  = field(default_factory=dict)
    total_nan_rows:     int             = 0
    is_valid:           bool            = True
    warnings:           List[str]       = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "file_name":        self.file_name,
            "rows":             self.rows,
            "columns":          self.columns,
            "duration_s":       self.duration_s,
            "sampling_rate_hz": self.sampling_rate_hz,
            "channel_status":   self.channel_status,
            "missing_channels": self.missing_channels,
            "optional_present": self.optional_present,
            "duplicate_columns": self.duplicate_columns,
            "nan_counts":       self.nan_counts,
            "total_nan_rows":   self.total_nan_rows,
            "is_valid":         self.is_valid,
            "warnings":         self.warnings,
        }


def validate(df: pd.DataFrame, metadata: dict, config: RunConfig) -> ValidationReport:
    report = ValidationReport(
        file_name        = metadata.get("file_name", ""),
        rows             = metadata.get("rows", len(df)),
        columns          = metadata.get("columns", len(df.columns)),
        duration_s       = metadata.get("duration_s", 0.0),
        sampling_rate_hz = metadata.get("sampling_rate_hz", config.sampling_rate),
    )

    col_list = list(df.columns)
    seen     = {}
    for c in col_list:
        seen[c] = seen.get(c, 0) + 1
    report.duplicate_columns = [c for c, n in seen.items() if n > 1]
    if report.duplicate_columns:
        w = f"Duplicate columns: {report.duplicate_columns}"
        report.warnings.append(w)
        logger.warning(w)

    for ch in MANDATORY_CHANNELS:
        if ch in df.columns:
            report.channel_status[ch] = "PRESENT"
        else:
            report.channel_status[ch] = "MISSING"
            report.missing_channels.append(ch)

    for ch in OPTIONAL_MOMENT_CHANNELS:
        if ch in df.columns:
            report.optional_present.append(ch)

    if report.missing_channels:
        report.is_valid = False
        msg = f"Missing mandatory channels: {report.missing_channels}"
        report.warnings.append(msg)
        logger.warning(msg)

    for ch in MANDATORY_CHANNELS:
        if ch in df.columns:
            n = int(df[ch].isna().sum())
            if n > 0:
                report.nan_counts[ch] = n

    all_force = [ch for ch in MANDATORY_CHANNELS if ch in df.columns]
    if all_force:
        report.total_nan_rows = int(df[all_force].isna().all(axis=1).sum())

    out_path = config.run_output_dir / "validation_report.json"
    with open(out_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info("Validation report saved: %s", out_path)

    status_lines = []
    for ch, st in report.channel_status.items():
        symbol = "[OK]" if st == "PRESENT" else "[MISSING]"
        status_lines.append(f"  {symbol} {ch} {st}")
    logger.info("Channel validation:\n%s", "\n".join(status_lines))

    return report
