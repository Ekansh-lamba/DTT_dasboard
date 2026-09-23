"""Write a processed study as a FAMOS-style CSV, laid out like the team's
reference export ``csv/RLDA WFT PV data sample.csv``.

That file is what imc FAMOS produces and what the team's WFT Analyzer and
spreadsheets are built around, so the processed output comes off in the same
shape:

* two blank lines, then the header row, then data -- CRLF line endings;
* every field right-aligned in 16 characters, comma separated;
* ``Time`` first, every other channel in A-Z order (``Altitude, Dist,
  FL_Fx .. FL_Mz, FL_WS1, FR_.., Latacc, ...``);
* wheel channels under their standard names (``FR_Fx_2`` -> ``FR_Fx``, see
  :mod:`dtt.channel_names`) and ``Distance`` as ``Dist``;
* numbers as FAMOS prints them: 6 significant digits, trailing zeros dropped,
  ``x.0`` for a whole value, exponent form below 0.1 (``9.06273e-2``,
  ``1.0e-2``), a missing sample as an empty field;
* forces in **N** and moments as recorded, the units of the reference file.
  The platform works in daN internally (``processed_data.csv`` stays that
  way); the export multiplies forces by 10 so a reader that expects the FAMOS
  export -- the WFT Analyzer divides by 10 itself when the median force is
  above 1000 -- gets the numbers it expects.

The platform's own ``processed_data.csv`` is unchanged: every screen and the
analysis-only mode read it, and it is the file the FAMOS cross-check compares.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from dtt.channel_names import canonical_name
from dtt.channels import parse_channel

logger = logging.getLogger(__name__)

FIELD_WIDTH = 16
DAN_TO_N = 10.0
_RENAME = {"distance": "Dist"}
_CHUNK_ROWS = 50_000
# Windows ANSI, as FAMOS writes it: Excel and the WFT Analyzer (latin1) read
# "m/s²" and "daN·m" correctly; UTF-8 would show them as "m/sÂ²".
ENCODING = "cp1252"


def famos_number(x) -> str:
    """One value the way the FAMOS CSV export prints it."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(x):
        return ""
    if x == 0.0:
        return "-0.0" if math.copysign(1.0, x) < 0 else "0.0"
    ax = abs(x)
    if ax < 0.1 or ax >= 1e6:
        mant, exp = f"{x:.5e}".split("e")
        mant = mant.rstrip("0").rstrip(".")
        if "." not in mant:
            mant += ".0"
        return f"{mant}e{int(exp)}"
    s = f"{x:.6g}"
    if "." not in s and x.is_integer():
        s += ".0"
    return s


def export_name(column: str) -> str:
    """The column's name in the export: standard wheel names, ``Dist``."""
    name = canonical_name(column)
    return _RENAME.get(name.lower(), name)


def is_force(column: str) -> bool:
    parsed = parse_channel(str(column))
    return parsed is not None and parsed[1] in ("Fx", "Fy", "Fz")


def export_columns(columns: Iterable[str]) -> Dict[str, str]:
    """``{export name: source column}``, Time first then A-Z.

    Two source spellings of one channel (an imc ``- Copy`` duplicate) keep
    the first; the export has one column per channel.
    """
    mapping: Dict[str, str] = {}
    for col in columns:
        name = export_name(str(col))
        if name not in mapping:
            mapping[name] = col
    time = next((n for n in mapping if n.lower() == "time"), None)
    ordered = ([time] if time else []) + sorted(n for n in mapping if n != time)
    return {n: mapping[n] for n in ordered}


def _format_column(values: np.ndarray) -> List[str]:
    w = FIELD_WIDTH
    return [f"{famos_number(v):>{w}}" for v in values]


def write_famos_csv(df: pd.DataFrame, path: Path, forces_in_newton: bool = True,
                    columns: Optional[Dict[str, str]] = None) -> Path:
    """Write ``df`` (platform units, forces in daN) in the reference layout."""
    path = Path(path)
    cols = columns or export_columns(df.columns)
    scale = {name: (DAN_TO_N if forces_in_newton and is_force(src) else 1.0)
             for name, src in cols.items()}
    header = ",".join(f"{name:>{FIELD_WIDTH}}" for name in cols)
    n = len(df)
    with open(path, "w", encoding=ENCODING, errors="replace", newline="") as fh:
        fh.write("\r\n\r\n" + header + "\r\n")
        for start in range(0, n, _CHUNK_ROWS):
            stop = min(n, start + _CHUNK_ROWS)
            cells = []
            for name, src in cols.items():
                v = pd.to_numeric(df[src].iloc[start:stop], errors="coerce").to_numpy(float)
                if scale[name] != 1.0:
                    v = v * scale[name]
                cells.append(_format_column(v))
            fh.write("\r\n".join(",".join(row) for row in zip(*cells)))
            fh.write("\r\n")
    logger.info("FAMOS-style CSV saved: %s  (%d rows x %d channels%s)", path, n,
                len(cols), ", forces in N" if forces_in_newton else "")
    return path


def processed_export_path(run_output_dir: Path, vehicle: str, study: str) -> Path:
    return Path(run_output_dir) / f"WFT_Processed_{vehicle}_{study}.csv"


# ----------------------------------------------------------------- statistics

STATS_COLUMNS = ["Channel", "Unit", "Count", "Mean", "Median", "Std", "RMS",
                 "Min", "Max", "P80", "P90", "P95"]
_STATS_KEYS = [None, None, "count", "mean", "median", "std", "rms",
               "min", "max", "P80", "P90", "P95"]


def write_stats_csv(stats: Dict[str, dict], path: Path) -> Path:
    """The per-channel statistics in the same layout as the data export.

    One row per channel, in the export's channel order and names. Values are
    in the platform's units, stated in the Unit column (forces daN) -- the
    same numbers the Statistics screen and the report show.
    """
    path = Path(path)
    w = FIELD_WIDTH
    lines = ["", "", ",".join(f"{c:>{w}}" for c in STATS_COLUMNS)]
    for src, entry in stats.items():
        cells = []
        for col, key in zip(STATS_COLUMNS, _STATS_KEYS):
            if col == "Channel":
                cells.append(entry.get("name") or export_name(src))
            elif col == "Unit":
                cells.append(entry.get("unit", ""))
            else:
                cells.append(famos_number(entry.get(key)))
        lines.append(",".join(f"{c:>{w}}" for c in cells))
    with open(path, "w", encoding=ENCODING, errors="replace", newline="") as fh:
        fh.write("\r\n".join(lines) + "\r\n")
    logger.info("Statistics CSV saved: %s  (%d channels)", path, len(stats))
    return path


def stats_export_path(run_output_dir: Path, vehicle: str, study: str) -> Path:
    return Path(run_output_dir) / f"WFT_Statistics_{vehicle}_{study}.csv"
