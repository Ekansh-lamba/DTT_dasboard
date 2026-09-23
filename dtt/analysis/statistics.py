import json
import logging
from typing import Dict

import numpy as np
import pandas as pd

from dtt.config import PERCENTILES, RunConfig

logger = logging.getLogger(__name__)


def rms(x: np.ndarray) -> float:
    """Root-mean-square of the finite values in ``x``, NaN if none."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(x ** 2)))


def channel_statistics(df: pd.DataFrame) -> Dict:
    """Statistics for every channel of ``df`` except Time.

    Keyed by the column's own name (the key every screen uses to find the
    column again), in the order of the FAMOS-style CSV export -- Time aside,
    A-Z by standard name -- with that standard ``name`` and the ``unit``
    alongside, so the Statistics screen, its CSV and the processed-data CSV
    list the same channels the same way. It used to cover the force channels
    only, under their raw names (``FR_Fx_2``).
    """
    from dtt.channel_names import channel_unit
    from dtt.famos_csv import export_columns, is_force

    results = {}
    for name, ch in export_columns(df.columns).items():
        if name.lower() == "time":
            continue
        series = pd.to_numeric(df[ch], errors="coerce").dropna()
        if len(series) == 0:
            continue
        vals = series.values
        entry = {
            "name":   name,
            "unit":   channel_unit(ch) or ("daN" if is_force(ch) else ""),
            "count":  int(len(vals)),
            "mean":   round(float(np.mean(vals)),   4),
            "median": round(float(np.median(vals)),  4),
            "std":    round(float(np.std(vals)),     4),
            "rms":    round(rms(vals),                4),
            "min":    round(float(np.min(vals)),     4),
            "max":    round(float(np.max(vals)),     4),
        }
        for p in PERCENTILES:
            entry[f"P{p}"] = round(float(np.percentile(vals, p)), 4)
        results[ch] = entry
    return results


def compute_statistics(df: pd.DataFrame, config: RunConfig) -> Dict:
    results = channel_statistics(df)

    out_path = config.run_output_dir / "stats_summary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Statistics saved: %s  (%d channels)", out_path, len(results))
    try:
        from dtt.famos_csv import stats_export_path, write_stats_csv
        write_stats_csv(results, stats_export_path(
            config.run_output_dir, config.vehicle_name, config.study_name))
    except Exception as exc:                                  # noqa: BLE001
        logger.error("Statistics CSV export failed: %s", exc, exc_info=True)
    return results
