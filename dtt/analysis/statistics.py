import json
import logging
from typing import Dict

import numpy as np
import pandas as pd

from dtt.config import MANDATORY_CHANNELS, PERCENTILES, RunConfig

logger = logging.getLogger(__name__)


def compute_statistics(df: pd.DataFrame, config: RunConfig) -> Dict:
    if config.run_channels is not None:
        channels = [ch for ch in config.run_channels.mandatory_channels if ch in df.columns]
    else:
        channels = [ch for ch in MANDATORY_CHANNELS if ch in df.columns]
    results  = {}

    for ch in channels:
        series = pd.to_numeric(df[ch], errors="coerce").dropna()
        if len(series) == 0:
            continue
        vals = series.values
        entry = {
            "mean":   round(float(np.mean(vals)),   4),
            "median": round(float(np.median(vals)),  4),
            "std":    round(float(np.std(vals)),     4),
            "min":    round(float(np.min(vals)),     4),
            "max":    round(float(np.max(vals)),     4),
        }
        for p in PERCENTILES:
            entry[f"P{p}"] = round(float(np.percentile(vals, p)), 4)
        results[ch] = entry

    out_path = config.run_output_dir / "stats_summary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Statistics saved: %s", out_path)
    return results
