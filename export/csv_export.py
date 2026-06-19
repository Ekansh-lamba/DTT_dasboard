"""
export/csv_export.py
====================
CSV export for all DTT WFT engine results.

Produces flat, readable CSVs for:
  - Force histograms (bin centres + count + distance km per channel)
  - Trip statistics (overview + force summary + accel summary + wheel speeds)
  - Force severity bands
  - Fatigue summary (damage ratios, cycle counts, P95 ranges)

Also provides export_all_csv_bytes() which bundles everything into a
ZIP archive returned as bytes (for Streamlit st.download_button).
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def export_all_csv_bytes(
    hist_results: dict = None,
    trip_stats: dict = None,
    severity_results: dict = None,
    fatigue_results: dict = None,
) -> bytes:
    """
    Bundle all available engine results into a ZIP archive of CSVs.

    Returns bytes suitable for st.download_button.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if hist_results:
            df = _histogram_to_df(hist_results)
            zf.writestr("histogram_results.csv", df.to_csv(index=False))
            logger.info("CSV export: histogram_results.csv (%d rows)", len(df))

        if trip_stats:
            from engine.trip_stats import stats_to_dataframe
            dfs = stats_to_dataframe(trip_stats)
            for name, df in dfs.items():
                if not df.empty:
                    fname = f"trip_{name}.csv"
                    zf.writestr(fname, df.to_csv(index=False))
                    logger.info("CSV export: %s (%d rows)", fname, len(df))

        if severity_results:
            df = _severity_to_df(severity_results)
            zf.writestr("force_severity.csv", df.to_csv(index=False))
            logger.info("CSV export: force_severity.csv (%d rows)", len(df))

        if fatigue_results:
            df = _fatigue_to_df(fatigue_results)
            zf.writestr("fatigue_summary.csv", df.to_csv(index=False))
            logger.info("CSV export: fatigue_summary.csv (%d rows)", len(df))

    buf.seek(0)
    return buf.read()


def export_all_csv_dir(
    output_dir: str | Path,
    hist_results: dict = None,
    trip_stats: dict = None,
    severity_results: dict = None,
    fatigue_results: dict = None,
) -> list[Path]:
    """
    Write all available engine results as individual CSV files to output_dir.

    Returns list of written file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if hist_results:
        path = output_dir / "histogram_results.csv"
        _histogram_to_df(hist_results).to_csv(path, index=False)
        written.append(path)
        logger.info("CSV written: %s", path)

    if trip_stats:
        from engine.trip_stats import stats_to_dataframe
        dfs = stats_to_dataframe(trip_stats)
        for name, df in dfs.items():
            if not df.empty:
                path = output_dir / f"trip_{name}.csv"
                df.to_csv(path, index=False)
                written.append(path)
                logger.info("CSV written: %s", path)

    if severity_results:
        path = output_dir / "force_severity.csv"
        _severity_to_df(severity_results).to_csv(path, index=False)
        written.append(path)

    if fatigue_results:
        path = output_dir / "fatigue_summary.csv"
        _fatigue_to_df(fatigue_results).to_csv(path, index=False)
        written.append(path)

    return written


# ─── Converters ────────────────────────────────────────────────────────────

def _histogram_to_df(hist_results: dict) -> pd.DataFrame:
    rows = []
    for key, res in hist_results.items():
        if not isinstance(res, dict) or "bin_centers" not in res:
            continue
        for bc, hc, hd in zip(
            res["bin_centers"],
            res["hist_count"],
            res["hist_dist_km"],
        ):
            rows.append({
                "channel":        key,
                "wheel":          res.get("wheel", ""),
                "suffix":         res.get("suffix", ""),
                "display_unit":   res.get("display_unit", ""),
                "bin_center":     round(float(bc), 4),
                "count":          int(hc),
                "distance_km":    round(float(hd), 6),
            })
    # Percentiles as separate rows
    pct_rows = []
    for key, res in hist_results.items():
        for p, pval in res.get("percentiles", {}).items():
            pct_rows.append({
                "channel":      key,
                "percentile":   f"P{p}",
                "value":        round(float(pval), 4),
                "display_unit": res.get("display_unit", ""),
            })
    main_df = pd.DataFrame(rows)
    return main_df


def _severity_to_df(severity_results: dict) -> pd.DataFrame:
    rows = []
    for wheel, data in severity_results.items():
        row = {"wheel": wheel}
        for band, val in data.get("fz_bands", {}).items():
            row[f"Fz_{band}"] = round(float(val), 2)
        for band, frac in data.get("fz_band_fractions", {}).items():
            row[f"frac_{band}_pct"] = round(float(frac) * 100, 2)
        for suffix in ("fx", "fy"):
            stats = data.get(f"{suffix}_stats", {})
            for stat, val in stats.items():
                row[f"{suffix}_{stat}"] = round(float(val), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def _fatigue_to_df(fatigue_results: dict) -> pd.DataFrame:
    rows = []
    channels = fatigue_results.get("channels", {})
    for key, res in channels.items():
        rows.append({
            "channel":        key,
            "wheel":          res.get("wheel", ""),
            "suffix":         res.get("suffix", ""),
            "total_cycles":   res.get("total_cycles", 0),
            "damage_index":   res.get("damage", 0.0),
            "p95_range":      res.get("p95_range", 0.0),
            "p90_range":      res.get("p90_range", 0.0),
            "max_range":      res.get("max_range", 0.0),
            "mean_range":     res.get("mean_range", 0.0),
            "miner_m":        res.get("miner_m", 5),
            "display_unit":   res.get("display_unit", ""),
        })

    axles = fatigue_results.get("axles", {})
    for suffix, axle_data in axles.items():
        ratio = axle_data.get("damage_ratio_rear_over_front")
        rows.append({
            "channel":   f"AXLE_RATIO_{suffix}",
            "wheel":     "Front÷Rear",
            "suffix":    suffix,
            "damage_index": ratio if ratio is not None else float("nan"),
            "miner_m":   axle_data.get("miner_m", 5),
        })

    return pd.DataFrame(rows)
