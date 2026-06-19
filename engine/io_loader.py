"""
engine/io_loader.py
===================
Memory-safe, chunked CSV loading for the DTT WFT engine.

Key design rules (per §5 of the build spec):
  1. Never call pd.read_csv without a chunksize on a file > 100 MB.
  2. Downcast force/moment/accel columns to float32 after load (halves RAM).
  3. Rebuild the time axis from row_index × dt; do NOT trust the stored Time
     column for indexing or dt derivation past ~10,000 s.
  4. Detect sample rate dynamically from data (first clean rows); never hard-code 100.
  5. Validate column presence, NaN rates, and Fz plausibility — warn, don't fix.

Anti-patterns explicitly avoided:
  - No pd.read_excel fallback (truncates at ~1.05M rows).
  - No full-file read_csv(low_memory=False) without chunk strategy.
  - No hard-coded fs=100 Hz.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Header detection ──────────────────────────────────────────────────────

def detect_skiprows(path: str | Path, known_column_keywords: list[str]) -> int:
    """
    Auto-detect how many rows to skip before the header row.

    Strategy (ported from WFT_Analyzer_v5._parse_csv):
      Try skiprows = 2, 1, 0, 3 in that order.
      The first attempt that yields at least 2 recognisable channel names wins.

    Parameters
    ----------
    path : CSV file path
    known_column_keywords : list of substrings to look for in header
        (e.g. ['Fx', 'Fy', 'Fz', 'Time', 'Dist', 'Speed'])

    Returns
    -------
    int — the number of rows to skip (0, 1, 2, or 3)
    """
    kw_lower = [k.lower() for k in known_column_keywords]

    for skip in (2, 1, 0, 3):
        try:
            header_df = pd.read_csv(
                path,
                encoding="latin1",
                skiprows=skip,
                nrows=0,        # read only the header row
                skip_blank_lines=True,
                engine="c",
            )
            cols = [str(c).strip().lower() for c in header_df.columns]
            hits = sum(
                1 for col in cols
                if any(kw in col for kw in kw_lower)
            )
            if hits >= 2:
                logger.info("detect_skiprows: using skiprows=%d (%d channel hits)", skip, hits)
                return skip
        except Exception:
            continue

    logger.warning("detect_skiprows: could not auto-detect; defaulting to skiprows=2")
    return 2


# ─── Chunked CSV loading ───────────────────────────────────────────────────

def load_csv_chunked(
    path: str | Path,
    cfg: dict,
    usecols: Optional[list[str]] = None,
    skiprows: Optional[int] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Load a large CSV in chunks and concatenate into a single DataFrame.

    Memory optimisations (per refinement §1 of the approved plan):
      - Reads in chunks of `cfg.loading.chunksize` rows (default 200k)
      - Downcasts force / moment / accel columns to float32 after load
        (halves memory: ~730 MB float64 → ~365 MB float32)
      - Keeps Time and GPS columns as float64 for precision

    Parameters
    ----------
    path    : Path to the CSV file
    cfg     : Parsed defaults.yaml dict
    usecols : Optional list of column names to load (None = all)
    skiprows: Override for header skip detection

    Returns
    -------
    (df, meta) where:
      df   : DataFrame with stripped column names and rebuilt time axis
      meta : dict with {fs, dt, n_rows, duration_s, total_dist_km, warnings, load_time_s}
    """
    path = Path(path)
    warnings: list[str] = []
    t0 = time.perf_counter()

    # ── Step 1: detect header position ────────────────────────────────────
    header_kw = ["Fx", "Fy", "Fz", "Time", "Dist", "Speed", "Acc", "Lat", "Lon"]
    skip = skiprows if skiprows is not None else detect_skiprows(path, header_kw)

    chunksize = cfg.get("loading", {}).get("chunksize", 200_000)

    # ── Step 2: stream chunks ──────────────────────────────────────────────
    logger.info("Loading '%s' in chunks of %d rows …", path.name, chunksize)
    chunks: list[pd.DataFrame] = []
    total_rows = 0

    try:
        reader = pd.read_csv(
            path,
            encoding="latin1",
            skiprows=skip,
            skip_blank_lines=True,
            skipinitialspace=True,
            engine="c",
            chunksize=chunksize,
            usecols=usecols,
            dtype=str,          # read everything as str first → safe parse
            low_memory=False,
        )
        for chunk in reader:
            chunk.columns = [str(c).strip() for c in chunk.columns]
            # Convert to numeric (non-convertible → NaN)
            chunk = chunk.apply(pd.to_numeric, errors="coerce")
            chunks.append(chunk)
            total_rows += len(chunk)
            if total_rows % 500_000 == 0:
                logger.info("  … loaded %s rows so far", f"{total_rows:,}")
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV '{path}': {exc}") from exc

    if not chunks:
        raise RuntimeError("No data rows found in CSV after parsing.")

    logger.info("Concatenating %d chunks (%s rows total) …", len(chunks), f"{total_rows:,}")
    df = pd.concat(chunks, ignore_index=True)

    # ── Step 3: downcast to float32 where safe ─────────────────────────────
    df = _downcast_columns(df, cfg, warnings)

    # ── Step 4: detect sample rate & rebuild time axis ─────────────────────
    dt, fs = detect_sample_rate(df, cfg)
    df = rebuild_time_axis(df, dt, cfg, warnings)

    # ── Step 5: compute meta ───────────────────────────────────────────────
    duration_s = total_rows / fs if fs > 0 else 0.0
    total_dist_km = _compute_total_distance(df, cfg)
    load_time_s = time.perf_counter() - t0

    meta = {
        "fs": fs,
        "dt": dt,
        "n_rows": total_rows,
        "duration_s": duration_s,
        "total_dist_km": total_dist_km,
        "skiprows_used": skip,
        "warnings": warnings,
        "load_time_s": load_time_s,
    }

    logger.info(
        "Load complete: %s rows | fs=%.1f Hz | duration=%.0f s | dist=%.1f km | %.1f s elapsed",
        f"{total_rows:,}", fs, duration_s, total_dist_km, load_time_s,
    )
    return df, meta


def _downcast_columns(df: pd.DataFrame, cfg: dict, warnings: list[str]) -> pd.DataFrame:
    """
    Downcast force/moment/accel/speed/yaw/dist columns to float32.
    Keeps Time and GPS (lat/lon/alt) as float64 for precision.
    Modifies df in-place and returns it.
    """
    # Identify float32 candidates: any column whose name looks like a force/
    # moment / IMU / dist channel (NOT time, lat, lon, alt)
    float64_keywords = {"time", "latitude", "longitude", "altitude", "lat", "lon", "alt"}

    for col in df.select_dtypes(include="float64").columns:
        lname = col.lower().strip()
        if any(kw in lname for kw in float64_keywords):
            continue  # keep float64
        try:
            df[col] = df[col].astype(np.float32)
        except Exception:
            pass  # leave as float64 on failure

    return df


# ─── Sample-rate detection ─────────────────────────────────────────────────

def detect_sample_rate(df: pd.DataFrame, cfg: dict) -> tuple[float, float]:
    """
    Detect the sample interval dt and sample rate fs from the Time column.

    Uses the early portion of the file (first N rows, where N comes from config)
    because the Time column degrades in precision past ~10,000 s.

    Returns
    -------
    (dt, fs) — dt in seconds, fs in Hz
    """
    n_early = cfg.get("sample_rate", {}).get("detect_from_n_early_rows", 5000)
    time_col_name = cfg.get("sample_rate", {}).get("time_col", "Time")

    if time_col_name not in df.columns:
        logger.warning(
            "Time column '%s' not found; defaulting to dt=0.01 s (100 Hz).", time_col_name
        )
        return 0.01, 100.0

    time_vals = pd.to_numeric(df[time_col_name].iloc[:n_early], errors="coerce").dropna().values

    if len(time_vals) < 3:
        logger.warning("Too few valid Time values to detect fs; defaulting to 100 Hz.")
        return 0.01, 100.0

    diffs = np.diff(time_vals)
    # Use median to be robust against any early-file anomalies
    dt = float(np.median(diffs[diffs > 0]))

    if dt <= 0 or not np.isfinite(dt):
        logger.warning("Detected dt=%.6f is invalid; defaulting to dt=0.01 s.", dt)
        return 0.01, 100.0

    fs = 1.0 / dt
    logger.info("Detected sample rate: dt=%.6f s → fs=%.2f Hz", dt, fs)
    return dt, fs


# ─── Time axis rebuild ─────────────────────────────────────────────────────

def rebuild_time_axis(
    df: pd.DataFrame,
    dt: float,
    cfg: dict,
    warnings: list[str],
) -> pd.DataFrame:
    """
    Add column 't_rebuilt' = row_index × dt.

    Cross-check: compare t_rebuilt against the stored Time column in the early
    (clean) region.  If they diverge by more than the configured tolerance, emit
    a WARNING (not an assertion error — per refinement §2 of the approved plan).

    The stored Time column is kept as-is for reference; t_rebuilt is the
    authoritative time axis for all engine computations.
    """
    n_rows = len(df)
    df["t_rebuilt"] = (np.arange(n_rows, dtype=np.float64)) * dt

    # Cross-check in the clean early region only
    time_col_name = cfg.get("sample_rate", {}).get("time_col", "Time")
    tol_frac = cfg.get("sample_rate", {}).get("crosscheck_tolerance_frac", 0.15)
    n_check = cfg.get("sample_rate", {}).get("detect_from_n_early_rows", 5000)

    if time_col_name in df.columns:
        stored_early = pd.to_numeric(
            df[time_col_name].iloc[:n_check], errors="coerce"
        ).dropna()
        rebuilt_early = df["t_rebuilt"].iloc[stored_early.index]

        if len(stored_early) >= 10:
            offset = float(stored_early.iloc[0])  # handle non-zero start time
            relative_stored = stored_early.values - offset
            relative_rebuilt = rebuilt_early.values - rebuilt_early.iloc[0]

            with np.errstate(divide="ignore", invalid="ignore"):
                rel_err = np.abs(relative_stored - relative_rebuilt) / (
                    np.abs(relative_rebuilt) + 1e-9
                )

            max_err = float(np.nanpercentile(rel_err, 95))
            if max_err > tol_frac:
                msg = (
                    f"Rebuilt time axis diverges from stored '{time_col_name}' by "
                    f"{max_err*100:.1f}% (P95) in the early region — "
                    f"this may indicate a non-standard sampling pattern. "
                    f"t_rebuilt = row_index × {dt:.6f} s is used as the authoritative axis."
                )
                logger.warning(msg)
                warnings.append(msg)
            else:
                logger.info(
                    "Time cross-check passed: rebuilt vs stored diverges by ≤ %.1f%% (P95)",
                    max_err * 100,
                )

    logger.info("Time axis rebuilt: t_rebuilt = row_index × %.6f s (fs=%.2f Hz)", dt, 1/dt)
    return df


# ─── Data validation gate ──────────────────────────────────────────────────

def validate_data(df: pd.DataFrame, chan_map: dict, cfg: dict) -> list[str]:
    """
    Light validation gate (NOT re-cleaning — data is already FAMOS-processed).

    Checks:
      1. Expected columns present (warn for missing ones)
      2. NaN fraction per channel (warn if > cfg threshold)
      3. Fz plausibility range (warn if median outside physical bounds)

    Returns
    -------
    List of warning strings (empty if all checks pass).
    """
    warnings: list[str] = []
    val_cfg = cfg.get("validation", {})
    max_nan = val_cfg.get("max_nan_fraction", 0.05)
    fz_range = val_cfg.get("fz_range_n", [500, 60000])

    n_rows = len(df)

    # ── Check 1: missing channels ──────────────────────────────────────────
    critical = ["time", "dist", "speed"]
    for key in critical:
        if chan_map.get(key) is None:
            msg = f"Critical channel '{key}' not found in CSV."
            logger.warning(msg)
            warnings.append(msg)

    # ── Check 2: NaN rates ─────────────────────────────────────────────────
    for internal_key, col_name in chan_map.items():
        if col_name is None or col_name not in df.columns:
            continue
        nan_frac = df[col_name].isna().sum() / n_rows
        if nan_frac > max_nan:
            msg = (
                f"Channel '{internal_key}' ({col_name}) has {nan_frac*100:.1f}% NaN values "
                f"(threshold: {max_nan*100:.0f}%)."
            )
            logger.warning(msg)
            warnings.append(msg)

    # ── Check 3: Fz plausibility ───────────────────────────────────────────
    from .channel_map import WHEELS
    for wheel in WHEELS:
        fz_key = f"{wheel}_Fz"
        col = chan_map.get(fz_key)
        if col is None or col not in df.columns:
            continue
        fz_vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(fz_vals) == 0:
            continue
        fz_median = float(fz_vals.median())
        if fz_median < fz_range[0] or fz_median > fz_range[1]:
            msg = (
                f"{fz_key} median = {fz_median:.0f} N, which is outside the "
                f"expected range [{fz_range[0]}, {fz_range[1]}] N. "
                f"Check units or sign convention."
            )
            logger.warning(msg)
            warnings.append(msg)

    if not warnings:
        logger.info("Data validation passed — no issues found.")

    return warnings


# ─── Utility ───────────────────────────────────────────────────────────────

def _compute_total_distance(df: pd.DataFrame, cfg: dict) -> float:
    """
    Compute total distance from the cumulative Dist column.
    Returns distance in km.
    """
    dist_col = None
    for cname in ("Dist", "Distance"):
        if cname in df.columns:
            dist_col = cname
            break

    if dist_col is None:
        return 0.0

    dist_vals = pd.to_numeric(df[dist_col], errors="coerce").dropna()
    if len(dist_vals) < 2:
        return 0.0

    total_m = float(dist_vals.iloc[-1] - dist_vals.iloc[0])
    return total_m / 1000.0


def get_column_or_warn(
    df: pd.DataFrame,
    internal_key: str,
    chan_map: dict,
    warnings: list[str],
) -> Optional[pd.Series]:
    """
    Return df[col] for the channel mapped to internal_key, or None with a warning.
    """
    col = chan_map.get(internal_key)
    if col is None:
        msg = f"Channel '{internal_key}' not found in channel map."
        logger.debug(msg)
        return None
    if col not in df.columns:
        msg = f"Channel '{internal_key}' mapped to '{col}' but column not in DataFrame."
        logger.warning(msg)
        warnings.append(msg)
        return None
    return df[col]
