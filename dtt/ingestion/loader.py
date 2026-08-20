import re
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from dtt.config import (
    MANDATORY_CHANNELS,
    N_TO_DAN_THRESHOLD,
    N_TO_DAN_FACTOR,
    EXCLUDE_KEYWORDS,
    TIME_COLUMN,
    DEFAULT_SAMPLING_RATE,
    RunConfig,
)

logger = logging.getLogger(__name__)


def _looks_like_time_series(arr) -> bool:
    try:
        vals = pd.to_numeric(arr, errors="coerce").dropna().values
        if len(vals) < 3:
            return False
        diffs = np.diff(vals)
        if np.nanmedian(np.abs(diffs)) == 0:
            return False
        pos_frac   = np.sum(diffs >= -1e-8) / len(diffs)
        med_abs    = np.nanmedian(np.abs(diffs))
        return pos_frac > 0.85 and med_abs < 1.0
    except Exception:
        return False


def _try_load_csv(path: str, skiprows, nrows: Optional[int]) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(
            path,
            encoding="latin1",
            nrows=nrows,
            skiprows=skiprows,
            skip_blank_lines=True,
            skipinitialspace=True,
            engine="c",
            low_memory=False,
        )
        return df
    except Exception:
        try:
            df = pd.read_csv(
                path,
                encoding="latin1",
                nrows=nrows,
                skiprows=skiprows,
                skip_blank_lines=True,
                skipinitialspace=True,
                engine="python",
            )
            return df
        except Exception:
            return None


def _rename_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    for i, c in enumerate(df.columns):
        s = str(c).strip()
        if s == "" or s.lower().startswith("unnamed"):
            try:
                if _looks_like_time_series(df.iloc[:, i]):
                    new_cols.append(TIME_COLUMN)
                else:
                    new_cols.append(f"Unnamed_{i}")
            except Exception:
                new_cols.append(f"Unnamed_{i}")
        else:
            new_cols.append(s)
    df.columns = new_cols
    return df


def _sanitize_header(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if len(df) == 0:
        return df
    first = df.iloc[0]
    non_num = 0
    total   = len(first)
    for v in first:
        if v is None:
            non_num += 1
            continue
        s = str(v).strip()
        try:
            float(s)
        except Exception:
            if not any(ch.isdigit() for ch in s):
                non_num += 1
    if non_num >= 0.3 * total:
        df = df.drop(df.index[0]).reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_channel_strict(df: pd.DataFrame, wheel: str, force: str) -> Optional[str]:
    if df is None:
        return None
    pat = re.compile(
        rf"^{re.escape(wheel)}[_\s\-]*{re.escape(force)}$",
        re.IGNORECASE,
    )
    for c in df.columns:
        name  = str(c).strip()
        lname = name.lower()
        if any(k in lname for k in EXCLUDE_KEYWORDS):
            continue
        if pat.match(name):
            return name
    return None


def _detect_force_channels(df: pd.DataFrame) -> list:
    found = []
    for ch in MANDATORY_CHANNELS:
        if ch in df.columns:
            found.append(ch)
    if not found:
        wheel_parts = ["FL", "FR", "RL", "RR"]
        force_parts = ["Fx", "Fy", "Fz"]
        for w in wheel_parts:
            for f in force_parts:
                strict = find_channel_strict(df, w, f)
                if strict:
                    found.append(strict)
    if not found:
        # Generalized detection for any axle layout (A1L_Fx, A3RO_Fz, …).
        from dtt.channels import discover, FORCE_COMPONENTS
        cs = discover([str(c) for c in df.columns])
        found = [orig for orig, (pos, comp) in cs.source_map.items()
                 if comp in FORCE_COMPONENTS]
    return found


def _apply_n_to_dan(df: pd.DataFrame, channels: list) -> Tuple[pd.DataFrame, bool]:
    if not channels:
        return df, False
    medians = [df[ch].abs().median() for ch in channels if ch in df.columns]
    if not medians:
        return df, False
    max_med = max(m for m in medians if not np.isnan(m))
    if max_med > N_TO_DAN_THRESHOLD:
        df = df.copy()
        for ch in channels:
            if ch in df.columns:
                df[ch] = df[ch] / N_TO_DAN_FACTOR
        logger.info("Applied N->daN conversion (div 10) to %d channels", len(channels))
        return df, True
    return df, False


def normalise_force_units(df: pd.DataFrame, config: RunConfig) -> Tuple[pd.DataFrame, dict]:
    """Correct a whole-decade unit error on the force channels, anchored on Fz.

    ``_apply_n_to_dan`` is a magnitude heuristic with a single fixed threshold,
    so it cannot tell N from decanewtons when the source has *also* scaled the
    data — an imc FAMOS ``CR`` block that multiplies the stored counts by 10
    leaves the frame exactly one decade high, the heuristic's one division
    brings it back only to newtons, and the result gets labelled daN. On the
    reference recording that reported a 6841 kg static front wheel and a 24.4 t
    passenger car.

    Static Fz is the one channel with a known answer: it is the corner weight,
    so it must land inside the vehicle preset's ``fz_range_dan``. Comparing the
    two gives the error as a power of ten. Only clean decades are corrected —
    anything else is a real mismatch between the recording and the chosen
    preset, which is the operator's call, not something to paper over.

    Moments are deliberately left alone: they carry their own ``CR`` factor and
    their own unit (Nm), and the ``My/Fx`` ratio comes out at a correct 0.27 m
    loaded radius once the *forces* alone are fixed.
    """
    rc = getattr(config, "run_channels", None)
    tyre = getattr(config, "tyre", None)
    if rc is None or tyre is None:
        return df, {}

    fz_cols = [orig for orig, (_, comp) in rc.channel_set.source_map.items()
               if comp == "Fz" and orig in df.columns]
    force_cols = [orig for orig, (_, comp) in rc.channel_set.source_map.items()
                  if comp in ("Fx", "Fy", "Fz") and orig in df.columns]
    if not fz_cols or not force_cols:
        return df, {}

    mags = []
    for ch in fz_cols:
        v = pd.to_numeric(df[ch], errors="coerce").abs()
        v = v[np.isfinite(v) & (v > 0)]
        if len(v):
            mags.append(float(v.median()))
    hi = float(tyre.fz_range_dan[1])
    if not mags or hi <= 0:
        return df, {}
    med = float(np.median(mags))

    # A static wheel load sits inside the preset band, not at its very edge;
    # 0.5 * hi is the reference point, so the ratio is a clean decade only when
    # the data really is a decade out.
    decades = int(round(np.log10(med / (0.5 * hi)))) if med > 0 else 0
    if decades == 0:
        return df, {}

    factor = 10.0 ** decades
    df = df.copy()
    for ch in force_cols:
        df[ch] = pd.to_numeric(df[ch], errors="coerce") / factor
    logger.warning(
        "Force channels were 10^%d out against the '%s' preset (median |Fz| "
        "%.0f vs expected band 0-%.0f daN) - divided %d force channels by %g",
        decades, tyre.tyre_type, med, hi, len(force_cols), factor)
    logger.info("Static |Fz| now %.1f daN (%.0f kg per wheel)",
                med / factor, med / factor * 10 / 9.81)
    return df, {"force_scale_divisor": factor, "force_scale_decades": decades,
                "force_channels_scaled": len(force_cols)}


def load_csv(path: Path, config: RunConfig) -> Tuple[pd.DataFrame, dict]:
    path_str = str(path)
    logger.info("Loading CSV: %s", path_str)

    df_raw = None
    for skip in (2, 1, 0, 3):
        candidate = _try_load_csv(path_str, skip, None)
        if candidate is not None:
            candidate = _rename_unnamed_columns(candidate)
            candidate = _sanitize_header(candidate)
            candidate.columns = [str(c).strip() for c in candidate.columns]
            candidate = candidate.apply(pd.to_numeric, errors="coerce")
            if _detect_force_channels(candidate):
                df_raw = candidate
                logger.info("CSV loaded with skiprows=%d", skip)
                break

    if df_raw is None:
        raise ValueError(
            f"Could not parse CSV: {path.name}. No force channels detected."
        )

    force_channels = _detect_force_channels(df_raw)
    df_raw         = df_raw.dropna(subset=force_channels, how="all")
    df_raw, dan_applied = _apply_n_to_dan(df_raw, force_channels)

    sr = config.sampling_rate
    if TIME_COLUMN in df_raw.columns:
        time_col = pd.to_numeric(df_raw[TIME_COLUMN], errors="coerce").dropna()
        if len(time_col) > 1:
            dt = float(np.nanmedian(np.diff(time_col.values)))
            if dt > 0:
                sr = round(1.0 / dt, 2)
                logger.info("Sampling rate detected from Time column: %.2f Hz", sr)

    duration_s = len(df_raw) / sr

    metadata = {
        "file_name":       path.name,
        "rows":            len(df_raw),
        "columns":         len(df_raw.columns),
        "force_channels":  force_channels,
        "sampling_rate_hz": sr,
        "duration_s":      round(duration_s, 2),
        "n_to_dan_applied": dan_applied,
    }

    logger.info(
        "Loaded: %d rows, %d columns, %.1f s, SR=%.0f Hz",
        len(df_raw), len(df_raw.columns), duration_s, sr,
    )
    return df_raw, metadata


def _finalize_raw(df, imeta, config, name) -> Tuple[pd.DataFrame, dict]:
    if df.empty:
        raise ValueError(f"No imc channels found: {name}")
    force_channels = _detect_force_channels(df)
    if not force_channels:
        raise ValueError(f"No force channels detected: {name}")

    df, dan_applied = _apply_n_to_dan(df, force_channels)
    sr = imeta.get("output_fs_hz", config.sampling_rate or DEFAULT_SAMPLING_RATE)

    metadata = {
        "file_name":        name,
        "rows":             len(df),
        "columns":          len(df.columns),
        "force_channels":   force_channels,
        "sampling_rate_hz": sr,
        "duration_s":       imeta.get("duration_s", round(len(df) / sr, 2) if sr else 0),
        "n_to_dan_applied": dan_applied,
        "raw_fs_hz":        imeta.get("raw_fs_hz"),
        "source_type":      "imc_raw",
        "famos_recipe":     imeta.get("famos_recipe") or {},
        "famos_decimate":   imeta.get("famos_decimate", 1),
        "deglitch":         imeta.get("deglitch", False),
        "despike":          imeta.get("despike", False),
        # Carried through so the pipeline can publish the study's "before".
        "raw_frame":        imeta.get("raw_frame"),
    }
    logger.info(
        "Loaded imc raw: %d rows, %d cols, raw %.0f Hz -> %.0f Hz, %d force channels",
        len(df), len(df.columns), imeta.get("raw_fs_hz", 0), sr, len(force_channels),
    )
    return df, metadata


def load_raw_folder(folder: Path, config: RunConfig) -> Tuple[pd.DataFrame, dict]:
    """Load an imc ``.raw`` channel folder (FAMOS-grade) as a DataFrame."""
    from dtt.ingestion.imc_reader import read_folder
    folder = Path(folder)
    logger.info("Loading imc raw folder: %s", folder)
    target = config.sampling_rate or DEFAULT_SAMPLING_RATE
    df, imeta = read_folder(folder, target_fs=target,
                            famos=getattr(config, "famos_mode", True),
                            deglitch=getattr(config, "deglitch", False),
                            despike=getattr(config, "despike", False),
                            despike_rail_min_run=getattr(config, "despike_rail_min_run", 3),
                            despike_dropout_max_run=getattr(config, "despike_dropout_max_run", 5),
                            despike_hw_cutoff_hz=getattr(config, "despike_hw_cutoff_hz", 200.0),
                            despike_net=getattr(config, "despike_net", False),
                            despike_net_nsigma=getattr(config, "despike_net_nsigma", 6.0),
                            despike_net_window_s=getattr(config, "despike_net_window_s", 0.011))
    return _finalize_raw(df, imeta, config, folder.name)


def load_raw_files(files, config: RunConfig) -> Tuple[pd.DataFrame, dict]:
    """Load a specific subset of imc ``.raw`` files as a DataFrame."""
    from dtt.ingestion.imc_reader import read_files
    files = [Path(f) for f in files]
    logger.info("Loading %d imc raw files", len(files))
    target = config.sampling_rate or DEFAULT_SAMPLING_RATE
    df, imeta = read_files(files, target_fs=target,
                           famos=getattr(config, "famos_mode", True),
                           deglitch=getattr(config, "deglitch", False),
                           despike=getattr(config, "despike", False),
                           despike_rail_min_run=getattr(config, "despike_rail_min_run", 3),
                           despike_dropout_max_run=getattr(config, "despike_dropout_max_run", 5),
                           despike_hw_cutoff_hz=getattr(config, "despike_hw_cutoff_hz", 200.0),
                           despike_net=getattr(config, "despike_net", False),
                           despike_net_nsigma=getattr(config, "despike_net_nsigma", 6.0),
                           despike_net_window_s=getattr(config, "despike_net_window_s", 0.011))
    return _finalize_raw(df, imeta, config, f"{len(files)} files")
