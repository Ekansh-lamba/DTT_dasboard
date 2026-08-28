import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from dtt.config import RunConfig, TIME_COLUMN, N_TO_DAN_FACTOR
from dtt.ingestion.loader import load_csv
from dtt.validation.validator import validate
from dtt.sanitization.sanitizer import sanitize
from dtt.processing.signal_processor import apply_filter
from dtt.analysis.statistics import compute_statistics
from dtt.analysis.histograms import generate_histograms
from dtt.analysis.heatmaps import generate_heatmaps
from dtt.analysis.boxplots import generate_boxplots
from dtt.analysis.rainflow import generate_rainflow
from dtt.reporting.report_builder import build_report

logger = logging.getLogger("pipeline")

# Full float repr writes ~18 characters for a value the WFT resolved to about
# five: processed_data.csv came to 269 MB, and pandas has to scan every one of
# those bytes to pull a single column, so the GUI paid 2.5 s per channel switch.
# Six significant figures halves the file and the read time, and costs 5e-3 daN
# on values around 800 - four orders below the sensor's own noise floor.
CSV_FLOAT_FORMAT = "%.6g"


def _setup_logging(config: RunConfig) -> None:
    log_path = config.logs_dir / "pipeline.log"
    fmt      = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    # The GUI reads our stdout as UTF-8 to drive its live log. Left alone, a
    # frozen build writes the Windows ANSI codepage and every en/em-dash in a
    # log line arrives as a replacement char. PYTHONIOENCODING does not help
    # here — the PyInstaller bootloader ignores it — so set it on the stream.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass                                   # None or already-closed stream
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )



def _force_frame(df, config):
    """Time plus the force channels only — all the before/after view can plot."""
    rc = getattr(config, "run_channels", None)
    if rc is None:
        return None
    cols = [c for c in rc.mandatory_channels if c in df.columns]
    if not cols:
        return None
    keep = ([TIME_COLUMN] if TIME_COLUMN in df.columns else []) + cols
    return df[keep].copy()


def _write_raw_csv(raw_frame, config, metadata, scale_meta) -> None:
    """Save the unconditioned force channels as the study's `raw_data.csv`."""
    import pandas as _pd

    divisor = 1.0
    if metadata.get("n_to_dan_applied"):
        divisor *= N_TO_DAN_FACTOR
    divisor *= float(scale_meta.get("force_scale_divisor", 1.0) or 1.0)

    out = raw_frame.copy()
    if divisor != 1.0:
        for c in out.columns:
            if c != TIME_COLUMN:
                out[c] = _pd.to_numeric(out[c], errors="coerce") / divisor

    path = config.run_output_dir / "raw_data.csv"
    out.to_csv(path, index=False, float_format=CSV_FLOAT_FORMAT)
    logger.info("Raw (unconditioned) data saved: %s  (%d rows, %d channels, "
                "scaled by 1/%g)", path, len(out), len(out.columns) - 1, divisor)




def _warn_if_stationary(df, config) -> None:
    """Say so when a recording holds stops and they are being kept.

    Silence here would be the dangerous option. A parked vehicle still reads a
    static longitudinal load -- on the reference recording |Fx/Fz| sits at 0.667
    while stopped against 0.026 while driving -- and because G-severity is an RMS
    it is dominated by those large ratios. 280 s of standstill in a 5548 s record,
    5% of it, inflated Gx from 0.045 to 0.116. The number is not wrong so much as
    answering a different question, and nothing on the report says which.
    """
    from dtt.preprocessing import stop_mask_frame

    try:
        fs = config.sampling_rate
        speed = next((c for c in ("Vehicle_Speed", "Speed_kmph", "Speed2D")
                      if c in df.columns), None)
        mask, basis = stop_mask_frame(df, fs, speed_column=speed,
                                      min_stop_s=getattr(config, "stop_min_s", 5.0))
    except Exception:                                            # noqa: BLE001
        return
    if mask is None or not mask.any():
        return
    seconds = float(mask.sum()) / fs
    pct = 100.0 * seconds / (len(df) / fs) if len(df) else 0.0
    logger.warning(
        "%.0f s stationary (%.1f%% of the recording, from %s) is being kept. "
        "Standstill carries a static Fx/Fz that an RMS severity is dominated by, "
        "so Gx/Gy/Gxy and the histograms describe the parked vehicle as much as "
        "the road. Re-run with --remove-stops to exclude it.",
        seconds, pct, basis)


def _drop_stationary(df, raw_frame, config, metadata):
    """Excise stationary and paused stretches from the record.

    Placed before validation and statistics: a stop is not road load, and left
    in it drags the mean toward the static wheel weight, inflates the time spent
    near zero in every histogram, and adds cycles to the rainflow count that the
    vehicle never saw.

    One mask cuts both frames. They still share an index here -- sanitisation
    has not yet dropped rows -- and cutting them separately would leave the
    before/after view comparing two different drives.
    """
    from dtt.preprocessing import stop_mask_frame, apply_stop_mask

    fs = config.sampling_rate
    speed = next((c for c in ("Vehicle_Speed", "Speed_kmph", "Speed2D")
                  if c in df.columns), None)
    mask, basis = stop_mask_frame(df, fs, speed_column=speed,
                                  min_stop_s=getattr(config, "stop_min_s", 5.0))
    if mask is None:
        logger.info("No stationary stretches found (basis: %s)", basis or "n/a")
        return df, raw_frame

    n_stops = int(np.count_nonzero(np.diff(np.r_[False, mask].astype(np.int8)) == 1))
    seconds = float(mask.sum()) / fs
    before = len(df)
    df = apply_stop_mask(df, mask, fs, TIME_COLUMN)
    if raw_frame is not None and len(raw_frame) == before:
        raw_frame = apply_stop_mask(raw_frame, mask, fs, TIME_COLUMN)
    elif raw_frame is not None:
        logger.warning("Raw copy is %d rows against %d - left uncut, so the "
                       "before/after view will not line up",
                       len(raw_frame), before)
    metadata["stops_removed"] = n_stops
    metadata["stop_seconds"] = round(seconds, 2)
    metadata["stop_basis"] = basis
    logger.info("Removed %d stationary stretch(es), %.0f s (%.1f%% of the "
                "recording), detected from %s", n_stops, seconds,
                100.0 * seconds / (before / fs) if before else 0.0, basis)
    return df, raw_frame


def run(
    csv_path: Path = None,
    vehicle_name: str = "Vehicle",
    study_name: str   = "",
    sampling_rate: float = None,
    filter_order: int    = None,
    filter_cutoff: float = None,
    apply_filter_flag: bool = True,
    miner_exponent: float   = None,
    output_dir: Path        = None,
    raw_folder: Path        = None,
    raw_files:  list        = None,
    raw_folders: list       = None,
    remove_stops: bool      = False,
    vehicle_type: str       = "",
    famos_mode: bool        = True,
    deglitch:   bool        = False,
) -> Path:
    if csv_path is None and raw_folder is None and not raw_files and not raw_folders:
        raise ValueError("Provide csv_path, raw_folder, raw_folders, or raw_files")

    kwargs = dict(
        csv_path     = Path(csv_path) if csv_path else None,
        raw_folder   = Path(raw_folder) if raw_folder else None,
        raw_files    = [Path(f) for f in raw_files] if raw_files else None,
        raw_folders  = [Path(f) for f in raw_folders] if raw_folders else None,
        vehicle_type = vehicle_type,
        vehicle_name = vehicle_name,
        study_name   = study_name,
    )
    if sampling_rate  is not None: kwargs["sampling_rate"]  = sampling_rate
    if filter_order   is not None: kwargs["filter_order"]   = filter_order
    if filter_cutoff  is not None: kwargs["filter_cutoff"]  = filter_cutoff
    if miner_exponent is not None: kwargs["miner_exponent"] = miner_exponent
    if output_dir     is not None: kwargs["output_dir"]     = Path(output_dir)
    kwargs["apply_filter"] = apply_filter_flag
    kwargs["famos_mode"]   = famos_mode
    kwargs["deglitch"]     = deglitch
    kwargs["remove_stops"] = remove_stops

    config = RunConfig(**kwargs)
    _setup_logging(config)
    logger = logging.getLogger("pipeline")

    source = (f"{len(config.raw_folders)} raw sessions" if getattr(config, "raw_folders", None)
              else f"{len(config.raw_files)} raw files" if config.raw_files
              else config.raw_folder or config.csv_path)
    logger.info("=" * 60)
    logger.info("DTT WFT Automation Pipeline  –  Starting")
    logger.info("Vehicle:  %s", config.vehicle_name)
    logger.info("Study:    %s", config.study_name)
    logger.info("Source:   %s", source)
    if config.vehicle_type:
        logger.info("Vehicle type: %s", config.vehicle_type)
    logger.info("Output:   %s", config.run_output_dir)
    logger.info("=" * 60)

    t0 = time.perf_counter()

    logger.info("[1/9]  Data Ingestion")
    if getattr(config, "raw_folders", None):
        from dtt.ingestion.loader import load_raw_sessions
        df, metadata = load_raw_sessions(config.raw_folders, config)
    elif config.raw_files:
        from dtt.ingestion.loader import load_raw_files
        df, metadata = load_raw_files(config.raw_files, config)
    elif config.raw_folder is not None:
        from dtt.ingestion.loader import load_raw_folder
        df, metadata = load_raw_folder(config.raw_folder, config)
    else:
        df, metadata = load_csv(config.csv_path, config)
    if metadata.get("sampling_rate_hz"):
        config.sampling_rate = metadata["sampling_rate_hz"]
    # imc raw ingestion runs smo/FiltLP at the native rate (before red()), which
    # is the only correct place for it; tell stage 4 not to condition twice.
    # A DataFrame must not travel in metadata: that dict is handed to validation
    # and the report builder, both of which treat it as plain descriptive values.
    raw_frame = metadata.pop("raw_frame", None)
    if metadata.get("sessions", 1) > 1:
        logger.info("Joined %d recording sessions -> %.0f s; seams at %s s",
                    metadata["sessions"], metadata.get("duration_s", 0),
                    ", ".join(f"{t:.0f}" for t in metadata.get("seam_times_s", [])))
    config.famos_applied = bool(metadata.get("famos_recipe"))
    if config.famos_applied:
        # Report how many channels were actually *conditioned*, not how many the
        # recipe looked at. The old count included every passthrough column, so a
        # run in which no force channel was recognised still logged "38 channels".
        from dtt.preprocessing import count_conditioned
        recipe = metadata["famos_recipe"]
        treated = count_conditioned(recipe)
        logger.info("FAMOS recipe applied at ingestion (red x%s): %d/%d channels conditioned",
                    metadata.get("famos_decimate", 1), treated, len(recipe))
        if not treated:
            logger.warning("No channel matched the FAMOS recipe — the data is "
                           "unconditioned and red() will have aliased it")

    # Build the axle-dynamic channel configuration from the actual columns
    # BEFORE validation so every stage (incl. validation) is axle-aware.
    from dtt.run_channels import build_run_channels
    config.run_channels = build_run_channels(list(df.columns))
    logger.info("Channel configuration: %s", config.run_channels.summary())

    from dtt.vehicles import get_preset, match_preset
    preset = get_preset(config.vehicle_type) or match_preset(config.run_channels.channel_set)
    config.tyre = preset.tyre
    logger.info("Vehicle preset: %s  (tyre %s)", preset.name, preset.tyre.tyre_type)

    # Units before anything reads a number: validation thresholds, histogram
    # ranges and every statistic are all quoted in daN, so a decade error here
    # silently invalidates all of them.
    from dtt.ingestion.loader import normalise_force_units
    df, scale_meta = normalise_force_units(df, config)
    metadata.update(scale_meta)

    if getattr(config, "remove_stops", False):
        logger.info("[1b/9] Removing stationary periods")
        df, raw_frame = _drop_stationary(df, raw_frame, config, metadata)
    else:
        _warn_if_stationary(df, config)

    logger.info("[2/9]  Channel Validation")
    validation_report = validate(df, metadata, config)

    logger.info("[3/9]  Data Sanitization")
    df, san_report = sanitize(df, config)

    logger.info("[4/9]  Signal Processing (%s)",
                "imc/FAMOS recipe" if config.famos_mode else "Butterworth LPF")
    if raw_frame is None and not config.famos_applied:
        # A CSV study is still unconditioned here — stage 4 is what conditions
        # it — so its "before" is simply the frame on the way in.
        raw_frame = _force_frame(df, config)
    df = apply_filter(df, config)

    # The raw copy is only useful if it is on the same scale as the conditioned
    # data. It bypassed both unit corrections (it is not in `df`), so re-apply
    # exactly what `df` received, or the before/after traces sit a decade apart.
    if raw_frame is not None:
        _write_raw_csv(raw_frame, config, metadata, scale_meta)

    # Publish the frame every later stage actually analyses — and that the GUI
    # re-reads as "study data". Saving before stage 4 shipped the raw frame.
    processed_csv = config.run_output_dir / "processed_data.csv"
    df.to_csv(processed_csv, index=False, float_format=CSV_FLOAT_FORMAT)
    logger.info("Processed data saved: %s  (%d rows)", processed_csv, len(df))

    logger.info("[5/9]  Statistical Analysis")
    stats = compute_statistics(df, config)

    logger.info("[5b/9] Load Severity  (Gx, Gy, Gxy, DLC)")
    try:
        from dtt.analysis.severity import compute_severity, generate_severity_figure
        severity = compute_severity(df, config)
        generate_severity_figure(severity, config)
    except Exception as exc:
        logger.error("Severity analysis failed: %s", exc, exc_info=True)

    logger.info("[6/9]  Histogram Generation")
    try:
        generate_histograms(df, config)
    except Exception as exc:
        logger.error("Histogram generation failed: %s", exc, exc_info=True)

    logger.info("[7/9]  Heatmap Generation")
    try:
        generate_heatmaps(df, config)
    except Exception as exc:
        logger.error("Heatmap generation failed: %s", exc, exc_info=True)

    logger.info("[8/9]  Boxplot Generation")
    try:
        generate_boxplots(df, config)
    except Exception as exc:
        logger.error("Boxplot generation failed: %s", exc, exc_info=True)

    logger.info("[9/9]  Rainflow Analysis")
    try:
        generate_rainflow(df, config)
    except Exception as exc:
        logger.error("Rainflow generation failed: %s", exc, exc_info=True)

    logger.info("[Report]  Building PowerPoint")
    report_path = build_report(config, validation_report, stats, metadata)

    elapsed = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1f s  (%.1f min)", elapsed, elapsed / 60)
    logger.info("Report: %s", report_path)
    logger.info("=" * 60)

    return report_path


def _cli() -> None:
    parser = argparse.ArgumentParser(description="DTT WFT Automation Pipeline")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv",                               help="Path to WFT CSV file")
    src.add_argument("--raw",                               help="Path to imc STUDIO .raw channel folder")
    src.add_argument("--raw-files", nargs="+",              help="Specific imc .raw files")
    src.add_argument("--raw-folders", nargs="+",
                     help="Several imc .raw session folders, joined end to end "
                          "in recording order (one route captured over several runs)")
    parser.add_argument("--vehicle",  default="Vehicle",    help="Vehicle name for report naming")
    parser.add_argument("--vehicle-type", default="",       help="Vehicle-type preset ('' = auto-detect)")
    parser.add_argument("--study",    default="",           help="Study identifier (default: timestamp)")
    parser.add_argument("--sr",       type=float,           help="Override sampling rate (Hz)")
    parser.add_argument("--cutoff",   type=float,           help="Override filter cutoff (Hz)")
    parser.add_argument("--order",    type=int,             help="Override filter order")
    parser.add_argument("--no-filter", action="store_true", help="Disable all filtering")
    parser.add_argument("--no-famos",  action="store_true",
                        help="Use the legacy Butterworth LPF instead of the imc/FAMOS recipe")
    parser.add_argument("--remove-stops", action="store_true",
                        help="Excise stationary/paused stretches so stops are not "
                             "counted as road load")
    parser.add_argument("--stop-min-s", type=float,
                        help="Shortest stationary stretch to remove (default 5 s)")
    parser.add_argument("--deglitch",  action="store_true",
                        help="Rolling-median de-glitch of DAQ artifact spikes before filtering")
    parser.add_argument("--miner",    type=float,           help="Miner's rule exponent (default 8)")
    parser.add_argument("--outdir",                         help="Override output directory")
    args = parser.parse_args()

    run(
        csv_path         = args.csv,
        raw_folder       = args.raw,
        raw_folders      = args.raw_folders,
        raw_files        = args.raw_files,
        vehicle_type     = args.vehicle_type,
        vehicle_name     = args.vehicle,
        study_name       = args.study,
        sampling_rate    = args.sr,
        filter_order     = args.order,
        filter_cutoff    = args.cutoff,
        apply_filter_flag= not args.no_filter,
        famos_mode       = not args.no_famos,
        deglitch         = args.deglitch,
        remove_stops     = args.remove_stops,
        miner_exponent   = args.miner,
        output_dir       = args.outdir,
    )


if __name__ == "__main__":
    _cli()
