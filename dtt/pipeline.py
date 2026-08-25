import argparse
import logging
import sys
import time
from pathlib import Path

from dtt.config import RunConfig, TIME_COLUMN, N_TO_DAN_FACTOR, SPEED_CANDIDATES
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



def _find_speed_column(df):
    """First usable speed-like column name, or None. Same candidate list
    histograms.py searches, so stop removal and distance weighting agree
    on which column is "the" speed channel for a given study."""
    cols = {c.lower(): c for c in df.columns}
    for name in SPEED_CANDIDATES:
        if name in df.columns:
            return name
        if name.lower() in cols:
            return cols[name.lower()]
    return None


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
    vehicle_type: str       = "",
    famos_mode: bool        = True,
    deglitch:   bool        = False,
) -> Path:
    if csv_path is None and raw_folder is None and not raw_files:
        raise ValueError("Provide csv_path, raw_folder, or raw_files")

    kwargs = dict(
        csv_path     = Path(csv_path) if csv_path else None,
        raw_folder   = Path(raw_folder) if raw_folder else None,
        raw_files    = [Path(f) for f in raw_files] if raw_files else None,
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

    config = RunConfig(**kwargs)
    _setup_logging(config)
    logger = logging.getLogger("pipeline")

    source = (f"{len(config.raw_files)} raw files" if config.raw_files
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
    if config.raw_files:
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

    # Stop removal produces a *derived* series for rainflow and statistics
    # only -- processed_data.csv above is already saved from the full `df`,
    # which stays the canonical processed artifact. This keeps the feature
    # reversible (nothing downstream of `df` itself depends on it) and keeps
    # a stop-containing study's "study data" file showing everything that
    # was actually measured.
    stats_df = df
    if getattr(config, "remove_stops", False):
        from dtt.preprocessing import remove_stops_frame
        speed_col = _find_speed_column(df)
        moving_df, stop_meta = remove_stops_frame(
            df, config.sampling_rate, speed_column=speed_col,
            min_stop_s=getattr(config, "stop_min_s", 3.0),
            moving_kph=getattr(config, "stop_speed_kph", 1.5),
            seam_search_s=getattr(config, "stop_seam_search_s", 1.0),
            seam_blend_s=getattr(config, "stop_seam_blend_s", 0.2))
        metadata["stop_removal"] = stop_meta
        if stop_meta.get("stops_removed"):
            logger.info(
                "Stop removal: %d stop(s), %.1fs removed (%s), basis=%s",
                stop_meta["stops_removed"], stop_meta["stop_seconds"],
                stop_meta.get("removed_intervals"), stop_meta.get("stop_basis"))
            stats_df = moving_df
        else:
            logger.info("Stop removal enabled but found nothing to cut")

    logger.info("[5/9]  Statistical Analysis")
    stats = compute_statistics(stats_df, config)

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
        generate_rainflow(stats_df, config)
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
    parser.add_argument("--vehicle",  default="Vehicle",    help="Vehicle name for report naming")
    parser.add_argument("--vehicle-type", default="",       help="Vehicle-type preset ('' = auto-detect)")
    parser.add_argument("--study",    default="",           help="Study identifier (default: timestamp)")
    parser.add_argument("--sr",       type=float,           help="Override sampling rate (Hz)")
    parser.add_argument("--cutoff",   type=float,           help="Override filter cutoff (Hz)")
    parser.add_argument("--order",    type=int,             help="Override filter order")
    parser.add_argument("--no-filter", action="store_true", help="Disable all filtering")
    parser.add_argument("--no-famos",  action="store_true",
                        help="Use the legacy Butterworth LPF instead of the imc/FAMOS recipe")
    parser.add_argument("--deglitch",  action="store_true",
                        help="Rolling-median de-glitch of DAQ artifact spikes before filtering")
    parser.add_argument("--miner",    type=float,           help="Miner's rule exponent (default 8)")
    parser.add_argument("--outdir",                         help="Override output directory")
    args = parser.parse_args()

    run(
        csv_path         = args.csv,
        raw_folder       = args.raw,
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
        miner_exponent   = args.miner,
        output_dir       = args.outdir,
    )


if __name__ == "__main__":
    _cli()
