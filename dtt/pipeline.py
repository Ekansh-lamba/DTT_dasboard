import argparse
import logging
import sys
import time
from pathlib import Path

from dtt.config import RunConfig, OUTPUTS_DIR
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


def _setup_logging(config: RunConfig) -> None:
    log_path = config.logs_dir / "pipeline.log"
    fmt      = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def run(
    csv_path: Path,
    vehicle_name: str = "Vehicle",
    study_name: str   = "",
    sampling_rate: float = None,
    filter_order: int    = None,
    filter_cutoff: float = None,
    apply_filter_flag: bool = True,
    miner_exponent: float   = None,
    output_dir: Path        = None,
) -> Path:
    kwargs = dict(
        csv_path     = Path(csv_path),
        vehicle_name = vehicle_name,
        study_name   = study_name,
    )
    if sampling_rate  is not None: kwargs["sampling_rate"]  = sampling_rate
    if filter_order   is not None: kwargs["filter_order"]   = filter_order
    if filter_cutoff  is not None: kwargs["filter_cutoff"]  = filter_cutoff
    if miner_exponent is not None: kwargs["miner_exponent"] = miner_exponent
    if output_dir     is not None: kwargs["output_dir"]     = Path(output_dir)
    kwargs["apply_filter"] = apply_filter_flag

    config = RunConfig(**kwargs)
    _setup_logging(config)
    logger = logging.getLogger("pipeline")

    logger.info("=" * 60)
    logger.info("DTT WFT Automation Pipeline  –  Starting")
    logger.info("Vehicle:  %s", config.vehicle_name)
    logger.info("Study:    %s", config.study_name)
    logger.info("CSV:      %s", config.csv_path)
    logger.info("Output:   %s", config.run_output_dir)
    logger.info("=" * 60)

    t0 = time.perf_counter()

    logger.info("[1/9]  Data Ingestion")
    df, metadata = load_csv(config.csv_path, config)
    if metadata.get("sampling_rate_hz"):
        config.sampling_rate = metadata["sampling_rate_hz"]

    logger.info("[2/9]  Channel Validation")
    validation_report = validate(df, metadata, config)

    logger.info("[3/9]  Data Sanitization")
    df, san_report = sanitize(df, config)

    logger.info("[4/9]  Signal Processing (Butterworth LPF)")
    df = apply_filter(df, config)

    logger.info("[5/9]  Statistical Analysis")
    stats = compute_statistics(df, config)

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
    parser.add_argument("--csv",      required=True,        help="Path to WFT CSV file")
    parser.add_argument("--vehicle",  default="Vehicle",    help="Vehicle name for report naming")
    parser.add_argument("--study",    default="",           help="Study identifier (default: timestamp)")
    parser.add_argument("--sr",       type=float,           help="Override sampling rate (Hz)")
    parser.add_argument("--cutoff",   type=float,           help="Override filter cutoff (Hz)")
    parser.add_argument("--order",    type=int,             help="Override filter order")
    parser.add_argument("--no-filter", action="store_true", help="Disable LPF filtering")
    parser.add_argument("--miner",    type=float,           help="Miner's rule exponent (default 8)")
    parser.add_argument("--outdir",                         help="Override output directory")
    args = parser.parse_args()

    run(
        csv_path         = args.csv,
        vehicle_name     = args.vehicle,
        study_name       = args.study,
        sampling_rate    = args.sr,
        filter_order     = args.order,
        filter_cutoff    = args.cutoff,
        apply_filter_flag= not args.no_filter,
        miner_exponent   = args.miner,
        output_dir       = args.outdir,
    )


if __name__ == "__main__":
    _cli()
