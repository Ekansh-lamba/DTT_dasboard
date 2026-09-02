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
    # `logging.basicConfig` is a silent no-op the moment the root logger already
    # has a handler -- which is true whenever the pipeline is called in-process
    # by a host that configured logging first (a notebook, a test harness, an
    # embedding GUI). The run then wrote no pipeline.log at all.
    #
    # An empty log is not a cosmetic loss. `PreprocessPage._study_is_conditioned`
    # read that file to decide whether the FAMOS recipe had run, so a fully
    # conditioned study was reported as unconditioned and the screen offered to
    # smooth it a second time. Attach the handlers explicitly instead.
    #
    # Only handlers this function owns are replaced, so a host's own logging
    # survives; and the console handler is skipped when the host already has
    # one, rather than double-printing every line.
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt)

    host_console = any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        and not getattr(h, "_dtt_owned", False)
        for h in root.handlers)

    for h in list(root.handlers):
        if getattr(h, "_dtt_owned", False):
            root.removeHandler(h)
            try:
                h.close()
            except (OSError, ValueError):
                pass

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler._dtt_owned = True
    root.addHandler(file_handler)

    if not host_console:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler._dtt_owned = True
        root.addHandler(stream_handler)



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


def _raw_reference_frame(df, config):
    """Time plus every channel the FAMOS recipe conditions — the before/after set.

    Keyed off :func:`dtt.preprocessing.conditions_channel`, the same question the
    imc reader asks when it snapshots raw, so the two ingestion paths store the
    same channel set. It used to be ``rc.mandatory_channels``, which is the
    Fx/Fy/Fz forces only: the moments and the smoothed aux channels reached
    ``processed_data.csv`` correctly but had no raw twin, so the Preprocess
    screen drew them as one unlabelled line.

    Falls back to the mandatory force list only if nothing matches, so a study
    with unrecognised channel names still stores whatever it can.
    """
    from dtt.preprocessing import conditions_channel

    cols = [c for c in df.columns if c != TIME_COLUMN and conditions_channel(c)]
    if not cols:
        rc = getattr(config, "run_channels", None)
        if rc is None:
            return None
        cols = [c for c in rc.mandatory_channels if c in df.columns]
    if not cols:
        return None
    keep = ([TIME_COLUMN] if TIME_COLUMN in df.columns else []) + cols
    return df[keep].copy()


def _write_raw_csv(raw_frame, config, metadata, scale_meta) -> None:
    """Save the unconditioned reference channels as the study's `raw_data.csv`."""
    import pandas as _pd

    divisor = 1.0
    if metadata.get("n_to_dan_applied"):
        divisor *= N_TO_DAN_FACTOR
    divisor *= float(scale_meta.get("force_scale_divisor", 1.0) or 1.0)

    out = raw_frame.copy()
    scaled = []
    if divisor != 1.0:
        # Force channels only. Both upstream corrections -- `_apply_n_to_dan`
        # and `normalise_force_units` -- divide Fx/Fy/Fz and nothing else, by
        # design: moments carry their own imc `CR` factor and their own unit
        # (Nm), and Latacc/Longacc/Vehicle_Speed are not forces at all.
        #
        # Dividing every column was harmless while this frame held the forces
        # alone. It stopped being harmless the moment the frame widened to every
        # conditioned channel: a moment would have been written 10-100x low and
        # then drawn against its correctly-scaled processed trace, which is a
        # far more convincing lie than the missing channel it replaced.
        from dtt.channels import parse_channel, FORCE_COMPONENTS

        for c in out.columns:
            if c == TIME_COLUMN:
                continue
            parsed = parse_channel(str(c))
            if parsed is not None and parsed[1] in FORCE_COMPONENTS:
                out[c] = _pd.to_numeric(out[c], errors="coerce") / divisor
                scaled.append(c)

    path = config.run_output_dir / "raw_data.csv"
    out.to_csv(path, index=False, float_format=CSV_FLOAT_FORMAT)
    logger.info("Raw (unconditioned) reference saved: %s  (%d rows, %d channels; "
                "1/%g applied to %d force channels, others unscaled)",
                path, len(out), len(out.columns) - 1, divisor, len(scaled))
    logger.info("    before/after pair available for: %s",
                ", ".join(c for c in out.columns if c != TIME_COLUMN) or "(none)")


def _write_famos_validation_csv(df, config) -> None:
    """TEMPORARY FAMOS-validation export.

    Writes the signal exactly as it stands right after the FAMOS recipe
    (despike/smo/FiltLP/decimate) and before any analysis stage touches it —
    the same `df` `processed_data.csv` saves, just reordered by the
    platform's own `RunChannels` model (Time, then the canonical force
    channels, then anything else) instead of incidental column order, and
    under a distinct, unambiguous filename so it can't be mistaken for a
    permanent study artifact. Opt-in only (`config.export_famos_validation_csv`)
    — this is a manual cross-check aid against a licensed FAMOS install, not
    part of the normal study output, and is a candidate for removal once
    that cross-check is done.
    """
    rc = getattr(config, "run_channels", None)
    ordered = [TIME_COLUMN] if TIME_COLUMN in df.columns else []
    if rc is not None:
        ordered += [c for c in rc.mandatory_channels if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered]
    out = df[ordered + remaining]

    path = config.run_output_dir / "preprocessed_for_famos_validation.csv"
    out.to_csv(path, index=False, float_format=CSV_FLOAT_FORMAT)
    logger.info("FAMOS-validation CSV saved (temporary, opt-in): %s  (%d rows, %d channels)",
                path, len(out), len(out.columns) - len(ordered[:1]))




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
                                      min_stop_s=getattr(config, "stop_min_s", 3.0))
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
    stop_min_s: float       = None,
    stop_speed_kph: float   = None,
    vehicle_type: str       = "",
    famos_mode: bool        = True,
    deglitch:   bool        = False,
    despike:    bool        = False,
    despike_rail_min_run:    int   = None,
    despike_dropout_max_run: int   = None,
    despike_hw_cutoff_hz:    float = None,
    despike_net:             bool  = False,
    transient_despike:                  bool  = False,
    transient_despike_pct:              float = None,
    transient_despike_window_s:         float = None,
    transient_despike_noise_floor_mult: float = None,
    transient_despike_max_spike_frac:   float = None,
    histogram_range_mode: str = None,
    export_famos_validation_csv: bool = False,
    workflow_mode: str      = "both",
) -> Path:
    """``workflow_mode``: ``"both"`` (default, today's full pipeline,
    unchanged) | ``"preprocess"`` (ingestion + FAMOS recipe only, writes the
    processed study output, then stops before any analysis stage) |
    ``"analysis"`` (skips ingestion/FAMOS entirely; re-analyzes an existing
    study's ``processed_data.csv`` in place, identified by ``study_name`` +
    ``output_dir``; checks that study's saved processing provenance and
    warns, never silently, if it looks stale or missing).

    Returns the PPTX report path for ``"both"``/``"analysis"``, or the
    ``processed_data.csv`` path for ``"preprocess"`` (no report exists yet
    in that mode) — the return type is workflow_mode-dependent. Confirmed by
    grep that no in-process caller inspects this return value today (the GUI
    only drives the pipeline via the CLI subprocess), so this is safe, but
    flagging it as a real signature change.
    """
    if workflow_mode == "analysis":
        if not study_name:
            raise ValueError(
                'workflow_mode="analysis" requires study_name — the existing '
                'study (under output_dir) whose processed_data.csv will be '
                're-analyzed in place.')
    elif csv_path is None and raw_folder is None and not raw_files and not raw_folders:
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
    if stop_min_s     is not None: kwargs["stop_min_s"]     = stop_min_s
    if stop_speed_kph is not None: kwargs["stop_speed_kph"] = stop_speed_kph
    kwargs["despike"]     = despike
    if despike_rail_min_run    is not None: kwargs["despike_rail_min_run"]    = despike_rail_min_run
    if despike_dropout_max_run is not None: kwargs["despike_dropout_max_run"] = despike_dropout_max_run
    if despike_hw_cutoff_hz    is not None: kwargs["despike_hw_cutoff_hz"]    = despike_hw_cutoff_hz
    kwargs["despike_net"] = despike_net
    kwargs["transient_despike"] = transient_despike
    if transient_despike_pct              is not None: kwargs["transient_despike_pct"]              = transient_despike_pct
    if transient_despike_window_s         is not None: kwargs["transient_despike_window_s"]         = transient_despike_window_s
    if transient_despike_noise_floor_mult is not None: kwargs["transient_despike_noise_floor_mult"] = transient_despike_noise_floor_mult
    if transient_despike_max_spike_frac   is not None: kwargs["transient_despike_max_spike_frac"]   = transient_despike_max_spike_frac
    if histogram_range_mode is not None: kwargs["histogram_range_mode"] = histogram_range_mode
    kwargs["export_famos_validation_csv"] = export_famos_validation_csv
    kwargs["workflow_mode"] = workflow_mode

    config = RunConfig(**kwargs)
    _setup_logging(config)
    logger = logging.getLogger("pipeline")

    if config.workflow_mode == "analysis":
        source = f"existing study '{config.study_name}' (workflow_mode=analysis, no fresh ingestion)"
    else:
        source = (f"{len(config.raw_folders)} raw sessions" if getattr(config, "raw_folders", None)
                  else f"{len(config.raw_files)} raw files" if config.raw_files
                  else config.raw_folder or config.csv_path)
    logger.info("=" * 60)
    logger.info("DTT WFT Automation Pipeline  –  Starting")
    logger.info("Vehicle:  %s", config.vehicle_name)
    logger.info("Study:    %s", config.study_name)
    logger.info("Source:   %s", source)
    logger.info("Mode:     %s", config.workflow_mode)
    if config.vehicle_type:
        logger.info("Vehicle type: %s", config.vehicle_type)
    logger.info("Output:   %s", config.run_output_dir)
    logger.info("=" * 60)

    t0 = time.perf_counter()

    if config.workflow_mode == "analysis":
        # No ingestion, no FAMOS recipe, no sanitization/filtering -- this
        # mode's entire point is to trust the existing processed_data.csv
        # and re-run only the analysis stages on it, in place.
        logger.info("[1-4/9]  Skipped (workflow_mode=analysis) -- reading "
                    "the existing processed_data.csv instead of re-preprocessing")
        processed_csv = config.run_output_dir / "processed_data.csv"
        if not processed_csv.exists():
            raise FileNotFoundError(
                f'workflow_mode="analysis" requires an existing '
                f'processed_data.csv in {config.run_output_dir}, found none. '
                f'Run workflow_mode="preprocess" or "both" on this study first.')
        import pandas as _pd
        df = _pd.read_csv(processed_csv)
        raw_frame, scale_meta = None, {}

        from dtt.run_channels import build_run_channels
        config.run_channels = build_run_channels(list(df.columns))
        logger.info("Channel configuration (rebuilt from the existing CSV): %s",
                    config.run_channels.summary())

        from dtt.vehicles import get_preset, match_preset
        preset = get_preset(config.vehicle_type) or match_preset(config.run_channels.channel_set)
        config.tyre = preset.tyre
        logger.info("Vehicle preset: %s  (tyre %s)", preset.name, preset.tyre.tyre_type)

        # Required provenance guard: this mode trusts a file it did not just
        # produce, so warn loudly -- never silently -- when that trust is
        # shaky. Missing provenance (a study from before provenance tracking
        # existed) is the most likely real case and is firmly in this warn
        # path, not a silent pass-through.
        from dtt.provenance import load_provenance, check_stale_provenance
        prov_warnings = check_stale_provenance(config.run_output_dir, config=config, df=df)
        if prov_warnings:
            logger.warning("=" * 60)
            logger.warning('workflow_mode="analysis" is trusting a previously-processed '
                           "file. %d provenance warning(s):", len(prov_warnings))
            for w in prov_warnings:
                logger.warning("  - %s", w)
            logger.warning("Proceeding as explicitly requested -- results may not "
                           "reflect the current pipeline/recipe.")
            logger.warning("=" * 60)
        stored_prov = load_provenance(config.run_output_dir)

        metadata = {
            "file_name": processed_csv.name,
            "rows": len(df),
            "columns": len(df.columns),
            "sampling_rate_hz": config.sampling_rate,
            "duration_s": (len(df) / config.sampling_rate) if config.sampling_rate else 0.0,
            "n_to_dan_applied": bool((stored_prov or {}).get("n_to_dan_applied")),
        }

        logger.info("[2/9]  Channel Validation  (re-checked against the existing processed data)")
        validation_report = validate(df, metadata, config)
    else:
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
            # Per-channel, not just a count. A reviewer checking that the
            # moments really were smoothed should not have to infer it from
            # "15/38" -- and a channel that silently fell through to
            # passthrough is only visible if every channel is named.
            for _ch, _ops in recipe.items():
                logger.info("    %-22s %s", _ch, _ops)
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

        if not getattr(config, "remove_stops", False):
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
            raw_frame = _raw_reference_frame(df, config)
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

        if getattr(config, "export_famos_validation_csv", False):
            _write_famos_validation_csv(df, config)

        # Recorded so a later two-study comparison (dtt.analysis.study_compare)
        # can tell whether both studies were produced by comparable processing --
        # a recipe-version or units difference should not masquerade as a real
        # load difference between two runs.
        from dtt.provenance import build_provenance, write_provenance
        write_provenance(config.run_output_dir, build_provenance(config, metadata))

        if config.workflow_mode == "preprocess":
            elapsed = time.perf_counter() - t0
            logger.info("=" * 60)
            logger.info("Pipeline complete (preprocess-only) in %.1f s  (%.1f min)",
                        elapsed, elapsed / 60)
            logger.info("Processed data: %s", processed_csv)
            logger.info("=" * 60)
            return processed_csv

    # Stop removal produces a *derived* series for every analysis stage below
    # -- processed_data.csv above is already saved from the full `df`, which
    # stays the canonical processed artifact regardless. This keeps the
    # feature reversible (nothing upstream of this point depends on it) and
    # keeps a stop-containing study's "study data" file showing everything
    # that was actually measured, while statistics/severity/histograms/
    # heatmaps/boxplots/rainflow all see the moving-only view: a parked
    # vehicle still carries a static Fx/Fz that distorts more than just
    # rainflow -- on the reference recording it inflated FR Gx from 0.045 to
    # 0.116 (G-severity is an RMS, dominated by those few large standstill
    # ratios) and would skew every histogram's near-zero bin too.
    analysis_df = df
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
            analysis_df = moving_df
        else:
            logger.info("Stop removal enabled but found nothing to cut")

    logger.info("[5/9]  Statistical Analysis")
    stats = compute_statistics(analysis_df, config)

    logger.info("[5b/9] Load Severity  (Gx, Gy, Gxy, DLC)")
    try:
        from dtt.analysis.severity import compute_severity, generate_severity_figure
        severity = compute_severity(analysis_df, config)
        generate_severity_figure(severity, config)
    except Exception as exc:
        logger.error("Severity analysis failed: %s", exc, exc_info=True)

    logger.info("[6/9]  Histogram Generation")
    try:
        generate_histograms(analysis_df, config,
                            range_mode=getattr(config, "histogram_range_mode", "full"))
    except Exception as exc:
        logger.error("Histogram generation failed: %s", exc, exc_info=True)

    logger.info("[7/9]  Heatmap Generation")
    try:
        generate_heatmaps(analysis_df, config)
    except Exception as exc:
        logger.error("Heatmap generation failed: %s", exc, exc_info=True)

    logger.info("[8/9]  Boxplot Generation")
    try:
        generate_boxplots(analysis_df, config)
    except Exception as exc:
        logger.error("Boxplot generation failed: %s", exc, exc_info=True)

    logger.info("[9/9]  Rainflow Analysis")
    try:
        generate_rainflow(analysis_df, config)
    except Exception as exc:
        logger.error("Rainflow generation failed: %s", exc, exc_info=True)

    logger.info("[9b/9] Welch PSD")
    try:
        from dtt.analysis.psd import generate_psd
        generate_psd(analysis_df, config)
    except Exception as exc:
        logger.error("PSD generation failed: %s", exc, exc_info=True)

    logger.info("[Report]  Building PowerPoint")
    report_path = build_report(config, validation_report, stats, metadata,
                               analysis_only=(config.workflow_mode == "analysis"))

    elapsed = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1f s  (%.1f min)", elapsed, elapsed / 60)
    logger.info("Report: %s", report_path)
    logger.info("=" * 60)

    return report_path


def _cli() -> None:
    parser = argparse.ArgumentParser(description="DTT WFT Automation Pipeline")
    # Not required at the argparse level: workflow_mode="analysis" needs none
    # of these (it reads an existing study's processed_data.csv instead).
    # Enforced manually below for the other two modes, where it still is.
    src = parser.add_mutually_exclusive_group(required=False)
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
                        help="Shortest stationary stretch to remove (default 3 s)")
    parser.add_argument("--stop-speed-kph", type=float,
                        help="Speed below which the vehicle reads as stopped (default 1.5 km/h)")
    parser.add_argument("--deglitch",  action="store_true",
                        help="Rolling-median de-glitch of DAQ artifact spikes before filtering")
    parser.add_argument("--despike",  action="store_true",
                        help="Physical-rule despike (rail/dropout/narrow-spike) ahead of smo/FiltLP")
    parser.add_argument("--despike-rail-min-run", type=int,
                        help="Samples pinned at channel min/max to count as saturation (default 3)")
    parser.add_argument("--despike-dropout-max-run", type=int,
                        help="Samples of any repeated value to count as a frozen sensor (default 5)")
    parser.add_argument("--despike-hw-cutoff-hz", type=float,
                        help="Hardware anti-alias cutoff; sets the minimum physical feature width (default 200 Hz)")
    parser.add_argument("--despike-net", action="store_true",
                        help="Optional loose adaptive Hampel net for gross leftovers after despike")
    parser.add_argument("--transient-despike", action="store_true",
                        help='Manual\'s "20%% within 1s" spike rule, run post-smooth/decimate')
    parser.add_argument("--transient-despike-pct", type=float,
                        help="Deviation from local trend, as a percent, to flag (default 20)")
    parser.add_argument("--transient-despike-window-s", type=float,
                        help="Local-trend window, seconds (default 1.0)")
    parser.add_argument("--transient-despike-noise-floor-mult", type=float,
                        help="Noise-floor multiple the deviation must also clear (default 10)")
    parser.add_argument("--transient-despike-max-spike-frac", type=float,
                        help="Cap on flagged run length as a fraction of the window (default 0.4)")
    parser.add_argument("--miner",    type=float,           help="Miner's rule exponent (default 8)")
    parser.add_argument("--outdir",                         help="Override output directory")
    parser.add_argument("--histogram-range-mode", choices=("full", "autoscale"), default=None,
                        help='Force-histogram x-axis: "full" (default) uses the configured '
                             'per-channel span; "autoscale" always fits to the actual data '
                             "(P0.5-P99.5).")
    parser.add_argument("--export-famos-validation-csv", action="store_true",
                        help="TEMPORARY: also write a wide, RunChannels-ordered CSV of the "
                             "post-recipe (pre-analysis) signal, for a manual FAMOS cross-check. "
                             "Off by default; not part of normal study output.")
    parser.add_argument("--mode", choices=("preprocess", "analysis", "both"), default="both",
                        help='"both" (default): today\'s full pipeline. "preprocess": ingestion '
                             '+ FAMOS recipe only, stop before any analysis stage. "analysis": '
                             "skip preprocessing, re-analyze an existing study's "
                             "processed_data.csv in place (requires --study).")
    args = parser.parse_args()

    if args.mode == "analysis":
        if not args.study:
            parser.error("--mode analysis requires --study (the existing study to re-analyze)")
    elif not (args.csv or args.raw or args.raw_files or args.raw_folders):
        parser.error("one of --csv, --raw, --raw-files, --raw-folders is required "
                     "unless --mode analysis is used")

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
        stop_min_s       = args.stop_min_s,
        stop_speed_kph   = args.stop_speed_kph,
        despike                             = args.despike,
        despike_rail_min_run                = args.despike_rail_min_run,
        despike_dropout_max_run             = args.despike_dropout_max_run,
        despike_hw_cutoff_hz                = args.despike_hw_cutoff_hz,
        despike_net                         = args.despike_net,
        transient_despike                   = args.transient_despike,
        transient_despike_pct               = args.transient_despike_pct,
        transient_despike_window_s          = args.transient_despike_window_s,
        transient_despike_noise_floor_mult  = args.transient_despike_noise_floor_mult,
        transient_despike_max_spike_frac    = args.transient_despike_max_spike_frac,
        histogram_range_mode = args.histogram_range_mode,
        miner_exponent   = args.miner,
        output_dir       = args.outdir,
        export_famos_validation_csv = args.export_famos_validation_csv,
        workflow_mode    = args.mode,
    )


if __name__ == "__main__":
    _cli()
