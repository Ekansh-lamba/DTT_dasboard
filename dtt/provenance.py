"""Lightweight per-study processing provenance.

Not the full audit manifest (`famos.audit.RunManifest`, which is built but
never wired into `pipeline.py` — checksums, x0 verification, per-stage FAMOS
chain). This is a much smaller record with one job: let a later comparison
between two *studies* (`dtt.analysis.study_compare`) tell whether they were
produced by comparable processing, so a recipe-version or units difference
doesn't masquerade as a load difference between the two runs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PROVENANCE_FILENAME = "run_provenance.json"

# Fields whose disagreement between two studies means "these were not
# processed the same way" -- a mismatch here can turn a real EV-vs-IC delta
# into a pipeline-version artifact.
_COMPARABILITY_FIELDS = ("famos_applied", "n_to_dan_applied", "famos_version", "git")


def build_provenance(config, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the provenance record for the study currently being processed.

    Called from ``pipeline.py`` once ``metadata``/``config`` hold their final
    per-run values (after unit normalisation, before stage 5). Library/git
    version fields reuse ``famos.audit`` rather than reimplementing them.
    """
    try:
        from famos.audit import library_versions, git_commit
        from dtt.config import BASE_DIR
        libs = library_versions()
        git = git_commit(BASE_DIR)
    except Exception as exc:                                      # noqa: BLE001
        logger.warning("Could not collect library/git provenance: %s", exc)
        libs, git = {}, "unavailable"

    return {
        "study_name": config.study_name,
        "famos_applied": bool(config.famos_applied),
        "famos_mode": bool(config.famos_mode),
        "n_to_dan_applied": bool(metadata.get("n_to_dan_applied")),
        "sampling_rate_hz": float(config.sampling_rate),
        "despike": bool(getattr(config, "despike", False)),
        "transient_despike": bool(getattr(config, "transient_despike", False)),
        "remove_stops": bool(getattr(config, "remove_stops", False)),
        "filter_order": getattr(config, "filter_order", None),
        "filter_cutoff": getattr(config, "filter_cutoff", None),
        "famos_version": libs.get("famos", "unknown"),
        "libraries": libs,
        "git": git,
    }


def write_provenance(run_output_dir: Path, record: Dict[str, Any]) -> Path:
    path = Path(run_output_dir) / PROVENANCE_FILENAME
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    logger.info("Provenance saved: %s", path)
    return path


def load_provenance(run_output_dir: Path) -> Optional[Dict[str, Any]]:
    path = Path(run_output_dir) / PROVENANCE_FILENAME
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def compare_provenance(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]],
                       label_a: str = "study A", label_b: str = "study B"
                       ) -> List[str]:
    """Human-readable warnings about processing-comparability, or ``[]``.

    Mirrors the pattern `dtt/comparison.py::_add_scale_note` already uses: a
    warning appended to the comparison, never a hard failure -- the caller
    can still see the numbers, just knows to distrust them if flagged.
    """
    warnings: List[str] = []
    if a is None or b is None:
        missing = label_a if a is None else label_b
        warnings.append(
            f"No processing provenance recorded for {missing} (older study, "
            f"run before provenance tracking was added) — comparability with "
            f"the other study's recipe/units could not be verified.")
        return warnings

    for field in _COMPARABILITY_FIELDS:
        va, vb = a.get(field), b.get(field)
        if va != vb:
            warnings.append(
                f"{label_a} and {label_b} differ in '{field}' ({va!r} vs "
                f"{vb!r}) — deltas below may reflect a processing/recipe "
                f"change rather than a real difference between the runs.")
    return warnings


def check_stale_provenance(run_output_dir: Path, config=None, df=None) -> List[str]:
    """Warnings about whether a study's saved provenance is fresh enough to
    trust for a workflow_mode="analysis" re-run — never a hard block, only a
    signal the caller must surface (never hide), same warn-don't-block shape
    as :func:`compare_provenance`, but the comparison here is one stored
    study against the code *currently running*, not two peer studies.

    Reuses :func:`load_provenance` and the exact ``famos.audit`` calls
    :func:`build_provenance` already makes for the version fields. When
    ``config``/``df`` are both given, also runs
    ``dtt.ingestion.loader.normalise_force_units``'s existing decade-error
    heuristic **read-only on a copy** of ``df`` — never applied to the real
    analysis frame — purely to flag a units mismatch, not to correct one
    (correcting here would be exactly the forbidden second N-to-daN
    conversion).
    """
    warnings: List[str] = []
    stored = load_provenance(run_output_dir)
    if stored is None:
        # Firmly in the warn path, not a silent pass-through: this is the
        # single most likely stale case in practice (a study processed
        # before provenance stamping existed).
        warnings.append(
            "No run_provenance.json found for this study — it predates "
            "provenance tracking (or was produced outside this pipeline). "
            "Its recipe/units cannot be verified against the current code; "
            "proceeding only because 'only analysis' was explicitly "
            "requested. Treat the results as unverified.")
        return warnings

    try:
        from famos.audit import library_versions, git_commit
        from dtt.config import BASE_DIR
        current_libs = library_versions()
        current_git = git_commit(BASE_DIR)
    except Exception as exc:                                      # noqa: BLE001
        logger.warning("Could not collect current library/git provenance: %s", exc)
        current_libs, current_git = {}, "unavailable"

    stored_famos = stored.get("famos_version", "unknown")
    current_famos = current_libs.get("famos", "unknown")
    if stored_famos != current_famos:
        warnings.append(
            f"This study was processed with famos v{stored_famos}; the code "
            f"now running is famos v{current_famos}. The recipe may have "
            f"changed since this study was preprocessed — re-preprocess if "
            f"in doubt rather than trusting the existing processed_data.csv.")

    stored_git = stored.get("git", "unavailable")
    if stored_git not in ("unavailable", current_git) and current_git != "unavailable":
        warnings.append(
            f"This study's processing commit ({stored_git[:12]}) differs "
            f"from the code currently running ({current_git[:12]}). Pipeline "
            f"internals may have changed since preprocessing.")

    if config is not None and df is not None:
        try:
            from dtt.ingestion.loader import normalise_force_units
            _, scale_meta = normalise_force_units(df.copy(), config)
            if scale_meta.get("force_scale_decades"):
                warnings.append(
                    f"The loaded processed data looks "
                    f"{scale_meta['force_scale_decades']:+d} decade(s) off "
                    f"the expected daN range for this vehicle preset — units "
                    f"may be inconsistent with what run_provenance.json "
                    f"recorded (n_to_dan_applied={stored.get('n_to_dan_applied')}).")
        except Exception as exc:                                   # noqa: BLE001
            logger.warning("Could not run the units consistency check: %s", exc)

    return warnings
