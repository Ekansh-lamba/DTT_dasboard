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
