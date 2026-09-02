"""Regressions for the two failure modes that made preprocessing *look* absent.

Neither is a signal-processing bug, which is what makes them worth pinning: both
produce a screen that quietly under-reports work the pipeline actually did.

1. ``_setup_logging`` used ``logging.basicConfig``, a no-op once the root logger
   has any handler. Called in-process by a host that had already configured
   logging, the run wrote an empty ``pipeline.log`` -- and the Preprocess screen
   read that file to decide whether the FAMOS recipe had run, so a conditioned
   study was reported as unconditioned and offered for a second smoothing pass.

2. That same decision now reads ``run_provenance.json``, which records
   ``famos_applied`` as a structured field instead of leaving the answer to
   whether logging happened to be wired up.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dtt.pipeline import _setup_logging
from dtt.provenance import PROVENANCE_FILENAME, load_provenance, write_provenance


class _Cfg:
    def __init__(self, logs_dir: Path):
        self.logs_dir = logs_dir


@pytest.fixture
def clean_root():
    """Restore the root logger, so these tests cannot leak into the others."""
    root = logging.getLogger()
    saved, level = list(root.handlers), root.level
    for h in list(root.handlers):
        root.removeHandler(h)
    yield root
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in saved:
        root.addHandler(h)
    root.setLevel(level)


def test_log_is_written_even_when_root_already_has_handlers(tmp_path, clean_root):
    """The regression: a host that configured logging first silenced the log."""
    logging.basicConfig(level=logging.INFO)          # the hostile precondition
    assert clean_root.handlers, "precondition: root must already have a handler"

    logs = tmp_path / "logs"
    logs.mkdir()
    _setup_logging(_Cfg(logs))
    logging.getLogger("pipeline").info("FAMOS recipe applied at ingestion (red x10)")
    for h in clean_root.handlers:
        h.flush()

    text = (logs / "pipeline.log").read_text(encoding="utf-8")
    assert "FAMOS recipe applied at ingestion" in text


def test_repeated_runs_do_not_stack_handlers(tmp_path, clean_root):
    """Two studies in one process must not each keep writing to the other."""
    first, second = tmp_path / "a", tmp_path / "b"
    for d in (first, second):
        d.mkdir()
    _setup_logging(_Cfg(first))
    _setup_logging(_Cfg(second))

    owned = [h for h in clean_root.handlers if getattr(h, "_dtt_owned", False)]
    files = [h for h in owned if isinstance(h, logging.FileHandler)]
    assert len(files) == 1, "the previous study's file handler was left attached"

    logging.getLogger("pipeline").info("second study only")
    for h in clean_root.handlers:
        h.flush()
    assert "second study only" not in (first / "pipeline.log").read_text(encoding="utf-8")
    assert "second study only" in (second / "pipeline.log").read_text(encoding="utf-8")


def test_host_console_handler_is_not_duplicated(tmp_path, clean_root):
    logging.basicConfig(level=logging.INFO)          # host's console handler
    logs = tmp_path / "logs"
    logs.mkdir()
    _setup_logging(_Cfg(logs))
    owned_streams = [h for h in clean_root.handlers
                     if getattr(h, "_dtt_owned", False)
                     and isinstance(h, logging.StreamHandler)
                     and not isinstance(h, logging.FileHandler)]
    assert owned_streams == [], "would double-print every line to the console"


# ---------------------------------------------------- provenance as the authority

def test_provenance_records_famos_applied(tmp_path):
    write_provenance(tmp_path, {"study_name": "S", "famos_applied": True})
    assert (tmp_path / PROVENANCE_FILENAME).exists()
    assert load_provenance(tmp_path)["famos_applied"] is True


def test_conditioned_flag_survives_an_empty_log(tmp_path):
    """A study whose log never got written is still known to be conditioned.

    Exercises the same resolution order ``_study_is_conditioned`` uses, without
    needing a Qt widget: provenance first, log only as the fallback.
    """
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "pipeline.log").write_text("", encoding="utf-8")
    write_provenance(tmp_path, {"study_name": "S", "famos_applied": True})

    prov = load_provenance(tmp_path)
    log_says = "FAMOS recipe applied at ingestion" in \
        (tmp_path / "logs" / "pipeline.log").read_text(encoding="utf-8")

    assert log_says is False, "precondition: the log carries no evidence"
    assert bool(prov["famos_applied"]) is True


def test_missing_provenance_falls_back_rather_than_crashing(tmp_path):
    assert load_provenance(tmp_path) is None
