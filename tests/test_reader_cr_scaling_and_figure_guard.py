"""Regression tests for two silent failures found during FAMOS validation.

Both bugs were silent, which is why they need guarding rather than just fixing:
neither raised, neither logged, and both produced output that looked ordinary.

  1. The imc ``CR`` block's FIRST field is a transformation flag. The reader
     read the factor and offset without consulting it, so a flag-0 channel
     ("already in physical units", and therefore shipping ``factor=0``) was
     multiplied by zero. Two genuinely-recorded channels in the project's own
     corpus -- Distance and GPS.speed -- came back as flat zero because of it.

  2. ``famos_repro.py`` ended its plot with an unconditional ``plt.show()``,
     which BLOCKS under a GUI backend. An unattended run saved the figure and
     then hung forever without printing its verdict.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from dtt.ingestion.imc_reader import read_famos, read_famos_all

DX = 1.0e-3


# --------------------------------------------------------------- file builder

def _key(name: str, content: bytes, ver: int = 1) -> bytes:
    """One ``|KEY,ver,len,<content>;`` block, the FAMOS on-disk unit."""
    return b"|" + name.encode() + b"," + str(ver).encode() + b"," \
        + str(len(content)).encode() + b"," + content + b";"


def famos_bytes(samples: np.ndarray, *, cr: bytes, dtype_code: int,
                name: str = "TestCh") -> bytes:
    """A minimal single-channel FAMOS file carrying the given CR block."""
    return (
        _key("CF", b"2,1,1", ver=2)
        + _key("CD", f"{DX:.15E},1,1,s,0,0,0, +0.000000000000000E+0,1".encode())
        + _key("CP", f"1,2,{dtype_code},16,0,0,1,0".encode())
        + _key("CR", cr)
        + _key("CN", f"0,0,0,{len(name)},{name},0,".encode())
        + _key("CS", b"1," + samples.tobytes())
    )


def write(tmp_path, data, cr, dtype_code, name="TestCh"):
    p = tmp_path / f"{name}.raw"
    p.write_bytes(famos_bytes(data, cr=cr, dtype_code=dtype_code, name=name))
    return p


# int16 counts, and the float32 payload a flag-0 channel actually ships
COUNTS = np.array([-30, -10, 0, 17, 250, -4, 99, 3], dtype="<i2")
PHYSICAL = np.array([0.0, 1.5, 3.25, 7.125, 11.0, 40.5, 56.85, 59.0],
                    dtype="<f4")

CR_SCALED = b"1,1.000000000000000E-2,0.000000000000000E+0,1,5,m/s"
CR_UNSCALED = b"0,0,0,1,5,km/h"        # flag 0 -> factor field is a literal 0


# ------------------------------------------------------------- the CR fix

def test_cr_flag_zero_leaves_samples_untouched(tmp_path):
    """flag 0 means "already physical" -- the factor-0 field must be ignored.

    This is the bug itself: applying field 1 unconditionally multiplied the
    whole channel by zero and returned a flat, plausible-looking empty signal.
    """
    ch = read_famos(write(tmp_path, PHYSICAL, CR_UNSCALED, 7))
    assert np.allclose(ch.data, PHYSICAL.astype(np.float64))
    assert ch.data.max() == pytest.approx(59.0)
    assert not np.all(ch.data == 0), "flag-0 channel was blanked"
    assert ch.factor == 1.0, "no transformation applied -> factor reads as 1"
    assert ch.unit == "km/h"


def test_cr_flag_one_still_applies_factor_and_offset(tmp_path):
    """The ordinary path must be untouched by the fix."""
    ch = read_famos(write(tmp_path, COUNTS, CR_SCALED, 4))
    assert np.allclose(ch.data, COUNTS.astype(np.float64) * 1.0e-2)
    assert ch.factor == pytest.approx(1.0e-2)
    assert ch.unit == "m/s"


def test_cr_offset_is_applied_when_the_flag_is_set(tmp_path):
    cr = b"1,1.000000000000000E-2,3.276800000000000E+2,1,3,m/s"
    ch = read_famos(write(tmp_path, COUNTS, cr, 4))
    assert np.allclose(ch.data, COUNTS.astype(np.float64) * 1.0e-2 + 327.68)


def test_malformed_cr_flag_scales_rather_than_blanking(tmp_path):
    """An unparseable flag must fall back to scaling, never to zeroing.

    Guessing "no transformation" on a malformed header would reintroduce the
    same silent blanking through a different door.
    """
    ch = read_famos(write(tmp_path, COUNTS, b"x,1.0E-2,0.0E+0,1,5,m/s", 4))
    assert np.allclose(ch.data, COUNTS.astype(np.float64) * 1.0e-2)


def test_both_famos_readers_agree_on_scaling(tmp_path):
    """read_famos and read_famos_all parse CR through one shared helper.

    They each used to carry their own copy of this block, which is exactly how
    the bug survived in one of them -- so pin that they agree.
    """
    for data, cr, code in ((PHYSICAL, CR_UNSCALED, 7), (COUNTS, CR_SCALED, 4)):
        path = write(tmp_path, data, cr, code)
        one, many = read_famos(path), read_famos_all(path)
        assert len(many) == 1
        assert one.factor == many[0].factor
        assert one.unit == many[0].unit
        np.testing.assert_array_equal(one.data, many[0].data)


# ------------------------------------------------- the plt.show() fix

class _Stream:
    def __init__(self, tty: bool):
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


@pytest.fixture
def repro():
    return pytest.importorskip("famos_repro")


@pytest.mark.parametrize("backend,stdin_tty,stdout_tty,expected", [
    ("qtagg", True,  True,  True),    # a real interactive session
    ("qtagg", True,  False, False),   # `python famos_repro.py > log` -- the hang
    ("qtagg", False, True,  False),   # stdin detached (cron, service)
    ("qtagg", False, False, False),   # fully unattended
    ("agg",   True,  True,  False),   # headless backend cannot show anything
    ("pdf",   True,  True,  False),
])
def test_figure_window_offered_only_when_someone_is_watching(
        repro, monkeypatch, backend, stdin_tty, stdout_tty, expected):
    monkeypatch.setattr(repro.matplotlib, "get_backend", lambda: backend)
    monkeypatch.setattr(sys, "stdin", _Stream(stdin_tty))
    monkeypatch.setattr(sys, "stdout", _Stream(stdout_tty))
    assert repro._figure_window_available() is expected


def test_show_figure_defaults_to_auto(repro):
    """None = decide per run. A hardcoded True is what caused the hang."""
    assert repro.SHOW_FIGURE is None
