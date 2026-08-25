"""Tests for the famos replacement library.

The emphasis is on the properties that prevent silent data corruption, since a
signal-processing bug does not crash -- it emits a plausible-looking wrong
answer that nobody questions:

  * chunked execution is BIT-identical to whole-record execution
  * float64 is enforced, never silently widened from float32
  * determinism across repeated runs
  * gaps survive as gaps, and never smear into neighbouring samples
  * an unknown or unimplemented operator raises instead of passing data through
"""

from __future__ import annotations

import numpy as np
import pytest

from famos import ops
from famos.chunked import (StreamingFiltLP, StreamingRed, StreamingSmo,
                           chunk_indices)

FS = 1000.0
CUTOFF = 5.0
ORDER = 4
WIDTH = 0.5
FACTOR = 10


def signal(n: int = 60_000, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    return (np.cumsum(rng.standard_normal(n)) * 0.01 + 0.18
            + 0.5 * np.sin(2 * np.pi * 3 * t))


# --------------------------------------------------------------- dtype policy

def test_float32_is_rejected_not_upcast():
    """float32 has already lost precision; widening it would hide that."""
    with pytest.raises(TypeError, match="lost precision"):
        ops.as_f64(np.zeros(100, dtype=np.float32))
    with pytest.raises(TypeError, match="lost precision"):
        ops.smo(np.zeros(5000, dtype=np.float32), WIDTH, FS)


def test_integers_are_accepted_because_widening_is_exact():
    out = ops.as_f64(np.arange(100, dtype=np.int32))
    assert out.dtype == np.float64
    assert np.array_equal(out, np.arange(100))


def test_every_operator_returns_float64():
    x = signal(20_000)
    assert ops.smo(x, WIDTH, FS).dtype == np.float64
    assert ops.filtlp(x, CUTOFF, ORDER, FS).dtype == np.float64
    assert ops.red(x, FACTOR).dtype == np.float64


# ------------------------------------------------------------- determinism

def test_repeated_runs_are_bit_identical():
    """Same input, same output -- byte for byte, every time."""
    x = signal(50_000)
    a = ops.smo(ops.filtlp(x, CUTOFF, ORDER, FS), WIDTH, FS)
    for _ in range(3):
        b = ops.smo(ops.filtlp(x, CUTOFF, ORDER, FS), WIDTH, FS)
        assert np.array_equal(a, b, equal_nan=True)


# ------------------------------------------- chunked == unchunked, bitwise

@pytest.mark.parametrize("chunk", [600, 1024, 4096, 7777, 20_000, 59_999])
def test_chunked_chain_is_bit_identical(chunk):
    """The corruption guarantee: processing in pieces changes nothing.

    Not 'close to' -- identical. A chunked pipeline that differs even at the
    last bit means the record length silently influences the result, and no
    downstream check would ever catch it.
    """
    x = signal()
    n = x.size
    ref_l = ops.filtlp(x, CUTOFF, ORDER, FS)
    ref_s = ops.smo(ref_l, WIDTH, FS)
    ref_r = ops.red(ref_s, FACTOR)

    f = StreamingFiltLP(CUTOFF, ORDER, FS)
    parts = [f.process(x[a:b], is_first=first)
             for a, b, first, _ in chunk_indices(n, chunk)]
    got_l = np.concatenate(parts)
    got_s = np.concatenate(list(StreamingSmo(WIDTH, FS).run(iter(parts))))
    r = StreamingRed(FACTOR)
    got_r = np.concatenate([r.process(got_s[a:b])
                            for a, b, _, _ in chunk_indices(n, chunk)])

    assert np.array_equal(got_l, ref_l), "FiltLP differs across chunk boundary"
    assert np.array_equal(got_s, ref_s), "Smo differs across chunk boundary"
    assert np.array_equal(got_r, ref_r), "red phase differs across chunks"


def test_streaming_red_keeps_global_phase():
    """Each chunk must not restart the stride pattern."""
    x = np.arange(1000, dtype=float)
    r = StreamingRed(10)
    got = np.concatenate([r.process(x[a:b])
                          for a, b, _, _ in chunk_indices(1000, 137)])
    assert np.array_equal(got, x[::10])


# ------------------------------------------------------------ gap handling

def test_smo_gap_does_not_smear():
    x = np.full(20_000, 2.0)
    x[9_000:9_050] = np.nan
    y = ops.smo(x, WIDTH, FS)
    assert np.all(np.isnan(y[9_000:9_050]))
    assert np.isfinite(y[:9_000]).all() and np.isfinite(y[9_050:]).all()


def test_filtlp_gap_survives_and_does_not_poison():
    """One NaN through a recursive filter would otherwise ruin everything after."""
    x = signal(20_000)
    x[5_000:5_010] = np.nan
    y = ops.filtlp(x, CUTOFF, ORDER, FS)
    assert np.all(np.isnan(y[5_000:5_010]))
    assert np.isfinite(y[5_010:]).all()


def test_filtlp_chunk_boundary_gap_raises_rather_than_differing():
    """A gap at a boundary would interpolate from different endpoints."""
    x = signal(4_000)
    x[:5] = np.nan
    f = StreamingFiltLP(CUTOFF, ORDER, FS)
    with pytest.raises(ValueError, match="reaches a chunk boundary"):
        f.process(x, is_first=True)


# ------------------------------------------------------ operator semantics

def test_filtlp_refuses_unimplemented_characteristics():
    """Substituting Butterworth for Bessel would look fine and be wrong."""
    with pytest.raises(NotImplementedError, match="SvCharacter"):
        ops.filtlp_sos(4, 5.0, FS, character=ops.CHARACTER_BESSEL)


def test_filtlp_refuses_cutoff_at_or_above_nyquist():
    with pytest.raises(ValueError, match="Nyquist"):
        ops.filtlp_sos(4, FS / 2, FS)


def test_red_is_a_stride_not_an_antialiased_decimation():
    x = np.arange(100, dtype=float)
    assert np.array_equal(ops.red(x, 10), x[::10])


def test_smo_window_longer_than_record_raises():
    """The manual: width may not exceed the length of the data set."""
    with pytest.raises(ValueError, match="record is only"):
        ops.smo(np.zeros(100), 10.0, FS)


# ------------------------------------------------------------ chain layer

def test_chain_rejects_unknown_operator():
    from famos.chain import Stage
    with pytest.raises(ValueError, match="unknown operator"):
        Stage("rainflow", {})


def test_seq_translation_of_the_real_imc_file():
    """The production recipe must translate to the chain we expect."""
    from pathlib import Path
    from famos.chain import from_seq

    seq = Path(__file__).resolve().parent.parent / \
        "imc coding_filtering_Channel mapping.txt"
    tr = from_seq(seq)

    def desc(ch):
        return " -> ".join(s.op for s in tr.chain.stages_for(ch))

    # Latacc:  FiltLP(Lat_acc,0,0,4,5) then smo(...,0.5), then the global red
    assert desc("AccelY") == "filtlp -> smo -> red"
    # Long_acc is written in place as `Long_acc = smo(Long_acc, 0.5)`
    assert desc("AccelX") == "smo -> red"
    # WFT forces take smo(0.1) only
    assert desc("WFT_Fx_fl") == "smo -> red"
    # GPS / yaw are passthrough apart from the global decimation
    assert desc("PosLat") == "red"
    assert desc("Message7_YawRate") == "red"

    lat = tr.chain.stages_for("AccelY")
    assert lat[0].params["order"] == 4
    assert lat[0].params["cutoff_hz"] == 5.0
    assert lat[0].params["character"] == ops.CHARACTER_BUTTERWORTH
    assert lat[1].params["width_s"] == 0.5
    assert tr.chain.stages_for("WFT_Fx_fl")[0].params["width_s"] == 0.1

    # The severity arithmetic is not preprocessing and must be reported as
    # untranslated rather than silently dropped.
    assert any("Gxyf" in u for u in tr.unmapped)


def test_chain_run_matches_direct_operator_calls():
    from famos.chain import Chain, Stage
    x = signal(20_000)
    chain = Chain(name="t", channels={"c": [
        Stage("filtlp", {"cutoff_hz": CUTOFF, "order": ORDER}),
        Stage("smo", {"width_s": WIDTH})]},
        final=[Stage("red", {"factor": FACTOR})])
    got, fs = chain.run("c", x, FS)
    ref = ops.red(ops.smo(ops.filtlp(x, CUTOFF, ORDER, FS), WIDTH, FS), FACTOR)
    assert np.array_equal(got, ref)
    assert fs == FS / FACTOR


def test_chain_requires_an_explicit_sample_rate():
    from famos.chain import Chain
    with pytest.raises(ValueError, match="sample rate"):
        Chain(name="t").run("c", np.zeros(1000))


# ---------------------------------------------------------------- audit

def test_manifest_marks_assumed_x0_as_unverified():
    from famos.audit import RunManifest, X0Provenance
    m = RunManifest()
    m.set_x0(X0Provenance.assumed(0.0, "not present in header"))
    assert m.x0["verified"] is False
    assert any("ASSUMED" in w for w in m.warnings)


def test_manifest_records_verified_x0_without_warning():
    from famos.audit import RunManifest, X0Provenance
    m = RunManifest()
    m.set_x0(X0Provenance(value_s=1.25, verified=True,
                          source="IMC2 CD block", field_index=8, byte_offset=0x81))
    assert m.x0["verified"] is True
    assert not m.warnings
