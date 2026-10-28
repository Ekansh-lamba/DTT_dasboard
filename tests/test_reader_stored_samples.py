"""The raw readers return each channel's stored samples, nothing more or less.

Round 9: the app's 100 Hz output drifted against FAMOS by 5.22 ppm (offset 0
at the start, -29 samples at 1 kHz by the end of a 5,548 s recording). Two
reader faults stacked:

  1. ``read_famos`` read the whole ``CS`` block instead of the ``Cb`` buffer
     inside it, so 8,296 bytes of recorder padding came back as 4,148 extra
     samples of 0 at the end of every channel.
  2. ``_assemble`` then stretched every channel end to end onto the length of
     the first file in the folder (``Acceleration.raw``, float32, padded to a
     different length), i.e. linearly resampled the WFT data.

And on the imc3 path, ``read_raw`` found the data start with a smoothness
heuristic that landed 58 samples early (PROJECT_HISTORY §4.6).

These tests pin the fix: counts come from the file's own framing, values are
the stored integers x the calibration factor, and channels share a timeline
by whole samples, never by interpolation.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dtt.ingestion.imc_reader import (
    _common_timeline, read_famos, read_raw, ImcChannel,
)

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"


def _ref_recording() -> Path:
    """The 2006-08-25 two-wheel recording (FR_*_2 / RR_*_1), gitignored.

    Set ``DTT_REF_RECORDING`` to its folder, or keep it at the repo-relative
    path the FAMOS cross-check tests already use.
    """
    env = os.environ.get("DTT_REF_RECORDING")
    if env:
        return Path(env)
    return Path("raw data in .dat format/2006-08-25 09-10-46 (1)")


REF = _ref_recording()
needs_ref = pytest.mark.skipif(not (REF / "FR_Fx_2.raw").exists(),
                               reason="reference recording not present "
                                      "(set DTT_REF_RECORDING)")
needs_cut = pytest.mark.skipif(
    not ((DATA / "WFT_Fx_fr.raw").exists() and (DATA / "Fx_raw_cut.csv").exists()),
    reason="data/WFT_Fx_fr.raw + data/Fx_raw_cut.csv are gitignored")


# ------------------------------------------------- classic FAMOS: |Cb buffer

def _key(name: str, content: bytes, ver: int = 1) -> bytes:
    return b"|" + name.encode() + b"," + str(ver).encode() + b"," \
        + str(len(content)).encode() + b"," + content + b";"


def _famos_with_padding(counts: np.ndarray, pad: int, factor: float = 10.0) -> bytes:
    """A FAMOS file shaped like the recorder's: a ``Cb`` buffer of exactly the
    samples, inside a ``CS`` block padded with ``pad`` zero bytes."""
    nbytes = counts.nbytes
    cb = (f"1,0,10461,1,0,{nbytes},0,{nbytes},1, +0.000000000000000E+0,"
          f" +8.409642460000000E+8,").encode()
    return (_key("CF", b"2,1,1", ver=2)
            + _key("CD", b"1.000000000000000E-3,1,1,s,0,0,0, +0.000000000000000E+0,1")
            + _key("CP", b"10461,2,4,16,0,0,1,0")
            + _key("CR", f"1,{factor:.15E},0.000000000000000E+0,1,1,N".encode())
            + _key("CN", b"0,0,0,7,FR_Fx_2,0,")
            + _key("Cb", cb)
            + _key("CS", b"1," + counts.tobytes() + b"\x00" * pad))


def test_famos_reads_the_cb_buffer_not_the_padded_cs_block(tmp_path):
    counts = np.array([-4530, -4530, -4532, 17, 250, -4, 99, 3], dtype="<i2")
    p = tmp_path / "FR_Fx_2.raw"
    p.write_bytes(_famos_with_padding(counts, pad=8296))
    ch = read_famos(p)
    assert ch.data.size == counts.size                 # not 8 + 4148
    np.testing.assert_array_equal(ch.data, counts.astype(float) * 10.0)
    assert ch.fs == 1000.0


@needs_ref
def test_fr_fx_2_count_and_values_are_the_stored_samples():
    """Sample count = the file's own ``Cb`` buffer length / 2; every value is
    the stored int16 x the ``CR`` factor (bit for bit -- no interpolation)."""
    raw = (REF / "FR_Fx_2.raw").read_bytes()
    cb_pos = raw.find(b"|Cb,")
    cb = raw[cb_pos:raw.find(b";", cb_pos)].split(b",")[3:]
    filled = int(cb[7])
    # "|CS,<ver>,<length>,<key index>,<data>": data follows the third comma.
    cs_pos = raw.find(b"|CS,")
    after_ver = raw.find(b",", cs_pos + 4)
    after_len = raw.find(b",", after_ver + 1)
    data_start = raw.find(b",", after_len + 1) + 1
    stored = np.frombuffer(raw[data_start:data_start + filled], dtype="<i2")

    ch = read_famos(REF / "FR_Fx_2.raw")
    assert ch.data.size == filled // 2 == 5_547_821
    np.testing.assert_array_equal(ch.data, stored.astype(float) * ch.factor)


@needs_ref
def test_reference_folder_keeps_fr_fx_2_bit_exact_through_assembly():
    """The folder reader end to end, without the recipe: FR_Fx_2 must still be
    the stored samples on a 1 kHz grid of exactly 5,547,821 rows."""
    from dtt.ingestion.imc_reader import read_folder

    df, meta = read_folder(REF, target_fs=None, famos=False)
    assert len(df) == 5_547_821
    assert meta["raw_fs_hz"] == 1000.0
    np.testing.assert_array_equal(df["FR_Fx_2"].to_numpy(),
                                  read_famos(REF / "FR_Fx_2.raw").data)
    assert df["Time"].iloc[1] == pytest.approx(0.001)


# -------------------------------------------------------- imc3: |RC5 framing

def _imc3_bytes(counts: np.ndarray, chunk: int = 5, dx: float = 1e-3,
                factor: float = 1.373291015625, code: int = 4) -> bytes:
    """A small imc3 file with the real layout: metadata, then ``|RR1|RT1``,
    ``|RC5`` data chunks, an ``|Ri1`` index of header copies, more chunks and
    ``|RE1`` -- the index copies are what the old reader mistook for data."""
    rt1 = b"|RT1" + b"\x01\x00\x00\x00" + b"\x00" * 24
    head = (b"|imc3,1;|CB1" + b"\x00" * 20
            + b"|CC1" + struct.pack("<I", 1) + struct.pack("<d", dx) + b"\x00" * 16
            + b"|CM1" + struct.pack("<I", code) + struct.pack("<d", factor)
            + b"\x00" * 8 + b"\x01\x00N"
            + b"|CD1" + b"\x00" * 20
            + b"|CN1" + b"\x00" + struct.pack("<H", 9) + b"WFT_Fx_fr"
            + struct.pack("<H", 0)
            + b"|CP1" + b"\x00" * 12)
    raw = counts.astype("<i2" if code == 4 else "<i4").tobytes()
    size = 2 if code == 4 else 4
    step = chunk * size
    pieces = [raw[i:i + step] for i in range(0, len(raw), step)]
    rc5 = lambda p: b"|RC5" + struct.pack("<iii", 1, 1, len(p))   # noqa: E731
    half = len(pieces) // 2
    body = b"|RR1" + rt1
    for p in pieces[:half]:
        body += rc5(p) + p
    body += b"|Ri1" + struct.pack("<ii", 0, half + 1) + rt1
    body += b"".join(rc5(p) for p in pieces[:half])
    for p in pieces[half:]:
        body += rc5(p) + p
    return head + body + b"|RE1|CS1" + b"\x00" * 30 + b"|CE1"


def test_imc3_reads_exactly_the_rc5_payloads(tmp_path):
    counts = np.arange(-50, 50, dtype=np.int16) * 7
    p = tmp_path / "WFT_Fx_fr.raw"
    p.write_bytes(_imc3_bytes(counts))
    ch = read_raw(p)
    assert ch.data.size == counts.size
    np.testing.assert_array_equal(ch.data, counts * 1.373291015625)
    assert ch.fs == 1000.0
    assert ch.name == "FR_Fx"


def test_imc3_sample_type_comes_from_the_header(tmp_path):
    counts = np.array([70000, -70000, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int32)
    p = tmp_path / "PosAlt.raw"
    p.write_bytes(_imc3_bytes(counts, code=6, factor=0.001, dx=0.2))
    ch = read_raw(p)
    np.testing.assert_allclose(ch.data, counts * 0.001)
    assert ch.fs == pytest.approx(5.0)


def test_imc3_unknown_record_fails_loudly(tmp_path):
    p = tmp_path / "WFT_Fx_fr.raw"
    p.write_bytes(_imc3_bytes(np.arange(20, dtype=np.int16)).replace(b"|RE1", b"|RX9"))
    with pytest.raises(ValueError, match="unknown data record"):
        read_raw(p)


@needs_cut
def test_wft_fx_fr_matches_famos_raw_cut_at_offset_zero():
    """The §4.6 / §10 regression check. FAMOS's ``Fx_raw_cut`` is samples
    1,000,000 .. 1,030,000 of ``WFT_Fx_fr``; the old reader matched it only at
    +58. Now offset 0, to the CSV export's 0.005 N rounding floor."""
    ch = read_raw(DATA / "WFT_Fx_fr.raw")
    assert ch.data.size == 4_219_125            # = imcdbc duration / x-delta
    ref = pd.read_csv(DATA / "Fx_raw_cut.csv", skiprows=[1], skipinitialspace=True)
    ref.columns = [c.strip() for c in ref.columns]
    fam = ref["Fx_raw_cut"].to_numpy(float)
    ours = ch.data[1_000_000:1_000_000 + fam.size]
    assert np.max(np.abs(ours - fam)) < 0.0051
    assert np.corrcoef(ours, fam)[0, 1] > 0.999999999


# ------------------------------------------------------- the shared timeline

def _ch(name, data, fs):
    return ImcChannel(name=name, raw_name=name, unit="", factor=1.0,
                      data=np.asarray(data, float), fs=fs)


def test_timeline_never_resamples_wft_channels():
    fx = np.arange(10, dtype=float) * 3.0
    acc = np.arange(13, dtype=float)                 # longer aux channel first
    out, fs, adj = _common_timeline([_ch("Acceleration", acc, 1000.0),
                                     _ch("FR_Fx_2", fx, 1000.0),
                                     _ch("RR_Fx_1", fx[:8], 1000.0)])
    assert fs == 1000.0 and out["Time"].size == 10   # the WFT length, not 13
    np.testing.assert_array_equal(out["FR_Fx_2"], fx)
    np.testing.assert_array_equal(out["RR_Fx_1"][:8], fx[:8])
    assert np.isnan(out["RR_Fx_1"][8:]).all()        # padded at the end
    np.testing.assert_array_equal(out["Acceleration"], acc[:10])   # cut at the end
    assert adj == {"Acceleration": -3, "RR_Fx_1": 2}


def test_timeline_places_a_slow_channel_on_its_own_time_axis():
    gps = np.array([0.0, 10.0, 20.0])                # 5 Hz: t = 0, 0.2, 0.4 s
    out, _, _ = _common_timeline([_ch("FR_Fx_2", np.zeros(501), 1000.0),
                                  _ch("GPS.speed", gps, 5.0)])
    g = out["GPS.speed"]
    assert g[0] == 0.0 and g[100] == pytest.approx(5.0) and g[400] == pytest.approx(20.0)
    assert np.isnan(g[401:]).all()                   # nothing invented past 0.4 s


def test_timeline_refuses_wft_channels_at_different_rates():
    with pytest.raises(ValueError, match="never resampled"):
        _common_timeline([_ch("FR_Fx_2", np.zeros(10), 1000.0),
                          _ch("RR_Fx_1", np.zeros(10), 500.0)])
