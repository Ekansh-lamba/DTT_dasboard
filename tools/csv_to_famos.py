"""Write channels as native imc FAMOS ``.dat`` files.

FAMOS opens these directly — no ASCII import assistant, no markers to place, no
"Marker must not point behind the file!". That assistant is where a corpus
capture stalls, and trial days are the one resource here that cannot be
replenished.

    python tools/csv_to_famos.py golden_corpus/famos_golden_inputs.csv \\
                                 --out golden_corpus/famos_in --dx 0.001

One file per channel, named after the column, which is also the name FAMOS shows
in its variable list — so ``gc_impulse.dat`` loads as ``gc_impulse`` and the
sequence finds it by name with nothing to rename.

Written as float64 (``CP`` number format 8) with factor 1 and offset 0. The real
recordings store int16 with a scale factor, which is right for a DAQ but wrong
here: these are synthetic references whose whole purpose is to be reproduced
exactly, and quantising them to 16 bits would put a floor under the very
measurement they exist to make.

The format is the classic FAMOS key stream: ``|KEY,version,length,content;``
repeated, ending with ``CS`` holding the samples. Every field not carrying our
data is copied from the structure of a genuine imc file, on the principle that
the closer the output sits to something FAMOS already accepts, the less there is
to be rejected.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# CP number-format codes, mirroring _FAMOS_DT in dtt.ingestion.imc_reader.
_FMT_F64 = 8
_BYTES_F64 = 8


def _key(name: str, version: int, content: bytes) -> bytes:
    """One ``|KEY,version,length,content;`` block.

    ``length`` counts the content bytes only. A real file pads the number with
    spaces in some keys and not others, so FAMOS clearly tolerates either; plain
    digits are used here.
    """
    return b"|" + name.encode("latin1") + b"," + str(version).encode() + b"," \
        + str(len(content)).encode() + b"," + content + b";"


def famos_bytes(y: np.ndarray, name: str, dx: float, unit: str = "",
                x0: float = 0.0, comment: str = "") -> bytes:
    """One channel as a complete FAMOS file."""
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    blob = y.tobytes()                      # little-endian float64, native order
    buf_ref = 1
    name_b = name.encode("latin1")
    comment_b = comment.encode("latin1")
    unit_b = unit.encode("latin1")

    out = bytearray()
    out += _key("CF", 2, b"1")                               # format version
    out += _key("CK", 1, b"1,1")                             # group start
    out += _key("NO", 1, b"0,50,dtt csv_to_famos,0,")         # origin
    out += _key("CG", 1, b"1,1,1")                           # one component
    # CD: dx, then unit of the x axis. Trailing fields copied from a real file.
    cd = (f"{dx:.15E},1,1,s,0,0,0, +0.000000000000000E+0,1").encode("latin1")
    out += _key("CD", 1, cd)
    out += _key("NT", 1, b" 1, 1,1980, 0, 0, 0.0000000")      # timestamp
    out += _key("CC", 1, b"1,1")                              # component start
    # CP: buffer ref, bytes per sample, number format, significant bits, ...
    cp = f"{buf_ref},{_BYTES_F64},{_FMT_F64},{_BYTES_F64 * 8},0,0,1,0".encode()
    out += _key("CP", 1, cp)
    # CR: transformation, factor, offset, calibrated, unit length, unit.
    # factor 1 / offset 0 -- the samples are already physical values.
    cr = (f"1,{1.0:.15E},{0.0:.15E},1,{len(unit_b)},".encode("latin1") + unit_b)
    out += _key("CR", 1, cr)
    # CN: language, ..., name length, name, comment
    cn = (f"0,0,0,{len(name_b)},".encode() + name_b + b"," + comment_b + b",")
    out += _key("CN", 1, cn)
    # Cb: buffer table -- one buffer covering the whole sample block.
    nbytes = len(blob)
    cb = (f"1,1,{buf_ref},1,0,{nbytes},0,{nbytes},0,"
          f"{x0:+.15E},0,{nbytes}").encode("latin1")
    out += _key("Cb", 1, cb)
    out += _key("CS", 1, b"1," + blob)                        # the samples
    return bytes(out)


def write_channel(path: Path, y: np.ndarray, name: str, dx: float,
                  unit: str = "", x0: float = 0.0) -> Path:
    path.write_bytes(famos_bytes(y, name, dx, unit, x0))
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--out", default="golden_corpus/famos_in")
    ap.add_argument("--dx", type=float, default=None,
                    help="sample step in seconds; taken from a Time column if absent")
    ap.add_argument("--time-column", default="Time")
    ap.add_argument("--verify", action="store_true", default=True,
                    help="read every file back and compare (default on)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    dx = args.dx
    if dx is None:
        if args.time_column not in df.columns:
            print(f"No --dx given and no '{args.time_column}' column to take it from")
            return 2
        t = pd.to_numeric(df[args.time_column], errors="coerce").to_numpy(float)
        dx = float(np.median(np.diff(t[:1000])))
    x0 = (float(df[args.time_column].iloc[0])
          if args.time_column in df.columns else 0.0)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    channels = [c for c in df.columns if c != args.time_column]

    print(f"{len(channels)} channels x {len(df)} samples, "
          f"dx = {dx:g} s ({1/dx:g} Hz) -> {out_dir}")
    written = []
    for c in channels:
        y = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        p = write_channel(out_dir / f"{c}.dat", y, c, dx, x0=x0)
        written.append((p, c, y))
        print(f"  {p.name:24s} {p.stat().st_size / 1e6:7.2f} MB")

    if not args.verify:
        return 0

    # Read every file back through the project's own FAMOS parser. This does not
    # prove FAMOS will accept them, but a file our reader cannot parse is
    # certainly wrong, and it catches a malformed key stream immediately.
    from dtt.ingestion.imc_reader import read_famos
    print("\nverifying round-trip:")
    bad = 0
    for p, name, y in written:
        ch = read_famos(p)
        if ch is None:
            print(f"  {p.name:24s} FAILED to parse"); bad += 1; continue
        ok_n = ch.data.size == y.size
        err = float(np.nanmax(np.abs(ch.data - y[:ch.data.size]))) if ok_n else float("nan")
        ok_fs = abs(ch.fs - 1.0 / dx) < 1e-6
        good = ok_n and ok_fs and err == 0.0
        bad += not good
        print(f"  {p.name:24s} name={ch.name:14s} n={ch.data.size:7d} "
              f"fs={ch.fs:8.2f} max|err|={err:.3g}  {'ok' if good else 'MISMATCH'}")
    print(f"\n{len(written) - bad}/{len(written)} verified")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
