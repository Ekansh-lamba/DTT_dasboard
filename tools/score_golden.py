"""Score the Python FAMOS operators against a golden corpus captured from FAMOS.

Run this after ``golden_corpus/famos_golden.seq`` has been executed in a licensed
imc FAMOS and its results exported. It re-derives every output channel with
:mod:`famos.ops` and reports how far each one sits from what FAMOS produced.

    python tools/score_golden.py golden_corpus/famos_out
    python tools/score_golden.py golden_corpus/famos_out.csv

The directory form is preferred: FAMOS ``.dat`` files are binary, so they carry
the full stored precision. An ASCII export rounds, and that rounding became the
floor on the last smo measurement (0.0056 N) - which is the wrong side of the
question when two candidate kernels are being told apart at 1e-6.

Exit status is 1 if any operator misses its tolerance, so this can gate a build.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from famos import ops                                            # noqa: E402
from dtt.ingestion.imc_reader import read_famos_all              # noqa: E402

FS = 1000.0                       # golden inputs are dx = 0.001 s
INPUTS = Path("golden_corpus/famos_golden_inputs.csv")

# FAMOS output channel  ->  (input channel, how we reproduce it)
CASES: Dict[str, tuple] = {}


def _register() -> None:
    """Build the case table, mirroring famos_golden.seq stage for stage."""
    sigs = ["impulse", "step", "dc", "chirp", "noise",
            "sine_2p5", "sine_5p0", "sine_10p0"]
    short = {"sine_2p5": "sine2p5", "sine_5p0": "sine5p0", "sine_10p0": "sine10p0"}

    for s in sigs:
        src = f"gc_{s}"
        tag = short.get(s, s)
        # Stage 1 - FiltLP alone
        CASES[f"gc_{tag}_lpf"] = (src, lambda x: ops.filtlp(x, 5.0, 4, FS))
        # Stage 2 - FiltLP -> smo(0.5)
        CASES[f"gc_{tag}_smo"] = (
            src, lambda x: ops.smo(ops.filtlp(x, 5.0, 4, FS), 0.5, FS))
        # Stage 3 - FiltLP -> smo(0.5) -> red(10)
        CASES[f"gc_{tag}_red"] = (
            src, lambda x: ops.red(ops.smo(ops.filtlp(x, 5.0, 4, FS), 0.5, FS), 10))

    for s in ["impulse", "step", "dc", "chirp", "noise"]:
        src = f"gc_{s}"
        # smo in isolation - the only measurement not contaminated by FiltLP
        CASES[f"gc_{s}_smoonly"] = (src, lambda x: ops.smo(x, 0.5, FS))
        CASES[f"gc_{s}_smo01"] = (src, lambda x: ops.smo(x, 0.1, FS))

    # red alone on the impulse states the decimation phase outright
    CASES["gc_impulse_redonly"] = ("gc_impulse", lambda x: ops.red(x, 10))


def _register_wft(raw_path: Path) -> Dict[str, tuple]:
    """Stage 4: the production chain on a real WFT record.

    This is the only case that puts our *reader* under test. Every gc_ channel
    starts from a signal Python generated and FAMOS merely imported, so a fault
    in the imc byte parsing could not show up there. Here both sides start from
    the same file on disk and FAMOS does its own reading, so the comparison
    covers ingestion and arithmetic together -- which is what "the processed
    output matches FAMOS" actually means.
    """
    from dtt.ingestion.imc_reader import read_famos

    ch = read_famos(raw_path)
    if ch is None or not ch.data.size:
        return {}
    x = ch.data
    fs = ch.fs
    return {
        "wft_accely_lpf":   (x, lambda v: ops.filtlp(v, 5.0, 4, fs)),
        "wft_accely_smo":   (x, lambda v: ops.smo(ops.filtlp(v, 5.0, 4, fs), 0.5, fs)),
        "wft_accely_red":   (x, lambda v: ops.red(
            ops.smo(ops.filtlp(v, 5.0, 4, fs), 0.5, fs), 10)),
        "wft_accely_smo01": (x, lambda v: ops.smo(v, 0.1, fs)),
    }


def load_outputs(path: Path) -> Dict[str, np.ndarray]:
    """FAMOS results, from a directory of .dat files or a single CSV."""
    out: Dict[str, np.ndarray] = {}
    if path.is_dir():
        for f in sorted(path.glob("*")):
            if f.suffix.lower() not in (".dat", ".raw"):
                continue
            try:
                chans = read_famos_all(f)
            except Exception as exc:                             # noqa: BLE001
                print(f"  ! could not read {f.name}: {exc}")
                continue
            # FAMOS writes every selected variable into one file, so a single
            # .dat routinely holds the whole export.
            for ch in chans:
                if ch.data.size:
                    # the name inside the file wins; the filename is a fallback
                    out[(ch.raw_name or f.stem).strip()] = ch.data
        return out

    # A FAMOS ASCII export may carry metadata lines above the channel names, and
    # the separator follows the machine's locale, so neither is assumed.
    df = None
    for skip in (0, 1, 2, 3, 4, 5):
        for sep in (",", ";", "	"):
            try:
                cand = pd.read_csv(path, skiprows=skip, sep=sep, engine="python")
            except Exception:                                    # noqa: BLE001
                continue
            names = [str(c).strip() for c in cand.columns]
            if sum(n.startswith("gc_") for n in names) >= 2:
                cand.columns = names
                df = cand
                break
        if df is not None:
            break
    if df is None:
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
    # FAMOS writes a units row directly under the channel names. Left in, every
    # channel shifts by one sample and the whole comparison silently misaligns,
    # so drop any leading row that holds no numbers at all.
    if len(df):
        first = pd.to_numeric(df.iloc[0], errors="coerce")
        if not np.isfinite(first.to_numpy(dtype=float)).any():
            df = df.iloc[1:].reset_index(drop=True)
    for c in df.columns:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        # One wide export, one shared x-axis. A channel FAMOS decimated is
        # written *sparsely* onto that axis -- for red(x,10), one value every
        # tenth row with blanks between -- so the column has to be compacted
        # back to the channel's own rate before it means anything. Comparing
        # across the blanks reads as a total mismatch (r ~ 0) while the data is
        # in fact identical, which is a very convincing way to be wrong.
        finite = np.flatnonzero(np.isfinite(v))
        if finite.size > 2:
            step = np.diff(finite)
            if step[0] > 1 and np.all(step == step[0]):
                v = v[finite]                       # regular stride: compact
            else:
                v = v[:finite[-1] + 1]              # merely padded at the end
        out[str(c).strip()] = v
    return out


def score(ours: np.ndarray, theirs: np.ndarray, skip: int = 0) -> dict:
    """Max/mean absolute error and correlation over the comparable region."""
    n = min(ours.size, theirs.size)
    a, b = ours[:n], theirs[:n]
    if skip:
        # A causal filter's startup transient is real output, not error, but it
        # swamps the comparison; report with and without so neither hides.
        a, b = a[skip:n - skip or None], b[skip:n - skip or None]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return {"n": 0, "max": float("nan"), "mean": float("nan"), "r": float("nan")}
    a, b = a[m], b[m]
    d = np.abs(a - b)
    r = (float(np.corrcoef(a, b)[0, 1])
         if np.std(a) > 0 and np.std(b) > 0 else float("nan"))
    return {"n": int(a.size), "max": float(d.max()),
            "mean": float(d.mean()), "r": r}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("outputs", help="FAMOS results: a directory of .dat, or a .csv")
    ap.add_argument("--inputs", default=str(INPUTS))
    ap.add_argument("--skip", type=int, default=3000,
                    help="samples to drop at each end for the settled score")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="max abs error allowed on the settled region")
    ap.add_argument("--raw", default=None,
                    help="the .raw file Stage 4 was run on, e.g. AccelY.raw; "
                         "scores our reader and operators against FAMOS end to end")
    args = ap.parse_args()

    _register()
    wft = _register_wft(Path(args.raw)) if args.raw else {}
    if args.raw and not wft:
        print(f"Could not read {args.raw} - Stage 4 will be skipped")
    CASES.update(wft)

    src = pd.read_csv(args.inputs)
    inputs = {c: pd.to_numeric(src[c], errors="coerce").to_numpy(float)
              for c in src.columns}
    got = load_outputs(Path(args.outputs))
    if not got:
        print(f"No FAMOS channels found in {args.outputs}")
        return 2

    print(f"inputs : {len(inputs) - 1} channels x {len(src)} samples @ {FS:.0f} Hz")
    print(f"outputs: {len(got)} channels from {args.outputs}\n")
    print(f"{'channel':24s} {'n':>8s} {'max err':>12s} {'mean err':>12s} "
          f"{'r':>12s}  verdict")
    print("-" * 84)

    missing, failed, passed = [], [], []
    for name, (src_name, fn) in sorted(CASES.items()):
        if name not in got:
            missing.append(name)
            continue
        if isinstance(src_name, str):
            if src_name not in inputs:
                missing.append(f"{name} (input {src_name} absent)")
                continue
            src_data = inputs[src_name]
        else:
            src_data = src_name                 # Stage 4 hands the array itself
        try:
            ours = np.asarray(fn(src_data), dtype=float)
        except Exception as exc:                                 # noqa: BLE001
            print(f"{name:24s} {'':>8s} {'':>12s} {'':>12s} {'':>12s}  ERROR {exc}")
            failed.append(name)
            continue

        # decimated channels are 10x shorter, so the transient skip scales too
        dec = name.endswith("_red") or name.endswith("_redonly")
        s = max(0, args.skip // (10 if dec else 1))
        st = score(ours, got[name], skip=s)
        ok = np.isfinite(st["max"]) and st["max"] <= args.tol
        (passed if ok else failed).append(name)
        print(f"{name:24s} {st['n']:8d} {st['max']:12.3e} {st['mean']:12.3e} "
              f"{st['r']:12.9f}  {'ok' if ok else 'MISS'}")

    print("-" * 84)
    print(f"{len(passed)} within {args.tol:g}, {len(failed)} outside, "
          f"{len(missing)} not exported")
    if missing:
        print("\nnot found in the export (run those stages, or ignore if skipped):")
        for m in missing:
            print(f"   {m}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
