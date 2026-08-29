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


# The force channels captured by famos_forces.seq, and the .raw each comes from.
_FORCE_CHANNELS = ("FR_Fx_2", "FR_Fy_2", "FR_Fz_2",
                   "RR_Fx_1", "RR_Fy_1", "RR_Fz_1")


def _register_forces(raw_dir: Path) -> Dict[str, tuple]:
    """Stage F: the WFT force chain on the real recording.

    The force path is ``smo(0.1)`` then ``red(10)`` and no ``FiltLP`` -- that
    operator belongs to Latacc alone. Scoring a force channel through the accel
    chain would validate something the pipeline never runs.

    Compared in the file's own units, deliberately. The N-to-daN step and the
    decade correction are decisions this project makes and FAMOS does not, so
    folding them in here would score our unit policy rather than our signal
    processing, and a mismatch could not be attributed to either.
    """
    from dtt.ingestion.imc_reader import read_famos

    cases: Dict[str, tuple] = {}
    for ch_name in _FORCE_CHANNELS:
        f = raw_dir / f"{ch_name}.raw"
        if not f.exists():
            continue
        ch = read_famos(f)
        if ch is None or not ch.data.size:
            print(f"  ! could not read {f.name}")
            continue
        x, fs = ch.data, ch.fs
        cases[f"{ch_name}_smo"] = (x, lambda v, fs=fs: ops.smo(v, 0.1, fs))
        cases[f"{ch_name}_red"] = (
            x, lambda v, fs=fs: ops.red(ops.smo(v, 0.1, fs), 10))
        # No smoothing at all: isolates the reader from the operators.
        cases[f"{ch_name}_rawred"] = (x, lambda v: ops.red(v, 10))
    return cases


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


def load_outputs(path: Path, wanted=None) -> Dict[str, np.ndarray]:
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
    # the separator follows the machine's locale, so neither is assumed. The
    # layout is settled from the first few lines only: a real force export runs
    # to 1.2 GB, and probing that by re-parsing the whole file (let alone with
    # the python engine) is not a thing that finishes.
    head = []
    with open(path, "r", encoding="latin1", errors="replace") as fh:
        for _ in range(6):
            line = fh.readline()
            if not line:
                break
            head.append(line)

    sep, hdr_row = ",", 0
    for i, line in enumerate(head):
        for cand in (",", ";", "	"):
            names = [c.strip() for c in line.split(cand)]
            if len(names) > 2 and sum(bool(n) for n in names) > 2:
                sep, hdr_row = cand, i
                break
        else:
            continue
        break

    # FAMOS writes a units row directly under the channel names. Left in, every
    # channel shifts by one sample and the comparison silently misaligns.
    skip = []
    if len(head) > hdr_row + 1:
        cells = [c.strip() for c in head[hdr_row + 1].split(sep)]
        numeric = 0
        for c in cells:
            try:
                float(c)
                numeric += 1
            except ValueError:
                pass
        if numeric == 0:
            skip = [hdr_row + 1]

    # FAMOS pads the header names out to a fixed width, and usecols matches the
    # raw text, so the comparison has to strip before it decides.
    use = (lambda c: str(c).strip() in wanted) if wanted else None
    df = pd.read_csv(path, sep=sep, header=hdr_row, skiprows=skip,
                     usecols=use, engine="c", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

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


def quantum_ratio(err: np.ndarray, ref: np.ndarray, sig_figs: int = 6) -> float:
    """Worst error as a multiple of *that sample's* own export quantum.

    Six significant figures is a different absolute precision at every
    magnitude: 0.1 at 79,000 but 1.0 the moment a peak crosses 100,000. Judging
    a channel by one quantum taken from its typical value therefore mis-scores
    exactly the samples where the largest errors live -- the biggest ones. So
    each sample is compared against the quantum at its own magnitude, and the
    verdict is the worst of those ratios. At or below 0.5 means every sample
    agrees to within half a stored digit, which is as close as the file can
    record.
    """
    m = np.isfinite(err) & np.isfinite(ref) & (np.abs(ref) > 0)
    if not m.any():
        return 0.0
    q = 10.0 ** (np.floor(np.log10(np.abs(ref[m]))) - (sig_figs - 1))
    return float(np.max(np.abs(err[m]) / q))


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
            "mean": float(d.mean()), "r": r,
            "qratio": quantum_ratio(a - b, b)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("outputs", help="FAMOS results: a directory of .dat, or a .csv")
    ap.add_argument("--inputs", default=str(INPUTS))
    ap.add_argument("--skip", type=int, default=3000,
                    help="samples to drop at each end for the settled score")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="max abs error allowed on the settled region")
    ap.add_argument("--raw-dir", default=None,
                    help="folder of WFT .raw files, to score the force chain "
                         "(smo(0.1) -> red(10)) end to end against FAMOS")
    ap.add_argument("--raw", default=None,
                    help="the .raw file Stage 4 was run on, e.g. AccelY.raw; "
                         "scores our reader and operators against FAMOS end to end")
    args = ap.parse_args()

    _register()
    wft = _register_wft(Path(args.raw)) if args.raw else {}
    if args.raw and not wft:
        print(f"Could not read {args.raw} - Stage 4 will be skipped")
    CASES.update(wft)
    if args.raw_dir:
        forces = _register_forces(Path(args.raw_dir))
        if not forces:
            print(f"No WFT .raw files found in {args.raw_dir}")
        CASES.update(forces)

    src = pd.read_csv(args.inputs)
    inputs = {c: pd.to_numeric(src[c], errors="coerce").to_numpy(float)
              for c in src.columns}
    got = load_outputs(Path(args.outputs), wanted=set(CASES))
    if not got:
        print(f"No FAMOS channels found in {args.outputs}")
        return 2

    print(f"inputs : {len(inputs) - 1} channels x {len(src)} samples @ {FS:.0f} Hz")
    print(f"outputs: {len(got)} channels from {args.outputs}\n")
    print(f"{'channel':24s} {'n':>8s} {'max err':>12s} {'mean err':>12s} "
          f"{'r':>12s} {'err/q':>8s}  verdict")
    print("-" * 93)
    print("   err/q = max error as a fraction of the export quantum; "
          "<= 0.5 is the rounding floor")
    print()

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
        ratio = st["qratio"]
        # Half a quantum is the most a correctly-rounded value can differ by;
        # a little headroom on top, because the last stored digit of a filtered
        # value carries the rounding of everything that fed into it.
        ok = np.isfinite(st["max"]) and (st["max"] <= args.tol or ratio <= 0.75)
        (passed if ok else failed).append(name)
        print(f"{name:24s} {st['n']:8d} {st['max']:12.3e} {st['mean']:12.3e} "
              f"{st['r']:12.9f} {ratio:8.2f}  {'ok' if ok else 'MISS'}")

    print("-" * 93)
    print(f"{len(passed)} at or below the export's precision, {len(failed)} above, "
          f"{len(missing)} not exported")
    if missing:
        print("\nnot found in the export (run those stages, or ignore if skipped):")
        for m in missing:
            print(f"   {m}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
