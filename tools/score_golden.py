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
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from famos import ops                                            # noqa: E402
# The all-channel cross-check core lives in the library, not here: the GUI's
# "Validate vs FAMOS..." action calls the same code, and the packaged app
# bundles dtt/ but not tools/. This file keeps the CLI, the synthetic corpus
# case table, and the exit status.
from dtt.validation.famos_validation import (                    # noqa: E402,F401
    crosscheck_all_channels, load_outputs, quantum_ratio, score,
)

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


def run_all_channels(args) -> int:
    """CLI front end for :func:`crosscheck_all_channels`.

    The measurement lives in the library so the GUI's "Validate vs FAMOS..."
    action and this command cannot disagree about what they scored or what
    counts as a pass. Everything here is printing and the exit status.
    """
    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_dir():
        print(f"--raw-dir is not a folder: {raw_dir}")
        return 2
    cut = None
    if args.cut:
        a, _, b = args.cut.partition(":")
        cut = (int(a or 0), int(b) if b else None)

    export_paths = [Path(p) for p in args.outputs_list]
    print(f"recording: {raw_dir}")
    print(f"export   : {', '.join(str(p) for p in export_paths)}")
    if cut:
        print(f"cut      : samples {cut[0]}:{cut[1]}  (applied before "
              f"conditioning, exactly as the sequence does)")

    res = crosscheck_all_channels(export_paths, raw_dir, cut=cut,
                                  skip=args.skip, tol=args.tol,
                                  report_path=args.report)
    if not res.rows and not res.skipped:
        print("No conditionable channels found in the recording.")
        return 2

    hdr = (f"{'channel':22s} {'stage':8s} {'n':>9s} {'max err':>12s} "
           f"{'mean err':>12s} {'r':>12s} {'err/q':>7s} {'within':>8s}  verdict")
    print()
    print(hdr)
    print("-" * len(hdr))
    print("   err/q  = worst error as a fraction of that sample's own export "
          "quantum; <= 0.5 is the rounding floor")
    print("   within = % of samples inside that floor")
    print()
    for r in res.rows:
        print(f"{r['channel']:22s} {r.get('stage', ''):8s} "
              f"{r.get('n', 0):9d} {r.get('max_err', ''):>12s} "
              f"{r.get('mean_err', ''):>12s} {r.get('r', ''):>12s} "
              f"{r.get('err_q', ''):>7s} {r.get('within_pct', ''):>8s}  "
              f"{r['verdict']}")

    if res.direction:
        print("\nLatacc / Latacc_LPF direction (both ways, measured):")
        for d in res.direction:
            print(f"   {d}")

    print("-" * len(hdr))
    print(f"{res.n_ok} at or below the export's precision, {res.n_miss} above, "
          f"{len(res.not_found)} not in the export, "
          f"{len(res.skipped)} channels not registered")
    lines = res.not_compared_lines()
    if lines:
        print("\nNot compared, and why:")
        for line in lines:
            print(f"   {line}")

    if res.report_path:
        print(f"\nreport: {res.report_path}  and  "
              f"{res.report_path.with_suffix('.md')}")
    return 1 if res.n_miss else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("outputs", nargs="+",
                    help="FAMOS results: a directory of .dat, or one or more "
                         ".csv exports (the 1000 Hz and 100 Hz files of an "
                         "all-channel run are usually separate)")
    ap.add_argument("--inputs", default=None,
                    help=f"synthetic corpus inputs (default {INPUTS}); only "
                         f"needed for the gc_* operator corpus, and not read "
                         f"at all by --all-channels")
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
    ap.add_argument("--all-channels", action="store_true",
                    help="score EVERY channel in --raw-dir at every recipe "
                         "stage, matched canonically to the export, instead "
                         "of the synthetic corpus and the six named forces")
    ap.add_argument("--cut", default=None, metavar="A:B",
                    help="sample range the sequence's own Cut() used, e.g. "
                         "0:600000 — applied before conditioning on our side "
                         "too, so both sides' filter start-up begins on the "
                         "same sample")
    ap.add_argument("--report", default=None, metavar="PATH.csv",
                    help="write the per-channel table to PATH.csv and the "
                         "reasons alongside it as PATH.md")
    args = ap.parse_args()
    args.outputs_list = list(args.outputs)
    args.outputs = args.outputs_list[0]

    if args.all_channels:
        if not args.raw_dir:
            print("--all-channels needs --raw-dir: the recording is what the "
                  "channel list and the comparison both come from")
            return 2
        return run_all_channels(args)

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

    # The synthetic corpus is 15 MB and gitignored, so it is routinely absent
    # on a machine that only wants the real-recording stages. Reading it
    # unconditionally made those runs die on a file they never needed.
    inputs: Dict[str, np.ndarray] = {}
    inputs_path = Path(args.inputs) if args.inputs else INPUTS
    if inputs_path.exists():
        src = pd.read_csv(inputs_path)
        inputs = {c: pd.to_numeric(src[c], errors="coerce").to_numpy(float)
                  for c in src.columns}
    elif args.inputs:
        print(f"--inputs not found: {inputs_path}")
        return 2
    else:
        print(f"note: {INPUTS} absent — the gc_* corpus stages will be "
              f"reported as not exported. Run tools/make_golden_inputs.py to "
              f"regenerate it.")
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
