"""Stage the all-channel validation inputs for imc FAMOS.

FAMOS dispatches on the file **extension**, not on the contents. The recorder's
``.raw`` files are already FAMOS format inside (``|CF,2,1,``), but handed a
``.raw`` FAMOS decides it is looking at bus data and asks for a *FlexRay
Description File (*.FRY)* instead of opening it. Copying them to ``.dat`` is the
whole fix, and it is the step that otherwise stops the validation on its first
click.

So this copies the channels `famos_allchannel.seq` needs into one folder with
the right extension, checks each one actually reads, and prints what to expect
on screen. Originals are never touched.

    python tools/stage_famos_inputs.py

Then follow golden_corpus/ALLCHANNEL_EXPORT.md from step 2.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dtt.ingestion.imc_reader import read_famos                  # noqa: E402

# The recording famos_allchannel.seq is written against. Deliberately the
# classic |CF folder: the newer imc3 recordings in "Raw data_no sanitisation"
# open in FAMOS but our own read_famos returns nothing for them, which would
# leave the comparison with a FAMOS side and no Python side.
DEFAULT_SOURCE = Path("raw data in .dat format/2006-08-25 09-10-46 (1)")
DEFAULT_DEST = Path("golden_corpus/allchannel_in")

# 12 wheel force/moment channels + the three aux inputs the recipe maps:
#   Long_acc = AccelX      Lat_acc = AccelY      Speed_kmph = Speed2D * 3.6
CHANNELS = [
    "FR_Fx_2", "FR_Fy_2", "FR_Fz_2", "FR_Mx_2", "FR_My_2", "FR_Mz_2",
    "RR_Fx_1", "RR_Fy_1", "RR_Fz_1", "RR_Mx_1", "RR_My_1", "RR_Mz_1",
    "AccelX", "AccelY", "Speed2D",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=str(DEFAULT_SOURCE),
                    help=f"recording folder (default: {DEFAULT_SOURCE})")
    ap.add_argument("--dest", default=str(DEFAULT_DEST),
                    help=f"where to stage the .dat copies (default: {DEFAULT_DEST})")
    args = ap.parse_args()

    src, dst = Path(args.source), Path(args.dest)
    if not src.is_dir():
        print(f"Source folder not found: {src}")
        return 2
    dst.mkdir(parents=True, exist_ok=True)

    ok, missing, unreadable = [], [], []
    n_samples = set()
    rates = set()

    for name in CHANNELS:
        f = src / f"{name}.raw"
        if not f.exists():
            missing.append(name)
            continue
        # Read before copying: a file FAMOS can open but we cannot is exactly
        # the case that wastes a licensed session, and it is cheap to catch now.
        try:
            ch = read_famos(f)
        except Exception as exc:                                 # noqa: BLE001
            unreadable.append(f"{name} ({type(exc).__name__}: {exc})")
            continue
        if ch is None or not ch.data.size:
            unreadable.append(f"{name} (reader returned nothing)")
            continue
        shutil.copyfile(f, dst / f"{name}.dat")
        ok.append(name)
        n_samples.add(int(ch.data.size))
        rates.add(round(float(ch.fs)))

    print(f"source : {src}")
    print(f"staged : {dst.resolve()}")
    print()
    print(f"{len(ok)} of {len(CHANNELS)} channels copied as .dat")
    if missing:
        print(f"  missing from the recording : {', '.join(missing)}")
    if unreadable:
        print("  present but unreadable     :")
        for u in unreadable:
            print(f"     {u}")

    if ok:
        print()
        print("In FAMOS you should see exactly these variable names "
              "(they come from inside the files, not the filenames):")
        for i in range(0, len(ok), 6):
            print("   " + "  ".join(ok[i:i + 6]))
        print()
        lo, hi = min(n_samples), max(n_samples)
        rate = sorted(rates)
        if lo == hi:
            print(f"Each one: {lo:,} samples at {rate[0]} Hz.")
        else:
            # A recorder stops its channels a sample apart; four of these end
            # one sample early. Harmless -- the scorer compares over the
            # shorter of the two arrays -- but worth saying, so a length that
            # is *not* within a sample or two stands out as the real problem.
            print(f"Each one: {lo:,}-{hi:,} samples at {rate[0]} Hz.")
            print(f"A {hi - lo}-sample spread across channels is the recorder "
                  f"stopping them a moment apart, and is fine.")
        print("A channel that is wildly shorter means the wrong file was opened.")

    print()
    print("Next: golden_corpus/ALLCHANNEL_EXPORT.md, step 2.")
    return 0 if ok and not missing and not unreadable else 1


if __name__ == "__main__":
    raise SystemExit(main())
