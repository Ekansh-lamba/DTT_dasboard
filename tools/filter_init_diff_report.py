#!/usr/bin/env python3
"""TASK 3 artifact: which archived studies move when FiltLP init is corrected.

Run:  python tools/filter_init_diff_report.py

This reports. It does not re-issue anything.

Why this is analytic rather than a literal re-run
-------------------------------------------------
Two facts make an exact answer possible without reprocessing:

1. The difference between the two conventions is the filter's ZERO-INPUT
   response from the state `zi = sosfilt_zi(sos) * x[0]`. It therefore depends
   on the first sample and the filter alone -- not on the rest of the record:

       y_famos[n] - y_legacy[n]  ==  x0 * g[n]        (verified to 5.2e-16)

   so the divergence for any study is a single multiplication, and the settling
   window is a property of the filter, identical for every study at a given fs.

2. A literal re-run is impossible for every study archived before the raw
   snapshot was widened. `raw_data.csv` used to store the mandatory FORCE
   channels only (`_force_frame` in dtt/pipeline.py kept
   `run_channels.mandatory_channels`); it now stores every channel the recipe
   conditions (`_raw_reference_frame`, keyed off
   `dtt.preprocessing.conditions_channel`), `Latacc` included. Studies ingested
   before that change still retain no raw `Latacc`. Re-deriving it from the processed column
   would mean inverting a low-pass, which amplifies quantisation noise without
   bound. The analytic route is the only exact one available.

Blast radius
------------
Exactly one channel in the imc recipe sets `apply_filter=True`: `Latacc`
(alias `Lat_acc`). Every WFT force and moment channel gets `smo(0.1)` and no
FiltLP, and `smo` is a convolution with no recursive state, so it cannot be
affected by an initial-condition change. Consequently no force-derived result
moves -- including the Gxy/Gxyf severity figures, which are computed from
RL_Fx/RL_Fy/RL_Fz.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, sosfilt_zi

from dtt.preprocessing import FAMOS_LPF_CUTOFF, FAMOS_LPF_ORDER

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "dtt" / "outputs"
AFFECTED_CHANNELS = ("Latacc", "Lat_acc", "Latacc_LPF")
REPORT_PATH = ROOT / "filter_init_diff_report.json"
# Level below which the divergence is smaller than the FAMOS CSV export's own
# rounding floor, i.e. not observable in any delivered artifact.
NEGLIGIBLE = 1e-6


def divergence_kernel(fs: float, n: int = 4000) -> np.ndarray:
    """g[n]: the exact per-unit-x0 divergence between the two conventions."""
    sos = butter(FAMOS_LPF_ORDER, FAMOS_LPF_CUTOFF / (0.5 * fs),
                 btype="low", output="sos")
    g, _ = sosfilt(sos, np.zeros(n), zi=sosfilt_zi(sos) * 1.0)
    return g


def settling_samples(g: np.ndarray, tol: float) -> int:
    below = np.abs(g) < tol
    return int(np.argmax(below)) if below.any() else g.size


def main() -> int:
    rows = []
    studies = sorted(p for p in OUTPUTS.iterdir() if p.is_dir())
    print(f"scanning {len(studies)} archived studies under {OUTPUTS}\n")

    kernels: dict[float, np.ndarray] = {}

    for st in studies:
        csv = st / "processed_data.csv"
        if not csv.exists():
            rows.append({"study": st.name, "status": "no processed_data.csv"})
            continue
        head = pd.read_csv(csv, nrows=0)
        cols = [c.strip() for c in head.columns]
        present = [c for c in AFFECTED_CHANNELS if c in cols]
        if not present:
            rows.append({"study": st.name, "status": "unaffected",
                         "reason": "no FiltLP'd channel present"})
            continue

        use = present[0]
        df = pd.read_csv(csv, usecols=lambda c: c.strip() in ("Time", use))
        df.columns = [c.strip() for c in df.columns]
        t = pd.to_numeric(df["Time"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(df[use], errors="coerce").to_numpy(float)
        fs = 1.0 / float(np.median(np.diff(t[:2000])))

        if fs not in kernels:
            kernels[fs] = divergence_kernel(fs)
        g = kernels[fs]

        # x0 proxy: the local signal level at the record start. Latacc is a
        # low-frequency channel, so the median of the first second is a sound
        # estimate of the sample FiltLP actually started from. Labelled a proxy
        # because the true raw x0 was not retained (see module docstring).
        head_n = max(1, int(round(fs)))
        x0_proxy = float(np.nanmedian(y[:head_n]))
        max_div = abs(x0_proxy) * float(np.abs(g).max())

        rows.append({
            "study": st.name,
            "status": "AFFECTED" if max_div > NEGLIGIBLE else "affected-negligible",
            "channel": use,
            "fs_hz": round(fs, 4),
            "x0_proxy": round(x0_proxy, 6),
            "max_divergence": round(max_div, 6),
            "unit": "m/s^2",
            "window_1e3_s": round(settling_samples(g, 1e-3) / fs, 3),
            "window_1e6_s": round(settling_samples(g, 1e-6) / fs, 3),
            "pct_of_record": round(100.0 * (settling_samples(g, 1e-6) / fs)
                                   / max(t[-1] - t[0], 1e-9), 6),
        })

    affected = [r for r in rows if r.get("status") == "AFFECTED"]
    negligible = [r for r in rows if r.get("status") == "affected-negligible"]
    clean = [r for r in rows if r.get("status") == "unaffected"]

    print("=" * 78)
    print("FiltLP INITIAL-CONDITION DIFF REPORT")
    print("=" * 78)
    print(f"  studies scanned            : {len(studies)}")
    print(f"  contain a FiltLP'd channel : {len(affected) + len(negligible)}")
    print(f"  materially affected        : {len(affected)}")
    print(f"  affected below {NEGLIGIBLE:g}       : {len(negligible)}")
    print(f"  unaffected (no such chan)  : {len(clean)}")
    print("\n  NOT affected, in any study: every WFT force/moment channel, and")
    print("  therefore every severity/fatigue figure derived from them (Gxy,")
    print("  Gxyf use RL_Fx/RL_Fy/RL_Fz). smo has no recursive state.\n")

    if affected:
        print(f"  {'study':<24}{'chan':<12}{'fs':>8}{'x0~':>10}"
              f"{'max div':>10}{'window(s)':>11}{'% of rec':>10}")
        for r in sorted(affected, key=lambda d: -d["max_divergence"]):
            print(f"  {r['study']:<24}{r['channel']:<12}{r['fs_hz']:>8.1f}"
                  f"{r['x0_proxy']:>10.4f}{r['max_divergence']:>10.4f}"
                  f"{r['window_1e6_s']:>11.3f}{r['pct_of_record']:>10.4f}")
    else:
        print("  no study is materially affected.")

    REPORT_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n  full report written to: {REPORT_PATH}")
    print("\n  NO OUTPUT HAS BEEN RE-ISSUED. This is the decision artifact only.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
