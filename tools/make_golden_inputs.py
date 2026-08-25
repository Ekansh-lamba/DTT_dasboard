#!/usr/bin/env python3
"""TASK 2: generate the golden-corpus INPUT signals for the FAMOS trial run.

Run:  python tools/make_golden_inputs.py

Why Python generates these rather than the FAMOS sequence
---------------------------------------------------------
Two reasons, both about not wasting a trial licence:

1. The only FAMOS functions used in the sequence are ones observed working in
   the customer's own imc file -- FiltLP, smo, red. Synthesis functions (ramp,
   noise, chirp generators) would be names I cannot verify without a licence,
   and a wrong name burns trial time debugging a sequence instead of capturing
   data.
2. FAMOS's RNG is not reproducible in Python. If FAMOS generated the noise we
   could never regenerate the exact input, so the corpus would only ever be as
   good as its exported copy. Generating here makes the input bit-exact and
   permanently reproducible from a seed.

Precision warning
-----------------
FAMOS's default CSV export is SIX SIGNIFICANT FIGURES (verified against
`csv/RLDA WFT PV data sample.csv`: 6121.62, 169.672, -22.3716, 0.180000). That
caps any comparison at ~1e-6 relative. The impulse amplitude below is scaled up
so its response still lands in a well-resolved range even on that path, but the
corpus should be exported as imc binary if at all possible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "golden_corpus"
OUT_CSV = OUT_DIR / "famos_golden_inputs.csv"

# --- Acquisition geometry, matching the real WFT recordings ----------------
FS_HZ = 1000.0
DX = 1.0 / FS_HZ
DURATION_S = 120.0                  # one rectangular file; see note below
N = int(round(DURATION_S * FS_HZ))

# --- Signal parameters -----------------------------------------------------
IMPULSE_INDEX = 10_000              # t = 10.000 s exactly
# FiltLP(4, 5 Hz) at 1000 Hz has b0 ~ 1.5e-08, so a unit impulse would produce a
# response far below a 6-sig-fig export's resolution. Scaling the impulse puts
# the response peak near 1.0. Python divides it back out.
IMPULSE_AMPLITUDE = 1000.0
STEP_INDEX = 10_000                 # step at t = 10.000 s
STEP_AMPLITUDE = 1.0
DC_LEVEL = 1.0
CHIRP_F0_HZ = 0.1
CHIRP_F1_HZ = 100.0
NOISE_SEED = 42
NOISE_SIGMA = 1.0
SINE_FREQS_HZ = (2.5, 5.0, 10.0)
SINE_AMPLITUDE = 1.0

# Full float64 round-trip. %.17g is the shortest form guaranteed to reproduce
# the exact double, so the input side of the corpus loses nothing.
FLOAT_FMT = "%.17g"


def build() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    t = np.arange(N, dtype=np.float64) * DX
    sig: dict[str, np.ndarray] = {}

    # 1. unit impulse at a known index -- identifies the filter itself
    imp = np.zeros(N, dtype=np.float64)
    imp[IMPULSE_INDEX] = IMPULSE_AMPLITUDE
    sig["gc_impulse"] = imp

    # 2. step at a known index -- clean step response, away from the record
    #    start so it is not entangled with initialisation
    step = np.zeros(N, dtype=np.float64)
    step[STEP_INDEX:] = STEP_AMPLITUDE
    sig["gc_step"] = step

    # 3. DC -- this is the one that pins the INITIALISATION convention, because
    #    it is constant from sample 0. Step-response init gives y[0]=1.0
    #    throughout; zero state ramps up from ~0. It also gives passband gain,
    #    which must be exactly 1.
    sig["gc_dc"] = np.full(N, DC_LEVEL, dtype=np.float64)

    # 4. linear chirp -- magnitude AND phase response across frequency. Phase is
    #    the discriminator between single-pass and forward-backward filtering.
    #    Linear-in-frequency sweep: f(t) = f0 + (f1-f0)*t/T, phase = 2*pi*integral
    k = (CHIRP_F1_HZ - CHIRP_F0_HZ) / DURATION_S
    phase = 2.0 * np.pi * (CHIRP_F0_HZ * t + 0.5 * k * t * t)
    sig["gc_chirp"] = np.sin(phase)

    # 5. white noise, fixed seed -- broadband statistical check on the chain
    rng = np.random.default_rng(NOISE_SEED)
    sig["gc_noise"] = rng.standard_normal(N) * NOISE_SIGMA

    # 6. sines -- 5 Hz settles the cutoff convention (is 5 Hz the -3 dB point,
    #    the passband edge, or the stopband edge?); 2.5 and 10 Hz bracket it
    for f in SINE_FREQS_HZ:
        name = f"gc_sine_{str(f).replace('.', 'p')}"
        sig[name] = SINE_AMPLITUDE * np.sin(2.0 * np.pi * f * t)

    return t, sig


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    t, sig = build()

    names = list(sig)
    data = np.column_stack([t] + [sig[n] for n in names])
    header = ",".join(["Time"] + names)
    np.savetxt(OUT_CSV, data, delimiter=",", header=header, comments="",
               fmt=FLOAT_FMT)

    b, a = butter(4, 5.0 / (0.5 * FS_HZ), btype="low")
    print("=" * 78)
    print("GOLDEN CORPUS INPUTS")
    print("=" * 78)
    print(f"  written        : {OUT_CSV}")
    print(f"  size           : {OUT_CSV.stat().st_size / 1e6:.1f} MB")
    print(f"  samples        : {N:,} per channel  ({DURATION_S:g} s at {FS_HZ:g} Hz)")
    print(f"  dx             : {DX:.9g} s")
    print(f"  channels       : {len(names)}")
    for n in names:
        v = sig[n]
        print(f"    {n:<16} min {v.min():>12.6g}  max {v.max():>12.6g}  "
              f"rms {np.sqrt(np.mean(v**2)):>10.6g}")
    print(f"\n  impulse at     : index {IMPULSE_INDEX:,} (t = {IMPULSE_INDEX*DX:.3f} s), "
          f"amplitude {IMPULSE_AMPLITUDE:g}")
    print(f"  step at        : index {STEP_INDEX:,} (t = {STEP_INDEX*DX:.3f} s)")
    print(f"  noise seed     : {NOISE_SEED} (numpy default_rng, reproducible)")
    print(f"\n  designed FiltLP(4, 5 Hz) at {FS_HZ:g} Hz has b0 = {b[0]:.6e}")
    print(f"  -> a UNIT impulse would peak near {b[0]:.2e}, far below a 6-sig-fig")
    print(f"     export. Scaled by {IMPULSE_AMPLITUDE:g} so the response is resolvable;")
    print( "     divide back out in analysis.")
    print("\n  NOTE: all channels are 120 s so the file is rectangular and needs")
    print("        one import action. The spec asked for 60 s on impulse/step;")
    print("        longer only adds settling observation, it removes nothing.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
