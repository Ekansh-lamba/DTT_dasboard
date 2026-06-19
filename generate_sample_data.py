"""
Synthetic WFT sample-data generator for the DTT platform.

Produces CSV files that match the real RLDA export format exactly
(38 columns, forces in Newtons, 100 Hz) but with a fully simulated drive
so every downstream screen — histograms, heatmaps, boxplots, rainflow —
has rich, non-trivial content.

Usage:
    python generate_sample_data.py                       # default 200k-row mixed drive
    python generate_sample_data.py --rows 120000 --name WFT_City --profile city
    python generate_sample_data.py --profile rough --name WFT_Rough_Road

Profiles:
    mixed   idle -> accel -> highway cruise -> cornering -> braking, repeated
    city    frequent stop/go, lower speeds, sharper events
    rough   highway speeds over a rough surface (large vertical dynamics)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CSV_DIR = Path(__file__).resolve().parent / "csv"
FS = 100.0  # Hz

# Exact column order from the real export.
COLUMNS = [
    "Time", "Altitude", "Dist",
    "FL_Fx", "FL_Fy", "FL_Fz", "FL_Mx", "FL_My", "FL_Mz", "FL_WS1",
    "FR_Fx", "FR_Fy", "FR_Fz", "FR_Mx", "FR_My", "FR_Mz", "FR_WS1",
    "Latacc", "Latacc_LPF", "Latitude", "Longacc", "Longitude",
    "RL_Fx", "RL_Fy", "RL_Fz", "RL_Mx", "RL_My", "RL_Mz", "RL_WS1",
    "RR_Fx", "RR_Fy", "RR_Fz", "RR_Mx", "RR_My", "RR_Mz", "RR_WS1",
    "Vehicle_Speed", "Yawrate",
]

# Per-wheel baseline static vertical load (N) and Fx/Fy character.
# Values chosen to land near the real dataset after the N->daN (/10) step.
WHEEL_BASE = {
    #         Fz_static   Fx_bias   Fy_bias   Mx_bias    My_bias   Mz_bias
    "FL": dict(fz=6080.0,  fx=180.0,  fy=240.0,  mx=-250.0,  my=180.0,  mz=-200.0),
    "FR": dict(fz=5700.0,  fx=150.0,  fy=190.0,  mx=-650.0,  my=290.0,  mz=-180.0),
    "RL": dict(fz=7050.0,  fx=-70.0,  fy=35.0,   mx=-12900.0, my=-470.0, mz=540.0),
    "RR": dict(fz=7100.0,  fx=-85.0,  fy=270.0,  mx=12200.0, my=-530.0, mz=-560.0),
}


def _speed_profile(n: int, profile: str, rng: np.random.Generator) -> np.ndarray:
    """Build a smooth, realistic vehicle-speed trace (km/h)."""
    t = np.arange(n) / FS
    if profile == "city":
        base = 28 + 22 * np.sin(2 * np.pi * t / 95.0) + 12 * np.sin(2 * np.pi * t / 31.0)
        stops = (np.sin(2 * np.pi * t / 47.0) > 0.85)
        base[stops] *= 0.15
    elif profile == "rough":
        base = 85 + 18 * np.sin(2 * np.pi * t / 160.0)
    else:  # mixed
        base = (45
                + 35 * np.sin(2 * np.pi * t / 140.0)
                + 18 * np.sin(2 * np.pi * t / 53.0)
                + 8 * np.sin(2 * np.pi * t / 17.0))
    base += rng.normal(0, 1.2, n)
    return np.clip(base, 0.0, 140.0)


def _lateral_profile(n: int, profile: str, rng: np.random.Generator) -> np.ndarray:
    """Lateral acceleration (g) — drives cornering loads, Fy, yaw."""
    t = np.arange(n) / FS
    amp = {"city": 0.45, "rough": 0.20, "mixed": 0.32}.get(profile, 0.32)
    lat = (amp * np.sin(2 * np.pi * t / 23.0)
           + 0.18 * np.sin(2 * np.pi * t / 7.3)
           + rng.normal(0, 0.03, n))
    return lat


def _road_roughness(n: int, profile: str, rng: np.random.Generator) -> np.ndarray:
    """Normalised vertical road excitation in [~ -1, 1], plus pothole spikes."""
    sigma = {"city": 0.5, "rough": 1.0, "mixed": 0.6}.get(profile, 0.6)
    # Coloured noise: integrate white noise then high-pass by differencing.
    white = rng.normal(0, 1, n)
    smooth = np.convolve(white, np.ones(15) / 15, mode="same")
    rough = sigma * (0.6 * smooth + 0.4 * white)
    # Occasional bumps / potholes
    n_events = max(1, n // 8000)
    for _ in range(n_events):
        c = rng.integers(0, n)
        w = rng.integers(8, 40)
        lo, hi = max(0, c - w), min(n, c + w)
        shape = np.hanning(hi - lo) * rng.choice([-1, 1]) * rng.uniform(2.5, 6.0)
        rough[lo:hi] += shape
    return rough


def generate(rows: int, profile: str, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = rows
    t = np.arange(n) / FS

    speed_kmh = _speed_profile(n, profile, rng)
    speed_mps = speed_kmh / 3.6
    dist = np.cumsum(speed_mps) / FS                       # metres
    long_acc = np.gradient(speed_mps) * FS / 9.81          # g
    lat_acc = _lateral_profile(n, profile, rng)
    lat_acc_lpf = np.convolve(lat_acc, np.ones(25) / 25, mode="same")
    yaw = lat_acc / np.clip(speed_mps, 1.0, None) * 9.81   # rad/s approx
    rough = _road_roughness(n, profile, rng)
    altitude = 180 + 12 * np.sin(2 * np.pi * t / 600.0)

    data = {
        "Time": np.round(t, 2),
        "Altitude": np.round(altitude, 3),
        "Dist": np.round(dist, 3),
        "Latacc": np.round(lat_acc, 5),
        "Latacc_LPF": np.round(lat_acc_lpf, 5),
        "Latitude": 0.0,
        "Longacc": np.round(long_acc, 5),
        "Longitude": 0.0,
        "Vehicle_Speed": np.round(speed_kmh, 3),
        "Yawrate": np.round(yaw, 5),
    }

    # Load transfer: braking pushes load forward, cornering side to side.
    front_share = 0.5 - 0.18 * long_acc          # >0.5 under braking
    left_share = 0.5 - 0.30 * lat_acc

    for wheel, base in WHEEL_BASE.items():
        is_front = wheel[0] == "F"
        is_left = wheel[1] == "L"
        fshare = front_share if is_front else (1.0 - front_share)
        sshare = left_share if is_left else (1.0 - left_share)

        # Vertical: static load redistributed by fore/aft + lateral transfer
        # (each share ~0.5 at nominal, so 2*share ~1.0 -> fz ~ static baseline)
        fz = (base["fz"] * (2 * fshare) * (2 * sshare)
              + base["fz"] * 0.035 * rough
              + rng.normal(0, base["fz"] * 0.006, n))
        fz = np.clip(fz, base["fz"] * 0.3, base["fz"] * 1.9)

        # Longitudinal: braking/traction + per-wheel bias + noise
        fx = (base["fx"]
              + (-1400.0 if is_front else -600.0) * long_acc
              + 250.0 * rough * (0.3 if is_front else 0.9)
              + rng.normal(0, 60 if is_front else 180, n))

        # Lateral: cornering + bias + noise
        fy = (base["fy"]
              + (900.0 if is_left else -900.0) * lat_acc
              + rng.normal(0, 120, n))

        # Moments roughly track forces around per-wheel bias.
        mx = base["mx"] + 0.02 * fz + rng.normal(0, 30, n)
        my = base["my"] + 0.5 * fx + rng.normal(0, 20, n)
        mz = base["mz"] + 0.3 * fy + rng.normal(0, 25, n)
        ws1 = rng.normal(0, 0.4, n)

        data[f"{wheel}_Fx"] = np.round(fx, 4)
        data[f"{wheel}_Fy"] = np.round(fy, 4)
        data[f"{wheel}_Fz"] = np.round(fz, 4)
        data[f"{wheel}_Mx"] = np.round(mx, 4)
        data[f"{wheel}_My"] = np.round(my, 4)
        data[f"{wheel}_Mz"] = np.round(mz, 4)
        data[f"{wheel}_WS1"] = np.round(ws1, 4)

    return pd.DataFrame(data)[COLUMNS]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic WFT sample CSV")
    ap.add_argument("--rows", type=int, default=200_000, help="Number of samples (100 Hz)")
    ap.add_argument("--profile", choices=["mixed", "city", "rough"], default="mixed")
    ap.add_argument("--name", default=None, help="Output filename stem")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    stem = args.name or f"WFT_Synthetic_{args.profile}"
    CSV_DIR.mkdir(exist_ok=True)
    out = CSV_DIR / f"{stem}.csv"

    df = generate(args.rows, args.profile, args.seed)
    df.to_csv(out, index=False)

    dur = args.rows / FS
    print(f"Wrote {out}")
    print(f"  rows={len(df):,}  cols={len(df.columns)}  duration={dur:.0f}s ({dur/60:.1f} min)")
    print(f"  profile={args.profile}  mean speed={df['Vehicle_Speed'].mean():.1f} km/h"
          f"  distance={df['Dist'].iloc[-1]/1000:.2f} km")


if __name__ == "__main__":
    main()
