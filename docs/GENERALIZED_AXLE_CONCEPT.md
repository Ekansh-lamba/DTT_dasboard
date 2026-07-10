# Concrete Concept — Generalized Multi-Axle WFT Platform

## The problem

The current pipeline hardcodes **four wheel positions** (`FL, FR, RL, RR`) and
**12 channels** — a 2-axle passenger car. Real commercial vehicles break this:

| Vehicle | Config | Axles | Wheel positions |
|---|---|---|---|
| Passenger car | 4x2 | 2 | 4 (FL FR RL RR) |
| LCV | 4x2 | 2 | 4–6 (dual rear) |
| Rigid truck | 6x2 / 6x4 | 3 | 6–10 (tandem, duals) |
| Heavy truck | 8x4 | 4 | 8–12 |
| Bus | 4x2 / 6x2 | 2–3 | 4–8 |
| Trailer | — | 1–3 | 2–12 |

And the raw data proves naming is inconsistent: one recording uses
`_fl/_fr/_rl/_rr`; another uses `FR_*_2 / RR_*_1` (only two WFTs, sensor-numbered).

**So positions cannot be assumed — they must be discovered, mapped, and driven by
a vehicle/axle configuration.**

---

## 1. Canonical position model

Replace the fixed `FL/FR/RL/RR` with a structured identifier:

```
Position = Axle(1..N, front→rear) × Side(L/R) × Wheel(single | Inner/Outer)

Canonical ID:  A{axle}{side}[{I|O}]
    A1L  = axle 1, left            (steer-left)
    A1R  = axle 1, right
    A2LO = axle 2, left-outer      (dual wheel)
    A2LI = axle 2, left-inner
    A3R  = axle 3, right
```

- **Backward compatible:** a 2-axle car maps `FL→A1L, FR→A1R, RL→A2L, RR→A2R`, and
  the GUI can still *display* the friendly labels (FL/FR/…) while the engine uses
  canonical IDs.
- A **channel** is then `Position × Component`, e.g. `A2R_Fz`, `A3LI_Fx`.
- Components stay the six WFT signals: `Fx, Fy, Fz, Mx, My, Mz`.

## 2. Vehicle-type / axle-layout configuration

A declarative config (one entry per vehicle class) defines the expected layout and
the **per-tyre-type parameters** (aims 2 & 7):

```yaml
vehicle_types:
  - name: "6x4 Heavy Truck"
    class: HCV
    axles:
      - {id: 1, role: steer, dual: false}        # A1L A1R
      - {id: 2, role: drive, dual: true }        # A2LO A2LI A2RO A2RI
      - {id: 3, role: drive, dual: true }        # A3LO A3LI A3RO A3RI
    tyre:
      type: "315/80 R22.5"
      Fz_range_daN: [0, 4000]                    # → histogram axis + heatmap limits
      Fx_range_daN: [-1500, 1500]
      Fy_range_daN: [-1500, 1500]
      lower_threshold_daN: 50                     # aim 5 (remove data below this)
```

- The config also carries **heatmap scaling limits** and **filter defaults** per
  tyre type — so aims 4, 5, 7 are all driven from one place.
- Ships with **standard CV presets** (LCV / MCV / HCV / bus / trailer, 4x2 / 6x2 /
  6x4 / 8x4) using typical load ranges, since the company isn't supplying types.

## 3. Channel discovery + mapping (auto)

On loading a raw folder or CSV, the engine **discovers** what's present instead of
assuming:

```
1. List all channels (raw filenames / CSV columns / FAMOS mapping file).
2. Regex-extract position tokens against a set of known patterns:
     _(fl|fr|rl|rr) | (FL|FR|RL|RR) | axle+side (A2R, 2R, R2) | sensor-id (_1,_2)
3. Normalise each to a canonical Position (A{n}{side}[{io}]).
4. Build the actual position list for THIS recording (2, 4, 6, 8, …).
5. Match to the closest vehicle-type preset (by axle count) OR let the user pick.
```

A small, editable **mapping table** handles vendor naming variants, so a new
convention is a config change, not a code change.

## 4. Dynamic pipeline

Every stage iterates over the **discovered** positions, not a fixed four:

- **Validation** — mandatory = whatever the chosen vehicle-type expects; reports
  present/missing per position.
- **Statistics / Histograms / Boxplots / Rainflow** — generated per discovered
  position; figure grids size themselves to N positions (2→8+).
- **Heatmaps** — one per position group; axis + colour limits from the tyre config
  (aim 4).
- **Sanitize / Filter / Resample** — per channel, parameters from config + GUI.

`config.py` changes from constants to a **`ChannelSet`** built at runtime from the
discovery + vehicle-type selection.

## 5. How the 8 aims all attach to this concept

| Aim | Where it lives in the concept |
|---|---|
| **B, 5, 6** sanitize/filter/resample in GUI | Per-channel stage params; lower-threshold + outlier removal from tyre config + live GUI controls |
| **2** vehicle types | `vehicle_types` config + selector; auto-matched by discovered axle count |
| **3** graph validation (FAMOS-like) | Per-channel time + FFT spectrum views over the canonical ChannelSet |
| **4** heatmap scaling limits | `tyre.*_range_daN` → fixed colour/axis limits (also fixes per-panel colour bug) |
| **7** tyre-type parameters | `tyre` block per vehicle type drives ranges, thresholds, limits |
| **8** time synchronization | Align channels by imcdbc start-epoch onto a shared master clock |
| raw ingestion | Discovery feeds the ChannelSet; FAMOS-grade imc reader already built |

## 6. End-to-end data flow

```
 raw .raw folder ─┐
                  ├─▶ Discovery ─▶ ChannelSet ─▶ Sanitize ─▶ Filter ─▶ Resample ─┐
 FAMOS CSV ───────┘        │            │                                         │
                           ▼            ▼                                         ▼
                    Vehicle-type   time-sync (imcdbc)                      Analysis (dynamic)
                     preset + tyre                                    stats·hist·heat·box·rainflow·FFT
                       params                                                     │
                           └───────────────── heatmap/axis limits ───────────────┤
                                                                                  ▼
                                                                         GUI screens + report
```

## 7. Module plan

```
dtt/
├── channels.py         # NEW: Position, Component, ChannelSet, discovery + mapping
├── vehicles.py         # NEW: VehicleType / Axle / TyreParams presets (standard CV)
├── ingestion/
│   ├── imc_reader.py   # DONE: FAMOS-grade raw reader
│   └── loader.py       # extended to emit a ChannelSet
├── (analysis/*)        # refactored to iterate ChannelSet.positions
gui/
├── pages/new_study     # + pick raw folder, pick/auto-detect vehicle type
├── pages/preprocess    # NEW: sanitize/filter/resample controls + live preview
├── pages/signals       # NEW: FAMOS graph validation + FFT spectrum + time-sync
├── pages/heatmaps      # + scaling-limit controls
└── models/vehicle_ctrl # vehicle-type + tyre-param state
```

## 8. Build sequence

1. **`channels.py` + `vehicles.py`** — the generalized model + discovery + standard
   presets (the foundation everything else needs).
2. Refactor the pipeline stages to iterate the `ChannelSet` (removes the FL/FR/RL/RR
   hardcoding) — keeps the 2-axle car working, unlocks N-axle.
3. **Preprocess screen** — sanitize (lower-threshold), filter, resample, live preview.
4. **Signals screen** — FAMOS time + FFT spectrum + channel time-sync.
5. **Heatmap scaling** from tyre config + colour-bug fix.
6. **Vehicle-type selector** wired through discovery.

This concept makes "how many axles" a *data + config* question, not a code
assumption — a 2-axle car and an 8x4 truck run the same pipeline.