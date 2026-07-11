# Digital Tyre Testing (DTT) — WFT Automation Platform

A Windows desktop application that automates **Wheel Force Transducer (WFT)**
analysis for vehicle durability engineering. It ingests raw road-load recordings
(imc `.raw` or FAMOS-exported CSV), validates and cleans them, runs the full
signal-processing / statistics / fatigue pipeline, and presents the results —
interactive plots, tables, and an auto-generated PowerPoint — in a dark-themed
PySide6 (Qt) GUI.

---

## Download & run (for reviewers)

A prebuilt Windows build is published on the **Releases** page:

1. Go to **Releases** → download **`DTT-Platform-windows.zip`**.
2. **Extract** the zip anywhere.
3. Run **`DTT-Platform.exe`** inside the extracted folder. No Python install required.

Studies you create are saved under **`Documents\..\DTT-Platform\outputs`** (a
`DTT-Platform` folder in your home directory), so results persist between runs.

(The build is produced automatically by GitHub Actions — see *Building the exe*.)

> **Note:** measurement data is **not** shipped in the repo (confidential). Use
> the built-in **New Study** screen to load your own imc `.raw` folder/files or a
> CSV, or generate synthetic test data (below).

---

## Run from source

```bash
pip install -r dtt/requirements.txt      # backend: pandas, numpy, scipy, matplotlib, rainflow, python-pptx
pip install -r gui/requirements.txt      # GUI: PySide6
python -m gui.main
```

Generate synthetic test data (no confidential data needed):

```bash
python generate_sample_data.py --profile mixed --rows 200000 --name WFT_Test
```

---

## What it does

**Data sources** — auto-detected:
- imc STUDIO `.raw` folders or **individual files** (FAMOS-grade calibration)
- Classic imc **FAMOS** `.raw` (`|CF`) exports
- FAMOS-exported **CSV**

**Any axle configuration** — positions are discovered from the data, so a 2-axle
car, a 6x4 tandem truck, a bus, or a trailer all run the same pipeline. Vehicle
type + tyre parameters (load ranges, thresholds) are auto-matched or user-selected.

**Screens:**
- **Dashboard / New Study / Processing** — configure and launch a run, watch the
  9 stages complete live.
- **Preprocess** — interactive sanitisation (lower threshold, outlier removal),
  Butterworth filtering, and resampling with a live before/after preview.
- **Signals** — FAMOS-style time-domain overlay, **FFT amplitude spectrum**
  (dB vs log-Hz), and cross-correlation **time-synchronisation**.
- **Validation / Statistics / Histograms / Heatmaps / Boxplots / Rainflow** —
  the analysis outputs (searchable tables, zoomable figures).
- **Reports / History** — open/export/regenerate PowerPoint reports; manage
  (and delete) past studies.

For the engineering concepts and the maths, see
[docs/PROJECT_CONCEPTS_GUIDE.md](docs/PROJECT_CONCEPTS_GUIDE.md); for the
multi-axle design, [docs/GENERALIZED_AXLE_CONCEPT.md](docs/GENERALIZED_AXLE_CONCEPT.md).

---

## Command-line (no GUI)

```bash
python -m dtt.pipeline --csv "data.csv" --vehicle "Truck" --study "Run_01"
python -m dtt.pipeline --raw "path/to/imc_raw_folder" --vehicle-type "Heavy Truck (6x4 tandem)"
python -m dtt.pipeline --raw-files ch1.raw ch2.raw ch3.raw
```

Results are written to `dtt/outputs/{study}/` (git-ignored).

---

## Building the .exe

Automatic (GitHub Actions, [`.github/workflows/build-exe.yml`](.github/workflows/build-exe.yml)):
- **Push a version tag** to publish a Release with the exe:
  ```bash
  git tag v1.0
  git push origin v1.0
  ```
- Or trigger it manually from the **Actions** tab (**Run workflow**) — the exe is
  attached as a downloadable artifact.

Locally:
```bash
pip install -r dtt/requirements.txt -r gui/requirements.txt -r build-requirements.txt
pyinstaller --noconfirm DTT-Platform.spec
```
The result is `dist/DTT-Platform.exe`. The build config is in
[DTT-Platform.spec](DTT-Platform.spec) (only Qt Core/Gui/Widgets/Svg are bundled;
heavy unused Qt modules are excluded to keep the exe small).

---

## Project structure

```
dtt/                 # analytical backend (the pipeline)
  channels.py        #   generalized axle/position model + discovery
  vehicles.py        #   vehicle-type + tyre-parameter presets
  run_channels.py    #   runtime channel config (axle-dynamic)
  ingestion/         #   CSV + imc .raw readers (imc3 & classic FAMOS)
  validation/ sanitization/ processing/   analysis/ reporting/
  preprocessing.py spectral.py            #   interactive-analysis helpers
  pipeline.py        #   9-stage orchestration (CLI entry)
gui/                 # PySide6 desktop app (MVC)
  main.py main_window.py theme.py
  models/ workers/ widgets/ pages/
generate_sample_data.py   # synthetic WFT data generator
```

---

## Confidentiality

Measurement data, reports, and proprietary engineering coding are **git-ignored**
and never committed: `csv/`, `*.raw`, `*.dat`, `*.pptx`, `dtt/outputs/`, the imc
channel-mapping file, and the raw-data folders. See [.gitignore](.gitignore).