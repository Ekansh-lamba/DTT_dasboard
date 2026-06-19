# Digital Tyre Testing (DTT) — WFT Automation Platform

A desktop application that automates **Wheel Force Transducer (WFT)** analysis for
vehicle durability engineering. It ingests raw road-load CSV recordings, validates
and cleans them, runs the full signal-processing / statistics / fatigue pipeline,
and presents the results — figures, tables, and an auto-generated PowerPoint — in a
dark-themed **PySide6 (Qt)** GUI.

> **In one line:** drop in a multi-hour recording of the forces at all four wheels →
> get cleaned data, severity statistics, distribution plots, and a fatigue/durability
> report, fully automatically.

---

## What it does

The backend runs a deterministic **9-stage pipeline** (no machine learning — same
input always gives the same output):

1. **Data Ingestion** — robust CSV parsing, channel detection, sampling-rate
   detection, automatic N→daN unit conversion.
2. **Validation** — checks all 12 mandatory force channels, NaNs, duplicates →
   `validation_report.json`.
3. **Sanitization** — removes duplicate timestamps, fills small gaps, flags (keeps)
   outliers → `processed_data.csv`.
4. **Signal Processing** — zero-phase Butterworth low-pass filter (default 4th
   order, 10 Hz) to remove noise without time-shifting peaks.
5. **Statistics** — mean / median / std / min / max + severity percentiles
   (P80/P90/P95) → `stats_summary.json`.
6. **Histograms** — distance-weighted and percentage force distributions.
7. **Heatmaps** — Fx×Fy / Fz×Fy / Fz×Fx occurrence maps + per-wheel hexbin density.
8. **Boxplots** — cross-wheel load-distribution comparison.
9. **Rainflow** — fatigue cycle counting, Siemens-style From-To matrices, Miner's
   damage.

Plus an automatic **PowerPoint report** assembling all of the above.

The **GUI** wraps this with 11 screens: Dashboard, New Study, Processing (live
progress), Validation, Statistics, Histograms, Heatmaps, Boxplots, Rainflow,
Reports, and Study History.

For a full plain-language explanation of the engineering concepts (what Fx/Fy/Fz
mean, what each chart shows, the maths used), see
[docs/PROJECT_CONCEPTS_GUIDE.md](docs/PROJECT_CONCEPTS_GUIDE.md).

---

## Project structure

```
apollo tyres/
├── dtt/                     # Analytical backend (the pipeline)
│   ├── ingestion/           #   CSV loading + channel detection
│   ├── validation/          #   data-quality checks
│   ├── sanitization/        #   cleaning (non-destructive)
│   ├── processing/          #   Butterworth filtering
│   ├── analysis/            #   statistics, histograms, heatmaps, boxplots, rainflow
│   ├── reporting/           #   PowerPoint builder
│   ├── config.py            #   channels, colours, bins, thresholds
│   ├── pipeline.py          #   orchestrates all 9 stages (CLI entry)
│   └── outputs/             #   per-study results (git-ignored)
├── gui/                     # PySide6 desktop front-end (MVC)
│   ├── main.py              #   entry point
│   ├── main_window.py       #   sidebar + QStackedWidget + controller
│   ├── theme.py             #   dark engineering theme
│   ├── models/              #   StudyRepository — reads pipeline outputs
│   ├── workers/             #   QThread runner that drives the pipeline
│   ├── widgets/             #   reusable Qt widgets (zoomable images, cards)
│   └── pages/               #   the 11 screens
├── docs/                    # Concept & feature guide
├── generate_sample_data.py  # Synthetic WFT CSV generator (for testing)
├── csv/                     # Input datasets (git-ignored)
└── README.md
```

---

## Requirements

- **Python 3.10+** (developed on 3.13)
- Multi-core CPU recommended (figure generation is CPU-bound); no GPU needed.

---

## Installation

```bash
# from the project root
pip install -r dtt/requirements.txt      # backend: pandas, numpy, scipy, matplotlib, rainflow, python-pptx
pip install -r gui/requirements.txt      # GUI: PySide6
```

---

## How to start the project

### Run the desktop app (recommended)

```bash
python -m gui.main
```

Then:
1. The window opens on the **Dashboard**.
2. Go to **New Study**, pick a CSV, set the vehicle name and (optionally) the filter
   cutoff / order / Miner's exponent, and click **Start Analysis**.
3. Watch the **Processing** screen show each stage complete live.
4. Browse the results in **Validation / Statistics / Histograms / Heatmaps /
   Boxplots / Rainflow / Reports**.
5. Re-open any past run from **History** or the study selector (top-right).

### Run the pipeline directly (CLI, no GUI)

```bash
python -m dtt.pipeline --csv "csv/your_data.csv" --vehicle "Vehicle_Name" --study "Run_01"
```

| Argument | Description |
|----------|-------------|
| `--csv` | (required) path to the WFT CSV |
| `--vehicle` | vehicle name (used in report title) |
| `--study` | output folder name (default: timestamp) |
| `--cutoff` | filter cutoff frequency (Hz) |
| `--order` | Butterworth filter order |
| `--no-filter` | skip filtering |
| `--miner` | Miner's-rule exponent (default 8.0) |

Results are written to `dtt/outputs/{study}/`.

---

## Generating test data

No real dataset? Generate realistic synthetic WFT recordings (38 columns, forces in
Newtons, 100 Hz, with cornering/braking/road-roughness events):

```bash
python generate_sample_data.py --rows 200000 --profile mixed --name WFT_Test
```

Profiles: `mixed` (varied driving), `city` (stop/go), `rough` (highway over rough
surface). Files are written to `csv/`.

---

## Output structure

Each study produces:

```
dtt/outputs/{study}/
├── validation_report.json     # channel validation + data-quality metrics
├── stats_summary.json         # per-channel statistics
├── processed_data.csv         # cleaned + filtered dataset
├── figures/
│   ├── hist_*.png             # histograms
│   ├── heatmap_*.png          # heatmaps + hexbin
│   ├── boxplot_*.png          # boxplots
│   ├── rainflow_*.png         # rainflow matrices
│   └── rainflow_cycles_*.csv  # raw cycle data
├── logs/pipeline.log
└── WFT_Report_{vehicle}_{date}.pptx
```

---

## Notes

- `csv/`, `dtt/outputs/`, and `instructions/` are git-ignored (large datasets and
  generated/internal artifacts). Clone, install, then generate or supply your own
  CSV to get started.
- The analytical methods (Butterworth filtering, rainflow counting) intentionally
  mirror the team's validated legacy engineering workflow.
