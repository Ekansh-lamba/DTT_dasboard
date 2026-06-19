# Digital Tyre Testing (DTT) Automation Platform

## Overview

The Digital Tyre Testing (DTT) Automation Platform is a centralized, automated pipeline designed to process Wheel Force Transducer (WFT) data. The platform modernizes existing engineering analysis methodologies by transitioning fragmented manual processes into a robust, single-command pipeline. 

Module 1 of the platform automates the ingestion, validation, signal processing, statistical analysis, and reporting of high-frequency WFT datasets. It leverages validated legacy engineering logic while introducing significant performance optimizations and eliminating manual intervention.

## Architectural Objectives

1. **Automation**: Replace the manual CSV-FAMOS-Python-PowerPoint workflow with a unified, automated process.
2. **Reproducibility**: Ensure all engineering calculations are deterministic and traceably documented.
3. **Performance**: Process high-density datasets (exceeding 1 million rows) efficiently using CPU-bound operations.
4. **Modularity**: Structure the codebase into discrete, maintainable components allowing for future expansion into GPS, IMU, and durability analytics.
5. **Preservation of Validated Logic**: Retain existing Siemens-style Rainflow counting and zero-phase Butterworth filtering methodologies already validated by the engineering team.

## Pipeline Architecture

The platform executes a sequential 9-stage pipeline orchestrating data from raw ingestion to final presentation:

### 1. Data Ingestion
Robust CSV loading architecture utilizing lazy scanning techniques to handle large datasets without memory saturation. The ingestion module automatically detects varying header structures, identifies the target sampling rate from time arrays, and intelligently normalizes forces from Newtons (N) to decaNewtons (daN) when required.

### 2. Channel Validation
Analyzes the dataset schema to ensure the presence of all 12 mandatory wheel force channels (FL/FR/RL/RR forces across X, Y, and Z axes). Emits a structured JSON validation report detailing missing channels, data quality metrics, and duplicate column flags.

### 3. Data Sanitization
Ensures data integrity prior to processing. Responsibilities include forward-filling minor missing data gaps, dropping completely empty records, removing duplicate timestamps, and applying percentile-based flags for outlier tracking without destructive data removal.

### 4. Signal Processing
Applies a zero-phase Butterworth low-pass filter (default: 4th order, 10 Hz cutoff) across all mandatory force channels using `scipy.signal.filtfilt`. This ensures high-frequency noise removal without introducing phase distortion, directly mirroring the legacy FAMOS processing standards.

### 5. Statistical Analysis
Computes fundamental descriptive statistics (Mean, Median, Standard Deviation, Minimum, Maximum) alongside severity percentiles (P80, P90, P95) for load quantification.

### 6. Histogram Generation
Produces comprehensive force distribution visualizations. Generates both distance-weighted histograms (scaled by vehicle speed) and percentage-based distribution plots utilizing standardized engineering axis ranges.

### 7. Heatmap Analysis
Generates two-dimensional percentage occurrence heatmaps and logarithmic hexbin density plots mapping the relationships between force vectors (Fx vs Fy, Fz vs Fy, and Fz vs Fx) for each wheel.

### 8. Boxplot Generation
Visualizes load distributions across channels with fixed-axis constraints to facilitate cross-vehicle comparisons. Accurately maps the 5th percentile, median, mean, 95th percentile, and interquartile ranges.

### 9. Rainflow & Damage Analysis
Executes fatigue and durability analysis utilizing the standard Rainflow counting algorithm. Outputs include Siemens-style 16x16 from-to matrices, cycle distribution datasets, and cumulative Miner's rule damage calculations.

### Reporting
Compiles all generated statistics and visualizations into a comprehensive 16-slide PowerPoint presentation native to the Microsoft Office ecosystem, structured for immediate engineering review.

## System Requirements

- Python 3.10+
- Multi-core CPU recommended for parallel figure generation
- GPU is explicitly not required (pipeline is 100% CPU-bound)

## Installation

Install the required dependencies via the provided requirements file:

```bash
pip install -r requirements.txt
```

## Execution

The platform is designed to be executed via a command-line interface. A single execution command triggers the entire 9-stage pipeline.

```bash
python -m dtt.pipeline --csv "path/to/dataset.csv" --vehicle "Vehicle_Name" --study "Optional_Study_Name"
```

### Command Line Arguments

- `--csv`: (Required) Absolute or relative path to the source WFT CSV file.
- `--vehicle`: Vehicle identifier utilized for report titling.
- `--study`: Unique identifier for the output directory. Defaults to the current timestamp.
- `--sr`: Override the automatically detected sampling rate (Hz).
- `--cutoff`: Override the default low-pass filter cutoff frequency.
- `--order`: Override the default Butterworth filter order.
- `--no-filter`: Bypass the signal processing stage entirely.
- `--miner`: Override the default Miner's rule exponent (default: 8.0).

## Output Structure

Upon successful execution, all artifacts are securely stored within an isolated study directory:

```
dtt/outputs/{Study_Name}/
├── validation_report.json     # Channel validation metadata
├── stats_summary.json         # Computed statistical metrics
├── processed_data.csv         # Sanitized and filtered dataset
├── logs/
│   └── pipeline.log           # Verbose execution logs
├── figures/
│   ├── hist_*.png             # Histogram exports
│   ├── heatmap_*.png          # Heatmap & hexbin exports
│   ├── boxplot_*.png          # Boxplot exports
│   ├── rainflow_*.png         # Rainflow matrix exports
│   └── rainflow_cycles_*.csv  # Raw Rainflow cycle data
└── WFT_Report_{Vehicle}_{Date}.pptx  # Final presentation report
```
