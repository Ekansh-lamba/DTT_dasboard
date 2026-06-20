# DTT — WFT Analysis Dashboard (Module 1)

> **Apollo Tyres | Tyre Duty Cycle Dashboard**  
> Replaces manual 4–5 day FAMOS + Python + PowerPoint workflow with an
> automated under-30-minute interactive dashboard.

---

## Quick start

```bash
# 1. Create & activate environment (Python 3.11+ recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).  
Enter the CSV path in the sidebar and click **Load**.

---

## Project structure

```
dtt_wft/
├── app.py                  # Streamlit entry point (Layer B)
├── engine/                 # Pure-Python analysis engine (Layer A)
│   ├── io_loader.py        # Chunked CSV load, time rebuild, validation
│   ├── channel_map.py      # Name-based channel resolver (regex + exclude-list)
│   ├── histograms.py       # Force distribution histograms
│   ├── heatmaps.py         # 2D correlation density heatmaps
│   ├── trip_stats.py       # Trip summary statistics
│   ├── gg_severity.py      # g-g diagram + Fz severity banding
│   ├── fatigue.py          # Rainflow + Miner's rule damage
│   └── box_distance.py     # Box plots + distance-weighted histograms
├── ui/
│   ├── components.py       # Sidebar widgets, status card, filters
│   └── plots.py            # Plotly figure builders
├── export/
│   ├── csv_export.py       # Flat CSV export + ZIP bundle
│   ├── html_export.py      # Standalone interactive HTML
│   └── pdf_report.py       # PDF via reportlab + kaleido
├── config/
│   └── defaults.yaml       # All tunable parameters (bins, ranges, units …)
├── tests/
│   └── test_engine.py      # pytest unit tests (synthetic data, no real file needed)
└── requirements.txt
```

---

## Architecture

| Layer | Location | Depends on |
|---|---|---|
| **A — Engine** | `engine/` | numpy, pandas, rainflow only |
| **B — UI** | `ui/`, `app.py` | Streamlit, Plotly, Layer A |
| **C — Export** | `export/` | reportlab, kaleido, Layer A results |

The engine has **zero UI imports** — it can be reused by future modules
(IMU, GPS) without modification.

---

## Key design decisions

### Memory safety
- CSV loaded in 200 k-row chunks (configurable in `config/defaults.yaml`)
- Force/moment/IMU columns downcast to `float32` after load (halves RAM)
- `Time` and GPS columns kept as `float64` for precision
- `@st.cache_data` prevents re-loading on every re-render

### Time axis
- The stored `Time` column loses floating-point precision past ~10,000 s
- Rebuilt as `t_rebuilt = row_index × dt` where `dt = 1/fs`
- `fs` is detected dynamically from the first 5000 rows (median of `Time` diffs)
- Cross-check against stored `Time` emits a **warning** (not an assertion)
  if divergence exceeds 15% — allows graceful handling of quirky files

### Channel resolution
- Channels are **never** identified by column position (the legacy anti-pattern)
- `find_channel_strict()` uses regex `^{wheel}[_ -]*{signal}$` with an
  exclude-list (`lat`, `lon`, `latacc`, `acc`, `speed`, `dist`, `time`, `ws1`) 
  to prevent false matches
- All exclude keywords and patterns are config-driven

### Unit conversion
- Input unit is **declared in config** (`input_units.force: N`)
- Conversion is `value × unit_scale` where `unit_scale` is also from config
  (`display.unit_scales.daN: 0.1`)
- **No heuristics** ("if median > 1000, divide by 10") — ever

### g-g diagram
- Axes are labelled in **g** (not m/s²)
- Limits: ±1.2 g (configurable)
- Reference circles at 0.2, 0.4, 0.6, 0.8, 1.0 g
- `Latacc` / `Longacc` channels are declared as `g` in config — no conversion applied

### Histogram range clipping
- A warning is emitted if > 1% of samples fall outside the configured histogram range
- The threshold is `histogram.clip_warn_fraction` in `defaults.yaml`
- Default Fy range is ±300 daN — increase if cornering data shows clipping warnings

---

## Running tests

```bash
pytest tests/test_engine.py -v
```

All tests use a synthetic 2000-row DataFrame — no real file required.
Key tests:
- `test_histogram_unit_conversion_is_deterministic` — verifies N→daN is exactly ÷10
- `test_gg_values_are_in_g` — verifies no accidental ×9.81 conversion
- `test_fatigue_damage_formula` — verifies Σ(n × r^m)
- `test_channel_map_does_not_confuse_latitude` — verifies exclude-list

---

## Configuration

All tunable parameters are in `config/defaults.yaml`.  
Key sections:

| Section | What it controls |
|---|---|
| `input_units` | Declared units for all channel types |
| `display.unit_scales` | N→daN scale factors |
| `channel_exclude_keywords` | Columns excluded from force-channel matching |
| `channel_patterns` | Regex patterns for scalar channels (speed, latacc, etc.) |
| `histogram.force_ranges_dan` | X-axis limits for histogram bins |
| `histogram.clip_warn_fraction` | Warn threshold for samples outside range |
| `gg.axis_limit_g` | g-g plot axis limits |
| `fatigue.miner_slope_m` | Wöhler exponent m (default 5) |
| `loading.chunksize` | CSV rows per chunk (default 200 000) |

---

## Real-file smoke test

```python
# Run from the dtt_wft\ directory
from engine.io_loader import load_csv_chunked
from engine.channel_map import build_channel_map, log_channel_map
import yaml

with open("config/defaults.yaml") as f:
    cfg = yaml.safe_load(f)

CSV = r"C:\Users\ekans\Downloads\DTT_WFT\For SRM\RLDA WFT PV data sample.csv"
df, meta = load_csv_chunked(CSV, cfg)
print(meta)

chan_map = build_channel_map(df, cfg)
log_channel_map(chan_map)
```

Expected output:
```
n_rows: ~2,400,000  fs: ~100.0 Hz  duration_s: ~24,000  dist_km: ~240
```

---

## Export

| Format | How | Notes |
|---|---|---|
| **CSV** | "Export CSV Results" button → ZIP archive | One CSV per result type |
| **HTML** | "Export Standalone HTML" button | Self-contained, works offline |
| **PDF** | "Export PDF Report" button | Requires kaleido (`pip install kaleido`) |

---

## Extending to IMU / GPS modules

The engine layer (`engine/`) has no UI or Streamlit imports.  
To add a new module:
1. Create `engine/gps_analysis.py` (new functions)
2. Add a new tab in `app.py`
3. Add new Plotly builders in `ui/plots.py`
4. The existing `io_loader`, `channel_map`, and `config` are reused as-is
