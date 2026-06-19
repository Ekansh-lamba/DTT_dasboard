# DTT WFT Automation Platform — Desktop GUI

A PySide6 (Qt) desktop front-end for the `dtt` analysis pipeline. It launches
the backend, watches its progress live, and presents every artifact the
pipeline writes to `dtt/outputs/{study}/`.

This GUI **does not** implement any analytical logic. It runs the backend's
documented CLI (`python -m dtt.pipeline …`) in a worker thread and reads the
JSON / CSV / PNG / PPTX outputs back through a single model layer.

## Run

```bash
# from the project root  (…/apollo tyres)
pip install -r gui/requirements.txt      # PySide6
pip install -r dtt/requirements.txt      # backend (pandas, scipy, …)
python -m gui.main
```

## Architecture (MVC)

```
gui/
├── main.py                 # entry point — QApplication + theme
├── main_window.py          # CONTROLLER: QMainWindow shell, sidebar nav,
│                           #   QStackedWidget routing, active-study state,
│                           #   pipeline lifecycle wiring
├── theme.py                # dark engineering theme (palette + global QSS)
├── models/
│   └── repository.py       # MODEL: StudyRepository + Study + ChannelStat.
│                           #   The ONLY place that knows the on-disk output
│                           #   contract. All file reads go through here.
├── workers/
│   └── pipeline_worker.py  # BACKEND INTEGRATION: QThread worker runs the CLI,
│                           #   parses [n/9] log markers into stage signals.
├── widgets/
│   ├── common.py           # Card, KpiCard, Badge, ZoomableImageView,
│   │                       #   FigureCard, FigureGrid, ScrollPage
│   └── image_viewer.py     # zoomable fullscreen figure dialog
└── pages/                  # VIEWS — one file per screen
    ├── base_page.py        #   BasePage.load_study(study) contract
    ├── dashboard_page.py   #   1. Dashboard
    ├── new_study_page.py   #   2. New Study
    ├── processing_page.py  #   3. Processing (live stages + log)
    ├── validation_page.py  #   4. Validation Results
    ├── statistics_page.py  #   5. Statistics (searchable table)
    ├── histograms_page.py  #   6. Histograms (wheel/signal selectors)
    ├── heatmaps_page.py    #   7. Heatmaps (+ hexbin, fullscreen)
    ├── boxplots_page.py    #   8. Boxplots
    ├── rainflow_page.py    #   9. Rainflow (+ cycle CSV export)
    ├── reports_page.py     #  10. Reports (open / export / regenerate)
    └── history_page.py     #  11. Study History
```

### How data flows

- **Reading outputs** — every screen receives the active `Study` via
  `load_study()`. `Study` exposes typed accessors (`stats()`, `validation()`,
  `histogram(wheel, signal, kind)`, `rainflow_cycle_csvs()`, …) so a change to
  the backend's file layout is a one-file edit in `repository.py`.
- **Monitoring progress** — `PipelineWorker` runs the subprocess and matches
  each `[n/9]  Stage` log line to a GUI stage, emitting `stage_started`,
  `stage_finished(name, seconds)` and `progress(done, total)`. The Processing
  screen renders these as a live checklist with per-stage timings.
- **Threading** — the subprocess is owned by a `QThread`; all signals cross to
  the GUI thread via Qt's queued connections, so the UI never blocks.
- **Displaying PNGs** — `ZoomableImageView` (a `QGraphicsView`) gives
  scroll-to-zoom + drag-pan; `FigureCard`/`FigureGrid` render auto-scaling
  thumbnails that open the fullscreen viewer.
- **Loading JSON into tables** — `repository._read_json` parses
  `validation_report.json` / `stats_summary.json`; the Statistics table sorts
  numerically and is filterable by text + wheel.

## Packaging

Build a standalone executable with PyInstaller (run from project root):

```bash
pip install pyinstaller
pyinstaller --noconfirm --windowed --name "DTT-Platform" \
    --collect-all PySide6 \
    --add-data "dtt;dtt" \
    gui/main.py
```

Notes:
- `--windowed` hides the console; `--collect-all PySide6` bundles Qt plugins.
- The bundled app still shells out to `python -m dtt.pipeline`. For a fully
  self-contained build, either ship a Python runtime or refactor the worker to
  import `dtt.pipeline.run` in-process (kept as a subprocess here to isolate the
  heavy backend from the UI process and stream logs cleanly).
- On Windows use `;` as the `--add-data` separator (shown above); on
  macOS/Linux use `:`.
```
