# Task: rainflow histogram restyle, FAMOS-validation CSV export, workflow modes

Three items. Do them in this order, contained ones first: (1) restyle the rainflow-page
histograms to the new light look, (2) add a temporary raw-preprocessed CSV export for
FAMOS validation, (3) split the workflow into three modes (only preprocessing / only
analysis / both). Item 3 is the largest and touches how studies run, so it comes last
and gets its own validation.

Read `PROJECT_HISTORY.md` first. Same workflow as prior rounds: Phase 1 analyse and
report, Phase 2 ordered plan and stop for approval, Phase 3 implement incrementally
with validation per item and a running `MIGRATION_NOTES.md`. Surgical changes, branch,
commit per validated item.

---

## Item 1 — restyle the rainflow-page histograms to the new light style

The rainflow matrix panels include a range-distribution histogram row that is still on
the OLD dark banded style, while the Histograms page now uses the new light soft-fill
style (`plot_style.py::draw_histogram(..., style="soft")`, the light `_style_ax`
theming from the last round). Match them exactly to that new light style: light
background, thin single-tone curve, soft low-opacity fill, no bars, no banding, quiet
grid, small dark text.

This reverses the earlier decision to keep `rainflow.py` on `style="banded"` for
byte-identical output. That is intended now, the user is choosing the new look. So:
- Switch the rainflow range-distribution histograms to `style="soft"` and the light
  theming, the same path the Histograms page uses.
- These panels are small subplots inside a larger figure, so match the style exactly
  but keep them readable at panel size (line weight and fonts scaled to the panel, not
  necessarily identical point sizes to the full-page version).
- Do NOT change the rainflow matrix (the Siemens from-to heatmap) itself, only the
  histogram row beneath it. Do not change any rainflow math, cycle counts, Miner
  damage, or CSV export.
- Note in `MIGRATION_NOTES.md` that this supersedes the earlier byte-identical
  constraint on rainflow output.

Validate: regenerate a rainflow figure before/after, confirm the histogram row matches
the Histograms page look and the from-to matrix and all numbers are unchanged.

---

## Item 2 — temporary raw-preprocessed CSV export for FAMOS validation

Add an export that writes the signal right AFTER the FAMOS recipe (despike, smo,
FiltLP, decimate) and BEFORE any analysis stage, so it is exactly the conditioned data
FAMOS would produce, for manual cross-check in FAMOS.

- One wide CSV: a time column plus one column per channel (all channels), at the
  post-recipe rate (100 Hz after decimation). Not per-channel files.
- Written at the point in `pipeline.py` where the FAMOS recipe has finished and the
  analysis stages have not yet run, so it captures the conditioned signal, not
  analysis output.
- Use the platform's real channel names and units (no N-to-daN re-conversion, no
  FL/FR/RL/RR hardcoding, channel order from the RunChannels model).
- This is explicitly a temporary validation aid. Mark it clearly as such (a distinct
  filename like `preprocessed_for_famos_validation.csv`, a code comment saying it is a
  temporary FAMOS-validation export, and a note in `MIGRATION_NOTES.md`). Gate it
  behind a flag/mode so it is not written on every run unless asked.

Validate: run it on a real study, confirm the CSV has the time column plus all
channels, row count matches the decimated length, and values match the platform's
processed output for a couple of channels.

---

## Item 3 — split the workflow into three modes

Add a workflow mode chosen when a study is run: **only preprocessing**, **only
analysis**, or **both** (both = today's behaviour). Chosen up front at run time, not a
sidebar/menu reshuffle, this is workflow logic, not navigation.

- **only preprocessing**: run ingestion + the FAMOS recipe (+ despike/stop-removal as
  configured) and write the processed study output, then stop. No analysis stages
  (histograms, heatmaps, boxplots, rainflow, severity, PSD, reports).
- **both**: today's full pipeline, unchanged. This must remain the default so existing
  runs behave exactly as now.
- **only analysis**: do NOT re-run preprocessing. Read an already-processed study's
  existing wide processed CSV (`processed_data.csv`) as input and run only the analysis
  stages on it.

**Provenance guard on "only analysis" (required).** Because it trusts a
previously-processed file, it must check the study's saved processing provenance
(pipeline/recipe version and units, whatever is already recorded with a study) before
analyzing. If the provenance is missing, or from an older pipeline version than the
current code, or the units look inconsistent, WARN clearly and do not silently
analyze stale data. Surface the warning to the user; let them proceed knowingly, but
never hide it. This is the same class of guard the two-run comparison feature uses,
reuse that provenance-reading approach if it exists.

Where the mode lives: a `RunConfig` field / CLI-and-run-config choice (e.g.
`workflow_mode: "preprocess" | "analysis" | "both" = "both"`). Do not build a new GUI
picker this round (GUI restructure is still deferred); wiring it into the run
config/CLI is enough, and if there is an existing run-setup surface that trivially
takes one more field, add it there, otherwise leave the GUI for the restructure round
and say so.

Validate: run all three modes on a real study. Confirm "only preprocessing" produces
the processed output and no analysis artifacts; "both" is byte-identical to today;
"only analysis" runs analysis off the existing processed CSV without re-preprocessing,
and actually emits the provenance warning when pointed at a deliberately stale/old
processed file.

---

## Traps / do-nots (unchanged from prior rounds)

- No N-to-daN re-conversion, no FL/FR/RL/RR hardcoding, sample rate from config.
- Do not change the FAMOS recipe math, the raw `.raw` reader / 58-sample offset, or
  ingestion internals.
- Do not do the full GUI restructure (popups to embedded views) this round.
- "both" mode must stay the default and behave exactly as the pipeline does today.

## Deliverables

Restyled rainflow histograms; the temporary FAMOS-validation CSV export; the three
workflow modes with the provenance guard; `MIGRATION_NOTES.md` updated per item;
validation evidence per item.

Start with Phase 1: confirm the exact functions and the study-provenance mechanism the
"only analysis" guard will read, then give the ordered plan and stop.
