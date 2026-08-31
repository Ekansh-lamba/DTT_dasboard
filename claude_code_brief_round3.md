# Task: add the missing analysis features and fix histogram readability

Two workstreams in this round: (A) port the analysis features the platform is
missing versus the standalone WFT Analyzer, and (B) fix the force-distribution
histograms so they are readable and consistent with the rest of the app. GUI
restructure (popups to embedded views) is explicitly NOT in this round, leave it
for later.

Read `PROJECT_HISTORY.md` first (architecture, units, channel model, existing
analysis stages, the `plot_style.py` banded/KDE styling from Round 2). The
feature-gap comparison against the standalone analyzer is already done; its
conclusions are the scope below, do not re-derive them, but do confirm the exact
functions before editing.

Ground truth for units and channel model: everything here runs on already-processed
platform output, uses the platform's channel abstraction (not FL/FR/RL/RR hardcoding),
and must NOT re-apply any N-to-daN conversion (the platform owns units).

---

## How to work

Same workflow as prior rounds. Phase 1: analyse and report, no edits. Phase 2:
ordered plan, stop for my approval. Phase 3: implement incrementally, validate each
step, keep the running `MIGRATION_NOTES.md`. Surgical changes, flag signature
changes, branch, commit per validated step.

Suggested order, contained/low-risk first: (1) histogram readability fixes, (2)
distance-weighted heatmap wiring, (3) Welch PSD, (4) two-run comparison tables +
styling, (5) RF Compare. Refine in your plan.

---

## Workstream A — the four missing analysis features

These four are the confirmed gaps from the comparison. Everything else in the
standalone analyzer (percentiles, box plots, rainflow from-to matrix, AUC/KDE,
G-severity/DLC formulas, PNG export) already exists in the platform and is equal or
better, so do NOT port those.

**A1 — Distance-weighted heatmaps (wiring, not new logic).** The platform's
`heatmaps.py` is currently count-weighted. The robust distance-weighting already
exists in `histograms.py::_get_speed_weights` (it handles dead/stuck speed channels
and unit auto-detection, and is more careful than the analyzer's version). Wire
`heatmaps.py` to use that existing function so the force-pair heatmaps weight by
distance travelled instead of sample count. Reuse the existing helper, do not copy
the analyzer's `_dist_km_per_sample`. Where the speed channel is unusable, fall back
to count weighting AND surface that clearly (see B3, this is the same speed-channel
fallback issue).

**A2 — Welch PSD.** Genuinely missing and distinct from the platform's existing
single-FFT dB spectrum. Add a Welch power-spectral-density analysis per wheel
(`scipy.signal.welch`). Sample rate must come from platform config/ingestion, NOT a
typed value. Put it in the spectral module alongside the existing FFT spectrum, as a
separate output, not a replacement. Lowest priority of the four.

**A3 — Two-run comparison tables.** This is the real architectural gap: the platform
has no concept of comparing two completed runs. Add the ability to load two finished
studies and produce delta tables between them: RMS delta, DLC delta, and G-severity
delta, per channel/wheel, showing DS1, DS2, absolute delta, and percent change.
Reuse the platform's existing RMS / DLC / G-severity functions (do not reimplement
the formulas, they are confirmed identical to the analyzer's). Check whether the
Compare page already has any two-run plumbing to build on before adding new.

**A4 — RF Compare.** A rainflow comparison view overlaying two datasets, front and
rear axle, showing the Miner damage ratio between them. This reuses the platform's
existing rainflow primitives (from-to matrix, Miner damage), it is new orchestration
on top of them, not new rainflow math. Belongs with A3 as part of the two-run
comparison capability.

### Two-run comparison plot styling (applies to A3/A4)

Style the two-run comparison plots to match the reference EV-vs-IC slides:
- Two runs overlaid on one axis as filled KDE curves, one colour per run, with a
  legend naming each run.
- Dashed vertical lines marking P5 and P95 of each run.
- A compact statistics strip under each plot: Min, P5, Q1, Median, Mean, Q3, P95,
  Max, one row per run. This strip is the "more informative" part of the reference,
  make it a standard element under the comparison plots.
- Support the cross-position overlay option (e.g. RL of run A vs FL of run B on the
  same axis), as in the reference slides.
- Do all of this through the platform's `plot_style.py`, not the analyzer's colour
  constants. Aim for clean and consistent; exact pixel-matching of the slides is a
  nice-to-have, not the bar.

---

## Workstream B — force-distribution histogram readability

The current force-distribution histograms (the `FR - Force Distribution` style, Fx/
Fy/Fz in a cramped row on a dark navy panel) are hard to read. Three fixes:

**B1 — Apply the existing banded/KDE style.** These plots are still on the old dark
style while the rest of the app moved to the Round 2 banded-with-KDE look. Apply the
existing `plot_style.py` banded styling (light background, dark-centre-to-light
bands, red KDE overlay) to these force-distribution histograms so they match the
rest of the app. This is a consistency fix, the styling helper already exists.

**B2 — X-axis range option.** The current plots span the full sensor range (Fx
-300..300, etc.) while the real data sits in a narrow band near zero, crushing all
the shape into the middle. Provide the x-axis range as a user option with at least
two modes: full sensor range (current behaviour), and autoscale-to-data (clip to
where the data actually sits, e.g. a percentile-based range like P1..P99 with a
margin). Make autoscale available but do not remove the full-range option, the user
picks. Default is your call, state it in the plan.

**B3 — Fix the distance-weighting speed-channel fallback.** The reference plot shows
"Sample-count weighting, no usable speed channel", meaning it wanted distance
weighting but the speed channel was unusable and it silently fell back. This is the
same speed-channel handling as A1. Use the robust `histograms.py::_get_speed_weights`
detection so a usable speed channel is actually found where one exists, and when it
genuinely isn't, keep the clear label but make sure the fallback is only hit when
truly necessary. Fix this together with A1, it is one underlying issue.

---

## Traps (do not reintroduce)

- No N-to-daN heuristic. The platform owns units; a second conversion causes a
  decade-off channel.
- No FL/FR/RL/RR hardcoding. Use the platform's generalized axle channel model.
- Sample rate from config, never a typed box. Note the analyzer's 50k-rainflow /
  20k-KDE caps and treat any cap as an explicit accuracy-vs-speed choice, do not
  silently inherit them.
- All new/restyled plots go through `plot_style.py`, and if stop removal is on they
  read the same moving-only derived series the other analysis stages use.

---

## Validation

- B1/B2: regenerate a force-distribution set (Fx/Fy/Fz) on a real study, show
  before/after, and show both x-axis modes.
- A1/B3: show a heatmap and a histogram distance-weighted on a study WITH a usable
  speed channel (confirm it is no longer falling back), and confirm graceful labelled
  fallback on one without. Sanity check: distance weights sum to the recording's
  total km.
- A2: show a Welch PSD on a real channel; sanity check the frequency axis against the
  known sample rate.
- A3/A4: run the two-run comparison on two real studies; confirm the delta tables
  reconcile (DS2 minus DS1 equals the shown delta) and the comparison plots carry the
  filled overlays, P5/P95 lines, and stats strip. For RF Compare, confirm the Miner
  ratio equals the two studies' individual damage numbers divided.

---

## Deliverables

1. The four features (A1–A4) on the platform's channel model / units / config /
   styling.
2. The histogram readability fixes (B1–B3).
3. `MIGRATION_NOTES.md` updated per change.
4. Validation evidence per above.

Do NOT touch: the raw `.raw` reader / 58-sample offset, the FAMOS recipe math, or
ingestion/preprocessing. Do NOT do the GUI restructure (popups to embedded views)
this round. This is analysis-layer and plot-styling work only.

Start with Phase 1: confirm the exact platform functions involved, then give me the
ordered plan and stop.
