# Task: FAMOS all-channel cross-validation, Preprocess graph scaling, two-colour AUC comparison, dataset labels

> **Version control: do not run any git commands.** Do not stage, commit, push,
> branch, tag, checkout, merge, or reset. Only create and edit files. I handle all
> version control myself. This overrides any "commit after each step" phrasing.

Four items. Items 3 and 4 touch the same comparison/AUC surface so do them together;
items 1 and 2 are independent. Keep the existing structure — additive/surgical, no
restructure. Read `PROJECT_HISTORY.md` and `MODULE1_STATUS.md` first. Same workflow:
Phase 1 analyse and report, Phase 2 short plan and stop for approval, Phase 3 implement
incrementally, continue `MIGRATION_NOTES.md`, leave everything on disk (no commits).

Reference script for items 3 and 4 is `WFT_Analyzer_All_AUC_Gxy_Heatmap_boxplot_PV_July26 (1).py`
in the repo root. Port its *look and behaviour*, on the platform's channel model and
units — never its hardcoded FL/FR/RL/RR names, never its N-to-daN heuristic.

---

## Item 1 — FAMOS cross-validation across ALL channels

I am going to load the same raw data into imc FAMOS, run the production recipe, and
export the result as CSV. Your job is the platform side: given that FAMOS CSV plus the
same raw input, cross-validate **every channel**, not a hand-picked few, and report
per-channel agreement.

**You do not run FAMOS.** It is licensed and the trial used for the 2026-08-29 golden
corpus is spent. Build the tooling so it takes my exported CSV as an input file. Do not
plan any step that requires a live FAMOS session.

- Start from what already exists rather than writing a third validator:
  `dtt/validation/famos_validation.py` (`crosscheck_csv`, `match_metrics`,
  `find_reference_pairs`) and `tools/score_golden.py` (which already scores against the
  FAMOS 6-significant-figure quantum and has a `--raw-dir` end-to-end force mode). In
  Phase 1, report which of the two is the right spine for an all-channel run and why.
- **Coverage is the point of this round.** `MODULE1_STATUS.md` records two gaps that were
  inferred rather than measured: the **moment channels (Mx/My/Mz)** and **Latacc on a real
  recording**. This round must close both, and must cover every column in the export —
  forces, moments, Latacc/Latacc_LPF, speed, anything else present — with an explicit
  "not compared, and why" line for any channel it cannot match instead of silently
  dropping it.
- Channels must be matched **canonically**, via `dtt/channels.py::parse_channel` /
  `dtt.comparison.canonical_map`, not by raw column string. FAMOS export names will not
  equal our column names.
- **Score against the per-sample quantum, never a fixed absolute tolerance.** This trap
  has already bitten once: FAMOS writes 6 significant figures, which is 1e-6 at order 1
  but 1.0 once an Fz peak crosses 1e5, so a fixed tolerance reports healthy channels as
  failing while r = 1.000000000. Reuse `score_golden.quantum_ratio`; <= 0.5 is the
  rounding floor. Report `err/q`, correlation, and % of samples within tolerance per
  channel.
- **Expect, and account for, the three deliberate differences.** `processed_data.csv` is
  intentionally NOT byte-identical to a FAMOS export: the decade unit fix, dropout
  blanking, and optional stop removal are corrections FAMOS does not make. The validator
  must either compare at the pre-correction stage or state the correction per channel —
  it must not report our improvements as failures. Say clearly in the output which
  differences are expected.
- **The `Latacc` / `Latacc_LPF` direction trap:** per the imc recipe
  `Latacc = smo(Latacc_LPF, 0.5)` — the bare column is the *output*, `_LPF` is the
  intermediate. An assumed (raw, filtered) direction scores ~96.8% regardless of code
  quality and measures nothing. Test both directions and report which relationship
  actually holds, as `crosscheck_csv` already does.
- Surface the result where the existing FAMOS check already lives (the Preprocess page's
  "Validate vs FAMOS…" action) plus a written per-channel table (CSV or markdown) I can
  read outside the GUI. A pass/fail summary alone is not enough; I want the number for
  every channel.
- Benchmark stays >= 95% match = FAMOS-grade.

Deliverable: a documented command I can run as `python tools/<script>.py <famos_export.csv>
--raw-dir <raw folder>` (or equivalent), and a per-channel report. Tell me exactly what to
export from FAMOS — which channels, which recipe, what format, header layout — before I
run it, so the export is usable on the first attempt.

---

## Item 2 — Preprocess graph scaling: raw spikes, processed over-filtered, at full recording

This has been wrong for a long time. On the Preprocess page's before/after preview, when
the span is switched to **"Full recording"**, the raw trace reads as a dense wall of
spikes and the processed trace reads as a flat over-filtered line. The gap between them is
far larger than what the filter actually removes, so the plot misrepresents the
conditioning. At short spans (10 s / 120 s) the pair looks correct.

**The cause is in the reduction, not the filter** — confirm this in Phase 1 before
changing anything. In `gui/pages/preprocess_page.py`, the two traces are reduced to screen
width by **different statistics**:

- raw → `_envelope_line()` → each bucket's **min→max** stroke (every extreme reaches the
  screen)
- processed → `_median_line()` → each bucket's **median** (a centre line)

At 120 s that is ~13 samples per bucket and the two agree. Over a full multi-thousand-second
recording it is hundreds of samples per bucket, and min/max vs median diverge by most of
the channel's range *on identical data*. The code comments at `_reduce()` and
`_median_line()` already acknowledge this ("at full-run zoom part of the gap between them
is the reduction itself, not only the filter") — that acknowledgement is the bug, not a
justification. `_density_style()` currently compensates by fading the raw as density
grows, which treats the symptom.

What I want:
- The two traces reduced **comparably**, so that the visual difference between blue and
  red is the conditioning and nothing else. The honest options are (a) draw both as
  min→max envelopes, (b) draw both as median centre lines, or (c) keep the raw envelope
  but add the processed channel's envelope behind its median so like is compared with
  like. Pick one, justify it in Phase 1, and say what it costs.
- Whatever is chosen must still not lie about the raw: a genuine isolated pothole spike
  must remain visible as a spike, and dropouts must still break the line rather than
  being ruled across (the NaN-bucket behaviour in `_reduce` stays).
- Do not fix this by stride-decimating — that was the earlier bug, it lets isolated
  samples masquerade as spikes.
- Whatever residual reduction artefact remains at full-recording zoom must be **stated on
  screen** (a caption or note saying the view is reduced to N buckets), not left for the
  operator to infer.
- Check the same asymmetry does not exist in the spike/de-glitch markers
  (`_spikes()` scores the raw-minus-processed residual at full sample rate, so it should
  be unaffected — confirm, don't assume).

Validate on a full recording: same channel, same settings, screenshot the preview at 10 s,
120 s and Full recording, before and after the fix. The before/after gap at Full recording
must shrink to what the short spans show.

---

## Item 3 — two-dataset AUC comparison, two colours only

Port the reference script's AUC comparison rendering into the platform, for comparing
**two already-processed datasets**.

Current state to confirm in Phase 1: `gui/pages/comparison_page.py` already draws a
KDE + histogram distribution overlay using `dtt/analysis/auc.py`
(`auc_grid`, `compare_distributions`), but it compares two **raw imc folders**.
`dtt/analysis/study_compare.py` compares two **finished studies** (it already carries
`label_a` / `label_b`) but has **no GUI surface at all**. Report which of these should own
the two-processed-dataset AUC view before building it.

Match the reference's `_auc_generate` (around line 1287):

- **Exactly two colours, fixed:** dataset 1 green `#27AE60` solid, dataset 2 red
  `#E74C3C` dashed, per `AUC_DS1_COLOR` / `AUC_DS2_COLOR`. Every element for a dataset —
  KDE line, fill, histogram bars, P5 and P95 vlines — uses that dataset's one colour,
  distinguished by line style (`:` for P5, `--` for P95), never by a third or fourth
  colour. The platform's current `_PREV_COLOR` blue / `_CURR_COLOR` red pair is replaced
  by this scheme on this view.
- Two panels side by side: **"Shaded Area — KDE Curves"** left, **"Density Histogram
  Overlay"** right, shared x-range, "Normalised density" y-label on both.
- The stats footer strip underneath: reference P5/P95, current P95 and delta, % of cycles
  exceeding the reference P95, % within the reference normal zone. The platform already
  computes all of this in `AucComparison` / `compare_distributions` — reuse it, do not
  recompute.
- **Keep the platform's `dtt.analysis.auc.kde_curve`. Do not port the reference's
  `KDE_MAX_SAMPLES = 20000` subsampling.** Scott's bandwidth scales as n^(-1/5), so
  fitting on 20k points instead of 600k widens the kernel ~1.7x and smears separate peaks
  into one — faster and quietly wrong. Our binned-convolution KDE is already O(n) and
  agrees with the exact full-data KDE to ~1e-6.
- **Fix the reference's legend, do not copy it.** It sets `labelcolor="white"` on a
  `#F5F5F5` light panel — that is the unreadable legend in my screenshot. Use the
  platform's light-theme legend colours from `dtt/analysis/plot_style.py` /
  `histograms.py`, and keep the readability standards already agreed for the Histogram and
  AUC sections (legible tick numbers, axis labels not clipped, nothing overlapping).
- Channel selection follows the platform's channel model — all available force **and
  moment** channels, whatever the study recorded. No fixed 12-channel FL/FR/RL/RR list.

---

## Item 4 — labelled datasets

Wherever two datasets are shown together, I must be able to name them, and the name must
appear everywhere that dataset is referenced. Right now the comparison view hardcodes the
strings `"previous"` and `"current"` into the plot legends and titles
(`gui/pages/comparison_page.py`, lines ~352, ~380, ~387), which is useless when the two
runs are EV and IC.

- Two editable label fields in the UI, exactly like the reference's `Label 1:` / `Label 2:`
  entries (reference lines 204–205, 311), defaulting to something sensible (the study or
  folder name if available, otherwise "Dataset 1" / "Dataset 2").
- Editing a label re-renders with the new name. No restart, no re-run of the analysis.
- The labels must flow through **everything** on that comparison: AUC panel legends, P5/P95
  legend entries, the figure suptitle, the stats footer, the histogram overlay legend, the
  delta/summary table headers, and any exported CSV column names — the reference does this
  via `f"P95_{lbl2}"` style keys, and `ComparisonResult` / `StudyComparisonResult` already
  carry `previous_label`/`current_label` and `label_a`/`label_b`, so plumb those through
  rather than adding a parallel label store.
- This covers the histogram comparison too: when two raw datasets are overlaid, both must
  be renameable, not just the AUC view.
- Labels are cosmetic only — they must not change which dataset is the reference for
  P5/P95 and exceedance maths.

---

## Constraints

- Platform channel model and units throughout; no FL/FR/RL/RR hardcoding, no N-to-daN
  re-conversion, no fixed 12-channel assumption.
- Reuse the existing helpers: `dtt.analysis.auc`, `plot_style`, `histograms`' style
  constants and axis-range logic, `famos/ops.py` for any smo/filtlp/red (that file is the
  **only** implementation — never add a second copy), `score_golden.quantum_ratio` for
  quantum scoring.
- Item 1 adds validation tooling only; it must not change preprocessing behaviour.
- Item 2 is a display fix; it must not change what the pipeline actually computes or
  writes to `processed_data.csv`.
- Items 3 and 4 change rendering and labelling only; no change to KDE, binning,
  percentile, or exceedance maths.

## Validation

1. Run the all-channel cross-check against my FAMOS export; show the per-channel table,
   including the moment channels and Latacc, with `err/q` and correlation per channel, and
   an explicit list of channels not compared with the reason.
2. Preprocess preview screenshots at 10 s / 120 s / Full recording, before and after, same
   channel and settings.
3. AUC comparison on two processed studies: two colours only, both panels, readable
   legend, footer stats present.
4. Rename both datasets to "EV" and "IC", re-render, and show that the new names appear in
   every legend, title, footer, table header and exported column.

Start with Phase 1: report (a) which validator is the right spine for an all-channel FAMOS
check and what the export needs to contain, (b) confirmation of the reduction asymmetry in
the Preprocess preview and which of the three fixes you recommend, (c) which module should
own the two-processed-dataset AUC view and where the label fields belong. Then give the
plan and stop.
