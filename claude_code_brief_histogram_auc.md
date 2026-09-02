# Task: split histogram and AUC into two sections, and fix readability/scaling

> **Version control: do not run any git commands.** Do not stage, commit, push,
> branch, tag, checkout, merge, or reset. Only create and edit files. I handle all
> version control myself. This overrides any "commit after each step" phrasing.

Two related changes to the distribution plots. Keep the existing page structure, this is
additive/refactor within the analysis-plot layer, not a restructure. Read
`PROJECT_HISTORY.md` first. Same workflow: Phase 1 analyse and report, Phase 2 plan and
stop for approval, Phase 3 implement incrementally, continue `MIGRATION_NOTES.md`, leave
everything on disk (no commits).

Scope note: this round is SINGLE-study only. Do NOT build the two-run comparison / AUC
overlay (EV-vs-IC, SA1-vs-SA2) from the reference images, that is a separate future
round with its own study-picker design. The reference comparison images attached are
here only as the visual bar for how a clean, readable AUC plot should look (axes, labels,
stats strip, spacing), not as a feature to build this round.

## Item A — separate "Histogram" and "Area Under Curve" into two sections

Currently histogram and AUC content are mixed/conflated. Split them into two distinct
sections/views: one plain **Histogram** section and one separate **Area Under Curve
(AUC)** section, each on its own, so the user views them independently rather than mixed.

- In Phase 1, report how histograms and AUC are currently produced and surfaced (which
  modules, which pages), so the split is done cleanly against the real structure.
- Keep both on the existing page structure (e.g. two views/tabs or two entries), do not
  restructure navigation. If they are currently one page, splitting into two clearly
  labelled sections/views is enough.
- No change to the underlying histogram or AUC math, only how they are organised/surfaced
  and styled (Item B).

## Item B — fix readability and axis scaling on histograms AND AUC

The current histograms and AUC plots have cramped, hard-to-read axes, labels, and
numbers. Fix readability on both, using the attached reference AUC images as the visual
bar (clean readable axis tick numbers, legible axis labels, visible per-plot stats where
present, enough spacing that nothing overlaps or is crushed).

Concretely:
- **Axis tick numbers and axis labels**: large enough to read, not overlapping, not
  clipped. Fix the font sizes and padding so the numbers are clearly visible.
- **Axis ranges/scaling**: where an axis is scaled so the data is crushed into a corner
  or the ticks are unreadable, fix the scaling so the distribution and its axis read
  clearly (this connects to the existing histogram x-axis full/autoscale option, reuse
  it, do not invent a parallel mechanism).
- **Layout spacing**: enough margin/spacing that labels, titles, and any stats text are
  not cut off or piled on top of each other, matching how the reference images breathe.
- Keep the clean light style already adopted for the force histograms (light background,
  thin curve, soft fill); this is about legibility and spacing, not re-theming.
- Apply the same readability treatment to both the Histogram section and the AUC section
  so they are consistent.

## Constraints

- Single-study only; no two-run comparison view this round.
- No change to histogram/AUC math, binning, KDE, or weighting, this is organisation and
  legibility only.
- Reuse the existing plot-style helpers and the existing x-axis range option; do not add
  a parallel styling or scaling mechanism.
- Platform channel model and units throughout; no FL/FR/RL/RR hardcoding, no N-to-daN
  re-conversion.

## Validation

- Regenerate the Histogram section and the AUC section on a real study; confirm they are
  now two separate views, and that axis numbers, labels, and any stats text are clearly
  readable and not crushed, comparable in legibility to the reference images.
- Show before/after of one wheel's plots for both sections.

Start with Phase 1: report how histograms and AUC are currently produced/surfaced and
where the readability/scaling is set, then give the plan and stop.
