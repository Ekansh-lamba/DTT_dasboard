# Task: compare the standalone WFT Analyzer against the platform (comparison only, no changes)

I am giving you a standalone script,
`WFT_Analyzer_All_AUC_Gxy_Heatmap_boxplot_PV_July26__1_.py` (attached). I am NOT
asking you to integrate it yet. I want a clear comparison of what it does versus
what the platform already has, so I can decide afterwards what is worth porting.

Do not change any code. Do not plan an integration. Just analyse and report.

Read `PROJECT_HISTORY.md` first for the platform's architecture, units story,
channel model, and existing analysis stages; it was just updated and has the
context you need. Then read the actual `dtt/analysis` modules (`statistics.py`,
`histograms.py`, `heatmaps.py`, `rainflow.py`, `severity.py`, `auc.py`,
`spectral.py`, `plot_style.py`, and any others), do not guess from names.

Context you need about the script: it is a single-file Tkinter/matplotlib GUI with
one god-class. None of the UI matters. It has no ingestion, no FAMOS recipe, no
preprocessing, it reads an already-processed CSV and analyzes it. So the only thing
worth comparing is its analysis and visualization math against ours.

## What to produce

A feature-by-feature comparison table. The analyses in the script are: percentile
(P80/P90/P95), rainflow From-To matrix (16x16 Siemens-style), box plots (fixed
per-type Y-axis), AUC/KDE comparison (plain + cross-position + moments), RMS table
(2-dataset delta), DLC (std(Fz)/mean(Fz)), G-severity (Gx=RMS(Fx/Fz), Gy=RMS(Fy/Fz),
Gxy=sqrt(mean(Fx²+Fy²)/mean(Fz)²)), distance-weighted force-pair heatmaps
(Fy-Fx / Fx-Fz / Fy-Fz, weighted by km not sample count), Welch PSD per wheel, and
export-to-PNG.

For each analysis, report:
- **Does the platform already have it?** Yes/no, and the exact platform
  function/module it maps to.
- **If yes, whose version is better, and specifically how?** Richer, equivalent, or
  worse, with the concrete difference (e.g. "ours does 1D range histogram only,
  script does a full 16x16 from/to matrix").
- **Verdict:** one of `already-have (skip)`, `port — genuinely new`,
  `port — better than ours`, or `superseded by platform (skip)`.
- **Any catch** that would matter if we later ported it (see the three traps below).

End with a short recommended port-list in priority order, your honest opinion of
what is actually worth taking, but frame it as a recommendation for me to decide on,
not a plan you are about to execute.

## My own read, to check against — not to take on faith

- **Distance-weighted heatmaps** — likely the most valuable thing here: weights the
  force-plane density by distance (Vehicle_Speed x sample rate) instead of sample
  count. If ours are count-weighted, this is a real upgrade.
  (`_dist_km_per_sample`, `_open_heatmap_pair_popup`.)
- **Rainflow From-To matrix** — if our `rainflow.py` only does 1D range histograms,
  the 16x16 from/to matrix is worth it. (`_build_fromto_matrix`.)
- **G-severity / DLC** — pure formulas, but we already have G-severity (the
  merge-conflict story in PROJECT_HISTORY mentions an FR Gx figure). Diff the
  formulas; if they differ, flag it as a correctness question.
- **AUC/KDE, percentiles, box plots, PSD, export** — probably already covered or
  superseded by our `auc.py` / `plot_style.py` / `spectral.py` / report builder.
  Confirm rather than assume.

## Three traps to note in the table (relevant only if we later port)

These do not need fixing now, just flag them per-feature where they apply, so the
comparison is honest about porting cost:
1. **N-to-daN heuristic** — `_parse_csv` divides by 10 if max median > 1000. We own
   units; a second conversion is how a channel ends up a decade off. Any port must
   drop it.
2. **Hardcoded FL/FR/RL/RR** — the script hardcodes 4-wheel naming everywhere; we
   have a generalized axle model. Any port must move to our channel abstraction.
3. **Typed sample rate + sample caps** — script reads SR from a text box and caps
   rainflow at 50k / KDE at 20k for GUI speed. Ours comes from config, and the caps
   are an accuracy-vs-speed choice, not a default to inherit.

## Do not

Do not modify any code, do not port anything, do not start an integration plan. This
is a read-and-report task only. When the comparison table and port-list are done,
stop. I will decide what to move forward with from there.
