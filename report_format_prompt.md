# Prompt: rebuild the PPTX report to match the reference format

Copy everything below the line into Claude Code, run from the root of the DTT/WFT platform repo. Before running, copy the reference deck into the repo as `docs/report_reference.pptx` (it is the "Project Charter - DTT Tyre Duty Cycle Dashboard" deck; the report format is on slides 8 to 14).

---

## Task

Change the generated PowerPoint report so its result slides match the reference deck `docs/report_reference.pptx`, slides 8 to 14, as closely as possible. The goal is that an analyst opens the generated report and only has to add their own observations, not rearrange or restyle anything.

Do not change any analysis numbers. This is a presentation change only. The FAMOS-validated preprocessing (`famos/`, `dtt/preprocessing.py`) must not be touched, and the statistics, KDE, percentile and heatmap calculations must give identical values before and after.

## Before writing code

1. Read `PROJECT_HISTORY.md` sections 9A, 9B and 9C, then read `dtt/report_builder.py` (or wherever the PPTX is assembled), `dtt/analysis/plot_style.py` (especially `draw_histogram`, `draw_comparison_kde`, `draw_stats_strip`), `dtt/analysis/histograms.py`, `dtt/analysis/heatmaps.py`, `dtt/analysis/boxplots.py` and `dtt/analysis/study_compare.py`.
2. Render the reference slides to PNG (LibreOffice headless to PDF, then `pdftoppm`) and look at them. Also dump their shape geometry with python-pptx. Use these as the specification. The measurements below come from that deck; if anything here disagrees with the deck, the deck wins.
3. Reuse what exists. The two-run comparison plots in `plot_style.py` already draw filled KDE overlays, P5/P95 lines, a Min to Max stats strip and an optional cross-position overlay. Extend those rather than writing a parallel implementation.
4. Write a short plan listing which functions change and which are new, then implement.

## General rules for every slide

- Slide size 13.333 x 7.5 in (16:9).
- Build with python-pptx. Every title, label, pill, divider and summary box is a native, editable PowerPoint shape or text box, not part of a picture. Only the plots themselves are PNG images. This matters because analysts edit the text afterwards.
- Footer text "APOLLO TYRES LTD", bold, dark grey, bottom-left (about x 0.5 in, y 6.9 in) on result slides.
- Analyst observations in the reference (the green, orange and navy speech-bubble callouts, the circled stats values, sentences like "EV rear tyres spend most of their operating life...") are human interpretation. Do not invent conclusions. Instead, create the summary/takeaway boxes in the right position and style containing placeholder text such as "Key observation: ..." so the analyst types into them. Do not generate the circle or bubble annotations.
- Dataset names (EV / IC, MRF / Apollo, M310 / M110), the route label (City, Highway, ...) and the slide title wording come from config or the run's comparison metadata, never hard-coded.
- Plot PNGs at 200 dpi or higher, sized for their placeholder box so text is not stretched.

## Plot style shared by all distribution plots

Match the reference plots:

- Light grey plot background (about #F2F2F2) with a light grid, white figure background.
- Y label "Normalised density"; x label "<CHANNEL> (unit)", for example "FL_Fy (daN)" or "RL_My vs FL_My (Nm)".
- Wheel or side label ("FL", "FR", "RL", "RR", "Left", "Right") written large and bold inside the axes at top-left.
- Compact dark legend box (dark navy/charcoal background, small white text) at top-right, listing both datasets and their P5/P95 values, e.g. "IC P5: -37", "EV P95: 30".
- Stats strip directly under each plot: a grey rounded panel, small monospaced text, one row per dataset, columns separated by "|": `<name> | Min: | P5: | Q1: | Median: | Mean: | Q3: | P95: | Max:`. Values rounded to integers. Extend `draw_stats_strip` if needed. The strip can be part of the plot image.

## Part A: PV format (2 pages, 4 wheel positions, dataset A vs dataset B)

Used for passenger vehicles. Reference slides 9 and 10.

PV colours: dataset A (EV) solid green line (about #27AE60) with light green fill; dataset B (IC) red dashed line (about #E74C3C) with light red fill. P5 lines dotted, P95 lines dashed, each in its dataset colour.

### PV page 1: "Force Distribution - <A> vs <B> - Fy & Fz"

- Title top-left, large bold purple (about #7030A0, 36 to 40 pt).
- Orange rounded "pill" at top-right (about x 11.9 in, y 0.23 in, 1.35 x 0.34 in, fill about #ED7D31, white text) showing the route label, e.g. "City".
- Thin black vertical divider at x about 6.74 in, from y 0.55 in to about 6.85 in.
- Left half: heading "Fy" centred over the half (bold, 16 pt). A 2 x 2 grid of plots: FL top-left, FR top-right, RL bottom-left, RR bottom-right, each about 3.2 x 2.0 in with its stats strip below. Approximate image positions: (0.1, 1.25), (3.5, 1.25), (0.1, 4.1), (3.5, 4.1).
- Right half: heading "Fz", same 2 x 2 layout. Approximate image positions: (6.8, 1.25), (10.1, 1.25), (6.8, 3.95), (10.1, 3.95).
- X ranges for PV: Fy -300 to 300 daN, Fz 100 to 1000 daN. Put these in config as per-vehicle-type defaults.
- Takeaway box bottom-right (about x 6.88, y 6.66, 6.36 x 0.52 in): light grey fill, bold purple text (#6C2DC7, about 11 pt), placeholder text.
- Footer bottom-left.

### PV page 2: "Force Distribution - <A> vs <B> - Fx"

- Same title style and route pill. Same vertical divider.
- Left half: navy rounded pill "By Wheel Position" (about x 0.13, y 0.65, fill dark navy about #0B1F8C, white text) and heading "Fx". 2 x 2 grid FL / FR / RL / RR of Fx, x range -300 to 300 daN, with stats strips.
- Right half top: navy pill "Drive Tyres Comparison" and heading "Fx". Two plots side by side, "Left" and "Right", comparing the driven axle of each dataset (cross-position overlay). In the reference, EV is rear-wheel drive and IC is front-wheel drive, so "Left" overlays RL_Fx (A) with FL_Fx (B) and "Right" overlays RR_Fx (A) with FR_Fx (B); the x label reads "RL_Fx vs FL_Fx (daN)". The driven axle per dataset must come from config (e.g. `drive_axle: rear | front`), not be assumed.
- Right half bottom: heading "My", the same Left / Right cross-position comparison for My in Nm, x range -900 to 700 Nm.
- Takeaway box bottom-left (about x 0.15, y 6.86, 8.9 x 0.52 in), and an "Overall" summary box bottom-right (about x 9.25, y 6.49, 3.6 x 0.97 in) with bold purple text and bullet placeholders. Both light grey fill.
- Footer.

## Part B: CV format (2 axles, reference dataset vs candidate dataset)

Used for commercial vehicles. Reference slide 12. One slide per comparison.

- Title: "Force Distribution" large bold black, followed on the same line by a smaller bold subtitle from config, e.g. "Diff Btw MRF vs Apollo - 10x2 Rigid Drive Axle & Tag - RWUP - Lift Axle up".
- Two rows separated by a full-width horizontal black line at y about 3.75 in. Row 1 is the drive axle (channels like D1_Fz), row 2 the tag axle (T1_Fz). Axles and their channel prefixes come from the existing generalized axle model (`docs/GENERALIZED_AXLE_CONCEPT.md`), so other axle configurations work too.
- Row label at the left (about x 0.15, 1.3 in wide): two lines, bold black "<Axle> Force dstrb'n" then bold light blue (#00B0F0) "<Candidate> vs <Reference>".
- Three columns of plots per row, in this order: Fz, Fx, Fy, with bold column headings above the top row. Each plot about 3.6 x 2.4 in at x about 1.6, 5.5, 9.3 in, with its stats strip below.
- CV plot style differs from PV. The reference dataset is the baseline:
  - Reference (e.g. MRF): grey dashed line, with the area between reference P5 and P95 filled green ("MRF normal zone (P5-P95)").
  - Candidate (e.g. Apollo): solid line, blue (about #2E75D0) in the drive row and red (about #E74C3C) in the tag row, with the area where the candidate exceeds the reference P95 filled red ("Apollo exceeds MRF P95").
  - Vertical lines: reference P5 green dotted, reference P95 red dashed, candidate P95 dashed in the candidate colour.
  - Legend entries in this order: normal zone, exceeds P95, reference, candidate, reference P5, reference P95, candidate P95, each with its value.
- X ranges for CV: Fz 5000 to 12000 daN, Fx -1000 to 1000 daN, Fy -1000 to 1000 daN (config defaults per vehicle type).
- One empty editable text box inside each plot area for the analyst's note (the reference has notes like "3700 daN in tag single tyre"), no fill, 11 to 14 pt.
- Footer.

## Part C: Box plots

The reference deck says "For box plot, both CV and PV remain the same", with no example slide. Keep the current box plot slides as they are (they were already matched to the analyzer in Round 6). Only apply the shared footer and title style so they look consistent with the new slides.

## Part D: Fx and Fy combination heatmap slide

Reference slide 14. Compares two datasets, all four wheels.

- Title "Combination of Fx & Fy" large bold black, then " - All wheels" smaller on the same line. Subtitle below in magenta (about #BC32BE, 16 pt): "<A> Vs <B> - Heatmap of Fx & Fy".
- Vertical light blue divider at x about 6.7 in.
- Each half: a 2 x 2 grid of heatmaps titled "FL - Fx vs Fy", "FR - Fx vs Fy", "RL - Fx vs Fy", "RR - Fx vs Fy", with a figure title "Fx vs Fy Heatmap - % of Occurrences" above. Dataset name in large bold text centred under each half.
- Heatmap details to match:
  - Units N, both axes -4000 to 4000, bin width 1000 N.
  - Y axis Fx with negative values at the top (-4000 top, 4000 bottom). X axis Fy with positive values on the left (4000 left, -4000 right). Check these against the reference image and the existing sign conventions; if the current code uses the opposite orientation, follow the reference.
  - Empty and near-zero cells drawn in the darkest colour of the colormap, so the background reads dark purple.
  - Colormap like the reference (dark purple through blue, cyan, green, yellow, orange to dark red, for example matplotlib "turbo").
  - Each non-empty cell (0.1% or more) annotated with its percentage, one decimal, small white bold text.
  - One shared vertical colorbar on the far right labelled "Percentage of Occurrences (%)", same scale for both datasets so colours are comparable.
  - On the first plot only, grey axis tags: "Acc" (upper Fx) and "Brake" (lower Fx) beside the y axis, "Left turn" and "Right turn" under the x axis. Check which sign means which against the existing channel conventions.
- Top-right: an editable PowerPoint table "Projected tyre life" with columns Left / Right per dataset and rows Front / Rear, left empty for the analyst (the tool does not compute tyre life). Same layout as the reference.
- Bottom: a rounded rectangle with a dashed magenta border and light grey fill spanning almost the full width (about x 1.64, y 6.76, 11.7 x 0.67 in) with two bullet placeholders.
- Footer.

## Slide order in the generated report

1. Existing cover / summary slides (unchanged).
2. A section divider slide "PV Format" or "CV Format" depending on vehicle type (simple centred title, like reference slides 8 and 11).
3. PV: page 1 (Fy & Fz) and page 2 (Fx & My). CV: the axle comparison slide.
4. Box plots.
5. Fx & Fy combination heatmap.
6. Remaining existing slides (rainflow, PSD, etc.) unchanged.

Single-study runs (no second dataset) should still produce the same layouts with a single dataset drawn in the dataset A style, and the stats strip showing one row.

## Validation

1. Run the existing test suite; all tests must still pass.
2. Add a test that a sample report contains the expected slide titles in order, and that every text element listed above is a native text shape, not baked into an image.
3. Generate one PV comparison report and one CV comparison report from real studies (current-build studies only, not stale ones). Render the generated slides and the reference slides to PNG and compare them side by side. Fix layout differences until positions, sizes, colours and fonts match closely.
4. Confirm the numbers in the stats strips equal the existing statistics output for the same channels (values must not change).
5. Report back with the rendered PNGs of every new slide next to the matching reference slide, a list of anything that could not be matched and why, and the list of new config keys with their defaults.
