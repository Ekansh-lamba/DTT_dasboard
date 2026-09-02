# Task: four small, contained GUI/analysis changes

> **Version control: do not run any git commands.** Do not stage, commit, push,
> branch, tag, checkout, merge, or reset. Only create and edit files. I handle all
> version control myself. This overrides any "commit after each step" phrasing.

Four small, low-risk changes. Keep the existing structure, additive/surgical only, no
restructure. Read `PROJECT_HISTORY.md` first. Same workflow: Phase 1 analyse and report,
Phase 2 short plan and stop for approval, Phase 3 implement incrementally, continue
`MIGRATION_NOTES.md`, leave everything on disk (no commits).

Three of these (items 1, 2, 4) all touch the new-study / add-dataset flow, so do them
together; item 3 is a contained plot port.

## Item 1 — IMC Raw folder as the default source option

On the new-study screen, make "IMC Raw folder" the pre-selected default source type
(instead of whatever is currently default, likely CSV). One default change. Confirm it
does not break the CSV path, the user can still switch to CSV; only the initial
selection changes.

## Item 2 — add multiple raw sessions at once

When adding raw dataset sessions, the user currently has to add one session at a time.
Change the file/folder picker so multiple sessions can be selected and added in one go
(multi-select), instead of repeating the add action per session. Confirm the downstream
handling accepts a batch (the pipeline already reads multiple raw folders/files, per
`imc_reader.read_folder`/`read_files`, so this is a picker-and-list change, not a
pipeline change). Preserve single-add behaviour as a subset (selecting one still works).

## Item 3 — file picker remembers the last-used location

Every "add dataset" picker (raw AND processed/CSV) currently opens at a fixed default
location every time. Change it to open at the last location the user picked from, so
they resume where they left off instead of navigating from scratch. Persist the
last-used directory (per the app's existing settings/QSettings mechanism if one exists;
if not, a simple stored path is fine) and reopen there next time. Apply this to every
dataset-adding picker, raw and processed, consistently.

## Item 4 — replace the boxplot with the analyzer's version exactly

Replace the platform's current boxplot with the boxplot from the attached analyzer
script (`WFT_Analyzer_All_AUC_Gxy_Heatmap_boxplot_PV_July26__1__1.py`), so it renders
exactly like the analyzer's, no visible difference. Replace entirely, do not add a
second style or a new page, the existing boxplot page/stage stays where it is and just
produces the analyzer's look.

- Port the analyzer's boxplot drawing (its fixed per-force-type Y-axis ranges, layout,
  styling) into the platform's existing boxplot module, on the platform's channel model
  and units, NOT the analyzer's hardcoded FL/FR/RL/RR naming and NOT its N-to-daN
  heuristic (the platform owns units).
- Match the visual output exactly; the only differences from the analyzer allowed are
  the channel-model and units plumbing needed to fit the platform.
- Validate by generating the boxplot on a real study and comparing side by side with the
  analyzer's output on the same data; they should look the same.

## Validation

- Item 1: new-study screen opens with IMC Raw preselected; CSV still selectable and works.
- Item 2: select several raw sessions at once, confirm all are added and the run picks
  them all up.
- Item 3: add a dataset from folder X, close and reopen the picker, confirm it opens at
  X, not the fixed default; check both raw and processed pickers.
- Item 4: boxplot matches the analyzer's output on the same data.

Start with Phase 1: confirm the new-study picker code, the settings mechanism for the
last-used path, and the platform's boxplot module, then give the short plan and stop.
