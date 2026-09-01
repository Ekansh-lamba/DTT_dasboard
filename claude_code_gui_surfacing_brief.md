# Task: surface terminal-only features into the GUI (conservative, additive pass)

> **Version control: do not run any git commands.** Do not stage, commit, push,
> branch, tag, checkout, merge, or reset. Only create and edit files. I handle all
> version control myself. This rule overrides any per-step "commit after each item"
> phrasing elsewhere, wherever a step says to commit, just leave the changes on disk
> for me to review and commit.


This is NOT a restructure. Keep every existing page and its layout exactly as they are.
The goal is narrow: recent rounds added features that currently have to be driven from
the terminal/CLI, and I want them reachable from the GUI, with sensible default values
pre-filled and with controls that are clearly visible (proper contrast, matching the
existing style). No page reorganisation, no navigation changes, no deeper restructure,
that is explicitly high-risk and out of scope.

Read `PROJECT_HISTORY.md` and `MIGRATION_NOTES.md` first. Same workflow as prior
rounds: Phase 1 analyse and report, Phase 2 ordered plan and stop for approval, Phase 3
implement incrementally with a control still working after each change, continue
`MIGRATION_NOTES.md`, branch, commit per step.

---

## Phase 1 — audit what is GUI-reachable vs terminal-only (this is the core of Phase 1)

I do not have a reliable list of which recent features got a GUI surface and which are
still CLI-only, so do not assume, find out from the code. Go through the features added
in recent rounds and, for each, report whether it is currently reachable from the GUI or
only from the CLI / `RunConfig`. At minimum cover:

- The three workflow modes (`workflow_mode`: preprocess / analysis / both).
- The FAMOS-validation CSV export (`export_famos_validation_csv`).
- Despike controls (rail/dropout/sub-width, the 20% post-smooth transient rule) and their
  parameters.
- Stop removal (speed threshold, min duration) and its on/off.
- The histogram x-axis range mode (full vs autoscale).
- Two-run comparison (RMS/DLC/G-severity deltas, RF Compare) outputs.
- Welch PSD outputs.
- Anything else added recently that only has a CLI flag or `RunConfig` field and no GUI
  control or no way to view its output.

Produce a table: feature, current GUI status (reachable / partial / terminal-only), and
where in the GUI it would naturally belong given the existing page structure. End Phase 1
with that table plus a proposed additive plan, then stop for approval. Do not start
adding controls before I approve the list, so we agree on what gets surfaced.

---

## What to add (additive only, on existing pages)

For each terminal-only feature the audit finds, add a control on the page it naturally
belongs to, without moving or restructuring anything:

- **Workflow mode**: a dropdown on the run / new-study screen with options
  preprocess / analysis / both. **"both" preselected as the default**, so the app
  behaves exactly as it does today unless the user changes it.
- **FAMOS-validation CSV export**: a clearly-labelled checkbox near where preprocessing
  is configured/run, off by default, matching the CLI flag.
- **Despike / stop-removal / x-axis-mode / any other CLI-only parameters**: expose as
  controls on their relevant existing page, defaulting to the current defaults.
- **Comparison / PSD outputs**: if these are produced but have no way to be viewed in the
  GUI, add a way to open/view them (open the output folder, or show the saved PNG/JSON),
  consistent with how existing outputs like rainflow/severity are surfaced. Do not build
  a new picker or a new page, reuse the existing pattern.

## Recommendation / default values (required)

Every input control that takes a value must be pre-filled with a sensible recommended
default, so the user picks from good starting values instead of typing from memory. Use
the values already validated in the pipeline, not invented ones:
- smo widths 0.1 s (force/moment) / 0.5 s (accel/speed), FiltLP order 4 / 5 Hz,
  decimation 10.
- despike defaults as currently in `RunConfig` (rail run, dropout run, hardware cutoff
  200 Hz, the 20% / 1 s transient rule, net off by default).
- stop removal 1.5 km/h, 3 s minimum.
- x-axis mode default matching the current code default.
Pull these from the existing `RunConfig` defaults so the GUI and the pipeline agree,
rather than hardcoding a second copy that can drift.

## Visibility / contrast (required)

- New buttons and controls must have proper contrast against the dark theme, no
  low-contrast text or controls that blend into the background.
- Match the styling of existing controls so additions look native, not bolted on.
- Buttons must be clearly visible and readable at normal window size.

---

## Constraints / do-nots

- No page reorganisation, no navigation restructure, no popups-to-embedded rework. Keep
  the current structure.
- Do not touch backend / pipeline / analysis logic. This is surfacing existing
  functionality only, the controls set the same `RunConfig` fields / CLI behaviour that
  already exist.
- Do not change defaults. "both" is the default mode; every other control defaults to its
  current value so behaviour is unchanged unless the user acts.
- Do not duplicate a control that already exists, the Phase 1 audit is what prevents this.
- Keep each page working after each change; add controls incrementally, not in one big
  rewrite.

## Validation

- After each control is added: launch the GUI, confirm the control appears with proper
  contrast, its default matches the current pipeline default, and setting it produces the
  same result as the equivalent CLI flag/field (e.g. selecting "preprocess" mode runs
  preprocess-only just as `--mode preprocess` does; ticking the FAMOS-CSV box writes the
  file).
- Confirm the default state of the whole screen reproduces today's behaviour (mode
  "both", every other control at its current default) so nothing changes for a user who
  just clicks run.

## Deliverables

The Phase 1 audit table; the added GUI controls with defaults and proper contrast;
`MIGRATION_NOTES.md` updated; validation evidence per control.

Start with Phase 1: audit GUI-reachable vs terminal-only, produce the table and the
additive plan, and stop for my approval.
