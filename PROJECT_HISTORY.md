# DTT / WFT Automation Platform — Project History & Context

A single narrative record of the project: what it is, what was built, every
technical issue found and how it was fixed, and every conflict (design
disagreements and literal git merge conflicts) hit along the way and how they
were resolved. Written so someone new to the project — or picking it back up
after a gap — can get the full picture without reconstructing it from commit
messages and chat logs.

**Companion documents**, each narrower and more detailed on its own topic:
- [`MIGRATION_NOTES.md`](MIGRATION_NOTES.md) — step-by-step technical log of
  the FAMOS-exact preprocessing migration (Round 1 and Round 2 below), with
  before/after numbers for every change.
- [`MODULE1_STATUS.md`](MODULE1_STATUS.md) — current validation status
  against a licensed imc FAMOS, feature completion table.
- [`golden_corpus/VALIDATION_GUIDE.md`](golden_corpus/VALIDATION_GUIDE.md) —
  how to reproduce the FAMOS validation on another machine/licence.
- [`docs/PROJECT_CONCEPTS_GUIDE.md`](docs/PROJECT_CONCEPTS_GUIDE.md) — plain
  language explanation of every domain concept (WFT, Fx/Fy/Fz, rainflow,
  G-severity, etc.) and every pipeline stage/GUI screen.
- [`docs/GENERALIZED_AXLE_CONCEPT.md`](docs/GENERALIZED_AXLE_CONCEPT.md) —
  the axle-dynamic channel model (any axle count, not just a 4-wheel car).

---

## 1. What this project is

The DTT (Digital Tyre Testing) WFT Automation Platform replaces a manual
CSV → FAMOS → Python → PowerPoint workflow with a single automated pipeline.
It ingests Wheel Force Transducer (WFT) recordings — the forces and moments
at each wheel, hundreds of samples per second, captured over real-world
durability routes — reproduces the exact signal conditioning a licensed imc
**FAMOS** installation would apply, and turns the result into statistics,
histograms, heatmaps, rainflow fatigue counts, load-severity numbers, and a
PowerPoint report, through a PySide6 desktop GUI or a CLI.

The one architectural fact that explains almost every decision in this
document: **the platform's entire value proposition is that its numbers
match a real FAMOS export.** Every operator (`smo`, `FiltLP`, `red`) was
reverse-engineered against genuine matched raw/processed FAMOS data, not
implemented from a spec sheet. Every change to that recipe had to be
re-validated against ground truth, and several "obviously correct" textbook
implementations turned out to be measurably wrong against the real thing.

### Codebase shape
- `dtt/` — the pipeline: ingestion, validation, sanitization, signal
  processing, statistics, histograms, heatmaps, boxplots, rainflow,
  severity, reporting.
- `famos/` — the canonical FAMOS operator library (`smo`, `FiltLP`, `red`,
  and a chain/translation layer), added mid-project to end three drifting
  copies of the same math (see §4.9).
- `gui/` — the PySide6 desktop app, one page per pipeline concern.
- `golden_corpus/` — synthetic and real-recording test sequences captured
  from a licensed FAMOS trial, used to score the Python operators against
  ground truth without needing a permanent licence.
- `tests/` — unit tests for the `famos/` library, the moment-channel
  preprocessing path, session-join time mapping, logging/provenance, and the
  golden-corpus scores against licensed-FAMOS output (§9.6).
- `data/` — real (gitignored) FAMOS ground-truth exports used to validate
  changes locally; never committed.

---

## 2. Timeline

| when | what | by |
|---|---|---|
| project start | initial pipeline, GUI, exe packaging | prior work |
| 2026-08-09 | **Round 1**: FAMOS-exact preprocessing migration (§4.1–4.7) | Claude + user |
| 2026-08-25 (in between) | `famos/` operator library, golden corpus, stationary-removal library-only | nik123-py + Claude Opus 5 |
| 2026-08-25 | **Round 2**: resample confirmation, histogram restyle, 20% spike rule, stop removal wired into pipeline (§5) | Claude + user |
| 2026-08-28–29 (parallel) | session joining, early stop-removal wiring, golden-corpus scoring, preprocess GUI work, unit/channel-matching fixes | nik123-py + Claude Opus 5 |
| 2026-08-29 | **merge**: reconciling two independent stop-removal implementations that landed at once (§6) | Claude + user |
| 2026-08-29 | leak/credential audit before push (§7) | Claude |
| 2026-08-29 | "spiky preprocess graph" investigation — real data, not a bug (§8) | Claude |
| 2026-08-29 | validation guide + Module 1 status docs | nik123-py + Claude Opus 5 |
| 2026-08-30 | this document, plus detailed method appendices §12–14 (imc3 format, smo kernel search, despike basis) | Claude |
| 2026-09-01 | packaged Windows `.exe` rebuilt and shipped (onefile) | Claude + user |
| 2026-09-02 | **Round 3**: moment (`f*_m*`) channels had no stored raw reference — preprocessing was invisible, not absent; golden-corpus scoring automated (§9) | Claude + user |

---

## 3. Verified-exact recipe (the ground truth everything else builds on)

Reverse-engineered against `data/Fx_raw_cut.csv` (a matched raw-vs-processed
FAMOS export) and, later, a licensed FAMOS trial's golden corpus:

```
FL_Fx      = smo(FL_Fx1, 0.1)              # every WFT force / moment
Latacc_LPF = FiltLP(Lat_acc, 0, 0, 4, 5)   # accel only
Latacc     = smo(Latacc_LPF, 0.5)
Vehicle_Speed = smo(Speed_kmph, 0.5)
cutdata    = red(cutd, 10)                 # decimate x10
```

- **`smo`** — triangular (Bartlett) kernel, exact half-width
  `a = (round(width_s·fs) - 1) / 2`, zero-phase. *Not* a moving average and
  *not* a boxcar-of-boxcars (§4.2; full candidate search in §13).
- **`FiltLP`** — causal, single-pass Butterworth. *Not* zero-phase (§4.1).
- **`red`** — plain stride decimation, keep every Nth sample from index 0,
  no anti-alias filtering of its own (confirmed twice: Round 1 build-out,
  Round 2 Step 1 re-confirmation).

This recipe now lives once, in `famos/ops.py`; `dtt/preprocessing.py`
delegates to it (§4.9).

---

## 4. Round 1 — FAMOS-exact preprocessing migration (2026-08-09)

Full numbers for every item below are in `MIGRATION_NOTES.md`; this section
is the narrative of what went wrong and why.

### 4.1 FiltLP: `filtfilt` → `lfilter`

**Issue:** the existing code used `scipy.signal.filtfilt` (zero-phase,
forward-backward) for the lateral-acceleration low-pass. Validated against
real FAMOS output, `filtfilt` only reached r = 0.93 with ~0.66 max abs
error. A single causal `lfilter` pass matched FAMOS to 5e-6 max abs error,
r = 1.000000000.

**Root cause:** a plausible-sounding but wrong assumption — "zero-phase is
obviously better/more correct" — was never checked against real FAMOS
output. FAMOS's manual documents plain `FiltLP` as phase-shifting; zero-phase
is a *separate*, Professional-edition-only operator (`FiltLPZ`) that this
recipe never calls.

**Fix:** switch to `lfilter`. Consumers of the lateral channel now carry a
real causal lag, which is correct (matches FAMOS), not a regression — this
was explicitly audited (Check 1) to confirm nothing downstream assumed zero
lag.

**Validation quirk:** the filter's own startup transient needed 1000+
samples to settle before the comparison was meaningful; a naive skip of 300
samples still showed 3.1e-3 error, mistakenly looking like a near-miss.

### 4.2 `smo`: double-boxcar → exact triangular kernel

**Issue:** the existing `smo` was two cascaded boxcar (rectangular) filters,
reaching "only" 99.995% match / r ≈ 0.996 against FAMOS — close enough to
look done, but not exact.

**Root cause:** a boxcar-of-boxcars *is* triangular in shape, but its
half-width is `N` samples (the boxcar length), while the true FAMOS kernel
identified by least-squares fit has half-width `a = (round(width_s·fs) -
1)/2` — about 2x narrower, and specifically wrong at the tails.

**Fix:** replace with the single explicit triangular kernel, applied once.
Result: max abs error 5.6e-3 N on force channels (the CSV export's own
rounding floor), r = 1.000000000.

The full list of kernel candidates tried and rejected before landing on the
exact half-width of 49.5 is in §13; the value is not one most people would
guess, and the search is instructive.

### 4.3 `smo` edge handling: nearest-pad → window-shrink

**Issue:** near the true start/end of a channel (not just interior NaN
gaps), the kernel was edge-padded (repeating the boundary value), which is
subtly wrong.

**Fix:** window-shrink renormalization at every true channel boundary — the
kernel is renormalized by the fraction of it that actually overlapped real
data. Affects ~0.25s at each channel end; interior unaffected.

**Later superseded, correctly:** when the `famos/` library was built (§4.9),
direct measurement against a real FAMOS export found the *opposite*
convention — the FAMOS manual states edges are extended with the repeated
edge value, and measured replication beats shrink renormalization by 4.5–
5.4x. **This reverted Step 2b's decision.** It's recorded here because it's
an instructive case: the original decision was principled but *not measured
against real edge data* (the CSV used only had a mid-recording cut, no true
channel start/end); the later team had access to a real edge-containing
export and measured the actual answer. Lesson: prefer measurement over
principle whenever ground truth is available.

### 4.4 Audit: smooth-then-cut ordering

No violation found. Traced every caller of `famos_smooth`/`butterworth_lpf`/
`apply_famos_recipe` and confirmed the full channel is always conditioned
before any sub-range windowing happens (the flagged suspect,
`comparison_io.py::load_recording`, reads whole recording folders). No code
change.

### 4.5 New despike step (rail / dropout / sub-hardware-width rules)

**Issue:** the manual FAMOS workflow included spike removal that wasn't
automated at all.

**Design constraint:** WFT road load is *genuinely* spiky — a naive
amplitude threshold would remove real fatigue-relevant peaks along with
artifacts. Every rule had to be structural (what shape is physically
possible), not "how big is it."

Three rules added to `dtt/preprocessing.py::despike()`:
1. **Rail** — a run pinned at the channel's own min/max value (saturation),
   detected by value so it needs no raw-ADC-counts plumbing.
2. **Dropout** — exact zeros/nulls, or any repeated value longer than a
   configurable run length (frozen sensor).
3. **Sub-hardware-width spike** — the raw channel was hardware low-pass
   filtered at 200Hz before digitizing at 1kHz, so nothing narrower than
   `fs/(2×200)` samples can be real.

**Bug found during validation:** a bare width test on rule 3 flagged 2.8% of
a real channel — ordinary sample-to-sample chatter is *always* exactly 1
sample wide relative to its own local median, so a width test alone can't
tell it apart from a real glitch. **Fix:** added a noise-floor gate (2x the
local residual's robust scale) — far below any real spike/peak amplitude, so
it still doesn't discriminate by size, only excludes chatter indistinguishable
from measurement noise. Brought the rate down to 0.023%.

**Validated on real data** (`data/Fx_raw_cut.csv`): 0.617% flagged total
across all three rules, each spot-checked with before/after plots — genuine
dropout runs (repeated floats that can't occur from real analog noise) and
saturation correctly caught, all real load swings on either side untouched.

The measurement that ruled out a statistical amplitude threshold and forced
the structural-rules approach is in §14.

### 4.6 Checks

- **Check 1** — audited every consumer of `Latacc`/`Lat_acc` for a zero-lag
  assumption after the FiltLP causality fix. None found; the one general
  cross-correlation utility (`spectral.py::estimate_lag`) actually benefits
  (it now measures the real lag correctly).
- **Check 2** — confirmed despike actually runs on the CSV ingestion path
  (`signal_processor.py::apply_filter`), not only the raw `.raw` path.
- **Check 3** — **found a real, still-open latent bug.** Read
  `data/WFT_Fx_fr.raw` with the project's own reader and compared the
  1000–1030s window against FAMOS's own `Fx_raw_cut` column. At naive
  alignment: r = 0.854, way off. Searching a ±2000-sample lag found the true
  match at **+58 samples**: r = 0.9999999999992875, max err 0.005 N (the
  export's own rounding floor). **The byte-level decode is correct**
  (int16, calibration factor, event-record stripping); `read_raw`'s
  smoothness-heuristic data-start detector (`_find_data_start`) just lands
  58 samples early relative to FAMOS's own sample 0. Left unfixed —
  explicitly out of scope for that round, flagged for whoever owns
  `imc_reader.py` — but it means **every channel read via `read_raw` is
  silently misaligned by 58ms** unless corrected. Not yet addressed as of
  this document (§10).

### 4.7 Result

The recipe now matches FAMOS to the CSV export's own rounding floor on every
operator. This became the baseline the rest of the project builds on.

---

## 5. Round 2 — histogram restyle, spike rule, resampling, stop removal (2026-08-25)

Full numbers in `MIGRATION_NOTES.md`'s "Round 2" section. Four tasks, done
in increasing order of risk.

### 5.1 Step 1 — resampling confirmation

No change needed. `famos_red` re-confirmed exact, already runs last in the
per-channel order, no caller uses the interpolating `resample_mode`.

### 5.2 Step 2 — histogram restyle

**Goal:** banded bars (dark centre → light edges) with a KDE curve overlay,
matching a reference FAMOS-style image, but adapted for real (often skewed)
WFT data rather than forcing a symmetric bell everywhere.

**Built:** `dtt/analysis/plot_style.py` — `detect_shape` (symmetric vs
one-sided, by boundary proximity or skew), `banded_colors` (sigma-bands for
symmetric, quantile-bands for one-sided), `draw_histogram` (orchestrates
both plus a scaled KDE curve). Extended `dtt/analysis/auc.py::kde_curve`
with optional per-sample weights and boundary reflection.

**Bug found during validation:** the first cut always reflected the
one-sided KDE curve around the plotted axis's left edge, even when the
"one-sided" classification came from mere skew rather than a real physical
boundary. For a skewed-but-unbounded force channel, that axis edge is just
wherever percentile-trimming happened to start the plot — reflecting around
it invented density that wasn't there. **Fix:** only reflect when the
caller passes an explicit, known boundary (rainflow ranges do — 0 is real);
otherwise use a plain, unreflected KDE, which still respects the actual
skewed shape without a false correction.

**Validated** on real bimodal Fx data and rainflow range distributions;
both archetypes render correctly (see the migration notes for the actual
before/after images).

### 5.3 Step 3 — the manual's "20% within 1 second" spike rule

This is the round's central cautionary story.

**As planned:** add a 4th despike rule — flag a sample/short run that
deviates from its local trend by >20% and returns within ~1 second,
alongside the existing rail/dropout/sub-width rules, on the **raw** channel
(matching the existing despike architecture).

**What happened when built exactly as planned:** validated against
`data/Fx_raw_cut.csv` and it flagged **25–90%** of the channel depending on
window/threshold tuning — not a tuning problem. Measured directly: 25.7% of
raw 1kHz samples already deviate more than 20% from even a 21ms local
median, **before any width test at all**. Raw, unsmoothed 1kHz WFT data is
simply that noisy sample-to-sample; the rule was catching ordinary raw
jitter, not isolated transients.

**This was surfaced to the user rather than silently patched** — it's a
genuine design/interpretation problem, not a threshold to nudge. The
resolution (explicit user direction, combining two of three offered
options): move the rule to run **post-smooth/post-decimate** (the manual's
"20% within 1 second" describes what an operator saw on a *conditioned*
FAMOS trace, not literal raw samples — the raw-stage rules stay raw because
they catch genuine digitizer glitches, which is a different job) **and**
make the effective threshold the *larger* of 20%-of-local-level and a
noise-floor multiple, so it can never sit below the channel's own noise.

**Second bug found during re-validation:** even on the correctly-placed
post-smooth signal, "narrow" was defined as "run length < the full 1-second
window" — which let a genuine ~0.7s real braking/cornering swing (smooth,
monotonic, verified on the actual recording) get flagged, exactly the
"eating real peaks" failure the brief warned about. **Fix:** decoupled the
width cap from the trend window — `max_spike_frac` (default 0.4) caps the
flagged run at 40% of the window, not the full window.

**Final validation:** 0% flagged on clean real data; correctly catches an
injected narrow (0.3s) synthetic transient; correctly ignores an injected
wide (0.7s) synthetic swing of the same peak magnitude.

### 5.4 Step 4 — stop / stationary-period removal

**Goal:** exclude parked/stopped stretches from analysis so they don't
distort statistics and rainflow — wired into the pipeline as an opt-in
stage, off by default.

**User amendments before starting** (both honored): (1) never overwrite
`processed_data.csv`/`raw_data.csv` — only a *derived* series feeds
downstream analysis, so the canonical artifact stays reversible and
complete; (2) validate with an actual rainflow before/after cycle-count and
Miner-damage comparison, plus a real seam-level plot, not just an asserted
number.

**Seam handling** (the hard part): concatenating "moving" segments after
cutting out stops leaves a level step at every join, which rainflow can
misread as a load cycle — corrupting the exact fatigue numbers this project
exists to compute correctly. Two defences: (1) boundary nudge — extend each
cut boundary up to 1s to the nearest low-load moment across all force
channels jointly; (2) seam blend — a short (0.2s) linear blend across
whatever step survives the nudge, as backstop.

**Validation surprise, resolved productively:** first validation pass (stop
removal alone, no despike/transient cleanup first) found a rainflow damage
ratio (after/before) of **0.38** — alarming, since removing genuine dead
time should barely move it. Traced to a single ~900 daN spike sitting
*inside* the detected stop interval on the real test recording — not a
stop-removal artifact, a genuine uncleaned sensor glitch that the transient
rule (§5.3) is specifically built to catch. Re-ran with the actual target
pipeline order (despike + transient rule enabled first, matching production
configuration) and the spike disappeared before stop removal ever saw it.
**Final damage ratio: 1.000000** — confirms the seam handling injects no
fake cycles. This is a good example of validating a feature against the
*real* pipeline it will run inside, not in isolation.

---

## 6. The merge conflict — two independent stop-removal designs

While Round 2 Step 4 was being built, a teammate (working with a separate
Claude Opus 5 session, commits authored as `nik123-py`) was independently
building their **own** stop-removal wiring, pushed directly to the shared
`nikhilesh` branch. Neither side knew about the other's version until the
push step.

### 6.1 What each side built

| | this round's version | teammate's version |
|---|---|---|
| architecture | `remove_stops_frame` did mask + cut + seam handling in one function | split into `stop_mask_frame` (decide) + `apply_stop_mask` (cut), so one decision can cut more than one frame identically |
| seam handling | boundary nudge + linear blend | none — hard concatenation |
| pipeline placement | **late**: only a derived series feeds stats/rainflow; `processed_data.csv`/`raw_data.csv` stay the full recording | **early**: cuts `df` and `raw_frame` directly, before validation/stage 4, so the canonical CSVs themselves reflect the cut |
| rationale | explicit user instruction: keep the canonical artifact reversible | measured that a parked vehicle's static load distorts G-severity too (FR Gx 0.045→0.116 from 5% standstill), not just rainflow |
| exposure | library + config only | CLI flag (`--remove-stops`), GUI checkbox, `_warn_if_stationary` log warning when stops are kept |

### 6.2 Literal git conflicts

`dtt/preprocessing.py` and `dtt/pipeline.py` had real, unresolvable-by-git
textual conflicts (two different function bodies for the same functions).
`dtt/config.py` auto-merged but ended up with **duplicate field
declarations** (`remove_stops`, `stop_min_s` defined twice in the same
dataclass — Python silently uses the last one, which is exactly the kind of
error that's easy to miss).

### 6.3 Resolution

**`preprocessing.py`** — not a pick-one-side situation, the two designs were
complementary: kept the teammate's mask/apply split (genuinely useful —
lets one decision cut a conditioned frame *and* its raw twin identically)
and folded this round's seam-aware nudge into `stop_mask_frame` and the
blend into `apply_stop_mask`, so **every** caller gets seam handling
automatically, including the teammate's own code, without them changing
anything. `remove_stops_frame` became a thin single-frame convenience
wrapper over both.

**`config.py`** — de-duplicated; kept the fuller field set
(`stop_speed_kph`/`stop_seam_search_s`/`stop_seam_blend_s` alongside
`stop_min_s`, defaulting to 3.0s).

**`pipeline.py`** — this was a genuine **product decision**, not a
technical merge, and was explicitly escalated to the user rather than
picked unilaterally (three options were laid out: take the early-cut
design entirely, keep the late-derived-series design exactly as approved,
or broaden the late design to cover more analysis stages). **User's
decision: broaden.** Final behavior: `processed_data.csv`/`raw_data.csv`
stay the full, untouched recording (the explicit reversibility requirement
holds); *every* analysis stage — statistics, severity, histograms,
heatmaps, boxplots, rainflow — now reads the moving-only derived series
when stop removal is on, so the teammate's G-severity distortion finding
is fixed too, without changing what the canonical CSVs mean. The
teammate's early hard-cut stage (`_drop_stationary`) was removed; their
`_warn_if_stationary` (logs when stops are present but not being removed)
was kept.

**Re-validated after reconciling:** the exact same rainflow damage-ratio
test from §5.4 re-run against the merged code — still 1.000000, confirming
the reconciliation didn't disturb the seam-handling correctness.

### 6.4 A process note on the merge itself

The remote moved twice while this was in progress — new commits landed on
`origin/nikhilesh` mid-review, once after the first merge had already
started. The in-progress (partially resolved) merge was aborted cleanly
and redone from scratch against the newer tip, after first confirming none
of the newest commits touched the three conflicted files. This is the safe
pattern for "the remote moved while I was resolving a conflict": abort,
re-check scope, redo — not attempt to reconcile a stale merge with a moving
target.

---

## 7. Data governance — the leak/credential audit

Before pushing, an explicit audit was run (user request: confirm no dataset
or credentials go up with the code).

**Clean:** every new commit in both rounds touched only `.py`/`.md` files.
`git grep` across the whole tree for credential-shaped patterns (API keys,
passwords, private-key headers) found nothing. No `.env`/`.pem`/`.key`
files tracked anywhere.

**Pre-existing exposure found, not introduced by this work:** two files
already on `origin/nikhilesh` from an earlier, separate push:
- `famos_repro.png` — an actual rendered plot of a real `Latacc` channel
  from a real test recording (raw vs. processed, residuals, PSD, ~11,000s).
  Proprietary measurement data made visible as an image, not raw data
  itself, but genuine test data nonetheless.
- `filter_init_diff_report.json` — lists 29 real internal study
  names/timestamps with per-study diagnostic values. Internal test-session
  metadata.

Both were flagged to the user with the specific finding and left in place
at the user's explicit direction (git history would retain them either
way; removing them now only stops *future* exposure). **Not yet
addressed** — see §10.

---

## 8. Investigation: "the preprocess graph got spikier after the merge"

The user reported the Preprocess screen's raw-vs-sanitized view looking
worse (more jagged) than before, suspecting the merge had regressed the
smoothing. Investigated rather than assumed:

1. Diffed the actual smoothing functions (`famos_smooth`, `apply_pipeline`,
   `famos_recipe`) between pre- and post-merge commits — byte-identical.
   The merge touched only despike/stop-removal code, which this GUI preview
   path doesn't call.
2. Pulled the actual `processed_data.csv` values for the study/channel in
   the screenshot: standard deviation genuinely jumps from 0.26 to 1.16
   daN between the two halves of the visible window — real data, not a
   rendering artifact.
3. Checked `Vehicle_Speed` for the same window: **0 throughout.** The
   entire 30-second view is a parked/stationary stretch; the "spikes" are
   real sensor micro-noise during a stop, visually exaggerated by the
   plot's tight auto-scaled Y-axis (~6 daN total range).

**Conclusion:** not a regression, not a bug. The FAMOS-verified 0.1s
smoothing width is doing exactly what it's supposed to and shouldn't be
widened (that would deviate from the validated spec). This is precisely
the kind of stretch stop removal (§5.4, §6) is meant to exclude from
*analysis* — though by design it will still show up on the Preprocess
screen, which always displays the true, complete measured data.

---

## 9. Round 3 — the missing raw reference for moment channels (2026-09-02)

An industry mentor reviewed the platform and reported three things: filtering
was "not being done/displayed properly", files referred to as `f*_m*` showed
"no preprocessed data", and the system "did not behave like FAMOS" when the
preprocessing was checked.

The brief that came with it assumed the filtering mathematics was suspect. It
was not. This round is a good example of why the instruction to *trace the data
flow before touching a validated operator* is worth following literally.

### 9.1 What `f*_m*` actually meant

Nothing in the codebase used that string. Resolved by listing the dataset:
`f*_m*` matches `FR_Mx_2.raw`, `FR_My_2.raw`, `FR_Mz_2.raw` in
`raw data in .dat format/2006-08-25 09-10-46 (1)/` — **WFT moment channels**,
named `<position>_<component>_<WFT serial>`. On the other dataset the same
channels are `FL_Mx` … `RR_Mz`.

Checked first, before assuming a mapping bug: `parse_channel("FR_Mx_2")`
returns `(A1R, Mx)` and `is_wft_channel` is `True`. Channel mapping was fine.
So was the recipe — `famos_recipe` gives moments `smo(0.1)`, identical to
forces — and so was the output: `FR_Mx_2` sits in `processed_data.csv` with
554,785 samples and no NaNs, measurably smoothed.

**The preprocessing had always run on moments. It was invisible, not absent.**

### 9.2 Root cause — the raw snapshot was force-only

Two places captured the unconditioned "before" copy, and both filtered it to
forces:

- `imc_reader.py` kept `is_wft_channel(c) and parse_channel(c)[1] in
  FORCE_COMPONENTS` — i.e. `Fx/Fy/Fz` only.
- `pipeline.py::_force_frame` kept `rc.mandatory_channels`, which
  `run_channels.py` builds under the same `if comp in FORCE_COMPONENTS` filter.

Measured on a real run: **31 of 37 channels had no raw counterpart**, moments
among them. `raw_data.csv` held 6 columns where `processed_data.csv` held 37.

The Preprocess screen reads the two files as a pair. With no raw column,
`_read_channel` returns `(None, None)`, `have_raw` goes false, and `_update`
falls into a branch that draws a single trace in `_RAW_COLOR` — **the raw
colour** — labelled "study data". A fully smoothed moment was therefore
rendered identically to an unprocessed one, while forces beside it showed a
proper blue/red pair. That is precisely the report: forces look preprocessed,
`f*_m*` does not.

**Fix:** ask the recipe itself. `preprocessing.conditions_channel(name)` returns
true when `famos_recipe` actually changes a channel (`smo` and/or `FiltLP`), and
both capture sites now key off it. Moments, `Latacc`, `Longacc` and
`Vehicle_Speed` all gain a before/after pair; genuine passthrough channels
(GPS, yaw, angles) correctly do not, because there is nothing to compare.
`raw_data.csv` went from 6 to 15 channels on the 2-wheel set, 12 to 26 on the
4-wheel set.

### 9.3 The bug the fix introduced, caught in validation

Widening that frame broke an assumption somewhere else — the kind of failure
that only shows up if validation checks *magnitudes* and not merely presence.

`_write_raw_csv` divided **every** column by the force scale divisor (the
N→daN conversion times any decade correction). That was correct only while the
frame contained forces alone. The first post-fix run gave, for `FR_Mx_2`:

```
raw  std =   8.24        proc std = 812.04        ← a factor of 100
```

Both upstream corrections (`_apply_n_to_dan`, `normalise_force_units`) touch
`Fx/Fy/Fz` and nothing else, by design — the loader's own docstring says so:
moments carry their own imc `CR` factor and their own unit (Nm). The divisor is
now applied to force channels only.

Left unnoticed this would have been **worse than the bug it replaced**: a
moment drawn 100x low against its correctly scaled processed trace is a
plausible-looking wrong answer, where a missing channel at least announces
itself. Pinned by `test_raw_csv_scales_forces_only`.

### 9.4 Making the conditioning observable

The mentor's underlying complaint was that preprocessing could not be *seen*,
so three things were added on top of the root-cause fix:

- **`famos_stages()`** returns every intermediate of the recipe for one channel
  (raw → despike → FiltLP → smo → red) as labelled `PreprocessStage` records.
  It calls the same operators in the same order as `apply_famos_recipe` rather
  than reimplementing them, and a test pins its final stage to that function's
  output at `atol=0` — a debugging view free to disagree with the data is worse
  than no view.
- **`log_preprocess_trace()`** emits a per-channel `[PREPROCESS]` block (sample
  counts, rate, detected type, which operators ran, `VALID`/`EMPTY`/all-NaN
  verdict), surfaced in the GUI behind a "Preprocessing trace…" button. The
  pipeline log now also names every channel and its operations instead of only
  a "15/38 conditioned" count.
- **Naming.** The conditioned trace is labelled *"Final FAMOS-Equivalent
  Preprocessed Output"*, not "sanitized" — the old word invited the reading that
  the real preprocessing lived somewhere else, unshown. A passthrough channel
  now says it is passthrough rather than looking unprocessed.

**Stage exactness.** The first cut of the stage overlay recomputed `smo`/`FiltLP`
from the stored raw, which is decimated — so it could not reproduce a pipeline
that runs them at 1 kHz *before* `red()`. That was replaced rather than
documented: on a conditioned study every stage drawn is now read from the
study's own files. Ingestion stores the FiltLP intermediate (`Latacc_LPF`,
one extra column, and a column a genuine FAMOS CSV export carries anyway), and
no separate smo curve is drawn at all — the recipe ends `smo → red`, so
`red(smo(x))` **is** the stored final output. The legend says that instead of
drawing a second, wrong curve.

### 9.5 An empty log made a conditioned study look unconditioned

`_setup_logging` used `logging.basicConfig`, which is a silent no-op once the
root logger has any handler. Called in-process by a host that had already
configured logging, the run wrote an **empty** `pipeline.log` — and
`PreprocessPage._study_is_conditioned` read that file to decide whether the
recipe had run, so the screen reported a fully conditioned study as
unconditioned and offered to smooth it a second time.

Two fixes: `_setup_logging` now attaches (and replaces) its own handlers,
leaving a host's alone and not double-printing to a console it already owns;
and `_study_is_conditioned` reads `famos_applied` from `run_provenance.json`
first, so the answer no longer depends on logging being wired up at all. The
log remains the fallback for studies written before provenance existed.

### 9.6 Turning the expired FAMOS trial into a standing test

`MODULE1_STATUS.md` recorded the licensed-FAMOS validation as done, but nothing
re-checked it — "our `smo` matches FAMOS" had degraded from a measurement into
a memory of one, and an edit to the kernel would have been caught by nothing.

The capture is still on disk, so it is now a test.
`golden_corpus/famos_out/FR_Fx_2.csv` carries, for six real force channels,
the raw signal beside FAMOS's own `_smo`, `_red` and `_rawred` results.
Scored over 200k samples:

| comparison | result |
|---|---|
| `smo(0.1)` vs FAMOS `_smo` | max err < 1 N on ~50,000 N signals, r > 0.99999999 |
| `smo → red(10)` vs FAMOS `_red` | same |
| `red(10)` alone vs FAMOS `_rawred` | **bit-exact, max err = 0** |

The last row has no rounding excuse available — both sides are copies of stored
samples — so it proves the decimation phase (start at index 0, stride 10) is
identical to FAMOS's. The tests skip cleanly when the 2 GB gitignored CSV is
absent, because a skip is honest and a pass on missing evidence is not.

The corpus covers *forces*; moments are not separately captured and do not need
to be, because `test_moment_and_force_are_conditioned_the_same_way` pins them to
byte-identical treatment. This narrows, but does not close, the §10 gap.

**Why the CSV and not the binary.** `golden_corpus/famos_out/*.dat` — and the
`Raw data_no sanitisation/` recordings — are the newer **imc3** container
(`|imc3,1;|CB1…`), which `read_famos_all` does not parse; it handles the classic
comma-keyed `|CF,2,1,1;` layout. The folder reader has its own imc3 path, which
is why ingestion works while `read_famos` returns nothing on the same files.
Scoring therefore went through the ASCII export, whose ~6-significant-figure
rounding (~0.05 N at these magnitudes) is the error floor above. See §12.

### 9.7 Multi-session joining — already built, now verified

A request arrived to "put data from 4220 to X time and have it map as well" —
i.e. join a second recording onto the end of the first. This already existed
(`load_raw_sessions` → `sessions.py::concat_sessions`, GUI "+ Session" button,
CLI `--raw-folders`) but had never been demonstrated, so it was verified rather
than rebuilt.

Splitting a real 4219 s recording in half, giving leg 2 its own clock starting
at zero as a recorder writes it, and rejoining: leg 2 lands at 2109.42 s, every
clock gap is exactly 0.01 s, and values are bit-identical to the unsplit
original. Joining the two real session folders end to end put the seam at
4219 s over a 9771.5 s record, and the joined `raw_data.csv` carries the moment
channels — so the before/after pair survives a stitch.

The join is deliberately **end to end**: a real-world gap between legs is not
represented, and the seam times in the report are what let a caller find where
one leg ended. That is the right model for "one route driven in several runs"
and the wrong one for absolute wall-clock placement, which does not exist.

### 9.8 Result

`raw_data.csv` now carries a "before" for every channel the recipe conditions,
the Preprocess screen names what it is showing, and the FAMOS operators are
re-scored on every test run. 130 tests pass. No change to the validated
mathematics: `famos/` is untouched, and `smo`, `red` and `FiltLP` are called,
not reimplemented.

Pushed as `ae4312f`, rebased onto a teammate's `c5fe316`.

**Carries a migration cost.** Conditioning is destructive and the raw was never
stored, so a study processed before this change cannot be repaired in place —
it must be re-ingested to gain the pair. The packaged `.exe` likewise had to be
rebuilt; the mentor's report was reproduced partly *because* the binary in use
predated the fix, which is worth remembering the next time a fix "doesn't
work".

---

## 10. Open items / known gaps

- **Raw reader 58-sample offset** (§4.6, Check 3) — still unfixed. A real,
  reproducible latent bug in `imc_reader.py::read_raw`'s data-start
  detection. Needs its own investigation (why 58, constant across
  files/channels?, is it `_find_data_start`'s window search or the
  leading-trim loop?). A regression check already exists: `WFT_Fx_fr.raw`
  sample `[1000000+58 : 1030001+58]` vs. `Fx_raw_cut.csv`. The structural
  imc3 format that lets a reader use explicit chunk lengths instead of a
  data-start heuristic (the deterministic fix at the source) is documented
  in §12.
- **`--stop-min-s` CLI flag is parsed but never wired into `run()`'s
  kwargs** — a small pre-existing gap from the teammate's own commit,
  noticed during the merge reconciliation, not yet fixed. Both the CLI
  flag and the GUI spinbox currently have no effect; `stop_min_s` only
  ever takes its `RunConfig` default (3.0s).
- **`famos_repro.png` / `filter_init_diff_report.json`** (§7) — flagged,
  left in place at user's direction. Revisit if/when the repo's visibility
  changes.
- **Moment channels (`Mx/My/Mz`) and `Latacc` on real data are inferred,
  not measured** — narrowed in Round 3 but still open. The golden corpus
  covers *forces*, and a test now pins moments to byte-identical treatment
  (`test_moment_and_force_are_conditioned_the_same_way`), so the operator
  they run is FAMOS-scored even though the channel is not. Capturing a real
  matched FAMOS moment pair still needs a licence; the trial is spent.
- **imc3 files are only readable through the folder path** — `read_famos`
  and `read_famos_all` parse the classic comma-keyed `|CF,2,1,1;` layout and
  return nothing for the newer `|imc3,1;` container, which is what both
  `golden_corpus/famos_out/*.dat` and the `Raw data_no sanitisation/`
  recordings are. Ingestion works because `_read_channels` has a separate
  imc3 path. The consequence is that golden-corpus scoring goes through the
  ASCII CSV export and inherits its ~0.05 N rounding floor instead of the
  binary's full precision (§9.6). Structural format notes in §12.
- **Absolute wall-clock placement of joined sessions does not exist** —
  `concat_sessions` joins end to end, so a real gap between legs is
  compressed to zero and only the seam time records where the join fell
  (§9.7).
- **Existing studies predate the unit-decade fix** — anything processed
  before that correction holds forces a decade high; re-run rather than
  compare.
- **Studies processed before Round 3 have no raw reference for moments** —
  conditioning is destructive and the "before" was never written, so these
  cannot be repaired in place; re-ingest to get the before/after pair
  (§9.8). The same applies to any packaged `.exe` built before 2026-09-02.

---

## 11. What's validated and where to check it

Full detail in `MODULE1_STATUS.md` and `golden_corpus/VALIDATION_GUIDE.md`.
Headline: the FAMOS reader is byte-exact (0.000e+00 vs. a licensed FAMOS on
real data), the force chain (`smo → red`) matches at r = 1.000000000 across
5.5M real samples, and the full operator corpus (`FiltLP`/`smo`/`red`,
synthetic + real) passes 33/33 at or below the CSV export's own rounding
floor. Despike, the transient-spike rule, and stop removal have no FAMOS
ground truth to check against (they're not FAMOS operations) and were
instead validated empirically on real recordings as described in §4.5,
§5.3, §5.4 above.

Since Round 3 those FAMOS scores are no longer a one-off record: 15 tests
re-score `smo` and `red` against the captured licensed-FAMOS output on every
run, `red` bit-exactly (§9.6). They skip rather than fail when the gitignored
2 GB corpus CSV is absent.

---

## 12. Appendix: reverse-engineering the imc3 raw binary format

This is the investigative detail behind the raw reader and the §4.6 Check 3 /
§10 offset bug. It was worked out during Round 1 while establishing ground
truth, and it is the structural reference a proper fix to `imc_reader.py`
should build on.

**The obstacle.** Reverse-engineering the recipe needed the raw channel values
in Python. The uploaded `.raw` files were imc's newer `imc3` format (magic
header `|imc3,1;`), not the older `|CF` FAMOS format. The obvious tool, the
`imctermite` pip library, installed and built but failed immediately with
"invalid block or corrupt buffer at byte 11", because it only understands the
older `|CF`-keyed layout. No off-the-shelf reader handled imc3, so the format
was cracked by hand.

**The container layout.** After the `|imc3,1;` magic, the file is a sequence
of keyed blocks. Metadata blocks come first (`|CB1`, `|CL1`, `|CO1`, `|CA1`,
`|CC1`, `|CM1`, `|CD1`, `|CN1`, `|CP1`: origin, channel name, x-axis delta,
calibration, properties), then the sample data, then a short trailer (`|CS1`,
`|CJ1`, `|CE1`). The `|CP1` properties block is where acquisition metadata
lives, including the hardware anti-alias filter cutoff (`eFilterCutoff1 = 200`)
that later justified the despike sub-width rule (§4.5, §14).

**The data is chunked, not contiguous.** This was the key finding. The samples
are not stored as one block. They are split across roughly 40,000 small
streaming chunks, each introduced by an `|RC5` marker, interspersed with
occasional `|RR1`/`|RT1` segment markers. Each `|RC5` chunk is: the 4-byte key
`|RC5`, then three little-endian int32 (the first two are always 1, the third
is the chunk's payload length in bytes), then that many bytes of little-endian
int16 samples.

**The bug that cost time.** The first parser read the length from the second
int32 (offset +8) instead of the third (offset +12), and read the payload from
+12 instead of +16. That desynced immediately and recovered only about 20,000
samples out of millions. Fixing the offset (length at +12, payload at +16,
advance by 16 + length) recovered the full channel: 4,364,839 samples, about
4365 seconds at 1 kHz, matching the recording length.

**Scaling.** The int16 counts are converted to physical units with a
per-channel factor read from the `|CM1` block (a little-endian double), for
example 0.01 for the acceleration channels and 1.373291... for the WFT force
channel.

**Validation.** The reconstructed physical values were checked against FAMOS's
own physical `Fx_raw_cut` column for the same 1000–1030 s window. Once aligned
they matched to 0.005 N, the CSV export's rounding floor, confirming the
structural decode (chunk framing, int16 parsing, calibration factor) is
correct.

**Relation to the project's reader and the open offset bug.** The platform's
own `imc_reader.py::read_raw` does not parse `|RC5` as chunk framing. It treats
the 16-byte `|RC5` header as an event record to strip and locates the data
start with a smoothness heuristic (`_find_data_start`) rather than by reading
the explicit length fields. That heuristic lands 58 samples early relative to
FAMOS's sample 0 (the §4.6 Check 3 finding), and because it always returns
something plausible it never fails loudly on a misread. The structural format
above is the deterministic alternative: reading the explicit `|RC5` length
fields and starting at the first chunk removes the need for a data-start
heuristic and would close the 58-sample offset at its source. This remains an
open item (§10); the structural understanding is recorded here as the reference
for that fix.

---

## 13. Appendix: pinning the exact `smo` kernel, every candidate tried

Expands §3 and §4.2. §4.2 states the conclusion (triangular, half-width
`(round(width_s·fs) - 1)/2`); this is the full search that got there, kept
because the dead-ends are instructive and because the final half-width is not
the value most people would guess.

All tests were against `data/Fx_raw_cut.csv`, comparing a candidate kernel's
output to FAMOS's `Fx_smo_cut` (0.1 s, force) and `Long_smo_cut` (0.5 s, accel)
columns, skipping the cut's first and last few hundred samples so window edges
did not pollute the comparison. The CSV values carry about six significant
figures, so the best achievable ("rounding floor") error is around 5e-3 N.

**Step 1, rejecting the boxcar.** A plain rectangular moving average was tried
across window sizes 98–102, both centered and trailing. The best reached
r ≈ 0.996 with about 200 N max error. Centered beat trailing by a wide margin,
which established one fact that held for everything after: the kernel is
symmetric and zero-phase. But a boxcar was decisively wrong.

**Step 2, identifying the shape directly.** Rather than keep guessing shapes,
the impulse response was solved for. Treating the smooth as an unknown
symmetric FIR filter, a least-squares fit of raw to smoothed over the signal
interior recovered the actual kernel: support of exactly ±49 samples (99 taps),
rising linearly to a peak at the center and falling linearly to the edges,
summing to 1. Its RMS distance to an ideal triangle was 2e-6, versus 0.003 to a
boxcar. The kernel was triangular, not rectangular.

**Step 3, the ideal triangle nearly but not quite.** A linear triangle of
half-width 50 (taps proportional to 50 − |k|, peak weight 0.02) reproduced the
0.5 s accel channel to 6e-4, essentially exact. But on the 0.1 s force channel
it left 4.25 N of max error, mean 0.25 N, r = 0.99999851, with the residual
concentrated at the single sharpest force spike in the window. Close, but not
at the rounding floor.

**Step 4, is 4.25 N real or just input rounding?** The least-squares kernel
from Step 2 reproduced the data everywhere to about 0.005 N (the rounding
floor), including at that spike, while the analytic half-width-50 triangle gave
4.25 N there. So the 4.25 N was a real, if tiny, kernel-shape error, not input
rounding: the true kernel differs from a pure half-width-50 triangle by up to
2e-4 per tap, being about 1% more peaked at the center and slightly lower at
the edges (measured peak 0.020201 versus the triangle's 0.02).

**Step 5, the candidates that did not work.** Searching for that
slightly-more-peaked triangle, the following were all tested against the CSV
and none reached the rounding floor:
- Boxcar convolutions (50×50, 50×51, 51×50, 51×51, 49×50, 49×51): all ~4.25 N
  or worse.
- `numpy.bartlett` at lengths 99, 100, 101, and 101 trimmed: no better.
- A triangle over ±50 (length 101): worse.
- A triangle raised to a power p: p = 1.05 was marginally better (3.72 N),
  p = 1.1 / 1.15 / 1.2 all worse. No clean fit.
- Gaussian at sigma 18 to 24: 10 to 70 N.
- Hann (99, 101), Welch, and a parabolic window: all worse.
- Three-boxcar convolutions (quadratic B-splines, e.g. 33×33×34): 80 to 90 N,
  peaked the wrong way.
- `scipy.signal.windows.triang(100)` (even length): 8 N; `triang(99)` is just
  the half-width-50 case.

**Step 6, the breakthrough.** Instead of guessing shapes, the triangle's
half-width was swept as a continuous parameter in `h[k] = max(0, 1 − |k|/a)`.
The error collapsed sharply at `a = 49.5`, giving 5.6e-3 N, the rounding floor,
with a peak weight of 0.020200 that matched the least-squares-identified
0.020201 exactly. Not 50, not 49, but 49.5.

**Step 7, the closed form.** 49.5 = (100 − 1)/2, and 100 = width_s × fs. So the
half-width is `a = (round(width_s·fs) − 1)/2`: the triangle reaches zero half a
sample past the last tap. Confirmed on the 0.5 s accel channel, where
`a = 249.5` gave 5e-6.

**Final kernel.** Symmetric triangular weights `h[k] = max(0, 1 − |k|/a)` with
`a = (round(width_s·fs) − 1)/2`, normalized to unit sum, applied centered
(zero-phase). Widths 0.1 s for force and moment channels, 0.5 s for
acceleration and speed. This is the kernel now in `famos/ops.py`, matching
FAMOS to the CSV export's rounding floor on every channel.

---

## 14. Appendix: why despike uses physical rules, not a statistical threshold

Expands §4.5. §4.5 states the design constraint (WFT road load is genuinely
spiky, so amplitude thresholds eat real peaks); this records the measurement
that proved it and set the whole despike direction.

Before committing to the structural rules, a standard statistical despiker was
tested on the full raw force channel (4.36M samples) to see whether it could
work: a Hampel filter, which flags any sample more than k times the local
scaled MAD (median absolute deviation) from a sliding median. The result:

| window | k = 3 | k = 5 | k = 6 |
|---|---|---|---|
| ±10 samples | 1.91% | 1.49% | 1.47% |

The telling part is not the absolute numbers but that loosening the threshold
from k = 3 to k = 6 barely moved the flagged fraction, from 1.9% down to only
1.47%. If these were genuine sensor glitches, the count would collapse as the
threshold loosened. It did not, because the thing being flagged is mostly real
signal: WFT road load is spiky by physics, sharp force peaks as the tyre hits
events, and those peaks are exactly what the fatigue analysis exists to count.
Any statistical amplitude threshold set tight enough to catch glitches also
deletes a percent or two of real load peaks, invisibly, which would quietly
lower the rainflow damage numbers.

That measurement is why the despike step (§4.5) uses only structural rules that
key on what is physically impossible rather than what is merely large: a run
pinned at the channel's own extreme (rail), a frozen or zero run that
continuous analog noise cannot produce (dropout), and a spike too narrow to
have survived the sensor's own 200 Hz hardware anti-alias filter (sub-width).
The hardware cutoff, read from the `|CP1` metadata during the format work
(§12), is what makes the sub-width rule defensible: at 1 kHz sampling, nothing
narrower than about 2.5 samples can be real. The one place a loose statistical
net survives is as an optional, off-by-default backstop at high k (~6), reused
from the existing Hampel machinery, for gross garbage the structural rules
miss, never as the primary rule.

A related decision recorded for completeness: no machine learning is used for
threshold selection anywhere in despike. The MAD-based net already adapts its
threshold to each channel's own noise, which is the only adaptivity needed. An
ML approach would be non-deterministic, impossible to validate against FAMOS
the way the rest of the recipe is, hard to justify in a durability report, and
has no labelled training data of past manual edits to learn from.
