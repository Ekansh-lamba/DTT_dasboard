# Migration Notes: FAMOS-exact preprocessing

Tracks the switch of `dtt/preprocessing.py` from the previous "close but not
exact" FAMOS approximation to the verified-exact recipe (reverse engineered
against `data/Fx_raw_cut.csv`, a matched raw-vs-processed FAMOS export). See
`famos_preprocess.py` for the reference implementation and validation numbers
this migration is targeting.

---

## Step 1 — FiltLP: filtfilt -> lfilter (2026-08-09)

**File:** `dtt/preprocessing.py::butterworth_lpf` (imported by `signal_processor.py`
and used by `imc_reader.py::_assemble` via `apply_famos_recipe`). Also updated
the module docstring and the GUI tooltip in `gui/pages/preprocess_page.py`
that both asserted FiltLP was zero-phase.

**Before:** `scipy.signal.filtfilt` — forward-backward, zero-phase Butterworth.
**After:** `scipy.signal.lfilter` — causal, single-pass Butterworth. Signature
unchanged (`butterworth_lpf(x, cutoff, order, fs)`).

**Why:** Reverse-engineered against a matched raw-vs-processed FAMOS export
(`data/Fx_raw_cut.csv`, columns `Lat_raw_cut` -> `Lat_lpf_cut`, order 4, cutoff
5 Hz). `filtfilt` only reached r = 0.93 with ~0.66 max abs error against the
real FAMOS `Latacc_LPF` column; `lfilter` matches it almost exactly. FAMOS's
own FiltLP is documented as a causal filter (the "0, 0" leading args in the
macro select causal mode; zero-phase is a separate Professional-edition
`FiltLPZ` operator FAMOS did not use here).

**Validation:** `lfilter` output vs `Lat_lpf_cut`, skipping the first/last N
samples of the cut to let the filter's startup transient settle (N=300 was
too tight — the transient does not fully decay until ~1000 samples in; the
reference script `famos_preprocess.py` uses N=3000 for this channel):

| skip | max abs err | mean abs err | r |
|---|---|---|---|
| 300  | 3.120e-03 | 6.048e-06 | 0.999999973 |
| 1000 | 5.014e-06 | 2.604e-07 | 1.000000000 |
| 3000 | 5.014e-06 | 2.695e-07 | 1.000000000 |

Matches the target (max abs error ~5e-6, r = 1.000000000).

**Consumers of the lateral channel now see a real causal lag** (this is
correct — it matches FAMOS, not a regression).

**Check 1 — downstream lag audit:** grepped every consumer of `Latacc` /
`Lat_acc` (`preprocessing.py`, `signal_processor.py`, `imc_reader.py`,
`famos_validation.py`, GUI pages, `generate_sample_data.py`). None combine
`Latacc` with another channel in a way that assumes zero lag (no load-transfer
or cross-channel correlation code touches it) — it is only ever smoothed,
displayed, or independently plotted/exported. The one general-purpose
cross-correlation utility that touches channel alignment, `dtt/spectral.py::
estimate_lag` (used by `gui/pages/signals_page.py` for manual channel
alignment in the UI), is unaffected: it now correctly measures the real lag
`Latacc` carries, which is the FAMOS-accurate behaviour. No fix needed.

---

## Step 2 — smo: double-boxcar -> exact triangular kernel (2026-08-09)

**File:** `dtt/preprocessing.py::famos_smooth_window` and `famos_smooth`.

**Before:** two cascaded centred box-cars of `N = round(width_s*fs/2)`
samples each (`uniform_filter1d` applied twice), giving an *approximate*
triangle of half-width `N`.
**After:** a single explicit triangular (Bartlett) kernel of half-width
`a = (round(width_s*fs) - 1) / 2`, applied once via `scipy.ndimage.convolve1d`.
`famos_smooth_window` now returns the normalised kernel array (`h`, unit
area) instead of an integer box-car length — this is an internal helper with
no external callers (checked: only `famos_smooth` calls it), so no public
signature elsewhere changed. `famos_smooth`'s own signature is unchanged.

**Why:** the double-boxcar's implied triangle half-width is `N` samples,
but the FAMOS kernel identified by least-squares fit against
`data/Fx_raw_cut.csv` has half-width `a = 49.5` for `width_s=0.1s, fs=1000Hz`
— about 2x off, and wrong specifically at the tails (a boxcar-of-boxcars
kernel is flatter and doesn't taper to zero the same way a true triangle
does). That mismatch was the ceiling on match quality (99.995%, r~0.996);
the exact kernel removes it.

**Validation** (vs `data/Fx_raw_cut.csv`, skipping first/last N samples of
the cut):

| channel | skip | max abs err | mean abs err | r |
|---|---|---|---|---|
| `smo(Fx_raw, 0.1s)` (force) | 300 | 5.557e-03 N | 1.227e-04 N | 1.000000000 |
| `smo(Long_raw, 0.5s)` (accel) | 600 | 5.112e-06 | 9.461e-07 | 1.000000000 |

Both hit the target numbers exactly (5.6e-3 N and 5.1e-6). `red(Fx_smo, 10)`
re-checked and remains bit-exact (max err 0.0) against `Fx_red_cut` —
unaffected by this change, as expected.

**Edge handling unchanged in this step** — still nearest-edge padding via
`convolve1d(mode="nearest")`, same behaviour as the previous
`uniform_filter1d(mode="nearest")`. Step 2b (next) replaces this with
window-shrink renormalization at the true channel ends.

---

## Step 2b — smo edge handling: nearest-pad -> window-shrink (2026-08-09)

**File:** `dtt/preprocessing.py::famos_smooth`.

**Before:** the all-finite ("fast path") case used
`convolve1d(arr, h, mode="nearest")` — the kernel effectively saw the edge
value repeated past the array boundary. Window-shrink renormalization (via
the filled/weight ratio) only kicked in for channels with an *interior* NaN
gap, since that was the only branch that computed it, and even that branch
used `mode="nearest"` for the boundary, not a true zero-weight shrink.
**After:** both the true start/end of every channel and interior NaN gaps
are treated the same way — as zero-weight regions — via
`convolve1d(..., mode="constant", cval=0.0)` on both the value and the
weight-mask arrays, then dividing. This is the "shrink" edge mode from the
reference implementation (`famos_preprocess.py::famos_smooth`, `edge="shrink"`):
the kernel is renormalised by the fraction of it that overlapped real data,
so a straight ramp value at sample 0 stays close to the ramp instead of
being pulled toward a repeated-edge plateau. The separate all-finite fast
path is gone — every call now goes through the mask/weight computation, at
negligible extra cost (one more `convolve1d` call on a float array of the
same size).

**Why:** the brief calls for window-shrink to be the default at the true
ends of *every* channel, not only ones that happen to have an interior gap
already exercising that code path. Affects roughly `width_s/2` seconds
(~0.25 s for the 0.5 s accel width) at each channel end; the interior is
identical either way.

**Validation:** re-ran the Step 2 CSV checks (interior, skip=300/600) —
identical to the numbers above (max err 5.557e-3 N / 5.112e-6, both
r=1.000000000), confirming the edge-mode change does not touch the interior.
Edge behaviour spot-checked with a synthetic ramp: `smo(ramp, 0.1s)[0]` now
tracks the ramp's local value (~16.17 for `ramp=arange(200)`) instead of
sitting near a flat, edge-padded plateau. NaN-gap handling re-checked and
still preserves gaps exactly.

---

## Step 3 — audit: smooth-then-cut ordering (2026-08-09)

**No code change** — audited every place that could violate "smooth the full
channel first, then cut a sub-range out of it" and found nothing to fix.

Traced every caller of `famos_smooth` / `butterworth_lpf` / `apply_famos_recipe`
/ `apply_pipeline` in the repo:

* `dtt/comparison_io.py::load_recording` — the brief's flagged suspect. It
  calls `imc_reader.read_folder(folder, ...)` on the *whole* recording
  folder, which runs `apply_famos_recipe` (smo/FiltLP at native rate, then
  `red()`) before any comparison-level alignment or windowing happens in
  `dtt/comparison.py`. `dtt/comparison.py` itself does no smoothing at all —
  it only computes distribution statistics (RMS, P95, etc.) on already
  band-limited, already-decimated channels. No violation.
* `dtt/pipeline.py` — ingestion (stage 1, full recording) always runs before
  signal processing (stage 4); there is no sub-range/time-window field
  anywhere in `RunConfig` (`dtt/config.py`) that could cut the data first.
* `gui/pages/preprocess_page.py::_load_channel` / `_update` — the interactive
  preview page. `_load_channel` reads the *entire* channel column (all rows)
  from `study.processed_csv`, and `_update` runs `apply_pipeline` over that
  full column — never a user-selected sub-range. Confirmed by reading
  `_load_channel`: `pd.read_csv(..., usecols=lambda c: c in cols)` with no
  row slicing.
* `dtt/validation/famos_validation.py` — calls `butterworth_lpf` /
  `famos_smooth` directly on whatever DataFrame it's given, for fitting
  against FAMOS ground-truth columns; callers pass full studies, not cuts.

Conclusion: the codebase already respects smooth-before-cut everywhere smo/
FiltLP is invoked. No fix needed for this step.

---

## Step 4 — new despike step (2026-08-09)

**Files:** `dtt/preprocessing.py` (new functions + `apply_famos_recipe` kwargs),
`dtt/config.py::RunConfig` (new fields), `dtt/processing/signal_processor.py::
apply_filter` (CSV path), `dtt/ingestion/imc_reader.py::_assemble`/`read_folder`/
`read_files` (raw path), `dtt/ingestion/loader.py::load_raw_folder`/
`load_raw_files` (threads `RunConfig` through to the raw reader).

**New functions in `preprocessing.py`:**

* `detect_rail(x, min_run=3)` — rule 1, saturation: a run of `>= min_run`
  samples pinned at the channel's own min or max value. By value, not raw
  ADC counts, so it needs no factor plumbing and works the same for raw and
  CSV sources.
* `detect_dropout(x, max_run=5)` — rule 2, dropout: exact zeros/NaN (always
  flagged) plus any run of an identical value repeated `>= max_run` samples
  (a frozen sensor). Shares the same run-detection helper (`_flag_value_runs`)
  as rule 1 — the brief calls these "nearly the same test."
* `detect_narrow_spikes(x, fs, hw_cutoff_hz=200.0)` — rule 3, sub-hardware-
  width spike: the raw channel was hardware low-pass filtered at
  `hw_cutoff_hz` before being digitised at `fs`, so nothing narrower than
  `fs / (2*hw_cutoff_hz)` samples (2 samples, by default) can be real. Flags
  a same-signed excursion off a local median trend that is narrower than
  that width, **regardless of its size** — so a genuine peak of any
  amplitude survives as long as it's wide (see the correction below for why
  a tiny noise floor was still needed).
* `despike(x, fs, ...)` — runs all three rules, unions the flags, and
  replaces them by reusing the existing `_interpolate_over` helper (never
  drops samples). Optional `net=True` runs a very loose adaptive
  `hampel_deglitch` pass afterwards for gross leftovers; off by default.
  Returns `(despiked, pct_flagged)`.

**Correction made during validation — rule 3 needed a noise floor.** The
first cut of `detect_narrow_spikes` used a width test alone (median trend
window `8*min_width+1`, flag any sign-consistent run shorter than
`min_width`) and flagged **2.8%** of the real `Fx_raw_cut` channel — nowhere
near "a small fraction of a percent." The reason: a bare structural width
test can't distinguish a genuine artifact from the ordinary sample-to-sample
chatter every noisy signal has around its own local median, and that
chatter is *always* exactly 1 sample wide by construction, so it trips the
test constantly. Added a light gate — the excursion's peak must clear `2x`
the local high-frequency residual's robust scale (`1.4826 * MAD`) — which is
far below any real spike or peak amplitude, so it still doesn't discriminate
by size, only filters out deviations indistinguishable from measurement
noise. After the fix, rule 3 alone flags **7 samples (0.023%)** on the same
channel.

**Config surface (`RunConfig`, mirrors the existing `deglitch` fields):**
`despike` (on/off, default `False`), `despike_rail_min_run` (default 3),
`despike_dropout_max_run` (default 5), `despike_hw_cutoff_hz` (default
200.0), `despike_net` (default `False`), `despike_net_nsigma` (default 6.0),
`despike_net_window_s` (default 0.011). All optional kwargs with defaults on
`apply_famos_recipe`, `_assemble`, `read_folder`, `read_files` — existing
callers are unaffected (confirmed: `apply_famos_recipe(df, fs,
decimate_factor=10)` with despike omitted produces byte-identical `applied`
output to before this step, `"smo(0.1s) -> red(10)"`, no `despike` step
listed).

**Scope note:** wired into `RunConfig` and both ingestion call sites per the
brief; did **not** add a GUI checkbox or `--despike` CLI flag (the pattern
`deglitch` uses in `pipeline.py`/`new_study_page.py`) since the brief scoped
this step to `apply_famos_recipe` and its two call sites, not full UI parity,
and despike has 6 parameters vs. deglitch's 1. Left as a natural follow-up if
wanted — the config fields and wiring are already in place, so it's just
argparse/GUI plumbing at that point.

**Check 2 — despike runs on the CSV path too, not just raw:** built a
`RunConfig(despike=True, famos_mode=True, famos_applied=False)` (the state a
plain CSV study is in — ingestion didn't run FAMOS, so stage 4 must) and
called `signal_processor.apply_filter` directly on `Fx_raw_cut` loaded as a
DataFrame. Log output confirms the despike step actually ran:
`FL_Fx  despike(0.617%) -> smo(0.1s)`. Confirmed.

**Validation** (real channel, `data/Fx_raw_cut.csv::Fx_raw_cut`, n=30001,
fs=1000 Hz, all default thresholds) — no FAMOS ground truth exists for this
step, so validated by running on a real channel and inspecting the result:

| rule | flagged | % |
|---|---|---|
| rail (saturation) | 3 | 0.010% |
| dropout (zero/null/frozen) | 175 | 0.583% |
| narrow-spike (sub-hardware-width) | 7 | 0.023% |
| **total (union)** | **185** | **0.617%** |

Well within "a small fraction of a percent." Spot-checked the dropout runs
directly: 122 runs total, mostly single exact-`0.0` samples plus a handful
of 5-6-sample runs of one exact repeated value (e.g. `-168.915` x5,
`-177.155` x6) — patterns that essentially cannot occur from continuous
sensor noise landing on the same float bit-for-bit, which is exactly what
rule 2 is meant to catch.

Before/after plots saved to
`C:\Users\ekans\AppData\Local\Temp\claude\d--Apollo-Project-Main\85ae4cfe-2ec7-4f03-816e-3a7bcff1fc60\scratchpad\despike_validation.png`
(session scratch dir, not committed) around four flagged points — a frozen
run, a two-point exact-zero cluster sitting mid-way up a large real
excursion, a pinned rail run inside a real dip, and the narrow-spike catch.
In all four, the large genuine road-load swings on either side of the flag
are untouched, and only the flagged samples themselves are smoothly bridged.

---

## Check 3 — raw reader vs. FAMOS's own raw column: **latent bug found** (2026-08-09)

**No code change** (explicitly out of scope — "do not change the reader").
Per the brief, read `data/WFT_Fx_fr.raw` directly with the project's current
reader (`dtt/ingestion/imc_reader.py::read_raw`, fs=1000 Hz, the
smoothness-heuristic data-start detector) and compared the 1000–1030 s
window against `data/Fx_raw_cut.csv::Fx_raw_cut` (FAMOS's own physical-unit
raw for that same window — the CSV's `x` column runs 1000.000–1030.000 s,
so it's literally this window).

**At the naive index alignment (`i = round(1000 * 1000) .. + 30001`):
mismatch.** `corr = 0.854`, max abs err = 1178 N, mean abs err = 82 N —
values are demonstrably the same shape and order of magnitude but not the
same signal sample-for-sample.

**Searching a +/-2000 sample lag window found the true alignment at a
+58 sample offset:** `corr = 0.9999999999992875`, max abs err = 0.0050 N,
mean abs err = 0.000134 N — i.e. once shifted by 58 samples (58 ms at
1 kHz), the reader's decoded values match FAMOS's own raw column to the
CSV's export-rounding floor. This confirms the byte-level decode itself is
correct (int16 parsing, the calibration factor `1.373291015625`, event-record
stripping) — the reader is reading the right bytes and scaling them right.

**The bug is in `read_raw`'s data-start detection.** `_find_data_start`
picks the byte offset that locally looks "smoothest," and `read_raw`'s
trailing trim (the `ref = samples[500:2500]` / robust-threshold walk,
`imc_reader.py` lines ~152–170) then nudges that further — but the net
result lands 58 samples early relative to where FAMOS itself considers
sample 0. Because the heuristic always returns *something plausible* rather
than failing loudly (a smooth, in-range signal either way), this offset is
invisible without an external ground truth to check against — exactly the
risk the brief flagged. It would silently misalign **every channel read via
`read_raw`**, not just force channels, against wall-clock time, against
other channels read with a different offset (if the offset varies per file —
not established here), and against anything computed from **absolute
sample position** (e.g. a fixed rainflow window, an event marker, or
cross-referencing to GPS/video timestamps). It does **not** by itself
corrupt derived per-channel statistics that don't depend on absolute
alignment (mean/RMS/histograms of a single channel are shift-invariant), but
it would corrupt anything that assumes two independently-read channels (or
a channel and an external time reference) start at the same sample.

**Left unfixed, as instructed** — this is a reader-internals question (why
58, is it constant across files/channels, is it the `_find_data_start`
window search or the post-hoc trim loop) that needs its own investigation,
not a one-line change alongside a preprocessing migration. Flagging for
follow-up: whoever owns `imc_reader.py` should scope how `_find_data_start`
and the leading-trim heuristic land relative to a **known-correct**
reference like this file (a repeatable regression check now exists: this
exact comparison, `WFT_Fx_fr.raw` sample `[1000000+58 : 1030001+58]` vs.
`Fx_raw_cut.csv`).

---
---

# Round 2: histogram restyle, stop removal, spike rule, resampling confirmation (2026-08-25)

Since the last round, `famos/` became the single home for the verified recipe
math (triangular smo, causal FiltLP, stride red) — `dtt/preprocessing.py`
now delegates to `famos.ops` rather than carrying its own copy. That math is
untouched in this round; everything below is new preprocessing/reporting
features layered on top of it.

## Step 1 — Resampling: confirmed, no change

`famos_red` (delegating to `famos.ops.red`) is plain stride decimation, keep
every 10th sample from index 0, no anti-alias filtering of its own — verified
exact again: `red(Fx_smo_cut, 10)` vs `Fx_red_cut` from `data/Fx_raw_cut.csv`,
max err 0.0. It already runs last in `apply_famos_recipe`'s per-channel order
(`despike -> deglitch -> FiltLP -> smo -> red`), after the signal is
band-limited, which is the FAMOS-correct order. `resample()`'s
`mode="antialias"` (scipy `decimate`) path exists in `preprocessing.py` but
grepping the whole tree found no caller that ever sets `resample_mode` to
anything but the default `"famos"` (plain stride). No code change needed.

## Step 2 — Histogram restyle (2026-08-25)

**New file** `dtt/analysis/plot_style.py`: `detect_shape`, `banded_colors`,
`draw_histogram` — one shared helper for every histogram in the app.
**Extended** `dtt/analysis/auc.py::kde_curve` with optional `weights` (so the
curve matches a distance-/percentage-weighted histogram, not a plain sample
count) and `reflect_boundary` (the standard reflection method: mirror data
about the boundary, fit on the pooled set, fold back and double — a
boundary-corrected KDE that doesn't droop toward zero right at a hard bound).
Backward compatible: default args reproduce the old unweighted, unreflected
curve exactly, so `auc_grid`'s existing callers (the comparison-page charts)
are unaffected.

**Wired into** `dtt/analysis/histograms.py::_plot_single_histogram` (replaced
the flat `ax.bar` call) and `dtt/analysis/rainflow.py::generate_rainflow`'s
range-distribution panel (replaced `ax2.hist(..., density=True)`, passing
`bar_is_density=True, boundary=0.0` since a rainflow range is `|from-to|`
and can never be negative — always the one-sided case with a genuine bound).
**Left out of scope, per your call**: the two-series overlay comparison
charts in `dtt/comparison_io.py` / `gui/pages/comparison_page.py` — they
already have their own purpose-built KDE-fill comparison style and the
single-distribution bands/bell don't have a natural two-series extension.

**Design, condensed:**
- `detect_shape`: `"one_sided"` if the data's low edge sits within 10% of its
  own range from 0 (a hard bound, like rainflow range), **or** if
  `|weighted skew| > 1.0` (a skewed-but-not-bounded force channel) —
  otherwise `"symmetric"`.
- `banded_colors`: 5 bands from a darkened to a lightened version of the
  channel's *own* colour (not a fixed palette), so the restyle doesn't
  break the existing wheel-colour coding used throughout the app. Symmetric
  bands by `|z|` at the brief's edges (0.5/1/1.5/2 sigma); one-sided bands
  by quantile position instead (distance-from-centre isn't meaningful for a
  distribution that only goes one way).
- Mean line: drawn (white, `axvline`) only for `"symmetric"`.
- KDE curve (red, `PLOT_COLORS["danger"]`): scaled to the bars' own units —
  a density integrates to 1 against the weights (or the sample count when
  unweighted), so `density * bin_width * total_weight` matches "weighted
  sum per bin" bars (the force histograms) and a raw `density` matches
  bars that are already `density=True` (rainflow).

**Bug found and fixed during validation, not in the original design**: my
first cut always reflected the one-sided curve around the plotted axis's
left edge when the caller didn't pass an explicit `boundary`. That's correct
for rainflow (0 is a real physical bound) but wrong for a force channel that
tests one-sided purely from **skew** — there the axis edge is just wherever
percentile-trimming happened to start the plot, not a real boundary, and
reflecting around it invents density that isn't there. Fixed: reflection
only happens when the caller passes an explicit `boundary` (rainflow does,
`histograms.py` doesn't); a skewed-but-unbounded channel gets a **plain**
(unreflected) KDE instead, which still follows the actual skewed shape with
no symmetry assumption, it just skips the edge correction there's no real
edge to correct.

**Validation**: a synthetic normal + exponential smoke test confirmed both
branches render as intended (scratch dir, not committed) before wiring into
the real draw sites. Then ran `generate_histograms` + `generate_rainflow`
end to end on a synthetic multi-channel frame built from
`data/Fx_raw_cut.csv`'s real (noisy, genuinely bimodal) Fx trace:
- `hist_distance_FL_Fx.png` (skewed/bimodal, real data) — bands and a
  plain KDE follow the actual double-hump shape rather than forcing a bell
  onto it; classified `one_sided` by skew even though it isn't hard-bounded,
  confirming the skew path works, not just the boundary path.
- `hist_distance_FL_Fz.png` (same skew-by-construction in this synthetic
  test, since Fz was derived from the same Fx trace) — confirmed the
  boundary-reflection fix: no droop/distortion at the plotted left edge
  after switching to a plain KDE for this case.
- `rainflow_FL.png` bottom row (`FL_Fx/Fy/Fz` range distributions) — genuine
  one-sided-with-a-real-bound case: curve correctly peaks at/near x=0
  without drooping toward zero, matching the true density right at the
  boundary. P95 line and damage annotation (untouched by the restyle) still
  render correctly alongside.
Both target archetypes confirmed against real data before moving on.

## Step 3 — 20% transient spike rule (2026-08-25)

**New function** `dtt/preprocessing.py::detect_transient_spikes` — despike
rule 4, encoding the manual step "a sudden spike of more than 20% within a
1 second time frame should be adjusted or eliminated." Wired into
`apply_famos_recipe()` as new `transient_*` kwargs, threaded through
`RunConfig`, `signal_processor.py::apply_filter` (CSV path),
`imc_reader.py::_assemble`/`read_folder`/`read_files`, and `loader.py`'s two
raw call sites — same chain as the despike round. Off by default; existing
callers unaffected (confirmed: `apply_famos_recipe` output is byte-identical
with the new kwargs omitted).

**Placement pivot, mid-implementation — this rule does NOT run on the raw
channel, unlike the other three despike rules.** The plan called for it
raw, alongside rail/dropout/narrow-spike. First validation against raw
1kHz data flagged 25-90% of the channel regardless of window/threshold
tuning — not a tuning problem, a scale mismatch: raw 1kHz WFT samples
routinely deviate more than 20% from even a 21ms local median purely from
ordinary sample-to-sample jitter (confirmed: 25.7% of samples already
exceed 20% of a 21ms trend with the width test removed entirely). Only
inflating the percent threshold to ~300-500% brought the raw-data flag
rate down to a believable range, which would have gutted the rule's
stated intent. Flagged this to you mid-step rather than picking an
arbitrary threshold; you confirmed the fix: **run this rule on the
already-conditioned (smo/FiltLP'd, decimated) signal instead** — the
manual's "20% within 1 second" describes what an operator saw on a
conditioned FAMOS trace, not literal raw samples — while the three raw-stage
rules (rail, dropout, sub-hardware-width) stay exactly where they were,
since they catch true digitizer glitches that only show up before
smoothing. `despike()` no longer carries a `transient_*` parameter; the new
rule runs as its own step in `apply_famos_recipe`, after `red()`, gated to
channels the recipe actually smoothed/filtered.

**Design, once moved to conditioned data:**
- Local trend = `window_s`-wide (default 1.0s) rolling median.
- Threshold = the *larger* of `pct_threshold`% (default 20%) of the local
  trend level, and `noise_floor_mult` (default 10.0) times the channel's own
  robust local noise scale (`1.4826 * median(|residual|)`) — same fix
  pattern as rule 3's noise floor, needed for the same reason: a bare 20%
  test, even on conditioned 100Hz data, still flagged ordinary chatter at
  low multipliers.
- **Second bug found during validation**: "narrow" was originally defined as
  "run length `< window_s`" — i.e. anything under the full second. A real,
  smooth, monotonic ~0.7s braking/cornering swing found in the actual
  reference recording passed that test and got flagged, which is exactly
  the "eating real peaks" failure mode the brief warned about. Fixed by
  decoupling the width cap from the trend window: `max_spike_frac` (default
  0.4) caps the flagged run at that fraction of `window_s` (0.4s of a 1s
  window), which is what actually distinguishes a spike-and-return from an
  ordinary wider real excursion that happens to resolve inside a second too.
- Replacement via the existing `_interpolate_over` helper, never dropped.

**Validation** (real channel, `data/Fx_raw_cut.csv::Fx_red_cut`, the 100Hz
decimated ground truth, n=3001, defaults):
- Baseline (unmodified real data): **0 samples flagged (0%)** — the clean
  FAMOS-equivalent output has no artifacts of this kind, as expected.
- Injected genuine narrow transient (Gaussian bump, ~0.3s, 800 daN peak,
  added to the real trace): **34 samples flagged and correctly bridged** —
  confirms the rule can actually catch something, not just stay silent.
- Injected wide sustained ramp (same 800 daN peak, ~0.7s): **0 samples
  flagged** — confirms real load of comparable size survives when it's wide
  enough, the core safety requirement.
- Full pipeline smoke test (`apply_famos_recipe(..., transient_enabled=True)`
  with the same injected narrow transient): correctly caught post-`red()`,
  reported as `transient(0.433%)` in the `applied` log — well within "a
  small fraction of a percent."
- Before/after plots saved to
  `C:\Users\ekans\AppData\Local\Temp\claude\d--Apollo-Project-Main\85ae4cfe-2ec7-4f03-816e-3a7bcff1fc60\scratchpad\transient_spike_validation.png`
  (scratch dir, not committed): the injected spike is cleanly bridged; the
  wide real swing is untouched, before/after traces overlapping exactly.
- Re-ran the Step 1/2-era smo/FiltLP/red ground-truth checks after these
  edits: identical numbers to before (5.6e-3 N / 5.0e-6 / 0.0 respectively)
  — confirms the verified recipe math in `preprocessing.py` wasn't disturbed.
- `pytest tests/` — 37/38 pass, same single pre-existing failure as last
  round (missing gitignored proprietary file, unrelated).
