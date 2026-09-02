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

## Step 4 — Stop / stationary-period removal (2026-08-25)

**Files:** `dtt/preprocessing.py::remove_stops_frame` (extended, new seam
helpers `_stop_intervals`, `_seam_reference`, `_nudge_stop_boundaries`,
`_blend_seams_inplace`), `dtt/config.py::RunConfig` (new `remove_stops`/
`stop_*` fields), `dtt/pipeline.py` (new stage + `_find_speed_column` helper).
`detect_stops`/`detect_stops_from_speed`/`remove_stops` (single-channel) were
already correct from the previous round and are unchanged.

**Amendments from you before starting, both implemented as specified:**

1. **`processed_data.csv` is never overwritten with stop-removed data.**
   `pipeline.py` still writes it from the full, canonical `df` right after
   stage 4, exactly as before. A *separate* `stats_df` is built via
   `remove_stops_frame` (only when `config.remove_stops` is set) and is the
   only thing that changes: `compute_statistics` and `generate_rainflow` now
   read `stats_df` instead of `df`; severity, histograms, heatmaps, boxplots,
   and both CSVs (`processed_data.csv`, `raw_data.csv`) still read the full
   `df`. Removed intervals are written to `metadata["stop_removal"]`
   regardless of whether anything was actually cut. Reversible by
   construction: turning `remove_stops` off is the only difference between
   `stats_df` and `df`.
2. **Rainflow before/after cycle-count and damage comparison added to
   validation**, alongside an actual seam level plot (not an asserted
   number) — see below.

**Seam handling** (`remove_stops_frame`'s new `seam_search_s`/`seam_blend_s`
kwargs, both against the *same* shared mask so every channel still cuts at
identical indices):
1. `_seam_reference` builds one robust-z-scored "how loaded is the frame"
   signal from every force/moment channel jointly (never from an individual
   channel, for the same reason the stop mask itself is one shared decision).
2. `_nudge_stop_boundaries` extends each cut boundary (up to `seam_search_s`,
   default 1.0 s, each side) to the nearest low-reference sample — "cut at a
   quiet moment," the first line of defence against a join step.
3. `_blend_seams_inplace` replaces a short stretch (`seam_blend_s`, default
   0.2 s, each side) at whatever step survives the nudge with a straight
   line, per channel — the backstop. Per your note: this is a small,
   deliberate distortion of already-short stretches at each seam, an
   acceptable default; worth revisiting only if a study turns out to have
   many stops. Not changed now.

**Validation — real data, not synthetic.** `data/WFT_Fx_fr.raw`'s
1000 Hz `FR_Fx` channel, t=3100-3300 s (a window containing two genuine
stationary stretches, confirmed against the recording's own rolling-σ
signature), run through `apply_famos_recipe(..., despike_enabled=True,
transient_enabled=True)` to 100 Hz — i.e. the actual target pipeline order
(despike raw -> smo/FiltLP -> decimate), not a shortcut. Built a 4-channel
frame (`FL/FR/RL/RR_Fx`, the real conditioned trace plus independent noise
per wheel — physically reasonable since all four wheels stop simultaneously)
with no speed column, exercising the force-dynamics-vote fallback path.

**A debugging detour that turned into useful evidence.** The first pass
skipped despike/transient entirely and found a rainflow damage ratio
(after/before stop removal) of 0.38 — alarming, since "barely move the
damage number" was the whole point. Traced it to a single ~900 daN spike
sitting *inside* one of the detected stop intervals (visible in the raw
trace, not a stop-removal artifact) that dominated the m=8 Miner-weighted
damage sum on its own. Re-ran with the real target pipeline order
(`despike_enabled=True, transient_enabled=True` — the raw-stage rules don't
catch this one, it's ~150ms wide, `detect_transient_spikes`'s job, not
`detect_narrow_spikes`'s) and the spike is gone from the conditioned trace
before stop removal ever sees it. This is a genuine cross-check that the
despike work from Steps 3/3-of-last-round and stop removal are pulling in
the same direction, not fighting each other — validate stop removal with
the artifacts it will actually run downstream of in production, not in
isolation.

**Final numbers** (full target pipeline, real data):

| metric | value |
|---|---|
| window | 200 s (`data/WFT_Fx_fr.raw`, t=3100-3300s, conditioned to 100 Hz) |
| stops found | 2 (`53.91-83.93s`, `87.25-144.21s`) |
| removed | 86.98 s (43.5% of this deliberately stop-heavy test window) |
| basis | force dynamics (4 channels), no speed column in this single-source test |
| rainflow cycles | 4382 -> 1524 (most of the drop is near-zero-amplitude noise cycles from the dead stretches, which is exactly what should disappear) |
| rainflow damage (m=8) | ratio after/before = **1.000000** |

All 4 channels confirmed cut at identical indices (single shared mask via one
`df.loc[keep]`, no NaNs introduced, row counts match across columns).
Before/after traces, a zoomed seam level plot (before: raw step at the join;
after: joined + blended, no visible step), and the rainflow comparison saved
to
`C:\Users\ekans\AppData\Local\Temp\claude\d--Apollo-Project-Main\85ae4cfe-2ec7-4f03-816e-3a7bcff1fc60\scratchpad\stop_removal_full_validation.png`
(scratch dir, not committed).

**Not done, out of scope per your original brief**: no GUI checkbox / CLI
flag, same scope decision as despike and the transient rule — `RunConfig`
fields and full pipeline wiring are in place, off by default.

---

# Round 3: missing analysis features + histogram readability (2026-08-31)

Ports the four confirmed gaps against the standalone WFT Analyzer script
(distance-weighted heatmaps, Welch PSD, two-run comparison tables, RF
Compare) plus a force-distribution histogram readability fix. Full plan in
the session's plan file; two Round 2 premises turned out to already be
resolved in code before this round started — noted where relevant below
rather than redone.

## Step 1 — B2: histogram x-axis range option (2026-08-31)

**Confirmed first, not redone:** B1 (banded/KDE styling) and B3's
speed-channel fallback labeling were already implemented in `histograms.py`
as of Round 2's `685519e` — verified by reading the code, not assumed from
the brief. The only real histogram-readability gap was the fixed axis.

**Issue:** `_axis_range()` only escaped the configured `FORCE_RANGES_DAN`
range (Fx/Fy ±300 daN, Fz 100-1000 daN) when more than 2% of samples fell
outside it — a safety net for a different vehicle class. Normal in-range
data (e.g. real Fx sitting ±40 daN inside the configured ±300) always got
the full configured span, crushing the actual distribution shape into the
middle third of the plot.

**Fix:** `generate_histograms(df, config, range_mode="full")` — new
`range_mode` parameter, `"full"` (default, byte-identical output/filenames
to before) or `"autoscale"` (always uses the existing P0.5-P99.5 percentile
range, previously only reached as an out-of-range fallback). Refactored the
percentile computation into `_percentile_range()` so both paths share it.
Non-default output gets a `_autoscale` filename suffix so both modes can be
generated side by side without overwriting each other.

**Validated:** synthetic Fx data (σ=15 daN, well inside the configured
±300) — `full` mode returns exactly `(-300.0, 300.0)`; `autoscale` mode
returns roughly `(-40, 40)`, matching the actual data spread. Confirmed
`range_mode="full"` produces identical filenames to the pre-change code
(no suffix), and an invalid `range_mode` raises `ValueError` rather than
silently doing something unexpected.

**Not wired to a `RunConfig` field or GUI toggle this round** — deliberate,
per the plan: adding a config field now risks not matching whatever toggle
the eventual GUI restructure round settles on. `range_mode` is exposed as a
function parameter, ready for either a config field or a direct GUI call
once that's scoped.

## Step 2 — A1 + B3: distance-weighted heatmaps (2026-08-31)

**Issue:** `heatmaps.py::generate_heatmaps` weighted its three force-pair
heatmaps (Fy-Fx, Fx-Fz, Fy-Fz) by sample count only (`np.histogram2d` with
no `weights=`), while the robust distance-weighting logic
(`histograms.py::_get_speed_weights` — handles dead/stuck/negative/
implausible speed channels, km/h vs m/s auto-detection) already existed one
module over and was already wired into `histograms.py`.

**Fix:** `heatmaps.py` imports `_get_speed_weights` from `histograms.py`
(reused, not copied — no second implementation of the speed-channel
validation logic). `generate_heatmaps` computes `weights_full` once and
threads it through `_pair_heatmap` -> `_hist2d_pct` (now accepts an optional
`weights=` array, applied via `np.histogram2d(..., weights=w[mask])` so each
cell's percentage is a share of distance, not sample count). Colorbar label
and figure suptitle switch between "% of Distance / Speed weighting: {unit}"
and "% of Occurrence / Sample-count weighting — no usable speed channel"
depending on whether a usable channel was found — same honest-fallback
pattern `histograms.py` already used, not a new convention.

**Deliberately left unweighted:** the per-wheel Fx/Fy hexbin density plot
(`generate_heatmaps`'s second loop). Hexbin has no clean per-point weights
API without manual pre-binning, and the brief's ask was specifically the
three force-pair heatmaps, not the hexbin.

**Validated:** synthetic 4-wheel data with a constant 72 km/h (20 m/s)
speed channel — `_get_speed_weights` returns a total distance whose km sum
exactly matches the recording's known distance (`20 m/s * 5000 samples /
100 Hz / 1000 = 1.000 km`, confirmed to 1e-6). A second run with a
stuck-at-zero speed channel correctly returns `None` (all-zero rejection
already in `_speed_as_mps`), and `generate_heatmaps` completes without
error, falling back to count-weighting with the labelled title.

## Step 3 — A2: Welch PSD (2026-08-31)

**Issue:** `dtt/spectral.py` had `amplitude_spectrum_db` (single windowed FFT,
dB magnitude — the FAMOS-style spectral-line view) but no Welch PSD anywhere
in the codebase (confirmed by grep, zero hits for "welch" outside the
standalone script). These answer different questions: Welch's segment
averaging gives a statistically stable noise floor; a single FFT shows exact
spectral lines. Durability/fatigue work commonly wants both.

**Fix:** `spectral.py::welch_psd(x, fs, nperseg=None)` wraps
`scipy.signal.welch`. Default `nperseg = min(len(x), max(256, fs*4))` — a
longer segment than a live-GUI tool would pick (the standalone script used
`min(len, max(64, fs*2))` for interactive responsiveness), documented in the
function's own docstring as a deliberate batch-pipeline accuracy choice, not
a value inherited from the script. New `dtt/analysis/psd.py::generate_psd`
follows the same per-wheel-figure pattern as `rainflow.py`/`histograms.py`
(iterates `config.run_channels.wheel_groups`, not hardcoded FL/FR/RL/RR;
sample rate from `config.sampling_rate`, never typed), saves `psd_{wheel}.png`
into `config.figures_dir`. Wired into `pipeline.py` as stage `[9b/9]`,
same try/except-and-log pattern as every other analysis stage so a PSD
failure can't take down the run.

**Validated:** a synthetic 5 Hz tone (buried in noise, 100 Hz sample rate,
200 s) — `welch_psd`'s returned frequency axis tops out at exactly `fs/2 =
50 Hz` as expected, and the PSD's peak lands at 5.0 Hz, recovering the known
tone. `generate_psd` on synthetic 4-wheel data produces one `psd_{wheel}.png`
per wheel (4 files). Confirmed `dtt.pipeline` still imports cleanly with the
new stage wired in.

## Step 4a — A3 prerequisites: RMS helper + processing provenance (2026-08-31)

Two small additions ahead of the two-run comparison tables themselves,
both requested explicitly rather than left to judgment call:

**RMS helper.** `dtt/analysis/statistics.py::rms(x)` — a basic statistic
placed alongside `mean`/`median`/`std` (not inside the new comparison
module), so there is exactly one RMS implementation to import from, not a
second one that could drift. Also added to `compute_statistics`'s per-channel
`stats_summary.json` entry, since it belongs there for the same reason.
Validated: `rms([3,4]) == sqrt((9+16)/2) == 3.5355...`, matches
`np.sqrt(np.mean(x**2))` to float precision; all-NaN input returns NaN
rather than raising.

**Processing provenance.** Confirmed by reading the code: no per-study
metadata is persisted anywhere today — `pipeline.py`'s `metadata` dict only
feeds text into the PPTX report and is discarded, and `famos/audit.py`'s
`RunManifest` (a full checksum/git/library-version audit manifest) exists
but has never been instantiated or written by `pipeline.py`. Comparing two
studies without recording what produced each one would let a recipe-version
or units difference between them look like a real EV-vs-IC load difference.

New `dtt/provenance.py` — deliberately smaller than wiring up the full
`RunManifest` (that would touch every pipeline stage and risks scope creep
into ingestion territory this round explicitly excludes). `build_provenance`
reuses `famos.audit.library_versions()`/`git_commit()` as-is rather than
reimplementing them, and records `famos_applied`, `famos_mode`,
`n_to_dan_applied` (the units signal, from `metadata`/`scale_meta`),
`sampling_rate_hz`, `despike`/`transient_despike`/`remove_stops` flags,
filter settings, and the famos/numpy/scipy/git versions. `write_provenance`
saves `run_provenance.json` into each study's output dir — wired into
`pipeline.py` right after `processed_data.csv` is saved, same point
`metadata` and `config` are both in their final per-run state.
`compare_provenance(a, b)` returns human-readable warning strings (not
errors) when `famos_applied`/`n_to_dan_applied`/`famos_version`/`git` differ
between two studies' records, or when one study predates provenance
tracking — same warn-don't-block pattern `dtt/comparison.py::_add_scale_note`
already established for its own Fz-ratio sanity check.

**Validated:** round-tripped `build_provenance` -> `write_provenance` ->
`load_provenance` and confirmed the loaded record matches exactly what was
written. `compare_provenance` correctly flags a `famos_applied` mismatch
between two synthetic records, returns no warnings for two identical
records, and correctly reports "no provenance recorded" (as a warning, not
an exception) when one side is `None`. Confirmed `dtt.pipeline` still
imports cleanly with the write wired in.

## Step 4 — A3: two-run comparison tables (2026-08-31)

**Issue:** no concept of comparing two *finished* studies existed anywhere.
`dtt/comparison.py` compares two raw imc folders (bypasses the whole
FAMOS/despike/stop-removal pipeline — its `comparison_page.py` worker calls
`dtt.comparison_io.load_recording` directly), which is a different data
source than two already-processed studies.

**Fix:** new `dtt/analysis/study_compare.py`. `load_study(study_dir)` reads
`processed_data.csv` and re-derives `RunChannels` from its columns via
`build_run_channels` — the same call `pipeline.py` makes at run time, so a
study's channel labels are recovered exactly as the pipeline saw them, no
FL/FR/RL/RR hardcoding. `compare_studies(study_a_dir, study_b_dir)`
orchestrates three delta tables, each reusing an existing, already-validated
formula rather than reimplementing it:

- **RMS** — `dtt.analysis.statistics.rms` (Step 4a), per channel
  (`{label}_{Fx,Fy,Fz}`).
- **DLC** — `dtt.analysis.severity.dynamic_load_coefficient`, per wheel.
- **G-severity** — `dtt.analysis.severity.g_severity`, per wheel, one row
  each for Gx/Gy/Gxy.

Each row is a `MetricDelta(label, metric, a, b, delta, pct)` —
`delta = b - a` and `pct = 100*delta/|a|`, mirroring `dtt/comparison.py`'s
`ChannelDelta` shape for consistency without reusing its raw-folder-specific
loading code. Wheel labels not present in *both* studies are skipped with a
logged warning rather than raising — lets a comparison still run on the
channels that do overlap. Provenance (Step 4a) is checked via
`compare_provenance` and surfaced as `result.warnings`; the comparison still
computes and returns all the numbers even when provenance disagrees or is
missing — warn, don't block, same as the RMS-delta shape it mirrors.

Output: `save_comparison` writes `study_comparison_{a}_vs_{b}.json`;
`generate_comparison_figure` renders all three tables as one PNG in the
`severity.py::generate_severity_figure` visual style (dark theme, same
banded-row table look), generalized from one table to three, with any
provenance warnings printed across the top in the danger colour. Both go
into a new `outputs/_comparisons/` folder — a comparison belongs to neither
study's own output directory. No GUI entry point this round (explicit
decision — backend-only, deferred to the GUI-restructure round); reviewed
via the saved JSON/PNG for now.

**Validated:** two synthetic 4-wheel studies (20,000 samples each, one with
an 80 daN Fz offset simulating a heavier vehicle) — `compare_studies`
produces 12 RMS rows (4 wheels × 3 components), 4 DLC rows, 12 G-severity
rows (4 wheels × Gx/Gy/Gxy); confirmed `DS2 - DS1 == delta` to 1e-9 for
every one of the 28 rows. Confirmed no warnings when both studies share
identical provenance, confirmed a `famos_applied` mismatch between two
studies is correctly flagged, and confirmed a study with no
`run_provenance.json` (simulating an older study) is reported as
"no provenance recorded" rather than raising. Confirmed a partial-overlap
case (one study missing most channels) skips the non-overlapping wheels
with a logged warning instead of crashing. JSON and PNG both write
successfully to a `_comparisons/` folder.

## Step 5 — A4: RF Compare (2026-08-31)

**Fix:** extended `dtt/analysis/study_compare.py` (belongs with A3, per the
plan) rather than a new module. `_axle_labels(rc, axle)` generalizes
"front axle (FL+FR)" beyond a hardcoded 2-axle car: front = the first half
of `RunChannels.labels` in the order the platform's own axle-dynamic model
reports them, rear = the rest — works for any axle count, not just a
4-wheel car. `compare_rainflow(study_a_dir, study_b_dir, component, axle)`
combines the axle's wheels' cycles within each study and computes Miner
damage via `rainflow.py::_miner_damage` **unchanged** — this step adds no
new rainflow math, only the axle-grouping-across-two-studies orchestration
the brief specified. `generate_rf_compare` renders the range-distribution
KDE overlay via the Step 4 `draw_comparison_kde`/`draw_stats_strip` helpers
(same shared styling as A3, not a separate look) with the Miner damage
ratio (`damage_b / damage_a`) and both raw damage numbers in the title, and
saves `rf_compare_{axle}_{component}_{a}_vs_{b}.png` into the same
`outputs/_comparisons/` folder A3 uses.

**Validated:** two synthetic studies with an oscillating Fx signal on
FL+FR (amplitude 50 vs. 90, everything else noise) — `_axle_labels`
correctly resolves `axle="front"` to `['FL', 'FR']`. Independently
recomputed the front axle's combined Miner damage by calling
`_extract_cycles`/`_miner_damage` directly (bypassing `compare_rainflow`
entirely) and confirmed it matches `result.damage_a` to 1e-6 — the
cross-check the brief's validation step asked for ("the Miner ratio equals
the two studies' individual damage numbers divided"). Confirmed
`damage_ratio == damage_b / damage_a` exactly, confirmed the higher-amplitude
study (90 vs. 50) produced the larger damage number as physically expected
(ratio ≈ 104x, consistent with Miner's `range^m` at `m=8` amplifying a
1.8x amplitude difference sharply), and confirmed the PNG renders
successfully.

## Step 6 — force-distribution histogram restyle: light reference match (2026-08-31)

**Diagnosis (done before any edit, as instructed).** The force-distribution
histograms were already routing through the one shared `plot_style.py::
draw_histogram` — no separate/older draw path existed, confirmed by tracing
`histograms.py::_plot_single_histogram`'s call into it. The dark look came
from two places instead: (1) `histograms.py`'s own `_style_ax` (a
module-local function, independent of every other module's) set a dark
navy `PLOT_COLORS["panel"]` background and light-on-dark text; (2)
`draw_histogram` itself defaulted to a dark/saturated look — a fixed red
`KDE_COLOR` and dark-centre-to-light-edge banded bars — shared with
`rainflow.py`'s range-distribution panel, its only other caller. So this
was case 2 from the brief: correctly wired to `plot_style`, but
`plot_style`'s own look needed softening, scoped so only the force
histograms changed.

**Reference clarified mid-task:** the attached reference turned out to be
the two-run EV-vs-IC comparison look (filled KDE curves, no bars, light
background, stats strip) — already built as `draw_comparison_kde`/
`draw_stats_strip` in Round 3 Step 4, explicitly out of scope for *this*
task. The force-distribution target is one strand of that same visual
language: a single soft-filled curve, no discrete bars at all.

**Fix, additive to `draw_histogram` (`plot_style.py`):** new `style: str =
"banded"` parameter. `"banded"` is the exact prior behaviour (bars +
banded colours + fixed red curve) — unchanged, so `rainflow.py`'s output is
untouched. New `style="soft"`: skips `ax.bar()`/`banded_colors()` entirely
and instead draws `ax.fill_between(x_grid, curve, alpha=0.25, color=color)`
+ a thin `ax.plot(..., linewidth=1.3, color=color)` — reusing the exact
same `kde_curve` call (same weighting, same boundary-reflection logic;
nothing about what's *computed* changed, only what's drawn). The mean/
percentile marker line's colour switches from hardcoded white (invisible
on a light background) to a muted dark grey when `style="soft"` — a
necessary visibility fix, called out rather than silently done.

**Fix, local to `histograms.py`:** replaced the module's `BG`/`PANEL`/
`TEXT_PRI`/`TEXT_SEC` constants — previously aliased from the shared dark
`PLOT_COLORS` dict — with new literal light-theme colours defined directly
in this module (white/pale-grey backgrounds, quiet light-grey gridlines,
thin light-grey spines, dark-grey text). Confirmed by grep that nothing
outside this file imports these names (only `_get_speed_weights` is
imported elsewhere, by `heatmaps.py`), so `boxplots.py`/`rainflow.py`/
`heatmaps.py`/`severity.py` — which import the same shared `PLOT_COLORS`
dict independently — are unaffected. `_style_ax` gained a light gridline
call. `_plot_single_histogram` now calls `draw_histogram(..., style="soft")`
and the per-channel title switched from the loud per-wheel primary colour
to the new dark `TEXT_PRI`, per the brief's "small dark unobtrusive text."
`PLOT_COLORS` import dropped from `histograms.py` (no longer used).

**Validated:** regenerated one wheel's Fx/Fy/Fz force-distribution
histograms (both `range_mode="full"` and `"autoscale"`, confirming last
round's option still works unchanged) on synthetic data — visually
confirmed light background, thin single-tone curve, soft fill, quiet grid,
small dark labels, matching the reference. Unit-level check on
`draw_histogram` directly: `style="banded"` still draws exactly one bar per
bin plus 2 lines (mean + KDE); `style="soft"` draws zero bars, one
fill-between collection, and the same 2 lines; the returned histogram
`heights` are identical between the two styles (same underlying math,
different rendering only); an invalid `style` value raises `ValueError`.
**Regression check:** generated `rainflow_FL.png` from identical seeded
synthetic data on the pre-change code and on this change, and compared
SHA-256 hashes — **byte-identical**, confirming `rainflow.py`'s output
(the shared function's other caller) is completely untouched.

---

# Round 4: rainflow restyle, FAMOS-validation export, workflow modes (2026-08-31)

## Step 1 — rainflow-page histogram row: light restyle (2026-08-31)

**This explicitly supersedes Round 3 Step 6's byte-identical constraint on
`rainflow.py`.** That constraint was correct for its round (styling was
scoped to the Histograms page only); this round the user chose to extend
the new light look to the rainflow range-distribution row too.

**Scope, precisely:** the rainflow figure has two rows per channel — the
Siemens from-to matrix (`ax1`, via `_draw_fromto_ax`) and the range
distribution (`ax2`, via `draw_histogram`). Both previously shared
`rainflow.py`'s own dark `_style_ax`. Only `ax2` changes; `ax1` and
`_draw_fromto_ax`/`_style_ax` are completely untouched.

**Fix:** `rainflow.py` imports `histograms.py`'s light `_style_ax` (aliased
`_style_ax_light`) plus its light `BG`/`TEXT_PRI`/`TEXT_SEC` constants —
reused, not duplicated, so the two pages' light themes can't drift apart.
`ax2` now calls `_style_ax_light(ax2)` and
`draw_histogram(..., style="soft")` (the same additive parameter Round 3
Step 6 added — no third styling path). The P95 marker line, title, axis
labels, damage-annotation textbox, and legend on `ax2` all switch from the
dark-theme colours to the light ones (dark text, white-background damage
box) to match. `ax1` keeps its dark styling and its own `_style_ax`
unchanged.

**Validated:** regenerated a rainflow figure on synthetic oscillating Fx
data (visual: light background, thin single-tone curve, soft fill, quiet
grid, small dark text on the bottom row; from-to matrix row still dark,
unchanged). Independently recomputed the from-to matrix and Miner damage
via `_extract_cycles`/`_build_fromto_matrix`/`_miner_damage` directly
(bypassing the plotting code entirely) and confirmed they match what the
figure's damage annotation shows (5.96e+16 for FL_Fx, m=8) — the rainflow
math itself was never touched by this styling change, confirmed both by
reading the code (no math functions edited) and by this independent
recomputation.

## Step 2 — temporary FAMOS-validation CSV export (2026-08-31)

**Confirmed first:** `processed_data.csv` already captures exactly the
right moment — right after `df = apply_filter(df, config)` (FAMOS recipe
done), before any analysis stage. It already holds all channels at full
precision. The only real gaps were: a distinct, unambiguous-as-temporary
filename; `RunChannels`-driven column order instead of incidental
DataFrame order; and an opt-in gate so it isn't written on every run.

**Fix:** `RunConfig.export_famos_validation_csv: bool = False` — new field,
inline-documented as temporary. New `pipeline.py::_write_famos_validation_csv(df,
config)` — a top-of-function docstring marks it explicitly as a temporary
FAMOS cross-check aid, a candidate for removal once that check is done.
Orders columns as `[Time] + rc.mandatory_channels (present) + anything else`
— no re-derivation of values, this is a pure reorder-and-save of the exact
`df` `processed_data.csv` already writes. Writes
`preprocessed_for_famos_validation.csv` into `config.run_output_dir`,
called right after `processed_data.csv` is saved, gated behind the flag.
CLI: `--export-famos-validation-csv`, following the existing
`--deglitch`/`--remove-stops` `store_true` pattern.

**Validated:** ran the pipeline twice on identical synthetic CSV input
(butterworth mode, no filter, for speed) — once without the flag (confirmed
`preprocessed_for_famos_validation.csv` absent), once with it (confirmed
present). Row count matches the decimated/processed length exactly (5000 ==
5000, no re-sampling). Spot-checked two channels (`FL_Fx`, `RR_Fz`) with
`np.allclose` against `processed_data.csv`'s own values — exact match, as
expected since both files are written from the same in-memory `df`. Column
order confirmed `Time` first, then the platform's canonical force-channel
order.

## Step 3 — three workflow modes: preprocess / analysis / both (2026-08-31)

**Confirmed before writing any code:** grepped every caller of
`pipeline.run(` across the repo — the GUI (`gui/workers/pipeline_worker.py::
RunRequest.to_cmd`) only shells out to the CLI subprocess, and `gui/main.py`'s
frozen-exe dispatch calls `dtt.pipeline._cli()` and discards its result too.
**No in-process caller inspects `run()`'s return value**, so making it
workflow_mode-dependent (PPTX path for `"both"`/`"analysis"`, CSV path for
`"preprocess"`) was safe to do without touching any caller — still called
out explicitly as a real signature change, per instruction, not silently
assumed safe.

**`RunConfig.workflow_mode: str = "both"`** — validated in `__post_init__`
against the three allowed values. `"both"` untouched: the new
`if config.workflow_mode == "analysis": ... else: <today's exact code,
now indented one level> ...` branch means `"both"`'s code path is
byte-for-byte the same statements that ran before this round, just inside
an `else:`.

**`"preprocess"`**: runs stages 1–4 exactly as `"both"` does (same `else`
branch), through `processed_data.csv`/`raw_data.csv`/the item-2 export/
`run_provenance.json`, then an early `return processed_csv` **before** stop
removal or any analysis stage — no `stats_summary.json`, no `figures/`, no
PPTX.

**`"analysis"`**: a new, much smaller branch. Skips ingestion/FAMOS/
sanitization/filtering entirely. Loads `df = pd.read_csv(run_output_dir /
"processed_data.csv")` (fails loudly with `FileNotFoundError` if absent),
rebuilds `config.run_channels`/`config.tyre` via the exact same
`build_run_channels`/`get_preset`/`match_preset` calls the normal path
uses, assembles a reduced `metadata` dict (`rows`/`columns`/`duration_s`/
`sampling_rate_hz` re-derived from the loaded frame, `n_to_dan_applied`
from the study's stored provenance), and **re-runs `validate(df, metadata,
config)`** — confirmed by reading `validator.py` that this is a read-only
channel-presence/NaN check with no data mutation, so re-running it is
validation-of-what-we're-about-to-analyze, not "re-running preprocessing."
Then converges into the same stop-removal + stage-5-onward code every mode
shares. `processed_data.csv`/`raw_data.csv` are never re-opened for writing
in this branch — confirmed by hash **and** mtime comparison in validation
below.

**Required provenance guard** — new `dtt/provenance.py::check_stale_provenance`,
reusing `load_provenance` exactly and the same `famos.audit.library_versions()`/
`git_commit()` calls `build_provenance` already makes, compared against the
*currently running* code (not a second study, which is what
`compare_provenance` is shaped for). Three cases, all warn-don't-block, all
logged loudly (`logger.warning`, wrapped in a `====` banner so they can't
scroll past unnoticed) and all still let the run proceed since the user
explicitly asked for `"analysis"` mode:
1. **No `run_provenance.json` at all** — firmly in the warn path per
   explicit instruction, not a silent pass. Confirmed by deleting a study's
   provenance file and re-running in `"analysis"` mode — the warning fires
   and the report still marks itself analysis-only.
2. **Stored `famos_version`/git commit differs from the code now running.**
3. **Units look inconsistent** — runs `dtt.ingestion.loader.
   normalise_force_units`'s existing decade-error heuristic **read-only on
   a copy** of the loaded `df` (never applied to the real analysis frame —
   that would be exactly the forbidden second N-to-daN conversion) purely
   to flag a mismatch against the vehicle preset's expected Fz band.

**Report marking (required addition):** `report_builder.py::build_report`
gained an `analysis_only: bool = False` parameter. When `True`, both the
title slide's subtitle (`"...  |  ANALYSIS-ONLY MODE"`) and its info block,
**and** the closing "Engineering Conclusions" slide, carry an explicit
"no fresh preprocessing/validation pass" line — a reduced-metadata report
can't be opened and mistaken for a full run. `pipeline.py` passes
`analysis_only=(config.workflow_mode == "analysis")`.

**CLI:** `--mode {preprocess,analysis,both}` (default `both`). The
source-argument group (`--csv`/`--raw`/`--raw-files`/`--raw-folders`)
changed from argparse-`required=True` to manually validated: still required
for `"preprocess"`/`"both"` (clear `parser.error` if omitted), not required
for `"analysis"` (which needs `--study` instead, also enforced with a clear
`parser.error` rather than a traceback).

**Validated, all three modes on one real synthetic study:**
- `"preprocess"`: confirmed `processed_data.csv`/`raw_data.csv`/
  `run_provenance.json` written, **zero** files in `figures/`, no
  `stats_summary.json`, no `.pptx` — and the early-return path actually
  returns the CSV path, not a report path.
- `"both"`: 63 figures produced, PPTX returned — unchanged from every prior
  round's runs of `"both"` (implicitly, since the code path is identical).
- `"analysis"`: re-ran on the study `"both"` had just produced. Confirmed
  `processed_data.csv`'s **SHA-256 hash and mtime are unchanged** after the
  analysis-only re-run (never re-opened for writing). `stats_summary.json`
  values matched the original `"both"` run to ~1e-6 relative — the tiny
  remaining difference is exactly `CSV_FLOAT_FORMAT="%.6g"` round-trip
  precision loss (`processed_data.csv` only carries 6 significant figures;
  `"both"` mode computes stats on the full-precision in-memory frame,
  `"analysis"` mode necessarily computes them on the re-read, rounded
  values) — expected and inherent to reading back a persisted CSV, not a
  bug, and confirmed by inspecting the actual diffs (all ~1e-6–1e-7
  relative). Confirmed the PPTX is marked `ANALYSIS-ONLY` on both the title
  and conclusions slides. Confirmed the stale-provenance warning actually
  fires (missing-provenance case, the most likely real one) when re-run
  against a study with its `run_provenance.json` deleted, and that the run
  still completes and produces a correctly-marked report rather than
  failing. Confirmed clear, typed errors (not tracebacks) for: `"analysis"`
  mode on a study with no `processed_data.csv`; `"analysis"` mode with no
  `study_name`; an invalid `workflow_mode` value; and the CLI-level
  equivalents via `parser.error`.

## Step 4 — GUI: workflow mode picker + FAMOS-validation checkbox (2026-08-31)

Round 4's plan deliberately deferred a GUI picker for the mode/export flag
to the later GUI-restructure round, on the reasoning that both were CLI-only
and that restructure would need to happen anyway. After seeing Step 1's
restyle land but having no way to reach Steps 2/3 without a terminal, the
user asked for these two specifically to be added to the existing New Study
screen now, ahead of that restructure — a narrower ask than the full
restructure, so done as a contained addition to the existing page rather
than pulled forward as part of it.

**`gui/workers/pipeline_worker.py::RunRequest`** — two new fields, `mode:
str = "both"` and `export_famos_validation_csv: bool = False`. `to_cmd()`
appends `--mode <mode>` only when not `"both"` (keeps the emitted command
identical to before for every existing caller) and
`--export-famos-validation-csv` when set. The existing `--csv`/`--raw*`
source-selection block is now skipped entirely when `mode == "analysis"` —
that mode needs none of them, matching the CLI's own validation.

**`gui/pages/new_study_page.py::NewStudyPage`**:
- New "Workflow mode" combo (`Full run` / `Preprocess only` / `Analysis
  only`) at the top of the Data Source card, driving a new `_on_mode_changed`
  handler.
- New "Existing study" combo, shown only for Analysis-only, populated from
  `self.repo.list_studies()` filtered to ones with a `processed_data.csv`
  (reusing `StudyRepository`, not a new discovery mechanism) — picking one
  also mirrors its name into the Study name field for clarity.
- Selecting Analysis-only hides the source picker, live channel detection,
  and channel listing (none apply — no source is being read), and disables
  every filtering control (`filter_check`/`famos_check`/
  `famos_validation_check`/`deglitch_check`/`stops_check`/`cutoff_spin`/
  `order_spin`) since nothing gets re-preprocessed in that mode. Miner's
  exponent stays enabled — rainflow still runs and reads it in Analysis-only
  mode.
- New "Export FAMOS-validation CSV (temporary)" checkbox next to the
  existing FAMOS-recipe checkbox, wired straight to
  `RunRequest.export_famos_validation_csv`.
- `_on_start` builds the mode/export fields into the existing `common` kwargs
  dict; for Analysis-only it short-circuits before any source validation and
  requires a study to actually be selected (clear inline error otherwise) —
  every other kwarg (vehicle, miner, remove_stops, etc.) still flows through
  unchanged, since several of them (miner, remove_stops) still matter to the
  analysis stages that do run.

**Known, accepted limitation (not fixed this pass):** the Processing
screen's stage tracker (`pipeline_worker.py::_STAGE_MARKERS`) matches
`[1/9]`/`[3/9]`/`[4/9]` log markers to show "Data Ingestion"/"Sanitization"/
"Filtering" as in-progress. In `"analysis"` mode those stages don't run and
the pipeline logs a single `[1-4/9] Skipped` line instead, so those three
rows on the Processing screen will simply show as never-started rather than
skipped — an honest reflection of what happened, just not a
purpose-built "skipped" state. Not addressed here to keep this addition to
the two things asked for; worth a follow-up if it reads as confusing in
practice.

**Validated (headless PySide6, `QT_QPA_PLATFORM=offscreen`, `page.show()` so
Qt's real visibility chain is exercised, not the false-negative
never-shown-widget state):**
- Switching the mode combo to Analysis-only correctly shows the existing-study
  picker and hides the source picker/channel panel; switching back to Full
  run correctly reverses both, restores the filtering controls' enabled
  state, and re-shows the CSV/raw picker per the underlying `source_type`
  selection.
- `_on_start()` in Analysis-only mode with no study selected sets the
  expected inline error and does not emit `start_requested`.
- End-to-end with a real on-disk study (`20260828_121831`): selecting it and
  starting produced a `RunRequest` with `mode="analysis"`, the study name,
  and `to_cmd()` correctly omitting every `--csv`/`--raw*` flag while
  including `--mode analysis --study 20260828_121831`.
- Preprocess-only with the FAMOS-validation checkbox checked and a real CSV
  selected produced `to_cmd()` containing `--mode preprocess`,
  `--export-famos-validation-csv`, and `--csv`, as expected.

---

# Round 5: GUI surfacing of terminal-only features (2026-09-01)

Additive-only pass per the brief: no page reorganisation, no navigation
restructure, no backend/pipeline math touched. Every new control sets an
existing `RunConfig` field through a new CLI flag (pure argument-parsing
plumbing) — the recipe/analysis code itself is unchanged.

**Phase 1 audit (full table given to the user separately) found:** workflow
mode, the FAMOS-validation CSV export, and stop removal on/off + min-duration
were already GUI-reachable (Round 4). Genuinely terminal-only: despike
(rail/dropout/sub-width + net), the transient/20%-spike rule, the stop-removal
speed threshold (`stop_speed_kph`), the histogram x-axis range mode, and the
Welch PSD output (generated automatically but with no GUI viewer). Two-run
study comparison (`compare_studies`/`compare_rainflow`, `study_compare.py`)
had **no CLI entry at all**, only a Python API — the user chose to defer both
of those (and the stop-removal seam-blend params, an internal tuning knob) out
of this round entirely.

## Step A — despike + transient-spike controls (2026-09-01)

**The GUI has no in-process pipeline import — it shells out to `python -m
dtt.pipeline`** (`gui/workers/pipeline_worker.py::RunRequest.to_cmd()`), so
every RunConfig field without a CLI flag was invisible to the GUI regardless
of any GUI work. Confirmed by grep: `despike`/`despike_*`/`transient_despike*`
existed on `RunConfig` (Round 2) and were already threaded into
`apply_famos_recipe` via `signal_processor.py::apply_filter` (confirmed by
reading the existing `getattr(config, "despike", False)`-style kwargs), but
`dtt/pipeline.py::_cli()` had no matching `argparse` entries and `run()` had
no matching keyword parameters — the only way to set them was constructing
`RunConfig(...)` directly in Python.

**Fix:** `dtt/pipeline.py::run()` gained keyword parameters for `despike`,
`despike_rail_min_run`, `despike_dropout_max_run`, `despike_hw_cutoff_hz`,
`despike_net`, `transient_despike`, and its 4 params — each only added to the
`kwargs` dict (and so only overrides the `RunConfig` default) when not `None`,
same pattern the existing `sampling_rate`/`filter_order` overrides use.
`_cli()` gained the matching `--despike`, `--despike-rail-min-run`,
`--despike-dropout-max-run`, `--despike-hw-cutoff-hz`, `--despike-net`,
`--transient-despike`, `--transient-despike-pct`,
`--transient-despike-window-s`, `--transient-despike-noise-floor-mult`,
`--transient-despike-max-spike-frac` flags.

**Bug found and fixed as a direct byproduct, not a silent side effect:**
`MIGRATION_NOTES.md`'s own Round 2 open-items note flagged that
`--stop-min-s` was parsed but never reached `run()`'s kwargs (`run()` had no
`stop_min_s` parameter at all). Since this step was already adding the
sibling `stop_speed_kph` parameter to the same function, `stop_min_s` was
added alongside it and wired into `_cli()`'s `run(...)` call — closing that
open bug. Flagging explicitly: this changes real behaviour (the existing GUI
spinbox and `--stop-min-s` flag previously had **no effect**; both now do).

**GUI (`new_study_page.py`):** two new checkboxes in the Filtering card —
"Despike (rail / dropout / sub-hardware-width)" and 'Transient spike rule
("20% within 1s")' — both off by default (matches `RunConfig`). Each reveals
its own indented `QFormLayout` sub-box **only when checked** (`setVisible`,
not just `setEnabled`) per the user's explicit clutter requirement — the
default view is two clean checkboxes, not a wall of greyed-out spinboxes.
Every numeric field is pre-filled from the matching `RunConfig` default:
rail-run 3, dropout-run 5, HW cutoff 200 Hz, transient threshold 20%, window
1.0 s, noise-floor 10x, max-spike-frac 0.4. A "Speed threshold" spinbox
(default 1.5 km/h, matching `stop_speed_kph`) was added next to the existing
"Shortest stop to cut" spinbox, same enable/disable wiring. `RunRequest`
(`pipeline_worker.py`) gained the matching fields and `to_cmd()` entries,
following the existing `remove_stops`/`stop_min_s` pattern (only emit a flag's
value args when the parent flag is set).

**Validated (headless PySide6, `QT_QPA_PLATFORM=offscreen`, `page.show()`):**
- Every new spinbox's default value read back exactly matches the
  corresponding `RunConfig` default (200.0, 3, 5, 20.0, 1.0, 10.0, 0.4, 1.5) —
  confirmed by reading each widget's `.value()` immediately after construction.
- Both `despike_box`/`transient_box` start hidden; toggling each checkbox
  shows/hides its box; the default (fully untouched) screen's `to_cmd()`
  produces the exact same command line as before this step (`--csv ... --vehicle
  ... --miner 8.0`, no new flags) — confirms the default state reproduces
  today's behaviour exactly.
- Setting despike (HW cutoff 250), transient, stop-removal (speed 2.5 km/h)
  and driving `_on_start()` produced a `RunRequest` whose `to_cmd()` contained
  every set flag with its edited value: `--despike --despike-rail-min-run 3
  --despike-dropout-max-run 5 --despike-hw-cutoff-hz 250.0 --transient-despike
  --transient-despike-pct 20.0 ... --remove-stops --stop-min-s 5.0
  --stop-speed-kph 2.5`.
- Backend-level check (isolated from the GUI): called
  `apply_famos_recipe(df, despike_enabled=True, despike_hw_cutoff_hz=250.0,
  transient_enabled=True, ...)` directly on a real synthetic CSV
  (`generate_sample_data.py --profile mixed`) — 37/38 channels came back with
  `despike(...)`/`transient(...)` in their applied-steps string, confirming
  the flag path actually reaches the pipeline math, not just argument parsing.
- `remove_stops_frame` called directly with `moving_kph=1.5` vs `5.0` (same
  min_stop_s) on the same synthetic study removed a different row count
  (1011 vs 1111 of 20000), and `min_stop_s=3.0` vs `20.0` (same speed)
  dropped from 1011 to 0 — confirms both newly-wired parameters actually
  change stop-detection output, not just pass through inertly.
- `pytest tests/` — 37/38 pass, same single pre-existing failure as every
  prior round (missing gitignored proprietary file, unrelated).

## Step B — stop-removal speed threshold: covered by Step A

The `stop_speed_kph` CLI flag, `RunRequest` field, and GUI spinbox were built
and validated together with Step A above (same Filtering-card row, same
`stops_check` enable/disable wiring) since they're the same feature group as
the `stop_min_s` bug fix — no separate work needed.

## Step C — histogram x-axis range mode (2026-09-01)

**Issue, more terminal-only than the rest:** `range_mode` wasn't even a
`RunConfig` field — `dtt/pipeline.py`'s call to `generate_histograms(analysis_df,
config)` never passed it, so the pipeline was hardcoded to `"full"` no matter
what. Deliberately left this way in Round 3 pending this GUI round.

**Fix:** new `RunConfig.histogram_range_mode: str = "full"` field (matches
today's default exactly, so a study run without touching this control is
unaffected). `pipeline.py`'s histogram-generation call now reads
`generate_histograms(analysis_df, config, range_mode=getattr(config,
"histogram_range_mode", "full"))`. `_cli()` gained
`--histogram-range-mode {full,autoscale}`; `run()` gained the matching keyword.

**GUI, two places:**
- **New Study page**: a "Histogram x-axis" combo in the Filtering card
  (Full range / Autoscale to data), default "Full range" — matches the
  `RunConfig` default, stays enabled in every workflow mode (rainflow/
  histograms still run in Analysis-only mode, same as Miner's exponent).
- **Histograms page**: a new "X-axis:" combo in the toolbar, so a study that
  was run with **both** modes generated (or only one) can be viewed either
  way. `Study.histogram()`/`histogram_combined()` (`gui/models/repository.py`)
  gained a `range_mode` parameter that resolves to the `_autoscale`-suffixed
  filename or the unsuffixed original, mirroring `generate_histograms`'s own
  filename convention exactly (no second naming scheme invented).

**Graceful degrade (explicit requirement):** no new empty-state code was
needed — `ZoomableImageView.load()` already renders "No figure available"
for a missing path and `HistogramsPage`'s caption already appends
"(not generated)"; passing a `range_mode` whose file doesn't exist for this
study just flows into that same existing path.

**Validated:**
- `python -m dtt.pipeline --csv ... --histogram-range-mode full` and
  `--histogram-range-mode autoscale` on the same synthetic study: `full`
  produced the unsuffixed `hist_distance_FL_Fx.png` (byte-identical filename
  convention to every prior round); `autoscale` produced
  `hist_distance_FL_Fx_autoscale.png` — confirms the CLI flag reaches
  `generate_histograms` correctly.
- Headless: loaded a study that only has `_autoscale` files into
  `HistogramsPage` with the range combo left at its "Full range" default —
  caption correctly read `hist_distance_FL.png  (not generated)`, no crash,
  no broken image. Switching the combo to "Autoscale to data" correctly
  resolved and displayed the real file.
- Default New Study page state (nothing touched) still emits no
  `--histogram-range-mode` flag at all — `to_cmd()` unchanged from before this
  step for the untouched-defaults case.

## Step D — Welch PSD viewer page (2026-09-01)

**Issue:** `dtt/analysis/psd.py::generate_psd` already runs unconditionally
every full/analysis run (pipeline stage `[9b/9]`, wired in Round 3) and saves
`psd_{wheel}.png` into `figures_dir` — no toggle needed, nothing to surface
there — but no GUI page displayed it; the only way to see it was opening the
study folder manually.

**Fix:** new `gui/pages/psd_page.py::PsdPage`, structurally identical to
`BoxplotsPage` (a `FigureGrid` of per-item cards, `ImageViewerDialog` on
click) rather than `HistogramsPage`'s single zoomable view + combo selectors
— PSD has exactly one figure per wheel, no signal/kind/range axis to pick,
so the simpler existing pattern fit without inventing a new one.
`Study.psd(wheel)` added to `gui/models/repository.py`, mirroring
`boxplot()`/`rainflow_image()` exactly. Registered in `main_window.py`'s
`NAV_ITEMS`/`_STUDY_PAGES`/page mapping, inserted between Rainflow and
Reports (closest existing neighbours in the pipeline's own stage order) —
a single added nav row, no reordering of the existing thirteen.

**Empty state (explicit requirement), inherited for free:** `FigureGrid.
set_figures()` already renders a "Figure not generated" placeholder per card
when a path is `None` (the exact mechanism `BoxplotsPage` already relies on)
— `PsdPage` needed no new empty-state code, only to build the `(title, path)`
list the same way `BoxplotsPage` does.

**Validated (headless PySide6):**
- A preprocess-only study (`workflow_mode="preprocess"`, stops before any
  analysis stage — confirmed 0 files in its `figures/` dir) loaded into
  `PsdPage`: 4 cards rendered (one per default wheel), each showing "Figure
  not generated" text, no broken-image icon, no crash. Screenshot confirmed
  visually.
- A full-run study (`workflow_mode="both"`, real `psd_FL/FR/RL/RR.png` on
  disk) loaded into the same page instance: all 4 cards show a real loaded
  pixmap (`card._pixmap is not None`). Screenshot confirmed the actual Welch
  PSD subplots render correctly inside the existing dark Card styling.
- Full `MainWindow` construction + `navigate("psd")` succeeds end to end,
  confirming the new page is wired into the real navigation stack, not just
  buildable in isolation.

## Contrast / styling — all four steps

No manual styling was added to any new widget. `QCheckBox`, `QSpinBox`,
`QDoubleSpinBox`, `QComboBox`, `QFormLayout`, and `QLabel` are all styled
globally by `gui/theme.py`'s app-wide stylesheet (confirmed by reading it —
explicit checked-state accent colour, focus borders, entry backgrounds) —
every new control automatically matches existing ones with no bolted-on look.
Confirmed visually via headless screenshot grabs (`page.grab()` with the real
`theme.stylesheet()` applied, same as `gui/main.py`'s actual startup path):
checkbox accent colouring, spinbox/combo borders, and indentation of the
collapsible despike/transient sub-forms all render as expected against the
dark theme.

## Deferred, per explicit user decision

- **Two-run study comparison / RF Compare** (`dtt/analysis/study_compare.py`)
  — left entirely as Python-API-only. No CLI flag, no GUI entry point. Still
  open for a future round if wanted; the module itself is unchanged.
- **Stop-removal seam-blend params** (`stop_seam_search_s`,
  `stop_seam_blend_s`) — left at their validated 1.0s/0.2s defaults, no CLI
  flag, no GUI control. Internal backstop tuning, not an operator-facing knob
  per the user's explicit call.

## Not touched

No pipeline/analysis math was edited in this round — every change is CLI
argument parsing (new `argparse` entries setting existing `RunConfig` fields)
or GUI widget wiring (new controls setting existing `RunRequest` fields).
`famos/ops.py`, `dtt/preprocessing.py`'s despike/transient rule
implementations, `dtt/analysis/psd.py`'s Welch computation, and
`dtt/analysis/histograms.py`'s range-mode logic are all byte-for-byte
unchanged from before this round.

---

## Round 4 — small batch: default source, multi-session picker, last-used
## directory, analyzer boxplot styling (2026-09-03)

Four small, independent GUI/analysis changes, three of which (1, 2, 3) all
touch `gui/pages/new_study_page.py`'s new-study/add-dataset flow and were done
together; item 4 is a contained plot-styling port. No pipeline/report
structure changed in any of the four.

### Item 1 — IMC Raw folder as the default source

**File:** `gui/pages/new_study_page.py`, `source_type` combo construction.
**Before:** `QComboBox` defaulted to index 0 ("CSV file"). **After:**
`setCurrentIndex(1)` right after populating the combo, so "imc raw folder" is
selected on open. CSV remains fully selectable; `_on_source_type()` (the
existing show/hide + detection wiring) is untouched, it just runs against the
new initial index.

**Validated:** headless construction of `NewStudyPage` shows the raw-folder
picker visible and the CSV picker hidden immediately after `__init__`;
switching the combo back to index 0 flips both correctly (existing CSV path
unaffected).

### Item 2 — add multiple raw sessions in one picker

**File:** `gui/pages/new_study_page.py::_add_session`.
**Before:** `QFileDialog.getExistingDirectory` — native picker, one folder per
click, so N sessions needed N clicks.
**After:** a new `_select_multiple_dirs()` helper builds a non-native
`QFileDialog` (`FileMode.Directory`, `ShowDirsOnly`, `DontUseNativeDialog`)
and switches its internal `listView`/`treeView` (Qt's own object names for the
actual file-browsing views — the unrelated `sidebar` quick-access list is
deliberately left single-select) to `ExtendedSelection`, so ctrl/shift-click
picks several folders in one dialog. `_add_session` now extends
`_raw_sessions` with every returned path instead of appending one.

No pipeline change needed: `RunRequest.raw_folders` (a list) already flows
through to `--raw-folders` / `imc_reader`'s session join — confirmed by
reading `pipeline_worker.py` before touching anything, per the brief's
"picker-and-list change, not a pipeline change" framing. Single-folder
selection is the N=1 case of the same code path, not a separate branch.

**Validated:** a non-interactive drive of the real `QFileDialog` (populating
its `listView` model against three temp folders, then programmatically
multi-selecting all three rows via `QItemSelectionModel`) confirmed
`selectedFiles()` returns all three paths; clearing to a single-row selection
confirmed it still returns exactly one. The multi-select dialog is
necessarily the non-native Qt one rather than the OS picker — Qt has no
native multi-folder dialog — so it looks different from Windows' own folder
browser; noted and accepted as the known tradeoff for this feature.

### Item 3 — pickers remember the last-used directory

**File:** `gui/pages/new_study_page.py`, new module-level `_last_dir()` /
`_remember_dir()` helpers backed by `QSettings` (key
`new_study/last_dataset_dir`). No settings mechanism existed anywhere in the
app before this — confirmed by grepping `gui/` for `QSettings`/`last_dir` and
finding nothing — so this introduces the app's first persisted GUI setting,
using Qt's own built-in mechanism (no new dependency). `QSettings()` picks up
its storage location from `QApplication`'s existing
`setOrganizationName("Apollo Tyres")` / `setApplicationName("DTT WFT
Automation Platform")` in `gui/main.py`, so no extra wiring was needed there.

All four dataset pickers — `_browse_csv`, `_browse_raw`, `_browse_raw_files`,
`_add_session` — now open via `_last_dir(default)` (falls back to the
existing fixed defaults, `repo.csv_dir`/`repo.csv_dir.parent`, only if nothing
is stored yet or the stored path no longer exists) and call `_remember_dir()`
on a successful pick, storing the picked file's/folder's parent. One shared
key across raw and processed/CSV pickers, per the brief's "apply consistently
to every dataset-adding picker" instruction.

**Validated:** `_last_dir()`/`_remember_dir()` round-trip confirmed
interactively; falls back to the default when `QSettings` is empty or the
stored directory has been deleted since.

### Item 4 — boxplot styling ported from the analyzer script

**File:** `dtt/analysis/boxplots.py::generate_boxplots`.

**Scope decision (asked and confirmed before implementing):** the analyzer's
boxplot (`_open_box_popup` in
`WFT_Analyzer_All_AUC_Gxy_Heatmap_boxplot_PV_July26 (1).py`) is structurally
per-channel — one popup per individual channel, up to 24 for a 4-wheel study.
The platform's boxplot is per-force-type with all wheels grouped in one plot
(3 images: `boxplot_Fx/Fy/Fz.png`), and both `dtt/reporting/report_builder.py`
(fixed one-image-per-force-type slide insert) and `gui/pages/boxplots_page.py`
(one grid card per signal) depend on that shape. Switching to per-channel
images would have meant reworking both of those — a restructure the brief
explicitly said not to do ("the existing boxplot page/stage stays where it
is"). Confirmed with the user: kept the platform's grouped 3-image structure,
ported only the analyzer's drawing/styling onto it.

**Ported (now matches the analyzer's `_open_box_popup` exactly):**
- Box width `0.38` (was `0.5`); `medianprops`/`whiskerprops`/`capprops` now
  solid black (`#000000`, was white/`TEXT_SEC`) at the analyzer's linewidths;
  `flierprops` markersize `2.5`/alpha `0.35` (was `2`/`0.3`); `boxprops`
  linewidth `1.6` (was `1.5`).
- Axes face colour `#F5F5F5` (was pure `#FFFFFF`) via `_style_ax`.
- P5/Mean/P95 reference lines are now scoped per box (`axhline(..., xmin=,
  xmax=)` spanning only that wheel's box, colours `#1E8449`/`#B7950B`/
  `#C0392B`, linewidth `2.5`/`2.5`/`2.6`) instead of the old full-plot-width
  `axhline` — the analyzer's `x_lo/x_hi = (i-1+0.31)/n, (i-1+0.69)/n` formula
  is already written generically over `n` boxes, so it generalizes from the
  analyzer's 1-2 datasets to the platform's 4 wheels with no change to the
  formula itself.
- Stats footer text: exact analyzer format and content, including `Q1`/`Q3`
  (which the platform's old text omitted), right-aligned 12-char labels,
  `│`-separated fields, `fontsize=6.5`, black text, box `facecolor="#F0F0F0"`/
  `edgecolor="#AAAAAA"`/`alpha=0.95` (was `TEXT_SEC` text on `ENTRY_BG`/
  `#1B3A5C` at `0.92`).
- Title format `"Box Plot — {force_type} — All Wheels"` (analyzer's em-dash
  spacing), replacing the old `"Box Plot  –  {force_type}  (All Wheels)"`.

**Deliberately not ported (platform keeps ownership, per the brief):**
- `BOX_YLIMS` stays sourced from `dtt/config.py` (Fx/Fy ±1200 daN, Fz
  5000-13000 daN — this platform's actual vehicle-class ranges), not the
  analyzer's hardcoded `BOX_YLIM` (±300 daN / 100-1000 daN, tuned for the
  analyzer's own reference vehicle). Confirmed these were already two
  independent constants before this change; no accidental overwrite risk.
- Channel labels/colours stay on the platform's own `chan_colors`/`CHAN_COLORS`
  (per-wheel channel names from the platform's channel model), not the
  analyzer's hardcoded `FL_Fx`/`FR_Fx`/... string literals.
- The legend block was already byte-identical between the two codebases
  before this round (same author wrote both, evidently copy-pasted once) —
  including a pre-existing minor inconsistency in the analyzer itself (legend
  swatch colours for P5/Mean/P95/Whiskers don't exactly match the actual
  plotted line colours). Left as-is rather than "fixed", since the brief asks
  to match the analyzer's actual output, inconsistency included, not to
  improve on it.
- No N-to-daN conversion heuristic was ported — the platform's data is
  already in daN by the time it reaches this module, so the `"Force (daN)"`
  ylabel (unchanged) is already correct for the platform's own units.

**Cleanup:** removed the now-unused `ENTRY_BG` module constant (was only
referenced by the stats-box styling this replaced).

**Validated:** generated boxplots on synthetic 4-wheel Fx/Fy/Fz data (5000
samples/wheel) through the real `generate_boxplots(df, RunConfig)` path,
confirmed all three `boxplot_Fx/Fy/Fz.png` render with black median/whisker/
cap lines, per-box-scoped P5/Mean/P95 markers, and the new stats-table format
(screenshot-inspected). Rendered the analyzer's own `_open_box_popup` drawing
code standalone (matplotlib-only, no tkinter) on the same channel's data for
a side-by-side: box proportions, line colours/styles, and stats-line format
match field-for-field; the only visible difference is the platform's 4 grouped
boxes in one axes vs. the analyzer's one box per popup, which is the
confirmed, deliberate structural difference. `pytest tests -k "not golden and
not corpus"` re-run after all four items: 114 passed, 1 pre-existing failure
unrelated to this round (`test_seq_translation_of_the_real_imc_file` — missing
a local `.txt` fixture file, fails identically on a clean checkout before any
of these changes).

### Not touched

No changes to `dtt/pipeline.py`, `dtt/reporting/report_builder.py`,
`gui/pages/boxplots_page.py`, `gui/workers/pipeline_worker.py`, or any
preprocessing/FAMOS-recipe code in this round. `RunRequest.raw_folders`
handling, `dtt/config.py::BOX_YLIMS`, and the platform's channel-colour model
are all unchanged, reused as-is.

---

## Round 5 — boxplot Y-axis limits reset to measured data (2026-09-03)

**File:** `dtt/config.py::BOX_YLIMS`. Follow-on, same day, to the Round 4
boxplot-styling work above — the styling port made the axis-scaling problem
*visible* (a readable box needs headroom, not empty space), which is what
surfaced this bug in the first place.

**Problem, confirmed by measurement, not assumed:** the old fixed limits
(`Fx`/`Fy` ±1200 daN, `Fz` 5000-13000 daN) were never derived from real data.
Measuring the real whisker envelope (Q1/Q3 ± 1.5×IQR, the actual box+whisker
extent, not the raw min/max which includes the outliers that are *supposed*
to sit outside it) across every real `processed_data.csv` this project has
produced gave `Fx` −200…+182, `Fy` −80…+91, `Fz` 484…980 — a 6-16x mismatch
against the old limits. At the old `Fz` range the box was invisible (0%
of a rendered plot showed anything at all); at the old `Fx`/`Fy` range the
box occupied roughly 7-16% of the plot height.

**Data-quality finding made while measuring, before trusting any number:**
`dtt/outputs/20260807_003524/processed_data.csv` had to be excluded from the
measurement corpus. It carries no `run_provenance.json` (predates provenance
tracking) and its `Fz` channels sit ~10x every other study's (median ~7300-
7900 vs ~600-800 elsewhere) — this matches this project's own known,
previously flagged, still-open gap: "existing studies predate the unit-decade
fix… anything processed before that correction holds forces a decade high"
(§10 of `PROJECT_HISTORY.md`). Including it would have derived the new fixed
limits from the same class of stale data the fix is meant to stop showing.
After that exclusion and collapsing duplicate re-ingestions of the same
source recording (several output folders are the identical Round-3 mentor
dataset, re-run repeatedly during earlier validation), the measurement corpus
was 6 distinct real studies / 20 real WFT channels.

**New values** (measured whisker envelope + ~50% margin, rounded):

| type | old | measured envelope | new |
|---|---|---|---|
| `Fx` | ±1200 | −200…+182 | **(−300, 300)** |
| `Fy` | ±1200 | −80…+91 | **(−150, 150)** |
| `Fz` | 5000-13000 | 484…980 | **(300, 1100)** |

The `Fx` value independently landing on ±300 — the exact number already used
by `dtt/config.py::FORCE_RANGES_DAN` (the histogram "full range" x-axis span)
and by the analyzer script's own `BOX_YLIM` constant — is a useful cross-
check that this isn't an arbitrary pick.

**Deliberately kept fixed, not switched to autoscale:** the user's explicit
requirement — the same channel must stay visually comparable across
different studies' boxplots, so the axis can't float per-plot. Confirmed
after the change: two different real studies (`20260828_121831` and
`NEW_Pipeline`, visibly different box positions/widths — one study's boxes
sit near the old plot's P95 line, the other's don't) render at the identical
`Fx` (−300, 300) axis range.

**Other plots checked for the same class of bug, none found needing a fix:**
grepped all of `dtt/` for axis-range constants — only `boxplots.py`,
`histograms.py`, `heatmaps.py`, and `psd.py` have any. `FORCE_RANGES_DAN`
(histograms) and `FZ_HEATMAP_BINS_DAN`/`FXY_HEATMAP_RANGE_DAN`/
`HEXBIN_EXTENT_N` (heatmaps) were already close to the measured real-data
envelope, not crushed the way `BOX_YLIMS` was — consistent with the user's
note that histograms had already been fixed for this in an earlier round.
`psd.py`'s only limit, `(0, fs/2)`, is physically derived from the sample
rate, not a guessed constant. **Reported, not touched:** `heatmaps.py::
_comp_bins` can already take a per-vehicle `tyre.fx_range_dan`/etc. override
in addition to the global constants — a more precise mechanism than
`boxplots.py` has. Left alone as out of scope for this fix; worth knowing
about if boxplot ranges ever need to be vehicle-specific rather than one
fixed value per force type.

**Validated:**
- Rendered `Fx`/`Fy`/`Fz` boxplots on a real study (`20260828_121831`) at
  both the old and new limits side by side: at the new limits the box,
  median, quartiles, and P5/P95 markers fill ~60-65% of the plot height
  (were ~7-16%, and 0% for `Fz`); the far outlier whiskers correctly run off
  the top/bottom of the plot instead of compressing everything else.
- Confirmed the stats-footer text (`fig.text`, independent of `ax.set_ylim`)
  still prints the true, unclipped `Min`/`Max` regardless of what the axis
  shows — e.g. `FL_Fy` prints `Max: 266` even though the `Fy` axis clips at
  150.
- Regenerated `Fx` boxplots through the real (unmodified) `generate_boxplots`
  path on two different real studies (`20260828_121831`, `NEW_Pipeline`) and
  confirmed both used the identical `(-300.0, 300.0)` axis range despite
  visibly different box positions/spreads — the fixed-per-force-type
  property holds.
- `pytest tests -k "not golden and not corpus"` re-run after the change: 114
  passed, the same 1 pre-existing unrelated failure
  (`test_seq_translation_of_the_real_imc_file`, missing local `.txt`
  fixture) as before this change.

### Not touched

Only the three `BOX_YLIMS` numbers changed. No drawing/styling code in
`dtt/analysis/boxplots.py` (the Round 4 analyzer-look port) was touched, no
other module's range constants were changed, and no other plot type was
modified.
