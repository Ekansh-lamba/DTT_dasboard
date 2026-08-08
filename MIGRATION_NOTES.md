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
