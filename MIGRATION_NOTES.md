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
