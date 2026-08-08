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
