# Module 1 — status and validation results

Where the WFT processing pipeline stands, what has been measured against imc
FAMOS, and what has not.

For **how to reproduce the validation**, see
[`golden_corpus/VALIDATION_GUIDE.md`](golden_corpus/VALIDATION_GUIDE.md).

---

## Headline

**The FAMOS signal conditioning is reproduced to the limit of measurement**, on
real WFT data, verified against a licensed imc FAMOS.

| what | result |
|---|---|
| imc file reader vs FAMOS | **0.000e+00** — byte-exact, 548,783 samples |
| Force chain `smo(0.1)` → `red(10)` | **r = 1.000000000**, 5.5 M samples × 6 channels |
| Worst disagreement | **0.54 × one stored digit** of the export |
| Operator corpus (`FiltLP`, `smo`, `red`) | **33 / 33**, two exact at 0.000e+00 |
| Project's own bar (`PASS_THRESHOLD`) | 95 % — cleared by a wide margin |

The residual is the CSV export's own rounding, not our error. FAMOS writes six
significant figures; we agree as closely as the file is capable of recording.

---

## What was validated, and how

### The reader

`_rawred` runs `red(10)` with **no filtering at all**, so it compares nothing but
the samples each side pulled out of the same file on disk.

```
FR_Fz_2_rawred   max err 0.000e+00   r = 1.000000000
RR_Fz_1_rawred   max err 0.000e+00   r = 1.000000000
```

This is the result that matters most, because the reader is reverse-engineered:
byte offsets found heuristically, `|RC5` event records stripped, leading and
trailing samples trimmed. Perfect operators on misread samples would still give
wrong answers. They do not — FAMOS and our parser return identical bytes.

### The force chain

`smo(0.1)` then `red(10)`, and **no `FiltLP`** — that operator applies to
`Latacc` alone. This is exactly what a force column in `processed_data.csv` is.

All 14 captured channels at `r = 1.000000000`, worst error 0.54 quantum.

### The operators

A synthetic corpus (impulse, step, DC, chirp, noise, three sines) through
`FiltLP`, `smo` and `red` in isolation and in combination: **33 / 33**.

`gc_impulse_redonly` — one spike at a known index through `red()` alone —
matched at **0.000e+00**, which fixes the decimation phase outright.
`gc_dc_smoonly` likewise exact.

This also settled a disagreement between two earlier reference exports about the
`smo` kernel. Every previous measurement ran `smo` *after* `FiltLP`, so the
filter's response contaminated it; the corpus runs `smo` alone on an impulse,
and an impulse response **is** the kernel. The exact triangular form is correct.

---

## Reading the numbers

A fixed absolute tolerance is meaningless here and will report healthy channels
as failures. Six significant figures is 1e-6 on a corpus signal of order 1, but
only **0.1** on a wheel force of order 1e5 — and **1.0** once an `Fz` peak
crosses 100,000.

Scoring is therefore done per sample against **that sample's own quantum**. The
`err/q` column is the error as a fraction of it, and **≤ 0.5 is the rounding
floor**. Every force channel lands at 0.54.

---

## What is *not* covered

`processed_data.csv` is the FAMOS chain **plus corrections FAMOS does not make**.
It is not expected to be byte-identical to a FAMOS export, and anyone diffing the
two files will see large differences that are improvements rather than errors.

1. **N→daN and the decade correction.** FAMOS emits 79,120 N; we emit 791 daN.
   The recording's `CR` block scales stored counts by 10, which left the frame a
   decade high — a 6,841 kg static front wheel on a 2.4 t car, and 0 % of `Fz`
   samples inside every configured histogram range. Corrected against the vehicle
   preset and confirmed independently by physics: 684 kg per wheel static, and a
   0.27 m loaded radius from the `My/Fx` ratio.
2. **`blank_dead_runs`.** A frozen-value sensor dropout becomes NaN; FAMOS keeps
   the zeros. `RR_Fz_1` dies 174 s early, which had been inflating its σ to 1526
   against 803 for the live front channel.
3. **Stop removal**, when enabled.

**Moment channels (`Mx/My/Mz`) are inferred, not measured.** They take the
identical `smo(0.1)` path as the forces, so the risk is low — but that is
inference. Six extra lines in `famos_forces.seq` would close it.

**`Latacc` on real data is inferred.** `FiltLP` + `smo(0.5)` is confirmed on
synthetic signals; the real-recording pair was not captured. Lower priority,
since `Latacc` is not the deliverable.

---

## Faults found and fixed along the way

These were live defects in the pipeline, not hypotheticals:

| fault | effect |
|---|---|
| FAMOS recipe never reached the force channels | `is_wft_channel` matched only the literal two-axle names, so `FR_Fx_2` / `A1L_Fz` got no `smo`; `red(10)` then aliased 1000 → 100 Hz unfiltered. Max sample step 7929 daN, against 1469 with the recipe applied |
| Forces a decade high | see above |
| `processed_data.csv` written before conditioning | the file on disk was the *unfiltered* frame while every statistic used the filtered one |
| No recoverable "before" | conditioning is destructive and runs at the native rate; `raw_data.csv` now preserves it |
| Sensor dropouts counted as measurements | `RR_Fz_1` mean 7104 against a median of 7336, σ 1526 → 734 once blanked |
| Stops counted as road load | a parked wheel still carries static `Fx`; `|Fx/Fz|` is 0.667 stopped against 0.026 driving, and 5 % standstill inflated `Gx` from 0.045 to 0.116 |

---

## Feature status

| | |
|---|---|
| imc raw ingestion (folder, file subset, **multi-session**) | done |
| FAMOS recipe (`smo`, `FiltLP`, `red`) | done, validated |
| Force unit normalisation | done |
| Dropout blanking | done |
| Stationary/stop removal | done — CLI, GUI, pipeline stage |
| Session joining (canonical channel matching, scale alignment) | done |
| Statistics, histograms, heatmaps, boxplots, rainflow | done |
| G-severity (Gx, Gy, Gxy) and DLC | done |
| AUC distribution comparison | done |
| Preprocess screen (raw vs sanitized, span control, channel list) | done |
| PowerPoint report | done |

---

## Notes for whoever runs this next

**Existing studies predate the fixes.** Anything processed before the unit
correction holds forces a decade high. Re-run rather than compare against them.

**Use `--remove-stops` when the output describes the road.** Severity,
histograms and rainflow are all distorted by standstill. A run that keeps its
stops now says so in the log.

**Sessions are joined in the order you add them** when the recordings carry no
imc timestamp — as these do not. Check the list before launching.

**Joining sessions from different vehicles is not prevented.** A clean decade
difference in static `Fz` is treated as calibration and corrected; anything else
warns and joins unscaled. Two of the sample folders sit 35 % apart in wheel load,
which is load state, not calibration — mechanically fine to join, questionable as
engineering.
