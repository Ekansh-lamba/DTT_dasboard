# All-channel FAMOS export — what to run, and what to export

Closes the two gaps `MODULE1_STATUS.md` records as **inferred rather than
measured**: the moment channels (`Mx/My/Mz`) and `Latacc` on a real recording.
Everything here is one FAMOS session and one exported CSV.

Companion to [`VALIDATION_GUIDE.md`](VALIDATION_GUIDE.md), which covers the
synthetic operator corpus (A) and the six-channel force capture (B). This is C,
and it supersedes B: it scores the same forces plus everything else.

---

## 0. What you need

| | |
|---|---|
| FAMOS | a **licensed** edition — the Reader has no Sequence menu |
| sequence | [`famos_allchannel.seq`](famos_allchannel.seq) |
| recording | `raw data in .dat format\2006-08-25 09-10-46 (1)\` |

**That folder, specifically.** It is the classic imc `|CF` layout, which our
`read_famos` parses. The `Raw data_no sanitisation\` recording has all four
wheels but is the newer `imc3` container, which FAMOS opens and our
`read_famos` does not (PROJECT_HISTORY §10) — so the comparison would have a
FAMOS side and no Python side. Verified present and readable in this folder, at
1000 Hz × 5,551,969 samples each:

```
FR_Fx_2  FR_Fy_2  FR_Fz_2  FR_Mx_2  FR_My_2  FR_Mz_2
RR_Fx_1  RR_Fy_1  RR_Fz_1  RR_Mx_1  RR_My_1  RR_Mz_1
AccelX   AccelY   Speed2D
```

---

## 1. Load the channels

**File → Open → Data**, select the **15 files above**.

**Do not load the `- Copy` duplicates.** `FR_Fx_2 - Copy.raw` and the others are
the same channel under a name nothing can be matched back to a wheel; the
scorer skips them and says so.

**Check:** the Variables panel lists exactly those 15 names, each 5,551,969
samples at dx = 0.001 s. If a channel arrives named `Column1` you opened it
through the ASCII import assistant instead — close it and use **Open → Data**.

---

## 2. Run the sequence

**Sequence → Open**, pick `famos_allchannel.seq`, run it.

It produces 31 result channels. Two checks that catch a bad run immediately:

- `FR_Fz_2_rawred` must have **555,197** samples (5,551,969 ÷ 10, rounded down)
  and its first value must equal `FR_Fz_2`'s first value exactly. `red` takes
  every tenth sample starting at the first, so nothing should have moved.
- `Latacc_LPF`'s first value must equal `AccelY`'s first value. FAMOS
  initialises `FiltLP` from the step response, not from zero state — if you
  see a value ~2400× smaller, the filter ran zero-state and something is
  configured differently from the recipe.

---

## 3. Export

**File → Export**, ASCII / CSV, **one file, all 31 selected variables**.

Select these and nothing else:

| group | count | names |
|---|---|---|
| production output | 24 | `FR_*_red`, `RR_*_red` (forces **and** moments) |
| unfiltered | 12 | `FR_*_rawred`, `RR_*_rawred` |
| Latacc | 3 | `Latacc_LPF_red`, `Latacc_red`, `Latacc_rawred` |
| Longacc | 2 | `Longacc_red`, `Longacc_rawred` |
| speed | 2 | `Vehicle_Speed_red`, `Vehicle_Speed_rawred` |

**Leave the un-decimated intermediates out** (`*_smo`, `Latacc_LPF`, `Latacc`,
`Longacc`, `Vehicle_Speed`). They are 1000 Hz. FAMOS writes every channel onto
one shared x-axis, so mixing rates in one file writes each decimated channel
*sparsely* — one value every tenth row, blanks between — which multiplies the
file by ten for no extra evidence. The scorer can compact that back, but not
creating it is better.

### Dialog settings that matter

| setting | value | why |
|---|---|---|
| separator | **comma** | the scorer sniffs `,` `;` and tab, so any of them works — comma is just the one already proven |
| decimal places | **the maximum offered** | FAMOS's default writes 6 significant figures, which is 1.0 of resolution once an `Fz` peak crosses 100,000. That rounding is the floor on every number in the report; do not accept it by default |
| header | **channel names row on** | the scorer reads names from it |
| units row | either | it is detected and skipped automatically |
| one file | **yes** | not one file per channel |

Expect roughly **400 MB** and 555,197 data rows.

---

## 4. Score it

```
python tools/score_golden.py <your_export.csv> ^
       --raw-dir "raw data in .dat format\2006-08-25 09-10-46 (1)" ^
       --all-channels ^
       --report golden_corpus/famos_allchannel_report.csv
```

Writes the per-channel table to `..._report.csv` and the reasons alongside it
as `..._report.md`. The same check is reachable from the GUI: **Preprocess →
Validate vs FAMOS…**.

If the sequence cut the recording rather than running it whole, pass the same
range as `--cut A:B`. Our side then cuts before conditioning too — otherwise
`FiltLP`'s start-up and `smo`'s edge padding begin on different samples and the
first half-window disagrees for a reason that has nothing to do with the
operators.

### Reading the result

Every number is scored against **that sample's own export quantum**, never a
fixed absolute tolerance. Six significant figures is 1e-6 at order 1 and 1.0 at
1e5, so one fixed tolerance would report healthy channels as failing while
`r = 1.000000000`. `err/q ≤ 0.5` is the rounding floor.

| row | expect |
|---|---|
| `rawred` | **0.000e+00**, exactly. No rounding excuse exists — both sides are copies of stored samples. Anything else is the reader or `red`'s phase |
| `red` | `err/q` ~0.5, `r = 1.000000000` |
| `Latacc_LPF_red` | same — this is `FiltLP` answering for itself |
| `Latacc_red` | same — `FiltLP` + `smo` together |

The project's bar is **≥ 95 % match = FAMOS-grade**; the force chain currently
clears it at `err/q` 0.52–0.55 with 98–99.5 % of samples inside the floor.

### What it will tell you it did *not* compare

Deliberately, and with the reason:

- **passthrough channels** (`PosLat`, `Yawrate`, `AngleRoll`, `VelForward`, …)
  — the imc recipe changes nothing about them, so there is no FAMOS operation
  to check.
- **`- Copy` duplicates** — same channel, unmatchable name.
- **`Speed_kmph.raw`** — the recipe derives it as `Speed2D * 3.6`; it is scored
  through `Vehicle_Speed` instead of twice.
- **the `smo` and `lpf` stages** — grouped into one line each, because this
  export is deliberately the decimated production output.

### What is *not* in this comparison, and why

The scoring path is `.raw` → `famos.ops` → your export, **in the file's own
units**. Three things `processed_data.csv` does that FAMOS does not are
therefore absent from it by construction, and the report names them per
channel rather than folding them in:

1. **N→daN and the CR decade correction** — forces only. FAMOS emits 79,120 N;
   we emit 791 daN.
2. **`blank_dead_runs`** — a frozen-value sensor dropout becomes NaN; FAMOS
   keeps the zeros.
3. **Stop removal**, when enabled — and it runs *after* `processed_data.csv` is
   written, on the analysis frame only.

All three are corrections. Folding them into this comparison would score our
unit and dropout policy rather than our signal processing, and a mismatch
could not be attributed to either. Diffing `processed_data.csv` against a
FAMOS export directly will show all three as large differences; they are
improvements, not errors.
