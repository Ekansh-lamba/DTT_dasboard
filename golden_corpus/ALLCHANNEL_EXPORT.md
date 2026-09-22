# All-channel FAMOS validation — step by step

Written for someone who has **not used imc FAMOS before**. Every step says what
to click and what you should see afterwards, so you can tell a success from a
plausible-looking failure before moving on.

What this proves: that our Python processing reproduces what FAMOS produces,
on **every channel** — forces, moments, Latacc, speed — not the six force
channels the 2026-08-29 round covered. It closes the two gaps
`MODULE1_STATUS.md` records as *inferred rather than measured*.

Companion to [`VALIDATION_GUIDE.md`](VALIDATION_GUIDE.md), which covers the
synthetic operator corpus. This supersedes its part B: it scores the same
forces plus everything else.

**Budget about 40 minutes**, most of it FAMOS chewing through 5.5 million
samples and writing the export.

---

## Before you start

Open FAMOS and read the **title bar**.

- `FAMOS Reader …` — **stop here.** The Reader is free but view-only: it cannot
  run sequences, and there will be no **Sequence** menu. You need a licensed
  edition (`Enterprise`, `Professional` or `Standard`).
- Anything else — carry on.

---

## Step 1 — Stage the input files (in a terminal, not FAMOS)

```
python tools/stage_famos_inputs.py
```

This copies the 15 channels the sequence needs into
`golden_corpus/allchannel_in/` and checks each one reads. Originals are not
touched.

**Why this step exists.** The recorder's files are named `.raw`. They are
already FAMOS format inside, but **FAMOS decides what a file is from its
extension** — handed a `.raw` it concludes this is bus data and pops up *"Open
database for FR_Fx_2.raw"* asking for a **FlexRay Description File (\*.FRY)**,
which you do not have. Copying them to `.dat` is the entire fix. This is the
step that otherwise stops the whole job on its first click.

**Expect:** `15 of 15 channels copied as .dat`, each about 5.55 million samples
at 1000 Hz. A one- or two-sample spread between channels is the recorder
stopping them a moment apart and is fine.

---

## Step 2 — Open the channels in FAMOS

**File → Open → Data**, navigate to `golden_corpus/allchannel_in/`, select
**all 15 `.dat` files**, open.

They open directly — no import assistant, nothing to configure. The sample rate
and the channel name are stored inside each file.

**Check the Variables panel lists exactly these 15 names:**

```
FR_Fx_2  FR_Fy_2  FR_Fz_2  FR_Mx_2  FR_My_2  FR_Mz_2
RR_Fx_1  RR_Fy_1  RR_Fz_1  RR_Mx_1  RR_My_1  RR_Mz_1
AccelX   AccelY   Speed2D
```

The names come from **inside** the files, so they are these regardless of what
the files are called. If you instead see `Column1`, `Column2`… you went through
the ASCII import assistant — close everything and use **Open → Data**.

---

## Step 3 — Run the sequence

**File → Open → Sequence** (or the Sequence menu), pick
`golden_corpus/famos_allchannel.seq`, then press **F9** to run it.

**Do not** open the `.seq` through *File → Open → Data*. A sequence is a
program, not data; opened as data, FAMOS throws the ASCII import assistant at
it and tries to parse the comment lines as numbers.

Give it a few minutes — 5.5 million samples through a 99-tap kernel, 15 times.

**Check:** Variables goes from 15 to about 46. The new names end in `_smo`,
`_red`, `_rawred`, plus `Latacc`, `Latacc_LPF`, `Longacc`, `Vehicle_Speed` and
their `_red` forms.

**Two quick sanity checks that catch a bad run immediately:**

| check | why it matters |
|---|---|
| `FR_Fz_2_rawred` has **555,197** samples, and its first value equals `FR_Fz_2`'s first value exactly | `red` takes every tenth sample starting at the first — nothing should have shifted |
| `Latacc_LPF`'s first value equals `AccelY`'s first value | FAMOS initialises `FiltLP` from the step response. A value ~2400× smaller means the filter ran from zero state, and the comparison will be measuring the wrong thing |

---

## Step 4 — Export to CSV

Select **all** the result variables → **File → Save**.

| setting | value | why |
|---|---|---|
| File name | a concrete name, e.g. `allchannel` | **not** a wildcard. `*_red` looks reasonable and the Save button simply does nothing, with no error |
| Save as type | the **ASCII / CSV** option | not `.dat` — see the note below |
| Decimal places | **the maximum offered** | the default writes 6 significant figures, which is 1.0 of resolution once an `Fz` peak crosses 100,000. That rounding becomes the floor under every number in the report |
| One file | **yes** | not one file per channel |
| Save to | `golden_corpus/famos_out/` | where the scorer looks by convention |

**Which variables to select — the 31 decimated ones:**

| group | count | names |
|---|---|---|
| production output | 24 | `FR_*_red`, `RR_*_red` — forces **and** moments |
| unfiltered | 12 | `FR_*_rawred`, `RR_*_rawred` |
| Latacc | 3 | `Latacc_LPF_red`, `Latacc_red`, `Latacc_rawred` |
| Longacc | 2 | `Longacc_red`, `Longacc_rawred` |
| speed | 2 | `Vehicle_Speed_red`, `Vehicle_Speed_rawred` |

**Leave the un-decimated intermediates out** — `*_smo`, `Latacc_LPF`, `Latacc`,
`Longacc`, `Vehicle_Speed`. Those are still at 1000 Hz. FAMOS writes every
channel onto one shared x-axis, so mixing rates in one file writes each
decimated channel *sparsely* — one value every tenth row with blanks between —
which multiplies the file by ten for no extra evidence.

Expect roughly **400 MB** and 555,197 data rows.

> **Why CSV and not `.dat`.** FAMOS 2026 writes `.dat` in the newer **imc3**
> container, which our reader does not parse — it reads the classic `|CF`
> layout the recorder writes. The channel names decode from an imc3 file but
> the sample layout does not.

---

## Step 5 — Score it (back in the terminal)

```
python tools/score_golden.py golden_corpus/famos_out/allchannel.csv ^
       --raw-dir "raw data in .dat format\2006-08-25 09-10-46 (1)" ^
       --all-channels ^
       --report golden_corpus/famos_allchannel_report.csv
```

Note `--raw-dir` points at the **original** recording folder, not the staged
copies: FAMOS read its copy, we read ours, and the comparison is of the same
bytes through two different readers.

Writes the per-channel table to `..._report.csv` and the reasons beside it as
`..._report.md`. The same check is in the app: **Preprocess → Validate vs
FAMOS…**, which will offer to take the recording folder as well.

If you cut the recording in the sequence rather than running it whole, pass the
same range as `--cut A:B` so our side cuts before conditioning too — otherwise
`FiltLP`'s start-up and `smo`'s edge padding begin on different samples on the
two sides.

---

## Step 6 — Read the result

| row | expect |
|---|---|
| `rawred` | **0.000e+00**, exactly |
| `red` | `err/q` ≈ 0.5, `r = 1.000000000` |
| `Latacc_LPF_red` | same — this is `FiltLP` answering for itself |
| `Latacc_red` | same — `FiltLP` and `smo` together |

**`rawred` is the one to look at first.** It runs `red(10)` with no filtering at
all, so it compares nothing but the samples each side pulled out of the file.
There is no rounding excuse available — both sides are copies of stored values.
**0.000e+00 means our imc parser is byte-exact against FAMOS.** Anything else
points at the reader or at `red`'s decimation phase, and no amount of operator
work would fix it.

**Do not read the raw error numbers as failures.** FAMOS writes 6 significant
figures, which is a different absolute precision at every magnitude: 0.1 at
79,000 but 1.0 once an `Fz` peak crosses 100,000. The **`err/q`** column is the
error as a fraction of *that sample's own* quantum, and **≤ 0.5 is the rounding
floor** — agreement as close as the file is capable of recording. A fixed
absolute tolerance reports healthy channels as failing while `r` reads
1.000000000.

The project's bar is **≥ 95 % match = FAMOS-grade**. The force chain currently
clears it at `err/q` 0.52–0.55 with 98–99.5 % of samples inside the floor.

### It will also tell you what it did *not* compare

Deliberately, and with the reason — a silent omission looks exactly like a pass:

- **passthrough channels** (`PosLat`, `Yawrate`, `AngleRoll`, `VelForward`, …)
  — the imc recipe changes nothing about them, so there is no FAMOS operation
  to check.
- **`- Copy` duplicates** — the same channel under a name that cannot be
  matched back to a wheel.
- **`Speed_kmph.raw`** — the recipe derives it as `Speed2D * 3.6`, so it is
  scored through `Vehicle_Speed` rather than twice.
- **the `smo` and `lpf` stages** — one line each, because this export is
  deliberately the decimated production output.

---

## If something goes wrong

| symptom | cause |
|---|---|
| *"Open database for … .raw"* asking for a `.FRY` file | you opened the `.raw` originals. Run step 1 and open the `.dat` copies |
| Variables named `Column1`, `Column2`… | the file went through the ASCII import assistant. Use **File → Open → Data** |
| No **Sequence** menu | FAMOS Reader edition — it cannot run sequences |
| The Save button does nothing | the file name is a wildcard. Type a concrete name |
| Sequence errors on its first line | it was opened as *data* instead of as a sequence |
| `err/q` in the thousands on a `_red` row | the decimated channel was written sparsely — the 1000 Hz intermediates were included in the same export file |
| Report says a channel is "absent from the export" | that stage was not selected at save time, or the variable was not created |

---

## What is **not** in this comparison, and why

The scoring path is `.raw` → `famos.ops` → your export, **in the file's own
units**. Three things `processed_data.csv` additionally does are therefore
absent from it by construction, and the report names them per channel rather
than folding them in:

1. **N→daN and the CR decade correction** — forces only. FAMOS emits 79,120 N;
   we emit 791 daN.
2. **`blank_dead_runs`** — a frozen-value sensor dropout becomes NaN; FAMOS
   keeps the zeros.
3. **Stop removal**, when enabled — and it runs *after* `processed_data.csv` is
   written, on the analysis frame only.

All three are corrections FAMOS does not make. Folding them in here would score
our unit and dropout policy rather than our signal processing, and a mismatch
could not be attributed to either. Diffing `processed_data.csv` directly
against a FAMOS export will show all three as large differences — they are
improvements, not errors.
