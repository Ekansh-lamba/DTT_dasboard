# Validating the pipeline against imc FAMOS

How to reproduce, on your own machine and your own FAMOS licence, the check that
our Python processing matches FAMOS.

There are two validations and they answer different questions:

| | what it proves | needs |
|---|---|---|
| **A. Operator corpus** | `FiltLP`, `smo` and `red` are arithmetically correct | `famos_golden.seq` |
| **B. Force chain** | our **imc file reader** *and* the operators agree on a real recording | `famos_forces.seq` |

**B is the one that matters for "does our processed output match FAMOS".** Every
signal in A is generated in Python and merely imported, so our file reader is
never under test there. Do B if you only do one.

Expected results are given at each step, so you can tell success from a
plausible-looking failure.

---

## 0. Before FAMOS: check you have the licensed product

Open FAMOS and read the **title bar**.

- `FAMOS Reader 2026` — **stop**. The Reader is free but view-only; it cannot run
  sequences. There will be no **Sequence** menu. Activate a licence
  (imc LICENSE Manager → *Activation of Trial Versions via Internet*), then
  restart FAMOS.
- `FAMOS Enterprise + AI 2026`, `Professional`, `Standard` — good, carry on.

---

# A. Operator corpus

## A1. Generate the test signals

```
python tools/make_golden_inputs.py
```

Writes `golden_corpus/famos_golden_inputs.csv` — 120,000 samples at 1000 Hz, of
8 signals with known responses (impulse, step, DC, chirp, noise, three sines).
It is gitignored because it is 15 MB, and it regenerates byte-identical from a
fixed seed, so this is safe to run any time.

## A2. Convert them to native FAMOS files

```
python tools/csv_to_famos.py golden_corpus/famos_golden_inputs.csv \
       --out golden_corpus/famos_in --dx 0.001
```

**Do this rather than importing the CSV.** FAMOS's ASCII import assistant is
where this process stalls: it remembers markers from whatever you opened last
("Marker must not point behind the file!"), it defaults the sampling interval to
1 s rather than 0.001, and it will happily name your channels `Column1…Column9`,
after which the sequence fails on its first line.

Native `.dat` files carry the sample rate and the channel name inside them, so
there is nothing to configure and nothing to get wrong.

Expect 8 files and `8/8 verified`.

## A3. Load them

**File → Open → Data**, select all 8 files in `golden_corpus/famos_in/`.

**Check:** the Variables panel lists exactly

```
gc_impulse   gc_step   gc_dc   gc_chirp
gc_noise     gc_sine_2p5   gc_sine_5p0   gc_sine_10p0
```

Watch `gc_sine_2p5` — underscore before the `2p5`. The sequence uses *both*
spellings deliberately (input `gc_sine_2p5`, output `gc_sine2p5_lpf`), so it is
easy to mistype.

Two channels are unmistakable and confirm the whole set: **`gc_dc` is flat at
1.0**, and **`gc_impulse` is a single spike at t = 10 s**.

## A4. Run the sequence

Open `golden_corpus/famos_golden.seq` and run it (F9).

**Not** via *File → Open → Data*. A `.seq` is a program, not data; opening it
that way puts FAMOS into the ASCII import assistant trying to parse comment
lines as numbers.

**Check:** Variables goes 8 → about 48, with names ending `_lpf`, `_smo`,
`_red`, `_smoonly`, `_smo01`.

## A5. Export

Select every `gc_*` variable → **File → Save**.

- **File name:** a concrete name such as `golden` — **not** a wildcard.
  `gc_*` looks reasonable and the Save button simply does nothing.
- **Save as type:** the **ASCII / CSV** option.

Save to `golden_corpus/famos_out/`.

> Why CSV and not `.dat`: FAMOS 2026 writes `.dat` in the newer **imc3**
> container, which our reader does not parse (it reads the classic `|CF` format
> the recorder writes). The channel *names* decode from it but the sample layout
> does not. CSV is lossless enough here — the corpus signals are of order 1 and
> FAMOS writes 6 significant figures, so ~1e-7, four orders below what we are
> measuring.

## A6. Score

```
python tools/score_golden.py golden_corpus/famos_out/golden.csv
```

**Expected:**

```
33 at or below the export's precision, 0 above
gc_impulse_redonly    max err  0.000e+00   r = 1.000000000
gc_dc_smoonly         max err  0.000e+00   r = 1.000000000
everything else       max err ~5e-07       r = 1.000000000
```

`~5e-7` is the CSV's own precision on values of order 1 — the operators agree to
the limit of what the file can express, not to 5e-7.

---

# B. Force chain on a real recording

This is the important one.

## B1. Copy the raw channels to a `.dat` extension

```
python - <<'EOF'
from pathlib import Path
import shutil
src = Path("raw data in .dat format/2006-08-25 09-10-46 (1)")
dst = Path("golden_corpus/forces_in"); dst.mkdir(parents=True, exist_ok=True)
for n in ["FR_Fx_2","FR_Fy_2","FR_Fz_2","RR_Fx_1","RR_Fy_1","RR_Fz_1"]:
    shutil.copyfile(src / f"{n}.raw", dst / f"{n}.dat")
    print(n)
EOF
```

The `.raw` files are already FAMOS format inside (`|CF,2,1,`), but FAMOS
dispatches on the **extension**: given `.raw` it decides this is bus data and
asks *"Open database for FR_Fx_2.raw"* with a **FlexRay Description File
(*.FRY)** filter. Renaming to `.dat` is the whole fix. Originals are untouched.

## B2. Load and run

**File → Open → Data** on the six `.dat` files in `golden_corpus/forces_in/`.
They open directly and arrive named `FR_Fx_2`, `FR_Fz_2` and so on — the name
comes from inside the file, not the filename.

**Check:** 6 variables, each about 5,551,969 samples at 1000 Hz.

Then open and run `golden_corpus/famos_forces.seq` (F9). It applies the **force**
path — `smo(0.1)` then `red(10)`, and **no FiltLP**; that operator belongs to
`Latacc` alone, and putting a force channel through it would validate something
the pipeline never does.

**Check:** Variables 6 → 20. Give it a minute — 5.5 M samples through a 99-tap
kernel, six times.

*Short of memory or time?* Load only the three `FR_` channels and comment out the
`RR_` lines with `;`. **`Fz` matters most** — it carries the static wheel load,
so an error there moves every severity and histogram figure.

## B3. Export as CSV

Same as A5: select all, concrete filename, ASCII/CSV, into
`golden_corpus/famos_out/`. It will be large — about 1.2 GB.

## B4. Score

```
python tools/score_golden.py golden_corpus/famos_out/<yourfile>.csv \
       --raw-dir "raw data in .dat format/2006-08-25 09-10-46 (1)"
```

Note `--raw-dir` points at the **original** folder: FAMOS reads its copy, we read
ours, and the comparison is of the same bytes through two readers.

**Expected:**

```
14 at or below the export's precision, 0 above

FR_Fz_2_rawred    0.000e+00   r = 1.000000000   err/q 0.00
RR_Fz_1_rawred    0.000e+00   r = 1.000000000   err/q 0.00
all _smo / _red   ~5e-2 .. 5e-1  r = 1.000000000  err/q ~0.54
```

## How to read that

**`_rawred` is the reader test.** It runs `red(10)` with no filtering at all, so
it compares nothing but the samples each side pulled out of the file.
**0.000e+00 means our imc parser is byte-exact against FAMOS** — the byte
offsets, the `|RC5` event-record stripping, the leading/trailing trim.

**`_smo` and `_red` are the production chain** — literally what a force column in
`processed_data.csv` is.

**Do not read the raw error numbers as failures.** FAMOS writes 6 significant
figures, which is a different absolute precision at every magnitude: 0.1 at
79,000 but 1.0 once an `Fz` peak crosses 100,000. The **`err/q`** column is the
error as a fraction of that sample's own quantum, and **≤ 0.5 is the rounding
floor** — agreement as close as the file can record. A fixed absolute tolerance
reports these as failures while `r = 1.000000000`, which is misleading.

---

## What this does and does not cover

**Covered:** the imc reader, and `smo` / `FiltLP` / `red`.

**Not covered — deliberately.** `processed_data.csv` is the FAMOS chain *plus*
corrections FAMOS does not make, so it is not expected to be byte-identical to a
FAMOS export:

1. **N→daN and the decade correction.** FAMOS emits 79,120 N; we emit 791 daN.
   Justified separately by physics — corner weight 684 kg/wheel, and a 0.27 m
   loaded radius from the `My/Fx` ratio.
2. **`blank_dead_runs`** — a frozen-value sensor dropout becomes NaN; FAMOS keeps
   the zeros.
3. **Stop removal**, when enabled.

Anyone diffing the two files will see large differences from these three. They
are improvements, not errors.

**Also not yet measured:** the moment channels (`Mx/My/Mz`). They take the
identical `smo(0.1)` path as the forces so the risk is low, but that is inference
rather than measurement. To close it, add six lines to `famos_forces.seq`
mirroring the force block.

---

## Traps, in one list

| symptom | cause |
|---|---|
| No **Sequence** menu | running FAMOS **Reader**; licence not active |
| *"Open database for … .FRY"* | `.raw` extension; copy to `.dat` |
| *"Marker must not point behind the file!"* | ASCII assistant on the wrong file, or stale markers |
| Channels named `Column1…N` | **Names etc.** tab not set to take names from row 1 |
| Sequence fails on line 1 | a channel name mismatch — check `gc_sine_2p5` |
| **Save** button does nothing | wildcard in the filename; use a concrete name |
| Everything reads as a failure at r = 1.0 | absolute tolerance on large values; read **err/q** |
| `_red` channels all fail at r ≈ 0 | decimated channels are written *sparsely* on the shared x-axis — the scorer compacts them, so this should not recur |
