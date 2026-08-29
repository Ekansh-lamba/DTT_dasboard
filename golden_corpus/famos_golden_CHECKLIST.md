# Golden corpus capture — step by step

Run this **once**, inside the 30-day imc FAMOS trial. Everything else in the
project can be redone later; this data cannot, because the licence expires.

Written for someone who has not used FAMOS before. Menu wording varies a little
between FAMOS versions, so where an exact label is given, treat it as "look for
something like this" rather than a promise.

---

## 0. Before you start

You need three things, all already in the repo:

| what | where |
|---|---|
| the test signals | `golden_corpus/famos_golden_inputs.csv` |
| the sequence | `golden_corpus/famos_golden.seq` |
| somewhere to put the results | make a folder `golden_corpus/famos_out/` |

If `famos_golden_inputs.csv` is missing, regenerate it — it is gitignored
because it is 15 MB and reproducible:

```
python tools/make_golden_inputs.py
```

It is byte-identical every time (fixed seed), so a regenerated file is the same
file.

---

## 1. Activate the trial

imc LICENSE Manager → **Activation of Trial Versions via Internet**. Tick
**imc FAMOS Trial version (30 days for free)**.

Write the expiry date somewhere. Steps 2–5 must happen before it.

---

## 2. Import the test signals

This is the fiddly step; the rest is easy.

In FAMOS, load `golden_corpus/famos_golden_inputs.csv`. Look for
**File → Import**, or an **ASCII / text import assistant** — FAMOS does not read
a CSV by simply opening it, it asks you to describe the format first.

The format to describe:

| setting | value |
|---|---|
| column separator | comma `,` |
| decimal separator | point `.` |
| header rows | 1 (the column names) |
| rows of data | 120,000 |
| columns | 9 (Time + 8 signals) |
| **x-axis step (dx)** | **0.001 s** (i.e. 1000 Hz) |

Two things matter more than anything else here:

**dx must be 0.001 s.** Every filter width in the sequence is in seconds, so a
wrong sample rate silently produces a corpus that characterises the wrong
filter. If FAMOS offers to take the x-axis from the `Time` column, that works
too — it is 0 to 119.999 s in 0.001 steps.

**The channel names must survive the import**, exactly:

```
gc_impulse   gc_step   gc_dc   gc_chirp   gc_noise
gc_sine_2p5  gc_sine_5p0  gc_sine_10p0
```

The sequence refers to them by name. If FAMOS names them `Column1`…`Column9`
or similar, rename them in the variable list before going on, or the sequence
will fail on its first line.

**Check before continuing:** the FAMOS variable list should now show 8 channels
(the `Time` column is consumed as the x-axis and need not appear), each 120,000
samples long, each spanning 0–120 s.

---

## 3. Optional — load the real recording

Stage 4 of the sequence uses a real WFT channel. It is the most valuable single
capture, but also the one most likely to go wrong, so it is optional.

Load:

```
raw data in .dat format/2006-08-25 09-10-46 (1)/AccelY.raw
```

These are already FAMOS-format files, so FAMOS opens them directly — no import
assistant needed. Rename the loaded channel to **`wft_accely`**.

If this gives you trouble, skip it. Stages 1–3 are the irreplaceable part.

---

## 4. Run the sequence

Open `golden_corpus/famos_golden.seq` in FAMOS's **sequence editor** (look for
**File → Open** with the file type set to sequence, or a *Sequences* panel), and
run it — usually a **Run/Start** button or **F9**.

It defines roughly 40 new variables named `gc_*` (and `wft_accely_*` if you did
step 3). It computes them into the workspace and writes nothing to disk.

If it stops with an error, the message names the line. By far the most common
cause is a channel name from step 2 not matching.

---

## 5. Export the results — do not skip this

The sequence deliberately does not save anything, so at this point the results
exist only in memory.

Select every `gc_*` variable in the variable list and save them to
`golden_corpus/famos_out/`.

**Save as FAMOS `.dat`, not CSV.** This matters more than it sounds: the last
`smo` measurement was floored at 0.0056 N by the CSV export's own rounding, and
the question these files answer is whether two candidate kernels differ at 1e-6.
An ASCII export destroys the evidence before Python ever sees it. The reader on
our side (`dtt.ingestion.imc_reader.read_famos`) parses FAMOS binary directly,
so `.dat` is also the *easier* path.

---

## 6. Score it

Back in the repo:

```
python tools/score_golden.py golden_corpus/famos_out
```

This re-derives every channel with `famos.ops` and prints max error, mean error
and correlation per operator, per test signal. It exits non-zero if anything
misses tolerance, so it can gate a build later. Channels you did not export are
listed as "not exported" rather than treated as failures.

---

## What this settles

**The smo kernel.** Two reference exports currently disagree — a
boxcar-of-boxcars scores better against one, the exact triangle against the
other, and they cannot both be right. Every previous measurement ran `smo`
*after* `FiltLP`, so the filter's response contaminated it. Stage 2 runs `smo`
alone on an impulse, and an impulse response **is** the kernel. That ends the
argument by measurement instead of inference.

**The decimation phase.** `gc_impulse_redonly` is `red()` applied to a single
non-zero sample at a known index. Whichever index survives states directly which
of each block of ten `red()` keeps. A wrong phase shows up downstream as a fixed
sample lag that is otherwise very hard to attribute.

**The filter initialisation.** The step and DC channels show what FAMOS does at
the very start of a record, where a causal filter's output depends on its
assumed initial state.

---

## Two things to check while you have a licence

You cannot check either afterwards, and both are load-bearing assumptions in
the code:

1. **Does `FiltLPZ` appear in the operator list?** The migration notes argue
   `FiltLP` is causal partly because zero-phase is a separate Professional-only
   operator. Seeing the operator list settles that.

2. **What does `smo` do at the very edges of a channel?** Local implementations
   drifted apart on exactly this, and the first/last half-window of every
   channel depends on it.
