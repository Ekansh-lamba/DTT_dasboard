# DTT WFT Automation Platform — Concepts & Features Guide

A plain-language guide to **every concept used in this project** and **what each
feature does**. Organized as:

1. [The physical things being measured](#1-the-physical-basics--what-is-being-measured)
2. [What each pipeline stage does](#2-what-each-pipeline-stage-does-the-9-steps)
3. [What each chart actually tells an engineer](#3-what-each-chartnumber-actually-tells-an-engineer)
4. [What each GUI feature does](#4-what-each-gui-feature-does-the-11-screens)
5. [The mathematics behind the processing](#5-the-mathematics-behind-the-processing)

> **One-sentence summary:** the project takes a multi-hour recording of the
> push/pull/bump forces at all four wheels, cleans and filters it, summarizes how
> big and how severe those loads were, visualizes their distribution and
> relationships, calculates how much fatigue damage the drive caused, and packages
> it all into a report — fully automatically, through a desktop app.

---

## 1. The physical basics — what is being measured

### The sensor: WFT (Wheel Force Transducer)
A special measuring wheel bolted on in place of a normal wheel. As the car drives
over real roads, it records the **forces and twisting moments** flowing through
that wheel hundreds of times per second. This is the raw input to the whole
project.

### The three forces: Fx, Fy, Fz
Every wheel pushes and pulls in three directions:

| Force | Direction | What causes it (driving feel) |
|-------|-----------|-------------------------------|
| **Fx** | Longitudinal (forward/back) | **Braking and acceleration.** Hard braking → big Fx. |
| **Fy** | Lateral (side to side) | **Cornering.** Sharp turn → big Fy (the sideways grip). |
| **Fz** | Vertical (up/down) | **Weight + bumps.** The load the tyre carries; spikes over potholes. |

For one wheel you get a constant story: how hard it's braking (Fx), how hard it's
cornering (Fy), and how much weight is on it (Fz). **Fz** is always large and
positive (the car's weight). **Fx** and **Fy** swing positive *and* negative
(brake vs accelerate, left vs right).

### The three moments: Mx, My, Mz
Forces *push*; moments *twist*. Mx/My/Mz are the rotational loads at the hub. In
this project they are **optional** — recorded if present but not required, because
the forces drive the durability analysis.

### The four wheels: FL, FR, RL, RR
**F**ront/**R**ear × **L**eft/**R**ight. A channel name like `RL_Fz` =
"vertical force at the Rear-Left wheel." 4 wheels × 3 forces = **12 mandatory
channels** — the backbone of everything.

### Units: Newton (N) vs decaNewton (daN)
Force is measured in Newtons, but tyre engineers report in **decaNewtons**
(1 daN = 10 N). The software auto-detects the file's unit and converts if needed
(if forces look ~10× too big, divide by 10).

### Time, Distance, Speed, Sampling rate
- **Sampling rate** = readings per second. Here **100 Hz** (100 rows = 1 second).
  The full sample file is ~3.2 hours of driving = 1.17 million rows.
- **Time / Distance / Vehicle_Speed** = used to know *where* and *how fast* the car
  was when each force was recorded (important for histograms).
- **RLDA (Road Load Data Acquisition)** = the umbrella name for the "record
  real-world loads on real roads" practice.

---

## 2. What each pipeline stage does (the 9 steps)

The pipeline runs these in order, each producing something the next one uses.

**1. Data Ingestion** — Reads the messy CSV from the instrument. Figures out which
row is the header, finds the 12 force channels, detects the sampling rate from the
Time column, and fixes units (N→daN). *Output: a clean table + metadata.*

**2. Validation** — Checks data quality before wasting time analyzing. Are all 12
channels present? Any missing? Any duplicate columns or empty (NaN) values?
*Output: `validation_report.json` — a pass/fail health check.*

**3. Sanitization** — Gentle cleanup **without destroying data**: removes duplicate
timestamps, fills tiny gaps, and *flags* (but keeps) extreme outliers — because in
durability the extremes are the important part. *Output: `processed_data.csv`.*

**4. Filtering (Butterworth low-pass)** — Removes high-frequency electrical/sensor
**noise** while keeping the real load signal. Uses a "zero-phase" filter so it
cleans the signal **without shifting it in time** (timing of peaks matters for
fatigue). Default: keep everything below 10 Hz.

**5. Statistics** — Computes the summary numbers for each channel (see section 3).
*Output: `stats_summary.json`.*

**6. Histograms → 7. Heatmaps → 8. Boxplots → 9. Rainflow** — The four
visualization/analysis families (see section 3). *Output: PNG figures + rainflow
CSVs.*

**+ Report** — Bundles everything into a ~16-slide **PowerPoint** automatically —
replacing what an engineer used to assemble by hand.

---

## 3. What each chart/number actually tells an engineer

Each one answers a different question.

### Statistics (the numbers)
For each of the 12 channels:
- **Mean / Median** — the typical load. (Median ignores freak spikes; mean doesn't.)
- **Std (standard deviation)** — how *variable* the load is. High std on rear Fx =
  lots of braking/traction swing.
- **Min / Max** — the extremes seen.
- **P80 / P90 / P95 (percentiles)** — **load severity.** "P95 = 671 daN" means
  *95% of the time the load stayed below 671 daN.* Engineers care about these far
  more than the absolute max, because they describe how punishing the loading
  *usually* is, not just the one worst instant.

### Histograms — "how often does each load level occur?"
A bar chart of how frequently each force value appears. Two versions:
- **Percentage** — what % of the time the load was at each level.
- **Distance-weighted** — weights each reading by how far the car travelled during
  it (speed ÷ sample rate). Answers *"over how many kilometres did the tyre see
  this load?"* — which is what matters for wear and fatigue. A load seen at
  120 km/h covers more road than the same load at 5 km/h.

**Use:** see the shape of the load distribution per wheel and axis — tightly
clustered or spread, symmetric or skewed.

### Heatmaps — "how do two forces relate to each other?"
A 2-D colored map showing which **combinations** of forces happen together:
- **Fx × Fy, Fz × Fy, Fz × Fx** — e.g. does heavy vertical load (Fz) coincide with
  hard cornering (Fy)? Bright zones = common combinations.
- **Hexbin density (per wheel)** — same idea with hexagonal cells and a **log color
  scale** so *rare but important* high-load combos stay visible instead of being
  drowned out by the common near-zero cluster.

**Use:** spot correlated loading that single-channel stats can't reveal (e.g.
cornering + bump hitting at once).

### Boxplots — "compare the spread across all four wheels at a glance"
For each force (Fx, Fy, Fz), one box per wheel showing the median, the middle 50%
(the box), and the outliers.

**Use:** instantly compare wheels — e.g. "rear wheels carry more vertical load than
front," or "left/right imbalance" — for symmetry and load-balance checks.

### Rainflow — the durability/fatigue heart
The most specialized analysis. A real load signal is a messy up-and-down squiggle.
**Rainflow counting** is a standard algorithm that breaks that squiggle into a tidy
list of **load cycles**, each with a *range* (how big the swing), a *mean*, and a
*count* (how many times). Outputs:
- **From-To matrix (16×16)** — a grid showing how often the load swung from one
  level to another. The classic durability fingerprint.
- **Range distribution** — which swing sizes dominate the fatigue exposure.
- **Miner's damage number** — a single "how damaging was this drive" figure,
  computed as Σ(count × rangeᵐ). The exponent **m** (default 8) reflects how much
  *big* swings dominate damage — large cycles hurt far more than small ones. It's a
  *relative* comparison number (study A vs study B), not an absolute lifespan.

**Use:** quantify fatigue damage so a test rig or simulation can reproduce
equivalent wear in the lab.

---

## 4. What each GUI feature does (the 11 screens)

| Screen | What it does for you |
|--------|----------------------|
| **Dashboard** | Overview: how many studies processed, recent ones, last report, and a "Start New Analysis" button. |
| **New Study** | Pick a CSV, name the vehicle, set the filter (cutoff/order) and Miner's exponent, and launch the pipeline. |
| **Processing** | Live view while it runs — each of the 10 stages shows status (○ pending → ◐ running → ● done), how long it took, a progress bar, and the live log. |
| **Validation** | The data health check — rows/columns/duration, sampling rate, each channel PRESENT/MISSING, NaN counts, and a VALID/INVALID badge. |
| **Statistics** | The searchable table of all 12 channels with mean/median/std/min/max/P80/P90/P95; filter by wheel, export to CSV. |
| **Histograms** | Pick a wheel (FL/FR/RL/RR) and signal (Fx/Fy/Fz) and view the distribution; zoom/pan, fullscreen. |
| **Heatmaps** | The Fx×Fy / Fz×Fy / Fz×Fx correlation maps + per-wheel hexbin density, with fullscreen view. |
| **Boxplots** | The Fx/Fy/Fz cross-wheel comparison plots. |
| **Rainflow** | The matrix images per wheel + a table of cycle-count CSVs you can export. |
| **Reports** | Lists the PowerPoint files; open, export, or regenerate them. |
| **History** | A table of every past study (name, vehicle, date, status) with open/folder actions. |

### Behind the scenes (technical features)
- The GUI runs the pipeline in a **background thread** so the window never freezes.
- It reads progress by **parsing the pipeline's log** output.
- Figures use a **zoom/pan image viewer**.
- The whole app uses a **dark engineering theme** matching the plot colors.
- It is built in **MVC** so the data-reading logic lives in one place
  (`gui/models/repository.py`).

---

## 5. The mathematics behind the processing

The exact formulas and algorithms used to turn raw force readings into the numbers
and charts. Notation: `x` = a channel's values, `N` = number of samples,
`fs` = sampling rate (Hz).

### 5.1 Sampling, time and the Nyquist limit
- **Sampling rate from the Time column** — the rate is *measured*, not assumed:
  ```
  fs = 1 / median( diff(Time) )
  ```
  Using the median of consecutive time differences makes it robust to occasional
  bad timestamps.
- **Duration** — `duration_s = N / fs`.
- **Nyquist frequency** — `f_nyq = fs / 2` (50 Hz at 100 Hz sampling). This is the
  highest frequency that can be represented; any filter cutoff must stay below it.

### 5.2 Unit normalization (N → daN)
A median-magnitude test decides whether to rescale:
```
if  max_over_channels( median(|x|) ) > 1000  →  x = x / 10
```
Forces in Newtons are ~10× larger than in decaNewtons, so a large median is the
tell-tale. Median (not mean) avoids being fooled by spikes.

### 5.3 Descriptive statistics (per channel)
Computed on the non-missing values of each channel:

| Quantity | Formula |
|----------|---------|
| Mean | `μ = (1/N) Σ xᵢ` |
| Median | middle value of the sorted data (average of two middles if N is even) |
| Std (population, ddof=0) | `σ = sqrt( (1/N) Σ (xᵢ − μ)² )` |
| Min / Max | smallest / largest value |

### 5.4 Percentiles (P80 / P90 / P95)
The **P-th percentile** is the value below which P% of the data falls. It is found
by **linear interpolation** on the sorted data:
```
rank      = (P/100) · (N − 1)
Pvalue    = x_sorted[floor(rank)] + frac(rank) · ( x_sorted[floor(rank)+1] − x_sorted[floor(rank)] )
```
These quantify **load severity** — "95% of the time the load stayed below this."

### 5.5 Outlier flagging (non-destructive)
Bounds are set by the 1st and 99th percentiles; samples outside are *counted*, not
removed:
```
lo = P1(x),   hi = P99(x)
flagged = count( x < lo  OR  x > hi )
```

### 5.6 Gap filling (forward fill)
Short missing runs are bridged by carrying the last valid value forward, capped at
10 samples so genuine long dropouts are never invented:
```
xᵢ = x_{last valid}   for gaps of length ≤ 10
```

### 5.7 Butterworth low-pass filter (zero-phase)
The core noise-removal math.

- **Normalized cutoff** — the cutoff is expressed as a fraction of Nyquist:
  ```
  Wn = f_cutoff / f_nyq          (default 10 / 50 = 0.2)
  ```
  If `f_cutoff ≥ f_nyq`, it is clamped to `0.9 · f_nyq`.
- **Filter design** — an order-`n` Butterworth filter (default n = 4) has a
  magnitude response that is *maximally flat* in the passband:
  ```
  |H(f)|² = 1 / ( 1 + (f / f_cutoff)^(2n) )
  ```
  `scipy.signal.butter(n, Wn)` returns the filter coefficients `b, a`.
- **Zero-phase application (`filtfilt`)** — the filter is run **forward then
  backward** over the signal. Each pass adds equal-and-opposite phase shift, so the
  net phase shift is **zero** (no time delay of peaks) and the effective magnitude
  response is squared:
  ```
  y = filtfilt(b, a, x)     ⇒   |H_eff(f)|² = ( |H(f)|² )²
  ```
  Preserving peak *timing* is essential because rainflow/fatigue depends on it.

### 5.8 Histograms (binned distributions)
- **Binning** — the force axis is split into `HIST_BINS = 60` equal bins between a
  fixed engineering range (or the data min/max). Bin centers: `(edge_i + edge_{i+1})/2`.
- **Weighted counts** — instead of counting samples, each sample contributes a
  weight:
  ```
  height(bin) = Σ  w_i   for x_i in that bin
  ```
- **Distance weighting** — the weight is the distance travelled during that sample:
  ```
  w_i = speed_i(m/s) / fs           (speed × Δt = distance per sample)
  total_distance = Σ w_i
  ```
- **Percentage version** — `height = (w_i / total_distance) · 100`.

### 5.9 Distance & speed (numerical integration)
- **Speed-unit detection** — if the median speed > 10 it is treated as km/h and
  converted: `v(m/s) = v(km/h) / 3.6`.
- **Cumulative distance** — a discrete integral of speed over time:
  ```
  Dist_k = Σ_{i≤k} v_i / fs          (≈ ∫ v dt)
  ```

### 5.10 Heatmaps (2-D occurrence + log color)
- **2-D binning** — a grid counts how often each `(force_A, force_B)` pair occurs.
- **Log color scaling** — to keep rare high-load combinations visible, the color
  maps `log10(count + 1)` (the `+1` avoids `log(0)`).
- **Hexbin** — the same density estimate using hexagonal cells over a fixed extent.

### 5.11 Boxplots (quartile summary)
Each box is built from the **quartiles**:
```
Q1 = P25,   Q2 = median = P50,   Q3 = P75
IQR = Q3 − Q1                      (height of the box)
whiskers reach to the furthest points within 1.5·IQR
points beyond the whiskers are drawn as outliers
```

### 5.12 Rainflow cycle counting
The fatigue core. The algorithm (ASTM E1049 standard) scans the load-time signal's
turning points and extracts closed hysteresis loops. Each extracted **cycle** has:
```
range  = peak − valley          (size of the load swing)
mean   = (peak + valley) / 2    (its midpoint)
count  = 1.0 or 0.5             (full or half cycle)
```
For the matrix, each cycle's swing is expressed as a **from→to** pair:
```
from = mean − range/2
to   = mean + range/2
```
(For very long signals the data is first decimated to ≤ 50,000 points to keep cycle
extraction tractable.)

### 5.13 From-To matrix (16×16 binning)
The "from" and "to" load levels are sorted into `RAINFLOW_BINS = 16` bins using a
search over shared bin edges, and each cycle's count is accumulated:
```
fi = bin_index(from),   ti = bin_index(to)
Matrix[ti, fi] += count
```
The image displays `log10(Matrix + 1)` so rare large swings remain visible.

### 5.14 Miner's rule (cumulative damage)
A single damage number summarizing the whole drive, based on the linear damage
hypothesis. From the S-N (stress-life) relationship `N·Sᵐ = C`, the damage of one
load level is `n/N ∝ n·Sᵐ`, so total damage is:
```
D = Σ  countᵢ · rangeᵢ^m          (default exponent m = 8)
```
A higher exponent `m` makes **large** swings dominate (a swing twice as big is
`2^8 = 256×` more damaging). `D` is a **relative** comparison metric between
studies/channels — not an absolute predicted lifespan.

---

## Glossary quick-reference

| Term | Meaning |
|------|---------|
| WFT | Wheel Force Transducer — the measuring wheel |
| RLDA | Road Load Data Acquisition |
| Fx / Fy / Fz | Longitudinal / Lateral / Vertical force |
| Mx / My / Mz | Twisting moments (optional channels) |
| FL / FR / RL / RR | Front-Left / Front-Right / Rear-Left / Rear-Right wheel |
| daN | decaNewton = 10 N (engineering force unit) |
| Sampling rate | Readings per second (100 Hz here) |
| Nyquist frequency | Half the sampling rate; the highest representable frequency |
| Butterworth LPF | Low-pass filter that removes high-frequency noise |
| Zero-phase (filtfilt) | Forward+backward filtering; no time shift of peaks |
| Percentile (P95) | The value below which 95% of readings fall — load severity |
| Std (σ) | Standard deviation — spread of values around the mean |
| IQR | Inter-quartile range, Q3 − Q1 — the box height in a boxplot |
| Rainflow | Algorithm that converts a load signal into counted cycles |
| Miner's rule | Cumulative fatigue-damage estimate, Σ(count × rangeᵐ) |
| From-To matrix | Grid of how loads swing between levels (durability fingerprint) |
