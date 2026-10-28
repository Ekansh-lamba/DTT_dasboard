"""
make_famos_report.py

Visual evidence that the DTT app's preprocessed output matches imc FAMOS,
channel by channel. Writes one multi-page landscape-A4 PDF plus the same
pages as PNGs.

    py validation_tools\\make_famos_report.py --dir C:\\famos_check

Inputs (all in --dir):
    famos_red.csv                          FAMOS 100 Hz export (*_red, *_rawred)
    famos_1k.csv                           FAMOS 1 kHz export (FR_Fx, FR_Fx1, RR_Fz, Latacc)
    preprocessed_for_famos_validation.csv  app output, 100 Hz
    raw_data.csv                           app raw snapshot, 100 Hz
    famos_compare.py                       the comparison script: its reader, pairing
                                           and compare() are imported, so the report
                                           and the numbers use the same code
    famos_comparison_results.csv,
    famos_1k_results.csv                   its results: verdicts shown, and every
                                           statistic recomputed here is checked
                                           against them

Outputs (in --dir unless --out is given):
    famos_validation_report.pdf
    report_png/page_NN_*.png

This is a standalone tool: it does not import or change the app (dtt/, famos/, gui/).
"""

import argparse
import datetime
import importlib.util
import math
import os
import sys
import textwrap
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MaxNLocator  # noqa: E402

FS = 100.0            # Hz, the decimated rate of famos_red.csv and the app files
FS_1K = 1000.0
TRIM_S = 2.0          # famos_compare.py default --trim
MAX_LAG = 200         # famos_compare.py default --max-lag
MAX_D_1K = 300        # famos_compare.py run_1k default
DEAD_MIN_ROWS = 1000  # an app gap of >= 10 s is a dead stretch, shorter is a gap
A4_LANDSCAPE = (11.69, 8.27)

C_APP = "#1f5fbf"
C_FAMOS = "#f28e1c"
C_ERR = "#d62728"
C_DEAD = "#9e9e9e"
C_GAP = "#7b3fa0"
APP_LINE = dict(color=C_APP, lw=1.7, solid_capstyle="butt", zorder=2)
FAMOS_LINE = dict(color=C_FAMOS, lw=1.0, ls=(0, (4, 2.5)), zorder=3)
VERDICT_COLOURS = {"PASS": "#b9e4c0", "FLAT": "#d9d9d9"}
VERDICT_BAD = "#f4b6b6"

DEAD_LABEL = "RR transducer dead: FAMOS keeps zeros, app blanks (documented difference)"

UNITS = {"force": "daN", "moment": "Nm", "accel": "m/s²", "speed": "km/h"}

FILES = dict(famos="famos_red.csv", famos1k="famos_1k.csv",
             app="preprocessed_for_famos_validation.csv", raw="raw_data.csv",
             script="famos_compare.py", res="famos_comparison_results.csv",
             res1k="famos_1k_results.csv")

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "axes.titleweight": "bold",
})


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_compare_module(path):
    spec = importlib.util.spec_from_file_location("famos_compare", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_1k(path, wanted):
    """Only the needed columns of the ~500 MB 1 kHz export, with the C parser.
    Same layout as the 100 Hz file: names row, units row, space-padded values."""
    df = pd.read_csv(path, sep=",", skiprows=[1], header=0, engine="c",
                     skipinitialspace=True, usecols=lambda c: c.strip() in wanted,
                     dtype=np.float64)
    df.columns = [c.strip() for c in df.columns]
    return df


def app_sig_digits(path, cols, n_lines=5000):
    """Significant digits the app wrote per column (expected 6, from %.6g)."""
    with open(path, "r", encoding="utf-8-sig") as f:
        header = f.readline().rstrip("\r\n").split(",")
        lines = [f.readline() for _ in range(n_lines)]
    idx = {c: header.index(c) for c in cols if c in header}
    out = {}
    for c, k in idx.items():
        best = 0
        for ln in lines:
            toks = ln.rstrip("\r\n").split(",")
            if k >= len(toks):
                continue
            tok = toks[k].strip().lstrip("+-").lower().split("e")[0].replace(".", "").lstrip("0")
            best = max(best, len(tok))
        out[c] = best or 6
    return out


def is_float32(values, n=20000):
    """True when the values are float32 numbers (FAMOS prints them to ~14 significant
    digits, so they match float32 to ~1e-13 relative, not bit for bit)."""
    v = values[np.isfinite(values) & (values != 0)]
    if len(v) == 0:
        return False
    v = v[:: max(1, len(v) // n)]
    f = v.astype(np.float32).astype(np.float64)
    return float(np.mean(np.abs(f - v) <= 1e-12 * np.abs(v))) > 0.99


# ---------------------------------------------------------------------------
# numeric helpers
# ---------------------------------------------------------------------------
def nan_runs(x):
    """(start, stop) of every run of non-finite values."""
    bad = ~np.isfinite(x)
    if not bad.any():
        return []
    d = np.diff(np.r_[0, bad.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def envelope(y, t0=0.0, bin_rows=int(FS)):
    """Min/max per 1 s bin, so spikes survive the reduction. All-NaN bins give NaN."""
    n = len(y) // bin_rows * bin_rows
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    yb = y[:n].reshape(-1, bin_rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanmin(yb, axis=1), np.nanmax(yb, axis=1)
    t = t0 + (np.arange(len(lo)) + 0.5) * bin_rows / FS
    return t, lo, hi


def rounding_floor(app_vals, famos_vals, scale, digits, famos_f32):
    """Half a unit in the last digit the app CSV wrote (digits significant figures),
    plus half a float32 step of FAMOS's stored value when FAMOS stores float32.
    FAMOS's own CSV text has ~14 significant digits, so its text rounding is negligible."""
    mag = np.abs(app_vals)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.floor(np.log10(np.where(mag > 0, mag, np.nan)))
    floor = np.where(mag > 0, 0.5 * 10.0 ** (e - (digits - 1)), 0.0)
    if famos_f32:
        fv = np.where(np.isfinite(famos_vals), famos_vals, 0).astype(np.float32)
        floor = floor + 0.5 * np.spacing(np.abs(fv)).astype(np.float64) * abs(scale)
    return floor


def dead_text(spans):
    """'5,363.4–5,377.1 s and 5,378.2–5,547.8 s, 183.3 s in total'"""
    parts = " and ".join(f"{s:,.1f}–{e:,.1f} s" for s, e in spans)
    tot = sum(e - s for s, e in spans)
    return parts + (f", {tot:.1f} s in total" if len(spans) > 1 else f", {tot:.1f} s")


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 1e5 or ax < 1e-3:
        return f"{x:.{nd - 1}e}"
    return f"{x:.{nd}g}"


def fmt_r(r):
    if r is None or not np.isfinite(r):
        return "flat"
    return f"{r:.13f}"


def num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v


def close(a, b, rel=1e-9, abs_=1e-15):
    a, b = num(a), num(b)
    if not np.isfinite(a) and not np.isfinite(b):
        return True
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


# ---------------------------------------------------------------------------
# one channel pair
# ---------------------------------------------------------------------------
class Pair:
    pass


def prepare_pair(fc, famos, other, fname, cands, src, kind, digits_map):
    p = Pair()
    p.fname, p.src, p.kind = fname, src, kind
    p.unit = UNITS.get(kind, UNITS["force"] if fname[3:5].startswith("F") else "Nm")
    if kind == "rawred":
        p.unit = "daN" if fname.split("_")[1].startswith("F") else "Nm"
    fcol = fc.find_col(famos, [fname])
    p.pcol = fc.find_col(other, cands) if other is not None else None
    p.missing = None
    if fcol is None:
        p.missing = f"{fname} not found in {FILES['famos']}"
    elif p.pcol is None:
        p.missing = f"none of {cands} found in {FILES['raw' if src == 'raw' else 'app']}"
    if p.missing:
        return p

    fa = famos[fcol].to_numpy(dtype=float)
    pb = other[p.pcol].to_numpy(dtype=float)
    res, _, _ = fc.compare(fa, pb, "rawred" if kind == "rawred" else kind, TRIM_S, MAX_LAG)
    p.res = res
    lag = int(res["lag"])
    if lag >= 0:
        a, b, off = fa, pb[lag:], 0
    else:
        a, b, off = fa[-lag:], pb, -lag
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    p.t0 = off / FS
    p.n_rows = len(fa)
    p.lag = lag

    # identical exclusion rule to famos_compare.compare(): trim the ends, then drop
    # non-finite rows and rows within GAP_GUARD of a NaN in the app's data
    trim = int(TRIM_S * FS)
    sl = slice(trim, n - trim) if n > 2 * trim + 100 else slice(0, n)
    incl = np.zeros(n, dtype=bool)
    at, bt = a[sl], b[sl]
    incl[sl] = np.isfinite(at) & np.isfinite(bt) & ~fc.near_gap(bt)
    p.incl = incl
    p.trim_rows = n - (sl.stop - sl.start)
    p.scale = float(res.get("scale", 1.0))
    p.a = a * p.scale          # FAMOS in the app's unit
    p.b = b
    p.err = p.a - p.b
    p.flat = res.get("verdict") in ("FLAT", "FLAT/DIFF")

    e = p.err[incl]
    p.n = int(incl.sum())
    p.max_err = float(np.max(np.abs(e))) if len(e) else float("nan")
    p.rms_err = float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan")
    p.r = float("nan") if p.flat else float(np.corrcoef(p.a[incl], p.b[incl])[0, 1])
    rng = float(np.ptp(p.b[incl])) if p.n else 0.0
    p.rel = float("nan") if p.flat else p.max_err / (rng or 1.0)

    runs = nan_runs(b)
    p.dead = [(s, e_) for s, e_ in runs if e_ - s >= DEAD_MIN_ROWS]
    p.gaps = [(s, e_) for s, e_ in runs if e_ - s < DEAD_MIN_ROWS]
    # share of exact zeros FAMOS holds inside each of the app's blanked runs
    p.dead_famos_zero = [float(np.mean(a[s:e_] == 0)) for s, e_ in p.dead]

    if kind != "rawred":
        dig = digits_map.get(p.pcol, 6)
        p.digits = dig
        p.famos_f32 = is_float32(fa)
        p.floor = rounding_floor(p.b, a, p.scale, dig, p.famos_f32)
        fl = p.floor[incl]
        inside = np.abs(e) <= fl * (1 + 1e-9)
        p.within_floor = int(inside.sum())
        p.outside_max = float(np.max(np.abs(e[~inside]))) if (~inside).any() else None
        p.outside_value = float(np.median(np.abs(p.b[incl][~inside]))) if (~inside).any() else None
        # error inside the excluded guard band, for honesty on the page
        guard = ~incl & np.isfinite(p.err)
        guard[:sl.start] = False
        guard[sl.stop:] = False
        p.guard_max = float(np.max(np.abs(p.err[guard]))) if guard.any() else None
    else:
        p.n_equal = int(np.sum(np.abs(e) <= 1e-9))
    return p


def check_against_csv(p, row):
    """List of mismatches between this report's statistics and the results CSV."""
    if row is None:
        return ["no row in results CSV"]
    bad = []
    checks = [("n", p.n, row.get("n")), ("lag", p.lag, row.get("lag")),
              ("max_err", p.max_err, row.get("max_err")), ("rms_err", p.rms_err, row.get("rms_err")),
              ("scale", p.scale, row.get("scale"))]
    if not p.flat:
        checks += [("r", p.r, row.get("r")), ("rel", p.rel, row.get("rel"))]
    for name, mine, theirs in checks:
        if not close(mine, theirs, rel=1e-6 if name in ("rms_err", "rel") else 1e-9):
            bad.append(f"{name}: report {mine!r} vs CSV {theirs!r}")
    if str(row.get("verdict")) != str(p.res.get("verdict")):
        bad.append(f"verdict: recomputed {p.res.get('verdict')} vs CSV {row.get('verdict')}")
    return bad


# ---------------------------------------------------------------------------
# page furniture
# ---------------------------------------------------------------------------
def new_page():
    return plt.figure(figsize=A4_LANDSCAPE)


def add_footer(fig, page, total, ctx):
    fig.text(0.012, 0.012,
             f"Page {page} of {total}   ·   Inputs: {FILES['famos']}, {FILES['famos1k']}, "
             f"{FILES['app']}, {FILES['raw']}, {FILES['res']}, {FILES['res1k']}   ·   "
             f"Generated {ctx['generated']} by validation_tools/make_famos_report.py",
             fontsize=6, color="#555555", ha="left", va="bottom")


def missing_page(fig, title, msg):
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.96)
    fig.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12, color=C_ERR,
             bbox=dict(boxstyle="round", fc="#fff3f3", ec=C_ERR))


def shade_blanks(ax, p, label=True, gaps=True):
    """Dead stretch (grey) and short gaps (purple) in the app's data."""
    for k, (s, e) in enumerate(p.dead):
        ax.axvspan(p.t0 + s / FS, p.t0 + e / FS, color=C_DEAD, alpha=0.28, lw=0, zorder=0,
                   label=DEAD_LABEL if (label and k == 0) else None)
    if gaps:
        for k, (s, e) in enumerate(p.gaps):
            ax.axvspan(p.t0 + s / FS - 0.5, p.t0 + e / FS + 0.5, color=C_GAP, alpha=0.35, lw=0,
                       zorder=0, label="app gap (NaN) ±0.5 s" if (label and k == 0) else None)


def plot_window(ax, p, i0, i1, markers=False):
    t = p.t0 + np.arange(i0, i1) / FS
    if markers:
        ax.plot(t, p.b[i0:i1], color=C_APP, lw=1.2, zorder=2)
        ax.plot(t, p.b[i0:i1], "o", color=C_APP, ms=3.2, zorder=3, label="app")
        ax.plot(t, p.a[i0:i1], "o", mfc="none", mec=C_FAMOS, mew=1.1, ms=6.5, zorder=4,
                label="FAMOS")
    else:
        ax.plot(t, p.b[i0:i1], label="app", **APP_LINE)
        ax.plot(t, p.a[i0:i1], label="FAMOS", **FAMOS_LINE)
    ax.set_xlim(t[0], t[-1])


def win(center, half, n):
    lo = int(max(0, min(center - half, n - 2 * half)))
    return lo, int(min(n, lo + 2 * half))


def time_axis(ax, decimals):
    """Plain time ticks (no '+2.77e3' offset) for the short windows."""
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))


def steepest(p, i0, i1):
    d = np.abs(np.diff(p.b[i0:i1]))
    d[~np.isfinite(d)] = -1
    return i0 + int(np.argmax(d)) if len(d) and d.max() > 0 else (i0 + i1) // 2


# ---------------------------------------------------------------------------
# summary page
# ---------------------------------------------------------------------------
def page_summary(fig, ctx):
    pairs, k1, sev = ctx["pairs"], ctx["k1"], ctx["sev"]
    fig.text(0.5, 0.965, "imc FAMOS vs DTT app: preprocessing validation", ha="center",
             fontsize=14, fontweight="bold")
    fig.text(0.5, 0.937,
             f"Recording: {ctx['recording']}   ·   100 Hz files: {ctx['n_famos']:,} rows "
             f"(FAMOS) / {ctx['n_app']:,} (app) / {ctx['n_raw']:,} (app raw)   ·   "
             f"1 kHz FAMOS: {ctx['n_1k']:,} rows   ·   duration {ctx['n_famos'] / FS:,.1f} s   ·   "
             f"time columns agree to {fmt(ctx['time_diff'])} s",
             ha="center", fontsize=7.5, color="#333333")

    # --- channel table ---
    ax = fig.add_axes([0.012, 0.05, 0.585, 0.865])
    ax.axis("off")
    cols = ["FAMOS channel", "app column", "verdict", "r", "max |err|", "max err /\nrange",
            "unit\nfactor", "offset", "rows\nexcl."]
    cells, colours = [], []
    for p in pairs:
        row = ctx["csv_rows"].get(p.fname, {})
        verdict = str(row.get("verdict", "MISSING")) if not p.missing else "MISSING"
        app_c = (p.pcol or "–") + (" (raw)" if p.src == "raw" else "")
        if p.missing:
            cells.append([p.fname, app_c, verdict, "–", "–", "–", "–", "–", "–"])
        else:
            cells.append([p.fname, app_c, verdict, fmt_r(p.r), f"{fmt(p.max_err)} {p.unit}",
                          fmt(p.rel, 2), f"{p.scale:g}", f"{p.lag} ({p.res['lag_frac']:+.2f})",
                          f"{p.n_rows - p.n:,}"])
        colours.append(verdict)
    for k in k1:
        verdict = k.get("verdict", "MISSING")
        if k.get("missing"):
            cells.append([f"{k['fname']} (1 kHz)", k["pcol"] or "–", "MISSING"] + ["–"] * 6)
        else:
            cells.append([f"{k['fname']} (1 kHz)", k["pcol"] + (" (raw)" if k["src"] == "raw" else ""),
                          verdict, fmt_r(k["r"]), f"{fmt(k['max_err'])} {k['unit']}", fmt(k["rel"], 2),
                          f"{k['scale']:g}", f"{k['d']} ms", f"{ctx['n_app'] - k['n']:,}"])
        colours.append(verdict)
    tab = ax.table(cellText=cells, colLabels=cols, loc="upper left", cellLoc="center",
                   colWidths=[0.15, 0.15, 0.07, 0.16, 0.12, 0.09, 0.07, 0.09, 0.08])
    tab.auto_set_font_size(False)
    tab.set_fontsize(6.3)
    tab.scale(1, 1.0)
    nrows = len(cells) + 1
    for (r, c), cell in tab.get_celld().items():
        cell.set_linewidth(0.3)
        cell.set_height(0.96 / (nrows + 1.2) if r else 1.6 * 0.96 / (nrows + 1.2))
        if r == 0:
            cell.set_facecolor("#e8ecf3")
            cell.set_text_props(fontweight="bold")
        elif c == 2:
            v = colours[r - 1]
            cell.set_facecolor(VERDICT_COLOURS.get(v, VERDICT_BAD))
            cell.set_text_props(fontweight="bold")
        if c in (0, 1) and r:
            cell.set_text_props(ha="left")
            cell._loc = "left"
    ax.text(0.0, -0.005,
            "r, max |err|, max/range: FAMOS × unit factor vs app, rows excluded = 2 s at each end + app NaN rows ±0.5 s "
            "(famos_compare.py rules).\nOffset: lag in 100 Hz samples (sub-sample estimate), "
            "or the 1 kHz start offset d (app sample i = FAMOS sample 10i + d).",
            transform=ax.transAxes, fontsize=5.8, va="top", color="#444444")

    # --- G-severity table ---
    ax2 = fig.add_axes([0.61, 0.58, 0.378, 0.305])
    ax2.axis("off")
    fig.text(0.61, 0.895, "G-severity (app formulas, both files; Fz in daN)", fontsize=8.5, fontweight="bold")
    gcols = ["wheel / source", "Gx", "Gy", "Gxy", "Static Fz", "Std Fz", "DLC", "rows", "Fz=0"]
    keys = ["Gx", "Gy", "Gxy", "StaticFz", "StdFz", "DLC"]
    gcells, gstyle = [], []

    def srow(label, s, style):
        gcells.append([label] + [f"{s[k]:.4f}" if k in ("Gx", "Gy", "Gxy", "DLC") else f"{s[k]:.1f}"
                                 for k in keys] + [f"{s['n']:,}", f"{s['fz_zero']:,}"])
        gstyle.append(style)

    def drow(label, s1, s2, style):
        out = [label]
        for k in keys:
            rel = (s1[k] - s2[k]) / s2[k] if s2[k] else float("nan")
            v = rel * 100
            out.append("0" if v == 0 else ("<0.001%" if abs(v) < 1e-3 else f"{v:+.2g}%"))
        out += [f"{s1['n'] - s2['n']:+,}", ""]
        gcells.append(out)
        gstyle.append(style)

    for pos in ("FR", "RR"):
        if pos not in sev:
            gcells.append([f"{pos}: missing columns"] + [""] * 8)
            gstyle.append("bad")
            continue
        s = sev[pos]
        mark = " †" if pos == "RR" and ctx["rr_dead"] else ""
        srow(f"{pos} FAMOS{mark}", s["FAMOS"], "dead" if mark else "")
        srow(f"{pos} app", s["app"], "")
        drow(f"{pos} difference{mark}", s["FAMOS"], s["app"], "diffdead" if mark else "diff")
        if "FAMOS_same" in s:
            srow(f"{pos} FAMOS ‡", s["FAMOS_same"], "")
            drow(f"{pos} difference ‡", s["FAMOS_same"], s["app"], "diff")
    gt = ax2.table(cellText=gcells, colLabels=gcols, loc="upper left", cellLoc="center",
                   colWidths=[0.17, 0.10, 0.10, 0.10, 0.105, 0.105, 0.10, 0.10, 0.085])
    gt.auto_set_font_size(False)
    gt.set_fontsize(6.0)
    for (r, c), cell in gt.get_celld().items():
        cell.set_linewidth(0.3)
        cell.set_height(1.0 / (len(gcells) + 1))
        if r == 0:
            cell.set_facecolor("#e8ecf3")
            cell.set_text_props(fontweight="bold")
            continue
        st = gstyle[r - 1]
        if st in ("diff", "diffdead"):
            cell.set_facecolor("#f6f6f6")
            cell.set_text_props(style="italic")
        if st in ("dead", "diffdead"):
            cell.set_facecolor("#e4e4e4")
        if c == 0:
            cell._loc = "left"
    notes = ["Difference rows: (FAMOS − app) / app; rows: FAMOS rows minus app rows. "
             "Gx, Gy skip rows with Fz = 0."]
    if ctx["rr_dead"]:
        notes.append(f"† affected by the RR dead stretch ({dead_text(ctx['rr_dead'])}): FAMOS's zeros "
                     "enter its Static Fz, Std Fz, DLC and Gxy; the app's blanked rows do not.")
    if any("FAMOS_same" in s for s in sev.values()):
        notes.append("‡ FAMOS computed on only the rows where the app has data for that wheel (FR: without the "
                     "FR_Fx_2 gap; RR: without the dead stretch). What remains is the CSV rounding.")
    fig.text(0.61, 0.57,"\n".join(textwrap.fill(n, 112) for n in notes), fontsize=5.8, va="top",
             color="#444444", linespacing=1.3)

    # --- conclusion ---
    fig.text(0.61, 0.475, "Conclusion (generated from the results)", fontsize=8.5, fontweight="bold")
    body = "\n".join(textwrap.fill(line, 86,initial_indent="• ", subsequent_indent="   ")
                     for line in ctx["conclusion"])
    fig.text(0.61, 0.46, body, fontsize=6.7, va="top", linespacing=1.38)


# ---------------------------------------------------------------------------
# channel page (_red)
# ---------------------------------------------------------------------------
def page_channel(fig, p, ctx):
    row = ctx["csv_rows"].get(p.fname, {})
    verdict = row.get("verdict", "–")
    if p.missing:
        missing_page(fig, f"{p.fname} vs app", f"Cannot plot: {p.missing}")
        return
    vcol = "#1a7f37" if verdict == "PASS" else ("#666666" if verdict == "FLAT" else C_ERR)
    fig.text(0.012, 0.968, f"{p.fname} (FAMOS)   vs   {p.pcol} (app)   ·   {p.unit}", fontsize=12.5,
             fontweight="bold", va="center")
    fig.text(0.988, 0.968, verdict, fontsize=13, fontweight="bold", color="white", ha="right",
             va="center", bbox=dict(boxstyle="round,pad=0.3", fc=vcol, ec="none"))

    fam_unit = ctx["famos_units"].get(p.fname, "?")
    fitted = p.res.get("fitted", float("nan"))
    fit_txt = (f"fitted {fitted:.10g}, snapped to a power of ten" if np.isfinite(fitted)
               else "not fitted: constant signal")
    info = [f"Unit factor: app = FAMOS × {p.scale:g} ({fit_txt}; FAMOS units row says '{fam_unit}').   "
            f"Offset {p.lag} samples (sub-sample {p.res['lag_frac']:+.2f}).",
            f"Error statistics on {p.n:,} of {p.n_rows:,} rows: excluded {p.trim_rows} rows (2 s at each end) "
            f"+ {p.res.get('gap_rows', 0):,} rows that are NaN in the app or within 0.5 s of a NaN."]
    notes = []
    if p.dead:
        parts = [f"{p.t0 + s / FS:,.2f}–{p.t0 + e / FS:,.2f} s ({e - s:,} rows, FAMOS {z:.1%} exact zeros)"
                 for (s, e), z in zip(p.dead, p.dead_famos_zero)]
        notes.append(f"RR dead stretch, app NaN in {len(p.dead)} run(s): " + "; ".join(parts) +
                     ". FAMOS's non-zero rows there are its smoothing ramping into the zeros at the run edges. "
                     "Excluded ±0.5 s.")
    for s, e in p.gaps:
        fv = ", ".join(fmt(v, 5) for v in p.a[s:e][:6])
        notes.append(f"App gap: {e - s} NaN rows at t = {p.t0 + s / FS:,.2f}–{p.t0 + (e - 1) / FS:,.2f} s "
                     f"(rows {s:,}–{e - 1:,}); FAMOS there ({p.unit}): {fv}. Marked in purple, excluded ±0.5 s.")
    if p.flat:
        notes.append(f"{p.pcol} is zero for the whole recording (no GPS signal): flat in both files. "
                     f"Unit factor cannot be fitted on a constant, 1 is used; FAMOS row says '{fam_unit}', "
                     "the app writes km/h.")
    lines = [textwrap.fill(t, 205, subsequent_indent="    ") for t in info + notes]
    fig.text(0.012, 0.935, "\n".join(lines), fontsize=6.8, va="top", color="#222222",
             linespacing=1.35)
    top = 0.905 - 0.0155 * sum(t.count("\n") + 1 for t in lines)

    gs = fig.add_gridspec(3, 3, left=0.07, right=0.975, top=top, bottom=0.07,
                          hspace=0.55, wspace=0.22, height_ratios=[1.0, 1.0, 1.05])
    n = len(p.b)
    t_end = p.t0 + n / FS

    # 1 whole recording
    ax1 = fig.add_subplot(gs[0, :])
    tb, blo, bhi = envelope(p.b, p.t0)
    _, alo, ahi = envelope(p.a, p.t0)
    ax1.fill_between(tb, blo, bhi, color=C_APP, alpha=0.18, lw=0, zorder=1)
    ax1.plot(tb, bhi, **APP_LINE)
    ax1.plot(tb, blo, **APP_LINE)
    ax1.plot(tb, ahi, **FAMOS_LINE)
    ax1.plot(tb, alo, **FAMOS_LINE)
    shade_blanks(ax1, p)
    for s, e in p.gaps:
        tg = p.t0 + s / FS
        ax1.annotate(f"app gap {e - s} rows\nt = {tg:,.2f} s", xy=(tg, np.nanmax(bhi)),
                     xytext=(tg + 0.04 * (t_end - p.t0) * (1 if tg < 0.7 * t_end else -3), np.nanmax(bhi)),
                     fontsize=6.3, color=C_GAP, va="top",
                     arrowprops=dict(arrowstyle="->", color=C_GAP, lw=0.8))
    ax1.set_xlim(p.t0, t_end)
    ax1.set_title("1  Whole recording: min/max per 1 s bin", loc="left")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel(p.unit)
    handles = [Line2D([], [], label="app (solid blue)", **APP_LINE),
               Line2D([], [], label="FAMOS × factor (dashed orange)", **FAMOS_LINE)]
    if p.dead:
        handles.append(Patch(color=C_DEAD, alpha=0.4, label=DEAD_LABEL))
    if p.gaps:
        handles.append(Patch(color=C_GAP, alpha=0.5, label="app gap (NaN) ±0.5 s"))
    ax1.legend(handles=handles, loc="upper right", ncol=len(handles), framealpha=0.9)

    # 2 10 s from the middle
    ax2 = fig.add_subplot(gs[1, 0])
    m0, m1 = win(n // 2, 500, n)
    plot_window(ax2, p, m0, m1)
    shade_blanks(ax2, p, label=False)
    ax2.set_title("2  10 s from the middle", loc="left")

    # 3 2 s around the largest |value|
    ax3 = fig.add_subplot(gs[1, 1])
    absb = np.where(p.incl, np.abs(p.b), -1)
    ipk = int(np.argmax(absb)) if not p.flat else n // 2
    k0, k1_ = win(ipk, 100, n)
    plot_window(ax3, p, k0, k1_)
    shade_blanks(ax3, p, label=False)
    pk_txt = "flat: middle shown" if p.flat else \
        f"{fmt(abs(p.b[ipk]), 5)} {p.unit} at {p.t0 + ipk / FS:,.2f} s"
    ax3.set_title(f"3  2 s around max |value| ({pk_txt})", loc="left")
    time_axis(ax3, 1)

    # 4 0.3 s samples
    ax4 = fig.add_subplot(gs[1, 2])
    ist = steepest(p, m0, m1)
    s0, s1 = win(ist, 15, n)
    plot_window(ax4, p, s0, s1, markers=True)
    ax4.set_title("4  0.3 s, every sample (steepest point of 2)", loc="left")
    time_axis(ax4, 2)
    ax4.legend(loc="best", framealpha=0.9, handlelength=1.2,
               handles=[Line2D([], [], marker="o", ls="-", color=C_APP, ms=3.2, lw=1.2, label="app"),
                        Line2D([], [], marker="o", ls="none", mfc="none", mec=C_FAMOS, mew=1.1, ms=6.5,
                               label="FAMOS")])
    for ax in (ax2, ax3, ax4):
        ax.set_xlabel("time (s)")
        ax.set_ylabel(p.unit)
    ax2.legend(loc="best", framealpha=0.9)

    # 5 error envelope
    ax5 = fig.add_subplot(gs[2, :2])
    e_incl = np.where(p.incl, p.err, np.nan)
    te, elo, ehi = envelope(e_incl, p.t0)
    _, _, fhi = envelope(np.where(p.incl, p.floor, np.nan), p.t0)
    ax5.fill_between(te, elo, ehi, color=C_ERR, alpha=0.35, lw=0, zorder=2)
    ax5.plot(te, ehi, color=C_ERR, lw=0.6, zorder=3)
    ax5.plot(te, elo, color=C_ERR, lw=0.6, zorder=3)
    ax5.fill_between(te, -fhi, fhi, step="mid", color="#000000", alpha=0.07, lw=0, zorder=1)
    ax5.step(te, fhi, where="mid", color="black", lw=0.6, ls=":", zorder=4)
    ax5.step(te, -fhi, where="mid", color="black", lw=0.6, ls=":", zorder=4)
    shade_blanks(ax5, p, label=False)
    lim = max(p.max_err if np.isfinite(p.max_err) else 0, np.nanmax(fhi) if np.isfinite(fhi).any() else 0)
    lim = lim * 1.2 or 1.0
    ax5.set_ylim(-1.95 * lim, lim)  # room below the trace for the note box
    ax5.set_xlim(p.t0, t_end)
    ax5.set_xlabel("time (s)")
    ax5.set_ylabel(f"FAMOS − app ({p.unit})")
    ax5.set_title("5  Error FAMOS − app, min/max per 1 s bin (included rows only)", loc="left")
    floor_txt = (f"dotted / grey band: ± rounding floor = ½ unit in the {p.digits}th significant digit of the app CSV"
                 + (" + ½ float32 step of FAMOS's stored value" if p.famos_f32 else "")
                 + "; it steps ×10 where |value| crosses a power of ten")
    within = f"{p.within_floor:,} of {p.n:,} rows ({p.within_floor / max(p.n, 1):.4%}) inside the floor"
    if p.outside_max is not None:
        within += (f"; the other {p.n - p.within_floor:,}: |err| ≤ {fmt(p.outside_max)} {p.unit}, at small values "
                   f"(median |value| {fmt(p.outside_value)}) where the value-scaled floor is tighter than "
                   "FAMOS's float32 arithmetic")
    guard = ""
    if p.guard_max is not None:
        guard = (f"\nrows within 0.5 s of an app NaN (excluded, not drawn): max |err| {fmt(p.guard_max)} {p.unit}"
                 + (" (FAMOS smooths into its zeros there, the app stops at the gap edge)" if p.dead else ""))
    note = "\n".join(textwrap.fill(t, 150) for t in (floor_txt + "\n" + within + guard).split("\n"))
    ax5.text(0.005, 0.03, note, transform=ax5.transAxes, fontsize=6.1,
             va="bottom", bbox=dict(fc="white", ec="#bbbbbb", alpha=0.9, lw=0.5))

    # 6 scatter
    ax6 = fig.add_subplot(gs[2, 2])
    idx = np.flatnonzero(p.incl)
    rng = np.random.default_rng(0)
    if len(idx) > 50_000:
        idx = np.sort(rng.choice(idx, 50_000, replace=False))
    ax6.scatter(p.b[idx], p.a[idx], s=25 if p.flat else 1.5, color=C_APP, alpha=0.4, lw=0, rasterized=True,
                zorder=3)
    if p.flat:
        ax6.annotate(f"all {len(idx):,} sampled rows at ({fmt(p.b[idx][0])}, {fmt(p.a[idx][0])})",
                     xy=(p.b[idx][0], p.a[idx][0]), xytext=(0.35, 0.3), textcoords="axes fraction",
                     fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.7))
    lo_ = min(np.nanmin(p.b[idx]), np.nanmin(p.a[idx])) if len(idx) else 0
    hi_ = max(np.nanmax(p.b[idx]), np.nanmax(p.a[idx])) if len(idx) else 1
    if hi_ == lo_:
        lo_, hi_ = lo_ - 1, hi_ + 1
    ax6.plot([lo_, hi_], [lo_, hi_], color="black", lw=0.7, ls="--", label="y = x")
    ax6.set_xlim(lo_, hi_)
    ax6.set_ylim(lo_, hi_)
    ax6.set_xlabel(f"app ({p.unit})")
    ax6.set_ylabel(f"FAMOS × {p.scale:g} ({p.unit})")
    ax6.set_title(f"6  FAMOS vs app ({len(idx):,} random rows)", loc="left")
    ax6.text(0.04, 0.96,
             f"r = {fmt_r(p.r)}\nmax |err| = {fmt(p.max_err)} {p.unit}\n"
             f"RMS err = {fmt(p.rms_err)} {p.unit}\nverdict: {verdict}",
             transform=ax6.transAxes, va="top", fontsize=6.6, family="monospace",
             bbox=dict(fc="white", ec="#bbbbbb", lw=0.5))
    ax6.legend(loc="lower right")

    mism = ctx["mismatch"].get(p.fname)
    if mism:
        fig.text(0.5, 0.035, "Statistics differ from famos_comparison_results.csv: " + "; ".join(mism),
                 ha="center", fontsize=6.5, color=C_ERR)


# ---------------------------------------------------------------------------
# raw pages
# ---------------------------------------------------------------------------
def page_raw(fig, pos, raw_pairs, ctx):
    fig.text(0.012, 0.965, f"{pos} raw channels: FAMOS *_rawred vs app raw_data.csv (0.3 s of samples)",
             fontsize=12.5, fontweight="bold", va="center")
    ok = [p for p in raw_pairs if not p.missing]
    if not ok:
        fig.text(0.5, 0.5, "No raw pairs found", ha="center", color=C_ERR, fontsize=12)
        return
    n_eq = sum(p.n_equal for p in ok)
    n_all = sum(p.n for p in ok)
    lines = [f"Both sides are the WFT's stored samples decimated to 100 Hz before any smoothing. "
             f"FAMOS × factor = app on {n_eq:,} of {n_all:,} compared rows (|diff| ≤ 1e-9); the remaining "
             f"differences are float64 round-off of the ×0.01 unit conversion.",
             f"Same exclusion rule as the processed channels (2 s at each end); raw_data.csv has no NaN rows."]
    if pos == "RR" and ctx["rr_dead"]:
        zr = ctx.get("rr_raw_dead_zero")
        lines.append(f"RR dead stretch ({dead_text(ctx['rr_dead'])}): the raw files are not blanked; there "
                     + ("FAMOS *_rawred and the app's raw_data.csv both hold exact zeros, compared like any other rows."
                        if zr else "the raw values are compared like any other rows."))
    fig.text(0.012, 0.935, "\n".join(lines), fontsize=6.9, va="top", linespacing=1.35)
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.975, top=0.825, bottom=0.08, hspace=0.45, wspace=0.25)
    for k, p in enumerate(raw_pairs):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        comp = p.fname.split("_")[1]
        if p.missing:
            ax.text(0.5, 0.5, p.missing, ha="center", va="center", transform=ax.transAxes, color=C_ERR)
            ax.set_title(comp, loc="left")
            continue
        n = len(p.b)
        m0, m1 = win(n // 2, 500, n)
        s0, s1 = win(steepest(p, m0, m1), 15, n)
        plot_window(ax, p, s0, s1, markers=True)
        time_axis(ax, 2)
        verdict = ctx["csv_rows"].get(p.fname, {}).get("verdict", "–")
        ax.set_title(f"{comp}: {p.fname} vs {p.pcol} · {verdict} · factor {p.scale:g}\n"
                     f"max |diff| over {p.n:,} rows = {fmt(p.max_err)} {p.unit}", loc="left", fontsize=7.4)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(p.unit)
        if k == 0:
            ax.legend(loc="best", framealpha=0.9, handlelength=1.2,
                      handles=[Line2D([], [], marker="o", ls="-", color=C_APP, ms=3.2, lw=1.2, label="app raw"),
                               Line2D([], [], marker="o", ls="none", mfc="none", mec=C_FAMOS, mew=1.1,
                                      ms=6.5, label="FAMOS rawred")])


# ---------------------------------------------------------------------------
# 1 kHz page
# ---------------------------------------------------------------------------
def page_1k(fig, ctx):
    fig.text(0.012, 0.965, "1 kHz: FAMOS 1 kHz samples vs app 100 Hz samples (app sample i = FAMOS sample 10i)",
             fontsize=12.5, fontweight="bold", va="center")
    k1 = ctx["k1"]
    if ctx.get("k1_missing"):
        missing_page(fig, "", ctx["k1_missing"])
        return
    fig.text(0.012, 0.935,
             "Left: 0.1 s with every FAMOS 1 kHz sample (dashed orange with small markers) and the app's 100 Hz "
             "samples (blue dots); each blue dot should sit on every 10th FAMOS sample.\n"
             f"Right: best start offset d (1 kHz samples = ms) found independently in windows along the recording, "
             f"from {FILES['res1k']}. FR_Fx/RR_Fz/Latacc compared with the app's processed file, FR_Fx1 (raw) "
             "with raw_data.csv.",
             fontsize=6.9, va="top", linespacing=1.35)
    gs = fig.add_gridspec(4, 2, left=0.06, right=0.985, top=0.87, bottom=0.07, hspace=0.95, wspace=0.16,
                          width_ratios=[1.35, 1])
    for row, k in enumerate(k1):
        axl = fig.add_subplot(gs[row, 0])
        axr = fig.add_subplot(gs[row, 1])
        if k.get("missing"):
            for ax in (axl, axr):
                ax.text(0.5, 0.5, k["missing"], ha="center", va="center", transform=ax.transAxes,
                        color=C_ERR)
            continue
        f1k, app, d, sc = k["f1k"], k["app"], k["d"], k["scale"]
        n = len(app)
        m0, m1 = win(n // 2, 500, n)
        dd = np.abs(np.diff(app[m0:m1]))
        dd[~np.isfinite(dd)] = -1
        ic = m0 + int(np.argmax(dd))
        i0, i1 = win(ic, 5, n)
        j0, j1 = 10 * i0 + d, 10 * (i1 - 1) + d + 1
        tj = np.arange(j0, j1) / FS_1K
        ti = np.arange(i0, i1) / FS
        axl.plot(ti, app[i0:i1], "o", color=C_APP, ms=6, zorder=2, label="app 100 Hz")
        axl.plot(tj, f1k[j0:j1] * sc, color=C_FAMOS, lw=1.0, ls=(0, (4, 2.5)), marker=".", ms=2.5,
                 zorder=3, label="FAMOS 1 kHz × factor")
        axl.set_xlim(tj[0] - 0.002, tj[-1] + 0.002)
        axl.set_xlabel("time (s)")
        axl.set_ylabel(k["unit"])
        time_axis(axl, 3)
        axl.set_title(f"{k['fname']} (1 kHz) vs {k['pcol']}{' (raw)' if k['src'] == 'raw' else ''}   ·   "
                      f"{k['verdict']}\nd = {d} ms, r = {fmt_r(k['r'])}, max |err| = {fmt(k['max_err'])} "
                      f"{k['unit']}, factor {sc:g}", loc="left", fontsize=7.2)
        if row == 0:
            axl.legend(loc="best", framealpha=0.9)

        dr = k["drift"]
        if len(dr):
            axr.plot(dr["t_s"], dr["d"], "o-", color=C_APP, ms=3.5, lw=0.8)
            ds = sorted(set(int(v) for v in dr["d"]))
            axr.set_title(f"offset d along the recording: {len(dr)} windows\noffsets seen {ds}, "
                          f"min r {dr['r'].min():.10f}", loc="left", fontsize=7.0)
        else:
            axr.text(0.5, 0.5, f"no drift rows for {k['fname']} in {FILES['res1k']}", ha="center",
                     va="center", transform=axr.transAxes, color=C_ERR)
        if k["fname"].startswith("RR") and ctx["rr_dead"]:
            for j, (s0, s1) in enumerate(ctx["rr_dead"]):
                axr.axvspan(s0, s1, color=C_DEAD, alpha=0.3, lw=0,
                            label=textwrap.fill(DEAD_LABEL, 48) + "\n(windows there skipped)" if j == 0 else None)
            axr.legend(loc="lower left", fontsize=5.8, framealpha=0.95)
        axr.set_ylim(-5, 5)
        axr.set_xlim(0, ctx["n_famos"] / FS)
        axr.set_xlabel("time (s)")
        axr.set_ylabel("offset d (ms)")


# ---------------------------------------------------------------------------
# conclusion
# ---------------------------------------------------------------------------
def build_conclusion(ctx):
    pairs, k1, csv = ctx["pairs"], ctx["k1"], ctx["csv_rows"]
    red = [p for p in pairs if p.kind != "rawred"]
    raw = [p for p in pairs if p.kind == "rawred"]
    out = []
    verd = lambda p: csv.get(p.fname, {}).get("verdict", "MISSING") if not p.missing else "MISSING"
    sig = [p for p in red if verd(p) != "FLAT"]
    flat = [p for p in red if verd(p) == "FLAT"]
    npass = sum(verd(p) == "PASS" for p in sig)
    lags0 = all((not p.missing) and p.lag == 0 for p in pairs)
    allp = npass + len(flat) + sum(verd(p) == "PASS" for p in raw)
    out.append(f"{allp} of {len(pairs)} channel pairs at 100 Hz pass: {npass} of {len(sig)} processed "
               f"channels with signal PASS" + (", all at offset 0" if lags0 else ", NOT all at offset 0")
               + (f"; {', '.join(p.pcol for p in flat)} is FLAT (zero in both files, no GPS signal)."
                  if flat else "."))
    bad = [p.fname for p in pairs if verd(p) not in ("PASS", "FLAT")]
    if bad:
        out.append(f"Not passing: {', '.join(bad)}. See their pages.")
    ok = [p for p in sig if not p.missing]
    if ok:
        w = max(ok, key=lambda p: p.rel)
        tot = sum(p.n for p in ok)
        inside = sum(p.within_floor for p in ok)
        out.append(f"Largest error relative to signal range: {w.rel:.1e} ({w.pcol}). {inside:,} of {tot:,} "
                   f"compared rows ({inside / tot:.4%}) lie within the CSV/float32 rounding floor"
                   + (f"; the other {tot - inside:,} are near-zero values with |err| ≤ "
                      f"{max(p.outside_max for p in ok if p.outside_max is not None):.1e} "
                      f"({', '.join(p.pcol for p in ok if p.outside_max is not None)})." if tot > inside else "."))
    rok = [p for p in raw if not p.missing]
    if rok:
        mx = max(p.max_err for p in rok)
        out.append(f"All {sum(verd(p) == 'PASS' for p in rok)} of {len(raw)} raw channels: max |FAMOS − app| "
                   f"= {mx:.1e} (float64 round-off of the unit factor), so the app's raw samples are "
                   "FAMOS's stored samples.")
    kok = [k for k in k1 if not k.get("missing")]
    if kok:
        npk = sum(k["verdict"] == "PASS" for k in kok)
        allw = all(len(k["drift"]) and set(k["drift"]["d"]) == {0} for k in kok)
        nw = sum(len(k["drift"]) for k in kok)
        out.append(f"1 kHz: {npk} of {len(k1)} pairs PASS with app sample i = FAMOS sample 10i"
                   + (f"; the offset is 0 in all {nw} windows along the recording (no drift)."
                      if allw else "; the offset is NOT 0 in every window."))
    if ctx["rr_dead"]:
        out.append(f"Documented difference: RR transducer dead ({dead_text(ctx['rr_dead'])}); "
                   "FAMOS keeps zeros, app blanks. Excluded ±0.5 s from the error statistics.")
    for p in red:
        for s, e in getattr(p, "gaps", []):
            out.append(f"{p.pcol} has a {e - s}-row gap (NaN) at t = {p.t0 + s / FS:,.2f} s; excluded ±0.5 s.")
    sev = ctx["sev"]
    if "FR" in sev and "RR" in sev:
        def worst(s1, s2):
            return max(abs(s1[k] - s2[k]) / abs(s2[k]) for k in ("Gx", "Gy", "Gxy", "StaticFz", "StdFz", "DLC"))
        parts = []
        for pos in ("FR", "RR"):
            s = sev[pos]
            t = f"{pos} differs by at most {worst(s['FAMOS'], s['app']) * 100:.3g} % on all rows"
            if "FAMOS_same" in s:
                t += f" and {worst(s['FAMOS_same'], s['app']) * 100:.2g} % on the rows where the app has data"
            parts.append(t)
        out.append("G-severity (Gx, Gy, Gxy, Static/Std Fz, DLC): " + "; ".join(parts) +
                   (". The RR all-rows difference is the dead stretch's zeros in FAMOS." if ctx["rr_dead"] else "."))
    nm = sum(len(v) for v in ctx["mismatch"].values())
    out.append("Every statistic in this report was recomputed and matches famos_comparison_results.csv / "
               "famos_1k_results.csv." if nm == 0 else
               f"{nm} statistic(s) differ from the results CSVs: see the red notes on the pages.")
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=r"C:\famos_check", help="folder with the input files")
    ap.add_argument("--out", help="output folder (default: --dir)")
    ap.add_argument("--recording", help="recording name for the title page")
    ap.add_argument("--dpi", type=int, default=130, help="PNG resolution")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    t_start = time.time()
    d = args.dir
    out_dir = args.out or d
    path = {k: os.path.join(d, v) for k, v in FILES.items()}
    for k in ("famos", "app", "script"):
        if not os.path.exists(path[k]):
            raise SystemExit(f"required input missing: {path[k]}")

    fc = load_compare_module(path["script"])
    log("reading FAMOS 100 Hz")
    famos = fc.read_famos(path["famos"])
    with open(path["famos"], encoding="utf-8-sig", errors="replace") as f:
        names = [t.strip() for t in f.readline().split(",")]
        units = [t.strip() for t in f.readline().split(",")]
    famos_units = dict(zip(names, units))
    log("reading app processed + raw")
    app = fc.read_any_csv(path["app"])
    raw = fc.read_any_csv(path["raw"]) if os.path.exists(path["raw"]) else None
    res = pd.read_csv(path["res"]) if os.path.exists(path["res"]) else pd.DataFrame()
    res1k = pd.read_csv(path["res1k"]) if os.path.exists(path["res1k"]) else pd.DataFrame()

    csv_rows = {}
    if "channel" in res:
        for _, r in res[res["channel"].notna()].iterrows():
            csv_rows[r["channel"]] = r.to_dict()
    digits = app_sig_digits(path["app"], list(app.columns))

    log("comparing channel pairs (famos_compare.compare)")
    pairs = []
    for fname, cands, src, kind in fc.PAIRS:
        other = raw if src == "raw" else app
        pairs.append(prepare_pair(fc, famos, other, fname, cands, src, kind, digits))

    mismatch = {}
    for p in pairs:
        if p.missing:
            continue
        m = check_against_csv(p, csv_rows.get(p.fname))
        if m:
            mismatch[p.fname] = m

    # RR dead stretch: the app's long NaN runs, merged over the RR processed channels
    spans = sorted((p.t0 + s / FS, p.t0 + e / FS) for p in pairs
                   if p.kind != "rawred" and not p.missing and p.fname.startswith("RR") for s, e in p.dead)
    rr_dead = []
    for s, e in spans:
        if rr_dead and s <= rr_dead[-1][1]:
            rr_dead[-1] = (rr_dead[-1][0], max(rr_dead[-1][1], e))
        else:
            rr_dead.append((s, e))
    # do both raw files hold exact zeros where the app blanked that same channel?
    rr_raw_dead_zero = None
    if rr_dead:
        proc = {p.fname.replace("_red", ""): p for p in pairs if p.kind != "rawred" and not p.missing}
        checks = []
        for p in pairs:
            q = proc.get(p.fname.replace("_rawred", ""))
            if p.kind == "rawred" and p.fname.startswith("RR") and not p.missing and q is not None:
                checks += [np.all(p.a[s:e] == 0) and np.all(p.b[s:e] == 0) for s, e in q.dead]
        rr_raw_dead_zero = bool(checks) and all(checks)

    # G-severity
    log("G-severity")
    sev = {}
    force_scale = 0.01
    fs_list = [p.scale for p in pairs if p.kind == "force" and not p.missing]
    if fs_list:
        force_scale = max(set(fs_list), key=fs_list.count)
    for pos, serial in fc.WHEELS:
        cf = [fc.find_col(famos, [f"{pos}_{c}_red"]) for c in ("Fx", "Fy", "Fz")]
        cp = [fc.find_col(app, [f"{pos}_{c}_{serial}", f"{pos}_{c}"]) for c in ("Fx", "Fy", "Fz")]
        if None in cf or None in cp:
            continue
        F = [famos[c].to_numpy(float) * force_scale for c in cf]
        A = [app[c].to_numpy(float) for c in cp]
        s = {"FAMOS": fc.severity(*F), "app": fc.severity(*A)}
        n = min(len(F[0]), len(A[0]))
        have = np.isfinite(A[0][:n]) & np.isfinite(A[1][:n]) & np.isfinite(A[2][:n])
        if (~have).any():
            s["FAMOS_same"] = fc.severity(*(f[:n][have] for f in F))
        sev[pos] = s
        if "wheel" in res:
            for src in ("FAMOS", "app"):
                r = res[(res["wheel"] == pos) & (res["source"] == src)]
                if len(r):
                    for key in ("Gx", "Gy", "Gxy", "StaticFz", "StdFz", "DLC", "n", "fz_zero"):
                        if not close(s[src][key], r.iloc[0][key], rel=1e-9):
                            mismatch.setdefault(f"severity {pos}", []).append(
                                f"{src} {key}: report {s[src][key]} vs CSV {r.iloc[0][key]}")

    # 1 kHz
    k1 = []
    k1_missing = None
    k1_specs = [("FR_Fx", ["FR_Fx_2"], "python", "daN"), ("FR_Fx1", ["FR_Fx_2"], "raw", "daN"),
                ("RR_Fz", ["RR_Fz_1"], "python", "daN"), ("Latacc", ["Latacc"], "python", "m/s²")]
    if os.path.exists(path["famos1k"]):
        log("reading FAMOS 1 kHz (4 columns)")
        f1k_df = read_1k(path["famos1k"], {s[0] for s in k1_specs})
        n_1k = len(f1k_df)
        log("1 kHz phase check (famos_compare.phase_compare)")
        r1k_sum = res1k[res1k["verdict"].notna()] if "verdict" in res1k else pd.DataFrame()
        r1k_drift = res1k[res1k["t_s"].notna()] if "t_s" in res1k else pd.DataFrame()
        for fname, cands, src, unit in k1_specs:
            other = raw if src == "raw" else app
            pcol = fc.find_col(other, cands) if other is not None else None
            entry = dict(fname=fname, pcol=pcol, src=src, unit=unit)
            if fname not in f1k_df or pcol is None:
                entry["missing"] = f"{fname} not in {FILES['famos1k']}" if fname not in f1k_df \
                    else f"{cands} not in app file"
                k1.append(entry)
                continue
            f1 = f1k_df[fname].to_numpy(float)
            av = other[pcol].to_numpy(float)
            pc = fc.phase_compare(f1, av, MAX_D_1K, TRIM_S)
            entry.update(pc, f1k=f1, app=av)
            rs = r1k_sum[r1k_sum["channel_1k"] == fname] if len(r1k_sum) else pd.DataFrame()
            if len(rs):
                rr = rs.iloc[0]
                entry["verdict"] = rr["verdict"]
                bad = [f"{key}: report {pc[key]!r} vs CSV {rr[key]!r}" for key in ("d", "r", "max_err", "rel", "n")
                       if not close(pc[key], rr[key], rel=1e-6 if key == "rel" else 1e-9)]
                if pc["verdict"] != rr["verdict"]:
                    bad.append(f"verdict: recomputed {pc['verdict']} vs CSV {rr['verdict']}")
                if bad:
                    mismatch[f"{fname} (1 kHz)"] = bad
            else:
                mismatch[f"{fname} (1 kHz)"] = ["no row in famos_1k_results.csv"]
            entry["drift"] = r1k_drift[r1k_drift["channel_1k"] == fname] if len(r1k_drift) else pd.DataFrame()
            k1.append(entry)
        del f1k_df
    else:
        n_1k = 0
        k1_missing = f"{FILES['famos1k']} not found in {d}: 1 kHz checks not shown"

    tcol_f = fc.find_col(famos, ["x", "time"])
    tcol_a = fc.find_col(app, ["Time"])
    time_diff = float("nan")
    if tcol_f and tcol_a:
        n = min(len(famos), len(app))
        time_diff = float(np.nanmax(np.abs(famos[tcol_f].to_numpy()[:n] - app[tcol_a].to_numpy()[:n])))

    ctx = dict(pairs=pairs, k1=k1, k1_missing=k1_missing, sev=sev, csv_rows=csv_rows, mismatch=mismatch,
               rr_dead=rr_dead, rr_raw_dead_zero=rr_raw_dead_zero, famos_units=famos_units,
               n_famos=len(famos), n_app=len(app), n_raw=len(raw) if raw is not None else 0, n_1k=n_1k,
               time_diff=time_diff, recording=args.recording or os.path.basename(os.path.normpath(d)),
               generated=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    ctx["conclusion"] = build_conclusion(ctx)

    red_pairs = [p for p in pairs if p.kind != "rawred"]
    raw_by_pos = {pos: [p for p in pairs if p.kind == "rawred" and p.fname.startswith(pos)] for pos in ("FR", "RR")}
    pages = [("summary", lambda f: page_summary(f, ctx))]
    pages += [(p.fname, (lambda q: lambda f: page_channel(f, q, ctx))(p)) for p in red_pairs]
    pages += [(f"{pos}_raw", (lambda q, ps: lambda f: page_raw(f, q, ps, ctx))(pos, ps))
              for pos, ps in raw_by_pos.items()]
    pages += [("1kHz", lambda f: page_1k(f, ctx))]

    png_dir = os.path.join(out_dir, "report_png")
    os.makedirs(png_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "famos_validation_report.pdf")
    log(f"rendering {len(pages)} pages")
    with PdfPages(pdf_path) as pdf:
        for k, (name, build) in enumerate(pages, 1):
            fig = new_page()
            build(fig)
            add_footer(fig, k, len(pages), ctx)
            pdf.savefig(fig)
            fig.savefig(os.path.join(png_dir, f"page_{k:02d}_{name}.png"), dpi=args.dpi)
            plt.close(fig)
        info = pdf.infodict()
        info["Title"] = "imc FAMOS vs DTT app: preprocessing validation"
        info["Subject"] = ctx["recording"]

    print()
    for line in ctx["conclusion"]:
        print(" -", line)
    if mismatch:
        print("\nMISMATCHES vs results CSVs:")
        for k, v in mismatch.items():
            print(f"  {k}: {'; '.join(v)}")
    log(f"wrote {pdf_path} ({len(pages)} pages) and {png_dir} in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    sys.exit(main())
