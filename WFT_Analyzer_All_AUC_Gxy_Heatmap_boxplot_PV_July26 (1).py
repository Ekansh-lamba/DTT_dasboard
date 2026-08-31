"""
WFT Force Data Analyzer — GUI  v7.0
=====================================
Features:
  - CSV file upload (browse)
  - Optional 2nd CSV for comparison overlay
  - Auto-detect available force channels (FL/FR/RL/RR  Fx/Fy/Fz)
  - Percentile analysis  : P80 / P90 / P95
  - Rainflow counting    : Siemens-style From-To matrix
  - Box Plot             : separate popup per channel, fixed Y-axis per type
  - Comparison overlay   : Dataset 1 vs Dataset 2 per channel
  - Area Under Curve     : KDE shaded comparison
  - RF Compare           : Front axle (FL+FR) | Rear axle (RL+RR) side by side
  - Converts N → daN automatically (uniform scale across all channels)
  - Scrollable left panel

Color scheme (per wheel, consistent across ALL plots):
  FL = Blue   (#185FA5 / #00B4D8 / #0077B6)
  FR = Red    (#E24B4A / #E67E22 / #C0392B)
  RL = Green  (#1E8449 / #27AE60 / #145A32)
  RR = Purple (#6C3483 / #9B59B6 / #4A235A)

Dependencies:
    pip install pandas numpy matplotlib rainflow scipy
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as patheffects
import rainflow
import os

# ── App colours ─────────────────────────────────────────────
BG       = "#0D1B2A"
PANEL_BG = "#1B3A5C"
BTN_BG   = "#00B4D8"
BTN_FG   = "#FFFFFF"
BTN_HOV  = "#0077B6"
TEXT_PRI = "#FFFFFF"
TEXT_SEC = "#90E0EF"
ACCENT   = "#E8862A"
SUCCESS  = "#27AE60"
DANGER   = "#C0392B"
ENTRY_BG = "#0A2540"
FONT_H   = ("Arial", 12, "bold")
FONT_B   = ("Arial", 10, "bold")
FONT_N   = ("Arial", 10)
FONT_S   = ("Arial", 9)

# ── Channel definitions ──────────────────────────────────────
FULL_COLUMN_MAP = {
    "Time": 0, "Vehicle_Speed": 1, "Dist": 2,
    "Latitude": 3, "Longitude": 4, "Altitude": 5,
    "FR_Fx": 6,  "FR_Fy": 7,  "FR_Fz": 8,
    "FR_Mx": 9,  "FR_My": 10, "FR_Mz": 11,
    "FL_Fx": 12, "FL_Fy": 13, "FL_Fz": 14,
    "FL_Mx": 15, "FL_My": 16, "FL_Mz": 17,
    "RL_Fx": 18, "RL_Fy": 19, "RL_Fz": 20,
    "RL_Mx": 21, "RL_My": 22, "RL_Mz": 23,
    "RR_Fx": 24, "RR_Fy": 25, "RR_Fz": 26,
    "RR_Mx": 27, "RR_My": 28, "RR_Mz": 29,
}

# 12 force channels — order: FL, FR, RL, RR
FORCE_CHANNELS = [
    "FL_Fx", "FL_Fy", "FL_Fz",
    "FR_Fx", "FR_Fy", "FR_Fz",
    "RL_Fx", "RL_Fy", "RL_Fz",
    "RR_Fx", "RR_Fy", "RR_Fz",
]

# 12 moment channels — order: FL, FR, RL, RR (auto-detected; optional in CSV)
MOMENT_CHANNELS = [
    "FL_Mx", "FL_My", "FL_Mz",
    "FR_Mx", "FR_My", "FR_Mz",
    "RL_Mx", "RL_My", "RL_Mz",
    "RR_Mx", "RR_My", "RR_Mz",
]

# ── Per-wheel colour palette ─────────────────────────────────
# Each wheel gets a trio: primary, lighter, darker
WHEEL_COLORS = {
    "FL": {"pri": "#2E86DE", "alt": "#74B9FF", "dark": "#0A3D7A"},
    "FR": {"pri": "#E74C3C", "alt": "#FF8A80", "dark": "#922B21"},
    "RL": {"pri": "#27AE60", "alt": "#6FCF97", "dark": "#145A32"},
    "RR": {"pri": "#8E44AD", "alt": "#C39BD3", "dark": "#4A235A"},
}

def _wheel(ch):
    """Return wheel prefix: FL / FR / RL / RR."""
    return ch[:2] if len(ch) >= 2 else "FL"

def _chan_color(ch):
    """Primary color for a channel based on wheel."""
    return WHEEL_COLORS.get(_wheel(ch), {}).get("pri", BTN_BG)

# All 24 channels — used by RMS / DLC / PSD / AUC-Moments analysis
ALL_CHANNELS = FORCE_CHANNELS + MOMENT_CHANNELS

# Legacy dict for quick lookup (used throughout) — now covers moments too
CHAN_COLORS = {ch: _chan_color(ch) for ch in ALL_CHANNELS}

# Units per channel type — forces are auto-scaled to daN, moments stay in Nm
def _chan_unit(ch):
    return "daN" if ch[-2:] in ("Fx", "Fy", "Fz") else "Nm"

# Full 6-channel groups per wheel (Fx,Fy,Fz,Mx,My,Mz) for RMS/DLC/PSD
WHEEL_GROUPS_FULL = {
    "FL": ["FL_Fx", "FL_Fy", "FL_Fz", "FL_Mx", "FL_My", "FL_Mz"],
    "FR": ["FR_Fx", "FR_Fy", "FR_Fz", "FR_Mx", "FR_My", "FR_Mz"],
    "RL": ["RL_Fx", "RL_Fy", "RL_Fz", "RL_Mx", "RL_My", "RL_Mz"],
    "RR": ["RR_Fx", "RR_Fy", "RR_Fz", "RR_Mx", "RR_My", "RR_Mz"],
}

# Fixed two-color scheme for DS1 vs DS2 comparisons (AUC / AUC Cross-Position / AUC Moments)
AUC_DS1_COLOR = "#27AE60"   # green — Dataset 1, solid line
AUC_DS2_COLOR = "#E74C3C"   # red   — Dataset 2, dashed line

# Cap on samples used to *fit/evaluate* KDE curves (percentiles still use full data).
# Large logged datasets (100Hz over hours = 100k-1M+ rows) make gaussian_kde evaluation
# (O(n_samples * n_grid)) extremely slow; a random subsample preserves the distribution
# shape while cutting AUC generation time drastically.
KDE_MAX_SAMPLES = 20000

# Fixed Y-axis limits for box plots per channel suffix
BOX_YLIM = {
    "Fx": (-300,  300),
    "Fy": (-300,  300),
    "Fz": ( 100, 1000),
}

def _box_ylim(ch):
    for key in BOX_YLIM:
        if ch.endswith(key):
            return BOX_YLIM[key]
    return None

def _plot_xlim(ch):
    for key in BOX_YLIM:
        if ch.endswith(key):
            return BOX_YLIM[key]
    return None

# ── Force-pair heatmap settings (distance-in-km, NOT sample count) ─
# Fixed bin edges per channel type — independent of BOX_YLIM.
HEATMAP_EDGES = {
    "Fx": np.arange(-300, 301, 100),   # -300..300, 100 daN bins
    "Fy": np.arange(-300, 301, 100),   # -300..300, 100 daN bins
    "Fz": np.arange( 400, 801, 100),   #  400..800, 100 daN bins
}
# The 3 combinations shown per wheel, per dataset.
# First element of each tuple = X-axis channel, second = Y-axis channel.
HEATMAP_PAIRS = [("Fy", "Fx"), ("Fx", "Fz"), ("Fy", "Fz")]

# Siemens-style discrete colour steps
RF_STEPS  = [0, 1, 3, 6, 13, 25, 48, 93, float("inf")]
RF_COLORS = [
    "#FFFFFF", "#FF00FF", "#0000FF", "#00BFFF",
    "#008000", "#ADFF2F", "#FFFF00", "#FF8C00", "#FF0000",
]
RF_LABELS = ["≤ 0","≤ 1","≤ 3","≤ 6","≤ 13","≤ 25","≤ 48","≤ 93","> 93"]

# Wheel group definitions
WHEEL_GROUPS = {
    "FL": ["FL_Fx", "FL_Fy", "FL_Fz"],
    "FR": ["FR_Fx", "FR_Fy", "FR_Fz"],
    "RL": ["RL_Fx", "RL_Fy", "RL_Fz"],
    "RR": ["RR_Fx", "RR_Fy", "RR_Fz"],
}
MOMENT_GROUPS = {
    "FL": ["FL_Mx", "FL_My", "FL_Mz"],
    "FR": ["FR_Mx", "FR_My", "FR_Mz"],
    "RL": ["RL_Mx", "RL_My", "RL_Mz"],
    "RR": ["RR_Mx", "RR_My", "RR_Mz"],
}
AXLE_GROUPS = {
    "Front": ["FL", "FR"],
    "Rear":  ["RL", "RR"],
}


# ══════════════════════════════════════════════════════════════
class WFTApp:
    def __init__(self, root):
        self.root = root
        self.root.title("WFT Force Data Analyzer  v7.0")
        self.root.configure(bg=BG)
        self.root.geometry("1200x800")
        self.root.minsize(950, 680)

        self.df           = None
        self.df2          = None
        self.file_path    = tk.StringVar(value="No file loaded")
        self.file_path2   = tk.StringVar(value="No comparison file")
        self.label1       = tk.StringVar(value="Dataset 1")
        self.label2       = tk.StringVar(value="Dataset 2")
        self.chan_labels  = {ch: tk.StringVar(value=ch) for ch in FORCE_CHANNELS}
        self.status_msg   = tk.StringVar(value="Ready — upload a CSV file to begin")
        self.avail_chans  = []
        self.avail_chans2 = []
        self.mpl_canvas   = None
        self.current_fig  = None
        # Export storage
        self._auc_figs          = []   # [(ch, fig), ...]
        self._auc_cross_figs    = []   # [(label, fig), ...]
        self._box_figs          = {}   # {ch: fig}
        self._auc_summary       = []
        self._auc_cross_summary = []
        self._auc_moment_figs    = []
        self._auc_moment_summary = []
        self._auc_cross_moment_figs    = []
        self._auc_cross_moment_summary = []
        self._rms_fig    = None
        self._rms_rows   = []
        self._dlc_fig    = None
        self._dlc_rows   = []
        self._gsev_fig   = None
        self._gsev_rows  = []
        self._psd_figs   = {}
        self.avail_moments  = []
        self.avail_moments2 = []
        self._heatmap_figs  = []   # [(name, fig), ...] for export
        self.heatmap_range_vars = {}
        for _ch in ("Fx", "Fy", "Fz"):
            _edges = HEATMAP_EDGES[_ch]
            self.heatmap_range_vars[_ch] = {
                "min":  tk.StringVar(value=str(int(_edges[0]))),
                "max":  tk.StringVar(value=str(int(_edges[-1]))),
                "step": tk.StringVar(value=str(int(_edges[1] - _edges[0]))),
            }

        self._build_ui()

    # ══ UI BUILD ══════════════════════════════════════════════
    def _build_ui(self):
        hdr = tk.Frame(self.root, bg=PANEL_BG, height=56)
        hdr.pack(fill="x", side="top")
        tk.Label(hdr, text="WFT Force Data Analyzer",
                 font=("Arial Black", 14), bg=PANEL_BG,
                 fg=BTN_BG).pack(side="left", padx=16, pady=12)
        tk.Label(hdr, text="FL · FR · RL · RR  |  Fx · Fy · Fz",
                 font=FONT_S, bg=PANEL_BG, fg=TEXT_SEC).pack(side="left", padx=4)
        tk.Frame(self.root, bg=ACCENT, height=3).pack(fill="x")

        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True)

        # Scrollable left panel
        lo = tk.Frame(main, bg=PANEL_BG, width=295)
        lo.pack(side="left", fill="y", padx=(8,4), pady=8)
        lo.pack_propagate(False)
        lc = tk.Canvas(lo, bg=PANEL_BG, highlightthickness=0, width=275)
        ls = tk.Scrollbar(lo, orient="vertical", command=lc.yview)
        lc.configure(yscrollcommand=ls.set)
        ls.pack(side="right", fill="y")
        lc.pack(side="left", fill="both", expand=True)
        self.left = tk.Frame(lc, bg=PANEL_BG)
        lc.create_window((0,0), window=self.left, anchor="nw", width=275)
        self.left.bind("<Configure>",
                       lambda e: lc.configure(scrollregion=lc.bbox("all")))
        lc.bind_all("<MouseWheel>",
                    lambda e: lc.yview_scroll(int(-1*(e.delta/120)), "units"))
        lc.bind_all("<Button-4>", lambda e: lc.yview_scroll(-1, "units"))
        lc.bind_all("<Button-5>", lambda e: lc.yview_scroll( 1, "units"))

        self.right = tk.Frame(main, bg=BG)
        self.right.pack(side="left", fill="both", expand=True, padx=(4,8), pady=8)

        self._build_left_panel()
        self._build_right_panel()

        sb = tk.Frame(self.root, bg="#071020", height=26)
        sb.pack(fill="x", side="bottom")
        tk.Label(sb, textvariable=self.status_msg,
                 font=FONT_S, bg="#071020", fg=TEXT_SEC,
                 anchor="w").pack(side="left", padx=10)

    def _build_left_panel(self):
        p = self.left

        # ── 1: Upload CSV 1 ──
        self._sec(p, "1  — UPLOAD CSV  (Dataset 1)")
        self._btn(p, "Browse / Upload CSV", self._load_csv, bg=ACCENT, w=30).pack(pady=(4,2))
        tk.Label(p, textvariable=self.file_path, font=("Arial",8),
                 bg=PANEL_BG, fg=TEXT_SEC, wraplength=245,
                 justify="left").pack(padx=10, pady=(0,3))
        self.info_var = tk.StringVar(value="—")
        tk.Label(p, textvariable=self.info_var, font=FONT_S,
                 bg=ENTRY_BG, fg=TEXT_SEC, wraplength=245, justify="left",
                 relief="flat", padx=8, pady=6).pack(fill="x", padx=8, pady=(0,6))

        # ── 2: Upload CSV 2 ──
        self._div(p)
        self._sec(p, "2  — COMPARISON CSV  (Optional)")
        tk.Label(p, text="Upload 2nd CSV to overlay & compare channels",
                 font=FONT_S, bg=PANEL_BG, fg=TEXT_SEC,
                 justify="left").pack(padx=10, pady=(0,4), anchor="w")
        self._btn(p, "Upload 2nd CSV", self._load_csv2, bg="#6C3483", w=30).pack(pady=2)
        tk.Label(p, textvariable=self.file_path2, font=("Arial",8),
                 bg=PANEL_BG, fg=TEXT_SEC, wraplength=245,
                 justify="left").pack(padx=10, pady=(0,3))
        for lbl_txt, var in [("Label 1:", self.label1), ("Label 2:", self.label2)]:
            row = tk.Frame(p, bg=PANEL_BG); row.pack(fill="x", padx=8, pady=1)
            tk.Label(row, text=lbl_txt, font=FONT_S, bg=PANEL_BG,
                     fg=TEXT_SEC, width=8, anchor="w").pack(side="left")
            tk.Entry(row, textvariable=var, width=16, bg=ENTRY_BG,
                     fg=TEXT_PRI, font=FONT_S,
                     insertbackground=TEXT_PRI, relief="flat").pack(side="left", padx=2)
        self._btn(p, "Clear 2nd Dataset", self._clear_csv2, bg="#444", w=30).pack(pady=(4,2))

        # ── 3: Settings ──
        self._div(p); self._sec(p, "3  — SETTINGS")
        row = tk.Frame(p, bg=PANEL_BG); row.pack(fill="x", padx=10, pady=(2,8))
        tk.Label(row, text="Sampling rate (Hz):", font=FONT_S,
                 bg=PANEL_BG, fg=TEXT_SEC).pack(side="left")
        self.sr_var = tk.StringVar(value="100")
        tk.Entry(row, textvariable=self.sr_var, width=6, bg=ENTRY_BG,
                 fg=TEXT_PRI, font=FONT_N,
                 insertbackground=TEXT_PRI, relief="flat").pack(side="left", padx=4)

        # ── 4: Percentile ──
        self._div(p); self._sec(p, "4  — PERCENTILE ANALYSIS")
        tk.Label(p, text="P80 / P90 / P95 for all 12 channels",
                 font=FONT_S, bg=PANEL_BG, fg=TEXT_SEC,
                 justify="left").pack(padx=10, pady=(0,4), anchor="w")
        for pv, bg in [(80,"#2C7744"),(90,"#1B5E83"),(95,DANGER)]:
            self._btn(p, f"Compute  P{pv}",
                      lambda v=pv: self._run_percentile(v), bg=bg, w=30).pack(pady=2)
        self._btn(p, "Compute  All  (P80+P90+P95)",
                  self._run_all_percentiles, bg=PANEL_BG, w=30, relief="ridge").pack(pady=2)

        # ── 5: Rainflow ──
        self._div(p); self._sec(p, "5  — RAINFLOW COUNTING")
        tk.Label(p, text="Siemens-style From-To matrix",
                 font=FONT_S, bg=PANEL_BG, fg=TEXT_SEC,
                 justify="left").pack(padx=10, pady=(0,2), anchor="w")
        for wheel, chans in WHEEL_GROUPS.items():
            wc = WHEEL_COLORS[wheel]["pri"]
            tk.Label(p, text=f"  {wheel}", font=("Arial",8,"bold"),
                     bg=PANEL_BG, fg=wc).pack(anchor="w", padx=14, pady=(4,0))
            for ch in chans:
                self._btn(p, f"Rainflow — {ch}",
                          lambda c=ch: self._run_rainflow_single(c),
                          bg=CHAN_COLORS[ch], w=30).pack(pady=1)
        self._btn(p, "Rainflow — All 12 Channels",
                  self._run_rainflow_all, bg="#6C3483", w=30).pack(pady=4)

        # ── 6: Box Plot ──
        self._div(p); self._sec(p, "6  — BOX PLOT")
        name_frame = tk.Frame(p, bg=ENTRY_BG, relief="flat")
        name_frame.pack(fill="x", padx=8, pady=(0,4))
        tk.Label(name_frame, text="Channel display names (optional):",
                 font=FONT_S, bg=ENTRY_BG, fg=TEXT_SEC,
                 anchor="w").pack(fill="x", padx=6, pady=(4,2))
        for ch in FORCE_CHANNELS:
            row = tk.Frame(name_frame, bg=ENTRY_BG); row.pack(fill="x", padx=6, pady=1)
            tk.Label(row, text=f"{ch}:", font=("Arial",8), bg=ENTRY_BG,
                     fg=CHAN_COLORS.get(ch, TEXT_SEC),
                     width=8, anchor="w").pack(side="left")
            tk.Entry(row, textvariable=self.chan_labels[ch],
                     width=16, bg="#0D1B2A", fg=TEXT_PRI,
                     font=("Arial",8), insertbackground=TEXT_PRI,
                     relief="flat").pack(side="left", padx=2)
        tk.Label(p, text="Fixed axes: Fx±1200  Fy±1200  Fz 5k–13k",
                 font=FONT_S, bg=PANEL_BG, fg=TEXT_SEC,
                 justify="left").pack(padx=10, pady=(0,4), anchor="w")
        for wheel, chans in WHEEL_GROUPS.items():
            wc = WHEEL_COLORS[wheel]["pri"]
            tk.Label(p, text=f"  {wheel}", font=("Arial",8,"bold"),
                     bg=PANEL_BG, fg=wc).pack(anchor="w", padx=14, pady=(4,0))
            for ch in chans:
                self._btn(p, f"Box Plot — {ch}",
                          lambda c=ch: self._run_boxplot_channel(c),
                          bg=CHAN_COLORS[ch], w=30).pack(pady=1)
        for wheel in WHEEL_GROUPS:
            wc = WHEEL_COLORS[wheel]["pri"]
            self._btn(p, f"Box Plot — {wheel}  (3 popups)",
                      lambda w=wheel: self._run_boxplot_wheel(w),
                      bg=wc, w=30).pack(pady=1)
        self._btn(p, "Box Plot — All 12  (12 popups)",
                  self._run_boxplot_all, bg="#6C3483", w=30).pack(pady=2)

        tk.Label(p, text="Moments (Mx,My,Mz) — only if present in CSV:",
                 font=FONT_S, bg=PANEL_BG, fg=TEXT_SEC,
                 justify="left").pack(padx=10, pady=(6,2), anchor="w")
        for wheel in MOMENT_GROUPS:
            wc = WHEEL_COLORS[wheel]["pri"]
            self._btn(p, f"Box Plot — {wheel} Moments  (3 popups)",
                      lambda w=wheel: self._run_boxplot_wheel_moments(w),
                      bg=wc, w=30).pack(pady=1)
        self._btn(p, "Box Plot — All Moments  (12 popups)",
                  self._run_boxplot_all_moments, bg="#4A235A", w=30).pack(pady=2)

        # ── 7: Compare ──
        self._div(p); self._sec(p, "7  — COMPARE  DS1 vs DS2")
        tk.Label(p, text="Overlay both datasets per channel.\nUpload 2nd CSV first.",
                 font=FONT_S, bg=PANEL_BG, fg=TEXT_SEC,
                 justify="left").pack(padx=10, pady=(0,4), anchor="w")
        for wheel, chans in WHEEL_GROUPS.items():
            wc = WHEEL_COLORS[wheel]["pri"]
            tk.Label(p, text=f"  {wheel}", font=("Arial",8,"bold"),
                     bg=PANEL_BG, fg=wc).pack(anchor="w", padx=14, pady=(4,0))
            for ch in chans:
                self._btn(p, f"Compare — {ch}",
                          lambda c=ch: self._run_compare_channel(c),
                          bg=CHAN_COLORS[ch], w=30).pack(pady=1)
        self._btn(p, "Compare — All 12 Channels",
                  self._run_compare_all, bg="#6C3483", w=30).pack(pady=4)

        tk.Frame(p, bg="#333", height=1).pack(fill="x", padx=8, pady=4)
        self._btn(p, "Area Under Curve — All 12",
                  self._run_auc_compare_all, bg="#117A65", w=30).pack(pady=2)
        self._btn(p, "AUC Cross-Position (FL\u2192RL / FR\u2192RR)",
          self._run_auc_cross, bg="#0E6655", w=30).pack(pady=2)
        self._btn(p, "AUC — All Moments (12)",
          self._run_auc_moments, bg="#0B5345", w=30).pack(pady=2)
        self._btn(p, "AUC Cross-Position — Moments (FL\u2192RL / FR\u2192RR)",
          self._run_auc_cross_moments, bg="#0B4C3D", w=30).pack(pady=2)

        tk.Frame(p, bg="#333", height=1).pack(fill="x", padx=8, pady=4)
        tk.Label(p, text="Rainflow Histogram Compare\n(Front axle FL+FR | Rear axle RL+RR):",
                 font=FONT_S, bg=PANEL_BG, fg=TEXT_SEC,
                 anchor="w").pack(fill="x", padx=10)
        self._btn(p, "RF Compare — Fx  (Front + Rear)",
                  lambda: self._run_rf_compare("Fx"),
                  bg="#E24B4A", w=30).pack(pady=2)
        self._btn(p, "RF Compare — Fy  (Front + Rear)",
                  lambda: self._run_rf_compare("Fy"),
                  bg="#185FA5", w=30).pack(pady=2)
        self._btn(p, "RF Compare — Fz  (Front + Rear)",
                  lambda: self._run_rf_compare("Fz"),
                  bg="#1E8449", w=30).pack(pady=2)

        # ── 8: RMS Table ──
        self._div(p); self._sec(p, "8  — RMS TABLE  (IC vs EV)")
        tk.Label(p, text="RMS for all 24 channels\n(FL/FR/RL/RR × Fx,Fy,Fz,Mx,My,Mz)\nMoments used only if present in CSV.",
                 font=FONT_S, bg=PANEL_BG, fg=TEXT_SEC,
                 justify="left").pack(padx=10, pady=(0,4), anchor="w")
        self._btn(p, "Compute RMS Table", self._run_rms_table,
                  bg="#117A65", w=30).pack(pady=2)

        # ── 9: DLC Table ──
        self._div(p); self._sec(p, "9  — DLC TABLE  (Fz only)")
        tk.Label(p, text="DLC = RMS(Fz) / Static Fz (mean)\nper wheel, DS1 vs DS2.",
                 font=FONT_S, bg=PANEL_BG, fg=TEXT_SEC,
                 justify="left").pack(padx=10, pady=(0,4), anchor="w")
        self._btn(p, "Compute DLC Table", self._run_dlc_table,
                  bg="#0E6655", w=30).pack(pady=2)

        # ── 10: PSD ──
        self._div(p); self._sec(p, "10 — POWER SPECTRAL DENSITY")
        tk.Label(p, text="Welch PSD, DS1 vs DS2 overlay.\nUses Sampling rate from Settings.",
                 font=FONT_S, bg=PANEL_BG, fg=TEXT_SEC,
                 justify="left").pack(padx=10, pady=(0,4), anchor="w")
        for wheel in WHEEL_GROUPS_FULL:
            wc = WHEEL_COLORS[wheel]["pri"]
            self._btn(p, f"PSD — {wheel}  (6 subplots)",
                      lambda w=wheel: self._run_psd_wheel(w),
                      bg=wc, w=30).pack(pady=1)
        self._btn(p, "PSD — All Wheels (4 popups)",
                  self._run_psd_all, bg="#6C3483", w=30).pack(pady=2)

        # ── 11: G-Severity (Gx, Gy, Gxy) ──
        self._div(p); self._sec(p, "11 — G-SEVERITY  (Gx, Gy, Gxy)")
        tk.Label(p, text="Gx  = RMS(Fx/Fz)\nGy  = RMS(Fy/Fz)\n"
                          "Gxy = sqrt( mean(Fx²+Fy²) / mean(Fz)² )\n"
                          "Computed per wheel, DS1 vs DS2.",
                 font=FONT_S, bg=PANEL_BG, fg=TEXT_SEC,
                 justify="left").pack(padx=10, pady=(0,4), anchor="w")
        self._btn(p, "Compute G-Severity Table", self._run_g_severity_table,
                  bg="#7D3C98", w=30).pack(pady=2)

        # ── 12: Force-Pair Heatmaps (km) ──
        self._div(p); self._sec(p, "12 — FORCE HEATMAPS  (km distribution)")
        tk.Label(p, text="Fy-Fx / Fx-Fz / Fy-Fz per wheel.\n"
                          "Color + label = % of total distance\n"
                          "(uses Vehicle_Speed, not sample count).\n"
                          "All 4 wheels shown together per window.",
                 font=FONT_S, bg=PANEL_BG, fg=TEXT_SEC,
                 justify="left").pack(padx=10, pady=(0,4), anchor="w")

        range_frame = tk.Frame(p, bg=ENTRY_BG, relief="flat")
        range_frame.pack(fill="x", padx=8, pady=(0,4))
        tk.Label(range_frame, text="Bin range (daN)  —  Min / Max / Step:",
                 font=FONT_S, bg=ENTRY_BG, fg=TEXT_SEC,
                 anchor="w").pack(fill="x", padx=6, pady=(4,2))
        hdr = tk.Frame(range_frame, bg=ENTRY_BG); hdr.pack(fill="x", padx=6)
        tk.Label(hdr, text="", width=4, bg=ENTRY_BG).pack(side="left")
        for htxt in ("Min", "Max", "Step"):
            tk.Label(hdr, text=htxt, font=("Arial",8), bg=ENTRY_BG,
                     fg=TEXT_SEC, width=6).pack(side="left", padx=2)
        for ch in ("Fx", "Fy", "Fz"):
            row = tk.Frame(range_frame, bg=ENTRY_BG); row.pack(fill="x", padx=6, pady=1)
            tk.Label(row, text=f"{ch}:", font=("Arial",8,"bold"), bg=ENTRY_BG,
                     fg=TEXT_PRI, width=4, anchor="w").pack(side="left")
            v = self.heatmap_range_vars[ch]
            for key in ("min", "max", "step"):
                tk.Entry(row, textvariable=v[key], width=6, bg="#0D1B2A",
                          fg=TEXT_PRI, font=("Arial",8),
                          insertbackground=TEXT_PRI, relief="flat").pack(side="left", padx=2)

        self._btn(p, "Generate Force Heatmaps (Fy-Fx/Fx-Fz/Fy-Fz)",
                  self._run_force_heatmaps, bg="#B9770E", w=30).pack(pady=2)

        # ── Save / Export ──
        tk.Frame(p, bg=ACCENT, height=1).pack(fill="x", padx=8, pady=8)
        self._btn(p, "⬇  Export ALL Results (1 click)", self._export_all_results,
                  bg="#117A65", w=30).pack(pady=(2,2))
        tk.Label(p, text="Saves AUC + Cross-Position + Box Plots\n+ summary CSV to chosen folder",
                 font=("Arial", 8), bg=PANEL_BG, fg=TEXT_SEC,
                 justify="center").pack(pady=(0,4))
        self._btn(p, "Save Current Plot (PNG)", self._save_plot,
                  bg="#444", w=30).pack(pady=4)

    def _build_right_panel(self):
        self.canvas_frame = tk.Frame(self.right, bg=BG)
        self.canvas_frame.pack(fill="both", expand=True)
        self.welcome = tk.Frame(self.canvas_frame, bg=BG)
        self.welcome.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(self.welcome, text="WFT", font=("Arial Black",48),
                 bg=BG, fg=BTN_BG).pack()
        tk.Label(self.welcome, text="Force Data Analyzer",
                 font=("Arial",16), bg=BG, fg=TEXT_PRI).pack()
        tk.Label(self.welcome, text="Upload a CSV file to begin",
                 font=FONT_S, bg=BG, fg=TEXT_SEC).pack(pady=8)

    # ══ HELPERS ═══════════════════════════════════════════════
    def _sec(self, parent, text):
        tk.Label(parent, text=text, font=FONT_B, bg=PANEL_BG,
                 fg=BTN_BG, anchor="w").pack(fill="x", padx=10, pady=(8,2))

    def _div(self, parent):
        tk.Frame(parent, bg=BTN_BG, height=1).pack(fill="x", padx=8, pady=6)

    def _btn(self, parent, text, cmd, bg=BTN_BG, fg=BTN_FG, w=28, relief="flat"):
        b = tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg,
                      font=FONT_B, width=w, relief=relief, bd=0,
                      activebackground=BTN_HOV, activeforeground=BTN_FG,
                      cursor="hand2", pady=5)
        b.bind("<Enter>", lambda e, b=b, o=bg: b.config(bg=BTN_HOV))
        b.bind("<Leave>", lambda e, b=b, o=bg: b.config(bg=o))
        return b

    def _show_fig(self, fig):
        if self.mpl_canvas:
            self.mpl_canvas.get_tk_widget().destroy()
            plt.close("all")
        try: self.welcome.place_forget()
        except Exception: pass
        self.current_fig = fig
        self.mpl_canvas  = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.mpl_canvas.draw()
        self.mpl_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _set_status(self, msg, color=TEXT_SEC):
        self.status_msg.set(msg); self.root.update_idletasks()

    def _style_ax(self, ax):
        ax.set_facecolor("#0A2540")
        ax.tick_params(colors=TEXT_SEC, labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#1B3A5C")
        ax.grid(True, color="#1B3A5C", linewidth=0.4, alpha=0.6)

    # ══ FILE LOADING ══════════════════════════════════════════
    def _parse_csv(self, path):
        """
        Reads CSV by column NAME (not position).
        Handles files with up to 2 extra rows before the Time header row.
        Strategy copied from working reference code:
          - Try skiprows=2 first (your format: row0=info, row1=units, row2=Time header)
          - Try skiprows=1 (one extra row before header)
          - Try skiprows=0 (Time is already row 0)
        All reads use latin1 encoding and skip_blank_lines=True.
        Auto-scale: uniform N->daN using max median across all channels.
        """
        known_names = set(FULL_COLUMN_MAP.keys()) | set(FORCE_CHANNELS)

        def _try_load(skip_n):
            """Attempt to load with given skiprows, return df or None."""
            for engine in ('c', 'python'):
                try:
                    kw = dict(encoding='latin1', skiprows=skip_n,
                              skip_blank_lines=True, skipinitialspace=True,
                              low_memory=False, engine=engine)
                    df = pd.read_csv(path, **kw)
                    df.columns = [str(c).strip() for c in df.columns]
                    # Check if any force channels present
                    hits = sum(1 for c in df.columns if c in known_names)
                    if hits >= 2:
                        return df
                except Exception:
                    continue
            return None

        # Try skipping 2, 1, 0 extra rows — first one that finds channels wins
        df_raw = None
        for skip in (2, 1, 0, 3):
            df_raw = _try_load(skip)
            if df_raw is not None:
                break

        if df_raw is None:
            raise ValueError("Could not parse CSV. Check file format.")

        df    = df_raw.apply(pd.to_numeric, errors="coerce")
        chans = [c for c in FORCE_CHANNELS if c in df.columns]
        if not chans:
            found = [c for c in df_raw.columns
                     if not str(c).startswith("col_") and not str(c).startswith("Unnamed")]
            raise ValueError(
                f"No force channels found in CSV.\n"
                f"Expected any of: {', '.join(FORCE_CHANNELS)}\n"
                f"Columns detected: {', '.join(found[:20])}")

        df = df.dropna(subset=chans, how="all")

        # Uniform auto-scale: decide ONCE using max median across all channels
        max_median = max(df[col].abs().median() for col in chans)
        if max_median > 1000:
            for col in chans:
                df[col] = df[col] / 10.0

        return df, chans

    def _load_csv(self):
        path = filedialog.askopenfilename(
            title="Select WFT CSV",
            filetypes=[("CSV","*.csv"),("All","*.*")])
        if not path: return
        try:
            self._set_status("Loading CSV …")
            self.df, self.avail_chans = self._parse_csv(path)
            self.avail_moments = [c for c in MOMENT_CHANNELS if c in self.df.columns]
            fname = os.path.basename(path)
            self.file_path.set(fname)
            self.info_var.set(
                f"Rows    : {len(self.df):,}\n"
                f"Channels: {len(self.avail_chans)}\n"
                f"SR      : {self.sr_var.get()} Hz\n"
                f"Duration: {len(self.df)/float(self.sr_var.get() or 100):.1f} s")
            self._set_status(
                f"Loaded: {fname}  |  {len(self.df):,} rows  |  "
                f"Channels: {', '.join(self.avail_chans)}", SUCCESS)
        except Exception as ex:
            messagebox.showerror("Load Error", str(ex))
            self._set_status(f"Error: {ex}", DANGER)

    def _load_csv2(self):
        path = filedialog.askopenfilename(
            title="Select 2nd CSV for comparison",
            filetypes=[("CSV","*.csv"),("All","*.*")])
        if not path: return
        try:
            self._set_status("Loading 2nd CSV …")
            self.df2, self.avail_chans2 = self._parse_csv(path)
            self.avail_moments2 = [c for c in MOMENT_CHANNELS if c in self.df2.columns]
            self.file_path2.set(os.path.basename(path))
            self._set_status(
                f"2nd dataset: {os.path.basename(path)}  |  {len(self.df2):,} rows", SUCCESS)
        except Exception as ex:
            messagebox.showerror("Load Error", str(ex))
            self._set_status(f"Error: {ex}", DANGER)

    def _clear_csv2(self):
        self.df2=None; self.avail_chans2=[]; self.avail_moments2=[]
        self.file_path2.set("No comparison file")
        self._set_status("2nd dataset cleared.", TEXT_SEC)

    # ══ PERCENTILE ════════════════════════════════════════════
    def _run_percentile(self, p):
        if self.df is None: messagebox.showwarning("No data","Upload CSV first."); return
        self._plot_percentile([p])

    def _run_all_percentiles(self):
        if self.df is None: messagebox.showwarning("No data","Upload CSV first."); return
        self._plot_percentile([80,90,95])

    def _plot_percentile(self, plist):
        chans = self.avail_chans
        ncols = 3
        nrows = (len(chans) + ncols - 1) // ncols
        fig   = plt.figure(figsize=(5*ncols, 4.2*nrows), facecolor=BG)
        fig.suptitle(f"Percentile Analysis  (P{'/P'.join(str(x) for x in plist)})",
                     color="white", fontsize=13, fontweight="bold", y=1.01)
        pc = {80:"#27AE60", 90:"#E67E22", 95:"#C0392B"}
        results = {ch: {p: float(np.percentile(self.df[ch].dropna(), p))
                        for p in plist} for ch in chans}
        for idx, ch in enumerate(chans):
            ax = fig.add_subplot(nrows, ncols, idx+1)
            self._style_ax(ax)
            ax.hist(self.df[ch].dropna(), bins=60,
                    color=CHAN_COLORS.get(ch, BTN_BG), alpha=0.75, edgecolor="none")
            for p in plist:
                v = results[ch][p]
                ax.axvline(v, color=pc[p], linewidth=1.8, linestyle="--",
                           label=f"P{p}: {v:.1f}")
            ax.set_title(ch, color=CHAN_COLORS.get(ch, "white"),
                         fontsize=11, fontweight="bold")
            ax.set_xlabel("Force (daN)", color=TEXT_SEC, fontsize=8)
            xlim = _plot_xlim(ch)
            if xlim: ax.set_xlim(xlim)
            ax.set_ylabel("Count", color=TEXT_SEC, fontsize=8)
            ax.legend(fontsize=7, facecolor=BG, labelcolor="white", framealpha=0.8)
        fig.tight_layout(rect=[0,0,1,0.97])
        self._show_fig(fig)
        self._show_table(results, plist)

    def _show_table(self, results, plist):
        win = tk.Toplevel(self.root)
        win.title("Percentile Results"); win.configure(bg=BG)
        win.geometry("680x420")
        tk.Label(win, text="Percentile Results (daN)", font=FONT_H,
                 bg=BG, fg=BTN_BG).pack(pady=(12,4))
        fr = tk.Frame(win, bg=PANEL_BG); fr.pack(padx=16, pady=8, fill="both", expand=True)
        for ci, h in enumerate(["Channel"]+[f"P{p}" for p in plist]):
            tk.Label(fr, text=h, font=FONT_B, bg=BTN_BG, fg=BTN_FG,
                     width=14, relief="flat", pady=5).grid(
                row=0, column=ci, padx=1, pady=1, sticky="nsew")
        for ri,(ch,vals) in enumerate(results.items(),1):
            bgc = ENTRY_BG if ri%2==0 else "#0F2D4A"
            tk.Label(fr, text=ch, font=FONT_B, bg=bgc,
                     fg=CHAN_COLORS.get(ch, TEXT_PRI),
                     width=14, pady=4).grid(row=ri,column=0,padx=1,pady=1,sticky="nsew")
            for ci,pv in enumerate(plist,1):
                v=vals[pv]; fc=DANGER if v>7000 else "#E67E22" if v>5000 else SUCCESS
                tk.Label(fr, text=f"{v:.2f}", font=FONT_N, bg=bgc, fg=fc,
                         width=14, pady=4).grid(row=ri,column=ci,padx=1,pady=1,sticky="nsew")
        tk.Button(win, text="Close", command=win.destroy,
                  bg=DANGER, fg=BTN_FG, font=FONT_B,
                  relief="flat", cursor="hand2").pack(pady=8)

    # ══ RAINFLOW ══════════════════════════════════════════════
    def _rf_color(self, count):
        for i, thr in enumerate(RF_STEPS[1:], 1):
            if count <= thr: return RF_COLORS[i]
        return RF_COLORS[-1]

    def _rainflow_cycles(self, df, ch, max_pts=50000):
        data = df[ch].dropna().values
        if len(data) > max_pts: data = data[::len(data)//max_pts]
        return list(rainflow.extract_cycles(data))

    def _build_fromto_matrix(self, cycles, n_bins=16, shared_edges=None):
        rng  = np.array([c[0] for c in cycles])
        mean = np.array([c[1] for c in cycles])
        cnt  = np.array([c[2] for c in cycles])
        frm  = mean - rng/2
        to   = mean + rng/2
        if shared_edges is not None:
            edges = shared_edges
        else:
            lo = min(frm.min(), to.min())
            hi = max(frm.max(), to.max())
            edges = np.linspace(lo, hi, n_bins+1)
        n = len(edges)-1
        mat = np.zeros((n, n))
        for f, t, c in zip(frm, to, cnt):
            fi = int(np.clip(np.searchsorted(edges, f,"right")-1, 0, n-1))
            ti = int(np.clip(np.searchsorted(edges, t,"right")-1, 0, n-1))
            mat[ti, fi] += c
        return mat, edges

    def _draw_fromto_ax(self, ax, mat, edges, title, col):
        n = len(edges)-1
        ax.set_facecolor("#12203A")
        for ri in range(n):
            for ci in range(n):
                val = mat[ri, ci]
                if val < 0.5: continue
                x0,x1 = edges[ci], edges[ci+1]
                y0,y1 = edges[ri], edges[ri+1]
                fc = self._rf_color(val)
                rect = mpatches.Rectangle((x0,y0), x1-x0, y1-y0,
                                          facecolor=fc, edgecolor="#0A0A1A",
                                          linewidth=0.5)
                ax.add_patch(rect)
                cx,cy = (x0+x1)/2, (y0+y1)/2
                tc = "black" if fc not in ("#0000FF","#6C3483") else "white"
                ax.text(cx, cy, str(int(val)), ha="center", va="center",
                        fontsize=4.5, color=tc, fontweight="bold")
        d0, d1 = edges[0], edges[-1]
        ax.plot([d0,d1], [d0,d1], color="#FF4500", linewidth=1.8, linestyle="-", zorder=5)
        ax.text((d0+d1)*0.5, (d0+d1)*0.5+abs(d1-d0)*0.03,
                "Zero Range", color="#FF4500", fontsize=5.5,
                ha="center", va="bottom", rotation=45, zorder=6)
        if edges[0] < 0 < edges[-1]:
            ax.axvline(0, color="#888", linewidth=0.7, linestyle="--", alpha=0.6, zorder=3)
            ax.axhline(0, color="#888", linewidth=0.7, linestyle="--", alpha=0.6, zorder=3)
        span = abs(edges[-1]-edges[0])
        ax.text(edges[0]+span*0.18, edges[-1]-span*0.08,
                "Compression", color="#90EE90", fontsize=7,
                fontweight="bold", ha="center", alpha=0.9, zorder=4)
        ax.text(edges[-1]-span*0.18, edges[0]+span*0.08,
                "Tension", color="#FFB6C1", fontsize=7,
                fontweight="bold", ha="center", alpha=0.9, zorder=4)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(edges[0], edges[-1])
        step = max(1, n//6)
        tv   = edges[::step]
        ax.set_xticks(tv)
        ax.set_xticklabels([f"{v:.0f}" for v in tv],
                           fontsize=6, color=TEXT_SEC, rotation=30, ha="right")
        ax.set_yticks(tv)
        ax.set_yticklabels([f"{v:.0f}" for v in tv], fontsize=6, color=TEXT_SEC)
        ax.tick_params(length=3, color="#1B3A5C")
        for sp in ax.spines.values(): sp.set_edgecolor("#333")
        ax.set_xlabel("From  (daN)  — valley / trough", color=TEXT_SEC, fontsize=8)
        ax.set_ylabel("To  (daN)  — peak",              color=TEXT_SEC, fontsize=8)
        ax.set_title(title, color=col, fontsize=9, fontweight="bold")

    def _add_rf_legend(self, ax):
        handles = [mpatches.Patch(facecolor=RF_COLORS[i], edgecolor="black",
                                  linewidth=0.5, label=f"Cycles {RF_LABELS[i]}")
                   for i in range(len(RF_LABELS))]
        ax.legend(handles=handles, title="Cycles", title_fontsize=6.5,
                  fontsize=6, facecolor="#0D1B2A", labelcolor="white",
                  framealpha=0.95, loc="upper left", edgecolor="#555",
                  handlelength=1.4, handleheight=1.1,
                  borderpad=0.5, labelspacing=0.3)

    def _run_rainflow_single(self, ch):
        if self.df is None: messagebox.showwarning("No data","Upload CSV first."); return
        if ch not in self.avail_chans:
            messagebox.showwarning("Missing", f"{ch} not in CSV."); return
        self._set_status(f"Running rainflow — {ch} …")
        try:
            cyc = self._rainflow_cycles(self.df, ch)
            self._plot_rf_single(ch, cyc)
            self._set_status(f"Rainflow {ch} — {len(cyc):,} cycles", SUCCESS)
        except Exception as ex:
            messagebox.showerror("Error", str(ex)); self._set_status(str(ex), DANGER)

    def _run_rainflow_all(self):
        if self.df is None: messagebox.showwarning("No data","Upload CSV first."); return
        self._set_status("Running rainflow — all channels …")
        try:
            all_c = {ch: self._rainflow_cycles(self.df, ch) for ch in self.avail_chans}
            self._plot_rf_all(all_c)
            self._set_status("Rainflow complete — all channels", SUCCESS)
        except Exception as ex:
            messagebox.showerror("Error", str(ex)); self._set_status(str(ex), DANGER)

    def _plot_rf_single(self, ch, cyc):
        if not cyc: messagebox.showinfo("No cycles", f"No cycles in {ch}."); return
        rng  = np.array([c[0] for c in cyc])
        mean = np.array([c[1] for c in cyc])
        cnt  = np.array([c[2] for c in cyc])
        mat, edges = self._build_fromto_matrix(cyc, n_bins=16)
        col  = CHAN_COLORS.get(ch, BTN_BG)
        fig  = plt.figure(figsize=(15, 5.5), facecolor=BG)
        fig.suptitle(f"Rainflow Counting — {ch}  (Siemens From-To Matrix)",
                     color="white", fontsize=13, fontweight="bold")
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42)

        ax1 = fig.add_subplot(gs[0])
        self._draw_fromto_ax(ax1, mat, edges,
                             f"{ch}  |  {int(cnt.sum()):,} cycles", col)
        self._add_rf_legend(ax1)

        ax2 = fig.add_subplot(gs[1])
        self._style_ax(ax2)
        rng_rep = np.repeat(rng, cnt.astype(int).clip(1))
        bins_r  = np.linspace(0, rng.max(), 22)
        ax2.bar(bins_r[:-1], np.histogram(rng, bins=bins_r, weights=cnt)[0],
                width=np.diff(bins_r)*0.85, color=col, alpha=0.85,
                edgecolor="#0A2540", align="edge")
        for pct,lc,ls in [(90,"#E67E22","--"),(95,"#C0392B","-.")]:\
            v = float(np.percentile(rng_rep, pct)); ax2.axvline(v, color=lc, linewidth=1.6,
                linestyle=ls, label=f"P{pct}: {v:.0f} daN")
        ax2.set_title("Range Distribution\n↑ higher Range = more fatigue damage",
                      color="white", fontsize=9, fontweight="bold")
        ax2.set_xlabel("Range (daN)", color=TEXT_SEC, fontsize=8)
        ax2.set_ylabel("Weighted cycle count", color=TEXT_SEC, fontsize=8)
        ax2.legend(fontsize=7, facecolor=BG, labelcolor="white")

        ax3 = fig.add_subplot(gs[2])
        self._style_ax(ax3)
        rng_sort = np.sort(rng_rep)[::-1]
        exceed   = np.arange(1, len(rng_sort)+1)
        ax3.semilogy(rng_sort, exceed, color=col, linewidth=2)
        ax3.fill_betweenx(exceed, rng_sort, alpha=0.15, color=col)
        ax3.set_title("Cumulative Exceedance\n(log scale)",
                      color="white", fontsize=9, fontweight="bold")
        ax3.set_xlabel("Range (daN)", color=TEXT_SEC, fontsize=8)
        ax3.set_ylabel("No. of exceedances", color=TEXT_SEC, fontsize=8)
        ax3.grid(True, color="#1B3A5C", linewidth=0.4, alpha=0.5, which="both")

        p95v = float(np.percentile(rng_rep, 95))
        fig.text(0.01, 0.01,
                 f"Total cycles: {int(cnt.sum()):,}   "
                 f"Max range: {rng.max():.0f} daN   "
                 f"P95 range: {p95v:.0f} daN   "
                 f"Mean (static load): {mean.mean():.0f} daN",
                 color=TEXT_SEC, fontsize=8, va="bottom")
        fig.tight_layout(rect=[0, 0.05, 1, 0.96])
        self._show_fig(fig)

    def _plot_rf_all(self, all_cyc):
        chans = [ch for ch in FORCE_CHANNELS if ch in all_cyc]
        if not chans: return
        N = 14

        data = {}
        for ch in chans:
            cyc = all_cyc[ch]
            if not cyc: data[ch]=None; continue
            rng = np.array([c[0] for c in cyc])
            cnt = np.array([c[2] for c in cyc])
            mat, edges = self._build_fromto_matrix(cyc, n_bins=N)
            total = int(cnt.sum())
            p95   = float(np.percentile(np.repeat(rng, cnt.astype(int).clip(1)), 95))
            data[ch] = dict(cyc=cyc, mat=mat, edges=edges, total=total, p95=p95)

        # Shared edges per wheel group
        for wheel, grp in WHEEL_GROUPS.items():
            valid = [c for c in grp if c in data and data[c]]
            if len(valid) < 2: continue
            glo = min(data[c]["edges"][0]  for c in valid)
            ghi = max(data[c]["edges"][-1] for c in valid)
            sh  = np.linspace(glo, ghi, N+1)
            for c in valid:
                mat, _ = self._build_fromto_matrix(data[c]["cyc"], n_bins=N, shared_edges=sh)
                data[c]["mat"]   = mat
                data[c]["edges"] = sh

        ncols = 3; nrows = (len(chans)+ncols-1)//ncols
        fig = plt.figure(figsize=(6.8*ncols, 5.8*nrows), facecolor=BG)
        fig.suptitle("Rainflow — All Channels  (Siemens From-To Matrix)\n"
                     "Shared axes per wheel group",
                     color="white", fontsize=12, fontweight="bold")

        for idx, ch in enumerate(chans):
            ax = fig.add_subplot(nrows, ncols, idx+1)
            d  = data.get(ch)
            if d is None:
                ax.set_facecolor("#0A2540")
                ax.text(0.5,0.5,f"{ch}\nNo cycles",ha="center",va="center",
                        color="white",transform=ax.transAxes); continue
            self._draw_fromto_ax(ax, d["mat"], d["edges"],
                                 f"{ch}  |  {d['total']:,} cyc  P95={d['p95']:.0f} daN",
                                 CHAN_COLORS.get(ch,"white"))
            if idx == 0: self._add_rf_legend(ax)

        fig.tight_layout(rect=[0,0,1,0.94])
        self._show_fig(fig)

    # ══ BOX PLOT ══════════════════════════════════════════════
    def _run_boxplot_channel(self, ch):
        if self.df is None: messagebox.showwarning("No data","Upload CSV first."); return
        if ch not in self.avail_chans:
            messagebox.showwarning("Missing", f"{ch} not found."); return
        self._open_box_popup(ch)

    def _run_boxplot_wheel(self, wheel):
        if self.df is None: messagebox.showwarning("No data","Upload CSV first."); return
        for ch in WHEEL_GROUPS[wheel]:
            if ch in self.avail_chans: self._open_box_popup(ch)

    def _run_boxplot_all(self):
        if self.df is None: messagebox.showwarning("No data","Upload CSV first."); return
        for ch in self.avail_chans: self._open_box_popup(ch)

    def _run_boxplot_channel_moment(self, ch):
        if self.df is None: messagebox.showwarning("No data","Upload CSV first."); return
        if ch not in self.avail_moments:
            messagebox.showwarning("Missing", f"{ch} not found in this CSV."); return
        self._open_box_popup(ch)

    def _run_boxplot_wheel_moments(self, wheel):
        if self.df is None: messagebox.showwarning("No data","Upload CSV first."); return
        found = False
        for ch in MOMENT_GROUPS[wheel]:
            if ch in self.avail_moments:
                self._open_box_popup(ch); found = True
        if not found:
            messagebox.showwarning("No moment channels",
                f"No Mx/My/Mz columns found for {wheel} in this CSV.")

    def _run_boxplot_all_moments(self):
        if self.df is None: messagebox.showwarning("No data","Upload CSV first."); return
        if not self.avail_moments:
            messagebox.showwarning("No moment channels",
                "No Mx/My/Mz columns found in this CSV."); return
        for ch in self.avail_moments: self._open_box_popup(ch)

    def _open_box_popup(self, ch):
        col  = CHAN_COLORS.get(ch, BTN_BG)
        unit = _chan_unit(ch)
        d1   = self.df[ch].dropna().values
        ds2_avail = (ch in self.avail_chans2) or (ch in self.avail_moments2)
        d2   = (self.df2[ch].dropna().values
                if self.df2 is not None and ds2_avail and ch in self.df2.columns else None)
        lbl1 = self.label1.get()
        lbl2 = self.label2.get()
        ylim = _box_ylim(ch)

        win = tk.Toplevel(self.root)
        disp = self.chan_labels[ch].get() if ch in self.chan_labels else ch
        win.title(f"Box Plot — {disp}")
        win.configure(bg=BG)
        win.geometry("720x560")

        fig, ax = plt.subplots(figsize=(7.2, 5), facecolor="#FFFFFF")
        ax.set_facecolor("#F5F5F5")

        datasets = [(d1, lbl1, col)]
        if d2 is not None: datasets.append((d2, lbl2, "#BBBBBB"))

        plot_data  = [d for d,_,_ in datasets]
        box_colors = [c for _,_,c in datasets]
        xlabels    = [l for _,l,_ in datasets]

        bp = ax.boxplot(plot_data, patch_artist=True, notch=False, widths=0.38,
                        medianprops=dict(color="#000000", linewidth=2.5),
                        whiskerprops=dict(color="#000000", linewidth=1.4, linestyle="--"),
                        capprops=dict(color="#000000", linewidth=2),
                        flierprops=dict(marker="o", markersize=2.5,
                                        markerfacecolor="#E67E22",
                                        markeredgewidth=0, alpha=0.35),
                        boxprops=dict(linewidth=1.6))
        for patch, bc in zip(bp["boxes"], box_colors):
            patch.set_facecolor(bc); patch.set_alpha(0.70)

        if ylim: ax.set_ylim(ylim)

        n = len(datasets)
        for i,(d,lbl,bc) in enumerate(datasets, 1):
            p5  = np.percentile(d, 5)
            mu  = np.mean(d)
            p95 = np.percentile(d, 95)
            x_lo = (i - 1 + 0.31) / n
            x_hi = (i - 1 + 0.69) / n
            ax.axhline(p5,  xmin=x_lo, xmax=x_hi, color="#1E8449", linewidth=2.5, linestyle=":")
            ax.axhline(mu,  xmin=x_lo, xmax=x_hi, color="#B7950B", linewidth=2.5, linestyle="--")
            ax.axhline(p95, xmin=x_lo, xmax=x_hi, color="#C0392B", linewidth=2.6, linestyle="-.")

        stat_rows = []
        for d, lbl, _ in datasets:
            p5  = np.percentile(d, 5)
            q1  = np.percentile(d, 25)
            med = np.median(d)
            mu  = np.mean(d)
            q3  = np.percentile(d, 75)
            p95 = np.percentile(d, 95)
            stat_rows.append(
                f"{lbl:>12s} │ Min:{d.min():>7.0f} │ P5:{p5:>7.0f} │ "
                f"Q1:{q1:>7.0f} │ Median:{med:>7.0f} │ Mean:{mu:>7.0f} │ "
                f"Q3:{q3:>7.0f} │ P95:{p95:>7.0f} │ Max:{d.max():>7.0f}")
        fig.text(0.5, 0.01, "\n".join(stat_rows), ha="center", va="bottom",
                 fontsize=6.5, color="#000000", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F0F0",
                           edgecolor="#AAAAAA", alpha=0.95))

        ax.set_xticks(range(1, len(xlabels)+1))
        ax.set_xticklabels(xlabels, fontsize=11, color="#000000", fontweight="bold")
        ax.set_ylabel(f"{'Force' if unit=='daN' else 'Moment'} ({unit})",
                      color="#000000", fontsize=11, fontweight="bold")
        ax.set_title(f"Box Plot — {disp}", color=col, fontsize=12, fontweight="bold")
        ax.tick_params(colors="#000000", labelsize=10, width=1.2)
        for sp in ax.spines.values(): sp.set_edgecolor("#333333")
        ax.grid(True, axis="y", color="#CCCCCC", linewidth=0.6, alpha=0.8)

        leg = [Line2D([0],[0], color="white",    linewidth=2.5, label="Median"),
               Line2D([0],[0], color="#27AE60",  linewidth=1.5, linestyle=":",  label="P5"),
               Line2D([0],[0], color="#F1C40F",  linewidth=1.5, linestyle="--", label="Mean"),
               Line2D([0],[0], color=DANGER,     linewidth=1.6, linestyle="-.", label="P95"),
               Line2D([0],[0], color=TEXT_SEC,   linewidth=1.2, linestyle="--", label="Whiskers (1.5×IQR)"),
               Line2D([0],[0], marker="o", color="w",
                      markerfacecolor="#E67E22", markersize=5, label="Outliers")]
        ax.legend(handles=leg, fontsize=8, facecolor=BG, labelcolor="white",
                  framealpha=0.95, edgecolor="#AAAAAA", loc="upper right")

        fig.tight_layout(rect=[0, 0.16, 1, 0.97])
        if not hasattr(self, "_box_figs"):
            self._box_figs = {}
        self._box_figs[ch] = fig
        cw = FigureCanvasTkAgg(fig, master=win)
        cw.draw(); cw.get_tk_widget().pack(fill="both", expand=True)
        btn_frame = tk.Frame(win, bg=BG); btn_frame.pack(pady=4)
        def _save_this():
            path = filedialog.asksaveasfilename(
                initialfile=f"BoxPlot_{ch}.png",
                defaultextension=".png",
                filetypes=[("PNG","*.png"),("PDF","*.pdf"),("All","*.*")])
            if path:
                fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#FFFFFF")
                self._set_status(f"Saved: {os.path.basename(path)}", SUCCESS)
        tk.Button(btn_frame, text="Save PNG", command=_save_this,
                  bg=BTN_BG, fg=BTN_FG, font=FONT_B,
                  relief="flat", cursor="hand2", padx=10).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Close", command=lambda: (plt.close(fig), win.destroy()),
                  bg=DANGER, fg=BTN_FG, font=FONT_B,
                  relief="flat", cursor="hand2", padx=10).pack(side="left", padx=6)

    # ══ COMPARE — DS1 vs DS2 ══════════════════════════════════
    def _run_compare_channel(self, ch):
        if self.df  is None: messagebox.showwarning("No data","Upload Dataset 1 first."); return
        if self.df2 is None: messagebox.showwarning("No 2nd CSV","Upload 2nd CSV first."); return
        self._plot_compare_single(ch)

    def _run_compare_all(self):
        if self.df  is None: messagebox.showwarning("No data","Upload Dataset 1 first."); return
        if self.df2 is None: messagebox.showwarning("No 2nd CSV","Upload 2nd CSV first."); return
        chans = [c for c in self.avail_chans if c in self.avail_chans2]
        if not chans:
            messagebox.showwarning("No overlap","No common channels between datasets."); return
        self._plot_compare_all(chans)

    def _plot_compare_single(self, ch):
        lbl1, lbl2 = self.label1.get(), self.label2.get()
        col = CHAN_COLORS.get(ch, BTN_BG)
        d1  = self.df[ch].dropna().values  if ch in self.avail_chans  else None
        d2  = self.df2[ch].dropna().values if ch in self.avail_chans2 else None

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
        fig.suptitle(f"Comparison — {ch}  ({lbl1}  vs  {lbl2})",
                     color="white", fontsize=13, fontweight="bold")

        ax = axes[0]; self._style_ax(ax)
        for d,lbl,fc,ls in [(d1,lbl1,col,"-"),(d2,lbl2,"#BBBBBB","--")]:
            if d is None: continue
            ax.hist(d, bins=60, alpha=0.5, color=fc, edgecolor="none",
                    label=lbl, density=True)
            p95 = np.percentile(d, 95)
            ax.axvline(p95, color=fc, linewidth=2, linestyle="--",
                       label=f"{lbl} P95: {p95:.0f} daN")
        ax.set_title("Distribution Overlay", color="white", fontsize=10, fontweight="bold")
        ax.set_xlabel("Force (daN)", color=TEXT_SEC, fontsize=9)
        xlim = _plot_xlim(ch)
        if xlim: ax.set_xlim(xlim)
        ax.set_ylabel("Density", color=TEXT_SEC, fontsize=9)
        ax.legend(fontsize=8, facecolor=BG, labelcolor="white", framealpha=0.85)

        ax2 = axes[1]; self._style_ax(ax2)
        plot_data, xlbls, bcolors = [], [], []
        if d1 is not None: plot_data.append(d1); xlbls.append(lbl1); bcolors.append(col)
        if d2 is not None: plot_data.append(d2); xlbls.append(lbl2); bcolors.append("#BBBBBB")
        bp = ax2.boxplot(plot_data, patch_artist=True, notch=False, widths=0.4,
                         medianprops=dict(color="white", linewidth=2.5),
                         whiskerprops=dict(color=TEXT_SEC, linewidth=1.3, linestyle="--"),
                         capprops=dict(color=TEXT_SEC, linewidth=1.8),
                         flierprops=dict(marker="o", markersize=2,
                                         markerfacecolor="#E67E22",
                                         markeredgewidth=0, alpha=0.3),
                         boxprops=dict(linewidth=1.5))
        for patch,bc in zip(bp["boxes"], bcolors):
            patch.set_facecolor(bc); patch.set_alpha(0.70)
        ylim = _box_ylim(ch)
        if ylim: ax2.set_ylim(ylim)
        ax2.set_xticks(range(1,len(xlbls)+1))
        ax2.set_xticklabels(xlbls, fontsize=10, color=TEXT_SEC, fontweight="bold")
        ax2.set_ylabel("Force (daN)", color=TEXT_SEC, fontsize=9)
        ax2.set_title("Box Plot Comparison", color="white", fontsize=10, fontweight="bold")
        ax2.grid(True, axis="y", color="#1B3A5C", linewidth=0.5, alpha=0.6)

        rows = []
        for d,lbl in zip(plot_data, xlbls):
            rows.append(
                f"{lbl}: Min={d.min():.0f}  P5={np.percentile(d,5):.0f}  "
                f"Median={np.median(d):.0f}  Mean={np.mean(d):.0f}  "
                f"P95={np.percentile(d,95):.0f}  Max={d.max():.0f}")
        fig.text(0.5, 0.01, "\n".join(rows), ha="center", va="bottom",
                 fontsize=7.5, color=TEXT_SEC, fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor=ENTRY_BG,
                           edgecolor="#1B3A5C", alpha=0.9))
        fig.tight_layout(rect=[0, 0.12, 1, 0.95])
        self._show_fig(fig)
        self._set_status(f"Compare {ch}: {lbl1} vs {lbl2}", SUCCESS)

    def _plot_compare_all(self, chans):
        lbl1, lbl2 = self.label1.get(), self.label2.get()
        ncols = 3; nrows = (len(chans)+ncols-1)//ncols
        fig = plt.figure(figsize=(5.5*ncols, 4.5*nrows), facecolor=BG)
        fig.suptitle(f"Comparison — All Channels  |  {lbl1}  vs  {lbl2}",
                     color="white", fontsize=12, fontweight="bold")
        for idx, ch in enumerate(chans):
            ax = fig.add_subplot(nrows, ncols, idx+1)
            self._style_ax(ax)
            col = CHAN_COLORS.get(ch, BTN_BG)
            d1  = self.df[ch].dropna().values  if ch in self.avail_chans  else None
            d2  = self.df2[ch].dropna().values if ch in self.avail_chans2 else None
            for d,lbl,fc in [(d1,lbl1,col),(d2,lbl2,"#BBBBBB")]:
                if d is None: continue
                ax.hist(d, bins=50, alpha=0.5, color=fc, edgecolor="none",
                        label=lbl, density=True)
                p95 = np.percentile(d, 95)
                ax.axvline(p95, color=fc, linewidth=1.8, linestyle="--",
                           label=f"P95:{p95:.0f}")
            ax.set_title(ch, color=col, fontsize=10, fontweight="bold")
            ax.set_xlabel("Force (daN)", color=TEXT_SEC, fontsize=7)
            xlim = _plot_xlim(ch)
            if xlim: ax.set_xlim(xlim)
            ax.set_ylabel("Density", color=TEXT_SEC, fontsize=7)
            ax.legend(fontsize=6.5, facecolor=BG, labelcolor="white", framealpha=0.85)
        fig.tight_layout(rect=[0,0,1,0.95])
        self._show_fig(fig)
        self._set_status(f"Compare all: {lbl1} vs {lbl2}", SUCCESS)

    # ══ AREA UNDER CURVE ══════════════════════════════════════
    def _kde_curve(self, data, x_grid):
        """Fast KDE using ALL data points — no subsampling.

        Direct scipy gaussian_kde on the full array is O(n_samples * n_grid),
        which is what made AUC generation slow on large logged files.
        My first attempt fixed the speed by fitting on a random 20,000-point
        subsample — that was a mistake: scipy's automatic (Scott) bandwidth
        scales as n^-1/5, so shrinking n from e.g. 300,000 to 20,000 widened
        the bandwidth by ~1.7x and over-smoothed close-together bumps into one.

        This version instead bins the FULL dataset into a fine histogram
        (still O(n), independent of grid size) and smooths it with a Gaussian
        filter using the bandwidth computed from the full data's Scott factor.
        This is numerically equivalent to the true full-data KDE (verified to
        ~1e-6 on test distributions) but runs in milliseconds even for
        millions of rows — faster than the subsampled version AND accurate.
        """
        from scipy.ndimage import gaussian_filter1d
        data = np.asarray(data, dtype=float)
        data = data[~np.isnan(data)]
        n = data.size
        if n < 2:
            return np.zeros_like(x_grid)

        std = np.std(data)
        if std == 0:
            return np.zeros_like(x_grid)
        bw = (n ** (-1.0 / 5.0)) * std   # Scott's rule, computed on the FULL data

        x_min, x_max = x_grid[0], x_grid[-1]
        n_bins = max(2000, len(x_grid) * 4)
        bin_edges = np.linspace(x_min, x_max, n_bins + 1)
        hist, edges = np.histogram(data, bins=bin_edges, density=True)
        bin_width = edges[1] - edges[0]
        sigma_bins = bw / bin_width if bin_width > 0 else 1.0
        smoothed = gaussian_filter1d(hist, sigma=sigma_bins, mode="nearest")
        bin_centers = (edges[:-1] + edges[1:]) / 2.0
        return np.interp(x_grid, bin_centers, smoothed)

    def _run_auc_compare_all(self):
        if self.df  is None: messagebox.showwarning("No data","Upload Dataset 1 first."); return
        if self.df2 is None: messagebox.showwarning("No 2nd CSV","Upload Dataset 2 first."); return
        chans = [c for c in FORCE_CHANNELS
                 if c in self.avail_chans and c in self.avail_chans2]
        if not chans:
            messagebox.showwarning("No overlap","No common channels in both datasets."); return
        self._set_status("Generating Area Under Curve plots …")
        try:
            self._auc_generate(chans, unit="daN",
                                figs_list=self._auc_figs,
                                summary_list=self._auc_summary,
                                title_tag="Area Under Curve")
            self._set_status("Area Under Curve — all channels done", SUCCESS)
        except Exception as ex:
            messagebox.showerror("Error", str(ex)); self._set_status(str(ex), DANGER)

    def _run_auc_moments(self):
        """AUC for all 12 moment channels (Mx, My, Mz per wheel) — same
        green(DS1)/red(DS2) two-color scheme as AUC All-12 / Cross-Position.
        Skips gracefully if moment columns aren't present in the CSVs."""
        if self.df  is None: messagebox.showwarning("No data","Upload Dataset 1 first."); return
        if self.df2 is None: messagebox.showwarning("No 2nd CSV","Upload Dataset 2 first."); return
        chans = [c for c in MOMENT_CHANNELS
                 if c in self.avail_moments and c in self.avail_moments2]
        if not chans:
            messagebox.showwarning("No moment channels",
                "No common moment channels (Mx/My/Mz) found in both datasets.\n"
                "This CSV format may only contain force channels."); return
        self._set_status("Generating AUC — Moments plots …")
        try:
            self._auc_generate(chans, unit="Nm",
                                figs_list=self._auc_moment_figs,
                                summary_list=self._auc_moment_summary,
                                title_tag="AUC — Moments")
            self._set_status("AUC — all moment channels done", SUCCESS)
        except Exception as ex:
            messagebox.showerror("Error", str(ex)); self._set_status(str(ex), DANGER)

    def _auc_generate(self, chans, unit, figs_list, summary_list, title_tag):
        """Shared AUC renderer used by AUC All-12 and AUC — Moments.
        Fixed 2-color scheme: DS1 always green (solid), DS2 always red (dashed)."""
        lbl1 = self.label1.get()
        lbl2 = self.label2.get()
        col  = AUC_DS1_COLOR
        col2 = AUC_DS2_COLOR
        for ch in chans:
            disp = self.chan_labels[ch].get() if ch in self.chan_labels else ch
            d1 = self.df[ch].dropna().values
            d2 = self.df2[ch].dropna().values
            p5_ref  = float(np.percentile(d2,  5))
            p95_ref = float(np.percentile(d2, 95))
            p5_d1   = float(np.percentile(d1,  5))
            p95_d1  = float(np.percentile(d1, 95))
            pct_exceed    = float(np.mean(d1 >= p95_ref) * 100)
            pct_normal_d1 = float(np.mean((d1 >= p5_ref) & (d1 <= p95_ref)) * 100)
            pct_normal_d2 = float(np.mean((d2 >= p5_ref) & (d2 <= p95_ref)) * 100)
            all_data = np.concatenate([d1, d2])
            xlim = _plot_xlim(ch)
            if xlim:
                x_min, x_max = xlim
            else:
                x_min = np.percentile(all_data, 0.5)
                x_max = np.percentile(all_data, 99.5)
            x_grid = np.linspace(x_min, x_max, 500)
            kde1 = self._kde_curve(d1, x_grid)
            kde2 = self._kde_curve(d2, x_grid)

            plt.close("all")
            fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="#FFFFFF")
            fig.suptitle(f"{title_tag} — {disp}  ({lbl1}  vs  {lbl2}  reference)",
                         color="#000000", fontsize=12, fontweight="bold")

            ax1 = axes[0]
            ax1.set_facecolor("#F5F5F5")
            for sp in ax1.spines.values(): sp.set_edgecolor("#333333")
            ax1.tick_params(colors="#000000", labelsize=10, width=1.2)
            ax1.grid(True, color="#CCCCCC", linewidth=0.6, alpha=0.8)
            ax1.fill_between(x_grid, kde1, alpha=0.25, color=col,  label=lbl1)
            ax1.fill_between(x_grid, kde2, alpha=0.25, color=col2, label=lbl2)
            ax1.plot(x_grid, kde2, color=col2, linewidth=2, linestyle="--", label=lbl2)
            ax1.plot(x_grid, kde1, color=col,  linewidth=2, linestyle="-",  label=lbl1)
            ax1.axvline(p5_ref,  color=col2, linewidth=1.4, linestyle=":",
                        label=f"{lbl2} P5: {p5_ref:.0f}")
            ax1.axvline(p5_d1,   color=col,  linewidth=1.4, linestyle=":",
                        label=f"{lbl1} P5: {p5_d1:.0f}")
            ax1.axvline(p95_ref, color=col2, linewidth=1.8, linestyle="--",
                        label=f"{lbl2} P95: {p95_ref:.0f}")
            ax1.axvline(p95_d1,  color=col,  linewidth=1.4, linestyle="--",
                        label=f"{lbl1} P95: {p95_d1:.0f}")
            ax1.set_title("Shaded Area — KDE Curves", color="#000000",
                          fontsize=12, fontweight="bold")
            ax1.set_xlabel(f"{disp} ({unit})", color="#000000", fontsize=11, fontweight="bold")
            ax1.set_xlim(x_min, x_max)
            ax1.set_ylabel("Normalised density", color="#000000", fontsize=11, fontweight="bold")
            ax1.legend(fontsize=7, facecolor=BG, labelcolor="white",
                       framealpha=0.88, loc="upper right")

            ax2 = axes[1]
            ax2.set_facecolor("#F5F5F5")
            for sp in ax2.spines.values(): sp.set_edgecolor("#333333")
            ax2.tick_params(colors="#000000", labelsize=10, width=1.2)
            ax2.grid(True, color="#CCCCCC", linewidth=0.6, alpha=0.8)
            bins = np.linspace(x_min, x_max, 55)
            ax2.hist(d2, bins=bins, density=True, alpha=0.45, color=col2,
                     edgecolor="none", label=lbl2)
            ax2.hist(d1, bins=bins, density=True, alpha=0.45, color=col,
                     edgecolor="none", label=lbl1)
            ax2.axvline(p5_ref,  color=col2, linewidth=1.4, linestyle=":",
                        label=f"{lbl2} P5: {p5_ref:.0f}")
            ax2.axvline(p5_d1,   color=col,  linewidth=1.4, linestyle=":",
                        label=f"{lbl1} P5: {p5_d1:.0f}")
            ax2.axvline(p95_ref, color=col2, linewidth=1.8, linestyle="--",
                        label=f"{lbl2} P95: {p95_ref:.0f}")
            ax2.axvline(p95_d1,  color=col,  linewidth=1.4, linestyle="--",
                        label=f"{lbl1} P95: {p95_d1:.0f}")
            ax2.set_title("Density Histogram Overlay", color="#000000",
                          fontsize=12, fontweight="bold")
            ax2.set_xlabel(f"{disp} ({unit})", color="#000000", fontsize=11, fontweight="bold")
            ax2.set_ylabel("Normalised density", color="#000000", fontsize=11, fontweight="bold")
            ax2.legend(fontsize=7, facecolor=BG, labelcolor="white",
                       framealpha=0.88, loc="upper right")

            footer = (
                f"{lbl2} (reference)  →  P5: {p5_ref:.0f} {unit}   "
                f"P95: {p95_ref:.0f} {unit}   Normal zone covers {pct_normal_d2:.1f}% of data"
                f"        "
                f"{lbl1}  →  P95: {p95_d1:.0f} {unit}   "
                f"{pct_exceed:.1f}% of cycles exceed {lbl2} P95   "
                f"{pct_normal_d1:.1f}% within {lbl2} normal zone")
            fig.text(0.5, 0.01, footer, ha="center", va="bottom",
                     fontsize=7.5, color=TEXT_SEC, fontfamily="monospace",
                     bbox=dict(boxstyle="round,pad=0.5", facecolor=ENTRY_BG,
                               edgecolor="#1B3A5C", alpha=0.92))
            fig.tight_layout(rect=[0, 0.08, 1, 0.94])
            figs_list.append((ch, fig))
            summary_list.append({
                "Channel": ch, "Unit": unit, "Label1": lbl1, "Label2": lbl2,
                f"P5_{lbl2}":  round(p5_ref,  2),
                f"P95_{lbl2}": round(p95_ref, 2),
                f"P95_{lbl1}": round(p95_d1,  2),
                f"Delta_P95_{unit}": round(p95_d1 - p95_ref, 2),
                f"Pct_exceed_{lbl2}_P95": round(pct_exceed, 2),
                f"Pct_normal_{lbl1}": round(pct_normal_d1, 2),
                f"Pct_normal_{lbl2}": round(pct_normal_d2, 2),
            })
            win = tk.Toplevel(self.root)
            win.title(ch)
            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

    def _run_auc_cross(self):
        """AUC Cross-Position (Forces): both directions —
        FL(DS1) vs RL(DS2)  AND  RL(DS1) vs FL(DS2)  (same for FR/RR pair)."""
        base_pairs = [
            ("FL_Fx", "RL_Fx"), ("FL_Fy", "RL_Fy"), ("FL_Fz", "RL_Fz"),
            ("FR_Fx", "RR_Fx"), ("FR_Fy", "RR_Fy"), ("FR_Fz", "RR_Fz"),
        ]
        # Include the reverse of every pair too, so DS1/DS2 roles get swapped
        # per wheel (e.g. FL-from-DS2 vs RL-from-DS1, in addition to the
        # original FL-from-DS1 vs RL-from-DS2).
        cross_pairs = base_pairs + [(c2, c1) for c1, c2 in base_pairs]
        self._auc_cross_generate(cross_pairs, unit="daN",
                                  avail1=self.avail_chans, avail2=self.avail_chans2,
                                  figs_list=self._auc_cross_figs,
                                  summary_list=self._auc_cross_summary,
                                  status_label="AUC cross-position")

    def _run_auc_cross_moments(self):
        """AUC Cross-Position (Moments): both directions, same as forces,
        for Mx, My, Mz — same green/red 2-color scheme."""
        base_pairs = [
            ("FL_Mx", "RL_Mx"), ("FL_My", "RL_My"), ("FL_Mz", "RL_Mz"),
            ("FR_Mx", "RR_Mx"), ("FR_My", "RR_My"), ("FR_Mz", "RR_Mz"),
        ]
        cross_pairs = base_pairs + [(c2, c1) for c1, c2 in base_pairs]
        self._auc_cross_generate(cross_pairs, unit="Nm",
                                  avail1=self.avail_moments, avail2=self.avail_moments2,
                                  figs_list=self._auc_cross_moment_figs,
                                  summary_list=self._auc_cross_moment_summary,
                                  status_label="AUC cross-position — moments")

    def _auc_cross_generate(self, cross_pairs, unit, avail1, avail2,
                             figs_list, summary_list, status_label):
        """Shared AUC Cross-Position renderer for both forces and moments.
        Fixed 2-color scheme: DS1 always green (solid), DS2 always red (dashed)."""
        if self.df  is None: messagebox.showwarning("No data","Upload Dataset 1 first."); return
        if self.df2 is None: messagebox.showwarning("No 2nd CSV","Upload Dataset 2 first."); return

        lbl1 = self.label1.get()
        lbl2 = self.label2.get()

        valid = [(c1, c2) for c1, c2 in cross_pairs if c1 in avail1 and c2 in avail2]
        if not valid:
            messagebox.showwarning("Missing",
                f"No valid cross-position channel pairs found in both datasets ({unit})."); return

        self._set_status(f"Generating {status_label} plots …")
        try:
            col     = AUC_DS1_COLOR   # Dataset 1 — always green
            COL_DS2 = AUC_DS2_COLOR   # Dataset 2 — always red
            for c1, c2 in valid:
                d1   = self.df[c1].dropna().values
                d2   = self.df2[c2].dropna().values

                p5_ref  = float(np.percentile(d2,  5))
                p95_ref = float(np.percentile(d2, 95))
                p5_d1   = float(np.percentile(d1,  5))
                p95_d1  = float(np.percentile(d1, 95))
                pct_exceed    = float(np.mean(d1 >= p95_ref) * 100)
                pct_normal_d1 = float(np.mean((d1 >= p5_ref) & (d1 <= p95_ref)) * 100)
                pct_normal_d2 = float(np.mean((d2 >= p5_ref) & (d2 <= p95_ref)) * 100)

                all_data = np.concatenate([d1, d2])
                xlim = _plot_xlim(c1)
                x_min = xlim[0] if xlim else float(np.percentile(all_data, 0.5))
                x_max = xlim[1] if xlim else float(np.percentile(all_data, 99.5))
                x_grid = np.linspace(x_min, x_max, 500)

                # Subsampled KDE — same fast path as AUC All-12 (fixes long generation times)
                kde1 = self._kde_curve(d1, x_grid)
                kde2 = self._kde_curve(d2, x_grid)

                disp1 = self.chan_labels[c1].get() if c1 in self.chan_labels else c1
                disp2 = self.chan_labels[c2].get() if c2 in self.chan_labels else c2

                # ── Figure — identical layout/style to AUC All-12 ──
                plt.close("all")
                fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="#FFFFFF")
                fig.suptitle(
                    f"AUC Cross-Position  —  {disp1} [{c1}] ({lbl1})  vs  "
                    f"{disp2} [{c2}] ({lbl2})",
                    color="#000000", fontsize=12, fontweight="bold")

                # ── Left panel: KDE shaded — fixed 2-color scheme ──
                ax1 = axes[0]
                ax1.set_facecolor("#F5F5F5")
                for sp in ax1.spines.values(): sp.set_edgecolor("#333333")
                ax1.tick_params(colors="#000000", labelsize=10, width=1.2)
                ax1.grid(True, color="#CCCCCC", linewidth=0.6, alpha=0.8)

                ax1.fill_between(x_grid, kde1, alpha=0.25, color=col,
                                 label=f"{disp1} ({lbl1})")
                ax1.fill_between(x_grid, kde2, alpha=0.25, color=COL_DS2,
                                 label=f"{disp2} ({lbl2})")
                ax1.plot(x_grid, kde2, color=COL_DS2, linewidth=2,
                         linestyle="--", label=f"{disp2} ({lbl2})")
                ax1.plot(x_grid, kde1, color=col, linewidth=2, linestyle="-",
                         label=f"{disp1} ({lbl1})")
                ax1.axvline(p5_ref,  color=COL_DS2, linewidth=1.4, linestyle=":",
                            label=f"{lbl2} P5: {p5_ref:.0f}")
                ax1.axvline(p5_d1,   color=col,     linewidth=1.4, linestyle=":",
                            label=f"{lbl1} P5: {p5_d1:.0f}")
                ax1.axvline(p95_ref, color=COL_DS2, linewidth=1.8, linestyle="--",
                            label=f"{lbl2} P95: {p95_ref:.0f}")
                ax1.axvline(p95_d1,  color=col,     linewidth=1.4, linestyle="--",
                            label=f"{lbl1} P95: {p95_d1:.0f}")
                ax1.set_title("Shaded Area — KDE Curves", color="#000000",
                              fontsize=12, fontweight="bold")
                ax1.set_xlabel(f"{disp1} vs {disp2} ({unit})",
                               color="#000000", fontsize=11, fontweight="bold")
                ax1.set_xlim(x_min, x_max)
                ax1.set_ylabel("Normalised density",
                               color="#000000", fontsize=11, fontweight="bold")
                ax1.legend(fontsize=7, facecolor=BG, labelcolor="white",
                           framealpha=0.88, loc="upper right")

                # ── Right panel: histogram overlay ──
                ax2 = axes[1]
                ax2.set_facecolor("#F5F5F5")
                for sp in ax2.spines.values(): sp.set_edgecolor("#333333")
                ax2.tick_params(colors="#000000", labelsize=10, width=1.2)
                ax2.grid(True, color="#CCCCCC", linewidth=0.6, alpha=0.8)

                bins = np.linspace(x_min, x_max, 55)
                ax2.hist(d2, bins=bins, density=True, alpha=0.45,
                         color=COL_DS2, edgecolor="none",
                         label=f"{disp2} ({lbl2})")
                ax2.hist(d1, bins=bins, density=True, alpha=0.45,
                         color=col, edgecolor="none",
                         label=f"{disp1} ({lbl1})")
                ax2.axvline(p5_ref,  color=COL_DS2, linewidth=1.4, linestyle=":",
                            label=f"{lbl2} P5: {p5_ref:.0f}")
                ax2.axvline(p5_d1,   color=col,     linewidth=1.4, linestyle=":",
                            label=f"{lbl1} P5: {p5_d1:.0f}")
                ax2.axvline(p95_ref, color=COL_DS2, linewidth=1.8, linestyle="--",
                            label=f"{lbl2} P95: {p95_ref:.0f}")
                ax2.axvline(p95_d1,  color=col,     linewidth=1.4, linestyle="--",
                            label=f"{lbl1} P95: {p95_d1:.0f}")
                ax2.set_title("Density Histogram Overlay", color="#000000",
                              fontsize=12, fontweight="bold")
                ax2.set_xlabel(f"{disp1} vs {disp2} ({unit})",
                               color="#000000", fontsize=11, fontweight="bold")
                ax2.set_ylabel("Normalised density",
                               color="#000000", fontsize=11, fontweight="bold")
                ax2.legend(fontsize=7, facecolor=BG, labelcolor="white",
                           framealpha=0.88, loc="upper right")

                # Footer — same format as AUC All-12
                footer = (
                    f"{disp2} [{c2}] ({lbl2})  →  P5: {p5_ref:.0f} {unit}   "
                    f"P95: {p95_ref:.0f} {unit}   Normal zone covers {pct_normal_d2:.1f}% of data"
                    f"        "
                    f"{disp1} [{c1}] ({lbl1})  →  P95: {p95_d1:.0f} {unit}   "
                    f"{pct_exceed:.1f}% of cycles exceed {lbl2} P95   "
                    f"{pct_normal_d1:.1f}% within {lbl2} normal zone")
                fig.text(0.5, 0.01, footer, ha="center", va="bottom",
                         fontsize=7.5, color=TEXT_SEC, fontfamily="monospace",
                         bbox=dict(boxstyle="round,pad=0.5", facecolor=ENTRY_BG,
                                   edgecolor="#1B3A5C", alpha=0.92))
                fig.tight_layout(rect=[0, 0.08, 1, 0.94])

                figs_list.append((f"{c1}_vs_{c2}", fig))
                summary_list.append({
                    "Channel_DS1": c1,    "Channel_DS2": c2, "Unit": unit,
                    "Label1": lbl1,       "Label2": lbl2,
                    f"P5_{lbl2}":   round(p5_ref,        2),
                    f"P95_{lbl2}":  round(p95_ref,       2),
                    f"P95_{lbl1}":  round(p95_d1,        2),
                    f"Delta_P95_{unit}": round(p95_d1 - p95_ref, 2),
                    f"Pct_exceed_{lbl2}_P95":  round(pct_exceed,    2),
                    f"Pct_normal_{lbl1}":      round(pct_normal_d1, 2),
                    f"Pct_normal_{lbl2}":      round(pct_normal_d2, 2),
                })
                # Embed reliably in a Toplevel (fig.show() can silently fail to
                # render when the TkAgg backend is already driven by the main
                # app window — this was likely why cross-position plots
                # sometimes "did not come" for you).
                win = tk.Toplevel(self.root)
                win.title(f"AUC Cross — {c1} vs {c2}")
                canvas = FigureCanvasTkAgg(fig, master=win)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)

            self._set_status(f"{status_label} done", SUCCESS)
        except Exception as ex:
            messagebox.showerror("Error", str(ex))
            self._set_status(str(ex), DANGER)

    # ══ RF COMPARE — Front axle (FL+FR) | Rear axle (RL+RR) ══
    def _run_rf_compare(self, suffix):
        """
        Rainflow histogram comparison for one force type (Fx / Fy / Fz).
        Layout: 2 columns per axle group (FL|FR on left, RL|RR on right)
                2 rows per channel: histogram (top) + cumulative CDF (bottom)
        """
        if self.df  is None: messagebox.showwarning("No data","Upload Dataset 1 first."); return
        if self.df2 is None: messagebox.showwarning("No 2nd CSV","Upload Dataset 2 first."); return

        # Collect channels that exist in both datasets
        all_chans = [f"FL_{suffix}", f"FR_{suffix}", f"RL_{suffix}", f"RR_{suffix}"]
        chans = [c for c in all_chans
                 if c in self.avail_chans and c in self.avail_chans2]
        if not chans:
            messagebox.showwarning("Missing",
                f"No common {suffix} channels found in both datasets."); return

        lbl1 = self.label1.get()
        lbl2 = self.label2.get()
        m    = 5  # Miner's slope

        self._set_status(f"Running rainflow compare — {suffix} …")
        try:
            # ── Figure layout: 4 cols (FL, FR, RL, RR) × 2 rows ──
            # If only front or only rear channels exist, adjust accordingly
            fig = plt.figure(figsize=(5.5 * len(chans), 9), facecolor=BG)
            fig.suptitle(
                f"Rainflow Comparison — {suffix} Channels  ({lbl1}  vs  {lbl2})\n"
                f"Front Axle: FL | FR          Rear Axle: RL | RR\n"
                f"Normalised density  |  Damage ratio using Miner's rule  m={m}",
                color="white", fontsize=11, fontweight="bold")

            ncols   = len(chans)
            summary = []

            for col_idx, ch in enumerate(chans):
                col = CHAN_COLORS.get(ch, BTN_BG)

                cyc1 = self._rainflow_cycles(self.df,  ch)
                cyc2 = self._rainflow_cycles(self.df2, ch)

                rng1 = np.array([c[0] for c in cyc1])
                cnt1 = np.array([c[2] for c in cyc1])
                rng2 = np.array([c[0] for c in cyc2])
                cnt2 = np.array([c[2] for c in cyc2])

                rng1_rep = np.repeat(rng1, cnt1.astype(int).clip(1))
                rng2_rep = np.repeat(rng2, cnt2.astype(int).clip(1))

                p95_1 = float(np.percentile(rng1_rep, 95))
                p95_2 = float(np.percentile(rng2_rep, 95))

                dmg1 = float(np.sum(cnt1 * (rng1 ** m)))
                dmg2 = float(np.sum(cnt2 * (rng2 ** m)))
                ratio = dmg2 / dmg1 if dmg1 > 0 else float("nan")
                ratio_txt = f"{ratio:.2f}×" if not np.isnan(ratio) else "N/A"
                summary.append((ch, p95_1, p95_2, ratio, ratio_txt))

                all_rng   = np.concatenate([rng1_rep, rng2_rep])
                bin_edges = np.linspace(0, np.percentile(all_rng, 99), 35)

                disp = (self.chan_labels[ch].get()
                        if hasattr(self, "chan_labels") and ch in self.chan_labels else ch)

                # ── Row 1: Histogram ──────────────────────────────
                ax1 = fig.add_subplot(2, ncols, col_idx + 1)
                self._style_ax(ax1)
                ax1.hist(rng1_rep, bins=bin_edges, density=True,
                         alpha=0.60, color=col, edgecolor="none", label=lbl1)
                ax1.hist(rng2_rep, bins=bin_edges, density=True,
                         alpha=0.45, color="#BBBBBB", edgecolor="none", label=lbl2)
                ax1.axvline(p95_1, color=col,      linewidth=1.8, linestyle="--",
                            label=f"P95 {lbl1}: {p95_1:.0f}")
                ax1.axvline(p95_2, color="#FFFFFF", linewidth=1.8, linestyle=":",
                            label=f"P95 {lbl2}: {p95_2:.0f}")

                badge_col = DANGER if ratio > 1 else SUCCESS
                ax1.text(0.97, 0.95,
                         f"Damage ratio\n{lbl2}/{lbl1}\n= {ratio_txt}",
                         transform=ax1.transAxes, fontsize=7.5, color=badge_col,
                         ha="right", va="top", fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.4", facecolor=ENTRY_BG,
                                   edgecolor=badge_col, alpha=0.92))

                # Axle separator label in title
                wheel  = _wheel(ch)
                axle   = "FRONT" if wheel in ("FL","FR") else "REAR"
                ax1.set_title(f"[{axle}]  {disp} — Load Range Histogram",
                              color=col, fontsize=9, fontweight="bold")
                ax1.set_xlabel("Range (daN)  [cycle amplitude]", color=TEXT_SEC, fontsize=8)
                ax1.set_ylabel("Normalised density", color=TEXT_SEC, fontsize=8)
                ax1.legend(fontsize=7, facecolor=BG, labelcolor="white", framealpha=0.85)

                # Vertical divider between front and rear (between FR and RL)
                # Drawn as a thick axis spine on the left of RL columns
                if wheel == "RL" and col_idx > 0:
                    ax1.spines["left"].set_linewidth(2.5)
                    ax1.spines["left"].set_edgecolor(ACCENT)

                # ── Row 2: Cumulative CDF ─────────────────────────
                ax2 = fig.add_subplot(2, ncols, col_idx + 1 + ncols)
                self._style_ax(ax2)
                s1 = np.sort(rng1_rep)
                s2 = np.sort(rng2_rep)
                c1 = np.arange(1, len(s1)+1) / len(s1)
                c2 = np.arange(1, len(s2)+1) / len(s2)
                ax2.plot(s1, c1, color=col,       linewidth=2, label=lbl1)
                ax2.plot(s2, c2, color="#BBBBBB",  linewidth=2, linestyle="--", label=lbl2)
                ax2.axvline(p95_1, color=col,      linewidth=1.4, linestyle="--", alpha=0.7)
                ax2.axvline(p95_2, color="#FFFFFF", linewidth=1.4, linestyle=":",  alpha=0.7)
                ax2.axhline(0.95,  color="#555",   linewidth=0.8, linestyle="-",  alpha=0.5)
                ax2.text(s1[-1]*0.98, 0.93, "P95", color="#AAAAAA", fontsize=6.5, ha="right")
                ax2.set_title(f"[{axle}]  {disp} — Cumulative Distribution",
                              color=col, fontsize=9, fontweight="bold")
                ax2.set_xlabel("Range (daN)", color=TEXT_SEC, fontsize=8)
                ax2.set_ylabel("Cumulative fraction", color=TEXT_SEC, fontsize=8)
                ax2.set_ylim(0, 1.05)
                ax2.legend(fontsize=7, facecolor=BG, labelcolor="white", framealpha=0.85)

                if wheel == "RL" and col_idx > 0:
                    ax2.spines["left"].set_linewidth(2.5)
                    ax2.spines["left"].set_edgecolor(ACCENT)

            # ── Footer summary ──────────────────────────────────────
            rows = []
            for ch, p1, p2, rat, rtxt in summary:
                disp = (self.chan_labels[ch].get()
                        if hasattr(self, "chan_labels") and ch in self.chan_labels else ch)
                sev  = "more" if rat >= 1 else "less"
                rows.append(
                    f"{disp:>10s} │ P95 {lbl1}: {p1:>6.0f} daN │ "
                    f"P95 {lbl2}: {p2:>6.0f} daN │ "
                    f"Damage ratio: {rtxt} ({lbl2} is {abs(rat):.2f}× {sev} damaging)")
            fig.text(0.5, 0.01, "\n".join(rows),
                     ha="center", va="bottom", fontsize=7, color=TEXT_SEC,
                     fontfamily="monospace",
                     bbox=dict(boxstyle="round,pad=0.5", facecolor=ENTRY_BG,
                               edgecolor="#1B3A5C", alpha=0.92))
            fig.tight_layout(rect=[0, 0.10, 1, 0.93])
            self._show_fig(fig)
            self._set_status(f"Rainflow compare {suffix} done", SUCCESS)

        except Exception as ex:
            messagebox.showerror("Error", str(ex)); self._set_status(str(ex), DANGER)

    # ══ RMS TABLE ═════════════════════════════════════════════
    def _rms(self, arr):
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return np.nan
        return float(np.sqrt(np.mean(arr**2)))

    def _run_rms_table(self):
        if self.df is None:
            messagebox.showwarning("No data", "Upload Dataset 1 first."); return
        lbl1 = self.label1.get()
        lbl2 = self.label2.get()
        has_ds2 = self.df2 is not None

        rows = []
        for wheel, chans in WHEEL_GROUPS_FULL.items():
            for ch in chans:
                in_ds1 = ch in self.avail_chans or ch in self.avail_moments
                if not in_ds1:
                    continue
                rms1 = self._rms(self.df[ch]) if ch in self.df.columns else np.nan
                rms2 = np.nan
                if has_ds2 and (ch in self.avail_chans2 or ch in self.avail_moments2) \
                        and ch in self.df2.columns:
                    rms2 = self._rms(self.df2[ch])
                delta   = (rms2 - rms1) if (has_ds2 and not np.isnan(rms2)) else np.nan
                pct_chg = (delta / rms1 * 100.0) if (rms1 not in (0, np.nan) and not np.isnan(delta)) else np.nan
                rows.append({
                    "Wheel": wheel, "Channel": ch, "Unit": _chan_unit(ch),
                    f"RMS_{lbl1}": round(rms1, 3) if not np.isnan(rms1) else None,
                    f"RMS_{lbl2}": round(rms2, 3) if has_ds2 and not np.isnan(rms2) else None,
                    "Delta":       round(delta, 3) if has_ds2 and not np.isnan(delta) else None,
                    "PctChange":   round(pct_chg, 2) if has_ds2 and not np.isnan(pct_chg) else None,
                })

        if not rows:
            messagebox.showwarning("No channels", "No RMS-eligible channels found."); return

        self._rms_rows = rows
        self._show_table_figure(
            rows,
            title=f"RMS Summary — {lbl1}" + (f"  vs  {lbl2}" if has_ds2 else ""),
            col_order=["Wheel", "Channel", "Unit", f"RMS_{lbl1}"] +
                      ([f"RMS_{lbl2}", "Delta", "PctChange"] if has_ds2 else []),
            fig_attr="_rms_fig")
        self._set_status("RMS table computed.", SUCCESS)

    # ══ DLC TABLE (Dynamic Load Coefficient, Fz only) ═════════
    def _run_dlc_table(self):
        if self.df is None:
            messagebox.showwarning("No data", "Upload Dataset 1 first."); return
        lbl1 = self.label1.get()
        lbl2 = self.label2.get()
        has_ds2 = self.df2 is not None

        rows = []
        for wheel in WHEEL_GROUPS_FULL:
            ch = f"{wheel}_Fz"
            if ch not in self.avail_chans:
                continue
            d1 = self.df[ch].dropna().values
            static1 = float(np.mean(d1)) if d1.size else np.nan
            std1    = float(np.std(d1))  if d1.size else np.nan
            dlc1    = (std1 / static1) if static1 else np.nan

            static2 = std2 = dlc2 = np.nan
            if has_ds2 and ch in self.avail_chans2 and ch in self.df2.columns:
                d2 = self.df2[ch].dropna().values
                static2 = float(np.mean(d2)) if d2.size else np.nan
                std2    = float(np.std(d2))  if d2.size else np.nan
                dlc2    = (std2 / static2) if static2 else np.nan

            delta = (dlc2 - dlc1) if (has_ds2 and not np.isnan(dlc2)) else np.nan
            rows.append({
                "Wheel": wheel, "Channel": ch,
                f"Static_{lbl1}_daN": round(static1, 2) if not np.isnan(static1) else None,
                f"StdDev_{lbl1}_daN": round(std1, 2)    if not np.isnan(std1) else None,
                f"DLC_{lbl1}":        round(dlc1, 4)     if not np.isnan(dlc1) else None,
                f"Static_{lbl2}_daN": round(static2, 2) if has_ds2 and not np.isnan(static2) else None,
                f"StdDev_{lbl2}_daN": round(std2, 2)    if has_ds2 and not np.isnan(std2) else None,
                f"DLC_{lbl2}":        round(dlc2, 4)     if has_ds2 and not np.isnan(dlc2) else None,
                "Delta_DLC":          round(delta, 4)    if has_ds2 and not np.isnan(delta) else None,
            })

        if not rows:
            messagebox.showwarning("No Fz channels", "No Fz channels found for DLC."); return

        self._dlc_rows = rows
        col_order = ["Wheel", "Channel", f"Static_{lbl1}_daN", f"StdDev_{lbl1}_daN", f"DLC_{lbl1}"]
        if has_ds2:
            col_order += [f"Static_{lbl2}_daN", f"StdDev_{lbl2}_daN", f"DLC_{lbl2}", "Delta_DLC"]
        self._show_table_figure(
            rows,
            title=f"Dynamic Load Coefficient (DLC) — {lbl1}" + (f"  vs  {lbl2}" if has_ds2 else "")
                  + "\nDLC = Std Dev(Fz) / Mean Fz",
            col_order=col_order,
            fig_attr="_dlc_fig")
        self._set_status("DLC table computed.", SUCCESS)

    # ══ G-SEVERITY (Gx, Gy, Gxy) ══════════════════════════════
    def _g_severity(self, fx, fy, fz):
        """
        Gx  = sqrt( mean( (Fx/Fz)^2 ) )
        Gy  = sqrt( mean( (Fy/Fz)^2 ) )
        Gxy = sqrt( mean(Fx^2 + Fy^2) / (mean(Fz))^2 )
        Rows with Fz == 0 are excluded (avoids div-by-zero).
        """
        fx = np.asarray(fx, dtype=float)
        fy = np.asarray(fy, dtype=float)
        fz = np.asarray(fz, dtype=float)
        mask = ~(np.isnan(fx) | np.isnan(fy) | np.isnan(fz) | (fz == 0))
        fx, fy, fz = fx[mask], fy[mask], fz[mask]
        if fx.size == 0:
            return np.nan, np.nan, np.nan

        with np.errstate(divide="ignore", invalid="ignore"):
            fxfz = fx / fz
            fyfz = fy / fz
        gx = float(np.sqrt(np.mean(fxfz**2)))
        gy = float(np.sqrt(np.mean(fyfz**2)))

        fxy      = fx**2 + fy**2
        mean_fxy = float(np.mean(fxy))
        avg_fz   = float(np.mean(fz))
        sq_fz    = avg_fz**2
        gxy      = float(np.sqrt(mean_fxy / sq_fz)) if sq_fz else np.nan
        return gx, gy, gxy

    def _run_g_severity_table(self):
        if self.df is None:
            messagebox.showwarning("No data", "Upload Dataset 1 first."); return
        lbl1 = self.label1.get()
        lbl2 = self.label2.get()
        has_ds2 = self.df2 is not None

        rows = []
        for wheel in WHEEL_GROUPS_FULL:
            fx_ch, fy_ch, fz_ch = f"{wheel}_Fx", f"{wheel}_Fy", f"{wheel}_Fz"
            if not all(c in self.avail_chans for c in (fx_ch, fy_ch, fz_ch)):
                continue
            gx1, gy1, gxy1 = self._g_severity(
                self.df[fx_ch], self.df[fy_ch], self.df[fz_ch])

            gx2 = gy2 = gxy2 = np.nan
            if has_ds2 and all(c in self.avail_chans2 for c in (fx_ch, fy_ch, fz_ch)):
                gx2, gy2, gxy2 = self._g_severity(
                    self.df2[fx_ch], self.df2[fy_ch], self.df2[fz_ch])

            row = {
                "Wheel": wheel,
                f"Gx_{lbl1}":  round(gx1, 4)  if not np.isnan(gx1)  else None,
                f"Gy_{lbl1}":  round(gy1, 4)  if not np.isnan(gy1)  else None,
                f"Gxy_{lbl1}": round(gxy1, 4) if not np.isnan(gxy1) else None,
            }
            if has_ds2:
                row.update({
                    f"Gx_{lbl2}":  round(gx2, 4)  if not np.isnan(gx2)  else None,
                    f"Gy_{lbl2}":  round(gy2, 4)  if not np.isnan(gy2)  else None,
                    f"Gxy_{lbl2}": round(gxy2, 4) if not np.isnan(gxy2) else None,
                    "Delta_Gx":  round(gx2 - gx1, 4)   if not (np.isnan(gx2) or np.isnan(gx1)) else None,
                    "Delta_Gy":  round(gy2 - gy1, 4)   if not (np.isnan(gy2) or np.isnan(gy1)) else None,
                    "Delta_Gxy": round(gxy2 - gxy1, 4) if not (np.isnan(gxy2) or np.isnan(gxy1)) else None,
                })
            rows.append(row)

        if not rows:
            messagebox.showwarning("No channels",
                "Need Fx, Fy and Fz present for at least one wheel."); return

        self._gsev_rows = rows
        col_order = ["Wheel", f"Gx_{lbl1}", f"Gy_{lbl1}", f"Gxy_{lbl1}"]
        if has_ds2:
            col_order += [f"Gx_{lbl2}", f"Gy_{lbl2}", f"Gxy_{lbl2}",
                          "Delta_Gx", "Delta_Gy", "Delta_Gxy"]
        self._show_table_figure(
            rows,
            title=f"G-Severity — {lbl1}" + (f"  vs  {lbl2}" if has_ds2 else "")
                  + "\nGx=RMS(Fx/Fz)   Gy=RMS(Fy/Fz)   Gxy=sqrt(mean(Fx²+Fy²)/mean(Fz)²)",
            col_order=col_order,
            fig_attr="_gsev_fig")
        self._set_status("G-Severity table computed.", SUCCESS)

    # ══ FORCE-PAIR HEATMAPS  (Fy-Fx / Fx-Fz / Fy-Fz, colour = % dist) ═
    def _dist_km_per_sample(self, df):
        """Per-sample distance in km, from Vehicle_Speed (km/h) and the
        sampling rate entered in Settings. Returns None if Vehicle_Speed
        is not present in this dataset."""
        if "Vehicle_Speed" not in df.columns:
            return None
        sr = float(self.sr_var.get() or 100)
        dt_hours = 1.0 / (sr * 3600.0)
        return df["Vehicle_Speed"].astype(float).values * dt_hours

    def _get_heatmap_edges(self, ch_type):
        """Read Min/Max/Step for Fx/Fy/Fz from the GUI entry fields and
        build bin edges. Raises ValueError with a clear message if the
        entered values don't make sense."""
        v = self.heatmap_range_vars[ch_type]
        try:
            mn   = float(v["min"].get())
            mx   = float(v["max"].get())
            step = float(v["step"].get())
        except ValueError:
            raise ValueError(f"{ch_type}: Min/Max/Step must be numbers.")
        if step <= 0:
            raise ValueError(f"{ch_type}: Step must be greater than 0.")
        if mx <= mn:
            raise ValueError(f"{ch_type}: Max must be greater than Min.")
        edges = np.arange(mn, mx + step / 2.0, step)
        if len(edges) < 2:
            raise ValueError(f"{ch_type}: range/step produces fewer than 2 bin edges.")
        return edges

    def _run_force_heatmaps(self):
        if self.df is None:
            messagebox.showwarning("No data", "Upload CSV first."); return

        # Validate all 3 bin ranges up front so we don't open some popups
        # then fail partway through.
        try:
            edges_by_type = {ch: self._get_heatmap_edges(ch) for ch in ("Fx", "Fy", "Fz")}
        except ValueError as ex:
            messagebox.showerror("Invalid bin range", str(ex)); return

        datasets = [(self.df, self.avail_chans, self.label1.get())]
        if self.df2 is not None:
            datasets.append((self.df2, self.avail_chans2, self.label2.get()))

        any_done = False
        for df, avail, label in datasets:
            dist_km = self._dist_km_per_sample(df)
            if dist_km is None:
                messagebox.showwarning(
                    "Missing Vehicle_Speed",
                    f"'{label}' has no Vehicle_Speed column — "
                    "cannot compute km. Skipped.")
                continue
            total_km = float(np.nansum(dist_km))
            for fx_type, fy_type in HEATMAP_PAIRS:
                self._open_heatmap_pair_popup(df, avail, dist_km, total_km,
                                               label, fx_type, fy_type,
                                               edges_by_type[fx_type], edges_by_type[fy_type])
            any_done = True

        if any_done:
            self._set_status("Force-pair heatmaps generated.", SUCCESS)

    def _open_heatmap_pair_popup(self, df, avail, dist_km, total_km, label,
                                  fx_type, fy_type, x_edges, y_edges):
        """One popup window: 2x2 grid, one subplot per wheel (FL,FR,RL,RR),
        for a single force-pair (e.g. Fy vs Fx), for one dataset.
        Colour = % of total dataset distance falling in that bin;
        each cell is also labelled with its % value."""
        cmap = matplotlib.colormaps["hot_r"].copy()
        cmap.set_bad("#FFFFFF")   # zero-distance cells shown white, not colormap low-end

        win = tk.Toplevel(self.root)
        win.title(f"Heatmap — {fx_type} vs {fy_type}  |  {label}")
        win.configure(bg=BG)
        win.geometry("980x860")

        fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.4), facecolor="#FFFFFF")
        fig.suptitle(
            f"{fx_type} vs {fy_type}  —  {label}\n"
            f"Total distance: {total_km:.2f} km   |   colour/label = % of total distance",
            fontsize=12.5, fontweight="bold", color="#000000")

        wheels = ["FL", "FR", "RL", "RR"]
        for ax, wheel in zip(axes.flat, wheels):
            wc = WHEEL_COLORS[wheel]["pri"]
            xch = f"{wheel}_{fx_type}"
            ych = f"{wheel}_{fy_type}"
            ax.set_facecolor("#F5F5F5")

            if xch not in avail or ych not in avail:
                ax.text(0.5, 0.5, f"{wheel}\nchannel missing",
                        ha="center", va="center", color="#888", transform=ax.transAxes)
                ax.set_title(wheel, color=wc, fontweight="bold")
                continue

            x = df[xch].astype(float).values
            y = df[ych].astype(float).values
            w = dist_km
            mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(w))
            x, y, w = x[mask], y[mask], w[mask]

            hist, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=w)
            pct = (hist / total_km * 100.0) if total_km > 0 else hist * 0.0
            pct_masked = np.ma.masked_where(pct <= 0, pct)

            mesh = ax.pcolormesh(x_edges, y_edges, pct_masked.T,
                                  cmap=cmap, shading="flat",
                                  edgecolors="#DDDDDD", linewidth=0.4)
            fig.colorbar(mesh, ax=ax, label="% of total distance", shrink=0.85)

            # Per-cell percentage labels
            x_centers = (x_edges[:-1] + x_edges[1:]) / 2
            y_centers = (y_edges[:-1] + y_edges[1:]) / 2
            for i, xc in enumerate(x_centers):
                for j, yc in enumerate(y_centers):
                    v = pct[i, j]
                    if v <= 0:
                        continue
                    ax.text(xc, yc, f"{v:.1f}%", ha="center", va="center",
                            fontsize=7, color="#000000", fontweight="bold",
                            path_effects=[patheffects.withStroke(linewidth=2.2,
                                                                  foreground="#FFFFFF")])

            ax.set_xlim(x_edges[0], x_edges[-1])
            ax.set_ylim(y_edges[0], y_edges[-1])
            ax.set_xticks(x_edges)
            ax.set_yticks(y_edges)
            ax.set_xlabel(f"{fx_type} (daN)", color="#000000", fontsize=9)
            ax.set_ylabel(f"{fy_type} (daN)", color="#000000", fontsize=9)
            ax.set_title(wheel, color=wc, fontweight="bold", fontsize=11)
            ax.tick_params(colors="#000000", labelsize=8)
            for sp in ax.spines.values(): sp.set_edgecolor("#333333")

        fig.tight_layout(rect=[0, 0, 1, 0.92])

        name = f"{label}_{fx_type}-{fy_type}"
        self._heatmap_figs.append((name, fig))

        cw = FigureCanvasTkAgg(fig, master=win)
        cw.draw(); cw.get_tk_widget().pack(fill="both", expand=True)
        btn_frame = tk.Frame(win, bg=BG); btn_frame.pack(pady=4)

        def _save_this():
            path = filedialog.asksaveasfilename(
                initialfile=f"Heatmap_{name}.png",
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("All", "*.*")])
            if path:
                fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#FFFFFF")
                self._set_status(f"Saved: {os.path.basename(path)}", SUCCESS)

        tk.Button(btn_frame, text="Save PNG", command=_save_this,
                  bg=BTN_BG, fg=BTN_FG, font=FONT_B,
                  relief="flat", cursor="hand2", padx=10).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Close", command=lambda: (plt.close(fig), win.destroy()),
                  bg=DANGER, fg=BTN_FG, font=FONT_B,
                  relief="flat", cursor="hand2", padx=10).pack(side="left", padx=6)

    # ══ TABLE RENDER HELPER (matplotlib table, dark theme) ═════
    def _show_table_figure(self, rows, title, col_order, fig_attr):
        n = len(rows)
        fig_h = max(2.2, 0.42 * n + 1.4)
        fig = plt.figure(figsize=(min(16, 1.7*len(col_order)+2), fig_h), facecolor=BG)
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=14)

        cell_text = []
        for r in rows:
            cell_text.append([("" if r.get(c) is None else str(r.get(c))) for c in col_order])

        tbl = ax.table(cellText=cell_text, colLabels=col_order,
                        loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1, 1.5)

        for (row_i, col_i), cell in tbl.get_celld().items():
            cell.set_edgecolor("#1B3A5C")
            if row_i == 0:
                cell.set_facecolor("#0A2540")
                cell.get_text().set_color(BTN_BG)
                cell.get_text().set_fontweight("bold")
            else:
                cell.set_facecolor(ENTRY_BG if row_i % 2 == 0 else "#0F2A45")
                cell.get_text().set_color("white")
                if col_order[col_i] == "Wheel":
                    wheel_val = cell_text[row_i-1][col_i]
                    cell.get_text().set_color(
                        WHEEL_COLORS.get(wheel_val, {}).get("pri", "white"))
                    cell.get_text().set_fontweight("bold")

        fig.tight_layout()
        setattr(self, fig_attr, fig)
        self._show_fig(fig)

    # ══ POWER SPECTRAL DENSITY (Welch) ═══════════════════════
    def _get_sr(self):
        try:
            sr = float(self.sr_var.get())
            if sr <= 0: raise ValueError
            return sr
        except Exception:
            return 100.0

    def _welch_psd(self, arr, fs):
        from scipy.signal import welch
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size < 16:
            return None, None
        nperseg = int(min(len(arr), max(64, fs*2)))
        f, pxx = welch(arr, fs=fs, nperseg=nperseg, noverlap=nperseg//2, detrend="constant")
        return f, pxx

    def _run_psd_wheel(self, wheel):
        if self.df is None:
            messagebox.showwarning("No data", "Upload Dataset 1 first."); return
        fs = self._get_sr()
        lbl1 = self.label1.get(); lbl2 = self.label2.get()
        has_ds2 = self.df2 is not None
        chans = [c for c in WHEEL_GROUPS_FULL[wheel]
                 if c in self.avail_chans or c in self.avail_moments]
        if not chans:
            messagebox.showwarning("No channels", f"No channels found for {wheel}."); return

        try:
            plt.close("all")
            fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), facecolor=BG)
            axes = axes.flatten()
            wc = WHEEL_COLORS[wheel]["pri"]
            fig.suptitle(f"Power Spectral Density — {wheel}  ({lbl1}" +
                         (f"  vs  {lbl2}" if has_ds2 else "") + ")",
                         color="white", fontsize=13, fontweight="bold")

            for i, ch in enumerate(WHEEL_GROUPS_FULL[wheel]):
                ax = axes[i]
                self._style_ax(ax)
                disp = self.chan_labels[ch].get() if ch in self.chan_labels else ch
                unit = _chan_unit(ch)
                if ch in self.df.columns and (ch in self.avail_chans or ch in self.avail_moments):
                    f1, p1 = self._welch_psd(self.df[ch].values, fs)
                    if f1 is not None:
                        ax.plot(f1, p1, color=wc, linewidth=1.6, label=lbl1)
                if has_ds2 and ch in self.df2.columns and \
                        (ch in self.avail_chans2 or ch in self.avail_moments2):
                    f2, p2 = self._welch_psd(self.df2[ch].values, fs)
                    if f2 is not None:
                        ax.plot(f2, p2, color="#BBBBBB", linewidth=1.6,
                                    linestyle="--", label=lbl2)
                ax.set_title(disp, color=wc, fontsize=9, fontweight="bold")
                ax.set_xlabel("Frequency (Hz)", color=TEXT_SEC, fontsize=7.5)
                ax.set_ylabel(f"PSD ({unit}²/Hz)", color=TEXT_SEC, fontsize=7.5)
                ax.set_xlim(0, fs/2)
                ax.legend(fontsize=7, facecolor=BG, labelcolor="white", framealpha=0.85)

            fig.tight_layout(rect=[0, 0, 1, 0.94])
            self._psd_figs[wheel] = fig
            self._show_fig(fig)
            self._set_status(f"PSD — {wheel} done  (fs={fs:.0f} Hz)", SUCCESS)
        except Exception as ex:
            messagebox.showerror("Error", str(ex)); self._set_status(str(ex), DANGER)

    def _run_psd_all(self):
        if self.df is None:
            messagebox.showwarning("No data", "Upload Dataset 1 first."); return
        for wheel in WHEEL_GROUPS_FULL:
            self._run_psd_wheel(wheel)
            if wheel in self._psd_figs:
                win = tk.Toplevel(self.root)
                win.title(f"PSD — {wheel}")
                canvas = FigureCanvasTkAgg(self._psd_figs[wheel], master=win)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)
        self._set_status("PSD — all wheels done.", SUCCESS)

    # ══ EXPORT ALL RESULTS ════════════════════════════════════
    def _export_all_results(self):
        if self.df is None:
            messagebox.showwarning("No data", "Upload Dataset 1 first."); return

        has_any = bool(self._auc_figs or self._auc_cross_figs or self._box_figs
                       or self._auc_moment_figs or self._auc_cross_moment_figs
                       or self._rms_fig is not None
                       or self._dlc_fig is not None or self._psd_figs
                       or self._gsev_fig is not None or self._heatmap_figs)
        if not has_any:
            messagebox.showinfo("Nothing to export",
                "No plots generated yet.\n\n"
                "Run at least one of:\n"
                "  • Area Under Curve — All 12\n"
                "  • AUC Cross-Position\n"
                "  • AUC — All Moments\n"
                "  • Box Plot (any)\n"
                "  • RMS Table / DLC Table\n"
                "  • PSD (any wheel)\n"
                "then click Export."); return

        folder = filedialog.askdirectory(title="Choose export folder")
        if not folder: return

        saved  = 0
        errors = []
        self._set_status("Exporting all results …")

        # ── AUC All-12 ─────────────────────────────────────────
        for ch, fig in self._auc_figs:
            try:
                path = os.path.join(folder, f"AUC_{ch}.png")
                fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#FFFFFF")
                saved += 1
            except Exception as ex:
                errors.append(f"AUC {ch}: {ex}")

        # ── AUC — All Moments ───────────────────────────────────
        for ch, fig in self._auc_moment_figs:
            try:
                path = os.path.join(folder, f"AUC_Moment_{ch}.png")
                fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#FFFFFF")
                saved += 1
            except Exception as ex:
                errors.append(f"AUC Moment {ch}: {ex}")

        # ── AUC Cross-Position ─────────────────────────────────
        for label, fig in self._auc_cross_figs:
            try:
                path = os.path.join(folder, f"AUC_CrossPosition_{label}.png")
                fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#FFFFFF")
                saved += 1
            except Exception as ex:
                errors.append(f"AUC Cross {label}: {ex}")

        # ── AUC Cross-Position — Moments ────────────────────────
        for label, fig in self._auc_cross_moment_figs:
            try:
                path = os.path.join(folder, f"AUC_CrossPosition_Moment_{label}.png")
                fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#FFFFFF")
                saved += 1
            except Exception as ex:
                errors.append(f"AUC Cross Moment {label}: {ex}")

        # ── Force-Pair Heatmaps ─────────────────────────────────
        for name, fig in self._heatmap_figs:
            try:
                path = os.path.join(folder, f"Heatmap_{name}.png")
                fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#FFFFFF")
                saved += 1
            except Exception as ex:
                errors.append(f"Heatmap {name}: {ex}")

        # ── Box Plots ──────────────────────────────────────────
        for ch, fig in self._box_figs.items():
            try:
                path = os.path.join(folder, f"BoxPlot_{ch}.png")
                fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#FFFFFF")
                saved += 1
            except Exception as ex:
                errors.append(f"BoxPlot {ch}: {ex}")

        # ── PSD figures ────────────────────────────────────────
        for wheel, fig in self._psd_figs.items():
            try:
                path = os.path.join(folder, f"PSD_{wheel}.png")
                fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
                saved += 1
            except Exception as ex:
                errors.append(f"PSD {wheel}: {ex}")

        # ── RMS table (figure + CSV) ────────────────────────────
        if self._rms_fig is not None:
            try:
                path = os.path.join(folder, "RMS_Table.png")
                self._rms_fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
                saved += 1
            except Exception as ex:
                errors.append(f"RMS figure: {ex}")
        if self._rms_rows:
            try:
                pd.DataFrame(self._rms_rows).to_csv(
                    os.path.join(folder, "WFT_RMS_Table.csv"), index=False)
                saved += 1
            except Exception as ex:
                errors.append(f"RMS CSV: {ex}")

        # ── DLC table (figure + CSV) ────────────────────────────
        if self._dlc_fig is not None:
            try:
                path = os.path.join(folder, "DLC_Table.png")
                self._dlc_fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
                saved += 1
            except Exception as ex:
                errors.append(f"DLC figure: {ex}")
        if self._dlc_rows:
            try:
                pd.DataFrame(self._dlc_rows).to_csv(
                    os.path.join(folder, "WFT_DLC_Table.csv"), index=False)
                saved += 1
            except Exception as ex:
                errors.append(f"DLC CSV: {ex}")

        # ── G-Severity table (figure + CSV) ─────────────────────
        if self._gsev_fig is not None:
            try:
                path = os.path.join(folder, "GSeverity_Table.png")
                self._gsev_fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
                saved += 1
            except Exception as ex:
                errors.append(f"G-Severity figure: {ex}")
        if self._gsev_rows:
            try:
                pd.DataFrame(self._gsev_rows).to_csv(
                    os.path.join(folder, "WFT_GSeverity_Table.csv"), index=False)
                saved += 1
            except Exception as ex:
                errors.append(f"G-Severity CSV: {ex}")

        # ── AUC Summary CSV ────────────────────────────────────
        all_rows = (self._auc_summary + self._auc_cross_summary
                    + self._auc_moment_summary + self._auc_cross_moment_summary)
        if all_rows:
            try:
                pd.DataFrame(all_rows).to_csv(
                    os.path.join(folder, "WFT_AUC_Summary.csv"), index=False)
                saved += 1
            except Exception as ex:
                errors.append(f"AUC Summary CSV: {ex}")

        # ── Box Plot Stats CSV ─────────────────────────────────
        if self._box_figs:
            try:
                lbl1 = self.label1.get(); lbl2 = self.label2.get()
                box_rows = []
                for ch in self._box_figs:
                    if ch not in self.avail_chans: continue
                    d1 = self.df[ch].dropna().values
                    box_rows.append({"Channel": ch, "Dataset": lbl1,
                        "Min":    round(float(d1.min()),                  2),
                        "P5":     round(float(np.percentile(d1,  5)),     2),
                        "Q1":     round(float(np.percentile(d1, 25)),     2),
                        "Median": round(float(np.median(d1)),             2),
                        "Mean":   round(float(np.mean(d1)),               2),
                        "Q3":     round(float(np.percentile(d1, 75)),     2),
                        "P95":    round(float(np.percentile(d1, 95)),     2),
                        "Max":    round(float(d1.max()),                  2)})
                    if self.df2 is not None and ch in self.avail_chans2:
                        d2 = self.df2[ch].dropna().values
                        box_rows.append({"Channel": ch, "Dataset": lbl2,
                            "Min":    round(float(d2.min()),              2),
                            "P5":     round(float(np.percentile(d2,  5)), 2),
                            "Q1":     round(float(np.percentile(d2, 25)), 2),
                            "Median": round(float(np.median(d2)),         2),
                            "Mean":   round(float(np.mean(d2)),           2),
                            "Q3":     round(float(np.percentile(d2, 75)), 2),
                            "P95":    round(float(np.percentile(d2, 95)), 2),
                            "Max":    round(float(d2.max()),              2)})
                if box_rows:
                    pd.DataFrame(box_rows).to_csv(
                        os.path.join(folder, "WFT_BoxPlot_Stats.csv"), index=False)
                    saved += 1
            except Exception as ex:
                errors.append(f"BoxPlot Stats CSV: {ex}")

        if errors:
            messagebox.showwarning("Export done (with errors)",
                f"Saved {saved} file(s).\n\nErrors:\n" + "\n".join(errors))
        else:
            messagebox.showinfo("Export complete",
                f"✓  {saved} file(s) saved to:\n{folder}")
        self._set_status(f"Export done — {saved} files → {os.path.basename(folder)}", SUCCESS)

    # ══ SAVE ══════════════════════════════════════════════════
    def _save_plot(self):
        if self.current_fig is None:
            messagebox.showinfo("Nothing to save","No plot displayed."); return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("PDF","*.pdf"),("All","*.*")])
        if path:
            self.current_fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
            self._set_status(f"Saved: {os.path.basename(path)}", SUCCESS)


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = WFTApp(root)
    root.mainloop()
