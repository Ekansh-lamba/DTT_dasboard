"""Loading two recordings for comparison, and exporting the result.

Kept separate from :mod:`dtt.comparison` so the comparison maths stays free of
file and report concerns and can be unit-tested on plain DataFrames.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from dtt.comparison import ComparisonResult, align_channel, merge_channel_frame

logger = logging.getLogger(__name__)

DEFAULT_TARGET_FS = 100.0


def load_recording(folder: Path, target_fs: float = DEFAULT_TARGET_FS,
                   famos: bool = True, deglitch: bool = False,
                   to_dan: bool = True) -> Tuple[pd.DataFrame, float, dict]:
    """Read one imc raw folder for comparison.

    The FAMOS recipe is applied exactly as it is for a normal study, so the two
    sides are conditioned identically — comparing a smoothed run against an
    unsmoothed one would show differences that are pure processing.

    ``to_dan`` applies the pipeline's newton→decanewton conversion, so the
    numbers here match what every other screen reports. Each recording is judged
    on its own magnitudes (that is what the pipeline does per study); if the two
    end up on different footing, :func:`dtt.comparison.compare_frames` flags it
    rather than quietly reporting a tenfold "change".
    """
    from dtt.ingestion.imc_reader import read_folder
    from dtt.ingestion.loader import _apply_n_to_dan, _detect_force_channels

    folder = Path(folder)
    df, meta = read_folder(folder, target_fs=target_fs, famos=famos, deglitch=deglitch)
    fs = float(meta.get("output_fs_hz") or target_fs)
    if to_dan and not df.empty:
        force_cols = _detect_force_channels(df)
        df, applied = _apply_n_to_dan(df, force_cols)
        meta = dict(meta, n_to_dan_applied=applied)
    logger.info("Loaded %s: %d rows x %d cols @ %.3f Hz (N->daN: %s)",
                folder.name, len(df), len(df.columns), fs,
                meta.get("n_to_dan_applied"))
    return df, fs, meta


# ------------------------------------------------------------------ exporting

def export_csv(result: ComparisonResult, path: Path) -> Path:
    """Per-channel comparison table as CSV, with the summary as a header block."""
    path = Path(path)
    lines = ["# WFT recording comparison"]
    lines += [f"# {ln}" for ln in result.summary_lines()]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(lines) + "\n\n")
        table = result.table()
        if not table.empty:
            table.to_csv(fh, index=False)
    return path


def export_excel(result: ComparisonResult, path: Path,
                 merged: Optional[pd.DataFrame] = None) -> Path:
    """Multi-sheet workbook: Summary, Channel comparison, Added/Removed, Merged."""
    path = Path(path)
    summary = result.summary()
    sum_rows = [{"Field": k, "Value": (", ".join(v) if isinstance(v, list) else v)}
                for k, v in summary.items()]

    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        pd.DataFrame(sum_rows).to_excel(xl, sheet_name="Summary", index=False)
        table = result.table()
        if not table.empty:
            table.to_excel(xl, sheet_name="Channel comparison", index=False)
        pd.DataFrame({
            "Removed (previous only)": pd.Series(result.only_previous, dtype=object),
            "Added (current only)": pd.Series(result.only_current, dtype=object),
        }).to_excel(xl, sheet_name="Added-Removed", index=False)
        if merged is not None and not merged.empty:
            # Excel caps at 1,048,576 rows; thin the merged trace if needed.
            step = max(1, len(merged) // 100_000)
            merged.iloc[::step].to_excel(xl, sheet_name="Merged data", index=False)
    return path


def export_json(result: ComparisonResult, path: Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(result.summary(), indent=2), encoding="utf-8")
    return path


def export_pdf(result: ComparisonResult, path: Path,
               prev: Optional[pd.DataFrame] = None,
               curr: Optional[pd.DataFrame] = None,
               fs_prev: float = 0.0, fs_curr: float = 0.0,
               max_channels: int = 12) -> Path:
    """Comparison report: summary page, delta chart, then per-channel pages."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    path = Path(path)
    changed_first = sorted(result.matched, key=lambda d: -d.max_pct)

    with PdfPages(path) as pdf:
        # --- page 1: summary -------------------------------------------------
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(0.06, 0.94, "WFT Recording Comparison", fontsize=20, fontweight="bold")
        fig.text(0.06, 0.90, "Previous vs Current session", fontsize=11, color="#555555")
        fig.text(0.06, 0.84, "\n".join(result.summary_lines()), fontsize=9.5,
                 family="monospace", va="top")
        pdf.savefig(fig); plt.close(fig)

        # --- page 2: % change per channel -----------------------------------
        if result.matched:
            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            labels = [d.label for d in changed_first]
            vals = [d.pct.get("rms", np.nan) for d in changed_first]
            colors = ["#C0392B" if d.changed else "#7F8C8D" for d in changed_first]
            ax.barh(labels, vals, color=colors)
            ax.axvline(0, color="#333333", lw=0.8)
            ax.set_xlabel("Change in RMS, current vs previous (%)")
            ax.set_title("Per-channel load change  (red = beyond tolerance)")
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # --- per-channel pages ----------------------------------------------
        if prev is not None and curr is not None:
            for d in changed_first[:max_channels]:
                t, a, b = align_channel(prev, curr, d, fs_prev, fs_curr, max_points=3000)
                if t.size == 0:
                    continue
                fig, axes = plt.subplots(2, 1, figsize=(11.7, 8.3),
                                         gridspec_kw={"height_ratios": [2, 1]})
                axes[0].plot(t, a, lw=0.6, color="#2E86DE", label="previous")
                axes[0].plot(t, b, lw=0.6, color="#E74C3C", alpha=0.8, label="current")
                axes[0].set_title(f"{d.label} — overlay on elapsed time"
                                  f"   ({'CHANGED' if d.changed else 'within tolerance'}"
                                  f", worst {d.max_pct_metric} {d.max_pct:+.1f}%)")
                axes[0].set_xlabel("Elapsed time (s)")
                axes[0].set_ylabel(d.label)
                axes[0].legend(fontsize=8)
                axes[0].grid(alpha=0.25)

                fa = a[np.isfinite(a)]
                fb = b[np.isfinite(b)]
                if fa.size and fb.size:
                    lo = min(np.percentile(fa, 0.5), np.percentile(fb, 0.5))
                    hi = max(np.percentile(fa, 99.5), np.percentile(fb, 99.5))
                    bins = np.linspace(lo, hi, 60)
                    axes[1].hist(fa, bins=bins, alpha=0.55, color="#2E86DE",
                                 label="previous", density=True)
                    axes[1].hist(fb, bins=bins, alpha=0.55, color="#E74C3C",
                                 label="current", density=True)
                    axes[1].set_title("Distribution")
                    axes[1].set_xlabel(d.label)
                    axes[1].legend(fontsize=8)
                    axes[1].grid(alpha=0.25)
                fig.tight_layout()
                pdf.savefig(fig); plt.close(fig)
    return path


def export_all(result: ComparisonResult, out_dir: Path,
               prev: Optional[pd.DataFrame] = None,
               curr: Optional[pd.DataFrame] = None,
               fs_prev: float = 0.0, fs_curr: float = 0.0,
               stem: str = "comparison") -> dict:
    """Write CSV + Excel + PDF + JSON into ``out_dir``; return {kind: path}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = (merge_channel_frame(prev, curr, result, fs_prev, fs_curr)
              if prev is not None and curr is not None else None)
    written = {
        "csv": export_csv(result, out_dir / f"{stem}.csv"),
        "excel": export_excel(result, out_dir / f"{stem}.xlsx", merged),
        "json": export_json(result, out_dir / f"{stem}.json"),
        "pdf": export_pdf(result, out_dir / f"{stem}.pdf",
                          prev, curr, fs_prev, fs_curr),
    }
    if merged is not None and not merged.empty:
        merged_path = out_dir / f"{stem}_merged.csv"
        merged.to_csv(merged_path, index=False)
        written["merged_csv"] = merged_path
    return written
