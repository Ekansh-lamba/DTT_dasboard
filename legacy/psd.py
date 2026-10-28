"""DISABLED 2026-10-28 -- the Welch PSD stage, formerly dtt/analysis/psd.py,
with ``welch_psd`` from dtt/spectral.py. Nothing in the app imports this; see
legacy/README.md.

Welch power spectral density, per wheel — a distinct view from the FAMOS-style
single-FFT dB spectrum in ``dtt/spectral.py``.

Segment-averaged, so it trades exact spectral lines for a smooth, statistically
stable noise floor: useful for "where does the road-load energy sit" in a way a
single windowed FFT (dominated by whatever one record's noise happened to look
like) is not.
"""

import logging
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dtt.config import (
    WHEEL_GROUPS,
    WHEEL_COLORS,
    CHAN_COLORS,
    PLOT_COLORS,
    FIGURE_DPI,
    RunConfig,
)
from dtt.channel_names import display_name


def welch_psd(x: np.ndarray, fs: float,
              nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density: ``(frequencies_hz, psd)``.

    ``nperseg`` defaults to a batch-pipeline-appropriate segment length
    derived from ``fs`` (``min(n, max(256, fs*4))``) — a longer segment than
    a live-GUI tool would use, trading update latency (irrelevant here, this
    runs once per study) for a smoother, more resolved PSD estimate. This is
    an explicit accuracy choice, not a default inherited from anywhere else.

    NaNs are dropped first. Returns a flat near-zero spectrum for input too
    short to window meaningfully.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8 or fs <= 0:
        return np.array([0.0]), np.array([0.0])
    from scipy.signal import welch
    if nperseg is None:
        nperseg = int(min(x.size, max(256, fs * 4)))
    nperseg = max(8, min(nperseg, x.size))
    freq, psd = welch(x, fs=fs, nperseg=nperseg)
    return freq, psd

logger = logging.getLogger(__name__)

BG       = PLOT_COLORS["bg"]
PANEL    = PLOT_COLORS["panel"]
TEXT_PRI = PLOT_COLORS["text_pri"]
TEXT_SEC = PLOT_COLORS["text_sec"]


def _style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT_SEC, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(PLOT_COLORS["accent"])
        sp.set_linewidth(0.5)
    ax.grid(True, color="#1B3A5C", linewidth=0.4, alpha=0.5, which="both")


def generate_psd(df: pd.DataFrame, config: RunConfig) -> None:
    """Welch PSD per wheel, one figure per wheel with one subplot per force
    channel present. Sample rate comes from ``config.sampling_rate`` — never a
    typed/GUI value."""
    out = config.figures_dir
    fs  = config.sampling_rate
    rc  = config.run_channels
    wheel_groups = rc.wheel_groups if rc is not None else WHEEL_GROUPS
    wheel_colors = rc.wheel_colors if rc is not None else WHEEL_COLORS
    chan_colors  = rc.chan_colors if rc is not None else CHAN_COLORS

    for wheel, channels in wheel_groups.items():
        present = [ch for ch in channels if ch in df.columns]
        if not present:
            continue
        col_pri = wheel_colors[wheel]["pri"]

        fig, axes = plt.subplots(1, len(present), figsize=(5.5 * len(present), 4.2), facecolor=BG)
        if len(present) == 1:
            axes = [axes]
        fig.suptitle(f"Welch PSD  –  {wheel}  (fs={fs:g} Hz)",
                     color=TEXT_PRI, fontsize=11, fontweight="bold")

        for ax, ch in zip(axes, present):
            col = chan_colors.get(ch, col_pri)
            _style_ax(ax)
            series = pd.to_numeric(df[ch], errors="coerce").dropna().values
            freq, psd = welch_psd(series, fs)
            ax.semilogy(freq, psd, color=col, linewidth=1.3)
            ax.set_title(display_name(ch), color=col, fontsize=9, fontweight="bold")
            ax.set_xlabel("Frequency (Hz)", color=TEXT_SEC, fontsize=8)
            ax.set_ylabel("PSD (daN²/Hz)", color=TEXT_SEC, fontsize=8)
            ax.set_xlim(0, fs / 2.0)

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fname = out / f"psd_{wheel}.png"
        fig.savefig(fname, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        logger.info("Saved PSD: %s", fname.name)
