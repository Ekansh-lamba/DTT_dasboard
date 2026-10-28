"""Figures for the result slides in the charter-deck format.

The report's "Force Distribution" pages and the "Combination of Fx & Fy"
heatmap follow the DTT project charter deck (slides 8-14): a PV format (two
pages, FL/FR/RL/RR, dataset A against dataset B), a CV format (one slide per
pair of axles, a candidate against a reference) and one combination heatmap
slide. This module decides what those slides show -- :func:`resolve_spec` --
and draws their plots; ``report_builder`` lays out every title, label and
summary box around them as native PowerPoint shapes.

Dataset A is the study being reported, on exactly the frame every other
analysis stage saw (stop-cut when stop removal is on). Dataset B is another
finished study, read from its ``processed_data.csv`` and given the same stop
removal, so both sides are cut the same way. Every curve is a plain
sample-count KDE and every stats strip is unweighted, so the strip's numbers
are the Statistics stage's numbers for the same channel.

Figure file names are fixed (:func:`figure_path`), so the slide builder can
place them, or say what is missing, without being handed anything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dtt.config import REPORT_RANGES, RunConfig

logger = logging.getLogger(__name__)

FIG_SUBDIR = "report"
DPI = 250

PV_WHEELS = ("FL", "FR", "RL", "RR")
PV_COLOR_A = "#27AE60"          # dataset A: solid green
PV_COLOR_B = "#E74C3C"          # dataset B: dashed red
CV_CANDIDATE_COLORS = ("#2E75D0", "#E74C3C")    # by row: blue, red, blue, ...

# Image sizes, inches -- each figure is drawn at the size it is placed at, so
# nothing is stretched. A distribution image is the plot plus its stats strip.
PV_SLOT_IN = (3.25, 2.55)
CV_SLOT_IN = (3.75, 2.75)
COMBO_IN = (5.9, 4.8)
COMBO_CBAR_IN = (0.75, 4.3)
_STRIP_IN = 0.32

_ROLE_TITLES = {"steer": "Steer Axle", "drive": "Drive Axle", "load": "Tag Axle",
                "trailer": "Trailer Axle"}


@dataclass
class CvRow:
    axle: int
    title: str              # "Drive Axle"
    wheel: str              # the wheel label plotted for this axle in A


@dataclass
class ReportSpec:
    fmt: str                                  # "pv" | "cv"
    label_a: str
    label_b: Optional[str]                    # None: single-study report
    route: str
    subtitle: str
    ranges: Dict[str, Tuple[float, float]]
    drive_a: str
    drive_b: str
    cv_rows: List[CvRow] = field(default_factory=list)
    combo_slots: List[str] = field(default_factory=list)

    @property
    def compared(self) -> bool:
        return self.label_b is not None


def figure_path(config: RunConfig, name: str) -> Path:
    return Path(config.figures_dir) / FIG_SUBDIR / f"{name}.png"


def drive_wheels(axle: str) -> Tuple[str, str]:
    """``("RL", "RR")`` for a rear-driven car, ``("FL", "FR")`` front."""
    return ("RL", "RR") if axle == "rear" else ("FL", "FR")


def _preset(config: RunConfig):
    from dtt.vehicles import PRESETS, get_preset, match_preset
    preset = get_preset(config.vehicle_type)
    if preset is None and config.run_channels is not None:
        preset = match_preset(config.run_channels.channel_set)
    return preset or PRESETS[0]


def _positions(rc) -> Dict[str, object]:
    from dtt.analysis.study_compare import _positions as positions
    return positions(rc) if rc is not None else {}


def _cv_rows(rc, preset) -> List[CvRow]:
    """One row per axle the study recorded, front to back. The row plots one
    wheel of the axle -- the left (outer) one when there is a choice -- and
    takes its title from the vehicle preset's axle role."""
    roles = {ax.id: ax.role for ax in preset.axles}
    by_axle: Dict[int, List[tuple]] = {}
    for label, pos in _positions(rc).items():
        by_axle.setdefault(pos.axle, []).append(
            ((pos.side, 0 if pos.sub in (None, "", "O") else 1), label))
    rows = []
    for axle in sorted(by_axle):
        wheel = sorted(by_axle[axle])[0][1]
        title = _ROLE_TITLES.get(roles.get(axle, ""), f"Axle {axle}")
        rows.append(CvRow(axle=axle, title=title, wheel=wheel))
    return rows


def resolve_spec(config: RunConfig) -> ReportSpec:
    """What the result slides show, from the config and this study's own
    channels -- the format, the dataset names, the axle rows."""
    preset = _preset(config)
    fmt = config.report_format
    if fmt == "auto":
        fmt = "pv" if preset.vclass == "PV" else "cv"
    ranges = dict(REPORT_RANGES[fmt])
    ranges.update({k: tuple(v) for k, v in (config.report_ranges or {}).items()})
    compare = config.report_compare_study
    label_b = None
    if compare:
        label_b = config.report_label_b or Path(compare).name
    rc = config.run_channels
    labels = rc.labels if rc is not None else list(PV_WHEELS)
    return ReportSpec(
        fmt=fmt,
        label_a=config.report_label_a or config.vehicle_name,
        label_b=label_b,
        route=config.report_route,
        subtitle=config.report_subtitle,
        ranges=ranges,
        drive_a=config.report_drive_axle_a,
        drive_b=config.report_drive_axle_b,
        cv_rows=_cv_rows(rc, preset) if fmt == "cv" else [],
        combo_slots=list(PV_WHEELS) if fmt == "pv" else labels,
    )


# ------------------------------------------------------------------ data

def _values(df: Optional[pd.DataFrame], col: Optional[str]) -> Optional[np.ndarray]:
    """Finite samples of one column -- the same set the Statistics stage
    reads (``to_numeric().dropna()``)."""
    if df is None or not col or col not in df.columns:
        return None
    v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return v if v.size >= 2 else None


def _index(rc) -> Dict[tuple, str]:
    from dtt.analysis.study_compare import _channel_index
    return _channel_index(rc) if rc is not None else {}


def load_compare_frame(config: RunConfig):
    """Dataset B's processed frame and channel model, cut for stops exactly
    as this study's analysis frame was, or ``(None, None)``."""
    from dtt.analysis.study_compare import load_study
    from dtt.provenance import compare_provenance, load_provenance

    study = Path(config.report_compare_study)
    df_b, rc_b, _ = load_study(study)
    for w in compare_provenance(load_provenance(config.run_output_dir), load_provenance(study),
                                config.study_name, study.name):
        logger.warning("Result slides: %s", w)
    if getattr(config, "remove_stops", False):
        from dtt.pipeline import _find_speed_column
        from dtt.preprocessing import remove_stops_frame
        df_b, meta = remove_stops_frame(
            df_b, config.sampling_rate, speed_column=_find_speed_column(df_b),
            min_stop_s=config.stop_min_s, moving_kph=config.stop_speed_kph,
            seam_search_s=config.stop_seam_search_s, seam_blend_s=config.stop_seam_blend_s)
        logger.info("Result slides: dataset B stop removal -- %d stop(s), %.0f s",
                    meta.get("stops_removed", 0), meta.get("stop_seconds", 0.0))
    return df_b, rc_b


# --------------------------------------------------------------- drawing

def _style_axes(ax, xlabel: str, xlim, fs: float) -> None:
    ax.set_facecolor("#F2F2F2")
    ax.grid(color="#D9D9D9", linewidth=0.5)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_edgecolor("#BFBFBF")
        sp.set_linewidth(0.6)
    ax.tick_params(labelsize=fs, length=2, pad=1.5)
    ax.yaxis.get_offset_text().set_fontsize(fs)
    ax.set_xlim(*xlim)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(xlabel, fontsize=fs + 0.5, fontweight="bold", labelpad=1.5)
    ax.set_ylabel("Normalised density", fontsize=fs, fontweight="bold", labelpad=2)


def _strip_fontsize(series, width_in: float, cap: float) -> float:
    """Largest monospace size (pt) at which the widest strip line fits."""
    from dtt.analysis.plot_style import stats_strip_lines, stats_strip_rows
    header, rows = stats_strip_rows(series, decimals=0)
    chars = max(len(l) for l in stats_strip_lines(header, rows))
    return min(cap, width_in * 72 * 0.97 / (0.602 * chars + 0.9))


def _distribution_figure(path: Path, size_in, xlabel: str, xlim,
                         draw, strip_series, fs: float) -> None:
    from dtt.analysis.plot_style import draw_stats_strip

    w, h = size_in
    left, right, bottom_lab = 0.46, 0.14, 0.30     # right: room for the last x tick
    fig = plt.figure(figsize=size_in, facecolor="white")
    ax = fig.add_axes([left / w, (_STRIP_IN + bottom_lab) / h,
                       (w - left - right) / w, (h - _STRIP_IN - bottom_lab - 0.06) / h])
    draw(ax)
    _style_axes(ax, xlabel, xlim, fs)
    strip = fig.add_axes([0.01, 0.01, 0.98, (_STRIP_IN - 0.03) / h])
    draw_stats_strip(strip, strip_series, layout="text", decimals=0,
                     fontsize=_strip_fontsize(strip_series, w * 0.98, fs - 0.6))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)


def _pv_plot(path: Path, xlabel: str, xlim, a, b, spec: ReportSpec) -> None:
    from dtt.analysis.plot_style import draw_comparison_kde

    series = [(spec.label_a, a)] + ([(spec.label_b, b)] if b is not None else [])
    _distribution_figure(
        path, PV_SLOT_IN, xlabel, xlim,
        lambda ax: draw_comparison_kde(ax, a, b, spec.label_a, spec.label_b or "",
                                       PV_COLOR_A, PV_COLOR_B, xlim=xlim,
                                       p5_in_legend=True, fill_alpha=0.22,
                                       line_width=1.1, legend_fontsize=3.6),
        series, fs=5.0)


def _cv_plot(path: Path, xlabel: str, xlim, cand, ref, spec: ReportSpec,
             color: str) -> None:
    from dtt.analysis.plot_style import draw_reference_kde

    series = [(spec.label_a, cand)] + ([(spec.label_b, ref)] if ref is not None else [])
    _distribution_figure(
        path, CV_SLOT_IN, xlabel, xlim,
        lambda ax: draw_reference_kde(ax, cand, ref, spec.label_a, spec.label_b or "",
                                      color, xlim=xlim, line_width=1.2,
                                      legend_fontsize=3.8),
        series, fs=5.5)


def _xlabel(col_a: str, col_b: Optional[str]) -> str:
    from dtt.channel_names import channel_unit, display_name
    name = display_name(col_a)
    if col_b is not None and display_name(col_b) != name:
        name = f"{name} vs {display_name(col_b)}"
    return f"{name} ({channel_unit(col_a)})"


# ---------------------------------------------------------------- stages

def _pv_figures(config, spec, df_a, ia, df_b, ib) -> int:
    n = 0
    for comp in ("Fx", "Fy", "Fz"):
        for wheel in PV_WHEELS:
            col_a, col_b = ia.get((wheel, comp)), ib.get((wheel, comp))
            a = _values(df_a, col_a)
            if a is None:
                continue
            b = _values(df_b, col_b)
            _pv_plot(figure_path(config, f"pv_{comp}_{wheel}"),
                     _xlabel(col_a, col_b if b is not None else None),
                     spec.ranges[comp], a, b, spec)
            n += 1
    # Drive tyres: each dataset's own driven axle, side by side
    for comp in ("Fx", "My"):
        for side, wa, wb in zip(("Left", "Right"), drive_wheels(spec.drive_a),
                                drive_wheels(spec.drive_b)):
            col_a, col_b = ia.get((wa, comp)), ib.get((wb, comp))
            a = _values(df_a, col_a)
            if a is None:
                continue
            b = _values(df_b, col_b)
            _pv_plot(figure_path(config, f"pv_drive_{comp}_{side}"),
                     _xlabel(col_a, col_b if b is not None else None),
                     spec.ranges[comp], a, b, spec)
            n += 1
    return n


def _cv_figures(config, spec, df_a, ia, df_b, rc_b, ib) -> int:
    pos_b = _positions(rc_b)
    n = 0
    for i, row in enumerate(spec.cv_rows):
        wheel_b = row.wheel if row.wheel in pos_b else next(
            (l for l, p in sorted(pos_b.items(), key=lambda kv: (kv[1].side, kv[0]))
             if p.axle == row.axle), None)
        color = CV_CANDIDATE_COLORS[i % 2]
        for comp in ("Fz", "Fx", "Fy"):
            col_a, col_b = ia.get((row.wheel, comp)), ib.get((wheel_b, comp))
            a = _values(df_a, col_a)
            if a is None:
                continue
            b = _values(df_b, col_b)
            _cv_plot(figure_path(config, f"cv_{comp}_axle{row.axle}"),
                     _xlabel(col_a, col_b if b is not None else None),
                     spec.ranges[comp], a, b, spec, color)
            n += 1
    return n


def _combo_figures(config, spec, df_a, rc_a, df_b, rc_b) -> int:
    from dtt.analysis.heatmaps import (bins_from_ranges, fx_fy_combination_colorbar,
                                       fx_fy_combination_figure,
                                       fx_fy_combination_panel, resolve_bin_ranges)
    # The same bins as every other force heatmap of this study, in daN.
    bins = bins_from_ranges(resolve_bin_ranges(config)[0])
    fy_edges, fx_edges = bins["Fy"], bins["Fx"]
    sides = [("a", df_a, rc_a, True)]
    if df_b is not None:
        sides.append(("b", df_b, rc_b, False))
    panels = {key: {w: fx_fy_combination_panel(df, rc, w, fy_edges, fx_edges)
                    for w in spec.combo_slots}
              for key, df, rc, _ in sides}
    peaks = [float(p[0].max()) for side in panels.values() for p in side.values()
             if p is not None]
    if not peaks:
        return 0
    vmax = max(peaks) or 1.0
    for key, _, _, tags in sides:
        path = figure_path(config, f"combo_{key}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fx_fy_combination_figure(panels[key], spec.combo_slots, path, fy_edges, fx_edges, vmax,
                                 COMBO_IN, tags=tags, dpi=DPI)
    fx_fy_combination_colorbar(figure_path(config, "combo_cbar"), vmax, COMBO_CBAR_IN, dpi=DPI)
    return len(sides)


def generate_result_figures(df_a: pd.DataFrame, config: RunConfig) -> ReportSpec:
    """Draw every result-slide plot for this study into ``figures/report/``.

    ``df_a`` is the frame the other analysis stages used. Returns the spec the
    slides are laid out from. A dataset B that cannot be read is logged and
    the slides fall back to dataset A alone -- a broken comparison path must
    not cost the study its report.
    """
    spec = resolve_spec(config)
    out = Path(config.figures_dir) / FIG_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):          # never place a stale plot from an earlier run
        old.unlink()

    df_b = rc_b = None
    if spec.compared:
        try:
            df_b, rc_b = load_compare_frame(config)
        except Exception as exc:                                  # noqa: BLE001
            logger.error("Result slides: comparison study %s could not be read (%s) -- "
                         "drawing %s alone", config.report_compare_study, exc, spec.label_a)
    rc_a = config.run_channels
    ia, ib = _index(rc_a), _index(rc_b)

    n = (_pv_figures(config, spec, df_a, ia, df_b, ib) if spec.fmt == "pv"
         else _cv_figures(config, spec, df_a, ia, df_b, rc_b, ib))
    n_combo = _combo_figures(config, spec, df_a, rc_a, df_b, rc_b)
    logger.info("Result slides: %s format, %s%s -- %d distribution plot(s), %d heatmap(s)",
                spec.fmt.upper(), spec.label_a,
                f" vs {spec.label_b}" if df_b is not None else " (single study)",
                n, n_combo)
    if spec.compared and df_b is None:
        spec.label_b = None
    return spec
