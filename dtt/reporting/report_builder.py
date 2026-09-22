import datetime
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Pt, Emu

from dtt.config import (
    MANDATORY_CHANNELS,
    WHEEL_GROUPS,
    PERCENTILES,
    PPTX_SLIDE_WIDTH_EMU,
    PPTX_SLIDE_HEIGHT_EMU,
    RunConfig,
)
from dtt.validation.validator import ValidationReport

logger = logging.getLogger(__name__)

_BG_RGB     = RGBColor(0x0D, 0x1B, 0x2A)
_PANEL_RGB  = RGBColor(0x1B, 0x3A, 0x5C)
_ACCENT_RGB = RGBColor(0xE8, 0x86, 0x2A)
_TEXT_RGB   = RGBColor(0xFF, 0xFF, 0xFF)
_SEC_RGB    = RGBColor(0x90, 0xE0, 0xEF)
_GREEN_RGB  = RGBColor(0x27, 0xAE, 0x60)
_RED_RGB    = RGBColor(0xC0, 0x39, 0x2B)


def _px_to_emu(px: float, dpi: int = 96) -> int:
    return int(px / dpi * 914400)


def _set_slide_bg(slide, prs: Presentation) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = _BG_RGB


def _add_textbox(slide, text: str, left, top, width, height,
                 font_size: int = 18, bold: bool = False,
                 color: RGBColor = _TEXT_RGB, align=PP_ALIGN.LEFT) -> None:
    txb  = slide.shapes.add_textbox(left, top, width, height)
    tf   = txb.text_frame
    tf.word_wrap = True
    p    = tf.paragraphs[0]
    p.alignment = align
    run  = p.add_run()
    run.text = text
    run.font.size  = Pt(font_size)
    run.font.bold  = bold
    run.font.color.rgb = color


def _add_title(slide, title: str, subtitle: str = "") -> None:
    _add_textbox(slide, title,
                 Emu(457200), Emu(228600),
                 Emu(PPTX_SLIDE_WIDTH_EMU - 914400), Emu(685800),
                 font_size=26, bold=True, color=_ACCENT_RGB, align=PP_ALIGN.LEFT)
    if subtitle:
        _add_textbox(slide, subtitle,
                     Emu(457200), Emu(914400),
                     Emu(PPTX_SLIDE_WIDTH_EMU - 914400), Emu(457200),
                     font_size=14, color=_SEC_RGB, align=PP_ALIGN.LEFT)


def _insert_image_or_message(slide, img_path: Path, left, top, width, height,
                             reason: str = "") -> bool:
    """Insert ``img_path`` if it exists, is non-empty, and decodes as an
    image; otherwise leave an explanatory textbox in its place instead of a
    bare title with nothing under it. Returns whether the image was inserted."""
    if img_path.exists() and img_path.stat().st_size > 0:
        try:
            slide.shapes.add_picture(str(img_path), left, top, width, height)
            return True
        except Exception as exc:
            logger.warning("Image could not be opened, skipping: %s (%s)", img_path.name, exc)
            msg = f"Not available: {img_path.name} could not be opened ({exc})."
    else:
        logger.warning("Image not found, skipping: %s", img_path.name)
        msg = reason or f"Not available: {img_path.name} was not generated for this study."
    _add_textbox(slide, msg, left, top, width, min(height, Emu(500000)),
                font_size=12, color=_SEC_RGB)
    return False


def _add_stats_table(slide, stats: Dict, channels: List[str], left, top, width, height) -> None:
    """``channels`` is this study's own mandatory-channel list
    (``config.run_channels.mandatory_channels``, generalized axle names like
    ``FR_Fx_2``) -- never the legacy ``FL_Fx``-style constant, which will
    never match a real study's stats keys."""
    channels = [ch for ch in channels if ch in stats]
    if not channels:
        _add_textbox(slide, "Not available: no statistics could be matched to "
                            "this study's channels.", left, top, width,
                    Emu(400000), font_size=13, color=_SEC_RGB)
        return
    cols    = ["Channel", "Count", "Mean", "Median", "Std", "Min", "Max"] + [f"P{p}" for p in PERCENTILES]
    n_rows  = len(channels) + 1
    n_cols  = len(cols)
    table   = slide.shapes.add_table(n_rows, n_cols, left, top, width, height).table

    for ci, hdr in enumerate(cols):
        cell = table.cell(0, ci)
        cell.text = hdr
        cell.fill.solid()
        cell.fill.fore_color.rgb = _PANEL_RGB
        for para in cell.text_frame.paragraphs:
            for run in para.runs:
                run.font.bold        = True
                run.font.color.rgb   = _SEC_RGB
                run.font.size        = Pt(9)

    for ri, ch in enumerate(channels, start=1):
        s = stats[ch]
        row_vals = [ch, s.get("count"), s.get("mean"), s.get("median"), s.get("std"),
                   s.get("min"), s.get("max")] + [s.get(f"P{p}") for p in PERCENTILES]
        for ci, val in enumerate(row_vals):
            cell = table.cell(ri, ci)
            if ci == 0:
                cell.text = str(val)
            elif val is None:
                cell.text = "N/A"
            elif ci == 1:
                cell.text = str(int(val))
            else:
                cell.text = f"{val:.2f}"
            cell.fill.solid()
            cell.fill.fore_color.rgb = _BG_RGB if ri % 2 == 0 else RGBColor(0x0A, 0x25, 0x40)
            for para in cell.text_frame.paragraphs:
                for run in para.runs:
                    run.font.color.rgb = _TEXT_RGB
                    run.font.size      = Pt(8)


def _grid_positions(n: int, left, top, width, height, max_cols: int = 2):
    """``n`` evenly-sized (left, top, width, height) cells, up to
    ``max_cols`` per row. Generic replacement for a hardcoded 2x2 quad --
    a fixed 4-slot grid silently drops any wheel past the 4th on a 6-wheel
    truck, since zip() truncates to the shorter of the two sequences."""
    if n <= 0:
        return []
    cols = min(max_cols, n)
    rows = -(-n // cols)  # ceil division
    cw, ch = width // cols, height // rows
    return [(left + (i % cols) * cw, top + (i // cols) * ch, cw, ch) for i in range(n)]


def _audit_slides(prs: Presentation) -> List[str]:
    """Slides that ended up with only a title and nothing else -- should be
    impossible now that every image slot falls back to an explanatory
    textbox via :func:`_insert_image_or_message`, so this is a regression
    guard, not expected to ever fire."""
    warnings = []
    for i, slide in enumerate(prs.slides, start=1):
        n_pics   = sum(1 for sh in slide.shapes if sh.shape_type == 13)
        n_tables = sum(1 for sh in slide.shapes if sh.has_table)
        n_text   = sum(1 for sh in slide.shapes if sh.has_text_frame and sh.text_frame.text.strip())
        if n_pics == 0 and n_tables == 0 and n_text <= 1:
            warnings.append(f"slide {i} has only a title, no content and no explanation")
    return warnings


def _add_validation_table(slide, vr: ValidationReport, left, top, width, height) -> None:
    items   = list(vr.channel_status.items())
    n_rows  = len(items) + 1
    table   = slide.shapes.add_table(n_rows, 2, left, top, width, height).table

    for ci, hdr in enumerate(["Channel", "Status"]):
        cell = table.cell(0, ci)
        cell.text = hdr
        cell.fill.solid()
        cell.fill.fore_color.rgb = _PANEL_RGB
        for para in cell.text_frame.paragraphs:
            for run in para.runs:
                run.font.bold      = True
                run.font.color.rgb = _SEC_RGB
                run.font.size      = Pt(9)

    for ri, (ch, st) in enumerate(items, start=1):
        is_ok  = st == "PRESENT"
        symbol = "✓" if is_ok else "✗"
        vals   = [ch, f"{symbol}  {st}"]
        for ci, val in enumerate(vals):
            cell = table.cell(ri, ci)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = _BG_RGB if ri % 2 == 0 else RGBColor(0x0A, 0x25, 0x40)
            for para in cell.text_frame.paragraphs:
                for run in para.runs:
                    run.font.color.rgb = _GREEN_RGB if (ci == 1 and is_ok) else (_RED_RGB if ci == 1 else _TEXT_RGB)
                    run.font.size      = Pt(8)


def build_report(
    config: RunConfig,
    validation_report: ValidationReport,
    stats: Dict,
    metadata: Dict,
    analysis_only: bool = False,
) -> Path:
    """``analysis_only=True`` marks the report as produced by
    ``workflow_mode="analysis"`` — no fresh preprocessing/validation pass, the
    channel-validation and file/duration/units figures on slide 1 describe
    the *existing* processed_data.csv this run read, not a run that just
    ingested and conditioned it. Made visible on the title slide so a
    reduced-metadata analysis-only report can't be mistaken for a full run.
    """
    prs = Presentation()
    prs.slide_width  = Emu(PPTX_SLIDE_WIDTH_EMU)
    prs.slide_height = Emu(PPTX_SLIDE_HEIGHT_EMU)

    blank_layout = prs.slide_layouts[6]
    figs = config.figures_dir

    # This study's own axle-generalized channel model, not the legacy fixed
    # FL/FR/RL/RR constants -- a recording with only FR/RR (or a 6-wheel
    # truck) must get exactly the slides its own data supports, matching
    # every analysis module upstream (dtt.analysis.statistics/histograms/
    # heatmaps/boxplots/rainflow already key off config.run_channels the
    # same way; report_builder was the one module still on the old constants).
    rc = config.run_channels
    wheel_groups = rc.wheel_groups if rc is not None else WHEEL_GROUPS
    stat_channels = rc.mandatory_channels if rc is not None else MANDATORY_CHANNELS
    wheels = list(wheel_groups.keys())

    W = Emu(PPTX_SLIDE_WIDTH_EMU)
    H = Emu(PPTX_SLIDE_HEIGHT_EMU)
    L = Emu(457200)
    T = Emu(1143000)
    IMG_W = W - Emu(914400)
    IMG_H = H - Emu(1600000)

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    slide1 = prs.slides.add_slide(blank_layout)
    _set_slide_bg(slide1, prs)
    subtitle = f"{config.vehicle_name}  |  {now_str}"
    if analysis_only:
        subtitle += "  |  ANALYSIS-ONLY MODE"
    _add_title(slide1, "WFT Automation  –  Study Overview", subtitle)
    info = (
        f"File:          {metadata.get('file_name', 'N/A')}\n"
        f"Rows:          {metadata.get('rows', 0):,}\n"
        f"Columns:       {metadata.get('columns', 0)}\n"
        f"Duration:      {metadata.get('duration_s', 0):.1f} s\n"
        f"Sampling Rate: {metadata.get('sampling_rate_hz', 0):.0f} Hz\n"
        f"Unit Applied:  {'N → daN (÷10)' if metadata.get('n_to_dan_applied') else 'daN (unchanged)'}\n"
        f"Study:         {config.study_name}"
    )
    if analysis_only:
        info += (
            "\n\nANALYSIS-ONLY MODE: this report was generated from an "
            "existing processed_data.csv with no fresh preprocessing or "
            "validation pass. Channel-validation and file/unit figures "
            "above describe that existing file, not a run that just "
            "ingested and conditioned it. See the pipeline log for any "
            "processing-provenance warnings."
        )
    _add_textbox(slide1, info, L, T, IMG_W, IMG_H, font_size=14, color=_TEXT_RGB)

    slide2 = prs.slides.add_slide(blank_layout)
    _set_slide_bg(slide2, prs)
    _add_title(slide2, "Channel Validation Summary")
    _add_textbox(
        slide2,
        f"Missing: {len(validation_report.missing_channels)}  |  "
        f"Optional present: {len(validation_report.optional_present)}  |  "
        f"Total NaN rows: {validation_report.total_nan_rows:,}",
        L, Emu(800000), IMG_W, Emu(400000), font_size=11, color=_SEC_RGB,
    )
    tbl_h = min(Emu(3000000), H - Emu(1800000))
    _add_validation_table(slide2, validation_report, L, Emu(1300000), IMG_W, tbl_h)

    slide3 = prs.slides.add_slide(blank_layout)
    _set_slide_bg(slide3, prs)
    _add_title(slide3, "Statistical Summary  –  P80 / P90 / P95")
    _add_stats_table(slide3, stats, stat_channels, L, T, IMG_W, IMG_H)

    slide_sev = prs.slides.add_slide(blank_layout)
    _set_slide_bg(slide_sev, prs)
    _add_title(slide_sev, "Load Severity  –  Gx / Gy / Gxy / DLC")
    _insert_image_or_message(slide_sev, figs / "severity_table.png", L, T, IMG_W, IMG_H)

    for wheel in wheels:
        slide = prs.slides.add_slide(blank_layout)
        _set_slide_bg(slide, prs)
        _add_title(slide, f"Histograms  –  {wheel}  (Distance & Percentage)")
        img_dist = figs / f"hist_distance_{wheel}.png"
        img_pct  = figs / f"hist_percentage_{wheel}.png"
        half_h   = (H - Emu(1400000)) // 2
        _insert_image_or_message(slide, img_dist, L, T, IMG_W, half_h)
        _insert_image_or_message(slide, img_pct, L, T + half_h + Emu(100000), IMG_W, half_h)

    for hm_name, hm_title in [
        ("heatmap_fx_fy_all",  "Heatmap  –  Fx vs Fy  (All Wheels)"),
        ("heatmap_fz_fy_all",  "Heatmap  –  Fz vs Fy  (All Wheels)"),
        ("heatmap_fz_fx_all",  "Heatmap  –  Fz vs Fx  (All Wheels)"),
    ]:
        slide = prs.slides.add_slide(blank_layout)
        _set_slide_bg(slide, prs)
        _add_title(slide, hm_title)
        _insert_image_or_message(slide, figs / f"{hm_name}.png", L, T, IMG_W, IMG_H)

    slide_hex = prs.slides.add_slide(blank_layout)
    _set_slide_bg(slide_hex, prs)
    _add_title(slide_hex, "Hexbin Density  –  Fy vs Fx  (All Wheels)")
    for (lf, tf, gw, gh), wheel in zip(_grid_positions(len(wheels), L, T, IMG_W, IMG_H), wheels):
        _insert_image_or_message(slide_hex, figs / f"heatmap_hexbin_{wheel}.png", lf, tf, gw, gh)

    slide_box = prs.slides.add_slide(blank_layout)
    _set_slide_bg(slide_box, prs)
    _add_title(slide_box, "Boxplots  –  Force Distribution (Fx / Fy / Fz)")
    third_w = IMG_W // 3
    for i, ft in enumerate(["Fx", "Fy", "Fz"]):
        _insert_image_or_message(slide_box, figs / f"boxplot_{ft}.png",
                                 L + i * third_w, T, third_w, IMG_H)

    for wheel in wheels:
        slide = prs.slides.add_slide(blank_layout)
        _set_slide_bg(slide, prs)
        _add_title(slide, f"Rainflow  –  {wheel}  (From-To Matrix + Range Distribution)")
        _insert_image_or_message(slide, figs / f"rainflow_{wheel}.png", L, T, IMG_W, IMG_H)

    for wheel in wheels:
        slide = prs.slides.add_slide(blank_layout)
        _set_slide_bg(slide, prs)
        _add_title(slide, f"AUC Distribution  –  {wheel}  (Distance & Percentage)")
        half_h = (H - Emu(1400000)) // 2
        _insert_image_or_message(slide, figs / f"auc_distance_{wheel}.png", L, T, IMG_W, half_h)
        _insert_image_or_message(slide, figs / f"auc_percentage_{wheel}.png",
                                 L, T + half_h + Emu(100000), IMG_W, half_h)

    for wheel in wheels:
        slide = prs.slides.add_slide(blank_layout)
        _set_slide_bg(slide, prs)
        _add_title(slide, f"Welch PSD  –  {wheel}")
        _insert_image_or_message(slide, figs / f"psd_{wheel}.png", L, T, IMG_W, IMG_H)

    slide_end = prs.slides.add_slide(blank_layout)
    _set_slide_bg(slide_end, prs)
    _add_title(slide_end, "Engineering Conclusions")
    present_ch  = [ch for ch, st in validation_report.channel_status.items() if st == "PRESENT"]
    missing_ch  = validation_report.missing_channels

    conclusions = f"Study: {config.vehicle_name}  |  {now_str}\n\n"
    if analysis_only:
        conclusions += (
            "*** ANALYSIS-ONLY MODE — no fresh preprocessing/validation pass. "
            "Generated from an existing processed_data.csv. ***\n\n"
        )
    conclusions += f"Channels validated:   {len(present_ch)} / {len(stat_channels)} mandatory\n"
    if missing_ch:
        conclusions += f"Missing channels:     {', '.join(missing_ch)}\n"
    conclusions += f"Total records:        {metadata.get('rows', 0):,}\n"
    conclusions += f"Study duration:       {metadata.get('duration_s', 0):.1f} s\n\n"
    if stats:
        fz_keys = [ch for ch in stats if "Fz" in ch]
        if fz_keys:
            max_fz_p95 = max(stats[ch]["P95"] for ch in fz_keys)
            conclusions += f"Max Fz P95 (all wheels): {max_fz_p95:.1f} daN\n"
    conclusions += "\nRefer to individual slides for detailed channel-level analysis."
    _add_textbox(slide_end, conclusions, L, T, IMG_W, IMG_H, font_size=13, color=_TEXT_RGB)

    date_str  = datetime.datetime.now().strftime("%Y%m%d")
    out_fname = (config.run_output_dir / f"WFT_Report_{config.vehicle_name}_{date_str}.pptx").resolve()
    prs.save(str(out_fname))

    blank_slides = _audit_slides(prs)
    if blank_slides:
        logger.warning("Report has %d title-only slide(s) with no content or "
                       "explanation: %s", len(blank_slides), "; ".join(blank_slides))
    logger.info("PowerPoint report saved: %s  (%d slides, %d wheel(s): %s, "
               "stats for %d/%d channels)",
               out_fname, len(prs.slides), len(wheels), ", ".join(wheels),
               len([ch for ch in stat_channels if ch in stats]), len(stat_channels))
    return out_fname
