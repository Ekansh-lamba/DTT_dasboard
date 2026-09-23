"""The study report, in the Apollo Tyres deck style.

Slides are built on ``templates/apollo_report_template.pptx`` -- the masters
and layouts of the DTT project charter deck, with its slides removed -- so the
report carries the same look as the charter: white 16:9 slides with the
Apollo logo, a black heading over an orange sub-heading and a grey rule,
purple-headed tables, the Apollo/Vredestein title slide and the purple-spiral
section dividers. Every slide is still built from this study's own channels
and figures; nothing from the charter's content is reused.

If the template is missing (a stripped install), a plain 16:9 deck with the
same layout geometry is produced instead, so a report is never lost to a
styling file.
"""

import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Pt, Emu, Inches

from dtt.config import (
    MANDATORY_CHANNELS,
    WHEEL_GROUPS,
    PERCENTILES,
    RunConfig,
)
from dtt.validation.validator import ValidationReport
from dtt.channel_names import display_name
from dtt.analysis.sessions_plot import session_info

logger = logging.getLogger(__name__)

TEMPLATE = Path(__file__).resolve().parent / "templates" / "apollo_report_template.pptx"

# Layout names in the template (the charter deck's own).
_LAYOUT_CONTENT = "Heading + text x2 columns (Apollo) A"
_LAYOUT_DIVIDER = "Divider opt 2 (Apollo) A"
_LAYOUT_COVER   = "1_Title Slide opt 1 (Apolo Tyres) A"

# Apollo palette, from the charter deck's theme.
_PURPLE    = RGBColor(0x95, 0x19, 0xEF)     # accent1: table headers, labels
_PURPLE_DK = RGBColor(0x5C, 0x2C, 0x91)     # accent4: panel outlines, footer
_ORANGE    = RGBColor(0xED, 0x7D, 0x31)     # sub-headings
_INK       = RGBColor(0x18, 0x18, 0x1E)     # dk1: body text
_GREY      = RGBColor(0x59, 0x59, 0x59)     # notes, slide numbers
_RULE      = RGBColor(0xBF, 0xBF, 0xBF)     # heading rule
_ROW_ALT   = RGBColor(0xF3, 0xEA, 0xFD)     # alternate table rows
_WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
_GREEN_RGB = RGBColor(0x1E, 0x84, 0x49)
_RED_RGB   = RGBColor(0xC0, 0x39, 0x2B)

# Slide geometry (16:9, 13.333 x 7.5 in). The Apollo logo sits bottom-left
# (0.1..2.63 in, from 6.37 in down), so content stops above it.
SLIDE_W = Emu(12192000)          # exact 16:9, as the template
SLIDE_H = Emu(6858000)
L = Inches(0.45)
T = Inches(1.45)
IMG_W = Inches(12.43)
IMG_H = Inches(4.85)

_FONT = "Calibri"


# ---------------------------------------------------------------- primitives

def _add_textbox(slide, text: str, left, top, width, height,
                 font_size: int = 14, bold: bool = False,
                 color: RGBColor = _INK, align=PP_ALIGN.LEFT,
                 italic: bool = False):
    txb = slide.shapes.add_textbox(left, top, width, height)
    tf = txb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(str(text).split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.size = Pt(font_size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.name = _FONT
        run.font.color.rgb = color
    return txb


def _layout(prs, name: str):
    for master in prs.slide_masters:
        for layout in master.slide_layouts:
            if layout.name == name:
                return layout
    # plain fallback: the emptiest layout there is
    return min((l for m in prs.slide_masters for l in m.slide_layouts),
               key=lambda l: len(l.placeholders))


def _drop_placeholders(slide, keep=()) -> None:
    """Remove placeholders we do not fill -- an empty one shows "Click to add
    text" to whoever opens the deck."""
    for ph in list(slide.placeholders):
        if ph.placeholder_format.idx not in keep:
            ph._element.getparent().remove(ph._element)


def _new_presentation():
    if TEMPLATE.exists():
        prs = Presentation(str(TEMPLATE))
    else:
        logger.warning("Report template missing (%s) -- using a plain deck", TEMPLATE)
        prs = Presentation()
    prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H
    return prs


class _Deck:
    """Adds slides in the charter's three styles and numbers them."""

    def __init__(self, prs, footer: str):
        self.prs = prs
        self.footer = footer

    def _number(self, slide) -> None:
        n = len(self.prs.slides)
        _add_textbox(slide, str(n), Inches(12.35), Inches(6.9), Inches(0.6),
                     Inches(0.3), font_size=10, color=_GREY, align=PP_ALIGN.RIGHT)

    def content(self, title: str, subtitle: str = ""):
        """Black heading, orange sub-heading, grey rule -- charter slides 3-6.
        The heading is the slide's first text shape, so it reads as its title."""
        slide = self.prs.slides.add_slide(_layout(self.prs, _LAYOUT_CONTENT))
        _drop_placeholders(slide)
        _add_textbox(slide, title, L, Inches(0.22), IMG_W, Inches(0.6),
                     font_size=28, bold=True, color=_INK)
        if subtitle:
            _add_textbox(slide, subtitle, L, Inches(0.8), IMG_W, Inches(0.4),
                         font_size=16, bold=True, color=_ORANGE)
        rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, L, Inches(1.25),
                                      IMG_W, Inches(0.03))
        rule.fill.solid()
        rule.fill.fore_color.rgb = _RULE
        rule.line.fill.background()
        if self.footer:
            _add_textbox(slide, self.footer, Inches(2.75), Inches(6.9),
                         Inches(8.0), Inches(0.35), font_size=11, italic=True,
                         color=_PURPLE_DK)
        self._number(slide)
        return slide

    def divider(self, title: str, subtitle: str = ""):
        """Purple-spiral section divider."""
        slide = self.prs.slides.add_slide(_layout(self.prs, _LAYOUT_DIVIDER))
        placed = {ph.placeholder_format.idx: ph for ph in slide.placeholders}
        if 0 in placed:
            placed[0].text = title
        else:
            _add_textbox(slide, title, L, Inches(2.1), Inches(6), Inches(0.7),
                         font_size=32, bold=True, color=_PURPLE)
        if 10 in placed and subtitle:
            placed[10].text = subtitle
        elif subtitle:
            _add_textbox(slide, subtitle, L, Inches(2.8), Inches(6), Inches(0.6),
                         font_size=18, color=_PURPLE_DK)
        _drop_placeholders(slide, keep={0} | ({10} if subtitle else set()))
        self._number(slide)
        return slide

    def cover(self, title: str, lines: List[str]):
        """Apollo / Vredestein title slide."""
        slide = self.prs.slides.add_slide(_layout(self.prs, _LAYOUT_COVER))
        placed = {ph.placeholder_format.idx: ph for ph in slide.placeholders}
        if 0 in placed:
            placed[0].text = title
        else:
            _add_textbox(slide, title, Inches(2.2), Inches(2.9), Inches(9), Inches(1.6),
                         font_size=36, bold=True, color=_PURPLE, align=PP_ALIGN.CENTER)
        _drop_placeholders(slide, keep={0})
        _add_textbox(slide, "\n".join(lines), Inches(2.2), Inches(4.55), Inches(9),
                     Inches(1.4), font_size=16, color=_PURPLE_DK, align=PP_ALIGN.CENTER)
        return slide


def _add_picture_fit(slide, img_path: Path, left, top, width, height):
    """Place the picture as large as fits the box, centred, aspect kept.

    Giving python-pptx both a width and a height stretches the image to the
    box: a tall box plot squeezed into a wide slot, text and all.
    """
    from PIL import Image
    with Image.open(img_path) as im:
        iw, ih = im.size
    scale = min(int(width) / iw, int(height) / ih)
    w, h = int(iw * scale), int(ih * scale)
    slide.shapes.add_picture(str(img_path), Emu(int(left) + (int(width) - w) // 2),
                             Emu(int(top) + (int(height) - h) // 2), Emu(w), Emu(h))


def _insert_image_or_message(slide, img_path: Path, left, top, width, height,
                             reason: str = "") -> bool:
    """Insert ``img_path`` if it exists, is non-empty, and decodes as an
    image; otherwise leave an explanatory textbox in its place instead of a
    bare title with nothing under it. Returns whether the image was inserted."""
    if img_path.exists() and img_path.stat().st_size > 0:
        try:
            _add_picture_fit(slide, img_path, left, top, width, height)
            return True
        except Exception as exc:
            logger.warning("Image could not be opened, skipping: %s (%s)", img_path.name, exc)
            msg = f"Not available: {img_path.name} could not be opened ({exc})."
    else:
        logger.warning("Image not found, skipping: %s", img_path.name)
        msg = reason or f"Not available: {img_path.name} was not generated for this study."
    _add_textbox(slide, msg, left, top, width, min(height, Emu(500000)),
                 font_size=12, italic=True, color=_GREY)
    return False


# --------------------------------------------------------------------- tables

def _style_cell(cell, text: str, *, header=False, label=False, alt=False,
                size=10, color: Optional[RGBColor] = None, bold=False,
                align=PP_ALIGN.LEFT) -> None:
    cell.text = str(text)
    cell.fill.solid()
    cell.fill.fore_color.rgb = (_PURPLE if (header or label)
                                else (_ROW_ALT if alt else _WHITE))
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.margin_top = cell.margin_bottom = Inches(0.03)
    for para in cell.text_frame.paragraphs:
        para.alignment = align
        for run in para.runs:
            run.font.name = _FONT
            run.font.size = Pt(size)
            run.font.bold = header or label or bold
            run.font.color.rgb = _WHITE if (header or label) else (color or _INK)


def _add_stats_table(slide, stats: Dict, channels: List[str], left, top, width, height) -> None:
    """``channels`` is this study's own mandatory-channel list
    (``config.run_channels.mandatory_channels``, generalized axle names like
    ``FR_Fx_2``) -- never the legacy ``FL_Fx``-style constant, which will
    never match a real study's stats keys."""
    channels = [ch for ch in channels if ch in stats]
    if not channels:
        _add_textbox(slide, "Not available: no statistics could be matched to "
                            "this study's channels.", left, top, width,
                     Emu(400000), font_size=13, italic=True, color=_GREY)
        return
    cols = ["Channel", "Count", "Mean", "Median", "Std", "Min", "Max"] + [f"P{p}" for p in PERCENTILES]
    row_h = min(int(height) // (len(channels) + 1), int(Inches(0.45)))
    table = slide.shapes.add_table(len(channels) + 1, len(cols), left, top, width,
                                   Emu(row_h * (len(channels) + 1))).table
    for ci, hdr in enumerate(cols):
        _style_cell(table.cell(0, ci), hdr, header=True, size=13,
                    align=PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER)

    for ri, ch in enumerate(channels, start=1):
        s = stats[ch]
        row_vals = [display_name(ch), s.get("count"), s.get("mean"), s.get("median"), s.get("std"),
                    s.get("min"), s.get("max")] + [s.get(f"P{p}") for p in PERCENTILES]
        for ci, val in enumerate(row_vals):
            if ci == 0:
                text = str(val)
            elif val is None:
                text = "N/A"
            elif ci == 1:
                text = f"{int(val):,}"
            else:
                text = f"{val:.2f}"
            _style_cell(table.cell(ri, ci), text, alt=ri % 2 == 0, bold=ci == 0, size=12,
                        align=PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER)


def _add_validation_table(slide, vr: ValidationReport, left, top, width, height) -> None:
    """Channel | Status, wrapped into side-by-side column pairs so a 40-channel
    recording still fits the slide instead of running off its bottom."""
    items = list(vr.channel_status.items())
    if not items:
        _add_textbox(slide, "No channel status was recorded.", left, top, width,
                     Emu(400000), font_size=13, italic=True, color=_GREY)
        return
    per_col = 14
    groups = -(-len(items) // per_col)
    rows = min(per_col, len(items)) + 1
    row_h = min(int(height) // rows, int(Inches(0.34)))
    table = slide.shapes.add_table(rows, 2 * groups, left, top, width,
                                   Emu(row_h * rows)).table
    for g in range(groups):
        _style_cell(table.cell(0, 2 * g), "Channel", header=True, size=12)
        _style_cell(table.cell(0, 2 * g + 1), "Status", header=True, size=12)
        for r in range(per_col):
            i = g * per_col + r
            if r + 1 >= rows:
                break
            if i >= len(items):
                _style_cell(table.cell(r + 1, 2 * g), "", alt=r % 2 == 1)
                _style_cell(table.cell(r + 1, 2 * g + 1), "", alt=r % 2 == 1)
                continue
            ch, st = items[i]
            ok = st == "PRESENT"
            _style_cell(table.cell(r + 1, 2 * g), display_name(ch), alt=r % 2 == 1, size=11)
            _style_cell(table.cell(r + 1, 2 * g + 1), f"{'✓' if ok else '✗'}  {st}",
                        alt=r % 2 == 1, size=11, bold=True,
                        color=_GREEN_RGB if ok else _RED_RGB)


def _add_key_value_table(slide, rows: List[tuple], left, top, width, height) -> None:
    """Purple label column, white value column -- charter slide 1."""
    row_h = min(int(height) // max(1, len(rows)), int(Inches(0.5)))
    table = slide.shapes.add_table(len(rows), 2, left, top, width,
                                   Emu(row_h * len(rows))).table
    table.first_row = False
    table.columns[0].width = Inches(2.3)
    table.columns[1].width = Emu(int(width) - int(Inches(2.3)))
    for r, (k, v) in enumerate(rows):
        _style_cell(table.cell(r, 0), k, label=True, size=14)
        _style_cell(table.cell(r, 1), v, size=13)


def _add_panel(slide, heading: str, lines: List[str], left, top, width, height) -> None:
    """Outlined panel with a purple heading -- charter slide 2's result boxes."""
    box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    box.fill.background()
    box.line.color.rgb = _PURPLE_DK
    box.line.width = Pt(1.25)
    _add_textbox(slide, heading, left + Inches(0.15), top + Inches(0.08),
                 width - Inches(0.3), Inches(0.4), font_size=18, bold=True,
                 color=_PURPLE_DK)
    _add_textbox(slide, "\n".join(f"•  {l}" for l in lines),
                 left + Inches(0.2), top + Inches(0.5), width - Inches(0.4),
                 height - Inches(0.6), font_size=15, color=_INK)


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


# ------------------------------------------------------------------- content

def _processing_lines(config: RunConfig, metadata: Dict) -> List[str]:
    """What was done to the data, in the order it was done -- so the report
    says how its numbers were produced, not only what they are."""
    lines = []
    if getattr(config, "famos_mode", True):
        lines.append("FAMOS recipe: forces/moments smo(0.1 s); Latacc FiltLP(4, 5 Hz) "
                     "→ smo(0.5 s); Longacc, speed smo(0.5 s); red(10)")
    else:
        lines.append(f"Butterworth low-pass {getattr(config, 'filter_cutoff', '?')} Hz, "
                     f"order {getattr(config, 'filter_order', '?')}")
    lines.append("Artefact blanking: samples beyond the P1–P99 band ± one band-width")
    stop = metadata.get("stop_removal") or {}
    if getattr(config, "remove_stops", False):
        lines.append(f"Stops removed: {stop.get('stops_removed', 0)} "
                     f"({stop.get('stop_seconds', 0):.0f} s) from every analysis")
    else:
        lines.append("Stops kept (standstill is in every statistic)")
    if getattr(config, "despike", False):
        lines.append("Physical despike: on")
    return lines


def _processing_summary(config: RunConfig, metadata: Dict) -> str:
    return "Processing:\n" + "\n".join(f"  {l}" for l in _processing_lines(config, metadata))


def _add_sessions_slide(deck: _Deck, sessions: Dict, figs: Path) -> None:
    from dtt.analysis.sessions_plot import SESSIONS_FIGURE, session_spans
    slide = deck.content(f"Recording Sessions  –  {sessions['sessions']} joined",
                         "Gaps between sessions removed · elapsed time continuous · "
                         "each session conditioned on its own before joining")
    rows = [f"Session {i}:  {Path(src).name or src or '—'}   "
            f"{s:.0f} – {e:.0f} s  ({(e - s) / 60:.1f} min)"
            for i, ((s, e), src) in enumerate(
                zip(session_spans(sessions), sessions["session_sources"]), start=1)]
    text_h = Emu(240000 * (len(rows) + 1))
    _add_textbox(slide, "\n".join(rows), L, T, IMG_W, text_h, font_size=13)
    _insert_image_or_message(slide, figs / SESSIONS_FIGURE, L, T + text_h,
                             IMG_W, IMG_H - text_h)


def build_report(
    config: RunConfig,
    validation_report: ValidationReport,
    stats: Dict,
    metadata: Dict,
    analysis_only: bool = False,
) -> Path:
    """``analysis_only=True`` marks the report as produced by
    ``workflow_mode="analysis"`` — no fresh preprocessing/validation pass, the
    channel-validation and file/duration/units figures describe the
    *existing* processed_data.csv this run read, not a run that just
    ingested and conditioned it. Made visible on the cover and overview so a
    reduced-metadata analysis-only report can't be mistaken for a full run.
    """
    figs = config.figures_dir

    # This study's own axle-generalized channel model, not the legacy fixed
    # FL/FR/RL/RR constants -- a recording with only FR/RR (or a 6-wheel
    # truck) must get exactly the slides its own data supports.
    rc = config.run_channels
    wheel_groups = rc.wheel_groups if rc is not None else WHEEL_GROUPS
    stat_channels = rc.mandatory_channels if rc is not None else MANDATORY_CHANNELS
    wheels = list(wheel_groups.keys())

    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M")
    sessions = session_info(metadata)
    deck = _Deck(_new_presentation(),
                 footer=f"WFT Road Load Report  ·  {config.vehicle_name}  ·  "
                        f"{config.study_name}")

    # ---- cover
    cover_lines = [f"{config.vehicle_name}  ·  Study {config.study_name}",
                   now.strftime("%d %B %Y")]
    if analysis_only:
        cover_lines.append("ANALYSIS-ONLY MODE")
    deck.cover("WFT Road Load Data – Tyre Duty Cycle Report", cover_lines)

    # ---- data & processing
    deck.divider("Data & Processing", "Recording · validation · statistics")

    slide = deck.content("Study Overview", f"{config.vehicle_name}  ·  {now_str}")
    overview = [
        ("Study", config.study_name),
        ("Source", str(metadata.get("file_name", "N/A"))),
        ("Duration", f"{metadata.get('duration_s', 0):,.1f} s  "
                     f"({metadata.get('duration_s', 0) / 60:.1f} min)"),
        ("Records", f"{metadata.get('rows', 0):,} rows × {metadata.get('columns', 0)} channels"),
        ("Sampling rate", f"{metadata.get('sampling_rate_hz', 0):.0f} Hz"),
        ("Units", "N → daN (÷10)" if metadata.get("n_to_dan_applied") else "daN (unchanged)"),
        ("Wheel positions", ", ".join(wheels) or "—"),
        ("Processing", "\n".join(_processing_lines(config, metadata))),
    ]
    if sessions:
        overview.append(("Sessions",
                         f"{sessions['sessions']} recordings joined end to end — seams at "
                         f"{', '.join(f'{t:.0f}' for t in sessions['seam_times_s'])} s"))
    if analysis_only:
        overview.append(("Mode", "ANALYSIS-ONLY: generated from an existing "
                                 "processed_data.csv, no fresh preprocessing or "
                                 "validation pass; see the pipeline log for "
                                 "provenance warnings"))
    _add_key_value_table(slide, overview, L, T, IMG_W, IMG_H)

    if sessions:
        _add_sessions_slide(deck, sessions, figs)

    slide = deck.content(
        "Channel Validation Summary",
        f"Missing: {len(validation_report.missing_channels)}   ·   "
        f"Optional present: {len(validation_report.optional_present)}   ·   "
        f"Total NaN rows: {validation_report.total_nan_rows:,}")
    _add_validation_table(slide, validation_report, L, T, IMG_W, IMG_H)

    slide = deck.content("Statistical Summary  –  P80 / P90 / P95",
                         "Per wheel force channel, daN")
    _add_stats_table(slide, stats, stat_channels, L, T, IMG_W, IMG_H)

    # ---- module 1
    deck.divider("Module 1", "Tyre load severity & distributions")

    slide = deck.content("Load Severity  –  Gx / Gy / Gxy / DLC",
                         "G-severity and dynamic load coefficient per wheel")
    _insert_image_or_message(slide, figs / "severity_table.png", L, T, IMG_W, IMG_H)

    half_h = (int(IMG_H) - int(Inches(0.1))) // 2
    for wheel in wheels:
        slide = deck.content(f"Histograms  –  {wheel}  (Distance & Percentage)",
                             "Share of distance travelled, then share of samples, per load bin")
        _insert_image_or_message(slide, figs / f"hist_distance_{wheel}.png", L, T, IMG_W, half_h)
        _insert_image_or_message(slide, figs / f"hist_percentage_{wheel}.png",
                                 L, T + half_h + Inches(0.1), IMG_W, half_h)

    for wheel in wheels:
        slide = deck.content(f"AUC Distribution  –  {wheel}  (Distance & Percentage)",
                             "Load density curves with P5 / P95")
        _insert_image_or_message(slide, figs / f"auc_distance_{wheel}.png", L, T, IMG_W, half_h)
        _insert_image_or_message(slide, figs / f"auc_percentage_{wheel}.png",
                                 L, T + half_h + Inches(0.1), IMG_W, half_h)

    for hm_name, hm_title in [
        ("heatmap_fx_fy_all",  "Heatmap  –  Fx vs Fy  (All Wheels)"),
        ("heatmap_fz_fy_all",  "Heatmap  –  Fz vs Fy  (All Wheels)"),
        ("heatmap_fz_fx_all",  "Heatmap  –  Fz vs Fx  (All Wheels)"),
    ]:
        slide = deck.content(hm_title, "Force combination occurrence")
        _insert_image_or_message(slide, figs / f"{hm_name}.png", L, T, IMG_W, IMG_H)

    slide = deck.content("Hexbin Density  –  Fy vs Fx  (All Wheels)",
                         "Sample density, log scale")
    for (lf, tf, gw, gh), wheel in zip(_grid_positions(len(wheels), L, T, IMG_W, IMG_H), wheels):
        _insert_image_or_message(slide, figs / f"heatmap_hexbin_{wheel}.png", lf, tf, gw, gh)

    for group, title in ((("Fx", "Fy", "Fz"), "Force Distribution"),
                         (("Mx", "My", "Mz"), "Moment Distribution")):
        present = [c for c in group if (figs / f"boxplot_{c}.png").exists()]
        if not present and group[0] == "Mx":
            continue                      # a forces-only study has no moment slide
        slide = deck.content(f"Boxplots  –  {title} ({' / '.join(group)})",
                             "Median, quartiles, P5 / mean / P95 per wheel")
        third_w = int(IMG_W) // 3
        for i, comp in enumerate(group):
            _insert_image_or_message(slide, figs / f"boxplot_{comp}.png",
                                     L + i * third_w, T, third_w, IMG_H)
    if (figs / "boxplot_parameters.png").exists():
        slide = deck.content("Boxplots  –  Vehicle & Other Parameters",
                             "Every other measured channel, each on its own unit")
        _insert_image_or_message(slide, figs / "boxplot_parameters.png", L, T, IMG_W, IMG_H)

    # ---- fatigue & frequency
    deck.divider("Fatigue & PSD", "Rainflow counting · Welch PSD")

    for wheel in wheels:
        slide = deck.content(f"Rainflow  –  {wheel}  (From-To Matrix + Range Distribution)",
                             "Siemens-style from-to matrix and cycle range distribution")
        _insert_image_or_message(slide, figs / f"rainflow_{wheel}.png", L, T, IMG_W, IMG_H)

    for wheel in wheels:
        slide = deck.content(f"Welch PSD  –  {wheel}", "Power spectral density per channel")
        _insert_image_or_message(slide, figs / f"psd_{wheel}.png", L, T, IMG_W, IMG_H)

    # ---- conclusions
    slide = deck.content("Engineering Conclusions", f"{config.vehicle_name}  ·  {now_str}")
    present_ch = [ch for ch, st in validation_report.channel_status.items() if st == "PRESENT"]
    missing_ch = validation_report.missing_channels
    results = [f"Channels validated: {len(present_ch)} / {len(stat_channels)} mandatory",
               f"Total records: {metadata.get('rows', 0):,}",
               f"Study duration: {metadata.get('duration_s', 0):,.1f} s"]
    if stats:
        fz_keys = [ch for ch in stats if "Fz" in ch and stats[ch].get("P95") is not None]
        if fz_keys:
            top = max(fz_keys, key=lambda ch: stats[ch]["P95"])
            results.append(f"Max Fz P95 (all wheels): {stats[top]['P95']:.1f} daN "
                           f"({display_name(top)})")
    notes = []
    if missing_ch:
        notes.append(f"Missing channels: {', '.join(missing_ch)}")
    if sessions:
        notes.append(f"Joined record: {sessions['sessions']} sessions, seams at "
                     f"{', '.join(f'{t:.0f}' for t in sessions['seam_times_s'])} s")
    if analysis_only:
        notes.append("ANALYSIS-ONLY MODE — no fresh preprocessing/validation pass; "
                     "generated from an existing processed_data.csv")
    notes.extend(_processing_lines(config, metadata))
    notes.append("Refer to individual slides for channel-level analysis")
    half_w = (int(IMG_W) - int(Inches(0.3))) // 2
    _add_panel(slide, "Key Results", results, L, T, half_w, IMG_H)
    _add_panel(slide, "Notes & Processing", notes, L + half_w + Inches(0.3), T,
               half_w, IMG_H)

    prs = deck.prs
    date_str = now.strftime("%Y%m%d")
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
