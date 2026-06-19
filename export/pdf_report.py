"""
export/pdf_report.py
====================
PDF report generator for the DTT WFT dashboard.

Strategy:
  1. Render each Plotly figure to a PNG via kaleido (Plotly's static image engine).
  2. Assemble PNGs into a titled, multi-section PDF using reportlab.
  3. If kaleido is unavailable, fall back to weasyprint (HTML→PDF).

Sections (matching the 5 analysis outputs):
  1. Force Histograms
  2. Heatmaps
  3. Trip Statistics
  4. g-g & Force Severity
  5. Fatigue / Rainflow
  6. Box / Distance Distribution
"""

from __future__ import annotations

import io
import logging
import tempfile
from datetime import datetime
from typing import Any
import streamlit as st

logger = logging.getLogger(__name__)

# Try to import PDF backends
try:
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    _REPORTLAB = True
except ImportError:
    _REPORTLAB = False
    logger.warning("reportlab not installed — PDF export unavailable. Run: pip install reportlab")

try:
    import plotly.io as pio
    _KALEIDO = True
    # Test kaleido availability
    pio.kaleido.scope.mathjax = None
except Exception:
    _KALEIDO = False
    logger.warning("kaleido not available for Plotly static export. Run: pip install kaleido")


def generate_pdf_bytes(
    figures: list[tuple[str, Any, str]],   # (section_title, plotly_figure, cache_key)
    title: str = "DTT WFT Analysis Report",
    subtitle: str = "Apollo Tyres Ltd \u2014 WFT Tyre Duty Cycle",
    dpi: int = 150,
    fig_width_px: int = 1200,
    fig_height_px: int = 700,
) -> bytes:
    """
    Generate a PDF report entirely in-memory from Plotly figures.

    Parameters
    ----------
    figures       : List of (section_title, plotly_figure, cache_key) tuples
    title         : Report title
    subtitle      : Subtitle line
    dpi           : Rasterisation DPI for figures (via kaleido)
    fig_width_px  : Figure pixel width for kaleido
    fig_height_px : Figure pixel height for kaleido

    Returns
    -------
    bytes of the generated PDF document.
    """
    if not _REPORTLAB:
        raise RuntimeError(
            "reportlab is not installed. Install with: pip install reportlab"
        )

    # Rasterise figures to PNG bytes
    png_list: list[tuple[str, bytes]] = []
    for section_title, fig, cache_key in figures:
        png_bytes = fig_to_png_bytes(fig, cache_key, fig_width_px, fig_height_px, dpi)
        if png_bytes:
            png_list.append((section_title, png_bytes))

    pdf_bytes = _build_reportlab_pdf_bytes(png_list, title=title, subtitle=subtitle)
    logger.info("PDF report generated in memory (%.1f MB)", len(pdf_bytes) / 1e6)
    return pdf_bytes


@st.cache_data(show_spinner=False)
def fig_to_png_bytes(_fig: Any, cache_key: str, width_px: int, height_px: int, dpi: int) -> bytes:
    """Render a Plotly figure to PNG bytes using kaleido. Cached via cache_key to avoid double-renders."""
    if not _KALEIDO:
        raise RuntimeError("PDF export needs the 'kaleido' package. Install it with: pip install kaleido==0.2.1")
    try:
        import plotly.io as pio
        return pio.to_image(
            _fig,
            format="png",
            width=width_px,
            height=height_px,
            scale=max(1, dpi // 72),
        )
    except Exception as exc:
        raise RuntimeError(f"PDF export needs the 'kaleido' package. Install it with: pip install kaleido==0.2.1\nError: {exc}")


def _build_reportlab_pdf_bytes(
    png_list: list[tuple[str, bytes]],
    title: str,
    subtitle: str,
) -> bytes:
    """Assemble PNG images into a reportlab PDF document."""
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DTTTitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=colors.HexColor("#2E86DE"),
        spaceAfter=6,
    )
    section_style = ParagraphStyle(
        "DTTSection",
        parent=styles["Heading1"],
        fontSize=14,
        textColor=colors.HexColor("#2E86DE"),
        spaceBefore=12,
        spaceAfter=6,
        borderPad=4,
    )
    sub_style = ParagraphStyle(
        "DTTSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#4A7FA5"),
        spaceAfter=24,
    )
    caption_style = ParagraphStyle(
        "DTTCaption",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#90B4CE"),
        spaceAfter=12,
    )

    # Use landscape A4 for wide figures
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(A4),
        rightMargin=1.5*cm,
        leftMargin=1.5*cm,
        topMargin=1.5*cm,
        bottomMargin=1.5*cm,
    )

    story = []

    # ── Cover page ────────────────────────────────────────────────────────
    story.append(Spacer(1, 2*cm))
    story.append(Paragraph(title, title_style))
    story.append(Paragraph(subtitle, sub_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        caption_style,
    ))
    story.append(Spacer(1, 1*cm))

    # Table of contents (section titles only)
    toc_data = [["Section", "Description"]]
    for i, (sec_title, _) in enumerate(png_list, start=1):
        toc_data.append([str(i), sec_title])
    toc = Table(toc_data, colWidths=[1.5*cm, 20*cm])
    toc.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1B3A5C")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#E0EAF4")),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8FC")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCDDEE")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(toc)
    story.append(PageBreak())

    # ── Figure pages ──────────────────────────────────────────────────────
    # Available width in landscape A4
    PAGE_W = landscape(A4)[0] - 3*cm
    PAGE_H = landscape(A4)[1] - 4*cm

    for section_title, png_bytes in png_list:
        story.append(Paragraph(section_title, section_style))
        # Write PNG bytes to temp file (reportlab needs a file path or BytesIO)
        img_buf = io.BytesIO(png_bytes)
        img = Image(img_buf, width=PAGE_W, height=PAGE_H * 0.85)
        story.append(img)
        story.append(Spacer(1, 0.3*cm))
        story.append(PageBreak())

    doc.build(story)
    return buf.getvalue()
