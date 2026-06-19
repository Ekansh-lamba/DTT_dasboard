"""Centralised dark theme: colour palette + global Qt stylesheet.

Palette mirrors the engineering colours already used by the backend plots
(`dtt/config.py` PLOT_COLORS) so the GUI and the generated figures feel like
one product.
"""

from __future__ import annotations

# ── Palette ──────────────────────────────────────────────────────────────────
BG          = "#0D1B2A"   # app background
SURFACE     = "#13243A"   # cards / panels
SURFACE_2   = "#1B3A5C"   # elevated panels, table headers
BORDER      = "#234263"
ENTRY_BG    = "#0A2540"

TEXT        = "#E6F1FF"
TEXT_MUTED  = "#90A4C4"
TEXT_FAINT  = "#5C7494"

ACCENT      = "#00B4D8"   # primary action / selection
ACCENT_DK   = "#0086A8"
ORANGE      = "#E8862A"   # Apollo accent
SUCCESS     = "#27AE60"
WARNING     = "#E0A92A"
DANGER      = "#C0392B"

# Per-wheel colours (match dtt/config.WHEEL_COLORS)
WHEEL_COLORS = {
    "FL": "#2E86DE",
    "FR": "#E74C3C",
    "RL": "#27AE60",
    "RR": "#8E44AD",
}

STATUS_COLORS = {
    "Complete":  SUCCESS,
    "Partial":   WARNING,
    "Validated": ACCENT,
    "Empty":     TEXT_FAINT,
    "Running":   ORANGE,
    "Error":     DANGER,
}


def stylesheet() -> str:
    return f"""
    * {{
        font-family: 'Segoe UI', 'Inter', sans-serif;
        font-size: 13px;
        color: {TEXT};
        outline: none;
    }}
    QMainWindow, QWidget#RootWidget {{ background: {BG}; }}

    /* ── Sidebar ─────────────────────────────────────────── */
    QFrame#Sidebar {{
        background: {SURFACE};
        border-right: 1px solid {BORDER};
    }}
    QLabel#BrandTitle {{ font-size: 16px; font-weight: 700; color: {TEXT}; }}
    QLabel#BrandSub   {{ font-size: 11px; color: {TEXT_MUTED}; }}

    QPushButton#NavButton {{
        text-align: left;
        padding: 10px 16px;
        border: none;
        border-radius: 8px;
        background: transparent;
        color: {TEXT_MUTED};
        font-size: 13px;
        font-weight: 500;
    }}
    QPushButton#NavButton:hover {{ background: {SURFACE_2}; color: {TEXT}; }}
    QPushButton#NavButton:checked {{
        background: {ACCENT_DK};
        color: #FFFFFF;
        font-weight: 600;
    }}

    /* ── Top bar ─────────────────────────────────────────── */
    QFrame#TopBar {{ background: {SURFACE}; border-bottom: 1px solid {BORDER}; }}
    QLabel#PageTitle {{ font-size: 18px; font-weight: 700; }}

    /* ── Cards ───────────────────────────────────────────── */
    QFrame#Card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 12px;
    }}
    QFrame#KpiCard {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 12px;
    }}
    QLabel#KpiLabel {{ font-size: 11px; font-weight: 600; color: {TEXT_MUTED}; }}
    QLabel#KpiValue {{ font-size: 30px; font-weight: 700; color: {TEXT}; }}
    QLabel#KpiSub   {{ font-size: 11px; color: {TEXT_FAINT}; }}
    QLabel#SectionTitle {{ font-size: 15px; font-weight: 700; color: {TEXT}; }}

    /* ── Inputs ──────────────────────────────────────────── */
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background: {ENTRY_BG};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 8px 10px;
        color: {TEXT};
        selection-background-color: {ACCENT};
    }}
    QLineEdit:focus, QComboBox:focus,
    QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {ACCENT}; }}
    QComboBox::drop-down {{ border: none; width: 22px; }}
    QComboBox QAbstractItemView {{
        background: {SURFACE_2};
        border: 1px solid {BORDER};
        selection-background-color: {ACCENT_DK};
        color: {TEXT};
        padding: 4px;
    }}
    QCheckBox {{ color: {TEXT}; spacing: 8px; }}
    QCheckBox::indicator {{
        width: 16px; height: 16px;
        border: 1px solid {BORDER};
        border-radius: 4px;
        background: {ENTRY_BG};
    }}
    QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

    /* ── Buttons ─────────────────────────────────────────── */
    QPushButton#Primary {{
        background: {ACCENT};
        color: #042027;
        border: none;
        border-radius: 8px;
        padding: 10px 18px;
        font-weight: 700;
    }}
    QPushButton#Primary:hover {{ background: #1FC6E8; }}
    QPushButton#Primary:disabled {{ background: {SURFACE_2}; color: {TEXT_FAINT}; }}

    QPushButton#Secondary {{
        background: transparent;
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 9px 16px;
        font-weight: 600;
    }}
    QPushButton#Secondary:hover {{ background: {SURFACE_2}; border-color: {ACCENT}; }}

    /* ── Tables ──────────────────────────────────────────── */
    QTableView, QTableWidget {{
        background: {SURFACE};
        alternate-background-color: {ENTRY_BG};
        gridline-color: {BORDER};
        border: 1px solid {BORDER};
        border-radius: 10px;
        selection-background-color: {ACCENT_DK};
        selection-color: #FFFFFF;
    }}
    QHeaderView::section {{
        background: {SURFACE_2};
        color: {TEXT};
        padding: 8px 10px;
        border: none;
        border-right: 1px solid {BORDER};
        font-weight: 600;
    }}
    QTableCornerButton::section {{ background: {SURFACE_2}; border: none; }}

    /* ── Misc ────────────────────────────────────────────── */
    QPlainTextEdit, QTextEdit {{
        background: #06121F;
        border: 1px solid {BORDER};
        border-radius: 10px;
        color: #C7E8F5;
        font-family: 'JetBrains Mono', 'Consolas', monospace;
        font-size: 12px;
    }}
    QProgressBar {{
        background: {ENTRY_BG};
        border: 1px solid {BORDER};
        border-radius: 6px;
        height: 10px;
        text-align: center;
        color: {TEXT};
    }}
    QProgressBar::chunk {{ background: {ACCENT}; border-radius: 5px; }}

    QScrollArea {{ background: transparent; border: none; }}
    QScrollBar:vertical {{ background: transparent; width: 12px; margin: 2px; }}
    QScrollBar::handle:vertical {{
        background: {SURFACE_2}; border-radius: 6px; min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {BORDER}; }}
    QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 2px; }}
    QScrollBar::handle:horizontal {{
        background: {SURFACE_2}; border-radius: 6px; min-width: 30px;
    }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}

    QComboBox#TabPill, QPushButton#TabPill {{
        background: {ENTRY_BG};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 6px 16px;
        color: {TEXT_MUTED};
        font-weight: 600;
    }}
    QPushButton#TabPill:checked {{ background: {ACCENT_DK}; color: #FFFFFF; border-color: {ACCENT}; }}

    QSplitter::handle {{ background: {BORDER}; }}
    QToolTip {{ background: {SURFACE_2}; color: {TEXT}; border: 1px solid {BORDER}; }}
    """
