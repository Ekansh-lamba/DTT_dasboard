"""
ui/components.py
================
Reusable Streamlit sidebar widgets and layout helpers for the DTT WFT dashboard.
"""

from __future__ import annotations

import streamlit as st

WHEELS = ["FL", "FR", "RL", "RR"]
FORCE_SUFFIXES = ["Fx", "Fy", "Fz"]


def render_sidebar(cfg: dict) -> dict:
    """
    Render the full sidebar configuration panel.

    Returns
    -------
    dict with keys:
        file_path, display_unit, sign_convention,
        selected_wheels, time_range, use_distance_weight,
        miner_m, heatmap_bins, hist_bins
    """
    st.sidebar.markdown(
        """
        <div style='text-align:center; padding: 8px 0 4px 0;'>
            <span style='font-size:22px; font-weight:900; color:#2E86DE;'>DTT</span>
            <span style='font-size:15px; color:#90B4CE; margin-left:6px;'>WFT Module 1</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    # ── File input ────────────────────────────────────────────────────────
    st.sidebar.markdown("### Data File")
    file_path = st.sidebar.text_input(
        "CSV file path",
        value=r"C:\Users\ekans\Downloads\DTT_WFT\For SRM\RLDA WFT PV data sample.csv",
        help="Full path to the pre-processed FAMOS WFT CSV file.",
    )

    st.sidebar.divider()

    # ── Display settings ──────────────────────────────────────────────────
    st.sidebar.markdown("### Display Settings")
    display_unit = st.sidebar.radio(
        "Force display unit",
        options=["daN", "N"],
        index=0,
        horizontal=True,
        help="daN = N ÷ 10 (matches Apollo reference charts). Conversion is deterministic.",
    )
    sign_convention = st.sidebar.selectbox(
        "Sign convention",
        options=["SAE", "ISO"],
        index=0,
        help="SAE J2047 (default). ISO 8855 flips Fy and Fz sign. Affects g-g and heatmaps.",
    )

    st.sidebar.divider()

    # ── Wheel selector ─────────────────────────────────────────────────────
    st.sidebar.markdown("### Wheel Positions")
    selected_wheels = st.sidebar.multiselect(
        "Select wheels",
        options=WHEELS,
        default=WHEELS,
        help="Choose which wheel positions to include in all analyses.",
    )
    if not selected_wheels:
        selected_wheels = WHEELS

    st.sidebar.divider()

    # ── Time / distance filter ────────────────────────────────────────────
    st.sidebar.markdown("### Range Filter")
    use_time_filter = st.sidebar.checkbox("Enable time/distance filter", value=False)
    time_range = None
    if use_time_filter:
        time_range = st.sidebar.slider(
            "Time range (s)",
            min_value=0,
            max_value=30_000,
            value=(0, 30_000),
            step=100,
        )

    # ── Histogram options ─────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.markdown("### Histogram Options")
    use_distance_weight = st.sidebar.checkbox(
        "Distance-weighted histograms",
        value=False,
        help="Weight each sample by distance increment (km). Y-axis = distance driven at that force level.",
    )
    hist_bins = st.sidebar.slider("Histogram bins", min_value=20, max_value=120, value=60, step=10)

    # ── Heatmap options ───────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.markdown("### Heatmap Options")
    heatmap_bins = st.sidebar.slider("Heatmap bins per axis", min_value=10, max_value=50, value=20, step=5)

    # ── Fatigue options ───────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.markdown("### Fatigue Options")
    miner_m = st.sidebar.slider(
        "Miner's slope (m)",
        min_value=2, max_value=10, value=5,
        help="Wöhler exponent for Miner's rule damage: D = Σ(n × range^m)",
    )

    return {
        "file_path":           file_path,
        "display_unit":        display_unit,
        "sign_convention":     sign_convention,
        "selected_wheels":     selected_wheels,
        "time_range":          time_range,
        "use_distance_weight": use_distance_weight,
        "hist_bins":           hist_bins,
        "heatmap_bins":        heatmap_bins,
        "miner_m":             miner_m,
    }


def render_load_status(meta: dict, warnings: list[str]) -> None:
    """
    Render a compact status card showing detected file metadata.
    """
    fs = meta.get("fs", 0)
    n_rows = meta.get("n_rows", 0)
    duration_s = meta.get("duration_s", 0)
    dist_km = meta.get("total_dist_km", 0)
    load_time = meta.get("load_time_s", 0)

    st.sidebar.success(
        f"**Loaded**  \n"
        f"Rows: **{n_rows:,}**  \n"
        f"fs: **{fs:.1f} Hz**  \n"
        f"Duration: **{duration_s/3600:.2f} h** ({duration_s:.0f} s)  \n"
        f"Distance: **{dist_km:.1f} km**  \n"
        f"Load time: **{load_time:.1f} s**"
    )

    if warnings:
        with st.sidebar.expander(f"{len(warnings)} warning(s)", expanded=False):
            for w in warnings:
                st.warning(w)


def wheel_color_badge(wheel: str) -> str:
    """Return an HTML colour badge for a wheel position."""
    colors = {
        "FL": "#2E86DE", "FR": "#E74C3C",
        "RL": "#27AE60", "RR": "#8E44AD",
    }
    c = colors.get(wheel, "#AAAAAA")
    return (
        f"<span style='background:{c};color:white;padding:2px 8px;"
        f"border-radius:4px;font-weight:bold;font-size:12px;'>{wheel}</span>"
    )


def render_tab_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
    st.divider()


def apply_time_filter(df, meta: dict, time_range: tuple | None):
    """
    Apply a time range filter to the DataFrame using the rebuilt time axis.
    Returns filtered DataFrame (or original if no filter applied).
    """
    if time_range is None or "t_rebuilt" not in df.columns:
        return df
    t_min, t_max = time_range
    mask = (df["t_rebuilt"] >= t_min) & (df["t_rebuilt"] <= t_max)
    filtered = df[mask].copy()
    return filtered
