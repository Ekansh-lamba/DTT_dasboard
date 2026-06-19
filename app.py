"""
app.py — DTT Module 1: WFT Analysis Dashboard
==============================================
Streamlit entry point.

Usage:
    streamlit run app.py

Architecture:
    Layer A (Engine):   engine/ — pure Python analysis, no UI
    Layer B (UI):       ui/ + this file — Streamlit + Plotly
    Layer C (Export):   export/ — PDF / CSV / HTML from engine results

The data is loaded ONCE via @st.cache_data and reused across all tabs.
All engine functions are called on the cached DataFrame; re-renders are
triggered only when sidebar filter values change.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import streamlit as st
import yaml

# ── Make engine & ui importable when running from project root ─────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Configure logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dtt.app")

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DTT — WFT Dashboard | Apollo Tyres",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "DTT Module 1 — WFT Tyre Duty Cycle Dashboard | Apollo Tyres Ltd",
    },
)

# ── Custom CSS for premium dark aesthetic ─────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp { background: #0D1B2A; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111E2E 0%, #0D1B2A 100%);
    border-right: 1px solid #1B3A5C;
}

.stTabs [data-baseweb="tab-list"] {
    background: #111E2E;
    border-radius: 8px;
    padding: 4px;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    color: #90B4CE;
    font-weight: 600;
    border-radius: 6px;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background: #1B3A5C !important;
    color: #E0EAF4 !important;
}

div[data-testid="metric-container"] {
    background: #111E2E;
    border: 1px solid #1B3A5C;
    border-radius: 8px;
    padding: 12px;
}

.apollo-header {
    background: linear-gradient(90deg, #0D1B2A 0%, #1B3A5C 50%, #0D1B2A 100%);
    border-bottom: 2px solid #2E86DE;
    padding: 12px 24px;
    margin-bottom: 16px;
}

.warning-box {
    background: rgba(232,134,42,0.12);
    border-left: 3px solid #E8862A;
    border-radius: 0 6px 6px 0;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 13px;
    color: #E8862A;
}
</style>
""", unsafe_allow_html=True)


# ─── Load config ───────────────────────────────────────────────────────────
@st.cache_resource
def load_config() -> dict:
    cfg_path = Path(__file__).parent / "config" / "defaults.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─── Cached data loader ────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading CSV — this may take 30–60 s for a 720 MB file …")
def cached_load(file_path: str, _cfg_hash: int) -> tuple:
    """
    Load the WFT CSV and build the channel map.
    Cached by file path; re-loads if path changes.

    Returns (df, meta, chan_map, warnings)
    """
    from engine.io_loader import load_csv_chunked, validate_data
    from engine.channel_map import build_channel_map, log_channel_map

    cfg = load_config()
    df, meta = load_csv_chunked(file_path, cfg)
    chan_map = build_channel_map(df, cfg)
    log_channel_map(chan_map)
    val_warnings = validate_data(df, chan_map, cfg)
    meta["warnings"] = meta.get("warnings", []) + val_warnings
    return df, meta, chan_map


# ─── App header ────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="apollo-header">
        <span style="font-size:28px; font-weight:900; color:#2E86DE; letter-spacing:2px;">
            DTT
        </span>
        <span style="font-size:18px; color:#E0EAF4; margin-left:12px;">
            Tyre Duty Cycle Dashboard
        </span>
        <span style="font-size:13px; color:#90B4CE; margin-left:16px;">
            Module 1 — WFT Analysis
        </span>
        <span style="float:right; font-size:12px; color:#4A7FA5; margin-top:6px;">
            Apollo Tyres Ltd
        </span>
    </div>
    """, unsafe_allow_html=True)


# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    cfg = load_config()

    # Sidebar
    from ui.components import render_sidebar, render_load_status, apply_time_filter
    sidebar = render_sidebar(cfg)

    render_header()

    file_path = sidebar["file_path"]
    display_unit = sidebar["display_unit"]
    sign_conv = sidebar["sign_convention"]
    selected_wheels = sidebar["selected_wheels"]
    time_range = sidebar["time_range"]
    use_dist_weight = sidebar["use_distance_weight"]
    hist_bins = sidebar["hist_bins"]
    heatmap_bins = sidebar["heatmap_bins"]
    miner_m = sidebar["miner_m"]

    # ── Load data ──────────────────────────────────────────────────────────
    if not file_path or not os.path.isfile(file_path):
        st.info(
            "Enter the path to your WFT CSV in the sidebar to begin.\n\n"
            "Expected: pre-processed FAMOS output (~720 MB, ~2.4 M rows, 38 columns)."
        )
        return

    try:
        cfg_hash = hash(str(cfg))
        df, meta, chan_map = cached_load(file_path, cfg_hash)
    except Exception as exc:
        st.error(f"Failed to load file:\n\n```\n{exc}\n```")
        logger.exception("Load failed")
        return

    render_load_status(meta, meta.get("warnings", []))

    # Apply time/distance filter
    df_filtered = apply_time_filter(df, meta, time_range)
    if time_range:
        n_filt = len(df_filtered)
        n_total = meta["n_rows"]
        st.caption(
            f"Filter active: {time_range[0]}–{time_range[1]} s  "
            f"| {n_filt:,} / {n_total:,} rows ({n_filt/n_total*100:.1f}%)"
        )

    # Override config bins from sidebar
    cfg["histogram"]["bins"] = hist_bins
    cfg["heatmap"]["bins"] = heatmap_bins
    cfg["fatigue"]["miner_slope_m"] = miner_m

    # ── Top-level metrics bar ─────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sample Rate", f"{meta['fs']:.1f} Hz")
    c2.metric("Total Rows", f"{meta['n_rows']:,.0f}")
    c3.metric("Duration", f"{meta['duration_s']/3600:.2f} h")
    c4.metric("Distance", f"{meta['total_dist_km']:.1f} km")
    c5.metric("Channels", str(sum(1 for v in chan_map.values() if v is not None)))

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Force Histograms",
        "Heatmaps",
        "Trip Statistics",
        "g-g & Severity",
        "Fatigue",
        "Box / Distance",
        "Reports"
    ])

    # ════════════════════════════════════════════════════════════════════
    # TAB 1 — Force Histograms
    # ════════════════════════════════════════════════════════════════════
    with tab1:
        from ui.components import render_tab_header
        render_tab_header(
            "Force Histograms",
            "Fx (longitudinal), Fy (lateral), Fz (vertical) distributions per wheel"
        )
        view_mode = st.radio(
            "View",
            ["Per wheel (2×2 grid)", "Combined (overlaid)"],
            horizontal=True,
            key="hist_view",
        )
        with st.spinner("Computing histograms …"):
            from engine.histograms import compute_force_histograms
            hist_results = compute_force_histograms(
                df_filtered, chan_map, cfg,
                wheels=selected_wheels,
                display_unit=display_unit,
                use_distance_weight=use_dist_weight,
            )
        from ui.plots import plot_force_histograms
        combined = view_mode.startswith("Combined")
        fig = plot_force_histograms(
            hist_results, selected_wheels, display_unit,
            use_distance=use_dist_weight, combined=combined,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════
    # TAB 2 — Heatmaps
    # ════════════════════════════════════════════════════════════════════
    with tab2:
        render_tab_header(
            "2D Correlation Heatmaps",
            "Fx×Fy, Fz×Fy, Fz×Fx density grids — per-cell % of total samples"
        )
        pairing = st.selectbox(
            "Force pairing",
            ["Fx×Fy", "Fz×Fy", "Fz×Fx"],
            key="heatmap_pairing",
        )
        hm_view = st.radio(
            "View",
            ["2×2 Grid (all wheels)", "Individual wheel"],
            horizontal=True,
            key="hm_view",
        )

        with st.spinner("Computing heatmaps …"):
            from engine.heatmaps import compute_all_heatmaps
            hm_results = compute_all_heatmaps(
                df_filtered, chan_map, cfg,
                wheels=selected_wheels,
                display_unit=display_unit,
            )

        from ui.plots import plot_all_heatmaps_grid, plot_heatmap
        min_pct = cfg.get("heatmap", {}).get("min_pct_label", 0.2)

        if hm_view.startswith("2×2"):
            fig = plot_all_heatmaps_grid(hm_results, pairing, selected_wheels, min_pct_label=min_pct)
            st.plotly_chart(fig, use_container_width=True)
        else:
            wheel_sel = st.selectbox("Wheel", selected_wheels, key="hm_wheel")
            result = hm_results.get((wheel_sel, pairing))
            if result:
                fig = plot_heatmap(result, min_pct_label=min_pct)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No data for {wheel_sel} — {pairing}")

    # ════════════════════════════════════════════════════════════════════
    # TAB 3 — Trip Statistics
    # ════════════════════════════════════════════════════════════════════
    with tab3:
        render_tab_header(
            "Trip Statistics",
            "Overall trip metrics, IMU acceleration summary, per-wheel force summary"
        )
        with st.spinner("Computing trip statistics …"):
            from engine.trip_stats import compute_trip_stats, stats_to_dataframe
            trip_stats = compute_trip_stats(
                df_filtered, chan_map, meta, cfg, display_unit=display_unit
            )
            dfs = stats_to_dataframe(trip_stats)

        # Trip overview
        st.markdown("#### Overall Trip")
        trip = trip_stats.get("trip", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Distance", f"{trip.get('total_dist_km', 0):.1f} km")
        col2.metric("Duration", f"{trip.get('duration_h', 0):.2f} h")
        col3.metric("Speed (mean)", f"{trip.get('mean_kmh', 0):.1f} km/h")
        col4.metric("Speed (max)", f"{trip.get('max_kmh', 0):.1f} km/h")

        # Accel stats
        accel = trip_stats.get("accel", {})
        if accel:
            st.markdown("#### IMU / Acceleration")
            col1, col2, col3 = st.columns(3)
            col1.metric("Lat Acc max |g|", f"{accel.get('latacc_max_abs', 0):.3f} g")
            col2.metric("Long Acc max |g|", f"{accel.get('longacc_max_abs', 0):.3f} g")
            col3.metric("Yawrate max", f"{accel.get('yawrate_max_abs', 0):.1f} deg/s")

        # Force summary table
        st.markdown(f"#### Per-Wheel Force Summary ({display_unit})")
        forces_df = dfs.get("forces")
        if forces_df is not None and not forces_df.empty:
            st.dataframe(forces_df.round(2), use_container_width=True)

        # Wheel speed (WS1) table
        ws_df = dfs.get("wheels")
        if ws_df is not None and not ws_df.empty:
            st.markdown("#### Wheel Speed (WS1) Summary (rpm)")
            st.dataframe(ws_df.round(1), use_container_width=True)

    # ════════════════════════════════════════════════════════════════════
    # TAB 4 — g-g & Severity
    # ════════════════════════════════════════════════════════════════════
    with tab4:
        render_tab_header(
            "g-g Diagram & Force Severity",
            "Longitudinal vs lateral acceleration (g) · Force severity banding by Fz percentiles"
        )
        with st.spinner("Computing g-g data …"):
            from engine.gg_severity import compute_gg, compute_force_severity
            gg_result = compute_gg(df_filtered, chan_map, cfg, sign_convention=sign_conv)
            severity = compute_force_severity(df_filtered, chan_map, cfg, display_unit=display_unit)

        from ui.plots import plot_gg, plot_force_severity

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("#### g-g Diagram")
            if gg_result.get("n_valid", 0) == 0:
                st.warning("No valid acceleration data found (check channel names).")
            else:
                fig_gg = plot_gg(gg_result)
                st.plotly_chart(fig_gg, use_container_width=True)

        with col2:
            st.markdown("#### Force Severity (Fz Banding)")
            if severity:
                fig_sev = plot_force_severity(
                    {w: severity[w] for w in selected_wheels if w in severity},
                    display_unit=display_unit,
                )
                st.plotly_chart(fig_sev, use_container_width=True)

                # Numeric band thresholds
                st.markdown("**Fz Percentile Thresholds**")
                rows = []
                for w in selected_wheels:
                    bands = severity.get(w, {}).get("fz_bands", {})
                    if bands:
                        rows.append({"Wheel": w, **bands})
                if rows:
                    import pandas as pd
                    band_df = pd.DataFrame(rows)
                    st.dataframe(band_df.round(1), use_container_width=True)

    # ════════════════════════════════════════════════════════════════════
    # TAB 5 — Fatigue
    # ════════════════════════════════════════════════════════════════════
    with tab5:
        render_tab_header(
            "Rainflow Fatigue Analysis",
            f"Miner's rule damage (m={miner_m}) · Load-range histogram · Cumulative exceedance"
        )
        force_suffix = st.selectbox(
            "Force channel suffix",
            ["Fx", "Fy", "Fz"],
            key="fatigue_suffix",
        )
        run_fatigue = st.button("Run Fatigue Analysis", type="primary", key="btn_fatigue")

        if run_fatigue or st.session_state.get("fatigue_cache") is not None:
            if run_fatigue:
                with st.spinner("Running rainflow cycle counting (may take 30–90 s on full file) …"):
                    from engine.fatigue import compute_fatigue_summary
                    fat_result = compute_fatigue_summary(
                        df_filtered, chan_map, cfg,
                        wheels=selected_wheels,
                        display_unit=display_unit,
                    )
                st.session_state["fatigue_cache"] = fat_result
            else:
                fat_result = st.session_state["fatigue_cache"]

            from ui.plots import plot_fatigue_channel, plot_axle_comparison

            channels_result = fat_result.get("channels", {})
            axle_result = fat_result.get("axles", {})

            # Per-channel plots
            st.markdown("#### Per-Channel Rainflow Results")
            for wheel in selected_wheels:
                key = f"{wheel}_{force_suffix}"
                ch_res = channels_result.get(key)
                if ch_res and ch_res.get("total_cycles", 0) > 0:
                    st.markdown(f"**{key}** — {ch_res['total_cycles']:,} cycles | "
                                f"P95 range = {ch_res['p95_range']:.1f} {display_unit}")
                    fig = plot_fatigue_channel(ch_res)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"{key}: No cycles extracted or data unavailable.")

            # Axle comparison
            axle_data = axle_result.get(force_suffix)
            if axle_data:
                st.markdown(f"#### Front vs Rear Axle — {force_suffix}")
                ratio = axle_data.get("damage_ratio_rear_over_front")
                norm_front = axle_data.get("damage_norm_front")
                norm_rear  = axle_data.get("damage_norm_rear")
                p95_ratio  = axle_data.get("p95_range_ratio")
                internal_unit = axle_data.get("internal_unit", "daN")

                if ratio is not None:
                    # ── Normalised index (primary display) ──
                    col1, col2, col3 = st.columns(3)
                    col1.metric(
                        "Front (FL+FR) — Normalised",
                        "1.00 (baseline)",
                        help="Reference axle = 1.00 (normalised damage index)",
                    )
                    col2.metric(
                        "Rear (RL+RR) — Normalised",
                        f"{norm_rear:.2f}× more damage" if norm_rear is not None else "N/A",
                        delta=f"{ratio:.2f}× vs front" if ratio else None,
                        delta_color="inverse" if ratio and ratio > 1 else "normal",
                        help="Normalised damage: Rear / Front ratio",
                    )
                    col3.metric(
                        "P95 Range ratio",
                        f"{p95_ratio:.2f}× (Rear / Front)" if p95_ratio else "N/A",
                        help=(
                            f"Rear P95 load range is {p95_ratio:.2f}× the front. "
                            f"With m={miner_m}, expected damage ratio = "
                            f"{p95_ratio**miner_m:.1f}×" if p95_ratio else ""
                        ),
                    )

                    # Explanatory caption
                    front_cycles = axle_data.get("front", {}).get("total_cycles", 0)
                    rear_cycles  = axle_data.get("rear",  {}).get("total_cycles", 0)
                    st.caption(
                        f"**Front axle has more cycles ({front_cycles:,}) "
                        f"but less damage than rear ({rear_cycles:,} cycles).** "
                        f"This is correct: Miner's rule is amplitude-dominated (m={miner_m}), "
                        f"so a smaller number of high-amplitude cycles on the rear outweigh "
                        f"many small-amplitude cycles on the front.\n"
                        f"Absolute damage values (e.g. 5.32e+13) are a "
                        f"**relative index** computed in {internal_unit} — "
                        f"they carry no physical unit and cannot be compared across different m values."
                    )

                fig_axle = plot_axle_comparison(axle_data, force_suffix)
                st.plotly_chart(fig_axle, use_container_width=True)

                # Raw absolute in collapsible expander (not deleted, just de-emphasised)
                with st.expander("Raw absolute damage index (advanced)", expanded=False):
                    front_dmg = axle_data.get("front", {}).get("damage", 0)
                    rear_dmg  = axle_data.get("rear",  {}).get("damage", 0)
                    st.markdown(
                        f"| Axle | Relative Damage Index (arb. units, m={miner_m}) |\n"
                        f"|---|---|\n"
                        f"| Front (FL+FR) | `{front_dmg:.3e}` |\n"
                        f"| Rear (RL+RR)  | `{rear_dmg:.3e}`  |\n\n"
                        f"*Computed in {internal_unit}. Changing m will change this number "
                        f"by orders of magnitude. Use the normalised ratio above for comparisons.*"
                    )

    # ════════════════════════════════════════════════════════════════════
    # TAB 6 — Box / Distance Distribution
    # ════════════════════════════════════════════════════════════════════
    with tab6:
        render_tab_header(
            "Box Plots & Distance Distribution",
            "Force distribution boxes · Distance-weighted force histograms (km on Y axis)"
        )
        suffix_sel = st.selectbox("Force suffix", ["Fx", "Fy", "Fz"], key="box_suffix")

        with st.spinner("Computing box stats and distance distributions …"):
            from engine.box_distance import compute_box_stats, compute_distance_distribution
            box_results = compute_box_stats(
                df_filtered, chan_map, cfg,
                wheels=selected_wheels, display_unit=display_unit
            )
            dist_results = compute_distance_distribution(
                df_filtered, chan_map, cfg,
                wheels=selected_wheels, display_unit=display_unit
            )

        from ui.plots import plot_box_all_wheels, plot_distance_distribution

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"#### Box Plot — {suffix_sel} ({display_unit})")
            fig_box = plot_box_all_wheels(box_results, suffix_sel, selected_wheels, display_unit)
            st.plotly_chart(fig_box, use_container_width=True)

        with col2:
            st.markdown(f"#### Distance Distribution — {suffix_sel}")
            fig_dist = plot_distance_distribution(
                dist_results, selected_wheels, suffix_sel, display_unit
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        # Stats table
        import pandas as pd
        rows = []
        for wheel in selected_wheels:
            key = f"{wheel}_{suffix_sel}"
            res = box_results.get(key)
            if res:
                rows.append({
                    "Wheel": wheel,
                    "Min": round(res["min"], 2),
                    "P5": round(res["p5"], 2),
                    "Q1": round(res["q1"], 2),
                    "Median": round(res["median"], 2),
                    "Mean": round(res["mean"], 2),
                    "Q3": round(res["q3"], 2),
                    "P95": round(res["p95"], 2),
                    "Max": round(res["max"], 2),
                })
        if rows:
            st.markdown(f"#### Statistics ({display_unit})")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ════════════════════════════════════════════════════════════════════
    # ════════════════════════════════════════════════════════════════════
    # TAB 7 — Reports & Export
    # ════════════════════════════════════════════════════════════════════
    with tab7:
        from ui.components import render_tab_header
        render_tab_header(
            "Reports & Export",
            "Generate consolidated or per-section PDF reports, or download raw CSV results"
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### PDF Reports")
            report_sel = st.selectbox(
                "Select report to generate",
                [
                    "Full Report (all analyses)",
                    "Force Histograms",
                    "Heatmaps",
                    "Trip Statistics",
                    "g-g & Severity",
                    "Fatigue",
                    "Box / Distance",
                ]
            )
            if st.button(f"Generate {report_sel.split(' (')[0]}", type="primary", key="btn_pdf_gen"):
                with st.spinner(f"Generating {report_sel} (this may take a moment on cold start) \u2026"):
                    try:
                        from export.pdf_report import generate_pdf_bytes
                        from ui.plots import (
                            plot_force_histograms, plot_all_heatmaps_grid, plot_trip_stats,
                            plot_gg, plot_box_all_wheels, plot_fatigue_channel, plot_distance_distribution
                        )
                        
                        figures = []
                        
                        # --- Force Histograms ---
                        if report_sel in ("Full Report (all analyses)", "Force Histograms"):
                            if "hist_results" not in locals():
                                from engine.histograms import compute_force_histograms
                                hist_results = compute_force_histograms(df_filtered, chan_map, cfg, wheels=selected_wheels, display_unit=display_unit, use_distance_weight=use_dist_weight)
                            fig_hist = plot_force_histograms(hist_results, selected_wheels, display_unit, use_distance=use_dist_weight, combined=False)
                            figures.append(("Force Histograms", fig_hist, f"hist_{display_unit}_{sign_conv}_{hist_bins}_{use_dist_weight}"))
                                
                        # --- Heatmaps ---
                        if report_sel in ("Full Report (all analyses)", "Heatmaps"):
                            if "hm_results" not in locals():
                                from engine.heatmaps import compute_all_heatmaps
                                hm_results = compute_all_heatmaps(df_filtered, chan_map, cfg, wheels=selected_wheels, display_unit=display_unit)
                            for pair in ["Fx\u00d7Fy", "Fz\u00d7Fy", "Fz\u00d7Fx"]:
                                fig_hm = plot_all_heatmaps_grid(hm_results, pair, selected_wheels, min_pct_label=cfg.get("heatmap", {}).get("min_pct_label", 0.2))
                                figures.append((f"Heatmaps \u2014 {pair}", fig_hm, f"hm_{pair}_{display_unit}_{sign_conv}_{heatmap_bins}"))

                        # --- Trip Statistics ---
                        if report_sel in ("Full Report (all analyses)", "Trip Statistics"):
                            if "trip_stats" not in locals():
                                from engine.trip_stats import compute_trip_stats
                                trip_stats = compute_trip_stats(df_filtered, chan_map, meta, cfg, display_unit=display_unit)
                            fig_ts = plot_trip_stats(trip_stats)
                            figures.append(("Trip Statistics", fig_ts, f"ts_{display_unit}_{sign_conv}"))
                                
                        # --- g-g & Severity ---
                        if report_sel in ("Full Report (all analyses)", "g-g & Severity"):
                            if "gg_result" not in locals():
                                from engine.gg_severity import compute_gg
                                gg_result = compute_gg(df_filtered, chan_map, cfg, sign_convention=sign_conv)
                            if gg_result.get("n_valid", 0) > 0:
                                fig_gg = plot_gg(gg_result)
                                figures.append(("g-g Diagram", fig_gg, f"gg_{display_unit}_{sign_conv}"))
                                
                        # --- Fatigue ---
                        if report_sel in ("Full Report (all analyses)", "Fatigue"):
                            if "fat_result" not in locals():
                                from engine.fatigue import compute_fatigue_summary
                                fat_result = compute_fatigue_summary(df_filtered, chan_map, cfg, wheels=selected_wheels, display_unit=display_unit)
                            
                            # Add a channel for demonstration in the report. We'll pick 'Fz' as default if not set.
                            fsfx = locals().get("force_suffix", "Fz")
                            channels_res = fat_result.get("channels", {})
                            for wheel in selected_wheels:
                                ch_key = f"{wheel}_{fsfx}"
                                ch_res = channels_res.get(ch_key)
                                if ch_res and ch_res.get("total_cycles", 0) > 0:
                                    fig_fat = plot_fatigue_channel(ch_res)
                                    figures.append((f"Fatigue \u2014 {ch_key}", fig_fat, f"fat_{ch_key}_{display_unit}_{miner_m}"))

                        # --- Box / Distance ---
                        if report_sel in ("Full Report (all analyses)", "Box / Distance"):
                            if "box_results" not in locals() or "dist_results" not in locals():
                                from engine.box_distance import compute_box_summaries, compute_distance_distribution
                                box_results = compute_box_summaries(df_filtered, chan_map, cfg, wheels=selected_wheels, display_unit=display_unit)
                                dist_results = compute_distance_distribution(df_filtered, chan_map, cfg, wheels=selected_wheels, display_unit=display_unit)
                            
                            suffixes = ["Fx", "Fy", "Fz"] if report_sel == "Full Report (all analyses)" else ["Fz"]
                            for sfx in suffixes:
                                fig_box = plot_box_all_wheels(box_results, sfx, selected_wheels, display_unit)
                                fig_dist = plot_distance_distribution(dist_results, selected_wheels, sfx, display_unit)
                                figures.append((f"Box Plot \u2014 {sfx}", fig_box, f"box_{sfx}_{display_unit}"))
                                figures.append((f"Distance \u2014 {sfx}", fig_dist, f"dist_{sfx}_{display_unit}"))

                        if figures:
                            pdf_bytes = generate_pdf_bytes(figures, title=report_sel.split(" (")[0])
                            st.session_state["pdf_bytes"] = pdf_bytes
                            st.session_state["pdf_name"] = f"dtt_wft_{report_sel.split(' (')[0].replace(' ', '_').lower()}.pdf"
                        else:
                            st.warning("No data available to generate this report.")
                    except Exception as e:
                        st.error(str(e))
            
            if st.session_state.get("pdf_bytes"):
                st.download_button(
                    "Download PDF",
                    data=st.session_state["pdf_bytes"],
                    file_name=st.session_state.get("pdf_name", "report.pdf"),
                    mime="application/pdf",
                    type="primary",
                )
                
        with col2:
            st.markdown("#### CSV Data")
            if st.button("Generate CSV Results", key="btn_csv"):
                with st.spinner("Generating CSV archive \u2026"):
                    from export.csv_export import export_all_csv_bytes
                    csv_zip = export_all_csv_bytes(
                        hist_results={} if "hist_results" not in locals() else hist_results,
                        trip_stats={} if "trip_stats" not in locals() else trip_stats,
                    )
                    if csv_zip:
                        st.session_state["csv_zip"] = csv_zip
                        
            if st.session_state.get("csv_zip"):
                st.download_button(
                    "Download CSV ZIP",
                    data=st.session_state["csv_zip"],
                    file_name="dtt_wft_csv_results.zip",
                    mime="application/zip",
                )


if __name__ == "__main__":
    main()
