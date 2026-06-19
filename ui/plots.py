"""
ui/plots.py
===========
Plotly figure builders for the DTT WFT dashboard.

Each function takes engine results (data arrays / dicts) and returns
a Plotly figure object.  No pandas reads, no file I/O here.

Per-wheel colour palette (consistent with WFT_Analyzer_v5):
  FL = #2E86DE (blue)
  FR = #E74C3C (red)
  RL = #27AE60 (green)
  RR = #8E44AD (purple)
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Colour palette ────────────────────────────────────────────────────────
WHEEL_COLORS = {
    "FL": "#2E86DE",
    "FR": "#E74C3C",
    "RL": "#27AE60",
    "RR": "#8E44AD",
}
SUFFIX_NAMES = {"Fx": "Longitudinal", "Fy": "Lateral", "Fz": "Vertical"}

# Dark theme tokens
BG_DARK   = "#0D1B2A"
PANEL_BG  = "#1B3A5C"
GRID_CLR  = "rgba(27,58,92,0.6)"
TEXT_CLR  = "#E0EAF4"
TEXT_SEC  = "#90B4CE"
ACCENT    = "#E8862A"


def _dark_layout(fig: go.Figure, title: str = "", height: int = 500) -> go.Figure:
    """Apply consistent dark theme to a figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG_DARK,
        plot_bgcolor="#0A2540",
        font=dict(family="Inter, Arial, sans-serif", color=TEXT_CLR, size=12),
        title=dict(text=title, font=dict(size=15, color=TEXT_CLR), x=0.5),
        height=height,
        legend=dict(bgcolor="rgba(13,27,42,0.9)", bordercolor=PANEL_BG, borderwidth=1),
        margin=dict(l=50, r=30, t=55, b=50),
    )
    fig.update_xaxes(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR)
    fig.update_yaxes(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR)
    return fig


# ─── §6.1 Force Histograms ─────────────────────────────────────────────────

def plot_force_histograms(
    hist_results: dict,
    wheels: list[str],
    display_unit: str = "daN",
    use_distance: bool = False,
    combined: bool = False,
) -> go.Figure:
    """
    Build force histogram figure.

    Parameters
    ----------
    hist_results : Output of engine.compute_force_histograms()
    wheels       : List of wheel positions to plot
    display_unit : 'N' or 'daN'
    use_distance : If True, plot distance (km) on Y axis; else count
    combined     : False = 2×2 small-multiples per wheel (3 clean vlines each)
                   True  = overlaid step histograms for comparison

    Returns
    -------
    Plotly Figure
    """
    from .components import FORCE_SUFFIXES
    suffixes = FORCE_SUFFIXES
    y_label = "Distance (km)" if use_distance else "Count"

    if not combined:
        # ─────────────────────────────────────────────────────────
        # Per-wheel 2×2 small-multiples grid.
        # Layout: rows = wheels (up to 4), cols = force suffixes (Fx, Fy, Fz)
        # Each subplot has exactly 3 percentile vlines for that wheel only.
        # ─────────────────────────────────────────────────────────
        n_wheels = len(wheels)
        n_suffix = len(suffixes)
        subplot_titles = []
        for wheel in wheels:
            for suffix in suffixes:
                subplot_titles.append(
                    f"{wheel} — {suffix} ({SUFFIX_NAMES.get(suffix, suffix)})"
                )

        fig = make_subplots(
            rows=n_wheels,
            cols=n_suffix,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.06,
            vertical_spacing=0.10,
        )

        for row_idx, wheel in enumerate(wheels, start=1):
            color = WHEEL_COLORS.get(wheel, "#AAAAAA")
            for col_idx, suffix in enumerate(suffixes, start=1):
                key = f"{wheel}_{suffix}"
                res = hist_results.get(key)
                if res is None:
                    continue

                y = res["hist_dist_km"] if use_distance else res["hist_count"]

                # Solid bar for this single wheel
                fig.add_trace(
                    go.Bar(
                        x=res["bin_centers"],
                        y=y,
                        name=wheel,
                        marker_color=color,
                        opacity=0.85,
                        showlegend=False,
                        hovertemplate=(
                            f"Force: %{{x:.1f}} {display_unit}<br>"
                            f"{y_label}: %{{y:,.0f}}<extra>{key}</extra>"
                        ),
                    ),
                    row=row_idx, col=col_idx,
                )

                # Percentile vlines — exactly 3 per subplot, no collision
                pct_styles = {
                    80: ("dot",    TEXT_SEC),
                    90: ("dash",   "#F39C12"),
                    95: ("dashdot","#C0392B"),
                }
                pct_y_stagger = {80: 0.98, 90: 0.93, 95: 0.88}
                for p, pval in res["percentiles"].items():
                    dash, pcolor = pct_styles.get(p, ("solid", TEXT_SEC))
                    fig.add_vline(
                        x=pval,
                        row=row_idx, col=col_idx,
                        line=dict(color=pcolor, width=1.5, dash=dash),
                        annotation_text=f"P{p}",
                        annotation_position="top right",
                        annotation_y=pct_y_stagger.get(p, 0.95),
                        annotation_font_color=pcolor,
                        annotation_font_size=9,
                    )

                # Auto-zoom X axis to P1–P99 view range
                vr = res.get("view_range")
                if vr and vr[0] is not None:
                    axis_key = f"xaxis{(row_idx - 1) * n_suffix + col_idx}"
                    fig.update_layout(
                        **{axis_key: dict(range=vr)}
                    )

        _dark_layout(
            fig,
            f"Force Histograms — Per Wheel  ({display_unit})",
            height=max(380, 240 * n_wheels),
        )
        # Axis labels on outermost row/col only to avoid clutter
        for col_idx in range(1, n_suffix + 1):
            fig.update_xaxes(
                title_text=f"Force ({display_unit})",
                row=n_wheels, col=col_idx,
            )
        for row_idx in range(1, n_wheels + 1):
            fig.update_yaxes(
                title_text=y_label,
                row=row_idx, col=1,
            )
        return fig

    else:
        # ─────────────────────────────────────────────────────────
        # Combined view: step/outline histograms for all wheels overlaid.
        # Percentiles for a single reference wheel (first selected) shown only.
        # This avoids 4×3 = 12 overlapping vlines.
        # ─────────────────────────────────────────────────────────
        fig = make_subplots(
            rows=1, cols=len(suffixes),
            subplot_titles=[f"{s} — {SUFFIX_NAMES.get(s, s)}" for s in suffixes],
        )
        ref_wheel = wheels[0] if wheels else None

        for col_idx, suffix in enumerate(suffixes, start=1):
            # Collect view_range from the reference wheel for this suffix
            ref_key = f"{ref_wheel}_{suffix}" if ref_wheel else None
            ref_res = hist_results.get(ref_key)

            for wheel in wheels:
                key = f"{wheel}_{suffix}"
                res = hist_results.get(key)
                if res is None:
                    continue
                y = res["hist_dist_km"] if use_distance else res["hist_count"]
                color = WHEEL_COLORS.get(wheel, "#AAAAAA")

                # Step histogram: use Scatter with fill for clean overlay
                # Build step-staircase from bin_edges + y
                edges = res["bin_edges"]
                step_x = np.repeat(edges, 2)[1:-1]
                step_y = np.repeat(y, 2)

                fig.add_trace(
                    go.Scatter(
                        x=step_x,
                        y=step_y,
                        mode="lines",
                        name=wheel,
                        line=dict(color=color, width=2),
                        fill="tozeroy",
                        fillcolor=_hex_to_rgba(color, 0.12),
                        opacity=0.85,
                        showlegend=(col_idx == 1),
                        legendgroup=wheel,
                        hovertemplate=(
                            f"Force: %{{x:.1f}} {display_unit}<br>"
                            f"{y_label}: %{{y:,.0f}}<extra>{wheel} {suffix}</extra>"
                        ),
                    ),
                    row=1, col=col_idx,
                )

            # Percentile vlines for reference wheel only (3 lines total)
            if ref_res is not None:
                ref_color = WHEEL_COLORS.get(ref_wheel, TEXT_SEC)
                pct_styles = {80: "dot", 90: "dash", 95: "dashdot"}
                pct_y_stagger = {80: 0.98, 90: 0.93, 95: 0.88}
                for p, pval in ref_res["percentiles"].items():
                    fig.add_vline(
                        x=pval, row=1, col=col_idx,
                        line=dict(color=ref_color, width=1.5, dash=pct_styles.get(p, "dot")),
                        annotation_text=f"{ref_wheel} P{p}",
                        annotation_position="top right",
                        annotation_y=pct_y_stagger.get(p, 0.95),
                        annotation_font_color=ref_color,
                        annotation_font_size=9,
                    )

                # Auto-zoom to reference wheel P1–P99 range
                vr = ref_res.get("view_range")
                if vr and vr[0] is not None:
                    fig.update_xaxes(range=vr, row=1, col=col_idx)

        _dark_layout(fig, f"Force Histograms — Combined ({display_unit})", height=480)
        fig.update_xaxes(title_text=f"Force ({display_unit})")
        fig.update_yaxes(title_text=y_label)
        return fig


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert #RRGGBB to rgba(r,g,b,alpha) string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ─── §6.2 Heatmaps ────────────────────────────────────────────────────────

def plot_heatmap(
    hm_result: dict,
    min_pct_label: float = 0.2,
) -> go.Figure:
    """
    Build a single 2D correlation density heatmap with % cell annotations.

    Matches the Apollo reference chart style:
      - turbo colormap
      - per-cell % annotation (omit cells < min_pct_label)
      - axes in display_unit (daN by default)
    """
    H_pct   = hm_result["H_pct"]
    xedges  = hm_result["xedges"]
    yedges  = hm_result["yedges"]
    x_label = hm_result["x_label"]
    y_label = hm_result["y_label"]
    title   = hm_result["title"]
    n_valid = hm_result.get("n_valid", 0)

    # Bin centres
    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])

    # Annotation text matrix
    text_matrix = []
    for j in range(H_pct.shape[1]):
        row_text = []
        for i in range(H_pct.shape[0]):
            val = H_pct[i, j]
            row_text.append(f"{val:.1f}%" if val >= min_pct_label else "")
        text_matrix.append(row_text)

    fig = go.Figure(go.Heatmap(
        z=H_pct.T,
        x=x_centers,
        y=y_centers,
        colorscale="turbo",
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=8, color="white"),
        colorbar=dict(title="% samples", tickfont=dict(color=TEXT_CLR)),
        hovertemplate=(
            f"{x_label}: %{{x:.1f}}<br>{y_label}: %{{y:.1f}}<br>"
            "% samples: %{z:.2f}<extra></extra>"
        ),
    ))

    _dark_layout(fig, f"{title}  (n={n_valid:,})", height=500)
    return fig


def plot_all_heatmaps_grid(
    heatmap_results: dict,
    pairing: str,
    wheels: list[str],
    min_pct_label: float = 0.2,
) -> go.Figure:
    """
    2x2 grid of heatmaps for the given pairing across all selected wheels.
    """
    nrows = 2
    ncols = 2

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[f"{w} — {pairing}" for w in wheels[:4]],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )
    positions = [(r + 1, c + 1) for r in range(nrows) for c in range(ncols)]

    for idx, wheel in enumerate(wheels[:4]):
        row, col = positions[idx]
        hm = heatmap_results.get((wheel, pairing))
        if hm is None or hm.get("n_valid", 0) == 0:
            continue
        H_pct   = hm["H_pct"]
        xedges  = hm["xedges"]
        yedges  = hm["yedges"]
        x_label = hm.get("x_label", "")
        y_label = hm.get("y_label", "")
        x_centers = 0.5 * (xedges[:-1] + xedges[1:])
        y_centers = 0.5 * (yedges[:-1] + yedges[1:])

        text_matrix = []
        for j in range(H_pct.shape[1]):
            row_text = []
            for i in range(H_pct.shape[0]):
                val = H_pct[i, j]
                row_text.append(f"{val:.1f}%" if val >= min_pct_label else "")
            text_matrix.append(row_text)

        fig.add_trace(
            go.Heatmap(
                z=H_pct.T,
                x=x_centers,
                y=y_centers,
                colorscale="turbo",
                text=text_matrix,
                texttemplate="%{text}",
                textfont=dict(size=7),
                showscale=(idx == 0),
                showlegend=False,
                colorbar=dict(x=1.02, title="% samples"),
                hovertemplate=(
                    f"{x_label}: %{{x:.1f}}<br>"
                    f"{y_label}: %{{y:.1f}}<br>"
                    "% samples: %{z:.2f}<extra></extra>"
                ),
            ),
            row=row, col=col,
        )

    _dark_layout(fig, f"Heatmap — {pairing} — All Wheels", height=700)
    return fig


# ─── §6.3 Trip Statistics ─────────────────────────────────────────────────

def plot_trip_stats(trip_stats: dict) -> go.Figure:
    """
    Render trip statistics as a Plotly Table figure (all sub-tables stacked).
    """
    import pandas as pd
    from .components import FORCE_SUFFIXES
    from engine.trip_stats import stats_to_dataframe

    dfs = stats_to_dataframe(trip_stats)

    fig = go.Figure()
    # Trip overview table
    trip_df = dfs.get("trip", pd.DataFrame())
    if not trip_df.empty:
        fig.add_trace(go.Table(
            header=dict(
                values=["Metric", "Value"],
                fill_color=PANEL_BG,
                font=dict(color=TEXT_CLR, size=12),
                align="left",
            ),
            cells=dict(
                values=[trip_df["Metric"], trip_df["Value"].round(3)],
                fill_color=BG_DARK,
                font=dict(color=TEXT_CLR, size=11),
                align="left",
            ),
        ))
    _dark_layout(fig, "Trip Overview", height=400)
    return fig


def plot_force_summary_table(trip_stats: dict, display_unit: str = "daN") -> go.Figure:
    """
    Per-wheel force summary as a Plotly table.
    """
    from engine.trip_stats import stats_to_dataframe
    dfs = stats_to_dataframe(trip_stats)
    forces_df = dfs.get("forces")
    if forces_df is None or forces_df.empty:
        return go.Figure()

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in forces_df.columns],
            fill_color=PANEL_BG,
            font=dict(color=TEXT_CLR, size=12),
            align="center",
        ),
        cells=dict(
            values=[forces_df[c].round(2) if forces_df[c].dtype in [float] else forces_df[c]
                    for c in forces_df.columns],
            fill_color=[
                [BG_DARK if i % 2 == 0 else "#0F2D4A" for i in range(len(forces_df))]
            ] * len(forces_df.columns),
            font=dict(color=TEXT_CLR, size=11),
            align="center",
        ),
    ))
    _dark_layout(fig, f"Force Summary by Wheel ({display_unit})", height=450)
    return fig


# ─── §6.4 g-g Plot ────────────────────────────────────────────────────────

def plot_gg(gg_result: dict) -> go.Figure:
    """
    Render the g-g diagram.

    CRITICAL: axes are labelled in g (not m/s²).
    Limits and reference circles are in g.
    """
    latacc  = gg_result.get("latacc_plot", np.array([]))
    longacc = gg_result.get("longacc_plot", np.array([]))
    axis_lim = gg_result.get("axis_limit_g", 1.2)
    ref_circles = gg_result.get("ref_circles_g", [0.2, 0.4, 0.6, 0.8])
    n_valid = gg_result.get("n_valid", 0)

    fig = go.Figure()

    # Reference circles
    theta = np.linspace(0, 2 * np.pi, 200)
    for r in ref_circles:
        fig.add_trace(go.Scatter(
            x=r * np.cos(theta),
            y=r * np.sin(theta),
            mode="lines",
            line=dict(color="rgba(144,180,206,0.3)", width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_annotation(
            x=0, y=r,
            text=f"{r}g",
            showarrow=False,
            font=dict(size=9, color=TEXT_SEC),
            yanchor="bottom",
        )

    # Scatter points (colour by density using 2D histogram proxy)
    if len(latacc) > 0 and len(longacc) > 0:
        fig.add_trace(go.Scattergl(
            x=latacc,
            y=longacc,
            mode="markers",
            marker=dict(
                size=2,
                color=longacc,
                colorscale="plasma",
                opacity=0.4,
                showscale=True,
                colorbar=dict(title="Long (g)", thickness=12, len=0.6),
            ),
            name=f"Data ({n_valid:,} pts)",
            hovertemplate="Lat: %{x:.3f}g<br>Long: %{y:.3f}g<extra></extra>",
        ))

    # Zero lines
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1))
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.2)", width=1))

    _dark_layout(fig, f"g-g Diagram  (n={n_valid:,} points plotted)", height=580)
    fig.update_xaxes(
        title_text="Lateral Acceleration (g)",
        range=[-axis_lim, axis_lim],
        scaleanchor="y",
        scaleratio=1,
    )
    fig.update_yaxes(
        title_text="Longitudinal Acceleration (g)",
        range=[-axis_lim, axis_lim],
    )
    return fig


def plot_force_severity(severity_result: dict, display_unit: str = "daN") -> go.Figure:
    """
    Horizontal stacked bar chart showing Fz severity band fractions per wheel.
    """
    BAND_COLORS = {
        "low":     "#27AE60",
        "medium":  "#F39C12",
        "high":    "#E67E22",
        "extreme": "#C0392B",
    }
    BAND_LABELS = {
        "low": "Low (<P80)", "medium": "Medium (P80–P90)",
        "high": "High (P90–P95)", "extreme": "Extreme (≥P95)"
    }

    wheels = list(severity_result.keys())
    bands = ["low", "medium", "high", "extreme"]

    fig = go.Figure()
    for band in bands:
        fracs = [
            severity_result.get(w, {}).get("fz_band_fractions", {}).get(band, 0) * 100
            for w in wheels
        ]
        fig.add_trace(go.Bar(
            name=BAND_LABELS[band],
            y=wheels,
            x=fracs,
            orientation="h",
            marker_color=BAND_COLORS[band],
            text=[f"{v:.1f}%" for v in fracs],
            textposition="inside",
        ))

    _dark_layout(fig, f"Force Severity — Fz Banding ({display_unit})", height=350)
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title_text="% of samples")
    fig.update_yaxes(title_text="Wheel")
    return fig


# ─── §6.4 Fatigue / Rainflow ──────────────────────────────────────────────

def plot_fatigue_channel(ch_result: dict) -> go.Figure:
    """
    3-panel fatigue plot for one channel:
      Left:   Load-range histogram
      Middle: Cumulative exceedance (log Y)
      Right:  Damage ratio badge (text summary)
    """
    key   = ch_result.get("key", "")
    color = WHEEL_COLORS.get(ch_result.get("wheel", ""), "#AAAAAA")
    unit  = ch_result.get("display_unit", "daN")
    m     = ch_result.get("miner_m", 5)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"{key} — Load Range Histogram",
            f"{key} — Cumulative Exceedance",
        ],
    )

    # Range histogram
    edges = ch_result.get("range_hist_edges", np.array([]))
    counts = ch_result.get("range_hist_counts", np.array([]))
    if len(edges) > 1:
        centers = 0.5 * (edges[:-1] + edges[1:])
        fig.add_trace(go.Bar(
            x=centers, y=counts,
            name="Cycle count",
            marker_color=color,
            opacity=0.8,
        ), row=1, col=1)

    # P95 marker
    p95 = ch_result.get("p95_range", 0.0)
    if p95 > 0:
        fig.add_vline(
            x=p95, row=1, col=1,
            line=dict(color="#C0392B", width=2, dash="dash"),
            annotation_text=f"P95={p95:.0f}",
            annotation_font_color="#C0392B",
        )

    # Cumulative exceedance
    rng_sorted = ch_result.get("cumexceed_range", np.array([]))
    exceedances = ch_result.get("cumexceed_count", np.array([]))
    if len(rng_sorted) > 0:
        # Downsample for plot speed
        step = max(1, len(rng_sorted) // 5000)
        fig.add_trace(go.Scatter(
            x=rng_sorted[::step],
            y=exceedances[::step],
            mode="lines",
            name="Exceedances",
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + (0.1,)}",
        ), row=1, col=2)

    _dark_layout(fig, f"Fatigue — {key}  (m={m})", height=450)
    fig.update_xaxes(title_text=f"Range ({unit})", row=1, col=1)
    fig.update_yaxes(title_text="Cycle count", row=1, col=1)
    fig.update_xaxes(title_text=f"Range ({unit})", row=1, col=2)
    fig.update_yaxes(title_text="No. exceedances", type="log", row=1, col=2)
    return fig


def plot_axle_comparison(axle_result: dict, suffix: str) -> go.Figure:
    """
    Front (FL+FR) vs Rear (RL+RR) fatigue comparison.

    Layout:
      Left:  Normalised damage bars (Front = 1.00, Rear = N×) — PRIMARY display
      Right: P95 load-range bar chart (front vs rear, in display_unit)

    The raw absolute damage index is shown in hover/expander only,
    labelled 'Relative Damage Index (arbitrary units, m=N)' to prevent
    mis-reading 5.32e+13 as a physical value.
    """
    front = axle_result.get("front") or {}
    rear  = axle_result.get("rear")  or {}
    ratio          = axle_result.get("damage_ratio_rear_over_front")
    norm_front     = axle_result.get("damage_norm_front", 1.0)
    norm_rear      = axle_result.get("damage_norm_rear")
    p95_ratio      = axle_result.get("p95_range_ratio")
    m              = axle_result.get("miner_m", 5)
    unit           = axle_result.get("display_unit", "daN")
    internal_unit  = axle_result.get("internal_unit", "daN")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"Normalised Damage  (Front = 1.00)",
            f"P95 Load Range ({unit})",
        ],
        column_widths=[0.6, 0.4],
    )

    # ── Left: Normalised damage bars ───────────────────────────────────
    labels  = ["Front (FL+FR)", "Rear (RL+RR)"]
    colors  = ["#2E86DE", "#E74C3C"]
    norms   = [norm_front, norm_rear]
    absvals = [front.get("damage", 0), rear.get("damage", 0)]
    ncycles = [front.get("total_cycles", 0), rear.get("total_cycles", 0)]

    for label, color, nv, av, nc in zip(labels, colors, norms, absvals, ncycles):
        if "Front" in label:
            text_str = f"<b>1.00 (baseline)</b><br><span style='font-size:10px'>{nc:,} cycles</span>"
        else:
            text_str = f"<b>{nv:.2f}× more damage</b><br><span style='font-size:10px'>{nc:,} cycles</span>"

        fig.add_trace(
            go.Bar(
                name=label,
                x=[label],
                y=[nv],
                marker_color=color,
                text=[text_str],
                textposition="auto",
                # Raw absolute in tooltip only
                hovertemplate=(
                    f"{label}<br>"
                    f"Normalised: {nv:.4f}×<br>"
                    f"Relative Damage Index: {av:.3e} (arb. units, m={m})<br>"
                    f"Cycles: {nc:,}<extra></extra>"
                ),
            ),
            row=1, col=1,
        )

    # ── Right: P95 range bars ───────────────────────────────────────
    p95_front = front.get("p95_range", 0.0)
    p95_rear  = rear.get("p95_range",  0.0)

    for label, color, p95 in zip(
        ["Front (FL+FR)", "Rear (RL+RR)"],
        ["#2E86DE", "#E74C3C"],
        [p95_front, p95_rear],
    ):
        ratio_note = ""
        if label.startswith("Rear") and p95_ratio is not None:
            ratio_note = f"<br>{p95_ratio:.2f}× Front"
        fig.add_trace(
            go.Bar(
                name=label,
                x=[label],
                y=[p95],
                marker_color=color,
                text=[f"{p95:.1f}{ratio_note}"],
                textposition="auto",
                hovertemplate=f"{label}<br>P95 range = %{{y:.2f}} {unit}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=2,
        )

    # Title with the ratio front and centre
    ratio_text = f"— Rear/Front = {ratio:.2f}×" if ratio is not None else ""
    _dark_layout(
        fig,
        f"Axle Damage Comparison — {suffix}  (m={m})  {ratio_text}",
        height=420,
    )
    fig.update_yaxes(
        title_text="Normalised Damage Index (Front = 1.00)",
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text=f"P95 Range ({unit})",
        row=1, col=2,
    )
    return fig


# ─── §6.5 Box Plots & Distance Distribution ───────────────────────────────

def plot_box_all_wheels(
    box_results: dict,
    suffix: str,
    wheels: list[str],
    display_unit: str = "daN",
) -> go.Figure:
    """
    Box plot comparing all selected wheels for one force suffix.
    """
    fig = go.Figure()
    for wheel in wheels:
        key = f"{wheel}_{suffix}"
        res = box_results.get(key)
        if res is None:
            continue
        color = WHEEL_COLORS.get(wheel, "#AAAAAA")
        fig.add_trace(go.Box(
            y=res["values"],
            name=wheel,
            marker_color=color,
            boxmean=True,
            boxpoints=False,
            width=0.4,
            line=dict(color=color, width=2),
        ))

    _dark_layout(
        fig,
        f"Box Plot \u2014 {suffix} ({SUFFIX_NAMES.get(suffix, suffix)}) [{display_unit}]",
        height=480,
    )
    fig.update_layout(boxmode="group", boxgap=0.3)
    fig.update_yaxes(title_text=f"Force ({display_unit})")
    return fig


def plot_distance_distribution(
    dist_results: dict,
    wheels: list[str],
    suffix: str,
    display_unit: str = "daN",
) -> go.Figure:
    """
    Distance-distribution histogram for one force suffix, all selected wheels.
    """
    fig = go.Figure()
    global_view_range = None

    for wheel in wheels:
        key = f"{wheel}_{suffix}"
        res = dist_results.get(key)
        if res is None:
            continue
        
        if global_view_range is None and res.get("view_range"):
            global_view_range = res["view_range"]

        color = WHEEL_COLORS.get(wheel, "#AAAAAA")
        # Use a step outline with light fill instead of dense bars
        centers = res["bin_centers"]
        counts = res["hist_dist_km"]
        if len(centers) > 0:
            step_x = np.repeat(centers, 2)[1:]
            step_y = np.repeat(counts, 2)[:-1]
            # duplicate edges to make it close correctly at ends
            step_x = np.concatenate(([centers[0]], step_x, [centers[-1]]))
            step_y = np.concatenate(([0], step_y, [0]))

            fig.add_trace(go.Scatter(
                x=step_x,
                y=step_y,
                mode="lines",
                name=f"{wheel} ({res['total_dist_km']:.1f} km)",
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=_hex_to_rgba(color, 0.15),
                opacity=0.85,
                line_shape="linear", # Since we built step coords manually
            ))

    _dark_layout(
        fig,
        f"Distance Distribution \u2014 {suffix}  ({display_unit})",
        height=450,
    )
    fig.update_xaxes(title_text=f"Force ({display_unit})")
    if global_view_range:
        fig.update_xaxes(range=global_view_range)
    fig.update_yaxes(title_text="Distance (km)")
    return fig
