"""
Plotly chart helpers for the Anomalies and Optimization pages.

All functions return go.Figure objects and have no Streamlit dependency.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# ── Severity / status palette ──────────────────────────────────────────────────

# "Warning" is what the stats module produces; "Mild" is the legacy mock label.
# Both map to the same yellow palette so the page works with either source.
SEV_COLOR: dict[str, str] = {
    "Critical": "#DC2626",
    "Severe": "#EA580C",
    "Warning": "#CA8A04",
    "Mild": "#CA8A04",
}
SEV_BG: dict[str, str] = {
    "Critical": "#FEE2E2",
    "Severe": "#FFEDD5",
    "Warning": "#FEF9C3",
    "Mild": "#FEF9C3",
}
STATUS_COLOR: dict[str, str] = {
    "New": "#2E7D32",
    "Reviewed": "#1976D2",
    "Handled": "#6B7280",
}


# ── Chart ─────────────────────────────────────────────────────────────────────


def anomaly_timeseries_chart(
    site_ts: pd.DataFrame,
    site_anom: pd.DataFrame,
    selected_event: pd.Series | None = None,
) -> go.Figure:
    """
    Build a Plotly LST time-series chart with ±1σ/±2σ bands and anomaly markers.

    Parameters
    ----------
    site_ts : pd.DataFrame
        Time-series for a single site.
        Required columns: ``date`` (datetime), ``lst``, ``baseline_mean``, ``baseline_std``.
    site_anom : pd.DataFrame
        Anomaly rows for the same site.
        Required columns: ``date`` (datetime), ``lst``, ``severity``.
    selected_event : pd.Series | None
        A single anomaly row to highlight with a star marker.

    Returns
    -------
    go.Figure
    """
    ts = site_ts.dropna(subset=["baseline_mean", "baseline_std"])

    fig = go.Figure()

    if not ts.empty:
        # ±2σ band
        fig.add_trace(
            go.Scatter(
                x=pd.concat([ts["date"], ts["date"][::-1]]),
                y=pd.concat([
                    ts["baseline_mean"] + 2 * ts["baseline_std"],
                    (ts["baseline_mean"] - 2 * ts["baseline_std"])[::-1],
                ]),
                fill="toself",
                fillcolor="rgba(46,125,50,0.07)",
                line={"width": 0},
                showlegend=True,
                name="±2σ band",
                hoverinfo="skip",
            )
        )

        # ±1σ band
        fig.add_trace(
            go.Scatter(
                x=pd.concat([ts["date"], ts["date"][::-1]]),
                y=pd.concat([
                    ts["baseline_mean"] + ts["baseline_std"],
                    (ts["baseline_mean"] - ts["baseline_std"])[::-1],
                ]),
                fill="toself",
                fillcolor="rgba(46,125,50,0.13)",
                line={"width": 0},
                showlegend=True,
                name="±1σ band",
                hoverinfo="skip",
            )
        )

        # Baseline mean
        fig.add_trace(
            go.Scatter(
                x=ts["date"],
                y=ts["baseline_mean"],
                name="Baseline mean",
                line={"color": "#2E7D32", "width": 1.8, "dash": "dot"},
                mode="lines",
            )
        )

    # Actual LST
    fig.add_trace(
        go.Scatter(
            x=site_ts["date"],
            y=site_ts["lst"],
            name="LST",
            line={"color": "#9CA3AF", "width": 1.8},
            mode="lines",
        )
    )

    # Anomaly markers per severity
    for sev_name, color in SEV_COLOR.items():
        if sev_name == "Mild":
            continue  # deduplicate — Mild == Warning visually
        pts = site_anom[site_anom["severity"] == sev_name]
        if pts.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=pts["date"],
                y=pts["lst"],
                name=sev_name,
                mode="markers",
                marker={
                    "color": color,
                    "size": 9,
                    "symbol": "circle",
                    "line": {"color": "#fff", "width": 1.5},
                },
                hovertemplate="%{x|%b %Y}<br>LST: %{y:.1f}°C<extra>" + sev_name + "</extra>",
            )
        )

    # Selected event star
    if selected_event is not None:
        fig.add_trace(
            go.Scatter(
                x=[selected_event["date"]],
                y=[selected_event["lst"]],
                name="Selected",
                mode="markers",
                marker={
                    "color": "#7C3AED",
                    "size": 14,
                    "symbol": "star",
                    "line": {"color": "#fff", "width": 2},
                },
                hovertemplate="%{x|%b %Y}<br>LST: %{y:.1f}°C<extra>Selected</extra>",
            )
        )

    fig.update_layout(
        height=380,
        margin={"t": 20, "b": 20, "l": 0, "r": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FAFAFA",
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        xaxis={
            "showgrid": True,
            "gridcolor": "#F3F4F6",
            "tickformat": "%Y",
            "title": None,
        },
        yaxis={
            "showgrid": True,
            "gridcolor": "#F3F4F6",
            "title": {"text": "LST (°C)", "font": {"size": 12}},
        },
    )
    return fig


def convergence_chart(history: list[float]) -> go.Figure:
    """
    Build a Plotly chart showing GA fitness convergence across generations.

    Parameters
    ----------
    history : list[float]
        Best fitness value at the end of each generation
        (monotonically non-decreasing).

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(history) + 1)),
            y=history,
            mode="lines",
            line={"color": "#2E7D32", "width": 2.5},
            fill="tozeroy",
            fillcolor="rgba(46,125,50,0.07)",
            hovertemplate="Gen %{x}<br>Fitness: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=340,
        margin={"t": 10, "b": 30, "l": 0, "r": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FAFAFA",
        xaxis={
            "showgrid": True,
            "gridcolor": "#F3F4F6",
            "title": {"text": "Generation", "font": {"size": 11}},
        },
        yaxis={
            "showgrid": True,
            "gridcolor": "#F3F4F6",
            "title": {"text": "Best fitness", "font": {"size": 11}},
        },
    )
    return fig


def logistic_growth_chart(df: pd.DataFrame, K: float) -> go.Figure:
    """
    Plotly chart of a logistic growth trajectory with carrying-capacity line.

    Parameters
    ----------
    df : pd.DataFrame
        Output of LogisticGrowth.simulate() — columns: t, population.
    K : float
        Carrying capacity, drawn as a dashed reference line.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    # Carrying capacity reference
    fig.add_hline(
        y=K,
        line={"color": "#2E7D32", "dash": "dash", "width": 1.5},
        annotation_text=f"K = {K:.2f}",
        annotation_position="right",
        annotation_font={"size": 10, "color": "#2E7D32"},
    )

    # Growth curve
    fig.add_trace(
        go.Scatter(
            x=df["t"],
            y=df["population"],
            mode="lines",
            name="Vegetation cover",
            line={"color": "#16A34A", "width": 2.5},
            fill="tozeroy",
            fillcolor="rgba(22,163,74,0.08)",
            hovertemplate="t = %{x:.1f}<br>Cover: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        height=280,
        margin={"t": 10, "b": 30, "l": 0, "r": 60},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FAFAFA",
        showlegend=False,
        xaxis={
            "showgrid": True,
            "gridcolor": "#F3F4F6",
            "title": {"text": "Time (years)", "font": {"size": 11}},
        },
        yaxis={
            "showgrid": True,
            "gridcolor": "#F3F4F6",
            "title": {"text": "Vegetation cover (NDVI proxy)", "font": {"size": 11}},
            "range": [0, K * 1.15],
        },
    )
    return fig


def energy_flow_chart(
    df: pd.DataFrame,
    df_planted: pd.DataFrame | None = None,
) -> go.Figure:
    """
    Plotly chart of the EnergyFlow heat-budget model trajectory.

    Parameters
    ----------
    df : pd.DataFrame
        Output of EnergyFlow.simulate() — columns: t, T, T_eq, Q_in.
    df_planted : pd.DataFrame | None
        Optional second trajectory (e.g. with higher NDVI after planting)
        to overlay for comparison.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    # Equilibrium envelope (shaded)
    fig.add_trace(
        go.Scatter(
            x=df["t"],
            y=df["T_eq"],
            mode="lines",
            name="Equilibrium T",
            line={"color": "#9CA3AF", "width": 1.2, "dash": "dot"},
            hovertemplate="Day %{x:.0f}<br>T_eq: %{y:.1f}°C<extra>Equilibrium</extra>",
        )
    )

    # Current surface temperature
    fig.add_trace(
        go.Scatter(
            x=df["t"],
            y=df["T"],
            mode="lines",
            name="Surface T (current)",
            line={"color": "#DC2626", "width": 2.2},
            hovertemplate="Day %{x:.0f}<br>T: %{y:.1f}°C<extra>Current</extra>",
        )
    )

    # After-planting scenario
    if df_planted is not None:
        fig.add_trace(
            go.Scatter(
                x=df_planted["t"],
                y=df_planted["T"],
                mode="lines",
                name="Surface T (after planting)",
                line={"color": "#2E7D32", "width": 2.2, "dash": "dash"},
                hovertemplate="Day %{x:.0f}<br>T: %{y:.1f}°C<extra>After planting</extra>",
            )
        )

    fig.update_layout(
        height=280,
        margin={"t": 10, "b": 30, "l": 0, "r": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FAFAFA",
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 11},
        },
        xaxis={
            "showgrid": True,
            "gridcolor": "#F3F4F6",
            "title": {"text": "Day of year", "font": {"size": 11}},
        },
        yaxis={
            "showgrid": True,
            "gridcolor": "#F3F4F6",
            "title": {"text": "Surface temperature (°C)", "font": {"size": 11}},
        },
    )
    return fig
