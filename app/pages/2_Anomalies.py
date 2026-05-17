from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_APP = _ROOT / "app"
for _p in (str(_ROOT / "src"), str(_APP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from ui import page_hero, section_label, set_page_config  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────

set_page_config(title="Anomaly Detection")

page_hero(
    title="🚨 Anomaly Detection & Alerts",
    subtitle="Land Surface Temperature anomalies detected via z-score analysis against a 30-day rolling baseline. Filter by site, date, or severity.",
    pills=["🌡️ LST z-Score", "📅 180-Day Window", "🔴 3 Severity Levels", "📍 8 Sites"],
    emoji="🚨",
)

# ── Data loading ──────────────────────────────────────────────────────────────

MOCK = _ROOT / "data" / "mock"


@st.cache_data
def _load_timeseries() -> pd.DataFrame:
    return pd.read_csv(MOCK / "lst_timeseries.csv", parse_dates=["date"])


@st.cache_data
def _load_anomalies() -> pd.DataFrame:
    return pd.read_csv(MOCK / "anomalies.csv", parse_dates=["date"])


ts_df = _load_timeseries()
anom_df = _load_anomalies()

SITES = ["All sites"] + sorted(ts_df["site"].unique())
SEV_COLOR = {"Critical": "#DC2626", "Severe": "#EA580C", "Mild": "#CA8A04"}
SEV_BG = {"Critical": "#FEE2E2", "Severe": "#FFEDD5", "Mild": "#FEF9C3"}
STATUS_COLOR = {"New": "#2E7D32", "Reviewed": "#1976D2", "Handled": "#6B7280"}

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── KPI strip ── */
    .kpi-row { display: flex; gap: 14px; margin-bottom: 4px; flex-wrap: wrap; }
    .kpi-card {
      flex: 1; min-width: 110px;
      background: #fff; border: 1px solid #e5e7eb;
      border-radius: 12px; padding: 18px 16px; text-align: center;
      box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .kpi-label { font-size: 0.65em; font-weight: 700; letter-spacing: 0.1em;
                 text-transform: uppercase; color: #6b7280; margin-bottom: 6px; }
    .kpi-value { font-size: 2em; font-weight: 800; color: #1C1B18; line-height: 1; }
    .kpi-sub   { font-size: 0.72em; color: #6b7280; margin-top: 4px; }

    /* ── anomaly table ── */
    .anom-table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
    .anom-scroll {
      max-height: 420px; overflow-y: auto;
      border: 1px solid #e5e7eb; border-radius: 10px;
    }
    .anom-table { border-radius: 0; }
    .anom-table thead th { position: sticky; top: 0; z-index: 1; }
    .anom-table th {
      background: #F0FDF4; color: #166534;
      font-weight: 700; font-size: 0.70em; letter-spacing: 0.07em;
      text-transform: uppercase; padding: 9px 12px;
      border-bottom: 2px solid #BBF7D0; text-align: left;
    }
    .anom-table td { padding: 9px 12px; border-bottom: 1px solid #F3F4F6;
                     color: #374151; vertical-align: middle; }
    .anom-table tr:hover td { background: #F9FAFB; }
    .sev-badge {
      display: inline-block; border-radius: 999px;
      padding: 2px 10px; font-size: 0.75em; font-weight: 700;
    }
    .status-dot {
      display: inline-block; width: 8px; height: 8px;
      border-radius: 50%; margin-right: 5px; vertical-align: middle;
    }

    /* ── explanation panel ── */
    .explain-panel {
      background: linear-gradient(135deg, #E8F5E9 0%, #F1F8E9 100%);
      border: 1px solid #A5D6A7; border-radius: 14px;
      padding: 24px 26px;
    }
    .explain-header {
      font-size: 0.68em; font-weight: 700; letter-spacing: 0.1em;
      text-transform: uppercase; color: #2E7D32; margin-bottom: 16px;
    }
    .explain-stat-row { display: flex; gap: 12px; margin-bottom: 18px; flex-wrap: wrap; }
    .explain-stat {
      flex: 1; min-width: 80px; background: #fff;
      border: 1px solid #C8E6C9; border-radius: 10px;
      padding: 12px 10px; text-align: center;
    }
    .es-label { font-size: 0.62em; color: #6b7280; font-weight: 600;
                text-transform: uppercase; letter-spacing: 0.06em; }
    .es-value { font-size: 1.25em; font-weight: 800; color: #1B5E20;
                margin: 4px 0; line-height: 1; }
    .explain-body { font-size: 0.87em; color: #374151; line-height: 1.75; }
    .explain-body p { margin: 0 0 0.7em; }
    .explain-body strong { color: #2E7D32; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Filters ───────────────────────────────────────────────────────────────────

section_label("Filters")

col_site, col_sev, col_status, col_date = st.columns([2, 1.5, 1.5, 2])
with col_site:
    site_filter = st.selectbox("Site", SITES)
with col_sev:
    sev_filter = st.selectbox("Severity", ["All", "Critical", "Severe", "Mild"])
with col_status:
    status_filter = st.selectbox("Status", ["All", "New", "Reviewed", "Handled"])
with col_date:
    min_d = anom_df["date"].min().date()
    max_d = anom_df["date"].max().date()
    date_range = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

# Apply filters
filtered = anom_df.copy()
if site_filter != "All sites":
    filtered = filtered[filtered["site"] == site_filter]
if sev_filter != "All":
    filtered = filtered[filtered["severity"] == sev_filter]
if status_filter != "All":
    filtered = filtered[filtered["status"] == status_filter]
if isinstance(date_range, tuple) and len(date_range) == 2:
    d0 = pd.Timestamp(date_range[0])
    d1 = pd.Timestamp(date_range[1])
    filtered = filtered[(filtered["date"] >= d0) & (filtered["date"] <= d1)]

filtered = filtered.sort_values("date", ascending=False).reset_index(drop=True)

# ── KPI strip ─────────────────────────────────────────────────────────────────

section_label("Summary")

total = len(filtered)
n_critical = (filtered["severity"] == "Critical").sum()
n_severe = (filtered["severity"] == "Severe").sum()
n_new = (filtered["status"] == "New").sum()
avg_z = filtered["z_score"].mean() if total else 0.0
max_z = filtered["z_score"].max() if total else 0.0

st.markdown(
    f"""
    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-label">Total Anomalies</div>
        <div class="kpi-value">{total}</div>
        <div class="kpi-sub">filtered events</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Critical</div>
        <div class="kpi-value" style="color:#DC2626">{n_critical}</div>
        <div class="kpi-sub">z ≥ 3σ</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Severe</div>
        <div class="kpi-value" style="color:#EA580C">{n_severe}</div>
        <div class="kpi-sub">2σ ≤ z &lt; 3σ</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Needs Review</div>
        <div class="kpi-value" style="color:#2E7D32">{n_new}</div>
        <div class="kpi-sub">status = New</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Avg z-Score</div>
        <div class="kpi-value">{avg_z:.2f}</div>
        <div class="kpi-sub">σ above baseline</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Peak z-Score</div>
        <div class="kpi-value">{max_z:.2f}</div>
        <div class="kpi-sub">worst event</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Main two-column layout ────────────────────────────────────────────────────

st.write("")
col_table, col_explain = st.columns([3, 2], gap="large")

# ── Left: anomaly table + drill-down selector ─────────────────────────────────

with col_table:
    section_label(f"Anomaly Events ({total})")

    if total == 0:
        st.info("No anomalies match the current filters.")
        selected_idx = None
    else:
        rows_html = ""
        for _, row in filtered.head(80).iterrows():
            sev = row["severity"]
            status = row["status"]
            bg = SEV_BG.get(sev, "#fff")
            fc = SEV_COLOR.get(sev, "#374151")
            sc = STATUS_COLOR.get(status, "#6B7280")
            ndvi = f"{row['ndvi_change']:+.3f}"
            rows_html += (
                f"<tr>"
                f'<td>{row["date"].strftime("%b %d, %Y")}</td>'
                f'<td>{row["site"]}</td>'
                f'<td>{row["lst"]:.1f}°C</td>'
                f'<td>{row["baseline"]:.1f}°C</td>'
                f"<td><b>{row['z_score']:.2f}σ</b></td>"
                f'<td><span class="sev-badge" style="background:{bg};color:{fc}">{sev}</span></td>'
                f'<td><span class="status-dot" style="background:{sc}"></span>{status}</td>'
                f'<td style="color:#DC2626">{ndvi}</td>'
                "</tr>"
            )

        st.markdown(
            f"""
            <div class="anom-scroll">
              <table class="anom-table">
                <thead>
                  <tr>
                    <th>Date</th><th>Site</th><th>LST</th><th>Baseline</th>
                    <th>z-Score</th><th>Severity</th><th>Status</th><th>ΔNDVI</th>
                  </tr>
                </thead>
                <tbody>{rows_html}</tbody>
              </table>
            </div>
            {"<p style='font-size:0.75em;color:#9ca3af;margin-top:8px'>Showing top 80 events</p>" if total > 80 else ""}
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        section_label("Drill-down")
        event_labels = [
            f"{row['date'].strftime('%b %d')} · {row['site']} · {row['severity']} ({row['z_score']:.2f}σ)"
            for _, row in filtered.iterrows()
        ]
        selected_idx = st.selectbox(
            "Select anomaly to inspect",
            range(len(event_labels)),
            format_func=lambda i: event_labels[i],
        )

# ── Right: explanation panel ──────────────────────────────────────────────────

with col_explain:
    section_label("Event Details")

    if total == 0 or selected_idx is None:
        st.markdown(
            '<div class="explain-panel">'
            '<div style="text-align:center;padding:40px 0;color:#9ca3af;font-size:0.88em;line-height:1.7">'
            "No anomaly selected.<br>Adjust filters to see events."
            "</div></div>",
            unsafe_allow_html=True,
        )
    else:
        sel = filtered.iloc[selected_idx]
        sev = sel["severity"]
        status = sel["status"]
        bg = SEV_BG.get(sev, "#fff")
        fc = SEV_COLOR.get(sev, "#374151")
        sc = STATUS_COLOR.get(status, "#6B7280")
        delta_t = sel["lst"] - sel["baseline"]

        if sev == "Critical":
            narrative = (
                f"A <strong>critical heat anomaly</strong> was recorded at <strong>{sel['site']}</strong> "
                f"on {sel['date'].strftime('%B %d, %Y')}. The land surface temperature reached "
                f"<strong>{sel['lst']:.1f}°C</strong>, exceeding the 30-day baseline by "
                f"<strong>+{delta_t:.1f}°C</strong> ({sel['z_score']:.2f}σ)."
                "<p>At this severity, prolonged heat stress may trigger vegetation die-off and soil "
                "moisture depletion. An NDVI drop of "
                f"<strong>{sel['ndvi_change']:+.3f}</strong> corroborates surface degradation. "
                "Immediate field inspection is recommended.</p>"
            )
        elif sev == "Severe":
            narrative = (
                f"A <strong>severe temperature spike</strong> was detected at <strong>{sel['site']}</strong> "
                f"on {sel['date'].strftime('%B %d, %Y')}. LST hit <strong>{sel['lst']:.1f}°C</strong>, "
                f"<strong>+{delta_t:.1f}°C</strong> above the rolling mean ({sel['z_score']:.2f}σ)."
                "<p>Severe anomalies at this site have historically coincided with reduced canopy "
                "reflectance. The concurrent NDVI change of "
                f"<strong>{sel['ndvi_change']:+.3f}</strong> suggests early-stage stress. "
                "Schedule a follow-up satellite pass within 7 days.</p>"
            )
        else:
            narrative = (
                f"A <strong>mild temperature anomaly</strong> occurred at <strong>{sel['site']}</strong> "
                f"on {sel['date'].strftime('%B %d, %Y')}. Observed LST: <strong>{sel['lst']:.1f}°C</strong> "
                f"(+{delta_t:.1f}°C, {sel['z_score']:.2f}σ above baseline)."
                "<p>Mild anomalies are common during dry spells and typically resolve within 3–5 days. "
                f"The NDVI change of <strong>{sel['ndvi_change']:+.3f}</strong> is within normal seasonal "
                "variation. Continue monitoring; escalate if sustained over multiple consecutive days.</p>"
            )

        st.markdown(
            f"""
            <div class="explain-panel">
              <div class="explain-header">⚡ Anomaly Explanation</div>
              <div class="explain-stat-row">
                <div class="explain-stat">
                  <div class="es-label">LST</div>
                  <div class="es-value">{sel["lst"]:.1f}°</div>
                </div>
                <div class="explain-stat">
                  <div class="es-label">Baseline</div>
                  <div class="es-value">{sel["baseline"]:.1f}°</div>
                </div>
                <div class="explain-stat">
                  <div class="es-label">z-Score</div>
                  <div class="es-value">{sel["z_score"]:.2f}σ</div>
                </div>
                <div class="explain-stat">
                  <div class="es-label">ΔNDVI</div>
                  <div class="es-value" style="color:#DC2626">{sel["ndvi_change"]:+.3f}</div>
                </div>
              </div>
              <div style="margin-bottom:14px">
                <span class="sev-badge" style="background:{bg};color:{fc}">{sev}</span>
                &nbsp;
                <span style="font-size:0.75em;color:{sc};font-weight:600">
                  <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
                         background:{sc};margin-right:4px;vertical-align:middle"></span>
                  {status}
                </span>
              </div>
              <div class="explain-body"><p>{narrative}</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── LST time-series chart ─────────────────────────────────────────────────────

st.write("")

# Pick chart site: selected event's site, or site filter, or first available
if total > 0 and selected_idx is not None:
    chart_site = filtered.iloc[selected_idx]["site"]
elif site_filter != "All sites":
    chart_site = site_filter
else:
    chart_site = sorted(ts_df["site"].unique())[0]

section_label(f"LST Time-Series with Anomalies — {chart_site}")

site_ts = (
    ts_df[ts_df["site"] == chart_site]
    .sort_values("date")
    .dropna(subset=["baseline_mean", "baseline_std"])
)
site_anom = anom_df[anom_df["site"] == chart_site]

fig = go.Figure()

# ±2σ band
fig.add_trace(
    go.Scatter(
        x=pd.concat([site_ts["date"], site_ts["date"][::-1]]),
        y=pd.concat([
            site_ts["baseline_mean"] + 2 * site_ts["baseline_std"],
            (site_ts["baseline_mean"] - 2 * site_ts["baseline_std"])[::-1],
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
        x=pd.concat([site_ts["date"], site_ts["date"][::-1]]),
        y=pd.concat([
            site_ts["baseline_mean"] + site_ts["baseline_std"],
            (site_ts["baseline_mean"] - site_ts["baseline_std"])[::-1],
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
        x=site_ts["date"],
        y=site_ts["baseline_mean"],
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
            hovertemplate="%{x|%b %d}<br>LST: %{y:.1f}°C<extra>" + sev_name + "</extra>",
        )
    )

# Selected event star
if total > 0 and selected_idx is not None:
    ev = filtered.iloc[selected_idx]
    if ev["site"] == chart_site:
        fig.add_trace(
            go.Scatter(
                x=[ev["date"]],
                y=[ev["lst"]],
                name="Selected",
                mode="markers",
                marker={
                    "color": "#7C3AED",
                    "size": 14,
                    "symbol": "star",
                    "line": {"color": "#fff", "width": 2},
                },
                hovertemplate="%{x|%b %d}<br>LST: %{y:.1f}°C<extra>Selected</extra>",
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
    xaxis={"showgrid": True, "gridcolor": "#F3F4F6", "tickformat": "%b %d", "title": None},
    yaxis={
        "showgrid": True,
        "gridcolor": "#F3F4F6",
        "title": {"text": "LST (°C)", "font": {"size": 12}},
    },
)

st.plotly_chart(fig, use_container_width=True)
st.caption(
    f"180-day LST record for {chart_site}. "
    "Coloured markers = detected anomalies (red=Critical, orange=Severe, yellow=Mild). "
    "★ = currently selected event."
)
