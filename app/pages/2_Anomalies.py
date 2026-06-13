from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_APP = _ROOT / "app"
for _p in (str(_ROOT / "src"), str(_APP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from ui import page_hero, section_label, set_page_config  # noqa: E402

from data_nature.viz.charts import (  # noqa: E402
    SEV_BG, SEV_COLOR, STATUS_COLOR, anomaly_timeseries_chart,
)

# ── Page config ───────────────────────────────────────────────────────────────

set_page_config(title="Anomaly Detection")

page_hero(
    title="🚨 Anomaly Detection & Alerts",
    subtitle="Land Surface Temperature anomalies detected via z-score analysis against a historical per-site baseline. Filter by site, date, or severity.",
    pills=["🌡️ LST z-Score", "📅 2000–2025", "🔴 3 Severity Levels", "📍 8 Sites"],
    emoji="🚨",
)

# ── Data loading ──────────────────────────────────────────────────────────────

_PROCESSED = _ROOT / "data" / "processed"
_MOCK = _ROOT / "data" / "mock"


def _build_from_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build (ts_df, anom_df) from real processed CSVs.

    ts_df  — date, site, lst, baseline_mean, baseline_std
    anom_df — date, site, lst, baseline, z_score, severity, status, ndvi_change
    """
    from data_nature.stats import compute_zscores, detect_anomalies

    monthly = pd.read_csv(_PROCESSED / "site_monthly.csv")
    baselines = pd.read_csv(_PROCESSED / "site_baselines.csv")

    # Build date column (first of month)
    monthly["date"] = pd.to_datetime(
        dict(year=monthly["year"], month=monthly["month"], day=1)
    )

    # Timeseries: merge monthly LST with leave-one-out baselines
    ts_df = (
        monthly[["date", "site", "year", "month", "lst"]]
        .merge(
            baselines[["site", "year", "month", "baseline_mean", "baseline_std"]],
            on=["site", "year", "month"],
            how="left",
        )
        .drop(columns=["year", "month"])
        .sort_values(["site", "date"])
        .reset_index(drop=True)
    )

    # Anomaly detection using the stats module
    enriched = compute_zscores(monthly)
    # detect_anomalies needs date, site, lst, ndvi, month columns
    anom_df = detect_anomalies(
        enriched[["date", "site", "lst", "ndvi", "month", "year"]],
    )
    anom_df["date"] = pd.to_datetime(anom_df["date"])

    return ts_df, anom_df


@st.cache_data(show_spinner="Loading anomaly data…")
def _load() -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Return (ts_df, anom_df, using_real_data)."""
    try:
        ts_df, anom_df = _build_from_processed()
        return ts_df, anom_df, True
    except Exception:
        pass

    ts_df = pd.read_csv(_MOCK / "lst_timeseries.csv", parse_dates=["date"])
    anom_df = pd.read_csv(_MOCK / "anomalies.csv", parse_dates=["date"])
    return ts_df, anom_df, False


ts_df, anom_df, _REAL_DATA = _load()

SITES = ["All sites"] + sorted(ts_df["site"].unique())

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
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

# ── Status banner ─────────────────────────────────────────────────────────────

if not _REAL_DATA:
    st.info("ℹ️ Showing mock data — place real CSVs in `data/processed/` to use live data.")

# ── Filters ───────────────────────────────────────────────────────────────────

section_label("Filters")

# Severity labels present in data — include both "Warning" (real) and "Mild" (mock)
_sev_opts = ["All", "Critical", "Severe", "Warning", "Mild"]
_sev_present = [s for s in _sev_opts if s == "All" or s in anom_df["severity"].unique()]

col_site, col_sev, col_status, col_date = st.columns([2, 1.5, 1.5, 2])
with col_site:
    site_filter = st.selectbox("Site", SITES)
with col_sev:
    sev_filter = st.selectbox("Severity", _sev_present)
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
        <div class="kpi-sub">z ≥ 3.5σ</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Severe</div>
        <div class="kpi-value" style="color:#EA580C">{n_severe}</div>
        <div class="kpi-sub">2.5σ ≤ z &lt; 3.5σ</div>
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
            ndvi_val = row.get("ndvi_change", float("nan"))
            ndvi_str = f"{ndvi_val:+.3f}" if pd.notna(ndvi_val) else "—"
            rows_html += (
                f"<tr>"
                f'<td>{row["date"].strftime("%b %Y")}</td>'
                f'<td>{row["site"]}</td>'
                f'<td>{row["lst"]:.1f}°C</td>'
                f'<td>{row["baseline"]:.1f}°C</td>'
                f"<td><b>{row['z_score']:.2f}σ</b></td>"
                f'<td><span class="sev-badge" style="background:{bg};color:{fc}">{sev}</span></td>'
                f'<td><span class="status-dot" style="background:{sc}"></span>{status}</td>'
                f'<td style="color:#DC2626">{ndvi_str}</td>'
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
            f"{row['date'].strftime('%b %Y')} · {row['site']} · {row['severity']} ({row['z_score']:.2f}σ)"
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
        ndvi_change = sel.get("ndvi_change", float("nan"))
        ndvi_str = f"{ndvi_change:+.3f}" if pd.notna(ndvi_change) else "—"

        if sev == "Critical":
            narrative = (
                f"A <strong>critical heat anomaly</strong> was recorded at <strong>{sel['site']}</strong> "
                f"in {sel['date'].strftime('%B %Y')}. The land surface temperature reached "
                f"<strong>{sel['lst']:.1f}°C</strong>, exceeding the historical baseline by "
                f"<strong>+{delta_t:.1f}°C</strong> ({sel['z_score']:.2f}σ)."
                "<p>At this severity, prolonged heat stress may trigger vegetation die-off and soil "
                f"moisture depletion. NDVI change: <strong>{ndvi_str}</strong>. "
                "Immediate field inspection is recommended.</p>"
            )
        elif sev == "Severe":
            narrative = (
                f"A <strong>severe temperature spike</strong> was detected at <strong>{sel['site']}</strong> "
                f"in {sel['date'].strftime('%B %Y')}. LST hit <strong>{sel['lst']:.1f}°C</strong>, "
                f"<strong>+{delta_t:.1f}°C</strong> above the historical mean ({sel['z_score']:.2f}σ)."
                "<p>Severe anomalies at this site have historically coincided with reduced canopy "
                f"reflectance. NDVI change: <strong>{ndvi_str}</strong>. "
                "Schedule a follow-up satellite pass within 7 days.</p>"
            )
        else:
            narrative = (
                f"A <strong>warning-level temperature anomaly</strong> occurred at <strong>{sel['site']}</strong> "
                f"in {sel['date'].strftime('%B %Y')}. Observed LST: <strong>{sel['lst']:.1f}°C</strong> "
                f"(+{delta_t:.1f}°C, {sel['z_score']:.2f}σ above baseline)."
                "<p>Warning anomalies are common during dry spells and typically resolve within one month. "
                f"NDVI change: <strong>{ndvi_str}</strong>. Continue monitoring; escalate if sustained.</p>"
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
                  <div class="es-value" style="color:#DC2626">{ndvi_str}</div>
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

if total > 0 and selected_idx is not None:
    chart_site = filtered.iloc[selected_idx]["site"]
elif site_filter != "All sites":
    chart_site = site_filter
else:
    chart_site = sorted(ts_df["site"].unique())[0]

section_label(f"LST Time-Series with Anomalies — {chart_site}")

site_ts = ts_df[ts_df["site"] == chart_site].sort_values("date").copy()
site_ts["date"] = pd.to_datetime(site_ts["date"])
site_anom = anom_df[anom_df["site"] == chart_site].copy()
site_anom["date"] = pd.to_datetime(site_anom["date"])

sel_event = None
if total > 0 and selected_idx is not None:
    ev = filtered.iloc[selected_idx]
    if ev["site"] == chart_site:
        sel_event = ev

fig = anomaly_timeseries_chart(site_ts, site_anom, selected_event=sel_event)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    f"Monthly LST record for **{chart_site}** (2000–2025). "
    "Coloured markers = detected anomalies (red=Critical, orange=Severe, yellow=Warning). "
    "★ = currently selected event. Shaded bands show ±1σ / ±2σ historical range."
)

# ── Energy-Flow / Heat-Budget Model (DN-A6) ───────────────────────────────────

st.write("")
section_label(f"Heat-Budget Model — {chart_site}")

from data_nature.models.ecology import EnergyFlow  # noqa: E402
from data_nature.viz.charts import energy_flow_chart  # noqa: E402

# Derive site-level defaults from the real data
_site_rows = ts_df[ts_df["site"] == chart_site]
_site_lst  = float(_site_rows["lst"].mean()) if not _site_rows.empty else 35.0

# Load monthly data (cached) to derive site NDVI
if "_monthly_df_cache" not in st.session_state:
    try:
        _m = pd.read_csv(_PROCESSED / "site_monthly.csv")
    except Exception:
        _m = pd.read_csv(_MOCK / "site_monthly.csv")
    st.session_state["_monthly_df_cache"] = _m
_monthly_all = st.session_state["_monthly_df_cache"]
_site_ndvi = float(
    _monthly_all[_monthly_all["site"] == chart_site]["ndvi"].mean()
    if chart_site in _monthly_all["site"].values else 0.4
)

ef_c1, ef_c2, ef_c3, ef_c4 = st.columns([1.5, 1.5, 1.5, 1.5])
with ef_c1:
    ef_T0 = st.slider(
        "Initial surface T (°C)", 20.0, 55.0,
        value=round(min(max(_site_lst, 20.0), 55.0), 1),
        step=0.5, key="ef_T0",
        help="Starting surface temperature for this site (from real data mean).",
    )
with ef_c2:
    ef_ndvi = st.slider(
        "Current NDVI", 0.05, 0.90,
        value=round(min(max(_site_ndvi, 0.05), 0.90), 2),
        step=0.05, key="ef_ndvi",
        help="Current vegetation index. Higher NDVI → stronger evapotranspirative cooling.",
    )
with ef_c3:
    ef_ndvi_plant = st.slider(
        "NDVI after planting", 0.05, 0.95,
        value=round(min(max(_site_ndvi + 0.20, 0.05), 0.95), 2),
        step=0.05, key="ef_ndvi_plant",
        help="Projected NDVI after a planting intervention.",
    )
with ef_c4:
    ef_solar = st.slider(
        "Solar input (W m⁻²)", 200, 700, 450, step=25, key="ef_solar",
        help="Mean incoming solar irradiance for this region.",
    )

_ef_kwargs = dict(T0=ef_T0, Q_solar=ef_solar, ndvi=ef_ndvi, T_amb=ef_T0 - 8.0)
ef_df         = EnergyFlow(**_ef_kwargs).simulate(steps=365)
ef_df_planted = EnergyFlow(**{**_ef_kwargs, "ndvi": ef_ndvi_plant}).simulate(steps=365)

st.plotly_chart(energy_flow_chart(ef_df, df_planted=ef_df_planted), use_container_width=True)

_delta_T = ef_df["T"].iloc[-1] - ef_df_planted["T"].iloc[-1]
st.caption(
    f"Heat-budget model for **{chart_site}**. "
    f"Equation: dT/dt = [Q_in(t) − k·(1 + β·NDVI)·(T − T_amb)] / C. "
    f"Dotted line = instantaneous equilibrium temperature. "
    f"Planting intervention (NDVI {ef_ndvi:.2f} → {ef_ndvi_plant:.2f}) "
    f"reduces steady-state surface temperature by ≈ **{_delta_T:.1f} °C**."
)
