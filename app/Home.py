"""Data Nature — Central Dashboard (Home)."""

from __future__ import annotations

import pathlib

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from ui import page_hero, section_label, set_page_config

from data_nature import __version__

set_page_config()

# ── Data paths ────────────────────────────────────────────────────────────────
_DATA = pathlib.Path(__file__).parent.parent / "data" / "mock"


@st.cache_data(show_spinner=False)
def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    anomalies = pd.read_csv(_DATA / "anomalies.csv", parse_dates=["date"])
    monthly = pd.read_csv(_DATA / "site_monthly.csv")
    locations = pd.read_csv(_DATA / "site_locations.csv")
    return anomalies, monthly, locations


anomalies, monthly, locations = _load()

# ── KPI computations ──────────────────────────────────────────────────────────
active_anomalies = anomalies[anomalies["status"] == "New"]
active_count = len(active_anomalies)

# Hottest site: site with highest mean LST in latest month of data
latest_year = int(monthly["year"].max())
latest_month = int(monthly[monthly["year"] == latest_year]["month"].max())
prev_month = latest_month - 1 if latest_month > 1 else 12
prev_year = latest_year if latest_month > 1 else latest_year - 1

latest_monthly = monthly[(monthly["year"] == latest_year) & (monthly["month"] == latest_month)]
prev_monthly = monthly[(monthly["year"] == prev_year) & (monthly["month"] == prev_month)]

hottest_site = latest_monthly.groupby("site")["lst"].mean().idxmax()
hottest_lst = latest_monthly.groupby("site")["lst"].mean().max()

avg_ndvi = latest_monthly["ndvi"].mean()
prev_ndvi = prev_monthly["ndvi"].mean() if not prev_monthly.empty else avg_ndvi
ndvi_delta = avg_ndvi - prev_ndvi

avg_lst_now = latest_monthly["lst"].mean()
avg_lst_prev = prev_monthly["lst"].mean() if not prev_monthly.empty else avg_lst_now
lst_delta = avg_lst_now - avg_lst_prev

# ── Risk level per site ───────────────────────────────────────────────────────
# Critical = red, Severe = yellow, else = green
site_max_sev: dict[str, str] = {}
for site in locations["site"]:
    site_anoms = active_anomalies[active_anomalies["site"] == site]
    if (site_anoms["severity"] == "Critical").any():
        site_max_sev[site] = "Critical"
    elif (site_anoms["severity"] == "Severe").any():
        site_max_sev[site] = "Severe"
    else:
        site_max_sev[site] = "Normal"

_SEV_COLOR = {"Critical": "#EF4444", "Severe": "#F59E0B", "Normal": "#22C55E"}


# ── Insights of the Week ──────────────────────────────────────────────────────
def _generate_insights() -> list[str]:
    insights: list[str] = []

    crit_sites = [s for s, sev in site_max_sev.items() if sev == "Critical"]
    if crit_sites:
        insights.append(
            f"🔴 **{len(crit_sites)} site{'s' if len(crit_sites) > 1 else ''} at Critical level** "
            f"this week: {', '.join(crit_sites[:3])}."
        )
    else:
        insights.append("✅ **No critical anomalies** detected across all monitored sites this week.")

    ndvi_sign = "increased" if ndvi_delta >= 0 else "decreased"
    ndvi_pct = abs(ndvi_delta)
    insights.append(
        f"🌿 Average NDVI has **{ndvi_sign} by {ndvi_pct:.3f}** compared to last month "
        f"({'improving' if ndvi_delta >= 0 else 'declining'} vegetation health)."
    )

    lst_sign = "warmer" if lst_delta >= 0 else "cooler"
    insights.append(
        f"🌡️ Mean surface temperature is **{abs(lst_delta):.1f} °C {lst_sign}** than last month — "
        f"hottest site is **{hottest_site}** at {hottest_lst:.1f} °C."
    )

    handled = anomalies[anomalies["status"] == "Handled"]
    if not handled.empty:
        recent_handled = handled.sort_values("date").iloc[-1]
        insights.append(
            f"📋 Most recently resolved anomaly: **{recent_handled['site']}** "
            f"on {recent_handled['date'].strftime('%b %d')}, severity {recent_handled['severity']}."
        )

    return insights


insights = _generate_insights()

# ── Map builder ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _build_map(_locations: pd.DataFrame, _site_sev: dict[str, str]) -> folium.Map:
    center_lat = _locations["lat"].mean()
    center_lng = _locations["lng"].mean()
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=9,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )
    for _, row in _locations.iterrows():
        site = str(row["site"])
        sev = _site_sev.get(site, "Normal")
        color = _SEV_COLOR[sev]
        folium.CircleMarker(
            location=[row["lat"], row["lng"]],
            radius=10,
            color="white",
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            tooltip=folium.Tooltip(
                f"<b>{site}</b><br>Status: {sev}<br>Land cover: {row['land_cover']}",
                sticky=True,
            ),
            popup=folium.Popup(
                f"<b>{site}</b><br>Status: {sev}<br>Lat: {row['lat']:.3f} | Lng: {row['lng']:.3f}",
                max_width=200,
            ),
        ).add_to(m)
    return m


# ── Page layout ───────────────────────────────────────────────────────────────
_notif_badge = (
    f'<span style="background:#EF4444;color:white;border-radius:999px;'
    f'padding:2px 9px;font-size:0.78em;font-weight:700;margin-left:8px;">'
    f'{active_count}</span>'
    if active_count > 0
    else ""
)

page_hero(
    title="Data Nature",
    subtitle=(
        "Satellite-driven ecological monitoring for Northern Israel — "
        "combining NDVI, LST, and ML-powered anomaly detection across 8 sites."
    ),
    pills=["📍 8 Sites", "📅 2000–2026", "🛰️ NDVI + LST", f"🚨 {active_count} Active Alerts"],
    emoji="🌿",
)

# ── KPI row ───────────────────────────────────────────────────────────────────
section_label("Key Metrics")

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    label="🚨 Active Anomalies",
    value=str(active_count),
    delta=None,
    help="Anomalies with status 'New' across all sites",
)
k2.metric(
    label="🌡️ Hottest Site",
    value=hottest_site,
    delta=f"{hottest_lst:.1f} °C",
    help=f"Site with highest mean LST in {latest_year}-{latest_month:02d}",
)
k3.metric(
    label="🌿 Average NDVI",
    value=f"{avg_ndvi:.3f}",
    delta=f"{ndvi_delta:+.3f} vs prev month",
    delta_color="normal",
    help="Mean NDVI across all sites this month",
)
k4.metric(
    label="🌡️ Temp Change",
    value=f"{avg_lst_now:.1f} °C",
    delta=f"{lst_delta:+.1f} °C vs prev month",
    delta_color="inverse",
    help="Mean LST change vs previous month",
)

st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

# ── Map + Insights row ────────────────────────────────────────────────────────
section_label("Site Overview")

map_col, insight_col = st.columns([3, 2], gap="large")

with map_col:
    st.markdown(
        "<p style='font-size:0.85em;color:#6b7280;margin-bottom:6px;'>"
        "Site risk levels — <span style='color:#22C55E;font-weight:600;'>● Normal</span>"
        " &nbsp;<span style='color:#F59E0B;font-weight:600;'>● Severe</span>"
        " &nbsp;<span style='color:#EF4444;font-weight:600;'>● Critical</span>"
        "</p>",
        unsafe_allow_html=True,
    )
    site_map = _build_map(locations, site_max_sev)
    st_folium(site_map, width=None, height=420, returned_objects=[], key="home_map")

with insight_col:
    st.markdown(
        f"<div style='display:flex;align-items:center;margin-bottom:10px;'>"
        f"<span style='font-size:0.68em;font-weight:700;letter-spacing:0.12em;"
        f"text-transform:uppercase;color:#6b7280;'>Insights of the Week</span>"
        f"{_notif_badge}</div>",
        unsafe_allow_html=True,
    )
    for insight in insights:
        st.markdown(
            f"<div style='background:rgba(232,245,233,0.6);border:1px solid #A5D6A7;"
            f"border-radius:10px;padding:12px 14px;margin-bottom:10px;"
            f"font-size:0.88em;line-height:1.6;'>{insight}</div>",
            unsafe_allow_html=True,
        )

# ── Active anomalies table ────────────────────────────────────────────────────
if active_count > 0:
    section_label("Active Alerts")
    display_cols = ["date", "site", "lst", "baseline", "z_score", "severity"]
    df_show = (
        active_anomalies[display_cols]
        .sort_values("date", ascending=False)
        .rename(
            columns={
                "date": "Date",
                "site": "Site",
                "lst": "LST (°C)",
                "baseline": "Baseline",
                "z_score": "Z-Score",
                "severity": "Severity",
            }
        )
        .reset_index(drop=True)
    )
    st.dataframe(df_show, use_container_width=True, hide_index=True)

# ── Navigation ────────────────────────────────────────────────────────────────
section_label("Modules")

nav_cols = st.columns(3)
_MODULES = [
    ("🗺️ Heatmap", "Heatmap", "Interactive NDVI & LST satellite layers over Northern Israel"),
    ("🚨 Anomalies", "Anomalies", "Z-score based heat anomaly detection with timeseries"),
    ("🔬 Simulator", "Simulator", "Cellular-automaton what-if scenarios for vegetation change"),
    ("🧬 Optimization", "Optimization", "Genetic algorithm for optimal tree-planting locations"),
    ("📈 Forecast", "Forecast", "7-day LST forecast with ML model"),
    ("📋 Reports", "Reports", "Export PDF summaries for any site or date range"),
]

for i, (label, _page, desc) in enumerate(_MODULES):
    with nav_cols[i % 3]:
        st.markdown(
            f"<div style='background:rgba(232,245,233,0.6);border:1px solid #A5D6A7;"
            f"border-left:4px solid #2E7D32;border-radius:10px;padding:14px 16px;"
            f"margin-bottom:12px;'>"
            f"<div style='font-weight:700;font-size:0.95em;margin-bottom:4px;'>{label}</div>"
            f"<div style='font-size:0.80em;color:#6b7280;line-height:1.5;'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
st.caption(
    f"Data Nature v{__version__} · Course: Ecological Models Lab — Spring 2026 · "
    "Team: Alaa Barazi, Ahmad Tawil"
)
