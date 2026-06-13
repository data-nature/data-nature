from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_APP = _ROOT / "app"
for _p in (str(_ROOT / "src"), str(_APP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import os

import ee  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402
from ui import page_hero, section_label, set_page_config  # noqa: E402

from data_nature.viz.maps import LAYER_CFG, build_site_map, legend_html  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────

set_page_config(title="Heat Map")

page_hero(
    title="🗺️ Interactive Heat Map",
    subtitle="Real-time MODIS satellite imagery via Google Earth Engine — LST (MOD11A1), NDVI (MOD13Q1), and Land Cover (MOD12Q1) across Northern Israel, 2000–2025.",
    pills=["🛰️ Google Earth Engine", "🌡️ MOD11A1 LST", "🌿 MOD13Q1 NDVI", "🗂️ MOD12Q1 Land Cover", "📅 2000–2025"],
    emoji="🗺️",
)

# ── Data ──────────────────────────────────────────────────────────────────────

_PROCESSED = _ROOT / "data" / "processed"
_MOCK = _ROOT / "data" / "mock"


def _compute_zscores(monthly: pd.DataFrame) -> pd.DataFrame:
    """Fill z_score_lst, z_score_ndvi, delta, is_anomaly using the stats module."""
    from data_nature.stats import compute_zscores

    # LST z-scores
    lst_enriched = compute_zscores(monthly)
    monthly = monthly.copy()
    monthly["z_score_lst"] = lst_enriched["z_score"]

    # NDVI z-scores (same logic, computed inline)
    ndvi_stats = (
        monthly.groupby(["site", "month"])["ndvi"]
        .agg(ndvi_mean="mean", ndvi_std="std")
        .reset_index()
    )
    monthly = monthly.merge(ndvi_stats, on=["site", "month"], how="left")
    safe_std = monthly["ndvi_std"].replace(0, float("nan"))
    monthly["z_score_ndvi"] = ((monthly["ndvi"] - monthly["ndvi_mean"]) / safe_std).round(4)
    monthly.drop(columns=["ndvi_mean", "ndvi_std"], inplace=True)

    monthly["delta"] = (monthly["z_score_lst"] - monthly["z_score_ndvi"]).round(4)
    monthly["is_anomaly"] = monthly["z_score_lst"] >= 1.5
    return monthly


@st.cache_data(show_spinner="Loading site data…")
def _load() -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    Returns (monthly_df, locs_df, using_real_data).
    Tries data/processed/ first; falls back to data/mock/ on any failure.
    """
    try:
        monthly = pd.read_csv(_PROCESSED / "site_monthly.csv")
        locs = pd.read_csv(_PROCESSED / "site_locations.csv")
        if monthly["z_score_lst"].isna().all():
            monthly = _compute_zscores(monthly)
        # Ensure is_anomaly is bool
        monthly["is_anomaly"] = monthly["is_anomaly"].astype(bool)
        return monthly, locs, True
    except Exception:
        pass

    monthly = pd.read_csv(_MOCK / "site_monthly.csv")
    locs = pd.read_csv(_MOCK / "site_locations.csv")
    # Mock CSVs store "True"/"False" as strings in some environments
    monthly["is_anomaly"] = monthly["is_anomaly"].astype(str).str.strip().str.lower() == "true"
    return monthly, locs, False


monthly_df, locs_df, _REAL_DATA = _load()
SITES = sorted(monthly_df["site"].unique())
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ── GEE initialisation ────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="🛰️ Connecting to Google Earth Engine…")
def _init_gee() -> bool:
    project = "datanature"
    try:
        key_json: str = (
            st.secrets.get("gee", {}).get("key_json")  # type: ignore[union-attr]
            or os.environ.get("GEE_KEY_JSON", "")
        )
        sa_email: str = (
            st.secrets.get("gee", {}).get("service_account")  # type: ignore[union-attr]
            or os.environ.get("GEE_SERVICE_ACCOUNT", "")
        )
        if key_json and sa_email:
            credentials = ee.ServiceAccountCredentials(sa_email, key_data=key_json)
            ee.Initialize(credentials, project=project)
            return True
    except Exception:
        pass
    try:
        ee.Initialize(project=project)
        return True
    except Exception:
        return False


GEE_OK: bool = _init_gee()

# ── Session state defaults ────────────────────────────────────────────────────

_defaults: dict[str, object] = {
    "hm_year": 2025, "hm_month": 7,
    "hm_playing": False, "hm_compare": False,
    "hm_site": SITES[0],
    "hm_year2": 2010, "hm_month2": 7,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── GEE tile URL (cached, TTL 1 h) ────────────────────────────────────────────


@st.cache_data(ttl=3600, show_spinner=False)
def _gee_tile_url(layer_name: str, y: int, mo: int) -> str | None:
    if not GEE_OK:
        return None
    try:
        region = ee.Geometry.Rectangle([34.5, 32.0, 36.3, 33.5])
        end_mo = mo % 12 + 1
        end_y = y + (1 if mo == 12 else 0)
        start = f"{y}-{mo:02d}-01"
        end = f"{end_y}-{end_mo:02d}-01"

        if layer_name == "LST Day (MOD11A1)":
            img = (
                ee.ImageCollection("MODIS/061/MOD11A1")
                .filterDate(start, end)
                .filterBounds(region)
                .select("LST_Day_1km")
                .mean()
                .multiply(0.02)
                .subtract(273.15)
            )
            vis = {
                "min": 20, "max": 55,
                "palette": ["#313695", "#74add1", "#e0f3f8",
                            "#fee090", "#f46d43", "#d73027", "#a50026"],
            }
        elif layer_name == "NDVI (MOD13Q1)":
            img = (
                ee.ImageCollection("MODIS/061/MOD13Q1")
                .filterDate(start, end)
                .filterBounds(region)
                .select("NDVI")
                .mean()
                .multiply(0.0001)
            )
            vis = {
                "min": 0, "max": 0.8,
                "palette": ["#d73027", "#f46d43", "#fdae61", "#fee08b",
                            "#d9ef8b", "#a6d96a", "#66bd63", "#1a9850"],
            }
        elif layer_name == "Land Cover (MOD12Q1)":
            img = (
                ee.ImageCollection("MODIS/061/MOD12Q1")
                .filterDate(f"{y}-01-01", f"{y}-12-31")
                .filterBounds(region)
                .first()
                .select("LC_Type1")
            )
            vis = {
                "min": 0, "max": 17,
                "palette": ["1c0dff", "05450a", "086a10", "54a708", "78d203",
                            "009900", "c6b044", "dcd159", "dade48", "fbff13",
                            "b6ff05", "27af87", "c24f44", "a5a5a5", "ff6d4c",
                            "69fff8", "f9ffa4", "1c0dff"],
            }
        else:
            return None

        map_id = img.clip(region).getMapId(vis)
        return str(map_id["tile_fetcher"].url_format)
    except Exception:
        return None


# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .detail-card {
      background: #fff; border: 1px solid #e5e7eb; border-radius: 12px;
      padding: 20px 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .detail-site { font-size: 1.05em; font-weight: 800; color: #1C1B18; margin-bottom: 14px; }
    .detail-row {
      display: flex; justify-content: space-between; align-items: center;
      padding: 7px 0; border-bottom: 1px solid #F3F4F6; font-size: 0.84em;
    }
    .detail-row:last-child { border-bottom: none; }
    .detail-key { color: #6b7280; font-weight: 500; }
    .detail-val { font-weight: 700; color: #1C1B18; }
    .anom-pill {
      display: inline-block; border-radius: 999px;
      padding: 3px 12px; font-size: 0.72em; font-weight: 700; margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Status banners ────────────────────────────────────────────────────────────

if not _REAL_DATA:
    st.info("ℹ️ Showing mock data — place real CSVs in `data/processed/` to use live data.")

if not GEE_OK:
    st.warning(
        "⚠️ Google Earth Engine is not authenticated — satellite layer will not load. "
        "Run `earthengine authenticate` in your terminal, then restart the app."
    )

# ── Auto-advance animation ────────────────────────────────────────────────────

if st.session_state["hm_playing"]:
    _m = int(st.session_state["hm_month"]) + 1  # type: ignore[arg-type]
    _y = int(st.session_state["hm_year"])  # type: ignore[arg-type]
    if _m > 12:
        _m = 1
        _y += 1
    if _y > 2025:
        st.session_state["hm_playing"] = False
    else:
        st.session_state["hm_month"] = _m
        st.session_state["hm_year"] = _y
        time.sleep(0.6)
        st.rerun()

# ── Controls ──────────────────────────────────────────────────────────────────

section_label("Controls")

c1, c2, c3, c4, c5, c6 = st.columns([1.8, 2, 1.4, 2, 1, 1])
with c1:
    layer: str = st.selectbox("MODIS Layer", list(LAYER_CFG.keys()))
with c2:
    year: int = st.slider("Year", 2000, 2025, key="hm_year")
with c3:
    month: int = st.slider("Month", 1, 12, key="hm_month", help="1 = Jan … 12 = Dec")
with c4:
    site_sel: str = st.selectbox("Highlight site", ["All"] + list(SITES))
with c5:
    st.write("")
    play_lbl = "⏹ Stop" if st.session_state["hm_playing"] else "▶ Play"
    if st.button(play_lbl, use_container_width=True):
        st.session_state["hm_playing"] = not st.session_state["hm_playing"]
        st.rerun()
with c6:
    st.write("")
    cmp_lbl = "❌ Single" if st.session_state["hm_compare"] else "⧉ Compare"
    if st.button(cmp_lbl, use_container_width=True):
        st.session_state["hm_compare"] = not st.session_state["hm_compare"]
        st.rerun()

cfg = LAYER_CFG[layer]

# ── Helpers ───────────────────────────────────────────────────────────────────


def _snap(y: int, mo: int) -> pd.DataFrame:
    return monthly_df[(monthly_df["year"] == y) & (monthly_df["month"] == mo)].merge(
        locs_df, on="site"
    )


def _make_map(snap_df: pd.DataFrame, layer_name: str, layer_cfg: dict,
              highlight: str, y: int, mo: int):
    tile_url = _gee_tile_url(layer_name, y, mo)
    return build_site_map(snap_df, layer_cfg, highlight, tile_url)


# ── Map layout ────────────────────────────────────────────────────────────────

compare_mode: bool = bool(st.session_state["hm_compare"])

if not compare_mode:
    snap = _snap(year, month)
    col_map, col_detail = st.columns([3, 1], gap="medium")

    with col_map:
        section_label(f"Map — {MONTH_NAMES[month - 1]} {year}")

        with st.spinner("Loading satellite layer…"):
            fmap = _make_map(snap, layer, cfg, site_sel, year, month)

        result = st_folium(
            fmap,
            height=460,
            returned_objects=["last_object_clicked"],
            key="main_map",
        )
        if result and result.get("last_object_clicked"):
            click = result["last_object_clicked"]
            if click and "lat" in click:
                dists = (
                    (locs_df["lat"] - click["lat"]) ** 2
                    + (locs_df["lng"] - click["lng"]) ** 2
                )
                st.session_state["hm_site"] = locs_df.loc[dists.idxmin(), "site"]

        st.markdown(legend_html(cfg), unsafe_allow_html=True)

    with col_detail:
        section_label("Site Detail")
        sel_site: str = str(st.session_state["hm_site"])
        sel_rows = snap[snap["site"] == sel_site]
        if sel_rows.empty:
            sel_rows = snap.iloc[[0]]
            sel_site = str(sel_rows.iloc[0]["site"])

        r = sel_rows.iloc[0]
        lc = locs_df[locs_df["site"] == sel_site].iloc[0]
        is_anom = bool(r["is_anomaly"])
        anom_pill = (
            '<span class="anom-pill" style="background:#FEE2E2;color:#DC2626">⚠️ Anomaly</span>'
            if is_anom
            else '<span class="anom-pill" style="background:#DCFCE7;color:#166534">✅ Normal</span>'
        )

        st.markdown(
            f"""
            <div class="detail-card">
              <div class="detail-site">📍 {sel_site}</div>
              {anom_pill}
              <div class="detail-row">
                <span class="detail-key">Land cover</span>
                <span class="detail-val">{lc["land_cover"]}</span>
              </div>
              <div class="detail-row">
                <span class="detail-key">LST</span>
                <span class="detail-val">{r["lst"]:.1f} °C</span>
              </div>
              <div class="detail-row">
                <span class="detail-key">NDVI</span>
                <span class="detail-val">{r["ndvi"]:.3f}</span>
              </div>
              <div class="detail-row">
                <span class="detail-key">LST z-score</span>
                <span class="detail-val">{r["z_score_lst"]:+.2f} σ</span>
              </div>
              <div class="detail-row">
                <span class="detail-key">NDVI z-score</span>
                <span class="detail-val">{r["z_score_ndvi"]:+.2f} σ</span>
              </div>
              <div class="detail-row">
                <span class="detail-key">Δ (LST − NDVI)</span>
                <span class="detail-val">{r["delta"]:+.2f} σ</span>
              </div>
              <div class="detail-row">
                <span class="detail-key">Period</span>
                <span class="detail-val">{MONTH_NAMES[month - 1]} {year}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    # ── Before / After compare mode ───────────────────────────────────────────
    cc1, cc2 = st.columns(2)
    with cc1:
        y1: int = st.slider("Year (left)",  2000, 2025, key="hm_year2")
        m1: int = st.slider("Month (left)", 1,    12,   key="hm_month2")
    with cc2:
        st.info(f"**Right:** {MONTH_NAMES[month - 1]} {year}  ·  use main sliders above")
        y2, m2 = year, month

    snap1, snap2 = _snap(y1, m1), _snap(y2, m2)
    cm1, cm2 = st.columns(2, gap="medium")

    with cm1:
        section_label(f"{MONTH_NAMES[m1 - 1]} {y1}")
        with st.spinner("Loading…"):
            fmap1 = _make_map(snap1, layer, cfg, "All", y1, m1)
        st_folium(fmap1, height=430, returned_objects=[], key="cmp_map1")
        st.markdown(legend_html(cfg), unsafe_allow_html=True)

    with cm2:
        section_label(f"{MONTH_NAMES[m2 - 1]} {y2}")
        with st.spinner("Loading…"):
            fmap2 = _make_map(snap2, layer, cfg, "All", y2, m2)
        st_folium(fmap2, height=430, returned_objects=[], key="cmp_map2")
        st.markdown(legend_html(cfg), unsafe_allow_html=True)

# ── Timeline ──────────────────────────────────────────────────────────────────

st.write("")
tl_site = str(st.session_state.get("hm_site", SITES[0]))
section_label(f"Timeline — LST z-Score History · {tl_site}")

tl_df = monthly_df[monthly_df["site"] == tl_site].sort_values(["year", "month"]).copy()
tl_df["date"] = pd.to_datetime(dict(year=tl_df["year"], month=tl_df["month"], day=1))

normal_pts = tl_df[~tl_df["is_anomaly"]]
anom_pts = tl_df[tl_df["is_anomaly"]]

fig = go.Figure()
fig.add_hrect(y0=-1, y1=1, fillcolor="rgba(46,125,50,0.07)", line_width=0)
fig.add_hrect(y0=1.5, y1=6, fillcolor="rgba(220,38,38,0.04)", line_width=0)
fig.add_trace(
    go.Scatter(
        x=normal_pts["date"], y=normal_pts["z_score_lst"],
        mode="markers", name="Normal",
        marker={"color": "#9CA3AF", "size": 4, "opacity": 0.6},
        hovertemplate="%{x|%b %Y}<br>z: %{y:.2f}<extra>Normal</extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=anom_pts["date"], y=anom_pts["z_score_lst"],
        mode="markers", name="Anomaly",
        marker={"color": "#DC2626", "size": 8, "line": {"color": "#fff", "width": 1}},
        hovertemplate="%{x|%b %Y}<br>z: %{y:.2f}<extra>Anomaly</extra>",
    )
)

cur_date = pd.Timestamp(year=year, month=month, day=1).isoformat()
fig.add_shape(
    type="line", x0=cur_date, x1=cur_date, y0=0, y1=1,
    xref="x", yref="paper",
    line={"color": "#2E7D32", "dash": "dash", "width": 1.5},
)
fig.add_annotation(
    x=cur_date, y=1, xref="x", yref="paper",
    text=f"{MONTH_NAMES[month - 1]} {year}",
    showarrow=False, xanchor="left",
    font={"color": "#2E7D32", "size": 10},
)
fig.update_layout(
    height=210,
    margin={"t": 10, "b": 20, "l": 0, "r": 70},
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FAFAFA",
    hovermode="x",
    legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    xaxis={"showgrid": True, "gridcolor": "#F3F4F6", "tickformat": "%Y", "title": None},
    yaxis={
        "showgrid": True, "gridcolor": "#F3F4F6",
        "title": {"text": "LST z-score", "font": {"size": 11}},
    },
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    f"Monthly LST z-score for **{tl_site}** (2000–2025). "
    "Red = anomaly months (z ≥ 1.5σ). Green dashed line = current selection. "
    "Click a site marker on the map to change the site shown here."
)

# ── Logistic Growth — Vegetation Recovery Model (DN-A6) ──────────────────────

st.write("")
section_label(f"Vegetation Recovery — Logistic Growth Model · {tl_site}")

from data_nature.models.ecology import LogisticGrowth  # noqa: E402
from data_nature.viz.charts import logistic_growth_chart  # noqa: E402

# Derive P0 from the site's current mean NDVI
_site_ndvi = float(
    monthly_df[monthly_df["site"] == tl_site]["ndvi"].mean()
)

lg_c1, lg_c2, lg_c3, lg_c4 = st.columns([1.5, 1.5, 1.5, 1.5])
with lg_c1:
    lg_P0 = st.slider(
        "Initial cover (P₀)", 0.01, 0.95,
        value=round(min(_site_ndvi, 0.9), 2),
        step=0.01, key="lg_P0",
        help="Starting vegetation density (NDVI proxy for this site).",
    )
with lg_c2:
    lg_r = st.slider(
        "Growth rate (r)", 0.05, 2.0, 0.4, step=0.05, key="lg_r",
        help="Intrinsic vegetation growth rate.",
    )
with lg_c3:
    lg_K = st.slider(
        "Carrying capacity (K)", 0.3, 1.0, 0.85, step=0.05, key="lg_K",
        help="Maximum sustainable vegetation cover for this land type.",
    )
with lg_c4:
    lg_years = st.slider(
        "Simulation years", 5, 50, 20, step=5, key="lg_years",
    )

lg_steps = lg_years * 10
lg_df = LogisticGrowth(P0=lg_P0, r=lg_r, K=lg_K).simulate(steps=lg_steps, dt=0.1)
lg_df["t"] = lg_df["t"] / 10  # convert to years

st.plotly_chart(logistic_growth_chart(lg_df, K=lg_K), use_container_width=True)
st.caption(
    f"Logistic growth model for **{tl_site}**. "
    f"Equation: dP/dt = r·P·(1 − P/K). "
    f"Initial cover P₀ = {lg_P0:.2f} (site mean NDVI). "
    "The curve saturates at carrying capacity K — the maximum vegetation density "
    "sustainable under local climate and land-use conditions."
)
