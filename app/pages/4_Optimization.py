from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_APP = _ROOT / "app"
for _p in (str(_ROOT / "src"), str(_APP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import folium  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402
from ui import page_hero, section_label, set_page_config  # noqa: E402

from data_nature.models.optimization import (  # noqa: E402
    COOLING, GRID, SEASON_MONTHS, VEG_BOOST,
    _CELL_LAT, _CELL_LNG,
    expected_delta_lst, init_site_grid, optimize_planting,
)
from data_nature.viz.charts import convergence_chart  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────

set_page_config(title="Optimization")

page_hero(
    title="🧬 Planting Optimization",
    subtitle=(
        "Genetic algorithm to find the optimal cells to plant for maximum "
        "heat reduction — evidence-based planting plans."
    ),
    pills=["🧬 Genetic Algorithm", "🌳 Optimal Planting", "🌡️ Heat Reduction", "📍 8 Sites"],
    emoji="🧬",
)

# ── Data ──────────────────────────────────────────────────────────────────────

_PROCESSED = _ROOT / "data" / "processed"
_MOCK = _ROOT / "data" / "mock"


@st.cache_data(show_spinner="Loading site data…")
def _load() -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    try:
        monthly = pd.read_csv(_PROCESSED / "site_monthly.csv")
        locs = pd.read_csv(_PROCESSED / "site_locations.csv")
        return monthly, locs, True
    except Exception:
        pass
    monthly = pd.read_csv(_MOCK / "site_monthly.csv")
    locs = pd.read_csv(_MOCK / "site_locations.csv")
    return monthly, locs, False


monthly_df, locs_df, _REAL_DATA = _load()
SITES = sorted(monthly_df["site"].unique())

# ── GEE initialisation ────────────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def _init_gee() -> bool:
    try:
        from data_nature.ingest.earth_engine import authenticate
        authenticate()
        return True
    except Exception:
        return False


GEE_OK: bool = _init_gee()

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .opt-card {
        background:#fff; border:1px solid #e5e7eb; border-radius:10px;
        padding:12px 14px; margin-bottom:8px;
        box-shadow:0 2px 4px rgba(0,0,0,0.04);
    }
    .opt-card-label {
        font-size:0.65em; font-weight:600; text-transform:uppercase;
        letter-spacing:0.07em; color:#6b7280; margin-bottom:2px;
    }
    .opt-card-value { font-size:1.5em; font-weight:800; color:#166534; margin-top:2px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Status banner ─────────────────────────────────────────────────────────────

if not _REAL_DATA:
    st.info("ℹ️ Showing mock data — place real CSVs in `data/processed/` to use live data.")

# ── Session state ─────────────────────────────────────────────────────────────

for _k, _v in {
    "opt_ok": False,
    "opt_site": SITES[0],
    "opt_season": "Summer",
    "opt_budget": 10,
    "opt_history": None,
    "opt_solutions": None,
    "opt_selected": 0,
    "opt_lst": None,
    "opt_ndvi": None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Controls ──────────────────────────────────────────────────────────────────

section_label("Algorithm Parameters")

c1, c2, c3, c4, c5 = st.columns([2, 1.5, 2, 1.5, 1.5])
with c1:
    site: str = st.selectbox("Site", SITES)
with c2:
    season: str = st.selectbox("Season", list(SEASON_MONTHS.keys()), index=1)
with c3:
    budget: int = st.slider("Planting budget (cells)", 1, 50, 10)
with c4:
    n_gen: int = st.number_input("Generations", min_value=10, max_value=300, value=60)
with c5:
    pop_sz: int = st.number_input("Population", min_value=10, max_value=100, value=30)

run_clicked = st.button("▶ Run Genetic Algorithm", type="primary", use_container_width=False)

# ── GA execution ──────────────────────────────────────────────────────────────

if run_clicked:
    _used_real_grid = False
    if GEE_OK:
        with st.spinner("🛰️ Fetching real MODIS pixel data from GEE…"):
            try:
                from data_nature.ingest.earth_engine import fetch_site_grid
                lst0, ndvi0 = fetch_site_grid(site, SEASON_MONTHS[season])
                _used_real_grid = True
            except Exception as _gee_err:
                st.warning(f"GEE fetch failed ({_gee_err}) — falling back to estimated grid.")
                lst0, ndvi0 = init_site_grid(monthly_df, site, season)
    else:
        lst0, ndvi0 = init_site_grid(monthly_df, site, season)

    if _used_real_grid:
        st.success("🛰️ Grid built from real MODIS LST & NDVI pixels (2015–2023 seasonal mean).")
    else:
        st.info("⚠️ Grid estimated from site seasonal averages (GEE unavailable).")

    pb = st.progress(0, text="Initialising population…")

    def _cb(gen: int, total: int, best_fit: float) -> None:
        pb.progress(gen / total, text=f"Generation {gen} / {total}  ·  best fitness: {best_fit:.3f}")

    result = optimize_planting(lst0, ndvi0, budget, int(n_gen), int(pop_sz), progress_cb=_cb)
    pb.empty()

    st.session_state.update(
        opt_ok=True,
        opt_site=site,
        opt_season=season,
        opt_budget=budget,
        opt_history=result["history"],
        opt_solutions=result["solutions"],
        opt_selected=0,
        opt_lst=lst0,
        opt_ndvi=ndvi0,
    )

# ── No-results placeholder ────────────────────────────────────────────────────

if not st.session_state["opt_ok"]:
    st.info(
        "Configure the parameters above and click **▶ Run Genetic Algorithm** "
        "to find the optimal planting locations."
    )
    st.stop()

# ── Retrieve results ──────────────────────────────────────────────────────────

history_: list[float] = st.session_state["opt_history"]
solutions_: list[dict] = st.session_state["opt_solutions"]
sel_: int = int(st.session_state["opt_selected"])
lst_: np.ndarray = st.session_state["opt_lst"]
ndvi_: np.ndarray = st.session_state["opt_ndvi"]
res_site: str = str(st.session_state["opt_site"])
res_season: str = str(st.session_state["opt_season"])
res_budget: int = int(st.session_state["opt_budget"])

best_mask = solutions_[sel_]["mask"]
site_row = locs_df[locs_df["site"] == res_site]
if site_row.empty:
    site_row = locs_df.iloc[[0]]
slat, slng = float(site_row.iloc[0]["lat"]), float(site_row.iloc[0]["lng"])

# ── Folium map builder ────────────────────────────────────────────────────────


def _opt_map(
    lst: np.ndarray,
    ndvi: np.ndarray,
    mask: np.ndarray,
    site_lat: float,
    site_lng: float,
) -> folium.Map:
    fmap = folium.Map(
        location=[site_lat, site_lng],
        zoom_start=13,
        tiles="CartoDB positron",
        control_scale=False,
        zoom_control=False,
    )
    zmin = float(lst.min()) - 0.5
    zmax = float(lst.max()) + 0.5
    norm = mcolors.Normalize(vmin=zmin, vmax=zmax)
    cmap = matplotlib.colormaps["RdYlBu_r"]
    half = GRID / 2.0

    for i in range(GRID):
        for j in range(GRID):
            lat_n = site_lat + (half - i) * _CELL_LAT
            lat_s = site_lat + (half - i - 1) * _CELL_LAT
            lng_w = site_lng + (j - half) * _CELL_LNG
            lng_e = site_lng + (j - half + 1) * _CELL_LNG
            is_planted = bool(mask[i, j])
            fill = "#22C55E" if is_planted else mcolors.to_hex(
                cmap(norm(float(lst[i, j])))
            )
            folium.Rectangle(
                bounds=[[lat_s, lng_w], [lat_n, lng_e]],
                color="#16A34A" if is_planted else "#cccccc",
                weight=3.5 if is_planted else 0.4,
                fill=True,
                fill_color=fill,
                fill_opacity=0.88 if is_planted else 0.65,
                tooltip=(
                    f"<b>{'🌿 Plant here' if is_planted else 'Cell'} ({i},{j})</b>"
                    f"<br>LST: {lst[i, j]:.1f} °C  ·  NDVI: {ndvi[i, j]:.3f}"
                ),
            ).add_to(fmap)

    return fmap


# ── Map + convergence ─────────────────────────────────────────────────────────

section_label(
    f"Results — {res_site} · {res_season} · Budget {res_budget} cells"
    + (f"  (Solution {sel_ + 1})" if sel_ > 0 else "  (Best solution)")
)

col_map, col_conv = st.columns([3, 2], gap="medium")

with col_map:
    st.markdown(
        '<p style="font-size:0.78em;font-weight:700;color:#6b7280;'
        'text-transform:uppercase;letter-spacing:0.08em;margin:0 0 4px">'
        "Optimal Planting Map  🟢 = plant here</p>",
        unsafe_allow_html=True,
    )
    st_folium(
        _opt_map(lst_, ndvi_, best_mask, slat, slng),
        height=340,
        returned_objects=[],
        key="opt_map",
    )

with col_conv:
    st.markdown(
        '<p style="font-size:0.78em;font-weight:700;color:#6b7280;'
        'text-transform:uppercase;letter-spacing:0.08em;margin:0 0 4px">'
        "Convergence</p>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(convergence_chart(history_), use_container_width=True)

# ── Summary metrics ───────────────────────────────────────────────────────────

best_delta = solutions_[sel_]["delta_lst"]
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Expected Avg ΔLST", f"{best_delta:+.2f} °C")
mc2.metric("Cells planted", f"{int(best_mask.sum())} / {res_budget}")
mc3.metric("Fitness score", f"{solutions_[sel_]['fitness']:.4f}")
mc4.metric("Generations run", str(len(history_)))

st.write("")

# ── Top 5 solutions table ─────────────────────────────────────────────────────

section_label("Top 5 Solutions")

col_t, col_sel = st.columns([3, 1], gap="medium")

with col_t:
    table_rows = [
        {
            "#": rank + 1,
            "ΔLST (avg)": f"{sol['delta_lst']:+.2f} °C",
            "Cells": int(sol["mask"].sum()),
            "Fitness": round(sol["fitness"], 4),
        }
        for rank, sol in enumerate(solutions_)
    ]
    st.dataframe(
        pd.DataFrame(table_rows),
        use_container_width=True,
        hide_index=True,
        column_config={"#": st.column_config.NumberColumn(width="small")},
    )

with col_sel:
    st.write("")
    chosen = st.radio(
        "Load on map",
        options=list(range(1, len(solutions_) + 1)),
        format_func=lambda x: f"Solution {x}",
        index=sel_,
        key="opt_radio",
    )
    if chosen - 1 != sel_:
        st.session_state["opt_selected"] = chosen - 1
        st.rerun()

# ── Export planting plan ──────────────────────────────────────────────────────

st.write("")
coords = []
for i in range(GRID):
    for j in range(GRID):
        if best_mask[i, j]:
            coords.append({
                "site": res_site,
                "cell_row": i,
                "cell_col": j,
                "lat": round(slat + (GRID / 2 - i - 0.5) * _CELL_LAT, 6),
                "lng": round(slng + (j - GRID / 2 + 0.5) * _CELL_LNG, 6),
                "lst_current_C": round(float(lst_[i, j]), 2),
                "ndvi_current": round(float(ndvi_[i, j]), 3),
                "expected_delta_lst_C": round(
                    -COOLING * float(
                        np.clip(ndvi_[i, j] * (1 + VEG_BOOST), 0.01, 0.99) - ndvi_[i, j]
                    ),
                    3,
                ),
            })

st.download_button(
    label="⬇️ Export planting plan (CSV)",
    data=pd.DataFrame(coords).to_csv(index=False),
    file_name=f"planting_plan_{res_site.replace(' ', '_')}_{res_season}.csv",
    mime="text/csv",
)
st.caption(
    f"Planting plan for **{res_site}** · {res_season} · Solution {sel_ + 1}. "
    "Coordinates are cell centres (±150 m accuracy). "
    "Expected ΔLST assumes +20 % NDVI increase per planted cell."
)
