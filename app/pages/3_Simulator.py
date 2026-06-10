from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_APP  = _ROOT / "app"
for _p in (str(_ROOT / "src"), str(_APP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib                                           # noqa: E402
import matplotlib.colors as mcolors                        # noqa: E402
import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402
import plotly.graph_objects as go                          # noqa: E402
import streamlit as st                                     # noqa: E402
import folium                                              # noqa: E402
from streamlit_folium import st_folium                     # noqa: E402
from ui import page_hero, section_label, set_page_config  # noqa: E402

from data_nature.models.cellular_automaton import (        # noqa: E402
    COOLING, DIFF, SAFE_LST, GRID_SIZE as GRID,
    HeatCA, init_grid, make_veg_mask, SEASON_MONTHS,
)
from data_nature.models.ecology import (                   # noqa: E402
    LotkaVolterra, simulate_lv,
    LogisticGrowth, simulate_logistic,
)

# ── Page config ───────────────────────────────────────────────────────────────

set_page_config(title="Simulator")

page_hero(
    title="🔬 What-If Simulator",
    subtitle=(
        "Cellular-automaton heat-diffusion model — test vegetation cover changes "
        "before going to the field."
    ),
    pills=["🔄 Cellular Automaton", "🌱 Vegetation Scenarios",
           "🌡️ Heat Diffusion", "📍 8 Sites"],
    emoji="🔬",
)

# ── Data loading (real data with mock fallback) ───────────────────────────────

PROCESSED = _ROOT / "data" / "processed"
MOCK      = _ROOT / "data" / "mock"


@st.cache_data
def _load() -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    real_monthly  = PROCESSED / "site_monthly.csv"
    real_locs     = PROCESSED / "site_locations.csv"
    if real_monthly.exists() and real_locs.exists():
        try:
            monthly = pd.read_csv(real_monthly)
            locs    = pd.read_csv(real_locs)
            if len(monthly) > 0 and len(locs) > 0:
                return monthly, locs, True
        except Exception:
            pass
    return pd.read_csv(MOCK / "site_monthly.csv"), pd.read_csv(MOCK / "site_locations.csv"), False


monthly_df, locs_df, using_real = _load()
SITES = sorted(monthly_df["site"].unique())

if not using_real:
    st.caption("ℹ️ Showing mock data — processed CSVs not found.")

_VEG_MASK = make_veg_mask(GRID)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.sim-card {
    background:#fff; border:1px solid #e5e7eb; border-radius:10px;
    padding:12px 14px; margin-bottom:8px;
    box-shadow:0 2px 4px rgba(0,0,0,0.04);
}
.sim-card-label {
    font-size:0.65em; font-weight:600; text-transform:uppercase;
    letter-spacing:0.07em; margin-bottom:2px;
}
.sim-card-value { font-size:1.55em; font-weight:800; margin-top:2px; }
.sim-grid-title {
    font-size:0.78em; font-weight:700; color:#6b7280;
    text-transform:uppercase; letter-spacing:0.08em; margin:0 0 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

for _k, _v in {
    "sim_ok": False, "sim_step": 0,
    "sim_site": SITES[0], "sim_season": "Summer",
    "sim_veg_pct": 20, "sim_n_steps": 5,
    "sim_lst0": None, "sim_ndvi0": None,
    "sim_grids": None, "sim_avgs": None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Controls ──────────────────────────────────────────────────────────────────

section_label("Scenario Parameters")

c1, c2, c3, c4 = st.columns([2, 2, 1.5, 1])
with c1:
    site: str = st.selectbox("Site", SITES)
with c2:
    veg_pct: int = st.slider(
        "Vegetation cover change (%)", -50, 100, 20, step=5,
        help="Positive = add vegetation  ·  Negative = remove",
    )
with c3:
    season: str = st.selectbox("Season", list(SEASON_MONTHS.keys()), index=1)
with c4:
    n_steps: int = st.number_input("Steps", min_value=1, max_value=20, value=5)

col_run, col_step, col_save = st.columns([1.2, 1, 1])
run_clicked  = col_run.button("▶ Run Simulation",  type="primary", use_container_width=True)
step_clicked = col_step.button("⏭ Step",            use_container_width=True)
save_clicked = col_save.button("💾 Save Scenario",  use_container_width=True)

# ── Simulation execution ──────────────────────────────────────────────────────

def _start(s: str, sea: str, vp: float, ns: int) -> None:
    lst0, ndvi0 = init_grid(s, sea, monthly_df)
    ca = HeatCA(lst0, ndvi0, season=sea)
    ca.apply_vegetation(vp)
    for _ in range(ns):
        ca.step()
    st.session_state.update(
        sim_ok=True, sim_step=ns,
        sim_site=s, sim_season=sea,
        sim_veg_pct=int(vp), sim_n_steps=ns,
        sim_lst0=lst0, sim_ndvi0=ndvi0,
        sim_grids=ca.history, sim_avgs=ca.avg_lst_trajectory,
    )


_at_max = False
if run_clicked:
    _start(site, season, float(veg_pct), n_steps)
elif step_clicked and st.session_state["sim_ok"]:
    _cur   = int(st.session_state["sim_step"])
    _total = len(st.session_state["sim_grids"]) - 1
    if _cur < _total:
        st.session_state["sim_step"] = _cur + 1
    else:
        _at_max = True

if not st.session_state["sim_ok"]:
    _start(site, season, float(veg_pct), n_steps)

# ── Read current state ────────────────────────────────────────────────────────

lst0_:    np.ndarray      = st.session_state["sim_lst0"]
ndvi0_:   np.ndarray      = st.session_state["sim_ndvi0"]
grids_:   list[np.ndarray]= st.session_state["sim_grids"]
avgs_:    list[float]     = st.session_state["sim_avgs"]
cur_step: int             = int(st.session_state["sim_step"])
lst_pred: np.ndarray      = grids_[cur_step]

zlo = float(min(lst0_.min(), lst_pred.min())) - 0.5
zhi = float(max(lst0_.max(), lst_pred.max())) + 0.5

delta   = lst_pred - lst0_
avg_d   = float(delta.mean())
max_d   = float(delta.max())
min_d   = float(delta.min())
new_safe = int((lst_pred < SAFE_LST).sum()) - int((lst0_ < SAFE_LST).sum())

# ── Map helpers ───────────────────────────────────────────────────────────────

_CELL_LAT    = 0.0027
_CELL_LNG    = 0.0032
_LST_PALETTE = ["#313695","#74add1","#e0f3f8","#fee090","#f46d43","#d73027","#a50026"]


def _grid_map(
    lst: np.ndarray, ndvi: np.ndarray,
    site_lat: float, site_lng: float,
    zmin: float, zmax: float,
) -> folium.Map:
    fmap = folium.Map(
        location=[site_lat, site_lng], zoom_start=13,
        tiles="CartoDB positron", control_scale=False, zoom_control=False,
    )
    norm = mcolors.Normalize(vmin=zmin, vmax=zmax)
    cmap = matplotlib.colormaps["RdYlBu_r"]
    half = GRID / 2.0
    for i in range(GRID):
        for j in range(GRID):
            lat_n = site_lat + (half - i)     * _CELL_LAT
            lat_s = site_lat + (half - i - 1) * _CELL_LAT
            lng_w = site_lng + (j - half)     * _CELL_LNG
            lng_e = site_lng + (j - half + 1) * _CELL_LNG
            fill    = mcolors.to_hex(cmap(norm(float(lst[i, j]))))
            is_veg  = bool(_VEG_MASK[i, j])
            folium.Rectangle(
                bounds=[[lat_s, lng_w], [lat_n, lng_e]],
                color="#22C55E" if is_veg else "#cccccc",
                weight=2.5 if is_veg else 0.4,
                fill=True, fill_color=fill, fill_opacity=0.78,
                tooltip=(
                    f"<b>Cell ({i},{j})</b><br>"
                    f"LST: {lst[i,j]:.1f} °C<br>"
                    f"NDVI: {ndvi[i,j]:.3f}"
                    + (" &nbsp;🌿 veg zone" if is_veg else "")
                ),
            ).add_to(fmap)
    return fmap


def _legend_html(zmin: float, zmax: float) -> str:
    gradient = ", ".join(_LST_PALETTE)
    mid = (zmin + zmax) / 2
    return (
        f'<div style="margin:4px 0 14px">'
        f'<div style="height:11px;border-radius:6px;'
        f'background:linear-gradient(90deg,{gradient});border:1px solid #e5e7eb"></div>'
        f'<div style="display:flex;justify-content:space-between;'
        f'font-size:0.68em;color:#6b7280;margin-top:3px">'
        f"<span>{zmin:.0f} °C</span><span>{mid:.0f} °C</span><span>{zmax:.0f} °C</span>"
        f"</div></div>"
    )


# ── Before / After maps ───────────────────────────────────────────────────────

section_label("Before vs. After")

site_row = locs_df[locs_df["site"] == str(st.session_state["sim_site"])].iloc[0]
_slat, _slng = float(site_row["lat"]), float(site_row["lng"])

col_grids, col_sum = st.columns([3, 1], gap="medium")

with col_grids:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<p class="sim-grid-title">Current State</p>', unsafe_allow_html=True)
        st_folium(_grid_map(lst0_, ndvi0_, _slat, _slng, zlo, zhi),
                  height=320, returned_objects=[], key="sim_map_curr")
    with col_b:
        step_lbl = f"After {cur_step} Step{'s' if cur_step != 1 else ''}"
        st.markdown(f'<p class="sim-grid-title">{step_lbl}</p>', unsafe_allow_html=True)
        st_folium(_grid_map(lst_pred, ndvi0_, _slat, _slng, zlo, zhi),
                  height=320, returned_objects=[], key="sim_map_pred")
    st.markdown(_legend_html(zlo, zhi), unsafe_allow_html=True)

with col_sum:
    st.markdown('<p class="sim-grid-title">Summary</p>', unsafe_allow_html=True)
    avg_fg  = "#166534" if avg_d  <= 0 else "#991B1B"
    max_fg  = "#166534" if max_d  <= 0 else "#991B1B"
    safe_bg = "#DCFCE7" if new_safe >= 0 else "#FEE2E2"
    safe_fg = "#166534" if new_safe >= 0 else "#991B1B"

    def _card(label: str, value: str, fg: str, bg: str = "#fff") -> str:
        return (
            f'<div class="sim-card" style="background:{bg}">'
            f'<div class="sim-card-label" style="color:{fg}">{label}</div>'
            f'<div class="sim-card-value" style="color:{fg}">{value}</div>'
            f"</div>"
        )

    st.markdown(
        _card("Avg ΔLST",      f"{avg_d:+.2f}°C",  avg_fg)
        + _card("Max ΔLST",    f"{max_d:+.2f}°C",  max_fg)
        + _card("Min ΔLST",    f"{min_d:+.2f}°C",  "#374151")
        + _card("New safe cells", f"{new_safe:+d}", safe_fg, safe_bg),
        unsafe_allow_html=True,
    )

if _at_max:
    st.info("Already at the final step — click **Run** to restart.")
if step_clicked and not _at_max and st.session_state["sim_ok"]:
    st.caption(
        f"**Step {cur_step}** — LST relaxes toward vegetation equilibrium "
        f"({COOLING:.0f} °C/NDVI unit) then diffuses across Moore neighbours (α = {DIFF})."
    )

# ── LST Trajectory chart ──────────────────────────────────────────────────────

lbl_site = str(st.session_state["sim_site"])
lbl_sea  = str(st.session_state["sim_season"])
lbl_veg  = int(st.session_state["sim_veg_pct"])

section_label(f"LST Trajectory — {lbl_site} · {lbl_sea} · Vegetation {lbl_veg:+d}%")

xs = list(range(len(avgs_)))
fig_ln = go.Figure()
fig_ln.add_hline(y=avgs_[0], line_dash="dot", line_color="#D1D5DB",
                 annotation_text="Baseline", annotation_position="bottom right",
                 annotation_font_color="#9CA3AF")
fig_ln.add_trace(go.Scatter(
    x=xs, y=avgs_, mode="lines+markers", name="Avg LST",
    line=dict(color="#2E7D32", width=2.5), marker=dict(size=7, color="#2E7D32"),
    hovertemplate="Step %{x}<br>Avg LST: %{y:.2f}°C<extra></extra>",
))
if 0 <= cur_step < len(avgs_):
    fig_ln.add_shape(type="line", x0=cur_step, x1=cur_step, y0=0, y1=1,
                     xref="x", yref="paper", line=dict(color="#7C3AED", dash="dash", width=1.5))
    fig_ln.add_annotation(
        x=cur_step, y=avgs_[cur_step],
        text=f"Step {cur_step}: {avgs_[cur_step]:.1f}°C",
        showarrow=True, arrowhead=2, arrowcolor="#7C3AED",
        font=dict(color="#7C3AED", size=11),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="#7C3AED", borderwidth=1, borderpad=4,
    )
fig_ln.update_layout(
    height=220, margin=dict(t=10, b=20, l=0, r=80),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFAFA", hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    xaxis=dict(showgrid=True, gridcolor="#F3F4F6", title=dict(text="Simulation step", font=dict(size=11)), dtick=1),
    yaxis=dict(showgrid=True, gridcolor="#F3F4F6", title=dict(text="Avg LST (°C)", font=dict(size=11))),
)
st.plotly_chart(fig_ln, use_container_width=True)
st.caption(
    f"Green = vegetation-change zone (radius 2.5 cells). "
    f"Cooling coefficient: {COOLING:.0f} °C per NDVI unit. "
    f"Diffusion α = {DIFF}. Safe threshold: {SAFE_LST:.0f} °C."
)

# ── Lotka-Volterra section ────────────────────────────────────────────────────

section_label("Population Dynamics — Native vs Invasive Vegetation")

lv_col1, lv_col2 = st.columns([1, 2])
with lv_col1:
    st.caption("Adjust Lotka-Volterra parameters")
    lv_N0  = st.slider("Initial native density",   0.1, 1.0, 0.8, 0.05)
    lv_I0  = st.slider("Initial invasive density", 0.0, 1.0, 0.2, 0.05)
    lv_rn  = st.slider("Native growth rate",       0.1, 1.0, 0.3, 0.05)
    lv_ri  = st.slider("Invasive growth rate",     0.1, 1.0, 0.5, 0.05)
    lv_alpha = st.slider("α (invasive→native competition)", 0.1, 2.0, 1.2, 0.1)
    lv_beta  = st.slider("β (native→invasive competition)", 0.1, 2.0, 0.8, 0.1)

with lv_col2:
    lv_df = simulate_lv(
        LotkaVolterra(N0=lv_N0, I0=lv_I0, r_native=lv_rn, r_invasive=lv_ri,
                      alpha=lv_alpha, beta=lv_beta),
        steps=300, dt=0.1,
    )
    fig_lv = go.Figure()
    fig_lv.add_trace(go.Scatter(
        x=lv_df["t"], y=lv_df["native"], name="Native",
        line=dict(color="#2E7D32", width=2), mode="lines",
    ))
    fig_lv.add_trace(go.Scatter(
        x=lv_df["t"], y=lv_df["invasive"], name="Invasive",
        line=dict(color="#C62828", width=2), mode="lines",
    ))
    fig_lv.update_layout(
        height=280, margin=dict(t=10, b=20, l=0, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFAFA",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=True, gridcolor="#F3F4F6", title="Time"),
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6", title="Population density",
                   range=[0, 1.1]),
    )
    st.plotly_chart(fig_lv, use_container_width=True)
    final_n = lv_df["native"].iloc[-1]
    final_i = lv_df["invasive"].iloc[-1]
    winner  = "Native 🌿" if final_n > final_i else "Invasive ⚠️" if final_i > final_n else "Coexistence"
    st.caption(f"Outcome at t=30: Native={final_n:.3f}, Invasive={final_i:.3f} → **{winner}**")

# ── Save scenario ─────────────────────────────────────────────────────────────

if save_clicked:
    if not st.session_state["sim_ok"]:
        st.warning("Run a simulation first.")
    else:
        scenarios_dir = _ROOT / "data" / "scenarios"
        scenarios_dir.mkdir(exist_ok=True)
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        fpath = scenarios_dir / f"scenario_{ts}.json"
        fpath.write_text(json.dumps({
            "timestamp": ts, "site": lbl_site, "season": lbl_sea,
            "veg_pct": lbl_veg, "n_steps": int(st.session_state["sim_n_steps"]),
            "avg_delta_lst": round(avg_d, 3), "max_delta_lst": round(max_d, 3),
            "min_delta_lst": round(min_d, 3), "new_safe_cells": new_safe,
            "avg_lst_trajectory": [round(v, 3) for v in avgs_],
        }, indent=2))
        st.success(f"Saved → `data/scenarios/scenario_{ts}.json`")