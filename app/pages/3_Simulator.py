from __future__ import annotations

import json
import sys
from datetime import datetime
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
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402
from ui import page_hero, section_label, set_page_config  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────

set_page_config(title="Simulator")

page_hero(
    title="🔬 What-If Simulator",
    subtitle=(
        "Cellular-automaton heat-diffusion model — test vegetation cover changes "
        "before going to the field."
    ),
    pills=[
        "🔄 Cellular Automaton",
        "🌱 Vegetation Scenarios",
        "🌡️ Heat Diffusion",
        "📍 8 Sites",
    ],
    emoji="🔬",
)

# ── Data ──────────────────────────────────────────────────────────────────────

MOCK = _ROOT / "data" / "mock"


@st.cache_data
def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(MOCK / "site_monthly.csv"), pd.read_csv(MOCK / "site_locations.csv")


monthly_df, locs_df = _load()
SITES = sorted(monthly_df["site"].unique())

# ── Model constants ───────────────────────────────────────────────────────────

GRID = 10
SEASON_MONTHS: dict[str, list[int]] = {
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Autumn": [9, 10, 11],
    "Winter": [12, 1, 2],
}
SEASON_HEAT: dict[str, float] = {
    "Spring": 0.0,
    "Summer": 0.4,
    "Autumn": -0.1,
    "Winter": -0.3,
}
SAFE_LST = 30.0   # °C — threshold for "safe zone"
COOLING = 9.0     # °C cooling per unit NDVI increase
DIFF = 0.30       # fraction of LST diffused to Moore neighbours per step
RELAX = 0.35      # relaxation rate toward vegetation-adjusted equilibrium

# ── Cellular automaton helpers ────────────────────────────────────────────────


def _make_veg_mask() -> np.ndarray:
    """Circular vegetation-change zone (radius 2.5 cells) centred on the grid."""
    cy, cx = (GRID - 1) / 2.0, (GRID - 1) / 2.0
    mask = np.zeros((GRID, GRID), dtype=bool)
    for i in range(GRID):
        for j in range(GRID):
            if (i - cy) ** 2 + (j - cx) ** 2 <= 2.5**2:
                mask[i, j] = True
    return mask


_VEG_MASK: np.ndarray = _make_veg_mask()


def _diffuse(lst: np.ndarray) -> np.ndarray:
    """Moore-neighbourhood (8-cell) heat diffusion — pure numpy, no scipy."""
    padded = np.pad(lst, 1, mode="edge")
    s = np.zeros_like(lst)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            s += padded[1 + di : GRID + 1 + di, 1 + dj : GRID + 1 + dj]
    return (1 - DIFF) * lst + DIFF * (s / 8.0)


@st.cache_data
def _init_grid(site: str, season: str) -> tuple[np.ndarray, np.ndarray]:
    """Build a reproducible 10×10 LST/NDVI grid from site monthly statistics."""
    months = SEASON_MONTHS[season]
    rows = monthly_df[
        (monthly_df["site"] == site) & (monthly_df["month"].isin(months))
    ]
    base_lst = rows["lst"].mean()
    base_ndvi = rows["ndvi"].mean()

    seed = sum(ord(c) for c in site + season) % (2**31)
    rng = np.random.default_rng(seed)
    cy, cx = (GRID - 1) / 2.0, (GRID - 1) / 2.0

    lst_g = np.empty((GRID, GRID))
    ndvi_g = np.empty((GRID, GRID))
    for i in range(GRID):
        for j in range(GRID):
            d = np.sqrt((i - cy) ** 2 + (j - cx) ** 2) / (GRID * 0.65)
            lst_g[i, j] = base_lst + 3.5 * d + rng.normal(0.0, 0.6)
            ndvi_g[i, j] = base_ndvi - 0.10 * d + rng.normal(0.0, 0.022)

    return lst_g, np.clip(ndvi_g, 0.01, 0.99)


def _run_sim(
    lst0: np.ndarray,
    ndvi0: np.ndarray,
    veg_pct: float,
    n: int,
    season: str,
) -> tuple[list[np.ndarray], list[float]]:
    """
    Run the CA for n steps.
    Returns (grids, avg_lst_per_step) where grids[0] = initial state.
    Physics:
      1. Vegetation change shifts NDVI in the mask zone.
      2. LST relaxes toward the NDVI-adjusted equilibrium each step.
      3. Moore-neighbourhood heat diffusion spreads temperature.
    """
    ndvi = ndvi0.copy()
    ndvi[_VEG_MASK] = np.clip(ndvi[_VEG_MASK] * (1.0 + veg_pct / 100.0), 0.01, 0.99)

    lst_eq = lst0 - COOLING * (ndvi - ndvi0)   # new thermal equilibrium
    sh = SEASON_HEAT[season] * 0.02             # tiny seasonal offset per step

    lst = lst0.copy()
    grids: list[np.ndarray] = [lst.copy()]
    avgs: list[float] = [float(lst.mean())]

    for _ in range(n):
        lst = lst + RELAX * (lst_eq - lst)
        lst = _diffuse(lst)
        lst = lst + sh
        grids.append(lst.copy())
        avgs.append(float(lst.mean()))

    return grids, avgs


# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────

for _k, _v in {
    "sim_ok": False,
    "sim_step": 0,
    "sim_site": SITES[0],
    "sim_season": "Summer",
    "sim_veg_pct": 20,
    "sim_n_steps": 5,
    "sim_lst0": None,
    "sim_ndvi0": None,
    "sim_grids": None,
    "sim_avgs": None,
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
        "Vegetation cover change (%)",
        -50,
        100,
        20,
        step=5,
        help="Positive = add vegetation  ·  Negative = remove",
    )
with c3:
    season: str = st.selectbox("Season", list(SEASON_MONTHS.keys()), index=1)
with c4:
    n_steps: int = st.number_input("Steps", min_value=1, max_value=20, value=5)

col_run, col_step, col_save = st.columns([1.2, 1, 1])
run_clicked = col_run.button("▶ Run Simulation", type="primary", use_container_width=True)
step_clicked = col_step.button("⏭ Step", use_container_width=True)
save_clicked = col_save.button("💾 Save Scenario", use_container_width=True)

# ── Simulation execution ──────────────────────────────────────────────────────


def _start(s: str, sea: str, vp: float, ns: int) -> None:
    lst0, ndvi0 = _init_grid(s, sea)
    grids, avgs = _run_sim(lst0, ndvi0, vp, ns, sea)
    st.session_state.update(
        sim_ok=True,
        sim_step=ns,
        sim_site=s,
        sim_season=sea,
        sim_veg_pct=int(vp),
        sim_n_steps=ns,
        sim_lst0=lst0,
        sim_ndvi0=ndvi0,
        sim_grids=grids,
        sim_avgs=avgs,
    )


_at_max = False

if run_clicked:
    _start(site, season, float(veg_pct), n_steps)
elif step_clicked and st.session_state["sim_ok"]:
    _cur = int(st.session_state["sim_step"])
    _total = len(st.session_state["sim_grids"]) - 1  # type: ignore[arg-type]
    if _cur < _total:
        st.session_state["sim_step"] = _cur + 1
    else:
        _at_max = True

if not st.session_state["sim_ok"]:
    _start(site, season, float(veg_pct), n_steps)

# ── Read current state ────────────────────────────────────────────────────────

lst0_: np.ndarray = st.session_state["sim_lst0"]
ndvi0_: np.ndarray = st.session_state["sim_ndvi0"]
grids_: list[np.ndarray] = st.session_state["sim_grids"]
avgs_: list[float] = st.session_state["sim_avgs"]
cur_step: int = int(st.session_state["sim_step"])
lst_pred: np.ndarray = grids_[cur_step]

zlo = float(min(lst0_.min(), lst_pred.min())) - 0.5
zhi = float(max(lst0_.max(), lst_pred.max())) + 0.5

delta = lst_pred - lst0_
avg_d = float(delta.mean())
max_d = float(delta.max())
min_d = float(delta.min())
new_safe = int((lst_pred < SAFE_LST).sum()) - int((lst0_ < SAFE_LST).sum())

# ── Map helpers ───────────────────────────────────────────────────────────────

# Each cell ≈ 300 m × 300 m expressed in degrees at ~33 °N
_CELL_LAT = 0.0027
_CELL_LNG = 0.0032
_LST_PALETTE = ["#313695", "#74add1", "#e0f3f8", "#fee090", "#f46d43", "#d73027", "#a50026"]


def _grid_map(
    lst: np.ndarray,
    ndvi: np.ndarray,
    site_lat: float,
    site_lng: float,
    zmin: float,
    zmax: float,
) -> folium.Map:
    """Render the 10×10 CA grid as coloured rectangles on a real map."""
    fmap = folium.Map(
        location=[site_lat, site_lng],
        zoom_start=13,
        tiles="CartoDB positron",
        control_scale=False,
        zoom_control=False,
    )

    norm = mcolors.Normalize(vmin=zmin, vmax=zmax)
    cmap = matplotlib.colormaps["RdYlBu_r"]
    half = GRID / 2.0

    for i in range(GRID):
        for j in range(GRID):
            lat_n = site_lat + (half - i) * _CELL_LAT
            lat_s = site_lat + (half - i - 1) * _CELL_LAT
            lng_w = site_lng + (j - half) * _CELL_LNG
            lng_e = site_lng + (j - half + 1) * _CELL_LNG
            fill = mcolors.to_hex(cmap(norm(float(lst[i, j]))))
            is_veg = bool(_VEG_MASK[i, j])
            folium.Rectangle(
                bounds=[[lat_s, lng_w], [lat_n, lng_e]],
                color="#22C55E" if is_veg else "#cccccc",
                weight=2.5 if is_veg else 0.4,
                fill=True,
                fill_color=fill,
                fill_opacity=0.78,
                tooltip=(
                    f"<b>Cell ({i},{j})</b><br>"
                    f"LST: {lst[i, j]:.1f} °C<br>"
                    f"NDVI: {ndvi[i, j]:.3f}"
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
        st_folium(
            _grid_map(lst0_, ndvi0_, _slat, _slng, zlo, zhi),
            height=320,
            returned_objects=[],
            key="sim_map_curr",
        )
    with col_b:
        step_lbl = f"After {cur_step} Step{'s' if cur_step != 1 else ''}"
        st.markdown(
            f'<p class="sim-grid-title">{step_lbl}</p>', unsafe_allow_html=True
        )
        st_folium(
            _grid_map(lst_pred, ndvi0_, _slat, _slng, zlo, zhi),
            height=320,
            returned_objects=[],
            key="sim_map_pred",
        )
    st.markdown(_legend_html(zlo, zhi), unsafe_allow_html=True)

with col_sum:
    st.markdown('<p class="sim-grid-title">Summary</p>', unsafe_allow_html=True)

    avg_fg = "#166534" if avg_d <= 0 else "#991B1B"
    max_fg = "#166534" if max_d <= 0 else "#991B1B"
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
        _card("Avg ΔLST", f"{avg_d:+.2f}°C", avg_fg)
        + _card("Max ΔLST", f"{max_d:+.2f}°C", max_fg)
        + _card("Min ΔLST", f"{min_d:+.2f}°C", "#374151")
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

# ── LST trajectory chart ──────────────────────────────────────────────────────

lbl_site = str(st.session_state["sim_site"])
lbl_sea = str(st.session_state["sim_season"])
lbl_veg = int(st.session_state["sim_veg_pct"])

section_label(f"LST Trajectory — {lbl_site} · {lbl_sea} · Vegetation {lbl_veg:+d}%")

xs = list(range(len(avgs_)))
fig_ln = go.Figure()

fig_ln.add_hline(
    y=avgs_[0],
    line_dash="dot",
    line_color="#D1D5DB",
    annotation_text="Baseline",
    annotation_position="bottom right",
    annotation_font_color="#9CA3AF",
)

fig_ln.add_trace(
    go.Scatter(
        x=xs,
        y=avgs_,
        mode="lines+markers",
        name="Avg LST",
        line=dict(color="#2E7D32", width=2.5),
        marker=dict(size=7, color="#2E7D32"),
        hovertemplate="Step %{x}<br>Avg LST: %{y:.2f}°C<extra></extra>",
    )
)

if 0 <= cur_step < len(avgs_):
    fig_ln.add_shape(
        type="line",
        x0=cur_step,
        x1=cur_step,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="#7C3AED", dash="dash", width=1.5),
    )
    fig_ln.add_annotation(
        x=cur_step,
        y=avgs_[cur_step],
        text=f"Step {cur_step}: {avgs_[cur_step]:.1f}°C",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#7C3AED",
        font=dict(color="#7C3AED", size=11),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#7C3AED",
        borderwidth=1,
        borderpad=4,
    )

fig_ln.update_layout(
    height=220,
    margin=dict(t=10, b=20, l=0, r=80),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FAFAFA",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    xaxis=dict(
        showgrid=True,
        gridcolor="#F3F4F6",
        title=dict(text="Simulation step", font=dict(size=11)),
        dtick=1,
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="#F3F4F6",
        title=dict(text="Avg LST (°C)", font=dict(size=11)),
    ),
)
st.plotly_chart(fig_ln, use_container_width=True)
st.caption(
    f"Green = vegetation-change zone (radius 2.5 cells). "
    f"Cooling coefficient: {COOLING:.0f} °C per NDVI unit. "
    f"Diffusion α = {DIFF}. Safe threshold: {SAFE_LST:.0f} °C."
)

# ── Save scenario ─────────────────────────────────────────────────────────────

if save_clicked:
    if not st.session_state["sim_ok"]:
        st.warning("Run a simulation first.")
    else:
        scenarios_dir = _ROOT / "data" / "scenarios"
        scenarios_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fpath = scenarios_dir / f"scenario_{ts}.json"
        fpath.write_text(
            json.dumps(
                {
                    "timestamp": ts,
                    "site": lbl_site,
                    "season": lbl_sea,
                    "veg_pct": lbl_veg,
                    "n_steps": int(st.session_state["sim_n_steps"]),
                    "avg_delta_lst": round(avg_d, 3),
                    "max_delta_lst": round(max_d, 3),
                    "min_delta_lst": round(min_d, 3),
                    "new_safe_cells": new_safe,
                    "avg_lst_trajectory": [round(v, 3) for v in avgs_],
                },
                indent=2,
            )
        )
        st.success(f"Saved → `data/scenarios/scenario_{ts}.json`")
