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
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402
from ui import page_hero, section_label, set_page_config  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────

set_page_config(title="Optimization")

page_hero(
    title="🧬 Planting Optimization",
    subtitle=(
        "Genetic algorithm to find the optimal cells to plant for maximum "
        "heat reduction — evidence-based planting plans."
    ),
    pills=[
        "🧬 Genetic Algorithm",
        "🌳 Optimal Planting",
        "🌡️ Heat Reduction",
        "📍 8 Sites",
    ],
    emoji="🧬",
)

# ── Data ──────────────────────────────────────────────────────────────────────

MOCK = _ROOT / "data" / "mock"


@st.cache_data
def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(MOCK / "site_monthly.csv"), pd.read_csv(MOCK / "site_locations.csv")


monthly_df, locs_df = _load()
SITES = sorted(monthly_df["site"].unique())

# ── Constants ─────────────────────────────────────────────────────────────────

GRID = 10
SEASON_MONTHS: dict[str, list[int]] = {
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Autumn": [9, 10, 11],
    "Winter": [12, 1, 2],
}
COOLING = 9.0      # °C per unit NDVI increase
VEG_BOOST = 0.20   # NDVI increase assumed per planted cell
_CELL_LAT = 0.0027
_CELL_LNG = 0.0032

# ── Grid init (same seed as simulator for consistency) ────────────────────────


@st.cache_data
def _init_grid(site: str, season: str) -> tuple[np.ndarray, np.ndarray]:
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


# ── Genetic algorithm ─────────────────────────────────────────────────────────


def _fitness(ind: np.ndarray, lst: np.ndarray, ndvi: np.ndarray) -> float:
    """
    Fitness = Σ over planted cells of (norm_LST × low_NDVI_factor)
              + adjacency bonus (contiguous clusters improve microclimate).
    Maximise → best cells are hot, bare ground, and clustered.
    """
    mask = ind.reshape(GRID, GRID)
    lst_n = (lst - lst.min()) / (lst.max() - lst.min() + 1e-9)

    base = float(np.sum(lst_n[mask] * (1.0 - ndvi[mask])))

    adj = 0
    for i in range(GRID):
        for j in range(GRID):
            if mask[i, j]:
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < GRID and 0 <= nj < GRID and mask[ni, nj]:
                        adj += 1

    return base + adj * 0.05


def _expected_delta_lst(
    mask: np.ndarray, ndvi: np.ndarray, boost: float = VEG_BOOST
) -> float:
    """Mean LST change (negative = cooling) if planted cells gain +boost NDVI."""
    ndvi_new = ndvi.copy()
    ndvi_new[mask] = np.clip(ndvi_new[mask] * (1.0 + boost), 0.01, 0.99)
    return -COOLING * float((ndvi_new - ndvi).mean())


def _tournament(
    pop: list[np.ndarray], fits: list[float], k: int, rng: np.random.Generator
) -> np.ndarray:
    idx = rng.choice(len(pop), k, replace=False)
    return pop[int(max(idx, key=lambda i: fits[i]))].copy()


def _crossover(
    p1: np.ndarray, p2: np.ndarray, budget: int, rng: np.random.Generator
) -> np.ndarray:
    combined = np.where(p1 | p2)[0]
    n = len(combined)
    chosen = (
        rng.choice(combined, budget, replace=False)
        if n >= budget
        else rng.choice(len(p1), budget, replace=False)
    )
    child = np.zeros(len(p1), dtype=bool)
    child[chosen] = True
    return child


def _mutate(ind: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    new = ind.copy()
    planted = np.where(ind)[0]
    unplanted = np.where(~ind)[0]
    new[rng.choice(planted)] = False
    new[rng.choice(unplanted)] = True
    return new


def _run_ga(
    lst: np.ndarray,
    ndvi: np.ndarray,
    budget: int,
    n_gen: int,
    pop_sz: int,
    progress_bar: st.delta_generator.DeltaGenerator,
) -> tuple[list[float], list[tuple[float, np.ndarray]]]:
    G2 = GRID * GRID
    rng = np.random.default_rng()

    pop = []
    for _ in range(pop_sz):
        ind = np.zeros(G2, dtype=bool)
        ind[rng.choice(G2, budget, replace=False)] = True
        pop.append(ind)

    history: list[float] = []
    seen: set[bytes] = set()
    top5: list[tuple[float, np.ndarray]] = []

    for gen in range(n_gen):
        fits = [_fitness(ind, lst, ndvi) for ind in pop]
        best_fit = max(fits)
        history.append(best_fit)

        for f, ind in sorted(zip(fits, pop), key=lambda x: -x[0])[:5]:
            key = ind.tobytes()
            if key not in seen:
                seen.add(key)
                top5.append((f, ind.copy()))
                top5.sort(key=lambda x: -x[0])
                top5 = top5[:5]

        best_idx = int(np.argmax(fits))
        new_pop: list[np.ndarray] = [pop[best_idx].copy()]

        while len(new_pop) < pop_sz:
            p1 = _tournament(pop, fits, k=3, rng=rng)
            p2 = _tournament(pop, fits, k=3, rng=rng)
            child = _crossover(p1, p2, budget, rng)
            if rng.random() < 0.25:
                child = _mutate(child, rng)
            new_pop.append(child)

        pop = new_pop
        progress_bar.progress(
            (gen + 1) / n_gen,
            text=f"Generation {gen + 1} / {n_gen}  ·  best fitness: {best_fit:.3f}",
        )

    return history, top5


# ── Map builder ───────────────────────────────────────────────────────────────


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
    .top5-row {
        display:flex; gap:8px; align-items:center;
        padding:8px 12px; border-radius:8px; margin-bottom:4px;
        background:#F9FAFB; border:1px solid #F3F4F6;
        font-size:0.84em; cursor:pointer;
    }
    .top5-row.selected { background:#DCFCE7; border-color:#BBF7D0; }
    .top5-badge {
        background:#2E7D32; color:#fff; border-radius:999px;
        padding:2px 9px; font-size:0.72em; font-weight:700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────

for _k, _v in {
    "opt_ok": False,
    "opt_site": SITES[0],
    "opt_season": "Summer",
    "opt_budget": 10,
    "opt_history": None,
    "opt_top5": None,
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
    site: str = st.selectbox("Site", SITES, index=SITES.index("Jordan Valley"))
with c2:
    season: str = st.selectbox("Season", list(SEASON_MONTHS.keys()), index=1)
with c3:
    budget: int = st.slider("Planting budget (cells)", 1, 50, 10)
with c4:
    n_gen: int = st.number_input("Generations", min_value=10, max_value=300, value=60)
with c5:
    pop_sz: int = st.number_input("Population", min_value=10, max_value=100, value=30)

run_clicked = st.button(
    "▶ Run Genetic Algorithm", type="primary", use_container_width=False
)

# ── GA execution ──────────────────────────────────────────────────────────────

if run_clicked:
    lst0, ndvi0 = _init_grid(site, season)
    _pb = st.progress(0, text="Initialising population…")
    history, top5 = _run_ga(lst0, ndvi0, budget, n_gen, pop_sz, _pb)
    _pb.empty()
    st.session_state.update(
        opt_ok=True,
        opt_site=site,
        opt_season=season,
        opt_budget=budget,
        opt_history=history,
        opt_top5=top5,
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
top5_: list[tuple[float, np.ndarray]] = st.session_state["opt_top5"]
sel_: int = int(st.session_state["opt_selected"])
lst_: np.ndarray = st.session_state["opt_lst"]
ndvi_: np.ndarray = st.session_state["opt_ndvi"]
res_site: str = str(st.session_state["opt_site"])
res_season: str = str(st.session_state["opt_season"])
res_budget: int = int(st.session_state["opt_budget"])

best_mask = top5_[sel_][1].reshape(GRID, GRID)
site_row = locs_df[locs_df["site"] == res_site].iloc[0]
slat, slng = float(site_row["lat"]), float(site_row["lng"])

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
    fig_conv = go.Figure()
    fig_conv.add_trace(
        go.Scatter(
            x=list(range(1, len(history_) + 1)),
            y=history_,
            mode="lines",
            line=dict(color="#2E7D32", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(46,125,50,0.07)",
            hovertemplate="Gen %{x}<br>Fitness: %{y:.4f}<extra></extra>",
        )
    )
    fig_conv.update_layout(
        height=340,
        margin=dict(t=10, b=30, l=0, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FAFAFA",
        xaxis=dict(
            showgrid=True,
            gridcolor="#F3F4F6",
            title=dict(text="Generation", font=dict(size=11)),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#F3F4F6",
            title=dict(text="Best fitness", font=dict(size=11)),
        ),
    )
    st.plotly_chart(fig_conv, use_container_width=True)

# ── Summary metrics ───────────────────────────────────────────────────────────

best_delta = _expected_delta_lst(best_mask, ndvi_)
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Expected Avg ΔLST", f"{best_delta:+.2f} °C")
mc2.metric("Cells planted", f"{int(best_mask.sum())} / {res_budget}")
mc3.metric("Fitness score", f"{top5_[sel_][0]:.4f}")
mc4.metric("Generations run", str(len(history_)))

st.write("")

# ── Top 5 solutions table ─────────────────────────────────────────────────────

section_label("Top 5 Solutions")

col_t, col_sel = st.columns([3, 1], gap="medium")

with col_t:
    table_rows = []
    for rank, (fit, ind) in enumerate(top5_):
        m = ind.reshape(GRID, GRID)
        d = _expected_delta_lst(m, ndvi_)
        table_rows.append(
            {
                "#": rank + 1,
                "ΔLST (avg)": f"{d:+.2f} °C",
                "Cells": int(ind.sum()),
                "Fitness": round(fit, 4),
            }
        )
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
        options=list(range(1, len(top5_) + 1)),
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
            coords.append(
                {
                    "site": res_site,
                    "cell_row": i,
                    "cell_col": j,
                    "lat": round(slat + (GRID / 2 - i - 0.5) * _CELL_LAT, 6),
                    "lng": round(slng + (j - GRID / 2 + 0.5) * _CELL_LNG, 6),
                    "lst_current_C": round(float(lst_[i, j]), 2),
                    "ndvi_current": round(float(ndvi_[i, j]), 3),
                    "expected_delta_lst_C": round(
                        -COOLING
                        * float(
                            np.clip(ndvi_[i, j] * (1 + VEG_BOOST), 0.01, 0.99)
                            - ndvi_[i, j]
                        ),
                        3,
                    ),
                }
            )

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
