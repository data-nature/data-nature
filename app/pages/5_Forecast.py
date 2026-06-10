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

set_page_config(title="Forecast")

page_hero(
    title="📈 Temperature Forecast",
    subtitle="7-month Land Surface Temperature forecast using machine learning trained on Landsat time-series data (2000–2026).",
    pills=["🤖 3 ML Models", "🌡️ LST Prediction", "📅 7-Month Horizon", "📍 8 Sites"],
    emoji="📈",
)

# ── Data loading (real data with mock fallback) ───────────────────────────────

PROCESSED = _ROOT / "data" / "processed"
MOCK      = _ROOT / "data" / "mock"


def _load(filename: str, date_col: str = "date") -> tuple[pd.DataFrame, bool]:
    """Load from processed/, fall back to mock/. Returns (df, is_real)."""
    real_path = PROCESSED / filename
    if real_path.exists():
        try:
            df = pd.read_csv(real_path, parse_dates=[date_col])
            if len(df) > 0:
                return df, True
        except Exception:
            pass
    df = pd.read_csv(MOCK / filename, parse_dates=[date_col])
    return df, False


@st.cache_data
def _load_history() -> tuple[pd.DataFrame, bool]:
    return _load("lst_history.csv")


@st.cache_data
def _load_forecast() -> tuple[pd.DataFrame, bool]:
    return _load("lst_forecast.csv")


@st.cache_data
def _load_metrics() -> tuple[pd.DataFrame, bool]:
    # model_metrics.csv has no date column
    real_path = PROCESSED / "model_metrics.csv"
    if real_path.exists():
        try:
            df = pd.read_csv(real_path)
            if len(df) > 0:
                return df, True
        except Exception:
            pass
    return pd.read_csv(MOCK / "model_metrics.csv"), False


hist_df,  hist_real  = _load_history()
fc_df,    fc_real    = _load_forecast()
met_df,   met_real   = _load_metrics()

using_real = hist_real and fc_real and met_real
if not using_real:
    st.caption("ℹ️ Showing mock data — run the forecast pipeline to load real results.")

SITES  = sorted(fc_df["site"].unique())
MODELS = [m for m in ["Random Forest", "Gradient Boosting", "LSTM"]
          if m in fc_df["model"].unique()]

# ── Controls ──────────────────────────────────────────────────────────────────

col_site, col_model, col_compare = st.columns([2, 2, 1])
with col_site:
    site = st.selectbox("Site", SITES, index=0)
with col_model:
    model = st.selectbox("Model", MODELS, index=MODELS.index("LSTM") if "LSTM" in MODELS else 0)
with col_compare:
    st.write("")
    compare = st.toggle("Compare all models")

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .conf-row { display: flex; gap: 10px; margin: 6px 0 24px; flex-wrap: wrap; }
    .conf-card {
        flex: 1; min-width: 90px;
        border-radius: 12px; padding: 14px 10px;
        text-align: center; border: 1px solid #e5e7eb;
        background: #fff;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .conf-day   { font-size: 0.68em; color: #6b7280; font-weight: 600;
                  text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 4px; }
    .conf-temp  { font-size: 1.35em; font-weight: 800; color: #1C1B18; margin: 4px 0; }
    .conf-badge {
        display: inline-block; border-radius: 999px;
        padding: 2px 10px; font-size: 0.68em; font-weight: 700;
        margin-top: 6px;
    }
    .conf-high  { background: #DCFCE7; color: #166534; }
    .conf-med   { background: #FEF9C3; color: #854D0E; }
    .conf-low   { background: #FEE2E2; color: #991B1B; }
    .metrics-table { width: 100%; border-collapse: collapse; font-size: 0.88em; }
    .metrics-table th {
        background: #F0FDF4; color: #166534;
        font-weight: 700; font-size: 0.72em; letter-spacing: 0.06em;
        text-transform: uppercase; padding: 8px 12px; border-bottom: 2px solid #BBF7D0;
    }
    .metrics-table td { padding: 8px 12px; border-bottom: 1px solid #F3F4F6; color: #374151; }
    .metrics-table tr:hover td { background: #F9FAFB; }
    .best-model td { font-weight: 700; color: #166534 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Main chart ────────────────────────────────────────────────────────────────

section_label("Forecast Chart")

site_hist = hist_df[hist_df["site"] == site].sort_values("date")
site_fc   = fc_df[fc_df["site"] == site].sort_values("date")

# Show last 24 months of history for readability
if len(site_hist) > 24:
    site_hist = site_hist.tail(24)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=site_hist["date"],
        y=site_hist["lst"],
        name="Historical LST",
        line={"color": "#9CA3AF", "width": 2},
        mode="lines",
    )
)

MODEL_COLORS = {
    "Random Forest":     "#2E7D32",
    "Gradient Boosting": "#1976D2",
    "LSTM":              "#7B1FA2",
}

models_to_draw = MODELS if compare else [model]

for m in models_to_draw:
    mfc   = site_fc[site_fc["model"] == m].sort_values("date")
    color = MODEL_COLORS.get(m, "#555")
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    rgba_band = f"rgba({r},{g},{b},0.10)"

    fig.add_trace(
        go.Scatter(
            x=pd.concat([mfc["date"], mfc["date"][::-1]]),
            y=pd.concat([mfc["lst_high"], mfc["lst_low"][::-1]]),
            fill="toself",
            fillcolor=rgba_band,
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
            name=f"{m} band",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=mfc["date"],
            y=mfc["lst_forecast"],
            name=m,
            line={"color": color, "width": 2.5, "dash": "dot" if m != model else "solid"},
            mode="lines+markers",
            marker={"size": 6, "color": color},
            hovertemplate="%{x|%b %Y}<br>LST: %{y:.1f}°C<extra>" + m + "</extra>",
        )
    )

split_date = site_fc["date"].min().isoformat()
fig.add_shape(
    type="line",
    x0=split_date, x1=split_date,
    y0=0, y1=1,
    xref="x", yref="paper",
    line={"color": "#D1D5DB", "dash": "dash", "width": 1.5},
)
fig.add_annotation(
    x=split_date, y=1,
    xref="x", yref="paper",
    text="Forecast →",
    showarrow=False,
    xanchor="left",
    font={"color": "#6B7280", "size": 11},
)

fig.update_layout(
    height=400,
    margin={"t": 20, "b": 20, "l": 0, "r": 0},
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FAFAFA",
    hovermode="x unified",
    legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    xaxis={
        "showgrid": True,
        "gridcolor": "#F3F4F6",
        "tickformat": "%b %Y",
        "title": None,
    },
    yaxis={
        "showgrid": True,
        "gridcolor": "#F3F4F6",
        "title": {"text": "LST (°C)", "font": {"size": 12}},
    },
)

st.plotly_chart(fig, use_container_width=True)

# ── Confidence cards ──────────────────────────────────────────────────────────

section_label("Forecast Confidence")

sel_fc = site_fc[site_fc["model"] == model].sort_values("date").reset_index(drop=True)


def _confidence(row: pd.Series) -> tuple[str, str]:
    band = row["lst_high"] - row["lst_low"]
    if band < 2.5:
        return "High",   "conf-high"
    if band < 3.5:
        return "Medium", "conf-med"
    return "Low",        "conf-low"


cards = ""
for _, row in sel_fc.iterrows():
    label, css = _confidence(row)
    month = row["date"].strftime("%b %Y")
    cards += (
        f'<div class="conf-card">'
        f'<div class="conf-day">{month}</div>'
        f'<div class="conf-temp">{row["lst_forecast"]:.1f}°</div>'
        f'<div class="conf-badge {css}">{label}</div>'
        f"</div>"
    )

st.markdown(f'<div class="conf-row">{cards}</div>', unsafe_allow_html=True)

# ── Model performance ─────────────────────────────────────────────────────────

section_label("Model Performance — Validation Set (2024–2026)")

# Try per-site metrics; fall back to ALL if site not in metrics
site_met = met_df[met_df["site"] == site].copy()
if site_met.empty:
    site_met = met_df[met_df["site"] == "ALL"].copy()

# Only show the 3 main models
site_met = site_met[site_met["model"].isin(MODELS)]

if not site_met.empty:
    best_model = site_met.loc[site_met["mae"].idxmin(), "model"]
    rows_html = ""
    for _, row in site_met.iterrows():
        is_best = "best-model" if row["model"] == best_model else ""
        star = " ★" if row["model"] == best_model else ""
        rows_html += (
            f'<tr class="{is_best}">'
            f"<td>{row['model']}{star}</td>"
            f"<td>{row['mae']:.2f} °C</td>"
            f"<td>{row['rmse']:.2f} °C</td>"
            f"<td>{row['r2']:.3f}</td>"
            f"</tr>"
        )

    st.markdown(
        f"""
        <table class="metrics-table">
          <thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>R²</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )
    st.caption("★ Best model for this site · Trained on 2000–2023 · Validated on 2024–2026")

# ── Heat alert ────────────────────────────────────────────────────────────────

st.write("")
if not sel_fc.empty:
    max_fc  = sel_fc["lst_forecast"].max()
    max_mon = sel_fc.loc[sel_fc["lst_forecast"].idxmax(), "date"].strftime("%B %Y")

    if max_fc >= 38:
        st.warning(
            f"⚠️ **Heat alert:** {model} forecasts {max_fc:.1f}°C at **{site}** in {max_mon}. "
            "Consider scheduling a field visit."
        )
    else:
        st.info(
            f"✅ No extreme heat events forecast at **{site}** in the next 7 months "
            f"(peak: {max_fc:.1f}°C)."
        )