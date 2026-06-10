from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_APP  = _ROOT / "app"
for _p in (str(_ROOT / "src"), str(_APP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd                                                   # noqa: E402
import plotly.graph_objects as go                                     # noqa: E402
import streamlit as st                                                # noqa: E402
from ui import page_hero, section_label, set_page_config             # noqa: E402
from data_nature.reports.generator import build_pdf, conclusions     # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────

set_page_config(title="Reports")

page_hero(
    title="📋 PDF Report Export",
    subtitle=(
        "Compile anomaly detections, forecasts, and planting plans into a "
        "professional document for sharing with colleagues and decision-makers."
    ),
    pills=["📄 PDF Export", "🚨 Anomaly Reports", "📈 Forecasts", "🌿 Planting Plans"],
    emoji="📋",
)

# ── Data loading (real data with mock fallback) ───────────────────────────────

PROCESSED = _ROOT / "data" / "processed"
MOCK      = _ROOT / "data" / "mock"


@st.cache_data
def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    real_anom    = PROCESSED / "anomalies.csv"
    real_ts      = PROCESSED / "lst_timeseries.csv"
    real_monthly = PROCESSED / "site_monthly.csv"

    if real_anom.exists() and real_ts.exists() and real_monthly.exists():
        try:
            anom    = pd.read_csv(real_anom,    parse_dates=["date"])
            ts      = pd.read_csv(real_ts,      parse_dates=["date"])
            monthly = pd.read_csv(real_monthly)
            if len(anom) > 0:
                return anom, ts, monthly, True
        except Exception:
            pass

    anom    = pd.read_csv(MOCK / "anomalies.csv",      parse_dates=["date"])
    ts      = pd.read_csv(MOCK / "lst_timeseries.csv", parse_dates=["date"])
    monthly = pd.read_csv(MOCK / "site_monthly.csv")
    return anom, ts, monthly, False


anom_df, ts_df, monthly_df, using_real = _load()

if not using_real:
    st.caption("ℹ️ Showing mock data — run the anomaly detection pipeline to load real results.")

ALL_SITES: list[str] = sorted(anom_df["site"].unique().tolist())
DATE_MIN:  date      = anom_df["date"].min().date()
DATE_MAX:  date      = anom_df["date"].max().date()

REPORT_TYPES = [
    "📅 Weekly Summary",
    "📆 Monthly Summary",
    "🚨 Anomaly Report",
    "🌱 Optimization Report",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _filter(sites: list[str], d0: date, d1: date) -> pd.DataFrame:
    t0, t1 = pd.Timestamp(d0), pd.Timestamp(d1)
    return anom_df[
        anom_df["site"].isin(sites)
        & (anom_df["date"] >= t0)
        & (anom_df["date"] <= t1)
    ].sort_values("z_score", ascending=False)


def _sev_color(sev: str) -> str:
    return {"Critical": "#DC2626", "Severe": "#F97316", "Mild": "#F59E0B"}.get(sev, "#6b7280")


# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.doc-wrap { background:#f0f2f6; padding:18px; border-radius:12px;
            max-height:540px; overflow-y:auto; }
.doc-page { background:#fff; max-width:640px; margin:0 auto;
            padding:32px 38px; box-shadow:0 4px 18px rgba(0,0,0,0.10);
            border-radius:4px; font-family:Georgia,serif; }
.doc-org   { font-size:0.72em; font-weight:700; color:#1B5E20;
             letter-spacing:0.1em; text-transform:uppercase; margin-bottom:6px; }
.doc-title { font-size:1.65em; font-weight:800; color:#1C1B18; margin-bottom:4px; }
.doc-sub   { font-size:0.84em; color:#6b7280; margin-bottom:16px; }
.doc-hr    { border:none; border-top:2px solid #1B5E20; margin:14px 0; }
.doc-meta  { display:flex; gap:18px; flex-wrap:wrap; font-size:0.76em;
             color:#374151; margin-bottom:14px; }
.doc-meta-item { display:flex; flex-direction:column; }
.doc-meta-lbl  { font-size:0.72em; text-transform:uppercase;
                 letter-spacing:0.06em; color:#9CA3AF; }
.doc-meta-val  { font-weight:600; color:#1C1B18; }
.doc-h2  { font-size:0.85em; font-weight:700; color:#1B5E20;
           text-transform:uppercase; letter-spacing:0.08em;
           margin:16px 0 6px; border-bottom:1px solid #E5E7EB; padding-bottom:4px; }
.doc-kpi-row { display:flex; gap:8px; flex-wrap:wrap; margin:6px 0 10px; }
.doc-kpi     { background:#F0FDF4; border:1px solid #BBF7D0; border-radius:8px;
               padding:7px 13px; text-align:center; min-width:72px; }
.doc-kpi-val { font-size:1.25em; font-weight:800; color:#166534; }
.doc-kpi-lbl { font-size:0.62em; color:#6b7280; text-transform:uppercase;
               letter-spacing:0.07em; }
.doc-table th { background:#F0FDF4; color:#166534; font-weight:700;
                padding:5px 9px; text-align:left; border-bottom:2px solid #BBF7D0;
                font-size:0.68em; text-transform:uppercase; letter-spacing:0.05em; }
.doc-table td { padding:5px 9px; border-bottom:1px solid #F3F4F6; }
.doc-table    { width:100%; border-collapse:collapse; font-size:0.76em; }
.doc-conclusions { background:#F9FAFB; border-left:3px solid #2E7D32;
                   padding:10px 14px; border-radius:0 8px 8px 0;
                   font-size:0.82em; color:#374151; margin-top:6px; line-height:1.6; }
.doc-notes { background:#FFFBEB; border-left:3px solid #F59E0B;
             padding:9px 13px; border-radius:0 8px 8px 0;
             font-size:0.80em; color:#374151; margin-top:6px; font-style:italic; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

if "rpt_init" not in st.session_state:
    st.session_state["rpt_d0"]   = DATE_MAX - pd.Timedelta(days=30)
    st.session_state["rpt_d1"]   = DATE_MAX
    st.session_state["rpt_init"] = True

for _k, _v in {
    "rpt_type":       REPORT_TYPES[2],
    "rpt_sites":      ALL_SITES[:4],
    "rpt_researcher": "Alaa Barazi",
    "rpt_org":        "Ecological Models Lab — Spring 2026",
    "rpt_notes":      "",
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Report type selector ──────────────────────────────────────────────────────

rpt_type: str = st.radio(
    "Report type", REPORT_TYPES,
    index=REPORT_TYPES.index(st.session_state["rpt_type"]),
    horizontal=True, label_visibility="collapsed",
)
# Reset date range when report type changes
if rpt_type != st.session_state.get("rpt_type_prev"):
    if "Weekly" in rpt_type:
        st.session_state["rpt_d0"] = DATE_MAX - pd.Timedelta(days=7).to_pytimedelta()
        st.session_state["rpt_d1"] = DATE_MAX
    elif "Monthly" in rpt_type:
        st.session_state["rpt_d0"] = DATE_MAX - pd.Timedelta(days=30).to_pytimedelta()
        st.session_state["rpt_d1"] = DATE_MAX
    else:
        st.session_state["rpt_d0"] = DATE_MIN
        st.session_state["rpt_d1"] = DATE_MAX
    st.session_state["rpt_type_prev"] = rpt_type

st.session_state["rpt_type"] = rpt_type
st.write("")

# ── Two-column layout ─────────────────────────────────────────────────────────

col_prev, col_cust = st.columns([3, 1.5], gap="large")

# ── Customization panel ───────────────────────────────────────────────────────

with col_cust:
    section_label("Customize")

    sel_sites: list[str] = st.multiselect(
        "Sites", ALL_SITES, default=st.session_state["rpt_sites"]
    ) or ALL_SITES[:4]
    st.session_state["rpt_sites"] = sel_sites

    _auto_d0 = (
        DATE_MAX - pd.Timedelta(days=7).to_pytimedelta()  if "Weekly"  in rpt_type else
        DATE_MAX - pd.Timedelta(days=30).to_pytimedelta() if "Monthly" in rpt_type else
        DATE_MIN
    )
    d_range = st.date_input(
        "Date range",
        value=(st.session_state["rpt_d0"], st.session_state["rpt_d1"]),
        min_value=DATE_MIN, max_value=DATE_MAX,
    )
    if isinstance(d_range, tuple) and len(d_range) == 2:
        d0, d1 = d_range[0], d_range[1]
    else:
        d0, d1 = _auto_d0, DATE_MAX
    st.session_state["rpt_d0"] = d0
    st.session_state["rpt_d1"] = d1

    researcher: str = st.text_input("Researcher", value=st.session_state["rpt_researcher"]) or ""
    st.session_state["rpt_researcher"] = researcher

    org: str = st.text_input("Organization", value=st.session_state["rpt_org"]) or ""
    st.session_state["rpt_org"] = org

    notes: str = st.text_area(
        "Researcher notes", value=st.session_state["rpt_notes"],
        height=110, placeholder="Add observations or recommendations…",
    ) or ""
    st.session_state["rpt_notes"] = notes

# ── Build report data ─────────────────────────────────────────────────────────

filt        = _filter(sel_sites, d0, d1)
conc        = conclusions(filt, sel_sites)
now_str     = pd.Timestamp.now().strftime("%d %b %Y, %H:%M")
period_str  = f"{d0.strftime('%d %b %Y')} – {d1.strftime('%d %b %Y')}"
rpt_label   = rpt_type.split(" ", 1)[1]
sites_str   = ", ".join(sel_sites) if len(sel_sites) <= 3 else f"{len(sel_sites)} sites"

n_tot  = len(filt)
n_crit = int((filt["severity"] == "Critical").sum()) if not filt.empty else 0
n_sev  = int((filt["severity"] == "Severe").sum())  if not filt.empty else 0
avg_z  = round(float(filt["z_score"].mean()), 2)    if not filt.empty else 0.0
pk_lst = round(float(filt["lst"].max()), 1)          if not filt.empty else 0.0

# ── Preview ───────────────────────────────────────────────────────────────────

with col_prev:
    section_label("Report Preview")

    st.markdown(f"""
    <div class="doc-wrap"><div class="doc-page">
      <div class="doc-org">{org}</div>
      <div class="doc-title">{rpt_label}</div>
      <div class="doc-sub">Data Nature — Vegetation Health &amp; Heat Anomaly Monitoring</div>
      <hr class="doc-hr">
      <div class="doc-meta">
        <div class="doc-meta-item">
          <span class="doc-meta-lbl">Researcher</span>
          <span class="doc-meta-val">{researcher}</span>
        </div>
        <div class="doc-meta-item">
          <span class="doc-meta-lbl">Period</span>
          <span class="doc-meta-val">{period_str}</span>
        </div>
        <div class="doc-meta-item">
          <span class="doc-meta-lbl">Sites</span>
          <span class="doc-meta-val">{sites_str}</span>
        </div>
        <div class="doc-meta-item">
          <span class="doc-meta-lbl">Generated</span>
          <span class="doc-meta-val">{now_str}</span>
        </div>
      </div>
      <div class="doc-h2">Key Metrics</div>
      <div class="doc-kpi-row">
        <div class="doc-kpi"><div class="doc-kpi-val">{n_tot}</div><div class="doc-kpi-lbl">Anomalies</div></div>
        <div class="doc-kpi"><div class="doc-kpi-val" style="color:#DC2626">{n_crit}</div><div class="doc-kpi-lbl">Critical</div></div>
        <div class="doc-kpi"><div class="doc-kpi-val" style="color:#F97316">{n_sev}</div><div class="doc-kpi-lbl">Severe</div></div>
        <div class="doc-kpi"><div class="doc-kpi-val">{avg_z}σ</div><div class="doc-kpi-lbl">Avg Z</div></div>
        <div class="doc-kpi"><div class="doc-kpi-val">{pk_lst}°C</div><div class="doc-kpi-lbl">Peak LST</div></div>
      </div>
    </div></div>
    """, unsafe_allow_html=True)

    if not filt.empty:
        by_day = filt.groupby("date").size().reset_index(name="count")
        fig_p  = go.Figure(go.Bar(
            x=by_day["date"], y=by_day["count"],
            marker_color="#2E7D32",
            hovertemplate="%{x|%b %Y}<br>%{y} anomalies<extra></extra>",
        ))
        fig_p.update_layout(
            height=160, margin=dict(t=8, b=16, l=0, r=8),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFAFA", bargap=0.2,
            xaxis=dict(showgrid=False, tickformat="%b %Y", tickfont=dict(size=10)),
            yaxis=dict(showgrid=True, gridcolor="#F3F4F6",
                       title=dict(text="Events/month", font=dict(size=10))),
        )
        st.plotly_chart(fig_p, use_container_width=True)

    top_rows = filt.head(8)
    if not top_rows.empty:
        rows_html = "".join(
            f"<tr><td>{str(r['date'])[:10]}</td><td>{r['site']}</td>"
            f"<td>{r['lst']:.1f}°C</td><td>{r['z_score']:.2f}σ</td>"
            f'<td style="color:{_sev_color(r["severity"])};font-weight:700">'
            f"{r['severity']}</td></tr>"
            for _, r in top_rows.iterrows()
        )
        notes_block = f'<div class="doc-notes">📝 {notes}</div>' if notes.strip() else ""
        st.markdown(f"""
        <div class="doc-wrap" style="max-height:260px"><div class="doc-page"
             style="padding:14px 22px">
          <div class="doc-h2">Top Anomaly Events</div>
          <table class="doc-table">
            <thead><tr>
              <th>Date</th><th>Site</th><th>LST</th><th>Z-Score</th><th>Severity</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
          <div class="doc-h2" style="margin-top:12px">Conclusions</div>
          <div class="doc-conclusions">{conc}</div>
          {notes_block}
        </div></div>
        """, unsafe_allow_html=True)
    else:
        st.info("No anomaly events for the selected filters — adjust the date range or sites.")

# ── Export buttons ────────────────────────────────────────────────────────────

st.write("")
section_label("Export")

col_pdf, col_email, col_csv = st.columns(3)

with col_pdf:
    pdf_bytes = build_pdf(rpt_type, sel_sites, d0, d1, researcher, org, notes, filt)
    fname = f"DataNature_{rpt_label.replace(' ', '_')}_{d1.strftime('%Y%m%d')}.pdf"
    st.download_button(
        label="⬇️ Download PDF", data=pdf_bytes, file_name=fname,
        mime="application/pdf", use_container_width=True, type="primary",
    )

with col_email:
    if st.button("✉️ Send by Email", use_container_width=True):
        subject = f"Data Nature — {rpt_label} — {period_str}"
        body = (
            f"Please find the attached {rpt_label} for {sites_str}.\n\n"
            f"Period: {period_str}\n"
            f"Anomalies detected: {n_tot}  (Critical: {n_crit})\n\n"
            f"{conc}\n\nGenerated by Data Nature · {org}"
        )
        st.info(
            f"**Email ready to send:**\n\n**Subject:** {subject}\n\n"
            f"**Body preview:**\n{body[:300]}…\n\n"
            "_Attach the downloaded PDF and send from your mail client._"
        )

with col_csv:
    csv_cols = ["date", "site", "lst", "baseline", "z_score", "severity", "status", "ndvi_change"]
    csv_data = filt[csv_cols].to_csv(index=False) if not filt.empty else "No data"
    st.download_button(
        label="⬇️ Export CSV / GIS", data=csv_data,
        file_name=f"DataNature_{rpt_label.replace(' ', '_')}_{d1.strftime('%Y%m%d')}.csv",
        mime="text/csv", use_container_width=True,
    )

st.caption(
    f"Report covers **{n_tot}** anomaly events across **{len(sel_sites)}** site(s) · {period_str}. "
    "PDF includes chart + table + conclusions. CSV is GIS-ready."
)