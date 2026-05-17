from __future__ import annotations

import io
import sys
from datetime import date, datetime
from typing import Any
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_APP = _ROOT / "app"
for _p in (str(_ROOT / "src"), str(_APP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from reportlab.lib import colors as rl_colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from ui import page_hero, section_label, set_page_config  # noqa: E402

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

# ── Data ──────────────────────────────────────────────────────────────────────

MOCK = _ROOT / "data" / "mock"


@st.cache_data
def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    anom = pd.read_csv(MOCK / "anomalies.csv", parse_dates=["date"])
    ts = pd.read_csv(MOCK / "lst_timeseries.csv", parse_dates=["date"])
    monthly = pd.read_csv(MOCK / "site_monthly.csv")
    return anom, ts, monthly


anom_df, ts_df, monthly_df = _load()
ALL_SITES: list[str] = sorted(anom_df["site"].unique().tolist())
DATE_MIN: date = anom_df["date"].min().date()
DATE_MAX: date = anom_df["date"].max().date()

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


def _conclusions(df: pd.DataFrame, sites: list[str]) -> str:
    if df.empty:
        return "No thermal anomalies were detected for the selected sites and period."
    n = len(df)
    n_c = (df["severity"] == "Critical").sum()
    worst = df.groupby("site")["z_score"].mean().idxmax()
    max_z = df["z_score"].max()
    parts = [
        f"A total of {n} thermal anomaly event{'s' if n > 1 else ''} were detected "
        f"across {len(sites)} site(s) during the reporting period."
    ]
    if n_c:
        parts.append(
            f"{n_c} event{'s' if n_c > 1 else ''} "
            f"{'were' if n_c > 1 else 'was'} classified as Critical (z ≥ 3.0σ)."
        )
    parts.append(
        f"The highest thermal deviation was recorded at {worst} (peak z = {max_z:.2f}σ). "
        "Field verification is recommended at all Critical and Severe sites."
    )
    return " ".join(parts)


def _sev_color(sev: str) -> str:
    return {"Critical": "#DC2626", "Severe": "#F97316", "Mild": "#F59E0B"}.get(
        sev, "#6b7280"
    )


# ── PDF builder ───────────────────────────────────────────────────────────────


def _make_chart_png(df: pd.DataFrame, sites: list[str]) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(7, 3))
    by_site = (
        df[df["site"].isin(sites)].groupby("site").size().sort_values(ascending=True)
        if not df.empty
        else pd.Series(dtype=int)
    )
    greens = ["#1B5E20", "#2E7D32", "#388E3C", "#43A047", "#66BB6A", "#A5D6A7", "#C8E6C9"]
    if not by_site.empty:
        ax.barh(
            list(by_site.index),
            by_site.to_numpy(),
            color=greens[: len(by_site)],
            height=0.6,
        )
        ax.set_xlabel("Anomaly count", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No anomalies in period", ha="center", va="center",
                color="#9CA3AF", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, axis="x")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _build_pdf(
    rpt_type: str,
    sites: list[str],
    d0: date,
    d1: date,
    researcher: str,
    org: str,
    notes: str,
    df: pd.DataFrame,
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2.5 * cm, bottomMargin=2 * cm,
    )
    base = getSampleStyleSheet()
    GREEN = rl_colors.HexColor("#1B5E20")
    LT_GREEN = rl_colors.HexColor("#F0FDF4")
    GRAY = rl_colors.HexColor("#6b7280")

    def _ps(name: str, **kw: Any) -> ParagraphStyle:
        return ParagraphStyle(name, parent=base["Normal"], **kw)

    title_s = _ps("T", fontSize=22, fontName="Helvetica-Bold", textColor=GREEN, spaceAfter=4)
    sub_s = _ps("S", fontSize=10, textColor=GRAY, spaceAfter=14)
    h2_s = _ps("H2", fontSize=11, fontName="Helvetica-Bold", textColor=GREEN,
                spaceBefore=14, spaceAfter=5)
    body_s = _ps("B", fontSize=9, leading=14, textColor=rl_colors.HexColor("#374151"), spaceAfter=6)
    foot_s = _ps("F", fontSize=7, textColor=GRAY, alignment=1)

    story = []

    # Cover
    rpt_label = rpt_type.split(" ", 1)[1]
    period = f"{d0.strftime('%d %b %Y')} – {d1.strftime('%d %b %Y')}"
    story.append(Paragraph(org, _ps("Org", fontSize=9, fontName="Helvetica-Bold", textColor=GREEN)))
    story.append(Spacer(1, 6))
    story.append(Paragraph(rpt_label, title_s))
    story.append(Paragraph("Data Nature — Vegetation Health & Heat Anomaly Monitoring", sub_s))

    meta = Table(
        [
            ["Researcher", researcher, "Period", period],
            ["Sites", (", ".join(sites))[:80], "Generated", datetime.now().strftime("%d %b %Y %H:%M")],
        ],
        colWidths=[2.8 * cm, 6 * cm, 2.8 * cm, 5.4 * cm],
    )
    meta.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), GRAY),
        ("TEXTCOLOR", (2, 0), (2, -1), GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(meta)
    story.append(Spacer(1, 12))

    # KPI strip
    if not df.empty:
        n_tot = len(df)
        n_crit = (df["severity"] == "Critical").sum()
        n_sev = (df["severity"] == "Severe").sum()
        avg_z = df["z_score"].mean()
        pk_lst = df["lst"].max()
        kpi_t = Table(
            [
                ["Anomalies", "Critical", "Severe", "Avg Z-Score", "Peak LST"],
                [str(n_tot), str(n_crit), str(n_sev), f"{avg_z:.2f}σ", f"{pk_lst:.1f}°C"],
            ],
            colWidths=[3.4 * cm] * 5,
        )
        kpi_t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), LT_GREEN),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 7),
            ("TEXTCOLOR", (0, 0), (-1, 0), GRAY),
            ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 1), (-1, 1), 13),
            ("TEXTCOLOR", (0, 1), (-1, 1), GREEN),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("BOX", (0, 0), (-1, -1), 1, rl_colors.HexColor("#BBF7D0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#E5E7EB")),
        ]))
        story.append(kpi_t)
        story.append(Spacer(1, 12))

    # Chart
    story.append(Paragraph("Anomaly Activity by Site", h2_s))
    story.append(RLImage(_make_chart_png(df, sites), width=14 * cm, height=5.5 * cm))
    story.append(Spacer(1, 10))

    # Anomaly table
    if not df.empty:
        story.append(Paragraph("Anomaly Events (top 10 by z-score)", h2_s))
        rows = [["Date", "Site", "LST (°C)", "Z-Score", "Severity", "Status"]]
        for _, r in df.head(10).iterrows():
            rows.append([
                str(r["date"])[:10], r["site"],
                f"{r['lst']:.1f}", f"{r['z_score']:.2f}σ",
                r["severity"], r["status"],
            ])
        tbl = Table(rows, colWidths=[2.3 * cm, 4 * cm, 2.2 * cm, 2.2 * cm, 2 * cm, 2.3 * cm])
        sev_clr = {
            "Critical": rl_colors.HexColor("#DC2626"),
            "Severe": rl_colors.HexColor("#F97316"),
            "Mild": rl_colors.HexColor("#F59E0B"),
        }
        cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), GREEN),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#E5E7EB")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        for i, row in enumerate(rows[1:], 1):
            if i % 2 == 0:
                cmds.append(("BACKGROUND", (0, i), (-1, i), LT_GREEN))
            if row[4] in sev_clr:
                cmds += [
                    ("TEXTCOLOR", (4, i), (4, i), sev_clr[row[4]]),
                    ("FONTNAME", (4, i), (4, i), "Helvetica-Bold"),
                ]
        tbl.setStyle(TableStyle(cmds))
        story.append(tbl)
        story.append(Spacer(1, 12))

    # Conclusions
    story.append(Paragraph("Conclusions & Recommendations", h2_s))
    story.append(Paragraph(_conclusions(df, sites), body_s))
    if notes.strip():
        story.append(Spacer(1, 6))
        story.append(Paragraph("Researcher Notes", _ps("NL", fontSize=9,
                                fontName="Helvetica-BoldOblique",
                                textColor=rl_colors.HexColor("#92400E"))))
        story.append(Paragraph(notes, _ps("NB", fontSize=9, fontName="Helvetica-Oblique",
                                textColor=rl_colors.HexColor("#374151"), leading=13)))

    # Footer
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        f"Data Nature · {org} · {datetime.now().strftime('%d %b %Y')}",
        foot_s,
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .doc-wrap {
        background:#f0f2f6; padding:18px; border-radius:12px;
        max-height:540px; overflow-y:auto;
    }
    .doc-page {
        background:#fff; max-width:640px; margin:0 auto;
        padding:32px 38px; box-shadow:0 4px 18px rgba(0,0,0,0.10);
        border-radius:4px; font-family:Georgia,serif;
    }
    .doc-org  { font-size:0.72em; font-weight:700; color:#1B5E20;
                letter-spacing:0.1em; text-transform:uppercase; margin-bottom:6px; }
    .doc-title { font-size:1.65em; font-weight:800; color:#1C1B18; margin-bottom:4px; }
    .doc-sub  { font-size:0.84em; color:#6b7280; margin-bottom:16px; }
    .doc-hr   { border:none; border-top:2px solid #1B5E20; margin:14px 0; }
    .doc-meta { display:flex; gap:18px; flex-wrap:wrap; font-size:0.76em;
                color:#374151; margin-bottom:14px; }
    .doc-meta-item { display:flex; flex-direction:column; }
    .doc-meta-lbl  { font-size:0.72em; text-transform:uppercase;
                     letter-spacing:0.06em; color:#9CA3AF; }
    .doc-meta-val  { font-weight:600; color:#1C1B18; }
    .doc-h2 { font-size:0.85em; font-weight:700; color:#1B5E20;
              text-transform:uppercase; letter-spacing:0.08em;
              margin:16px 0 6px; border-bottom:1px solid #E5E7EB; padding-bottom:4px; }
    .doc-kpi-row { display:flex; gap:8px; flex-wrap:wrap; margin:6px 0 10px; }
    .doc-kpi { background:#F0FDF4; border:1px solid #BBF7D0; border-radius:8px;
               padding:7px 13px; text-align:center; min-width:72px; }
    .doc-kpi-val { font-size:1.25em; font-weight:800; color:#166534; }
    .doc-kpi-lbl { font-size:0.62em; color:#6b7280; text-transform:uppercase;
                   letter-spacing:0.07em; }
    .doc-table { width:100%; border-collapse:collapse; font-size:0.76em; }
    .doc-table th { background:#F0FDF4; color:#166534; font-weight:700;
                    padding:5px 9px; text-align:left; border-bottom:2px solid #BBF7D0;
                    font-size:0.68em; text-transform:uppercase; letter-spacing:0.05em; }
    .doc-table td { padding:5px 9px; border-bottom:1px solid #F3F4F6; }
    .doc-conclusions { background:#F9FAFB; border-left:3px solid #2E7D32;
                       padding:10px 14px; border-radius:0 8px 8px 0;
                       font-size:0.82em; color:#374151; margin-top:6px; line-height:1.6; }
    .doc-notes { background:#FFFBEB; border-left:3px solid #F59E0B;
                 padding:9px 13px; border-radius:0 8px 8px 0;
                 font-size:0.80em; color:#374151; margin-top:6px; font-style:italic; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────

if "rpt_init" not in st.session_state:
    st.session_state["rpt_d0"] = DATE_MAX - pd.Timedelta(days=30)
    st.session_state["rpt_d1"] = DATE_MAX
    st.session_state["rpt_init"] = True

for _k, _v in {
    "rpt_type": REPORT_TYPES[2],
    "rpt_sites": ALL_SITES[:4],
    "rpt_researcher": "Ahmad Tawil",
    "rpt_org": "Ecological Models Lab — Spring 2026",
    "rpt_notes": "",
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Report type selector ──────────────────────────────────────────────────────

rpt_type: str = st.radio(
    "Report type",
    REPORT_TYPES,
    index=REPORT_TYPES.index(st.session_state["rpt_type"]),
    horizontal=True,
    label_visibility="collapsed",
)
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
        DATE_MAX - pd.Timedelta(days=7).to_pytimedelta()
        if "Weekly" in rpt_type
        else DATE_MAX - pd.Timedelta(days=30).to_pytimedelta()
        if "Monthly" in rpt_type
        else DATE_MIN
    )
    d_range = st.date_input(
        "Date range",
        value=(st.session_state["rpt_d0"], st.session_state["rpt_d1"]),
        min_value=DATE_MIN,
        max_value=DATE_MAX,
    )
    if isinstance(d_range, tuple) and len(d_range) == 2:
        d0, d1 = d_range[0], d_range[1]
    else:
        d0, d1 = _auto_d0, DATE_MAX
    st.session_state["rpt_d0"] = d0
    st.session_state["rpt_d1"] = d1

    researcher: str = st.text_input(
        "Researcher", value=st.session_state["rpt_researcher"]
    ) or ""
    st.session_state["rpt_researcher"] = researcher

    org: str = st.text_input("Organization", value=st.session_state["rpt_org"]) or ""
    st.session_state["rpt_org"] = org

    notes: str = st.text_area(
        "Researcher notes",
        value=st.session_state["rpt_notes"],
        height=110,
        placeholder="Add observations or recommendations…",
    ) or ""
    st.session_state["rpt_notes"] = notes

# ── Build report data ─────────────────────────────────────────────────────────

filt = _filter(sel_sites, d0, d1)
conclusions = _conclusions(filt, sel_sites)
now_str = pd.Timestamp.now().strftime("%d %b %Y, %H:%M")
period_str = f"{d0.strftime('%d %b %Y')} – {d1.strftime('%d %b %Y')}"
rpt_label = rpt_type.split(" ", 1)[1]
sites_str = ", ".join(sel_sites) if len(sel_sites) <= 3 else f"{len(sel_sites)} sites"

n_tot = len(filt)
n_crit = (filt["severity"] == "Critical").sum() if not filt.empty else 0
n_sev = (filt["severity"] == "Severe").sum() if not filt.empty else 0
avg_z = round(filt["z_score"].mean(), 2) if not filt.empty else 0.0
pk_lst = round(filt["lst"].max(), 1) if not filt.empty else 0.0

# ── Preview ───────────────────────────────────────────────────────────────────

with col_prev:
    section_label("Report Preview")

    # Cover page
    st.markdown(
        f"""
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
            <div class="doc-kpi">
              <div class="doc-kpi-val">{n_tot}</div>
              <div class="doc-kpi-lbl">Anomalies</div>
            </div>
            <div class="doc-kpi">
              <div class="doc-kpi-val" style="color:#DC2626">{n_crit}</div>
              <div class="doc-kpi-lbl">Critical</div>
            </div>
            <div class="doc-kpi">
              <div class="doc-kpi-val" style="color:#F97316">{n_sev}</div>
              <div class="doc-kpi-lbl">Severe</div>
            </div>
            <div class="doc-kpi">
              <div class="doc-kpi-val">{avg_z}σ</div>
              <div class="doc-kpi-lbl">Avg Z</div>
            </div>
            <div class="doc-kpi">
              <div class="doc-kpi-val">{pk_lst}°C</div>
              <div class="doc-kpi-lbl">Peak LST</div>
            </div>
          </div>
        </div></div>
        """,
        unsafe_allow_html=True,
    )

    # Anomaly timeline chart
    if not filt.empty:
        by_day = filt.groupby("date").size().reset_index(name="count")
        fig_p = go.Figure(
            go.Bar(
                x=by_day["date"],
                y=by_day["count"],
                marker_color="#2E7D32",
                hovertemplate="%{x|%d %b}<br>%{y} anomalies<extra></extra>",
            )
        )
        fig_p.update_layout(
            height=160,
            margin=dict(t=8, b=16, l=0, r=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#FAFAFA",
            bargap=0.2,
            xaxis=dict(showgrid=False, tickformat="%d %b", tickfont=dict(size=10)),
            yaxis=dict(
                showgrid=True,
                gridcolor="#F3F4F6",
                title=dict(text="Events/day", font=dict(size=10)),
            ),
        )
        st.plotly_chart(fig_p, use_container_width=True)

    # Anomaly table + conclusions
    top_rows = filt.head(8)
    if not top_rows.empty:
        rows_html = "".join(
            f"<tr>"
            f"<td>{str(r['date'])[:10]}</td>"
            f"<td>{r['site']}</td>"
            f"<td>{r['lst']:.1f}°C</td>"
            f"<td>{r['z_score']:.2f}σ</td>"
            f'<td style="color:{_sev_color(r["severity"])};font-weight:700">'
            f"{r['severity']}</td>"
            f"</tr>"
            for _, r in top_rows.iterrows()
        )
        notes_block = (
            f'<div class="doc-notes">📝 {notes}</div>' if notes.strip() else ""
        )
        st.markdown(
            f"""
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
              <div class="doc-conclusions">{conclusions}</div>
              {notes_block}
            </div></div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No anomaly events for the selected filters — adjust the date range or sites.")

# ── Export buttons ────────────────────────────────────────────────────────────

st.write("")
section_label("Export")

col_pdf, col_email, col_csv = st.columns(3)

with col_pdf:
    pdf_bytes = _build_pdf(rpt_type, sel_sites, d0, d1, researcher, org, notes, filt)
    fname = f"DataNature_{rpt_label.replace(' ', '_')}_{d1.strftime('%Y%m%d')}.pdf"
    st.download_button(
        label="⬇️ Download PDF",
        data=pdf_bytes,
        file_name=fname,
        mime="application/pdf",
        use_container_width=True,
        type="primary",
    )

with col_email:
    if st.button("✉️ Send by Email", use_container_width=True):
        subject = f"Data Nature — {rpt_label} — {period_str}"
        body = (
            f"Please find the attached {rpt_label} for {sites_str}.\n\n"
            f"Period: {period_str}\n"
            f"Anomalies detected: {n_tot}  (Critical: {n_crit})\n\n"
            f"{conclusions}\n\n"
            f"Generated by Data Nature · {org}"
        )
        st.info(
            f"**Email ready to send:**\n\n"
            f"**Subject:** {subject}\n\n"
            f"**Body preview:**\n{body[:300]}…\n\n"
            "_Attach the downloaded PDF and send from your mail client. "
            "Automated delivery requires SMTP server configuration._"
        )

with col_csv:
    csv_cols = ["date", "site", "lst", "baseline", "z_score", "severity", "status", "ndvi_change"]
    csv_data = filt[csv_cols].to_csv(index=False) if not filt.empty else "No data"
    st.download_button(
        label="⬇️ Export CSV / GIS",
        data=csv_data,
        file_name=f"DataNature_{rpt_label.replace(' ', '_')}_{d1.strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.caption(
    f"Report covers **{n_tot}** anomaly events across **{len(sel_sites)}** site(s) · {period_str}. "
    "PDF includes chart + table + conclusions. CSV is GIS-ready (lat/lng available in site_locations.csv)."
)
