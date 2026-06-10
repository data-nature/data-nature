"""
reports/generator.py — DN-B6 PDF report generator
====================================================
Extracts the PDF-building logic from 6_Reports.py into a reusable module.
Called by the Reports page and usable standalone (e.g. scheduled exports).

Public API
----------
    build_pdf       — generate a PDF report as bytes
    make_chart_png  — render anomaly-count bar chart as PNG bytes (BytesIO)
    conclusions     — generate natural-language summary text
"""

from __future__ import annotations

import io
from datetime import date, datetime
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
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

# ---------------------------------------------------------------------------
# Colour constants
# ---------------------------------------------------------------------------
_GREEN    = rl_colors.HexColor("#1B5E20")
_LT_GREEN = rl_colors.HexColor("#F0FDF4")
_GRAY     = rl_colors.HexColor("#6b7280")

_SEV_COLORS = {
    "Critical": rl_colors.HexColor("#DC2626"),
    "Severe":   rl_colors.HexColor("#F97316"),
    "Mild":     rl_colors.HexColor("#F59E0B"),
}

_GREENS_BAR = [
    "#1B5E20", "#2E7D32", "#388E3C",
    "#43A047", "#66BB6A", "#A5D6A7", "#C8E6C9",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def conclusions(df: pd.DataFrame, sites: list[str]) -> str:
    """Generate a natural-language conclusions paragraph.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered anomalies DataFrame.
    sites : list[str]
        Selected site names.

    Returns
    -------
    str — 2–3 sentence summary suitable for the report body.
    """
    if df.empty:
        return "No thermal anomalies were detected for the selected sites and period."

    n     = len(df)
    n_c   = int((df["severity"] == "Critical").sum())
    worst = df.groupby("site")["z_score"].mean().idxmax()
    max_z = float(df["z_score"].max())

    parts = [
        f"A total of {n} thermal anomaly event{'s' if n > 1 else ''} "
        f"were detected across {len(sites)} site(s) during the reporting period."
    ]
    if n_c:
        parts.append(
            f"{n_c} event{'s' if n_c > 1 else ''} "
            f"{'were' if n_c > 1 else 'was'} classified as Critical (z ≥ 3.0σ)."
        )
    parts.append(
        f"The highest thermal deviation was recorded at {worst} "
        f"(peak z = {max_z:.2f}σ). "
        "Field verification is recommended at all Critical and Severe sites."
    )
    return " ".join(parts)


def make_chart_png(df: pd.DataFrame, sites: list[str]) -> io.BytesIO:
    """Render anomaly-count bar chart as a PNG BytesIO object.

    Parameters
    ----------
    df : pd.DataFrame
        Anomalies DataFrame (may be empty).
    sites : list[str]
        Sites to include.

    Returns
    -------
    io.BytesIO  — PNG bytes at position 0.
    """
    fig, ax = plt.subplots(figsize=(7, 3))

    by_site = (
        df[df["site"].isin(sites)]
        .groupby("site")
        .size()
        .sort_values(ascending=True)
        if not df.empty
        else pd.Series(dtype=int)
    )

    if not by_site.empty:
        ax.barh(
            list(by_site.index),
            by_site.to_numpy(),
            color=_GREENS_BAR[: len(by_site)],
            height=0.6,
        )
        ax.set_xlabel("Anomaly count", fontsize=9)
    else:
        ax.text(
            0.5, 0.5, "No anomalies in period",
            ha="center", va="center", color="#9CA3AF", fontsize=10,
        )

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


def build_pdf(
    rpt_type:   str,
    sites:      list[str],
    d0:         date,
    d1:         date,
    researcher: str,
    org:        str,
    notes:      str,
    df:         pd.DataFrame,
) -> bytes:
    """Build a PDF report and return its raw bytes.

    Parameters
    ----------
    rpt_type : str
        Report type label (e.g. "🚨 Anomaly Report").
    sites : list[str]
        Selected site names.
    d0, d1 : date
        Reporting period start / end.
    researcher : str
        Name of the report author.
    org : str
        Organisation / project name.
    notes : str
        Free-text researcher notes (may be empty).
    df : pd.DataFrame
        Filtered anomalies DataFrame for the period and sites.

    Returns
    -------
    bytes — raw PDF content ready for st.download_button or file I/O.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2.5 * cm, bottomMargin=2 * cm,
    )
    base = getSampleStyleSheet()

    def _ps(name: str, **kw: Any) -> ParagraphStyle:
        return ParagraphStyle(name, parent=base["Normal"], **kw)

    title_s = _ps("T",  fontSize=22, fontName="Helvetica-Bold",
                        textColor=_GREEN, spaceAfter=4)
    sub_s   = _ps("S",  fontSize=10, textColor=_GRAY, spaceAfter=14)
    h2_s    = _ps("H2", fontSize=11, fontName="Helvetica-Bold",
                        textColor=_GREEN, spaceBefore=14, spaceAfter=5)
    body_s  = _ps("B",  fontSize=9, leading=14,
                        textColor=rl_colors.HexColor("#374151"), spaceAfter=6)
    foot_s  = _ps("F",  fontSize=7, textColor=_GRAY, alignment=1)

    story = []

    # ── Cover ────────────────────────────────────────────────────────────────
    rpt_label = rpt_type.split(" ", 1)[1] if " " in rpt_type else rpt_type
    period    = f"{d0.strftime('%d %b %Y')} – {d1.strftime('%d %b %Y')}"

    story.append(Paragraph(
        org, _ps("Org", fontSize=9, fontName="Helvetica-Bold", textColor=_GREEN)
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(rpt_label, title_s))
    story.append(Paragraph("Data Nature — Vegetation Health & Heat Anomaly Monitoring", sub_s))

    meta = Table(
        [
            ["Researcher", researcher, "Period",    period],
            ["Sites", (", ".join(sites))[:80], "Generated",
             datetime.now().strftime("%d %b %Y %H:%M")],
        ],
        colWidths=[2.8 * cm, 6 * cm, 2.8 * cm, 5.4 * cm],
    )
    meta.setStyle(TableStyle([
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("FONTNAME",   (0, 0), (0, -1),  "Helvetica-Bold"),
        ("FONTNAME",   (2, 0), (2, -1),  "Helvetica-Bold"),
        ("TEXTCOLOR",  (0, 0), (0, -1),  _GRAY),
        ("TEXTCOLOR",  (2, 0), (2, -1),  _GRAY),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(meta)
    story.append(Spacer(1, 12))

    # ── KPI strip ────────────────────────────────────────────────────────────
    if not df.empty:
        n_tot  = len(df)
        n_crit = int((df["severity"] == "Critical").sum())
        n_sev  = int((df["severity"] == "Severe").sum())
        avg_z  = float(df["z_score"].mean())
        pk_lst = float(df["lst"].max())

        kpi_t = Table(
            [
                ["Anomalies", "Critical", "Severe", "Avg Z-Score", "Peak LST"],
                [str(n_tot), str(n_crit), str(n_sev), f"{avg_z:.2f}σ", f"{pk_lst:.1f}°C"],
            ],
            colWidths=[3.4 * cm] * 5,
        )
        kpi_t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), _LT_GREEN),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, 0), 7),
            ("TEXTCOLOR",  (0, 0), (-1, 0), _GRAY),
            ("FONTNAME",   (0, 1), (-1, 1), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 1), (-1, 1), 13),
            ("TEXTCOLOR",  (0, 1), (-1, 1), _GREEN),
            ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("BOX",       (0, 0), (-1, -1), 1, rl_colors.HexColor("#BBF7D0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#E5E7EB")),
        ]))
        story.append(kpi_t)
        story.append(Spacer(1, 12))

    # ── Chart ────────────────────────────────────────────────────────────────
    story.append(Paragraph("Anomaly Activity by Site", h2_s))
    story.append(RLImage(make_chart_png(df, sites), width=14 * cm, height=5.5 * cm))
    story.append(Spacer(1, 10))

    # ── Anomaly table (top 10) ────────────────────────────────────────────────
    if not df.empty:
        story.append(Paragraph("Anomaly Events (top 10 by z-score)", h2_s))
        rows = [["Date", "Site", "LST (°C)", "Z-Score", "Severity", "Status"]]
        for _, r in df.head(10).iterrows():
            rows.append([
                str(r["date"])[:10], r["site"],
                f"{r['lst']:.1f}", f"{r['z_score']:.2f}σ",
                r["severity"], r["status"],
            ])
        tbl = Table(
            rows, colWidths=[2.3 * cm, 4 * cm, 2.2 * cm, 2.2 * cm, 2 * cm, 2.3 * cm]
        )
        cmds: list = [
            ("BACKGROUND", (0, 0), (-1, 0), _GREEN),
            ("TEXTCOLOR",  (0, 0), (-1, 0), rl_colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("GRID",       (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#E5E7EB")),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        for i, row in enumerate(rows[1:], 1):
            if i % 2 == 0:
                cmds.append(("BACKGROUND", (0, i), (-1, i), _LT_GREEN))
            if row[4] in _SEV_COLORS:
                cmds += [
                    ("TEXTCOLOR", (4, i), (4, i), _SEV_COLORS[row[4]]),
                    ("FONTNAME",  (4, i), (4, i), "Helvetica-Bold"),
                ]
        tbl.setStyle(TableStyle(cmds))
        story.append(tbl)
        story.append(Spacer(1, 12))

    # ── Conclusions ───────────────────────────────────────────────────────────
    story.append(Paragraph("Conclusions & Recommendations", h2_s))
    story.append(Paragraph(conclusions(df, sites), body_s))

    if notes.strip():
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "Researcher Notes",
            _ps("NL", fontSize=9, fontName="Helvetica-BoldOblique",
                textColor=rl_colors.HexColor("#92400E")),
        ))
        story.append(Paragraph(
            notes,
            _ps("NB", fontSize=9, fontName="Helvetica-Oblique",
                textColor=rl_colors.HexColor("#374151"), leading=13),
        ))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        f"Data Nature · {org} · {datetime.now().strftime('%d %b %Y')}",
        foot_s,
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()