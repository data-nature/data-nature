from __future__ import annotations

import html as _html
from typing import Literal

import streamlit as st

# ── Design tokens (mirrors .streamlit/config.toml) ───────────────────────────
# primaryColor = #2E7D32  |  backgroundColor = #FBFAF7
# secondaryBackgroundColor = #EFEDE6  |  textColor = #1C1B18

_GLOBAL_CSS = """
<style>
/* ── typography ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── layout ── */
.block-container { padding-top: 1.5rem !important; max-width: 1200px !important; }

/* ── gradient headings (green palette) ── */
h1, h2, h3 {
  background: -webkit-linear-gradient(45deg, #2E7D32, #43A047);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 700;
}

/* ── sidebar (glassmorphism) ── */
[data-testid="stSidebar"] {
  background-color: rgba(27, 94, 32, 0.85) !important;
  backdrop-filter: blur(12px) !important;
  border-right: 1px solid rgba(255, 255, 255, 0.08);
}

/* sidebar text — override theme dark textColor so it reads on dark green */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] a,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
  color: #D9EED9 !important;
  font-size: 1.05rem;
}
[data-testid="stSidebarNavLink"] {
  font-size: 1.05rem !important;
  font-weight: 500 !important;
  color: #D9EED9 !important;
}
[data-testid="stSidebarNavLink"]:hover,
[data-testid="stSidebarNavLink"][aria-current="page"] {
  background: rgba(255, 255, 255, 0.12) !important;
  color: #ffffff !important;
}

/* ── buttons ── */
.stButton > button {
  border-radius: 8px;
  transition: all 0.3s ease-in-out;
  background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%);
  border: none;
  color: white;
  box-shadow: 0 4px 6px -1px rgba(46, 125, 50, 0.4);
  font-weight: 600;
}
.stButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 15px -3px rgba(46, 125, 50, 0.5);
  color: white;
  border: none;
}

/* ── metric cards ── */
[data-testid="metric-container"] {
  background: rgba(232, 245, 233, 0.7);
  border: 1px solid #A5D6A7;
  border-radius: 12px;
  padding: 1rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
  transition: transform 0.3s ease;
}
[data-testid="metric-container"]:hover { transform: translateY(-2px); }

/* ── page hero ── */
.page-hero {
  background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 55%, #388E3C 100%);
  border-radius: 16px;
  padding: 40px 48px;
  color: #fff;
  margin-bottom: 28px;
  position: relative;
  overflow: hidden;
}
.page-hero::after {
  content: attr(data-emoji);
  position: absolute; right: 48px; top: 50%;
  transform: translateY(-50%);
  font-size: 7em; opacity: 0.1; pointer-events: none; line-height: 1;
}
.page-hero h1 {
  background: none !important;
  -webkit-background-clip: unset !important;
  -webkit-text-fill-color: #fff !important;
  color: #fff !important;
  font-size: 2.1em; font-weight: 800;
  margin: 0 0 10px; letter-spacing: -0.5px; line-height: 1.2;
}
.page-hero p {
  font-size: 1em; opacity: 0.88; margin: 0;
  max-width: 680px; line-height: 1.65;
}
.ph-pills { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 18px; }
.ph-pill {
  background: rgba(255,255,255,0.13);
  border: 1px solid rgba(255,255,255,0.22);
  border-radius: 999px; padding: 5px 14px;
  font-size: 0.76em; font-weight: 500;
}

/* ── section label ── */
.sec-label {
  font-size: 0.68em; font-weight: 700;
  letter-spacing: 0.12em; text-transform: uppercase;
  color: #6b7280; margin: 26px 0 12px;
}

/* ── empty / idle state ── */
.empty-state {
  text-align: center; padding: 56px 0 40px; color: #9ca3af;
}
.empty-icon { font-size: 2.8em; margin-bottom: 10px; }
.empty-msg  { font-size: 0.92em; line-height: 1.65; }
</style>
"""


def set_page_config(
    title: str = "Data Nature",
    icon: str = "🌿",
    layout: Literal["centered", "wide"] = "wide",
) -> None:
    """Call once at the top of every page before any other Streamlit command."""
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout=layout,
        initial_sidebar_state="expanded",
    )
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


def page_hero(
    title: str,
    subtitle: str,
    pills: list[str] | None = None,
    emoji: str = "🌿",
) -> None:
    """Render the green-gradient hero banner used at the top of every page."""
    pills_html = ""
    if pills:
        items = "".join(f'<div class="ph-pill">{_html.escape(p)}</div>' for p in pills)
        pills_html = f'<div class="ph-pills">{items}</div>'
    st.markdown(
        f"""
        <div class="page-hero" data-emoji="{_html.escape(emoji, quote=True)}">
          <h1>{_html.escape(title)}</h1>
          <p>{_html.escape(subtitle)}</p>
          {pills_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    """Render a small uppercase section-divider label."""
    st.markdown(
        f'<div class="sec-label">{_html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def empty_state(icon: str, message: str) -> None:
    """Render a centred idle / empty-state placeholder."""
    st.markdown(
        f"""
        <div class="empty-state">
          <div class="empty-icon">{icon}</div>
          <div class="empty-msg">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
