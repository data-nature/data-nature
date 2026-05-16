"""RAG – Research Papers  (Gemini 2.0 Flash)

TF-IDF retrieval over 5 indexed academic papers + Gemini-generated synthesis.
The GEMINI_API_KEY is loaded from .env – no per-user key entry needed.
"""

from __future__ import annotations

import html
import os
import re
import sys
from pathlib import Path

# ── path setup (must come before any local imports) ───────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
_APP  = _ROOT / "app"
for _p in (str(_ROOT / "src"), str(_APP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from ui import set_page_config  # noqa: E402  # app/ui.py

from data_nature.rag import PAPER_INFO, ChromaRetriever, load_all_chunks  # noqa: E402

load_dotenv(_ROOT / ".env")

PAPERS_DIR   = _ROOT / "papers"
_GEMINI_KEY: str = os.getenv("GEMINI_API_KEY", "")

# ── Page config (delegates to ui.py + .streamlit/config.toml) ────────────────

set_page_config(title="Research Assistant")

# ── CSS ───────────────────────────────────────────────────────────────────────
# Colors stay in sync with .streamlit/config.toml:
#   primaryColor             = #2E7D32  (Material Green 800)
#   backgroundColor          = #FBFAF7  (warm white)
#   secondaryBackgroundColor = #EFEDE6  (warm beige)
#   textColor                = #1C1B18

_CSS = """
<style>
/* ---------- reset / global ---------- */
.block-container { padding-top: 1.5rem !important; max-width: 1200px !important; }
*, *::before, *::after { box-sizing: border-box; }

/* ---------- hero banner ---------- */
.rag-hero {
  background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 55%, #388E3C 100%);
  border-radius: 16px;
  padding: 40px 48px;
  color: #fff;
  margin-bottom: 28px;
  position: relative;
  overflow: hidden;
}
.rag-hero::after {
  content: '🌿';
  position: absolute; right: 48px; top: 50%;
  transform: translateY(-50%);
  font-size: 7em; opacity: 0.1; pointer-events: none;
  line-height: 1;
}
/* override ui.py's gradient-text rule for headings inside the hero */
.rag-hero h1 {
  background: none !important;
  -webkit-background-clip: unset !important;
  -webkit-text-fill-color: #fff !important;
  color: #fff !important;
  font-size: 2.1em; font-weight: 800;
  margin: 0 0 10px; letter-spacing: -0.5px; line-height: 1.2;
}
.rag-hero p {
  font-size: 1em; opacity: 0.88; margin: 0;
  max-width: 660px; line-height: 1.65;
}
.rag-pills { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 18px; }
.rag-pill {
  background: rgba(255,255,255,0.13);
  border: 1px solid rgba(255,255,255,0.22);
  border-radius: 999px; padding: 5px 14px;
  font-size: 0.76em; font-weight: 500;
}

/* ---------- section label ---------- */
.sec-label {
  font-size: 0.68em; font-weight: 700;
  letter-spacing: 0.12em; text-transform: uppercase;
  color: #6b7280; margin: 26px 0 12px;
}

/* ---------- paper cards ---------- */
.papers-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 12px; margin-bottom: 6px;
}
.paper-card {
  background: #fff; border: 1px solid #e5e7eb;
  border-radius: 12px; padding: 18px 16px;
  min-height: 175px; position: relative; overflow: hidden;
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
.paper-card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, #2E7D32, #43A047);
  opacity: 0; transition: opacity 0.2s ease;
}
.paper-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 28px rgba(0,0,0,0.09);
  border-color: #A5D6A7;
}
.paper-card:hover::before { opacity: 1; }
.p-emoji { font-size: 1.8em; line-height: 1; margin-bottom: 10px; }
.p-title { font-weight: 700; font-size: 0.81em; color: #1C1B18; line-height: 1.4; margin-bottom: 7px; }
.p-topic { font-size: 0.71em; color: #6b7280; line-height: 1.4; }

/* ---------- search input & button ---------- */
div[data-testid="stTextInput"] input {
  border-radius: 10px !important;
  border: 2px solid #e5e7eb !important;
  padding: 11px 16px !important;
  font-size: 0.95em !important;
  background: #fff !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
}
div[data-testid="stTextInput"] input:focus {
  border-color: #2E7D32 !important;
  box-shadow: 0 0 0 3px rgba(46,125,50,0.12) !important;
  outline: none !important;
}
/* keep primary button consistent with ui.py's gradient */
div[data-testid="stButton"] > button[kind="primary"] {
  border-radius: 10px !important;
  font-weight: 600 !important; letter-spacing: 0.02em !important;
  padding: 0.58rem 1.6rem !important;
}

/* ---------- AI answer box ---------- */
.ai-box {
  background: linear-gradient(135deg, #E8F5E9 0%, #F1F8E9 100%);
  border: 1px solid #A5D6A7;
  border-radius: 14px; padding: 26px 30px;
  margin: 18px 0 10px;
}
.ai-header {
  display: flex; align-items: center; gap: 10px;
  margin-bottom: 16px;
}
.ai-header-label {
  font-size: 0.68em; font-weight: 700;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: #2E7D32;
}
.ai-content {
  color: #1C1B18; font-size: 0.93em; line-height: 1.8;
}
.ai-content p   { margin: 0 0 0.75em; }
.ai-content h2,
.ai-content h3,
.ai-content h4  {
  /* override ui.py gradient for answer headings */
  background: none !important;
  -webkit-background-clip: unset !important;
  -webkit-text-fill-color: #1B5E20 !important;
  color: #1B5E20 !important;
  font-weight: 700; margin: 0.9em 0 0.35em;
}
.ai-content ul,
.ai-content ol  { padding-left: 1.4em; margin: 0.4em 0 0.75em; }
.ai-content li  { margin-bottom: 0.3em; }
.ai-content strong { color: #2E7D32; }
.ai-content code {
  background: #C8E6C9; color: #1B5E20;
  padding: 1px 6px; border-radius: 4px;
  font-family: monospace; font-size: 0.88em;
}
.ai-footer {
  display: flex; align-items: center; gap: 10px;
  margin-top: 16px; padding-top: 12px;
  border-top: 1px solid #C8E6C9;
}
.ai-badge {
  background: #C8E6C9; color: #1B5E20;
  border-radius: 6px; font-size: 0.72em;
  font-weight: 700; padding: 3px 10px; font-family: monospace;
}
.ai-note { font-size: 0.72em; color: #6b7280; }

/* ---------- result cards ---------- */
.result-card {
  background: #fff; border: 1px solid #e5e7eb;
  border-left: 4px solid #2E7D32;
  border-radius: 0 12px 12px 0;
  padding: 18px 22px; margin-bottom: 12px;
  transition: box-shadow 0.15s ease;
}
.result-card:hover { box-shadow: 0 4px 18px rgba(0,0,0,0.07); }
.r-meta {
  display: flex; align-items: center;
  gap: 10px; margin-bottom: 8px; flex-wrap: wrap;
}
.r-rank {
  background: #2E7D32; color: #fff;
  border-radius: 6px; padding: 2px 9px;
  font-size: 0.68em; font-weight: 700;
  letter-spacing: 0.05em; flex-shrink: 0;
}
.r-title {
  font-weight: 700; font-size: 0.9em;
  color: #1C1B18; flex: 1; min-width: 0;
}
.score-track {
  background: #EFEDE6; border-radius: 999px;
  height: 6px; width: 80px; overflow: hidden; flex-shrink: 0;
}
.score-fill {
  height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, #2E7D32, #43A047);
  transition: width 0.4s ease;
}
.score-pct { font-size: 0.75em; color: #6b7280; flex-shrink: 0; }
.r-excerpt {
  font-size: 0.84em; color: #374151; line-height: 1.72;
  padding-top: 10px; border-top: 1px solid #EFEDE6; margin-top: 8px;
}
.r-source {
  display: inline-block; margin-top: 10px;
  background: #E8F5E9; border: 1px solid #A5D6A7;
  color: #2E7D32; border-radius: 6px;
  font-size: 0.7em; font-weight: 600;
  font-family: monospace; padding: 3px 10px;
}

/* ---------- empty / idle state ---------- */
.empty-state {
  text-align: center; padding: 56px 0 40px; color: #9ca3af;
}
.empty-icon { font-size: 2.8em; margin-bottom: 10px; }
.empty-msg  { font-size: 0.92em; line-height: 1.65; }
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────


CHROMA_DIR = _ROOT / "data" / "processed" / "chroma"


@st.cache_resource(show_spinner="📚 Building semantic index – first load may take ~30 s…")
def _build_index() -> ChromaRetriever:
    return ChromaRetriever(load_all_chunks(PAPERS_DIR), persist_dir=CHROMA_DIR)


def _generate_answer(query: str, context: list[dict]) -> str:
    from google import genai  # lazy import

    client = genai.Client(api_key=_GEMINI_KEY)
    ctx_text = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context
    )
    prompt = (
        "You are a research assistant for an ecological monitoring study about "
        "vegetation health (NDVI) and heat anomalies (LST) in Northern Israel.\n\n"
        "Answer the question using ONLY the provided paper excerpts. "
        "Cite source filenames inline after each claim. "
        "Use clear markdown formatting (headers, bullet points where helpful). "
        "Be structured and precise.\n\n"
        f"## Question\n{query}\n\n"
        f"## Paper Excerpts\n{ctx_text}"
    )
    return client.models.generate_content(  # type: ignore[union-attr]
        model="gemini-2.5-flash",
        contents=prompt,
    ).text


def _md_to_html(text: str) -> str:
    """Minimal Markdown → HTML for embedding in the styled AI answer div."""
    t = html.escape(text)
    t = re.sub(r"^### (.+)$", r"<h4>\1</h4>", t, flags=re.MULTILINE)
    t = re.sub(r"^## (.+)$",  r"<h3>\1</h3>", t, flags=re.MULTILINE)
    t = re.sub(r"^# (.+)$",   r"<h2>\1</h2>", t, flags=re.MULTILINE)
    t = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", t)
    t = re.sub(r"\*\*(.+?)\*\*",     r"<strong>\1</strong>",          t)
    t = re.sub(r"\*(.+?)\*",         r"<em>\1</em>",                  t)
    t = re.sub(r"`(.+?)`",           r"<code>\1</code>",              t)
    t = re.sub(r"^[\-\*] (.+)$",     r"<li>\1</li>",  t, flags=re.MULTILINE)
    t = re.sub(r"^\d+\. (.+)$",      r"<li>\1</li>",  t, flags=re.MULTILINE)
    t = re.sub(r"((?:<li>.*?</li>\n?)+)", r"<ul>\1</ul>", t, flags=re.DOTALL)
    parts = re.split(r"\n{2,}", t)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if re.match(r"^<(h[2-4]|ul|ol)", p):
            out.append(p.replace("\n", ""))
        else:
            out.append(f"<p>{p.replace(chr(10), '<br>')}</p>")
    return "\n".join(out)


# ── Hero ──────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="rag-hero">
      <h1>🔍 Research Assistant</h1>
      <p>Ask any question about vegetation health, land surface temperature, or
      ecological monitoring. The system retrieves the most relevant passages from
      five indexed academic papers and generates a grounded answer with Gemini AI.</p>
      <div class="rag-pills">
        <div class="rag-pill">📄 5 Papers Indexed</div>
        <div class="rag-pill">🔎 ChromaDB Semantic Search</div>
        <div class="rag-pill">✨ Gemini 2.5 Flash</div>
        <div class="rag-pill">🌍 Northern Israel Ecology</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Paper cards ───────────────────────────────────────────────────────────────

st.markdown('<div class="sec-label">Indexed Papers</div>', unsafe_allow_html=True)

cards_html = '<div class="papers-grid">'
for _fname, _info in PAPER_INFO.items():
    _t = html.escape(_info["short_title"])
    _k = html.escape(_info["topic"])
    cards_html += (
        f'<div class="paper-card">'
        f'<div class="p-emoji">{_info["emoji"]}</div>'
        f'<div class="p-title">{_t}</div>'
        f'<div class="p-topic">{_k}</div>'
        f"</div>"
    )
cards_html += "</div>"
st.markdown(cards_html, unsafe_allow_html=True)

# ── Search bar ────────────────────────────────────────────────────────────────

st.markdown('<div class="sec-label">Ask a Question</div>', unsafe_allow_html=True)

col_q, col_k, col_btn = st.columns([5, 1, 1])
with col_q:
    query = st.text_input(
        "q",
        placeholder="e.g. How does NDVI correlate with land surface temperature across land cover types?",
        label_visibility="collapsed",
    )
with col_k:
    top_k = st.slider("k", min_value=1, max_value=10, value=5, label_visibility="collapsed")
with col_btn:
    st.write("")
    go = st.button(
        "Search →",
        type="primary",
        use_container_width=True,
        disabled=not query.strip(),
    )

# ── Results ───────────────────────────────────────────────────────────────────

retriever = _build_index()

if go and query.strip():
    hits = retriever.retrieve(query, top_k=top_k)

    if not hits:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-icon">🔎</div>'
            "<div class='empty-msg'>No relevant passages found.<br>"
            "Try rephrasing your question.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    # ── Gemini synthesis ──
    if _GEMINI_KEY:
        with st.spinner("Generating answer with Gemini…"):
            try:
                raw_answer = _generate_answer(query, hits)
                body_html  = _md_to_html(raw_answer)
                st.markdown(
                    f"""
                    <div class="ai-box">
                      <div class="ai-header">
                        <span style="font-size:1.3em;color:#2E7D32">✦</span>
                        <span class="ai-header-label">AI Synthesis</span>
                      </div>
                      <div class="ai-content">{body_html}</div>
                      <div class="ai-footer">
                        <span class="ai-badge">gemini-2.5-flash</span>
                        <span class="ai-note">
                          Based on {len(hits)} retrieved excerpt{"s" if len(hits) != 1 else ""}
                        </span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                st.error(f"Gemini API error: {exc}")

    # ── Retrieved excerpts ──
    st.markdown(
        '<div class="sec-label" style="margin-top:20px">Retrieved Excerpts</div>',
        unsafe_allow_html=True,
    )
    for rank, chunk in enumerate(hits, start=1):
        _info   = PAPER_INFO.get(chunk["source"], {})
        emoji   = _info.get("emoji", "📄")
        title   = html.escape(_info.get("short_title", chunk["source"]))
        pct     = round(chunk["score"] * 100, 1)
        bar_w   = min(round(chunk["score"] * 500), 100)
        excerpt = html.escape(chunk["text"][:520]) + ("…" if len(chunk["text"]) > 520 else "")
        src     = html.escape(chunk["source"])
        src_short = (src[:65] + "…") if len(src) > 65 else src

        st.markdown(
            f"""
            <div class="result-card">
              <div class="r-meta">
                <span class="r-rank">#{rank}</span>
                <span class="r-title">{emoji}&nbsp;{title}</span>
                <div class="score-track">
                  <div class="score-fill" style="width:{bar_w}%"></div>
                </div>
                <span class="score-pct">{pct}%</span>
              </div>
              <div class="r-excerpt">{excerpt}</div>
              <div class="r-source">{src_short}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    st.markdown(
        '<div class="empty-state">'
        '<div class="empty-icon">💬</div>'
        "<div class='empty-msg'>Enter a research question above to get started.<br>"
        "The system will retrieve relevant passages and synthesise an answer.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
