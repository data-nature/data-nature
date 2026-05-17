"""Data Nature — Streamlit entry point."""

import streamlit as st
from ui import page_hero, section_label, set_page_config

from data_nature import __version__

set_page_config()

page_hero(
    title="Data Nature",
    subtitle=(
        "An interactive platform combining satellite data (NDVI & LST), "
        "ecological models, and machine learning to monitor vegetation health "
        "and detect heat anomalies across 8 sites in Northern Israel."
    ),
    pills=["📍 8 Sites Monitored", "📅 2000–2026", "🛰️ NDVI + LST", "🤖 ML Forecasting"],
    emoji="🌿",
)

section_label("Modules")

col1, col2, col3 = st.columns(3)
col1.metric("Sites monitored", "8")
col2.metric("Years of data", "2000–2026")
col3.metric("App version", __version__)

section_label("Navigation")

st.markdown(
    """
    Use the sidebar to navigate between modules:
    - **🗺️ Heatmap** — interactive map of LST and NDVI across the decade
    - **🚨 Anomalies** — automatic detection with z-score against a per-site baseline
    - **🔬 Simulator** — cellular-automaton "what if" scenarios for vegetation change
    - **🧬 Optimization** — genetic algorithm for optimal planting locations
    - **📈 Forecast** — 7-day temperature forecast with ML
    - **📋 Reports** — export PDF summaries
    - **🔍 Research Assistant** — RAG search over 5 academic papers
    """
)

st.caption("Course: Ecological Models Lab — Spring 2026 · Team: Alaa Barazi, Ahmad Tawil")
