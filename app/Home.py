"""Data Nature — Streamlit entry point."""

import streamlit as st
from ui import set_page_config

from data_nature import __version__

set_page_config()


st.title("🌿 Data Nature")
st.subheader("Vegetation health monitoring and heat anomaly detection — Northern Israel")

st.markdown(
    """
    Welcome to **Data Nature**, an interactive platform that combines satellite data
    (NDVI and LST), ecological models, genetic algorithms, and machine learning to
    detect heat anomalies across 8 ecological sites in Northern Israel.

    Use the sidebar to navigate between modules:
    - **Heatmap** — interactive map of LST and NDVI across the decade
    - **Anomalies** — automatic detection with z-score against a per-site baseline
    - **Simulator** — cellular-automaton "what if" scenarios for vegetation change
    - **Optimization** — genetic algorithm for optimal planting locations
    - **Forecast** — 7-day temperature forecast with ML
    - **Reports** — export PDF summaries
    """
)

st.divider()

col1, col2, col3 = st.columns(3)
col1.metric("Sites monitored", "8")
col2.metric("Years of data", "2000–2026")
col3.metric("App version", __version__)

st.caption("Course: Ecological Models Lab — Spring 2026 · Team: Alaa Barazi, Ahmad Tawil")
