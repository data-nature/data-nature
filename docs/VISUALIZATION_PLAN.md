# Data Nature — Visualization Plan

| | |
|---|---|
| **Team** | Ahmad Tawil · Alaa Barazi |
| **Project** | Northern Israel NDVI & LST Ecological Monitoring |
| **Sprint** | 31 May – 21 June 2026 |

---

## Visualization Table

| **Team** | **Key Pattern** | **Visualization — Why?** | **Tool** |
|---|---|---|---|
| Ahmad Tawil · Alaa Barazi | **Seasonal LST cycles across ecosystem types** — strong annual temperature cycle (peak Jul–Aug, trough Jan–Feb) with different amplitude per land-cover type (forest, arid, wetland, urban) → *Spatial patterns* | **Grid heatmap** (8 sites × 12 months, cell = mean LST °C). Collapses 96 (site × month) combinations into one panel — immediately shows which ecosystem is hottest and in which season, and which are insulated year-round | `Seaborn heatmap` / `Plotly imshow` |
| Ahmad Tawil · Alaa Barazi | **NDVI–LST vegetation cooling feedback** — negative correlation (r ≈ −0.4) between vegetation density and surface temperature; greener sites run cooler by ≈ 1–2 °C per 0.1 NDVI → *Species/ecosystem interactions* | **Scatter plot + OLS regression line**, points colored by land-cover type. Makes the cooling effect directly measurable; color-coding reveals whether each ecosystem follows or deviates from the global trend | `Plotly Express` `scatter(trendline="ols", color="land_cover")` |
| Ahmad Tawil · Alaa Barazi | **Heat diffusion across landscape cells** — LST propagates across neighboring grid cells over time; planting vegetation in one cell reduces heat in adjacent cells (modeled by cellular automaton) → *Complex spatiotemporal patterns* | **Animated grid** (10×10 cells colored by LST, animated over simulation steps). Animation is the only way to show how a spatial intervention (planting) propagates through the landscape over time — a static snapshot misses the diffusion dynamics | `Folium` grid map + `Plotly` step animation inside **Streamlit** |

---

## Pattern → Method Mapping

| Pattern Type | Method Chosen |
|---|---|
| Spatial patterns (site × season) | Grid heatmap |
| Species/ecosystem interactions (NDVI ↔ LST) | Scatter plot + OLS regression |
| Complex spatiotemporal patterns (heat diffusion) | Animated grid visualization |
