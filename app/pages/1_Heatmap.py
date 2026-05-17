from ui import empty_state, page_hero, set_page_config

set_page_config(title="Heatmap")

page_hero(
    title="🗺️ Heatmap",
    subtitle="Interactive Land Surface Temperature and NDVI heatmaps across 8 ecological monitoring sites in Northern Israel.",
    pills=["🛰️ Landsat Data", "📅 2000–2026", "📍 8 Sites"],
    emoji="🗺️",
)

empty_state("🚧", "Interactive map coming soon.<br>LST and NDVI visualisation across all sites.")
