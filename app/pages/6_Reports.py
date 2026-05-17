from ui import empty_state, page_hero, set_page_config

set_page_config(title="Reports")

page_hero(
    title="📋 Reports",
    subtitle="Export PDF summaries of anomaly detections, NDVI trends, and site-level analyses.",
    pills=["📄 PDF Export", "📊 Anomaly Summaries", "🌿 NDVI Trends"],
    emoji="📋",
)

empty_state("🚧", "Reports coming soon.<br>PDF export of anomalies and analyses.")
