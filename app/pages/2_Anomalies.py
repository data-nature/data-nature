from ui import empty_state, page_hero, set_page_config

set_page_config(title="Anomalies")

page_hero(
    title="🚨 Anomalies",
    subtitle="Automatic heat anomaly detection using z-score statistics against per-site historical baselines.",
    pills=["📊 Z-Score Detection", "📍 8 Sites", "📅 2000–2026"],
    emoji="🚨",
)

empty_state("🚧", "Anomaly detection coming soon.<br>Z-score analysis against per-site baselines.")
