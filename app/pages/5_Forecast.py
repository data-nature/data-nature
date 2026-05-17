from ui import empty_state, page_hero, set_page_config

set_page_config(title="Forecast")

page_hero(
    title="📈 Temperature Forecast",
    subtitle="7-day Land Surface Temperature forecast using machine learning trained on Landsat time series.",
    pills=["🤖 ML Forecasting", "🌡️ LST Prediction", "📅 7-Day Horizon"],
    emoji="📈",
)

empty_state("🚧", "Forecast coming soon.<br>7-day LST prediction with machine learning.")
