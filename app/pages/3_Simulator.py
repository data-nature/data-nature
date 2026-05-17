from ui import empty_state, page_hero, set_page_config

set_page_config(title="Simulator")

page_hero(
    title="🔬 What-If Simulator",
    subtitle="Cellular-automaton scenarios for modelling vegetation change under different environmental conditions.",
    pills=["🔄 Cellular Automaton", "🌱 Vegetation Scenarios", "📍 8 Sites"],
    emoji="🔬",
)

empty_state("🚧", "Simulator coming soon.<br>What-if scenarios for vegetation change.")
