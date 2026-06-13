"""
data_nature.models
~~~~~~~~~~~~~~~~~~
Forecast and simulation models for the Data Nature project.

Public API (DN-B2):
    train_forecasters  — train Random Forest + Gradient Boosting on site_monthly data
    forecast           — recursive multi-step LST forecast for a single site
    compute_metrics    — MAE / RMSE / R² on the held-out test split
    run_full_pipeline  — end-to-end: load → train → forecast all sites → save CSVs
"""

from data_nature.models.forecast import (
    compute_metrics,
    forecast,
    run_full_pipeline,
    train_forecasters,
)
from data_nature.models.ecology import (
    EnergyFlow,
    LogisticGrowth,
    LotkaVolterra,
    simulate_energy_flow,
    simulate_logistic,
    simulate_lv,
)
from data_nature.models.optimization import (
    COOLING,
    GRID,
    SEASON_MONTHS,
    VEG_BOOST,
    expected_delta_lst,
    fitness,
    init_site_grid,
    optimize_planting,
)

__all__ = [
    "train_forecasters",
    "forecast",
    "compute_metrics",
    "run_full_pipeline",
    "optimize_planting",
    "init_site_grid",
    "expected_delta_lst",
    "fitness",
    "GRID",
    "COOLING",
    "VEG_BOOST",
    "SEASON_MONTHS",
    "LogisticGrowth",
    "simulate_logistic",
    "EnergyFlow",
    "simulate_energy_flow",
    "LotkaVolterra",
    "simulate_lv",
]