"""
Mock data generator for the Temperature Forecast page.

Run:  python data/mock/generate_forecast.py

Produces three CSV files in data/mock/:
  - lst_history.csv    30-day historical LST per site
  - lst_forecast.csv   7-day forecast with bounds per model per site
  - model_metrics.csv  MAE / RMSE / R² per model per site

Replace these files with real GEE / ML outputs when ready.
Column names and dtypes must stay the same for the page to work.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
RNG = np.random.default_rng(42)

SITES: dict[str, float] = {
    "Hula Valley": 29.0,
    "Golan Heights": 24.5,
    "Galilee North": 27.0,
    "Galilee South": 28.5,
    "Mount Carmel": 26.0,
    "Jezreel Valley": 30.5,
    "Jordan Valley": 35.0,
    "Mount Tabor": 27.5,
}

MODELS = ["Random Forest", "Gradient Boosting", "LSTM"]

TODAY = date(2026, 5, 17)
HISTORY_DAYS = 30
FORECAST_DAYS = 7


# ── helpers ───────────────────────────────────────────────────────────────────


def _history_lst(base: float, n: int) -> np.ndarray:
    trend = np.linspace(0, 2.5, n)
    seasonal = 1.5 * np.sin(np.linspace(0, np.pi, n))
    noise = RNG.normal(0, 0.6, n)
    return np.round(base + trend + seasonal + noise, 2)


def _forecast_lst(last_hist: float, model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = {"Random Forest": 0.3, "Gradient Boosting": 0.5, "LSTM": 0.1}
    widths = {"Random Forest": 1.6, "Gradient Boosting": 1.4, "LSTM": 1.0}
    trend = np.linspace(0, 1.8, FORECAST_DAYS)
    noise = RNG.normal(offsets[model], 0.3, FORECAST_DAYS)
    band = widths[model]
    mean = np.round(last_hist + trend + noise, 2)
    low = np.round(mean - band, 2)
    high = np.round(mean + band, 2)
    return mean, low, high


def _metrics(model: str) -> dict[str, float]:
    base = {"Random Forest": (1.4, 1.9, 0.88), "Gradient Boosting": (1.2, 1.7, 0.91), "LSTM": (0.9, 1.3, 0.94)}
    mae, rmse, r2 = base[model]
    mae += round(RNG.uniform(-0.1, 0.1), 2)
    rmse += round(RNG.uniform(-0.1, 0.1), 2)
    r2 += round(RNG.uniform(-0.01, 0.01), 3)
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(min(r2, 0.99), 3)}


# ── lst_history.csv ───────────────────────────────────────────────────────────

hist_rows = []
for site, base in SITES.items():
    lst_vals = _history_lst(base, HISTORY_DAYS)
    for i, lst in enumerate(lst_vals):
        d = TODAY - timedelta(days=HISTORY_DAYS - i)
        hist_rows.append({"date": d.isoformat(), "site": site, "lst": lst})

pd.DataFrame(hist_rows).to_csv(OUT / "lst_history.csv", index=False)
print("wrote lst_history.csv")

# ── lst_forecast.csv ──────────────────────────────────────────────────────────

fc_rows = []
for site, base in SITES.items():
    last = _history_lst(base, HISTORY_DAYS)[-1]
    for model in MODELS:
        mean, low, high = _forecast_lst(last, model)
        for i in range(FORECAST_DAYS):
            d = TODAY + timedelta(days=i)
            fc_rows.append(
                {
                    "date": d.isoformat(),
                    "site": site,
                    "model": model,
                    "lst_forecast": mean[i],
                    "lst_low": low[i],
                    "lst_high": high[i],
                }
            )

pd.DataFrame(fc_rows).to_csv(OUT / "lst_forecast.csv", index=False)
print("wrote lst_forecast.csv")

# ── model_metrics.csv ─────────────────────────────────────────────────────────

met_rows = []
for site in SITES:
    for model in MODELS:
        m = _metrics(model)
        met_rows.append({"site": site, "model": model, **m})

pd.DataFrame(met_rows).to_csv(OUT / "model_metrics.csv", index=False)
print("wrote model_metrics.csv")
