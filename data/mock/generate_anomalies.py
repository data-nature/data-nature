"""
Mock data generator for the Anomaly Detection page.

Run:  python data/mock/generate_anomalies.py

Produces two CSV files in data/mock/:
  - lst_timeseries.csv   180-day LST per site with rolling baseline
  - anomalies.csv        Detected anomalies (z > 1.5) with severity + status

Replace these files with real GEE outputs when ready.
Column names and dtypes must stay the same for the page to work.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
RNG = np.random.default_rng(7)

SITES: dict[str, dict] = {
    "Hula Valley":    {"base": 22.0, "amp": 10.0},
    "Golan Heights":  {"base": 18.0, "amp":  9.0},
    "Galilee North":  {"base": 20.0, "amp":  9.5},
    "Galilee South":  {"base": 21.5, "amp": 10.0},
    "Mount Carmel":   {"base": 19.5, "amp":  9.0},
    "Jezreel Valley": {"base": 23.0, "amp": 11.0},
    "Jordan Valley":  {"base": 28.0, "amp": 12.0},
    "Mount Tabor":    {"base": 20.5, "amp": 10.0},
}

END   = date(2026, 5, 16)
START = END - timedelta(days=179)  # 180 days total
DATES = [START + timedelta(days=i) for i in range(180)]

WINDOW = 30   # baseline rolling window


def _seasonal(d: date, base: float, amp: float) -> float:
    """Smooth seasonal LST (peaks in July, troughs in January)."""
    doy = d.timetuple().tm_yday
    return base + amp * np.sin(2 * np.pi * (doy - 80) / 365)


# ── lst_timeseries.csv ────────────────────────────────────────────────────────

ts_rows = []
for site, params in SITES.items():
    lsts = []
    for d in DATES:
        seasonal = _seasonal(d, params["base"], params["amp"])
        # Occasional heat spike (5 % chance, +3–6 °C)
        spike = RNG.choice([0.0, RNG.uniform(3, 6)], p=[0.95, 0.05])
        noise = RNG.normal(0, 0.7)
        lsts.append(round(seasonal + spike + noise, 2))

    lst_series = pd.Series(lsts, index=DATES)
    roll_mean  = lst_series.rolling(WINDOW, min_periods=WINDOW // 2).mean()
    roll_std   = lst_series.rolling(WINDOW, min_periods=WINDOW // 2).std().clip(lower=0.5)

    for d, lst, mean, std in zip(DATES, lsts, roll_mean, roll_std):
        ts_rows.append({
            "date":          d.isoformat(),
            "site":          site,
            "lst":           lst,
            "baseline_mean": round(mean, 2) if not np.isnan(mean) else None,
            "baseline_std":  round(std,  2) if not np.isnan(std)  else None,
        })

ts_df = pd.DataFrame(ts_rows)
ts_df.to_csv(OUT / "lst_timeseries.csv", index=False)
print(f"wrote lst_timeseries.csv  ({len(ts_df)} rows)")

# ── anomalies.csv ─────────────────────────────────────────────────────────────

def _severity(z: float) -> str:
    if z >= 3.0:
        return "Critical"
    if z >= 2.0:
        return "Severe"
    return "Mild"


STATUSES = ["New", "New", "Reviewed", "Handled"]

anom_rows = []
ts_df["date"] = pd.to_datetime(ts_df["date"])
for _, row in ts_df.dropna(subset=["baseline_mean", "baseline_std"]).iterrows():
    z = (row["lst"] - row["baseline_mean"]) / row["baseline_std"]
    if z >= 1.5:
        ndvi_change = round(RNG.uniform(-0.18, -0.04), 3)
        anom_rows.append({
            "date":        row["date"].date().isoformat(),
            "site":        row["site"],
            "lst":         row["lst"],
            "baseline":    row["baseline_mean"],
            "z_score":     round(z, 2),
            "severity":    _severity(z),
            "status":      RNG.choice(STATUSES),
            "ndvi_change": ndvi_change,
        })

anom_df = pd.DataFrame(anom_rows).sort_values(["date", "site"]).reset_index(drop=True)
anom_df.to_csv(OUT / "anomalies.csv", index=False)
print(f"wrote anomalies.csv       ({len(anom_df)} rows)")
