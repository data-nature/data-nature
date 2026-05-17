"""
Mock data generator for the Interactive Heat Map page.

Run:  python data/mock/generate_heatmap.py

Produces two CSV files in data/mock/:
  - site_locations.csv   8 sites with lat/lng + land cover
  - site_monthly.csv     27 years × 12 months × 8 sites with LST, NDVI, z-scores

Column names and dtypes must stay the same for the page to work.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
RNG = np.random.default_rng(13)

SITES: dict[str, dict] = {
    "Hula Valley":    {"lat": 33.082, "lng": 35.610, "land_cover": "Wetland",              "base_lst": 22.0, "base_ndvi": 0.55},
    "Golan Heights":  {"lat": 33.000, "lng": 35.800, "land_cover": "Shrubland",            "base_lst": 18.0, "base_ndvi": 0.45},
    "Galilee North":  {"lat": 33.050, "lng": 35.500, "land_cover": "Forest",               "base_lst": 20.0, "base_ndvi": 0.60},
    "Galilee South":  {"lat": 32.750, "lng": 35.320, "land_cover": "Forest",               "base_lst": 21.5, "base_ndvi": 0.58},
    "Mount Carmel":   {"lat": 32.730, "lng": 34.960, "land_cover": "Mediterranean Forest", "base_lst": 19.5, "base_ndvi": 0.56},
    "Jezreel Valley": {"lat": 32.620, "lng": 35.200, "land_cover": "Cropland",             "base_lst": 23.0, "base_ndvi": 0.40},
    "Jordan Valley":  {"lat": 32.450, "lng": 35.550, "land_cover": "Arid Shrubland",       "base_lst": 28.0, "base_ndvi": 0.25},
    "Mount Tabor":    {"lat": 32.680, "lng": 35.390, "land_cover": "Forest",               "base_lst": 20.5, "base_ndvi": 0.52},
}

YEARS = range(2000, 2027)
MONTHS = range(1, 13)


def _lst(base: float, year: int, month: int) -> float:
    """LST with seasonal cycle peaking in July, long-term warming trend, occasional spikes."""
    seasonal = 12.0 * np.cos(2 * np.pi * (month - 7) / 12)
    trend = 0.025 * (year - 2000)
    noise = RNG.normal(0, 0.9)
    spike = RNG.choice([0.0, float(RNG.uniform(4, 9))], p=[0.95, 0.05])
    return round(base + seasonal + trend + noise + spike, 2)


def _ndvi(base: float, year: int, month: int) -> float:
    """NDVI with Mediterranean pattern: peak April, trough August. Slow long-term decline."""
    seasonal = 0.13 * np.cos(2 * np.pi * (month - 4) / 12)
    trend = -0.0007 * (year - 2000)
    noise = RNG.normal(0, 0.018)
    return round(float(np.clip(base + seasonal + trend + noise, 0.01, 0.99)), 3)


# ── site_monthly.csv ──────────────────────────────────────────────────────────

rows = []
for site, p in SITES.items():
    for year in YEARS:
        for month in MONTHS:
            rows.append({
                "year": year, "month": month, "site": site,
                "lst":  _lst(p["base_lst"], year, month),
                "ndvi": _ndvi(p["base_ndvi"], year, month),
            })

df = pd.DataFrame(rows)

# Climatological baseline: 2000–2019 mean + std per site × month
baseline = (
    df[df["year"] <= 2019]
    .groupby(["site", "month"])[["lst", "ndvi"]]
    .agg(["mean", "std"])
    .reset_index()
)
baseline.columns = ["site", "month", "lst_mean", "lst_std", "ndvi_mean", "ndvi_std"]
baseline["lst_std"]  = baseline["lst_std"].clip(lower=0.5)
baseline["ndvi_std"] = baseline["ndvi_std"].clip(lower=0.005)

df = df.merge(baseline, on=["site", "month"])
df["z_score_lst"]  = ((df["lst"]  - df["lst_mean"])  / df["lst_std"]).round(2)
df["z_score_ndvi"] = ((df["ndvi"] - df["ndvi_mean"]) / df["ndvi_std"]).round(2)
df["delta"]        = (df["z_score_lst"] - df["z_score_ndvi"]).round(2)
df["is_anomaly"]   = (df["z_score_lst"] >= 1.5) | (df["z_score_ndvi"] <= -1.5)

out_cols = ["year", "month", "site", "lst", "ndvi", "z_score_lst", "z_score_ndvi", "delta", "is_anomaly"]
df[out_cols].to_csv(OUT / "site_monthly.csv", index=False)
print(f"wrote site_monthly.csv  ({len(df)} rows)")

# ── site_locations.csv ────────────────────────────────────────────────────────

loc_rows = [
    {"site": s, "lat": p["lat"], "lng": p["lng"], "land_cover": p["land_cover"]}
    for s, p in SITES.items()
]
pd.DataFrame(loc_rows).to_csv(OUT / "site_locations.csv", index=False)
print("wrote site_locations.csv  (8 rows)")
