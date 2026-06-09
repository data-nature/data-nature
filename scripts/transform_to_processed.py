"""
Transform data/raw/ per-site CSVs into the canonical data-contract CSVs
under data/processed/, matching the mock column schemas exactly.

Columns that require DN-A2 (z-scores, baselines, anomalies) are left as
NaN / False placeholders so pages still load against the right schema.

Run from the project root:
    python scripts/transform_to_processed.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Coordinates for site_locations.csv (from the original GEE fetch)
SITE_COORDS = {
    "Upper_Galilee_Forest": {"lat": 33.05,  "lng": 35.25, "land_cover": "Forest"},
    "Jezreel_Valley_Agri":  {"lat": 32.65,  "lng": 35.20, "land_cover": "Agricultural"},
    "Hula_Valley_Wetland":  {"lat": 33.07,  "lng": 35.60, "land_cover": "Wetland"},
    "Nazareth_Urban":       {"lat": 32.70,  "lng": 35.30, "land_cover": "Urban"},
    "Carmel_Forest":        {"lat": 32.72,  "lng": 35.03, "land_cover": "Forest"},
    "Jordan_Valley_Arid":   {"lat": 32.50,  "lng": 35.55, "land_cover": "Arid"},
    "Haifa_Coastal_Urban":  {"lat": 32.82,  "lng": 34.99, "land_cover": "Urban"},
    "Lower_Galilee_Mixed":  {"lat": 32.80,  "lng": 35.40, "land_cover": "Mixed"},
}

# ── Load all raw per-site CSVs ─────────────────────────────────────────────────
raw_files = sorted(RAW_DIR.glob("*.csv"))
if not raw_files:
    print("No raw CSVs found in data/raw/ — run dn_a0_prepare_data.py first.")
    sys.exit(1)

dfs = []
for f in raw_files:
    df = pd.read_csv(f, parse_dates=["date"])
    dfs.append(df)
    print(f"  Loaded {f.name}  ({len(df)} rows)")

master = pd.concat(dfs, ignore_index=True)
master = master.sort_values(["site", "date"]).reset_index(drop=True)
print(f"\n  Master: {len(master)} rows, {master['site'].nunique()} sites\n")

# ── 1. site_locations.csv  (site, lat, lng, land_cover) ───────────────────────
loc_rows = []
for site in master["site"].unique():
    coords = SITE_COORDS.get(site, {})
    loc_rows.append({
        "site":       site,
        "lat":        coords.get("lat", np.nan),
        "lng":        coords.get("lng", np.nan),
        "land_cover": coords.get("land_cover", master.loc[master["site"] == site, "land_cover"].iloc[0]),
    })
loc_df = pd.DataFrame(loc_rows).sort_values("site").reset_index(drop=True)
loc_df.to_csv(PROCESSED_DIR / "site_locations.csv", index=False)
print(f"site_locations.csv     cols: {list(loc_df.columns)}  rows: {len(loc_df)}")

# ── 2. site_monthly.csv  (year, month, site, lst, ndvi,
#                          z_score_lst, z_score_ndvi, delta, is_anomaly) ────────
monthly = master[["year", "month", "site", "lst", "ndvi"]].copy()
monthly["z_score_lst"]  = np.nan   # filled by DN-A2
monthly["z_score_ndvi"] = np.nan   # filled by DN-A2
monthly["delta"]        = np.nan   # filled by DN-A2
monthly["is_anomaly"]   = False    # filled by DN-A2
monthly = monthly.sort_values(["site", "year", "month"]).reset_index(drop=True)
monthly.to_csv(PROCESSED_DIR / "site_monthly.csv", index=False)
print(f"site_monthly.csv       cols: {list(monthly.columns)}  rows: {len(monthly)}")

# ── 3. lst_timeseries.csv  (date, site, lst, baseline_mean, baseline_std) ──────
ts = master[["date", "site", "lst"]].copy()
ts["date"] = ts["date"].dt.strftime("%Y-%m-%d")
ts["baseline_mean"] = np.nan   # filled by DN-A2
ts["baseline_std"]  = np.nan   # filled by DN-A2
ts = ts.sort_values(["site", "date"]).reset_index(drop=True)
ts.to_csv(PROCESSED_DIR / "lst_timeseries.csv", index=False)
print(f"lst_timeseries.csv     cols: {list(ts.columns)}  rows: {len(ts)}")

# ── 4. lst_history.csv  (date, site, lst) ─────────────────────────────────────
hist = master[["date", "site", "lst"]].copy()
hist["date"] = hist["date"].dt.strftime("%Y-%m-%d")
hist = hist.sort_values(["site", "date"]).reset_index(drop=True)
hist.to_csv(PROCESSED_DIR / "lst_history.csv", index=False)
print(f"lst_history.csv        cols: {list(hist.columns)}  rows: {len(hist)}")

print("\nDone. data/processed/ now matches the mock column schemas.")
print("(z_score_lst, z_score_ndvi, delta, is_anomaly, baseline_mean, baseline_std")
print(" are NaN placeholders — DN-A2 will compute and fill them.)")
