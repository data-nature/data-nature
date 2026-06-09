"""
DN-A0: Pre-sprint data preparation — Northern Israel NDVI + LST (2000-2026)

Fetches real MODIS data from Google Earth Engine for 8 Northern Israel sites,
saves per-site raw files to data/raw/ and a plain master CSV + site_locations.csv
to data/processed/.  No statistics or z-scores — those come in DN-A2.

Collections used:
  - NDVI: MODIS/061/MOD13Q1  (16-day composite, 250 m, scale × 0.0001)
  - LST:  MODIS/061/MOD11A1  (daily → monthly mean, 1 km, Kelvin × 0.02 − 273.15)

Run from the project root:
    python scripts/dn_a0_prepare_data.py
"""

import json
import sys
from datetime import date
from pathlib import Path

import ee
import pandas as pd

# Force UTF-8 output on Windows so Unicode print statements work
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── GEE init ──────────────────────────────────────────────────────────────────
print("Initializing Google Earth Engine...")
ee.Initialize(project="datanature")
print("GEE ready.\n")

TODAY = date.today()

# ── sites ─────────────────────────────────────────────────────────────────────
SITES = {
    "Upper_Galilee_Forest": {"lat": 33.05,  "lng": 35.25, "land_cover": "Forest"},
    "Jezreel_Valley_Agri":  {"lat": 32.65,  "lng": 35.20, "land_cover": "Agricultural"},
    "Hula_Valley_Wetland":  {"lat": 33.07,  "lng": 35.60, "land_cover": "Wetland"},
    "Nazareth_Urban":       {"lat": 32.70,  "lng": 35.30, "land_cover": "Urban"},
    "Carmel_Forest":        {"lat": 32.72,  "lng": 35.03, "land_cover": "Forest"},
    "Jordan_Valley_Arid":   {"lat": 32.50,  "lng": 35.55, "land_cover": "Arid"},
    "Haifa_Coastal_Urban":  {"lat": 32.82,  "lng": 34.99, "land_cover": "Urban"},
    "Lower_Galilee_Mixed":  {"lat": 32.80,  "lng": 35.40, "land_cover": "Mixed"},
}

northern_israel = ee.Geometry.Rectangle([34.9, 32.4, 35.9, 33.3])

# ── Step 1: Fetch NDVI ────────────────────────────────────────────────────────
print("=" * 60)
print("[1/3] Fetching NDVI time series (2000–present)")
print("      MODIS/061/MOD13Q1 · 16-day · 250 m")
print("=" * 60)

ndvi_collection = (
    ee.ImageCollection("MODIS/061/MOD13Q1")
    .filterDate("2000-02-18", TODAY.strftime("%Y-%m-%d"))
    .filterBounds(northern_israel)
    .select("NDVI")
)
print(f"  Images in collection: {ndvi_collection.size().getInfo()}\n")


def _extract_ndvi_ts(collection, point):
    def _row(image):
        d = image.date().format("YYYY-MM-dd")
        v = image.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=500)
        return ee.Feature(None, v.set("date", d))
    return ee.FeatureCollection(collection.map(_row))


ndvi_data = {}
for name, info in SITES.items():
    pt = ee.Geometry.Point([info["lng"], info["lat"]])
    fc = _extract_ndvi_ts(ndvi_collection, pt)
    records = []
    for f in fc.getInfo()["features"]:
        p = f["properties"]
        raw = p.get("NDVI")
        if raw is not None:
            records.append({"date": p["date"], "ndvi": round(raw * 0.0001, 5)})
    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
    ndvi_data[name] = df
    print(f"  {name}: {len(records)} obs")

# ── Step 2: Fetch monthly LST ─────────────────────────────────────────────────
print()
print("=" * 60)
print("[2/3] Fetching monthly LST time series (2000–present)")
print("      MODIS/061/MOD11A1 · daily→monthly mean · 1 km")
print("      (~8 minutes — one API call per site-month)")
print("=" * 60)

lst_collection = (
    ee.ImageCollection("MODIS/061/MOD11A1")
    .filterDate("2000-01-01", TODAY.strftime("%Y-%m-%d"))
    .filterBounds(northern_israel)
    .select("LST_Day_1km")
)
print(f"  Images in collection: {lst_collection.size().getInfo()}\n")

lst_data = {}
for name, info in SITES.items():
    pt = ee.Geometry.Point([info["lng"], info["lat"]])
    records = []
    for year in range(2000, TODAY.year + 1):
        max_month = TODAY.month if year == TODAY.year else 12
        for month in range(1, max_month + 1):
            start = f"{year}-{month:02d}-01"
            end = f"{year + 1}-01-01" if month == 12 else f"{year}-{month + 1:02d}-01"
            try:
                img = lst_collection.filterDate(start, end).mean()
                val = (
                    img.reduceRegion(reducer=ee.Reducer.mean(), geometry=pt, scale=1000)
                    .get("LST_Day_1km")
                    .getInfo()
                )
                if val is not None:
                    records.append({
                        "date": f"{year}-{month:02d}-01",
                        "year": year,
                        "month": month,
                        "lst": round((val * 0.02) - 273.15, 2),
                    })
            except Exception:
                pass
    lst_data[name] = pd.DataFrame(records)
    print(f"  {name}: {len(records)} monthly records")

# ── Step 3: Save files ────────────────────────────────────────────────────────
print()
print("=" * 60)
print("[3/3] Saving files")
print("=" * 60)

all_dfs = []
for name, info in SITES.items():
    lst_df = lst_data[name].copy()
    ndvi_df = ndvi_data[name].copy()

    if lst_df.empty:
        print(f"  WARNING: {name} — no LST data, skipping")
        continue

    # Monthly NDVI mean (averaged from 16-day composites)
    if not ndvi_df.empty:
        ndvi_monthly = ndvi_df.groupby(["year", "month"])["ndvi"].mean().reset_index()
        merged = lst_df.merge(ndvi_monthly, on=["year", "month"], how="left")
    else:
        merged = lst_df.copy()
        merged["ndvi"] = None

    merged["site"] = name
    merged["land_cover"] = info["land_cover"]

    # Save per-site raw file (replace spaces with underscores for filename safety)
    out = RAW_DIR / f"{name.replace(' ', '_')}.csv"
    merged.to_csv(out, index=False)
    null_ndvi = int(merged["ndvi"].isna().sum())
    print(f"  data/raw/{out.name}  ({len(merged)} rows, NDVI nulls: {null_ndvi})")

    all_dfs.append(merged)

# site_locations.csv
loc_df = pd.DataFrame([
    {"site": name, "lat": v["lat"], "lng": v["lng"], "land_cover": v["land_cover"]}
    for name, v in SITES.items()
])
loc_df.to_csv(PROCESSED_DIR / "site_locations.csv", index=False)
print(f"\n  data/processed/site_locations.csv  ({len(loc_df)} sites)")

# master raw CSV (all sites combined, plain data — no stats)
if all_dfs:
    master = pd.concat(all_dfs, ignore_index=True)
    master["date"] = pd.to_datetime(master["date"]).dt.strftime("%Y-%m-%d")
    master = master.sort_values(["site", "date"]).reset_index(drop=True)
    master_path = PROCESSED_DIR / "master_raw.csv"
    master.to_csv(master_path, index=False)
    print(f"  data/processed/master_raw.csv      ({len(master)} rows, {master['site'].nunique()} sites)")

    # Sanity report
    report = {"generated": str(TODAY), "sites": {}}
    for site in sorted(master["site"].unique()):
        s = master[master["site"] == site]
        report["sites"][site] = {
            "rows": int(len(s)),
            "date_range": f"{s['date'].min()} → {s['date'].max()}",
            "lst_null": int(s["lst"].isna().sum()),
            "ndvi_null": int(s["ndvi"].isna().sum()),
            "lst_min": round(float(s["lst"].min()), 2),
            "lst_max": round(float(s["lst"].max()), 2),
        }
    with open(PROCESSED_DIR / "sanity_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"  data/processed/sanity_report.json")

print()
print("✓ DN-A0 data fetch complete.")
print(f"  data/raw/       → {len(SITES)} per-site CSVs")
print(f"  data/processed/ → site_locations.csv, master_raw.csv, sanity_report.json")
print()
print("Next: DN-A1 will formalise this into src/data_nature/ingest/")
print("      DN-A2 will add z-scores, baselines, and anomaly detection.")
