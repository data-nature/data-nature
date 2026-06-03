# Project Status — data-nature

## What this project is

A data science tool that monitors vegetation health and heat stress across 8 sites in Northern Israel.
It pulls real satellite data from NASA (via Google Earth Engine), runs statistical analysis on it,
and shows the results on a web dashboard built with Streamlit.

The 8 monitored sites cover different land types: forests, farmland, wetlands, urban areas, and arid zones.

---

## What works right now

The project has a running Streamlit dashboard with 7 pages (heatmap, anomalies, simulator, forecasting,
optimization, reports, and an AI Q&A page). It was previously running on mock/synthetic data.
The backend Python package is now being built out with real logic.

---

## What Ahmad did in this sprint

### DN-A1 — Satellite data ingest pipeline

**Purpose:** Connect to Google Earth Engine and pull real monthly temperature and vegetation data
for all 8 sites, going back to the year 2000.

**Files added:**

- `src/data_nature/ingest/earth_engine.py`
  The main ingest module. Contains three things:
  - `authenticate()` — logs into Google Earth Engine using a service account key (or falls back to
    interactive login if no key is set up). Gives a clear error message if login fails instead of crashing.
  - `fetch_site_series(site, start, end)` — pulls monthly LST (land surface temperature in °C) and
    NDVI (vegetation index) for one named site between two dates. Returns a clean table.
  - `run_pipeline()` — runs the above for all 8 sites and saves one CSV file per site into `data/raw/`.
  - Data comes from two NASA MODIS satellite products: `MODIS/061/MOD13Q1` for vegetation and
    `MODIS/061/MOD11A1` for temperature.

- `src/data_nature/ingest/__init__.py`
  Makes the above functions importable as `from data_nature.ingest import fetch_site_series`.

- `.env.example`
  Updated with clear instructions explaining how to set up the Google Earth Engine credentials,
  including both the service account method (for deployment) and the simple interactive login
  method (for local development).

- `tests/test_earth_engine.py`
  26 unit tests covering authentication, data extraction, and the pipeline runner.
  All tests use mocks — no real satellite connection needed to run them.

---

### DN-A2 — Statistical analysis engine

**Purpose:** Given the satellite data, detect temperature anomalies per site and run statistical
tests to understand the relationship between vegetation and temperature.

**Files added:**

- `src/data_nature/stats/anomaly.py`
  Two functions:
  - `compute_zscores(df)` — for each site and each calendar month, computes the long-run historical
    average temperature (the "baseline") and measures how far the current reading is from that average
    in standard deviations. A z-score of 0 means normal, a high positive z-score means unusually hot.
  - `detect_anomalies(df, thresholds)` — filters the data to only the anomalous readings and labels
    each one: **Warning** (z ≥ 1.5), **Severe** (z ≥ 2.5), or **Critical** (z ≥ 3.5).
    Returns a table in the exact format the dashboard expects (`anomalies.csv` schema).

- `src/data_nature/stats/regression.py`
  Two functions:
  - `fit_ndvi_lst(df)` — fits a linear regression to answer: "does more vegetation predict lower
    temperature?" Runs both a simple model (temperature vs vegetation only) and a richer model
    that also accounts for land cover type and month. Returns the slope, R², and p-values.
    The expected finding is a **negative coefficient** — greener sites run cooler.
  - `compare_site_types(df)` — runs a one-way ANOVA test to answer the research question:
    "is mean temperature statistically different across land cover types (forest vs arid vs urban etc.)?"
    Also runs pairwise t-tests between every pair of land types.
    H0 = all land types have the same mean temperature. H1 = at least one differs.

- `src/data_nature/stats/__init__.py`
  Makes all four functions importable from `data_nature.stats`.

- `tests/test_stats.py`
  34 unit tests. Key scenarios covered:
  - A synthetic 20°C temperature spike is correctly flagged as Critical.
  - A perfectly stable series produces zero anomalies.
  - The NDVI regression coefficient comes out negative (scientifically expected).
  - ANOVA correctly rejects H0 when groups have different means, and does not reject it when they are the same.

---

## What Alaa is working on (her branch: `dn-b1-eda-baselines`)

Alaa built the baseline computation module (`src/data_nature/stats/baselines.py`) and an EDA notebook.
Her work computes rolling historical baselines per site and per month, validates the data schema,
and produces `data/processed/site_baselines.csv`. Not yet merged into main.

---

## Overall file structure (what matters)

```
data-nature/
├── app/                    Streamlit dashboard (7 pages)
├── data/
│   ├── raw/                Per-site CSV files from Earth Engine (8 files)
│   ├── mock/               Synthetic data for dashboard testing
│   └── processed/          Combined datasets, baselines, vector index for AI Q&A
├── src/data_nature/
│   ├── ingest/             NEW — satellite data pipeline (DN-A1)
│   ├── stats/              NEW — anomaly detection + regression + ANOVA (DN-A2)
│   ├── rag/                AI Q&A over research papers (ChromaDB + Gemini)
│   └── models/             Forecasting models (not yet built)
├── tests/                  Unit tests (no internet or real credentials needed)
├── .env.example            How to configure credentials
└── pyproject.toml          Dependencies and project config
```

---

## How to run the tests

```bash
.venv\Scripts\python.exe -m pytest tests/ -v
```

## How to run the dashboard

```bash
streamlit run app/Home.py
```
