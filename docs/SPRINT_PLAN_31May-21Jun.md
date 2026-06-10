# Data Nature — Implementation Sprint Plan

**From mock data to real models**

| | |
|---|---|
| **Sprint window** | Sunday 31 May 2026 → Sunday 21 June 2026 (3 weeks) |
| **Team** | Ahmad Tawil · Alaa Barazi |
| **Goal** | Replace all mock CSV fixtures with real Earth Engine data and real, working models across every module |
| **Repo** | https://github.com/AhmadTawil1/data-nature.git |

---

## 1. Where the project stands today

The Streamlit app is fully built visually — all 7 pages render — but everything runs on **mock CSV fixtures** in `data/mock/`. The `src/data_nature` library is mostly empty scaffolding:

| Module | Status | This sprint |
|---|---|---|
| `src/data_nature/ingest/` | empty stub | Build the real Earth Engine pipeline |
| `src/data_nature/stats/` | empty stub | Build z-score, regression, significance tests |
| `src/data_nature/models/` | empty stub | Build ecological models, cellular automaton, GA, ML/DL forecast |
| `src/data_nature/viz/` | empty stub | Build shared Plotly/Folium helpers |
| `src/data_nature/reports/` | empty stub | Build real ReportLab PDF generation |
| `src/data_nature/rag/` | **done** | Verify only — already implemented |

By 21 June, every page should read from `data/processed/` (real data) or compute live, with mock CSVs kept only as an offline fallback.

---

## 2. How the work is split

The split is a **balanced shuffle**: both people do the same *types* of work — a data-layer module, a data-driven model, an algorithmic simulation model, ecological models, visualizations, and full pages. Nobody is "the models person" or "the visuals person".

Each person owns **complete vertical slices**: for every page they own, they build the model behind it, its visualization, and the page wiring — so they can develop and demo end-to-end without waiting on the other.

| Type of work | Ahmad | Alaa |
|---|---|---|
| Data layer | Earth Engine ingest pipeline | Raw → processed transform |
| Data-driven model | Z-score statistics + regression / ANOVA | ML forecast (Random Forest + Gradient Boosting) + LSTM |
| Algorithmic model | Genetic algorithm (planting optimization) | Cellular automaton (heat diffusion) |
| Ecological models | Logistic growth + energy-flow / heat budget | Lotka-Volterra |
| Visualization | Heatmap maps, anomaly charts, optimization charts | Forecast charts, simulator grid viz, report charts |
| Pages owned | Heatmap · Anomalies · Optimization | Forecast · Simulator · Reports |

Each person: **6 owned tickets** (2 per week) **+ 2 shared tickets**.

---

## 3. Timeline & milestones

| Week | Dates | Theme | Milestone |
|---|---|---|---|
| **Week 1** | Sun 31 May – Sat 6 Jun | Data layer + data-driven models | Real data downloaded & processed; stats engine + RF/GB models working |
| **Week 2** | Sun 7 Jun – Sat 13 Jun | First pages + deep learning | Heatmap + Forecast pages live on real data; LSTM trained |
| **Week 3** | Sun 14 Jun – Sun 21 Jun | Algorithmic models, remaining pages, ecology, verification | All pages on real data; ecological models in; final review & deploy |

**Hard checkpoints:** end of Sat 6 Jun, end of Sat 13 Jun, **final review Sat 20 – Sun 21 Jun**.

---

## 4. How we work (conventions)

- **Branch per ticket.** Branch name = ticket ID lowercased, e.g. `dn-a1-ingest-pipeline`. Open a PR into `main`; the other person reviews before merge.
- **The data contract is frozen.** Every processed CSV must match the existing mock schema exactly so pages keep working. Do not rename columns. The schemas:

  | File | Columns |
  |---|---|
  | `site_locations.csv` | `site, lat, lng, land_cover` |
  | `site_monthly.csv` | `year, month, site, lst, ndvi, z_score_lst, z_score_ndvi, delta, is_anomaly` |
  | `anomalies.csv` | `date, site, lst, baseline, z_score, severity, status, ndvi_change` |
  | `lst_timeseries.csv` | `date, site, lst, baseline_mean, baseline_std` |
  | `lst_history.csv` | `date, site, lst` |
  | `lst_forecast.csv` | `date, site, model, lst_forecast, lst_low, lst_high` |
  | `model_metrics.csv` | `site, model, mae, rmse, r2` |

- **Mock-first stays.** Pages must still fall back to `data/mock/` when real data or credentials are missing — keep the existing graceful-fallback pattern. This is also what lets both people work independently: develop a model against mock data, then swap to `data/processed/` with zero page changes.
- **Tests required.** Every model ticket ships with `pytest` tests under `tests/`. CI (`.github/workflows/ci.yml`) must stay green.
- **Daily sync.** Short written update each working day; full review call at each weekly checkpoint.

---

## 5. Shared tickets

### DN-S1 — Sprint kickoff & repo prep
- **Owner:** Both · **When:** Sun 31 May · **Effort:** 0.5 day
- **Description:** Align on the data contract above, agree branch/PR workflow, confirm Python 3.11/3.12 env on both machines, create `data/processed/` subfolders, and add an `ingest`/`models` section to the README roadmap.
- **Acceptance criteria:** Both can run `streamlit run app/Home.py` and `pytest` clean; PR review flow agreed; the data-contract table is committed to `docs/`.
- **Dependencies:** none.

### DN-S2 — Final integration, Home dashboard, review & deploy
- **Owner:** Both · **When:** Sat 20 – Sun 21 Jun · **Effort:** 1.5 days
- **Description:** Merge all branches, wire the **Home** dashboard KPIs to real aggregated data, run the full app end-to-end, cross-review each other's modules, fix integration bugs, update README, and redeploy the Streamlit Cloud app.
- **Acceptance criteria:** All 7 pages + Home render on real data with mock fallback intact; `pytest` green; CI green; deployed app live; README + roadmap updated.
- **Dependencies:** all Week 3 tickets.

---

## Week 1 — Data layer + data-driven models
**Sun 31 May – Sat 6 Jun**

### DN-A1 — Earth Engine ingest pipeline *(Ahmad — data)*
- **Effort:** 3 days · **Files:** `src/data_nature/ingest/earth_engine.py`, `src/data_nature/ingest/__init__.py`, `.env.example`
- **Description:** Build the real satellite ingest pipeline. Authenticate Earth Engine, define the 8 site AOIs (from `site_locations.csv`), and pull monthly **MODIS LST** and **NDVI** (MODIS/Landsat) for 2000–2026. Export raw exports to `data/raw/`. This is the live download Ahmad runs.
- **Acceptance criteria:**
  - `ingest.fetch_site_series(site, start, end)` returns a tidy DataFrame of monthly LST + NDVI.
  - Running the pipeline writes raw per-site files to `data/raw/`.
  - Missing/failed EE auth degrades gracefully with a clear message (no crash).
  - Docstrings name the EE collections used.
- **Dependencies:** DN-S1.

### DN-A2 — Statistical engine: z-score + regression / ANOVA *(Ahmad — data-driven model)*
- **Effort:** 3 days · **Files:** `src/data_nature/stats/anomaly.py`, `src/data_nature/stats/regression.py`, `src/data_nature/stats/__init__.py`, `tests/test_stats.py`
- **Description:** Build the statistical core. (1) Per-site, per-month rolling **baseline** + **z-score** of LST with severity classification (Warning / Severe / Critical). (2) Linear & multi-linear **regression** of LST on NDVI (+ land cover, month). (3) **t-test / ANOVA** comparing LST across land-cover types — answers the HW1 research question (H0 vs H1).
- **Acceptance criteria:**
  - `stats.compute_zscores(df)` and `stats.detect_anomalies(df, thresholds)` return DataFrames matching the `anomalies.csv` schema.
  - `stats.fit_ndvi_lst(df)` returns coefficients, R², p-values; `stats.compare_site_types(df)` returns an ANOVA table + pairwise t-tests.
  - `pytest tests/test_stats.py` covers a known synthetic spike, a clean series, and the expected negative NDVI→LST sign.
- **Dependencies:** DN-S1 (develop against mock data, then point at `data/processed/`).

### DN-B1 — Raw → processed data transform *(Alaa — data)*
- **Effort:** 2.5 days · **Files:** `src/data_nature/ingest/build_datasets.py`, `data/processed/*`
- **Description:** Build the script that converts raw EE exports into the canonical processed CSVs (`site_monthly.csv`, `lst_timeseries.csv`, `lst_history.csv`) — **exact mock schema**. Handle gaps, cloud-masked nulls, and unit conversion (MODIS LST Kelvin → °C).
- **Acceptance criteria:**
  - `python -m data_nature.ingest.build_datasets` produces all three CSVs in `data/processed/`.
  - Column names/types match the data contract; no NaN in required columns.
  - A row-count / date-range sanity report is printed.
- **Dependencies:** DN-A1 (can start against a small sample export, finalize once DN-A1 lands).

### DN-B2 — ML forecast models: Random Forest + Gradient Boosting *(Alaa — data-driven model)*
- **Effort:** 2.5 days · **Files:** `src/data_nature/models/forecast.py`, `src/data_nature/models/__init__.py`, `tests/test_forecast.py`
- **Description:** Train scikit-learn **Random Forest** and **Gradient Boosting** regressors to forecast LST 7 days ahead, using NDVI, season, month and site as features, on the real 2000–2024 series. Output predictions with uncertainty bands and an MAE/RMSE/R² metrics table.
- **Acceptance criteria:**
  - `models.train_forecasters(df)` and `models.forecast(site, horizon=7)` produce a DataFrame matching the `lst_forecast.csv` schema.
  - Metrics written matching the `model_metrics.csv` schema.
  - `pytest` checks output shape and that R² is computed on a held-out split.
- **Dependencies:** DN-S1 (develop against mock data, then point at `data/processed/`).

---

## Week 2 — First pages + deep learning
**Sun 7 Jun – Sat 13 Jun**

### DN-A3 — Heatmap page: map viz + real data *(Ahmad — visualization + page)*
- **Effort:** 2.5 days · **Files:** `src/data_nature/viz/maps.py`, `app/pages/1_Heatmap.py`
- **Description:** Build the Folium/Earth Engine map helpers in `viz/maps.py`, then wire the **Heatmap** page to real LST/NDVI layers for the 8 sites — layer/year/month controls, color legend, site detail panel, z-score history chart.
- **Acceptance criteria:**
  - Heatmap renders real LST & NDVI layers and the per-site z-score timeline.
  - Falls back to mock cleanly when data/credentials are missing.
- **Dependencies:** DN-A1, DN-B1.

### DN-A4 — Anomalies page: anomaly viz + real data *(Ahmad — visualization + page)*
- **Effort:** 2.5 days · **Files:** `src/data_nature/viz/charts.py` (anomaly charts), `app/pages/2_Anomalies.py`
- **Description:** Build anomaly visualizations (time-series with ±1σ/±2σ bands, severity-colored points, summary cards) and wire the **Anomalies** page to the DN-A2 statistical engine on real data, including the plain-language "explain anomaly" panel.
- **Acceptance criteria:**
  - Anomalies page renders real detected anomalies, filters (site/severity/status/date) work, summary cards correct.
  - Falls back to mock cleanly.
- **Dependencies:** DN-A2, DN-B1.

### DN-B3 — LSTM deep-learning forecast *(Alaa — data-driven model)*
- **Effort:** 2.5 days · **Files:** `src/data_nature/models/lstm.py`, `tests/test_lstm.py`
- **Description:** Build a PyTorch **LSTM** time-series model for 7-day LST forecasting trained on the historical series, registered as a third model alongside RF/GB. Include train/eval and a metrics entry.
- **Acceptance criteria:**
  - `models.LSTMForecaster` trains and predicts; output appended to the forecast/metrics tables under model name `LSTM`.
  - Training is reproducible (seeded) and runs on CPU in reasonable time.
  - A smoke test trains 1–2 epochs and asserts output shape.
- **Dependencies:** DN-B2.

### DN-B4 — Forecast page: forecast viz + real data *(Alaa — visualization + page)*
- **Effort:** 2.5 days · **Files:** `src/data_nature/viz/charts.py` (forecast charts), `app/pages/5_Forecast.py`
- **Description:** Build forecast visualizations (prediction line + uncertainty band, daily confidence cards, model-comparison table) and wire the **Forecast** page to the RF/GB/LSTM models on real data, with the automatic heat alert.
- **Acceptance criteria:**
  - Forecast page shows real 7-day predictions for all three models + real MAE/RMSE/R² table.
  - Falls back to mock cleanly.
- **Dependencies:** DN-B2, DN-B3.

---

## Week 3 — Algorithmic models, remaining pages, ecology & verification
**Sun 14 Jun – Sun 21 Jun**

### DN-A5 — Optimization: genetic algorithm + page *(Ahmad — algorithmic model + viz + page)*
- **Effort:** 3 days · **Files:** `src/data_nature/models/optimization.py`, `tests/test_optimization.py`, `src/data_nature/viz/charts.py` (convergence chart), `app/pages/4_Optimization.py`
- **Description:** Implement the **genetic algorithm** that finds optimal planting cell locations to minimize total heat load. Fitness balances predicted ΔLST (via the DN-A2 regression / NDVI relationship) against cost (cells/budget) — independent of Alaa's cellular automaton. Wire into the Optimization page: solution map, convergence chart, top-5 solutions, CSV export.
- **Acceptance criteria:**
  - `models.optimize_planting(site, params)` returns ranked solutions + convergence history.
  - Fitness improves monotonically across generations on a fixed seed — asserted in test.
  - Optimization page renders the solution map and exports a planting plan CSV.
- **Dependencies:** DN-A2.

### DN-A6 — Ecological models: logistic growth + energy flow *(Ahmad — ecological models + viz)*
- **Effort:** 2.5 days · **Files:** `src/data_nature/models/ecology.py` (logistic, energy flow), `tests/test_ecology_a.py`, sections in `app/pages/1_Heatmap.py` & `app/pages/2_Anomalies.py`
- **Description:** Implement two HW1 ecological models, each with a visualization: **logistic growth** for vegetation-cover dynamics (shown as a section in the Heatmap page) and an **energy-flow / heat-budget** model for a site (shown as a section in the Anomalies page).
- **Acceptance criteria:**
  - Each model exposes `simulate(params, steps)` returning a trajectory DataFrame, plus a Plotly chart.
  - Logistic curve saturates at carrying capacity — asserted in test.
  - Docstrings cite the governing equations.
- **Dependencies:** DN-A3, DN-A4.

### DN-B5 — Simulator: cellular automaton + Lotka-Volterra + page *(Alaa — algorithmic + ecological model + viz + page)*
- **Effort:** 3 days · **Files:** `src/data_nature/models/cellular_automaton.py`, `src/data_nature/models/ecology.py` (Lotka-Volterra), `tests/test_simulator.py`, `app/pages/3_Simulator.py`
- **Description:** Build the **cellular-automaton** heat-diffusion model (grid of land patches, transition rules driven by local NDVI, cell temperature and neighbours) and the **Lotka-Volterra** model for interacting populations (native vs invasive vegetation). Wire both into the **Simulator** page: before/after grids, ΔLST summary, LST trajectory chart, population-dynamics chart.
- **Acceptance criteria:**
  - `models.HeatCA(grid).step()` advances one generation; higher-NDVI cells measurably cool neighbours — asserted in test.
  - Lotka-Volterra produces stable oscillation on standard params — asserted in test.
  - Simulator page renders the before/after scenario and both charts; falls back to mock cleanly.
- **Dependencies:** DN-B1.

### DN-B6 — Reports: real PDF generation + page *(Alaa — visualization + page)*
- **Effort:** 2.5 days · **Files:** `src/data_nature/reports/pdf.py`, `src/data_nature/reports/__init__.py`, `app/pages/6_Reports.py`
- **Description:** Implement ReportLab PDF generation (cover page, KPI summary, anomaly table, charts) and wire the **Reports** page to export real data for any date range / site selection, plus the live preview.
- **Acceptance criteria:**
  - `reports.build_report(...)` produces a valid multi-page PDF from real data.
  - Reports page download button returns the generated PDF; CSV export still works.
- **Dependencies:** DN-A4, DN-B4 (pulls anomaly + forecast content).

---

## 6. Ticket summary

| Week | Ahmad | Alaa |
|---|---|---|
| **1** | DN-A1 EE ingest pipeline · DN-A2 Statistical engine (z-score + regression) | DN-B1 Raw→processed transform · DN-B2 ML forecast (RF/GB) |
| **2** | DN-A3 Heatmap page (maps) · DN-A4 Anomalies page (anomaly viz) | DN-B3 LSTM forecast · DN-B4 Forecast page (forecast viz) |
| **3** | DN-A5 Genetic-algorithm optimization + page · DN-A6 Ecological models (logistic + energy flow) | DN-B5 Cellular automaton + Lotka-Volterra + Simulator page · DN-B6 PDF reports + page |
| **Shared** | DN-S1 Kickoff (31 May) · DN-S2 Integration, Home & deploy (20–21 Jun) | |

Both people get the same job mix: a data module, a data-driven model, an algorithmic model, ecological models, visualizations, and 3 full pages each (~16 days of work apiece).

---

## 7. Risks & mitigations

- **Earth Engine quota / auth issues** — Ahmad starts DN-A1 on day 1; if EE blocks, fall back to a one-time bulk export and keep mock data as the offline path.
- **Cross-dependency on the data contract** — frozen schemas (Section 4) let each person develop models against mock data in parallel, then swap to `data/processed/` with no page changes.
- **LSTM training time** — keep the model small; RF/GB (DN-B2) is the guaranteed baseline if LSTM underperforms.
- **GA ↔ CA coupling avoided** — the genetic algorithm uses the regression-based ΔLST estimate, not Alaa's cellular automaton, so the two algorithmic tickets stay fully independent.
- **Week-3 crunch** — page-wiring is front-loaded into Weeks 2–3, leaving the final weekend for DN-S2.

---

## 8. Definition of done (sprint)

All 7 pages + Home run on real Earth Engine data with intact mock fallback; `ingest`, `stats`, `models`, `viz`, `reports` modules implemented and tested; `pytest` and CI green; research question answered with real statistics; Streamlit Cloud app redeployed; README / roadmap updated.
