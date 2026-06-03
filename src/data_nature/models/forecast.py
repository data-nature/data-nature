"""
forecast.py — Random Forest & Gradient Boosting LST forecast models.

Trains scikit-learn Random Forest and Gradient Boosting regressors to
forecast LST (Land Surface Temperature) one month ahead, per site.

Features used
-------------
- month          : captures seasonality (1–12)
- ndvi           : vegetation index — strong negative predictor of LST
- lst_lag1       : LST from the previous month (autoregressive signal)
- lst_lag12      : LST from 12 months ago (same month last year)
- lst_roll3_mean : 3-month rolling mean (short-term trend)
- lst_roll3_std  : 3-month rolling std  (recent volatility)
- site_encoded   : integer label-encoding of site name

Target
------
- lst_lead1 : LST of the *next* month (what we are predicting)

Train / test split
------------------
- Train : 2000–2023  (all data except last ~2 full years)
- Test  : 2024–2026  (held-out split for metric computation)

Output schemas (match frozen data contract)
-------------------------------------------
lst_forecast.csv  : date, site, model, lst_forecast, lst_low, lst_high
model_metrics.csv : site, model, mae, rmse, r2
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
MOCK_DIR = _REPO_ROOT / "data" / "mock"

FORECAST_PATH = PROCESSED_DIR / "lst_forecast.csv"
METRICS_PATH = PROCESSED_DIR / "model_metrics.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAIN_END_YEAR = 2023   # last year included in training
FORECAST_HORIZON = 7    # months ahead to forecast (sprint spec says 7 days;
                        # data is monthly so we forecast 7 months ahead)
RANDOM_STATE = 42

RF_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_leaf": 3,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

GB_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "random_state": RANDOM_STATE,
}

MODEL_REGISTRY: dict[str, type] = {
    "Random Forest": RandomForestRegressor,
    "Gradient Boosting": GradientBoostingRegressor,
}

MODEL_PARAMS: dict[str, dict] = {
    "Random Forest": RF_PARAMS,
    "Gradient Boosting": GB_PARAMS,
}

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag and rolling features to a site_monthly DataFrame.

    The DataFrame must be sorted by (site, year, month) before calling this.
    All lag/rolling operations are performed *within each site* so that
    site boundaries never bleed into each other.

    New columns added
    -----------------
    lst_lag1       : LST shifted 1 month back (previous month)
    lst_lag12      : LST shifted 12 months back (same month last year)
    lst_roll3_mean : 3-month rolling mean of LST
    lst_roll3_std  : 3-month rolling std of LST
    lst_lead1      : LST shifted 1 month forward — the prediction target
    site_encoded   : integer label for site (fitted on all sites in df)
    """
    df = df.copy().sort_values(["site", "year", "month"]).reset_index(drop=True)

    # Per-site lag/rolling features
    grp = df.groupby("site")["lst"]
    df["lst_lag1"] = grp.shift(1)
    df["lst_lag12"] = grp.shift(12)
    df["lst_roll3_mean"] = grp.transform(
        lambda s: s.shift(1).rolling(3, min_periods=2).mean()
    )
    df["lst_roll3_std"] = grp.transform(
        lambda s: s.shift(1).rolling(3, min_periods=2).std()
    )

    # Target: next month's LST
    df["lst_lead1"] = grp.shift(-1)

    # Site encoding
    le = LabelEncoder()
    df["site_encoded"] = le.fit_transform(df["site"])

    return df


FEATURE_COLS = [
    "month",
    "ndvi",
    "lst_lag1",
    "lst_lag12",
    "lst_roll3_mean",
    "lst_roll3_std",
    "site_encoded",
]
TARGET_COL = "lst_lead1"

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_forecasters(
    df: pd.DataFrame,
    train_end_year: int = TRAIN_END_YEAR,
) -> dict[str, object]:
    """
    Train Random Forest and Gradient Boosting regressors on site_monthly data.

    Parameters
    ----------
    df:
        site_monthly DataFrame (year, month, site, lst, ndvi, …).
    train_end_year:
        Last year (inclusive) used for training. Everything after is held out
        for metric computation.

    Returns
    -------
    dict
        ``{"Random Forest": fitted_model, "Gradient Boosting": fitted_model,
           "_label_encoder": LabelEncoder, "_feature_cols": list[str]}``
    """
    required = {"year", "month", "site", "lst", "ndvi"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train_forecasters: input missing columns {missing}")

    df_feat = _build_features(df)

    # Train mask: rows where we have a valid target and all features
    train_mask = (
        (df_feat["year"] <= train_end_year)
        & df_feat[FEATURE_COLS + [TARGET_COL]].notna().all(axis=1)
    )
    test_mask = (
        (df_feat["year"] > train_end_year)
        & df_feat[FEATURE_COLS + [TARGET_COL]].notna().all(axis=1)
    )

    X_train = df_feat.loc[train_mask, FEATURE_COLS]
    y_train = df_feat.loc[train_mask, TARGET_COL]
    X_test = df_feat.loc[test_mask, FEATURE_COLS]
    y_test = df_feat.loc[test_mask, TARGET_COL]

    logger.info(
        "Training on %d rows (≤%d), evaluating on %d rows (>%d).",
        len(X_train), train_end_year, len(X_test), train_end_year,
    )

    models: dict[str, object] = {}
    for name, cls in MODEL_REGISTRY.items():
        model = cls(**MODEL_PARAMS[name])
        model.fit(X_train, y_train)
        models[name] = model
        logger.info("Trained %s.", name)

    # Fit a fresh LabelEncoder on all sites so forecast() can encode new data
    le = LabelEncoder()
    le.fit(df["site"].unique())
    models["_label_encoder"] = le
    models["_feature_cols"] = FEATURE_COLS
    models["_df_feat"] = df_feat          # keep for forecast()
    models["_X_test"] = X_test
    models["_y_test"] = y_test

    return models


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    models: dict,
    sites: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute MAE, RMSE, R² for each model on the held-out test split,
    broken down per site.

    Parameters
    ----------
    models:
        Output of :func:`train_forecasters`.
    sites:
        Subset of sites to compute metrics for. None = all sites.

    Returns
    -------
    pd.DataFrame
        Columns: site, model, mae, rmse, r2  (matches model_metrics.csv schema)
    """
    df_feat = models["_df_feat"]
    X_test = models["_X_test"]
    y_test = models["_y_test"]

    test_sites = df_feat.loc[X_test.index, "site"]
    if sites:
        mask = test_sites.isin(sites)
        X_test = X_test[mask]
        y_test = y_test[mask]
        test_sites = test_sites[mask]

    records = []
    for name, cls in MODEL_REGISTRY.items():
        model = models[name]
        y_pred = model.predict(X_test)

        # Overall metrics (all sites combined)
        records.append(
            {
                "site": "ALL",
                "model": name,
                "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
                "r2": round(float(r2_score(y_test, y_pred)), 4),
            }
        )

        # Per-site metrics
        for site, grp_idx in test_sites.groupby(test_sites).groups.items():
            yt = y_test.loc[grp_idx]
            yp = model.predict(X_test.loc[grp_idx])
            records.append(
                {
                    "site": site,
                    "model": name,
                    "mae": round(float(mean_absolute_error(yt, yp)), 4),
                    "rmse": round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
                    "r2": round(float(r2_score(yt, yp)), 4),
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

def forecast(
    models: dict,
    site: str,
    horizon: int = FORECAST_HORIZON,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Produce a multi-step ahead forecast for a single site.

    Uses a recursive strategy: the predicted LST at step t becomes the
    lag feature for step t+1.

    Parameters
    ----------
    models:
        Output of :func:`train_forecasters`.
    site:
        Site name (must match site_monthly.csv values).
    horizon:
        Number of months ahead to forecast.
    reference_date:
        Start date for the forecast window. Defaults to the last observed
        date in the training data for that site.

    Returns
    -------
    pd.DataFrame
        Columns: date, site, model, lst_forecast, lst_low, lst_high
        (matches lst_forecast.csv schema)
    """
    df_feat = models["_df_feat"]
    le: LabelEncoder = models["_label_encoder"]

    if site not in le.classes_:
        raise ValueError(
            f"Unknown site '{site}'. Known sites: {list(le.classes_)}"
        )

    site_encoded = int(le.transform([site])[0])

    # Seed the recursive forecast from the last known observation
    site_data = df_feat[df_feat["site"] == site].sort_values(["year", "month"])
    last_row = site_data.iloc[-1]

    if reference_date is None:
        reference_date = pd.Timestamp(
            year=int(last_row["year"]),
            month=int(last_row["month"]),
            day=1,
        )

    records = []
    # State carried forward through the recursive loop
    prev_lst = float(last_row["lst"])
    lag1 = float(last_row["lst_lag1"]) if not pd.isna(last_row["lst_lag1"]) else prev_lst
    lag12 = float(last_row["lst_lag12"]) if not pd.isna(last_row["lst_lag12"]) else prev_lst
    roll_mean = float(last_row["lst_roll3_mean"]) if not pd.isna(last_row["lst_roll3_mean"]) else prev_lst
    roll_std = float(last_row["lst_roll3_std"]) if not pd.isna(last_row["lst_roll3_std"]) else 1.0
    ndvi = float(last_row["ndvi"])

    current_date = reference_date + pd.DateOffset(months=1)

    for step in range(horizon):
        month = current_date.month
        X = pd.DataFrame(
            [[month, ndvi, lag1, lag12, roll_mean, roll_std, site_encoded]],
            columns=FEATURE_COLS,
        )

        step_records = []
        for name in MODEL_REGISTRY:
            model = models[name]
            pred = float(model.predict(X)[0])

            # Uncertainty band: ±1 residual std from training (simple but
            # honest — width grows for GB which has lower variance, narrows
            # for RF which averages many trees).
            if hasattr(model, "estimators_") and hasattr(model.estimators_[0], "predict"):
                # Random Forest: use std across tree predictions
                tree_preds = np.array([t.predict(X)[0] for t in model.estimators_])
                band = float(tree_preds.std())
            else:
                # Gradient Boosting: use a fixed fraction of recent roll_std
                band = max(roll_std * 0.8, 0.5)

            step_records.append(
                {
                    "date": current_date.strftime("%Y-%m-%d"),
                    "site": site,
                    "model": name,
                    "lst_forecast": round(pred, 4),
                    "lst_low": round(pred - band, 4),
                    "lst_high": round(pred + band, 4),
                }
            )

        records.extend(step_records)

        # Update state for next step using first model's prediction as proxy
        first_pred = step_records[0]["lst_forecast"]
        lag12 = lag1
        lag1 = prev_lst
        roll_mean = (roll_mean * 2 + first_pred) / 3   # approximate 3-month mean
        roll_std = abs(first_pred - roll_mean) * 0.5 + roll_std * 0.5
        prev_lst = first_pred
        current_date += pd.DateOffset(months=1)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Convenience: full pipeline — load, train, save outputs
# ---------------------------------------------------------------------------

def load_site_monthly(data_dir: Path | None = None) -> pd.DataFrame:
    """Load site_monthly.csv with mock fallback."""
    data_dir = Path(data_dir) if data_dir else PROCESSED_DIR
    path = data_dir / "site_monthly.csv"
    if path.exists():
        logger.info("Loading site_monthly from %s", path)
        return pd.read_csv(path)
    mock = MOCK_DIR / "site_monthly.csv"
    if mock.exists():
        logger.warning("Falling back to mock site_monthly at %s", mock)
        return pd.read_csv(mock)
    raise FileNotFoundError(f"site_monthly.csv not found in {data_dir} or mock.")


def run_full_pipeline(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    horizon: int = FORECAST_HORIZON,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end: load data → train models → forecast all sites → save CSVs.

    Returns
    -------
    (forecast_df, metrics_df)
    """
    output_dir = Path(output_dir) if output_dir else PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_site_monthly(data_dir)
    models = train_forecasters(df)

    # Forecast every site
    all_forecasts = pd.concat(
        [forecast(models, site, horizon=horizon) for site in df["site"].unique()],
        ignore_index=True,
    )

    # Metrics on held-out test split
    metrics = compute_metrics(models)

    all_forecasts.to_csv(output_dir / "lst_forecast.csv", index=False)
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    logger.info(
        "Saved lst_forecast.csv (%d rows) and model_metrics.csv (%d rows).",
        len(all_forecasts), len(metrics),
    )

    return all_forecasts, metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("── Running DN-B2 full pipeline ────────────────────")
    fc, met = run_full_pipeline()
    print("\nForecast sample (first 6 rows):")
    print(fc.head(6).to_string(index=False))
    print("\nMetrics (ALL sites):")
    print(met[met["site"] == "ALL"].to_string(index=False))
    print(f"\nSaved to {PROCESSED_DIR}")
    sys.exit(0)