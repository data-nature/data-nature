"""
tests/test_forecast.py

Tests for src/data_nature/models/forecast.py covering:
  1. Feature engineering — lag/rolling columns created correctly
  2. train_forecasters — models train without error, return expected keys
  3. Output shape — forecast DataFrame matches lst_forecast.csv schema
  4. Metrics — computed on held-out split, R² is a real number
  5. Site isolation — forecasting one site doesn't affect another
  6. Unknown site — raises ValueError
  7. Integration — real processed data (skipped if unavailable)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_nature.models.forecast import (
    FEATURE_COLS,
    MODEL_REGISTRY,
    TRAIN_END_YEAR,
    _build_features,
    compute_metrics,
    forecast,
    train_forecasters,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_site_monthly(
    sites: list[str] | None = None,
    start_year: int = 2000,
    end_year: int = 2024,
) -> pd.DataFrame:
    """Generate a synthetic site_monthly DataFrame for testing."""
    if sites is None:
        sites = ["Forest_A", "Urban_B"]

    rows = []
    rng = np.random.default_rng(42)
    for site in sites:
        base_lst = rng.uniform(15, 30)
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # Realistic seasonal LST pattern + noise
                seasonal = 8 * np.sin(2 * np.pi * (month - 3) / 12)
                lst = base_lst + seasonal + rng.normal(0, 0.5)
                ndvi = 0.5 - 0.2 * np.sin(2 * np.pi * (month - 3) / 12) + rng.normal(0, 0.02)
                rows.append(
                    {
                        "year": year,
                        "month": month,
                        "site": site,
                        "lst": round(lst, 3),
                        "ndvi": round(float(np.clip(ndvi, 0.1, 0.9)), 4),
                        "z_score_lst": None,
                        "z_score_ndvi": None,
                        "delta": None,
                        "is_anomaly": False,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Feature engineering
# ---------------------------------------------------------------------------

class TestBuildFeatures:

    def test_lag1_shifts_by_one_month(self):
        """lst_lag1 at row i == lst at row i-1 within same site."""
        df = _make_site_monthly(sites=["S"], start_year=2000, end_year=2002)
        feat = _build_features(df)
        site_feat = feat[feat["site"] == "S"].reset_index(drop=True)

        for i in range(1, len(site_feat)):
            assert math.isclose(
                site_feat.loc[i, "lst_lag1"],
                site_feat.loc[i - 1, "lst"],
                rel_tol=1e-6,
            ), f"lag1 mismatch at row {i}"

    def test_lag12_shifts_by_12_months(self):
        """lst_lag12 at month M year Y == lst at month M year Y-1."""
        df = _make_site_monthly(sites=["S"], start_year=2000, end_year=2003)
        feat = _build_features(df).reset_index(drop=True)
        site = feat[feat["site"] == "S"].reset_index(drop=True)

        # First valid lag12 is at index 12
        assert math.isclose(
            site.loc[12, "lst_lag12"], site.loc[0, "lst"], rel_tol=1e-6
        )

    def test_lead1_is_next_month(self):
        """lst_lead1 at row i == lst at row i+1 within same site."""
        df = _make_site_monthly(sites=["S"], start_year=2000, end_year=2002)
        feat = _build_features(df)
        site = feat[feat["site"] == "S"].reset_index(drop=True)

        for i in range(len(site) - 1):
            if not pd.isna(site.loc[i, "lst_lead1"]):
                assert math.isclose(
                    site.loc[i, "lst_lead1"], site.loc[i + 1, "lst"], rel_tol=1e-6
                )

    def test_no_cross_site_leakage(self):
        """lag features for site A must not use site B values."""
        df = _make_site_monthly(sites=["A", "B"], start_year=2000, end_year=2001)
        feat = _build_features(df)

        # First row for site A after sorting: lag1 must be NaN (no prior row for A)
        first_a = feat[feat["site"] == "A"].sort_values(["year", "month"]).iloc[0]
        assert pd.isna(first_a["lst_lag1"])

    def test_all_feature_cols_present(self):
        """All required feature columns must exist after build_features."""
        df = _make_site_monthly()
        feat = _build_features(df)
        for col in FEATURE_COLS + ["lst_lead1"]:
            assert col in feat.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# 2. train_forecasters
# ---------------------------------------------------------------------------

class TestTrainForecasters:

    def test_returns_all_model_keys(self):
        """train_forecasters must return an entry for every model in registry."""
        df = _make_site_monthly()
        models = train_forecasters(df, train_end_year=2022)
        for name in MODEL_REGISTRY:
            assert name in models

    def test_models_are_fitted(self):
        """Each model must be fitted (has feature_importances_ attribute)."""
        df = _make_site_monthly()
        models = train_forecasters(df, train_end_year=2022)
        for name in MODEL_REGISTRY:
            assert hasattr(models[name], "feature_importances_")

    def test_missing_column_raises(self):
        """Raises ValueError if required column is absent."""
        df = _make_site_monthly().drop(columns=["ndvi"])
        with pytest.raises(ValueError, match="missing columns"):
            train_forecasters(df)

    def test_label_encoder_knows_all_sites(self):
        """LabelEncoder in models must contain all sites from input df."""
        sites = ["Site_X", "Site_Y", "Site_Z"]
        df = _make_site_monthly(sites=sites)
        models = train_forecasters(df, train_end_year=2022)
        le = models["_label_encoder"]
        for s in sites:
            assert s in le.classes_


# ---------------------------------------------------------------------------
# 3. forecast output shape and schema
# ---------------------------------------------------------------------------

class TestForecastOutput:

    @pytest.fixture(scope="class")
    def trained_models(self):
        df = _make_site_monthly(sites=["Forest_A", "Urban_B"])
        return train_forecasters(df, train_end_year=2022)

    def test_output_columns_match_schema(self, trained_models):
        """forecast() must return exactly the lst_forecast.csv schema columns."""
        fc = forecast(trained_models, "Forest_A", horizon=3)
        expected = {"date", "site", "model", "lst_forecast", "lst_low", "lst_high"}
        assert set(fc.columns) == expected

    def test_output_row_count(self, trained_models):
        """horizon × n_models rows expected."""
        horizon = 5
        fc = forecast(trained_models, "Forest_A", horizon=horizon)
        assert len(fc) == horizon * len(MODEL_REGISTRY)

    def test_all_models_present(self, trained_models):
        """Every model in the registry must appear in the forecast output."""
        fc = forecast(trained_models, "Forest_A", horizon=3)
        assert set(fc["model"].unique()) == set(MODEL_REGISTRY.keys())

    def test_uncertainty_band_is_valid(self, trained_models):
        """lst_low <= lst_forecast <= lst_high for every row."""
        fc = forecast(trained_models, "Forest_A", horizon=7)
        assert (fc["lst_low"] <= fc["lst_forecast"]).all()
        assert (fc["lst_forecast"] <= fc["lst_high"]).all()

    def test_dates_are_sequential(self, trained_models):
        """Forecast dates must be strictly increasing within each model."""
        fc = forecast(trained_models, "Forest_A", horizon=6)
        for model_name in MODEL_REGISTRY:
            dates = pd.to_datetime(fc[fc["model"] == model_name]["date"])
            diffs = dates.diff().dropna()
            assert (diffs > pd.Timedelta(0)).all()

    def test_unknown_site_raises(self, trained_models):
        """forecast() must raise ValueError for an unrecognised site."""
        with pytest.raises(ValueError, match="Unknown site"):
            forecast(trained_models, "Mars_Colony_1", horizon=3)


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------

class TestMetrics:

    @pytest.fixture(scope="class")
    def trained_models(self):
        df = _make_site_monthly(sites=["Forest_A", "Urban_B"])
        return train_forecasters(df, train_end_year=2022)

    def test_metrics_schema(self, trained_models):
        """compute_metrics must return site, model, mae, rmse, r2 columns."""
        met = compute_metrics(trained_models)
        assert set(met.columns) == {"site", "model", "mae", "rmse", "r2"}

    def test_r2_is_finite(self, trained_models):
        """R² must be a finite number (not NaN or inf)."""
        met = compute_metrics(trained_models)
        assert met["r2"].apply(lambda x: math.isfinite(x)).all()

    def test_mae_rmse_positive(self, trained_models):
        """MAE and RMSE must be non-negative."""
        met = compute_metrics(trained_models)
        assert (met["mae"] >= 0).all()
        assert (met["rmse"] >= 0).all()

    def test_rmse_geq_mae(self, trained_models):
        """RMSE >= MAE always (by definition of L2 vs L1)."""
        met = compute_metrics(trained_models)
        assert (met["rmse"] >= met["mae"] - 1e-6).all()

    def test_all_models_in_metrics(self, trained_models):
        """Every model in registry must appear in metrics output."""
        met = compute_metrics(trained_models)
        assert set(met["model"].unique()) == set(MODEL_REGISTRY.keys())

    def test_metrics_computed_on_held_out_split(self):
        """
        R² must be computed on data *after* train_end_year, not on training data.
        A model that perfectly memorises training data should not score R²=1
        on the test split.
        """
        df = _make_site_monthly(start_year=2000, end_year=2024)
        models = train_forecasters(df, train_end_year=2020)
        met = compute_metrics(models)
        # R² on unseen data should be < 1.0 (not memorised)
        assert (met["r2"] < 1.0).all()


# ---------------------------------------------------------------------------
# 5. Site isolation
# ---------------------------------------------------------------------------

class TestSiteIsolation:

    def test_forecasting_one_site_does_not_change_another(self):
        """Calling forecast() for site A should not alter site B predictions."""
        df = _make_site_monthly(sites=["A", "B"])
        models = train_forecasters(df, train_end_year=2022)

        fc_b_before = forecast(models, "B", horizon=3)
        _ = forecast(models, "A", horizon=3)   # call A in between
        fc_b_after = forecast(models, "B", horizon=3)

        pd.testing.assert_frame_equal(fc_b_before, fc_b_after)


# ---------------------------------------------------------------------------
# 6. Integration — real processed data (skipped if unavailable)
# ---------------------------------------------------------------------------

_PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


@pytest.mark.skipif(
    not (_PROCESSED / "site_monthly.csv").exists(),
    reason="Real processed data not available",
)
class TestRealData:

    @pytest.fixture(scope="class")
    def real_models(self):
        df = pd.read_csv(_PROCESSED / "site_monthly.csv")
        return train_forecasters(df, train_end_year=TRAIN_END_YEAR)

    def test_all_8_sites_forecastable(self, real_models):
        """forecast() must succeed for every site in the real dataset."""
        df = pd.read_csv(_PROCESSED / "site_monthly.csv")
        for site in df["site"].unique():
            fc = forecast(real_models, site, horizon=7)
            assert len(fc) == 7 * len(MODEL_REGISTRY)

    def test_metrics_r2_reasonable(self, real_models):
        """On real data R² should be > 0 (models beat a naive mean predictor)."""
        met = compute_metrics(real_models)
        overall = met[met["site"] == "ALL"]
        assert (overall["r2"] > 0).all(), f"R² below 0:\n{overall}"