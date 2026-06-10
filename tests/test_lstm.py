"""
tests/test_lstm.py — DN-B3 smoke & unit tests for the LSTM forecaster
======================================================================
All tests run on synthetic data so no data files are required.
CI must pass without any files under data/processed/.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from data_nature.models.lstm import (
    FEATURE_COLS,
    FORECAST_HORIZON,
    MODEL_NAME,
    LSTMForecaster,
    _build_features,
    _make_sequences,
    compute_metrics_lstm,
    forecast_lstm,
    train_lstm,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _synthetic_df(n_years: int = 26, seed: int = 42) -> pd.DataFrame:
    """Generate a minimal synthetic site_monthly DataFrame for testing.

    Two sites × n_years × 12 months = 2 × n_years × 12 rows.
    LST follows a simple sinusoidal seasonal pattern + small noise.
    NDVI is the inverse.
    """
    rng    = np.random.default_rng(seed)
    sites  = ["Site_A", "Site_B"]
    rows   = []
    for site in sites:
        for year in range(2000, 2000 + n_years):
            for month in range(1, 13):
                t    = (month - 1) / 12 * 2 * np.pi
                lst  = 25 + 10 * np.sin(t) + rng.normal(0, 0.5)
                ndvi = 0.5 - 0.2 * np.sin(t) + rng.normal(0, 0.02)
                rows.append({"year": year, "month": month, "site": site,
                             "lst": round(lst, 4), "ndvi": round(ndvi, 4)})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic_df():
    return _synthetic_df()


@pytest.fixture(scope="module")
def trained_artifacts(synthetic_df):
    """Train the LSTM for 2 epochs — fast enough for CI."""
    return train_lstm(synthetic_df, epochs=2, random_state=42)


# ---------------------------------------------------------------------------
# Unit tests — LSTMForecaster module
# ---------------------------------------------------------------------------

class TestLSTMForecasterModule:
    def test_forward_output_shape(self):
        """Forward pass returns (batch_size,) tensor."""
        model = LSTMForecaster(input_dim=len(FEATURE_COLS))
        x     = torch.randn(8, 12, len(FEATURE_COLS))   # (B=8, T=12, F)
        out   = model(x)
        assert out.shape == (8,), f"Expected (8,), got {out.shape}"

    def test_forward_single_sample(self):
        """Works with batch size of 1."""
        model = LSTMForecaster(input_dim=len(FEATURE_COLS))
        x     = torch.randn(1, 12, len(FEATURE_COLS))
        out   = model(x)
        assert out.shape == (1,)

    def test_output_is_finite(self):
        """Output contains no NaN or Inf."""
        model = LSTMForecaster(input_dim=len(FEATURE_COLS))
        x     = torch.randn(4, 12, len(FEATURE_COLS))
        out   = model(x)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"

    def test_deterministic_with_seed(self):
        """Same seed → same output (reproducibility check)."""
        torch.manual_seed(0)
        model = LSTMForecaster(input_dim=len(FEATURE_COLS))
        x     = torch.ones(2, 12, len(FEATURE_COLS))
        out1  = model(x).detach().clone()

        torch.manual_seed(0)
        model2 = LSTMForecaster(input_dim=len(FEATURE_COLS))
        out2   = model2(x).detach().clone()

        assert torch.allclose(out1, out2), "Different seeds give different outputs"


# ---------------------------------------------------------------------------
# Unit tests — helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_build_features_columns(self, synthetic_df):
        """_build_features adds all required columns."""
        df_feat = _build_features(synthetic_df)
        for col in FEATURE_COLS + ["lst_lead1"]:
            assert col in df_feat.columns, f"Missing column: {col}"

    def test_build_features_no_cross_site_bleed(self, synthetic_df):
        """Lag features must not bleed across site boundaries."""
        df_feat = _build_features(synthetic_df)
        for site in df_feat["site"].unique():
            site_rows = df_feat[df_feat["site"] == site].sort_values(["year", "month"])
            # The very first row of each site must have NaN lag1 (no prior month)
            first_lag1 = site_rows["lst_lag1"].iloc[0]
            assert pd.isna(first_lag1), (
                f"Site '{site}' first row lst_lag1 should be NaN, got {first_lag1}"
            )

    def test_make_sequences_shape(self):
        """_make_sequences returns correct (N-seq_len, seq_len, F) shape."""
        X = np.random.randn(50, len(FEATURE_COLS)).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)
        xs, ys = _make_sequences(X, y, seq_len=12)
        assert xs.shape == (38, 12, len(FEATURE_COLS)), f"Got {xs.shape}"
        assert ys.shape == (38,), f"Got {ys.shape}"

    def test_make_sequences_alignment(self):
        """Each sequence xs[i] covers X[i : i+seq_len] and target is y[i+seq_len]."""
        X = np.arange(20, dtype=np.float32).reshape(20, 1)
        y = np.arange(20, dtype=np.float32)
        xs, ys = _make_sequences(X, y, seq_len=5)
        # First sequence should be rows 0..4, target row 5
        np.testing.assert_array_equal(xs[0, :, 0], np.arange(5, dtype=np.float32))
        assert ys[0] == 5.0


# ---------------------------------------------------------------------------
# Integration tests — train_lstm
# ---------------------------------------------------------------------------

class TestTrainLSTM:
    def test_returns_expected_keys(self, trained_artifacts):
        """train_lstm output dict has all required keys."""
        required = {"model", "scaler_X", "scaler_y", "label_encoder",
                    "df_feat", "X_test_seq", "y_test_seq", "y_test_raw",
                    "test_sites", "seq_len", "feature_cols"}
        assert required <= set(trained_artifacts.keys())

    def test_model_in_eval_mode(self, trained_artifacts):
        """Returned model must be in eval mode."""
        assert not trained_artifacts["model"].training, \
            "Model should be in eval mode after training"

    def test_model_is_cpu(self, trained_artifacts):
        """Model parameters must be on CPU (sprint spec: CPU-only)."""
        for p in trained_artifacts["model"].parameters():
            assert p.device.type == "cpu"

    def test_missing_columns_raise(self):
        """train_lstm raises ValueError when required columns are absent."""
        bad_df = pd.DataFrame({"year": [2000], "month": [1], "site": ["X"]})
        with pytest.raises(ValueError, match="missing columns"):
            train_lstm(bad_df, epochs=1)


# ---------------------------------------------------------------------------
# Integration tests — forecast_lstm
# ---------------------------------------------------------------------------

class TestForecastLSTM:
    def test_output_shape(self, trained_artifacts):
        """forecast_lstm returns FORECAST_HORIZON rows per site."""
        df_out = forecast_lstm(trained_artifacts, site="Site_A",
                               horizon=FORECAST_HORIZON)
        assert len(df_out) == FORECAST_HORIZON, \
            f"Expected {FORECAST_HORIZON} rows, got {len(df_out)}"

    def test_output_columns(self, trained_artifacts):
        """Output has exactly the frozen data-contract columns."""
        df_out = forecast_lstm(trained_artifacts, site="Site_A")
        expected = {"date", "site", "model", "lst_forecast", "lst_low", "lst_high"}
        assert set(df_out.columns) == expected

    def test_model_name_tag(self, trained_artifacts):
        """All rows carry the correct model name tag."""
        df_out = forecast_lstm(trained_artifacts, site="Site_A")
        assert (df_out["model"] == MODEL_NAME).all()

    def test_uncertainty_band_ordering(self, trained_artifacts):
        """lst_low ≤ lst_forecast ≤ lst_high for every row."""
        df_out = forecast_lstm(trained_artifacts, site="Site_A")
        assert (df_out["lst_low"] <= df_out["lst_forecast"]).all()
        assert (df_out["lst_forecast"] <= df_out["lst_high"]).all()

    def test_forecast_values_finite(self, trained_artifacts):
        """No NaN or Inf in forecast output."""
        df_out = forecast_lstm(trained_artifacts, site="Site_B")
        for col in ["lst_forecast", "lst_low", "lst_high"]:
            assert df_out[col].notna().all(), f"NaN found in column {col}"
            assert np.isfinite(df_out[col].values).all(), f"Inf found in column {col}"

    def test_dates_are_sequential(self, trained_artifacts):
        """Forecast dates are monthly and strictly increasing."""
        df_out = forecast_lstm(trained_artifacts, site="Site_A")
        dates  = pd.to_datetime(df_out["date"])
        deltas = dates.diff().dropna()
        assert (deltas > pd.Timedelta(0)).all(), "Dates are not strictly increasing"

    def test_unknown_site_raises(self, trained_artifacts):
        """forecast_lstm raises ValueError for an unknown site name."""
        with pytest.raises(ValueError, match="Unknown site"):
            forecast_lstm(trained_artifacts, site="NoSuchSite")


# ---------------------------------------------------------------------------
# Integration tests — compute_metrics_lstm
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_metrics_columns(self, trained_artifacts):
        """Metrics DataFrame has the frozen data-contract columns."""
        metrics = compute_metrics_lstm(trained_artifacts)
        assert set(metrics.columns) == {"site", "model", "mae", "rmse", "r2"}

    def test_all_row_present(self, trained_artifacts):
        """An 'ALL' aggregate row is always included."""
        metrics = compute_metrics_lstm(trained_artifacts)
        assert "ALL" in metrics["site"].values

    def test_model_name_in_metrics(self, trained_artifacts):
        """All metric rows carry the correct model name."""
        metrics = compute_metrics_lstm(trained_artifacts)
        assert (metrics["model"] == MODEL_NAME).all()

    def test_metrics_are_finite(self, trained_artifacts):
        """MAE, RMSE, R² are finite numbers (model is not degenerate)."""
        metrics = compute_metrics_lstm(trained_artifacts)
        for col in ["mae", "rmse", "r2"]:
            assert np.isfinite(metrics[col].values).all(), \
                f"Non-finite value in {col}"

    def test_mae_rmse_positive(self, trained_artifacts):
        """MAE and RMSE must be non-negative."""
        metrics = compute_metrics_lstm(trained_artifacts)
        assert (metrics["mae"]  >= 0).all()
        assert (metrics["rmse"] >= 0).all()