"""
lstm.py — PyTorch LSTM time-series forecast model (DN-B3)
==========================================================
Trains a single-layer LSTM to forecast LST 7 months ahead, using the same
feature set and train/test split as the RF/GB models in forecast.py so that
all three models are directly comparable.

Public API
----------
    LSTMForecaster          — PyTorch Module (train / predict)
    train_lstm              — fit an LSTMForecaster on site_monthly data
    forecast_lstm           — recursive multi-step forecast for one site
    compute_metrics_lstm    — MAE / RMSE / R² on held-out test split
    run_lstm_pipeline       — end-to-end: load → train → forecast → append CSVs

Output schemas (frozen data contract)
--------------------------------------
    lst_forecast.csv  : date, site, model, lst_forecast, lst_low, lst_high
    model_metrics.csv : site, model, mae, rmse, r2
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (mirror forecast.py)
# ---------------------------------------------------------------------------
_REPO_ROOT    = Path(__file__).resolve().parents[3]
PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
MOCK_DIR      = _REPO_ROOT / "data" / "mock"

FORECAST_PATH = PROCESSED_DIR / "lst_forecast.csv"
METRICS_PATH  = PROCESSED_DIR / "model_metrics.csv"

# ---------------------------------------------------------------------------
# Constants  (match forecast.py so splits are identical)
# ---------------------------------------------------------------------------
TRAIN_END_YEAR   = 2023
FORECAST_HORIZON = 7
RANDOM_STATE     = 42
MODEL_NAME       = "LSTM"

# LSTM hyper-parameters
SEQ_LEN    = 12   # look-back window in months (one full year)
HIDDEN_DIM = 64
NUM_LAYERS = 1
DROPOUT    = 0.1
EPOCHS     = 50
LR         = 1e-3
BATCH_SIZE = 32

# Feature columns — identical to forecast.py
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
# Feature engineering (mirrors forecast.py _build_features)
# ---------------------------------------------------------------------------

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag/rolling features and target column, identical to forecast.py."""
    df = df.copy().sort_values(["site", "year", "month"]).reset_index(drop=True)

    grp = df.groupby("site")["lst"]
    df["lst_lag1"]       = grp.shift(1)
    df["lst_lag12"]      = grp.shift(12)
    df["lst_roll3_mean"] = grp.transform(
        lambda s: s.shift(1).rolling(3, min_periods=2).mean()
    )
    df["lst_roll3_std"]  = grp.transform(
        lambda s: s.shift(1).rolling(3, min_periods=2).std()
    )
    df["lst_lead1"] = grp.shift(-1)

    le = LabelEncoder()
    df["site_encoded"] = le.fit_transform(df["site"])

    return df


# ---------------------------------------------------------------------------
# PyTorch model
# ---------------------------------------------------------------------------

class LSTMForecaster(nn.Module):
    """Single-layer LSTM regressor for monthly LST forecasting.

    Input shape  : (batch, seq_len, n_features)
    Output shape : (batch, 1)  — point forecast of next-month LST
    """

    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        dropout:    float = DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, T, F)
        out, _ = self.lstm(x)                                     # (B, T, H)
        last    = out[:, -1, :]                                   # (B, H)
        return self.fc(self.dropout(last)).squeeze(-1)            # (B,)


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------

def _make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window of length seq_len over X/y arrays."""
    xs, ys = [], []
    for i in range(seq_len, len(X)):
        xs.append(X[i - seq_len : i])
        ys.append(y[i])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lstm(
    df: pd.DataFrame,
    train_end_year: int = TRAIN_END_YEAR,
    seq_len:        int = SEQ_LEN,
    hidden_dim:     int = HIDDEN_DIM,
    num_layers:     int = NUM_LAYERS,
    dropout:       float = DROPOUT,
    epochs:         int = EPOCHS,
    lr:            float = LR,
    batch_size:     int = BATCH_SIZE,
    random_state:   int = RANDOM_STATE,
) -> dict:
    """Train an LSTMForecaster on site_monthly data.

    Parameters
    ----------
    df : pd.DataFrame
        site_monthly DataFrame (year, month, site, lst, ndvi, …).
    train_end_year : int
        Last year (inclusive) in the training split (mirrors forecast.py).

    Returns
    -------
    dict with keys:
        model          — trained LSTMForecaster (eval mode, CPU)
        scaler_X       — fitted MinMaxScaler for features
        scaler_y       — fitted MinMaxScaler for target
        label_encoder  — fitted LabelEncoder for site names
        df_feat        — full featured DataFrame (for forecast seeding)
        X_test_raw     — unscaled test features  (for metrics)
        y_test_raw     — unscaled test targets    (for metrics)
        test_sites     — site Series aligned with test rows
        seq_len        — sequence length used during training
        feature_cols   — list of feature column names
    """
    required = {"year", "month", "site", "lst", "ndvi"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"train_lstm: missing columns {missing}")

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    df_feat = _build_features(df)

    # Keep only rows with complete features + target
    valid_mask = df_feat[FEATURE_COLS + [TARGET_COL]].notna().all(axis=1)
    df_feat    = df_feat[valid_mask].reset_index(drop=True)

    train_mask = df_feat["year"] <= train_end_year
    test_mask  = df_feat["year"] >  train_end_year

    X_all = df_feat[FEATURE_COLS].values.astype(np.float32)
    y_all = df_feat[TARGET_COL].values.astype(np.float32).reshape(-1, 1)

    # Scale features and target to [0, 1] for stable LSTM training
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_all)
    y_scaled = scaler_y.fit_transform(y_all).ravel()

    # Build sequences from training data only
    X_train_seq, y_train_seq = _make_sequences(
        X_scaled[train_mask.values], y_scaled[train_mask.values], seq_len
    )

    # Build sequences from test data for metric computation
    # We use full-dataset sequences but evaluate only on test indices
    X_all_seq, y_all_seq = _make_sequences(X_scaled, y_scaled, seq_len)
    # The i-th sequence predicts row (seq_len + i) in df_feat
    test_indices = np.where(test_mask.values[seq_len:])[0]

    X_test_seq  = X_all_seq[test_indices]
    y_test_seq  = y_all_seq[test_indices]
    test_sites  = df_feat["site"].iloc[seq_len:].reset_index(drop=True).iloc[test_indices]

    # ── DataLoader ────────────────────────────────────────────────────────
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train_seq),
        torch.from_numpy(y_train_seq),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # ── Model, optimiser, loss ────────────────────────────────────────────
    model     = LSTMForecaster(input_dim=len(FEATURE_COLS), hidden_dim=hidden_dim,
                               num_layers=num_layers, dropout=dropout)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # ── Training loop ─────────────────────────────────────────────────────
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimiser.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            epoch_loss += loss.item() * len(X_batch)

        if epoch % 10 == 0 or epoch == 1:
            logger.info("Epoch %3d/%d  loss=%.5f", epoch, epochs,
                        epoch_loss / len(dataset))

    model.eval()

    # Fit a fresh LabelEncoder on all sites for forecast()
    le = LabelEncoder()
    le.fit(df["site"].unique())

    # Inverse-transform test targets (safe even when test split is empty)
    if len(y_test_seq) > 0:
        y_test_raw = scaler_y.inverse_transform(
            y_test_seq.reshape(-1, 1)
        ).ravel()
    else:
        y_test_raw = np.array([], dtype=np.float32)

    return {
        "model":         model,
        "scaler_X":      scaler_X,
        "scaler_y":      scaler_y,
        "label_encoder": le,
        "df_feat":       df_feat,
        "X_test_seq":    X_test_seq,
        "y_test_seq":    y_test_seq,
        "y_test_raw":    y_test_raw,
        "test_sites":    test_sites.reset_index(drop=True),
        "seq_len":       seq_len,
        "feature_cols":  FEATURE_COLS,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics_lstm(artifacts: dict) -> pd.DataFrame:
    """Compute MAE, RMSE, R² per site + ALL on the held-out test split.

    Parameters
    ----------
    artifacts : dict
        Output of :func:`train_lstm`.

    Returns
    -------
    pd.DataFrame
        Columns: site, model, mae, rmse, r2
    """
    model      = artifacts["model"]
    scaler_y   = artifacts["scaler_y"]
    X_test_seq = artifacts["X_test_seq"]
    y_true_raw = artifacts["y_test_raw"]
    test_sites = artifacts["test_sites"]

    with torch.no_grad():
        y_pred_scaled = model(torch.from_numpy(X_test_seq)).numpy()
    y_pred_raw = scaler_y.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).ravel()

    def _row(site_label, yt, yp):
        return {
            "site":  site_label,
            "model": MODEL_NAME,
            "mae":   round(float(mean_absolute_error(yt, yp)),  4),
            "rmse":  round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
            "r2":    round(float(r2_score(yt, yp)),             4),
        }

    records = [_row("ALL", y_true_raw, y_pred_raw)]
    for site, idx in test_sites.groupby(test_sites).groups.items():
        records.append(_row(site, y_true_raw[idx], y_pred_raw[idx]))

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

def forecast_lstm(
    artifacts: dict,
    site:      str,
    horizon:   int = FORECAST_HORIZON,
) -> pd.DataFrame:
    """Recursive multi-step LSTM forecast for one site.

    Parameters
    ----------
    artifacts : dict
        Output of :func:`train_lstm`.
    site : str
        Site name (must exist in the training data).
    horizon : int
        Number of months ahead to forecast.

    Returns
    -------
    pd.DataFrame
        Columns: date, site, model, lst_forecast, lst_low, lst_high
    """
    model    = artifacts["model"]
    scaler_X = artifacts["scaler_X"]
    scaler_y = artifacts["scaler_y"]
    le       = artifacts["label_encoder"]
    df_feat  = artifacts["df_feat"]
    seq_len  = artifacts["seq_len"]

    if site not in le.classes_:
        raise ValueError(f"Unknown site '{site}'. Known: {list(le.classes_)}")

    site_data = df_feat[df_feat["site"] == site].sort_values(["year", "month"])
    if len(site_data) < seq_len:
        raise ValueError(
            f"Site '{site}' has only {len(site_data)} rows; need ≥{seq_len}."
        )

    # Seed sequence: last seq_len observed feature rows
    seed_X = site_data[FEATURE_COLS].values[-seq_len:].astype(np.float32)
    seed_X_scaled = scaler_X.transform(seed_X)             # (seq_len, F)

    last_row   = site_data.iloc[-1]
    start_date = pd.Timestamp(
        year=int(last_row["year"]), month=int(last_row["month"]), day=1
    ) + pd.DateOffset(months=1)

    records   = []
    window    = seed_X_scaled.copy()                        # rolling (seq_len, F)

    # Column indices for in-loop state updates
    col_idx = {c: i for i, c in enumerate(FEATURE_COLS)}
    ndvi_val       = float(last_row["ndvi"])
    site_enc_val   = float(le.transform([site])[0])

    # Keep an unscaled history of recent LST predictions for lag computation
    lst_history = list(site_data["lst"].values[-12:])       # last 12 observed

    with torch.no_grad():
        for step in range(horizon):
            x_tensor = torch.from_numpy(window).unsqueeze(0)  # (1, T, F)
            pred_scaled = model(x_tensor).item()
            pred_lst    = float(
                scaler_y.inverse_transform([[pred_scaled]])[0][0]
            )

            # Uncertainty band: std of predictions over small input perturbations
            # (Monte-Carlo dropout approximation — we use the model in eval mode
            #  so dropout is off; instead we derive the band from recent volatility)
            roll_std_col = col_idx["lst_roll3_std"]
            recent_std   = float(
                scaler_X.data_range_[roll_std_col] *
                window[-1, roll_std_col] + scaler_X.data_min_[roll_std_col]
            )
            band = max(recent_std * 0.8, 0.3)

            current_date = start_date + pd.DateOffset(months=step)
            records.append({
                "date":         current_date.strftime("%Y-%m-%d"),
                "site":         site,
                "model":        MODEL_NAME,
                "lst_forecast": round(pred_lst, 4),
                "lst_low":      round(pred_lst - band, 4),
                "lst_high":     round(pred_lst + band, 4),
            })

            # ── Roll the window forward ──────────────────────────────────
            lst_history.append(pred_lst)

            lag1  = lst_history[-2]  if len(lst_history) >= 2  else pred_lst
            lag12 = lst_history[-13] if len(lst_history) >= 13 else pred_lst
            roll3 = float(np.mean(lst_history[-3:])) if len(lst_history) >= 3 else pred_lst
            roll3_std = float(np.std(lst_history[-3:], ddof=1)) if len(lst_history) >= 3 else 0.5

            new_row_raw = np.array([[
                float(current_date.month),
                ndvi_val,
                lag1,
                lag12,
                roll3,
                roll3_std,
                site_enc_val,
            ]], dtype=np.float32)

            new_row_scaled = scaler_X.transform(new_row_raw)  # (1, F)
            window = np.vstack([window[1:], new_row_scaled])   # slide forward

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Full pipeline — load, train, append to existing forecast/metrics CSVs
# ---------------------------------------------------------------------------

def _load_site_monthly(data_dir: Path | None = None) -> pd.DataFrame:
    """Load site_monthly.csv with mock fallback."""
    data_dir = Path(data_dir) if data_dir else PROCESSED_DIR
    path = data_dir / "site_monthly.csv"
    if path.exists():
        return pd.read_csv(path)
    mock = MOCK_DIR / "site_monthly.csv"
    if mock.exists():
        logger.warning("Falling back to mock site_monthly.")
        return pd.read_csv(mock)
    raise FileNotFoundError(f"site_monthly.csv not found in {data_dir} or mock.")


def run_lstm_pipeline(
    data_dir:   Path | None = None,
    output_dir: Path | None = None,
    horizon:    int = FORECAST_HORIZON,
    **train_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end: load → train LSTM → forecast all sites → append CSVs.

    Appends LSTM rows to existing lst_forecast.csv and model_metrics.csv
    (created by forecast.py's run_full_pipeline) so all three models live
    in the same files.

    Returns
    -------
    (forecast_df, metrics_df)  — LSTM-only rows
    """
    output_dir = Path(output_dir) if output_dir else PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df        = _load_site_monthly(data_dir)
    artifacts = train_lstm(df, **train_kwargs)
    sites     = df["site"].unique()

    # ── Forecasts ──────────────────────────────────────────────────────────
    lstm_forecasts = pd.concat(
        [forecast_lstm(artifacts, site, horizon=horizon) for site in sites],
        ignore_index=True,
    )

    forecast_path = output_dir / "lst_forecast.csv"
    if forecast_path.exists():
        existing = pd.read_csv(forecast_path)
        existing  = existing[existing["model"] != MODEL_NAME]   # drop old LSTM rows
        combined  = pd.concat([existing, lstm_forecasts], ignore_index=True)
    else:
        combined = lstm_forecasts
    combined.to_csv(forecast_path, index=False)
    logger.info("Saved %d LSTM forecast rows → %s", len(lstm_forecasts), forecast_path)

    # ── Metrics ────────────────────────────────────────────────────────────
    lstm_metrics = compute_metrics_lstm(artifacts)

    metrics_path = output_dir / "model_metrics.csv"
    if metrics_path.exists():
        existing_m = pd.read_csv(metrics_path)
        existing_m = existing_m[existing_m["model"] != MODEL_NAME]
        combined_m = pd.concat([existing_m, lstm_metrics], ignore_index=True)
    else:
        combined_m = lstm_metrics
    combined_m.to_csv(metrics_path, index=False)
    logger.info("Saved %d LSTM metric rows → %s", len(lstm_metrics), metrics_path)

    return lstm_forecasts, lstm_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("── Running DN-B3 LSTM pipeline ────────────────────")
    fc, met = run_lstm_pipeline()
    print("\nForecast sample (first 6 rows):")
    print(fc.head(6).to_string(index=False))
    print("\nMetrics (ALL sites):")
    print(met[met["site"] == "ALL"].to_string(index=False))
    print(f"\nSaved to {PROCESSED_DIR}")
    sys.exit(0)