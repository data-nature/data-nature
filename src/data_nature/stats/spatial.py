"""
spatial.py — HW2 spatial-statistics layer for the LST / NDVI ecosystem model.

Adds the *spatial* analysis that the non-spatial models in ``regression.py``
do not cover:

- :func:`run_pca`            — Principal Component Analysis (Part C / D).
- :func:`morans_i`           — global Moran's I spatial-autocorrelation test (H1).
- :func:`spillover_regression` — neighbour-NDVI cooling spillover test (H2).
- :func:`pca_regression`     — OLS of LST on principal-component scores (Part D).
- :func:`kriging_cv`         — Gaussian-process (Kriging) spatial prediction (H3).

All functions operate on a tidy pixel-level DataFrame with at least the columns
``lst, ndvi, lat, lng, month, year, land_cover``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Numeric variables fed into the PCA (month encoded cyclically below).
NUMERIC_VARS = ["lst", "ndvi", "lat", "lng", "year", "month_sin", "month_cos"]


def add_cyclic_month(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the calendar month as two cyclic features (sin/cos)."""
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def run_pca(df: pd.DataFrame, cols: list[str] | None = None, n: int | None = None):
    """
    Standardise *cols* and run PCA.

    Returns
    -------
    scores : np.ndarray            (n_obs, n_components) component scores
    loadings : pd.DataFrame        variable × component loadings
    evr : pd.Series                explained-variance ratio per component
    model : sklearn PCA            the fitted estimator
    """
    cols = cols or NUMERIC_VARS
    X = StandardScaler().fit_transform(df[cols].values)
    model = PCA(n_components=n or len(cols)).fit(X)
    names = [f"PC{i + 1}" for i in range(model.n_components_)]
    scores = model.transform(X)
    loadings = pd.DataFrame(model.components_.T, index=cols, columns=names)
    evr = pd.Series(model.explained_variance_ratio_, index=names)
    return scores, loadings, evr, model


def _knn_weights(coords: np.ndarray, k: int = 8) -> np.ndarray:
    """Row-standardised k-nearest-neighbour spatial weights matrix."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    idx = idx[:, 1:]  # drop self
    n = len(coords)
    W = np.zeros((n, n))
    for i, nbrs in enumerate(idx):
        W[i, nbrs] = 1.0 / k
    return W


def morans_i(values, coords, k: int = 8, n_perm: int = 999, seed: int = 0) -> dict:
    """
    Global Moran's I with a permutation p-value (Hypothesis 1).

    Positive, significant I  →  LST is spatially clustered (reject H0).
    """
    x = np.asarray(values, float)
    W = _knn_weights(coords, k)
    z = x - x.mean()
    S0 = W.sum()
    I = (len(x) / S0) * (z @ (W @ z)) / (z @ z)
    rng = np.random.default_rng(seed)
    perm = np.empty(n_perm)
    for p in range(n_perm):
        zp = rng.permutation(z)
        perm[p] = (len(x) / S0) * (zp @ (W @ zp)) / (zp @ zp)
    p_val = (np.sum(np.abs(perm) >= abs(I)) + 1) / (n_perm + 1)
    EI = -1.0 / (len(x) - 1)
    zscore = (I - EI) / perm.std()
    return {
        "I": round(float(I), 4),
        "EI": round(float(EI), 4),
        "z": round(float(zscore), 3),
        "p": round(float(p_val), 4),
        "k": k,
        "n": int(len(x)),
    }


def spillover_regression(df: pd.DataFrame, coords: np.ndarray, k: int = 8):
    """
    H2 — does neighbouring NDVI cool a cell beyond its own NDVI?

    Fits ``lst ~ ndvi + ndvi_neighbors`` where ``ndvi_neighbors`` is the
    spatial (k-NN) lag of NDVI.  A significant negative ``ndvi_neighbors``
    coefficient is evidence of a spatial cooling spillover.
    """
    W = _knn_weights(coords, k)
    d = df.copy()
    d["ndvi_neighbors"] = W @ d["ndvi"].values
    return smf.ols("lst ~ ndvi + ndvi_neighbors", data=d).fit()


def pca_regression(df: pd.DataFrame, scores: np.ndarray, k_pcs: int = 2):
    """OLS of LST on the first *k_pcs* principal-component scores."""
    d = df.copy()
    for i in range(k_pcs):
        d[f"PC{i + 1}"] = scores[:, i]
    formula = "lst ~ " + " + ".join(f"PC{i + 1}" for i in range(k_pcs))
    return smf.ols(formula, data=d).fit()


def kriging_cv(df: pd.DataFrame, coords: np.ndarray, n_splits: int = 5, seed: int = 0) -> dict:
    """
    H3 — Kriging (Gaussian-process regression) spatial prediction.

    K-fold cross-validates an RBF + white-noise Gaussian process and compares
    its RMSE against a global-mean baseline.  Kriging RMSE well below the mean
    RMSE indicates exploitable spatial structure (reject H0).
    """
    X, y = coords, df["lst"].values
    kernel = ConstantKernel(1.0) * RBF(length_scale=0.1) + WhiteKernel(noise_level=1.0)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    krig_err, mean_err = [], []
    gp = None
    for tr, te in kf.split(X):
        gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=seed
        )
        gp.fit(X[tr], y[tr])
        pred = gp.predict(X[te])
        krig_err.append(np.sqrt(np.mean((pred - y[te]) ** 2)))
        mean_err.append(np.sqrt(np.mean((y[tr].mean() - y[te]) ** 2)))
    return {
        "kriging_rmse": round(float(np.mean(krig_err)), 3),
        "mean_rmse": round(float(np.mean(mean_err)), 3),
        "kernel": str(gp.kernel_),
    }
