"""
anomaly.py — Per-site, per-month LST z-score and anomaly detection.

Baseline strategy: historical mean and std of LST for each (site, month)
pair across all years present in the input DataFrame.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_THRESHOLDS: dict[str, float] = {
    "Warning": 1.5,
    "Severe": 2.5,
    "Critical": 3.5,
}


# ── helpers ───────────────────────────────────────────────────────────────────


def _classify_severity(
    z: float, thresholds: dict[str, float]
) -> str | None:
    if z >= thresholds["Critical"]:
        return "Critical"
    if z >= thresholds["Severe"]:
        return "Severe"
    if z >= thresholds["Warning"]:
        return "Warning"
    return None


# ── public API ────────────────────────────────────────────────────────────────


def compute_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-site, per-month historical baseline and z-score columns to *df*.

    For each (site, month) group the baseline is the mean and standard
    deviation of LST across all years present in the input.  The z-score
    measures how many standard deviations a given observation sits above
    (positive) or below (negative) that baseline.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``site``, ``month``, ``lst``.

    Returns
    -------
    pd.DataFrame
        Original df with three added columns:

        - ``baseline``     — historical mean LST for that (site, month)
        - ``baseline_std`` — historical std LST for that (site, month)
        - ``z_score``      — ``(lst − baseline) / baseline_std``

        Rows where ``baseline_std`` is zero or NaN yield ``z_score = NaN``.
    """
    stats = (
        df.groupby(["site", "month"])["lst"]
        .agg(baseline="mean", baseline_std="std")
        .reset_index()
    )

    result = df.copy().merge(stats, on=["site", "month"], how="left")

    safe_std = result["baseline_std"].replace(0, float("nan"))
    result["z_score"] = (
        (result["lst"] - result["baseline"]) / safe_std
    ).round(4)

    return result


def detect_anomalies(
    df: pd.DataFrame,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Detect LST anomalies by z-score threshold and return an anomaly table.

    If ``z_score`` or ``baseline`` columns are absent from *df*,
    :func:`compute_zscores` is called automatically.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``date``, ``site``, ``lst``, ``ndvi``, ``month``.
    thresholds : dict, optional
        Severity cut-offs keyed by ``"Warning"``, ``"Severe"``, ``"Critical"``.
        Defaults to ``DEFAULT_THRESHOLDS`` = {Warning: 1.5, Severe: 2.5,
        Critical: 3.5}.

    Returns
    -------
    pd.DataFrame
        Anomalous rows only, with columns matching the ``anomalies.csv`` schema:
        ``date``, ``site``, ``lst``, ``baseline``, ``z_score``, ``severity``,
        ``status``, ``ndvi_change``.

        ``status`` is always ``"New"`` — the dashboard layer tracks transitions.
        ``ndvi_change`` is the signed NDVI difference from the previous month
        for the same site (NaN for the first observation of each site).
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS.copy()

    if "z_score" not in df.columns or "baseline" not in df.columns:
        df = compute_zscores(df)

    enriched = df.copy().sort_values(["site", "date"]).reset_index(drop=True)
    enriched["severity"] = enriched["z_score"].apply(
        lambda z: _classify_severity(z, thresholds) if pd.notna(z) else None
    )
    enriched["ndvi_change"] = enriched.groupby("site")["ndvi"].diff().round(4)
    enriched["status"] = "New"

    anomalies = enriched.loc[enriched["severity"].notna()].copy()

    cols = ["date", "site", "lst", "baseline", "z_score", "severity", "status", "ndvi_change"]
    return anomalies[cols].reset_index(drop=True)
