"""
regression.py — Linear regression and ANOVA for LST / NDVI relationships.

Answers the sprint's core research question:
  H0: mean LST is equal across all land-cover types.
  H1: at least one land-cover type has a significantly different mean LST.
"""

from __future__ import annotations

import logging
from itertools import combinations

import pandas as pd
import scipy.stats
import statsmodels.formula.api as smf

log = logging.getLogger(__name__)


# ── public API ────────────────────────────────────────────────────────────────


def fit_ndvi_lst(df: pd.DataFrame) -> dict:
    """
    Fit simple and multilinear OLS regression of LST on NDVI.

    Simple model : ``LST ~ NDVI``
    Multi model  : ``LST ~ NDVI + land_cover (dummy) + month (numeric)``

    The expected sign of the NDVI coefficient is **negative** — higher
    vegetation density is associated with lower land-surface temperature.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``lst``, ``ndvi``, ``land_cover``, ``month``.
        Rows with NaN in ``lst`` or ``ndvi`` are dropped before fitting.

    Returns
    -------
    dict
        ``"simple"`` → dict with keys:
            ndvi_coef, intercept, r2, p_value, n

        ``"multi"``  → dict with keys:
            ndvi_coef, intercept, r2, adj_r2, p_values (dict), n
    """
    clean = df.dropna(subset=["lst", "ndvi"]).copy()
    if len(clean) < 3:
        raise ValueError(
            f"Need at least 3 non-null (lst, ndvi) rows to fit; got {len(clean)}."
        )

    # ── simple linear regression ──────────────────────────────────────────────
    slope, intercept, r, p, _ = scipy.stats.linregress(
        clean["ndvi"].astype(float), clean["lst"].astype(float)
    )

    simple: dict = {
        "ndvi_coef": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "r2": round(float(r**2), 4),
        "p_value": round(float(p), 6),
        "n": len(clean),
    }

    # ── multilinear regression ────────────────────────────────────────────────
    model = smf.ols("lst ~ ndvi + C(land_cover) + month", data=clean).fit()

    multi: dict = {
        "ndvi_coef": round(float(model.params.get("ndvi", float("nan"))), 4),
        "intercept": round(float(model.params.get("Intercept", float("nan"))), 4),
        "r2": round(float(model.rsquared), 4),
        "adj_r2": round(float(model.rsquared_adj), 4),
        "p_values": {k: round(float(v), 6) for k, v in model.pvalues.items()},
        "n": int(model.nobs),
    }

    log.info(
        "fit_ndvi_lst: simple R²=%.3f (p=%.4f), multi R²=%.3f, ndvi_coef=%.4f",
        simple["r2"],
        simple["p_value"],
        multi["r2"],
        multi["ndvi_coef"],
    )

    return {"simple": simple, "multi": multi}


def compare_site_types(df: pd.DataFrame) -> dict:
    """
    One-way ANOVA and pairwise Welch t-tests comparing LST across land-cover types.

    Tests research question:
        H0 — mean LST is equal across all land-cover types.
        H1 — at least one land-cover type has a different mean LST.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``lst``, ``land_cover``.
        Groups with fewer than 2 observations are silently dropped.

    Returns
    -------
    dict with keys:

    ``"anova_table"``
        Single-row pd.DataFrame with columns:
        ``F``, ``p_value``, ``df_between``, ``df_within``.

    ``"pairwise"``
        pd.DataFrame with one row per pair, columns:
        ``group_a``, ``group_b``, ``t_stat``, ``p_value``, ``significant``.

    ``"rejected_h0"``
        bool — True if ANOVA p-value < 0.05.

    Raises
    ------
    ValueError
        If fewer than two groups have sufficient data.
    """
    clean = df.dropna(subset=["lst", "land_cover"]).copy()
    groups: dict[str, list[float]] = {
        lc: grp["lst"].tolist()
        for lc, grp in clean.groupby("land_cover")
        if len(grp) >= 2
    }

    if len(groups) < 2:
        raise ValueError(
            f"Need at least two land-cover groups with ≥ 2 observations; "
            f"found {len(groups)}: {list(groups.keys())}"
        )

    # ── one-way ANOVA ─────────────────────────────────────────────────────────
    f_stat, p_anova = scipy.stats.f_oneway(*groups.values())
    n_groups = len(groups)
    n_total = sum(len(v) for v in groups.values())

    anova_table = pd.DataFrame(
        [{
            "F": round(float(f_stat), 4),
            "p_value": round(float(p_anova), 6),
            "df_between": n_groups - 1,
            "df_within": n_total - n_groups,
        }]
    )

    # ── pairwise Welch t-tests ────────────────────────────────────────────────
    rows = []
    for a, b in combinations(sorted(groups.keys()), 2):
        t_stat, p_val = scipy.stats.ttest_ind(
            groups[a], groups[b], equal_var=False
        )
        rows.append(
            {
                "group_a": a,
                "group_b": b,
                "t_stat": round(float(t_stat), 4),
                "p_value": round(float(p_val), 6),
                "significant": bool(p_val < 0.05),
            }
        )

    log.info(
        "compare_site_types: F=%.3f p=%.4f across %d groups, H0 %s",
        f_stat,
        p_anova,
        n_groups,
        "rejected" if p_anova < 0.05 else "not rejected",
    )

    return {
        "anova_table": anova_table,
        "pairwise": pd.DataFrame(rows),
        "rejected_h0": bool(p_anova < 0.05),
    }
