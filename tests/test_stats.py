"""
Tests for src/data_nature/stats/anomaly.py and regression.py.

All tests run on synthetic data — no file I/O or external services.

Coverage:
  - compute_zscores: baseline values, z-score arithmetic, NaN handling
  - detect_anomalies: known spike → Critical, clean series → empty,
    schema matches anomalies.csv, custom thresholds
  - fit_ndvi_lst: negative NDVI→LST coefficient, R²/p-value presence,
    multi-model keys
  - compare_site_types: H0 rejected on differentiated groups,
    H0 not rejected on identical groups, pairwise schema
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_nature.stats.anomaly import (
    DEFAULT_THRESHOLDS,
    compute_zscores,
    detect_anomalies,
)
from data_nature.stats.regression import compare_site_types, fit_ndvi_lst


# ── shared helpers ────────────────────────────────────────────────────────────

_RNG = np.random.default_rng(42)

ANOMALY_CSV_COLS = [
    "date", "site", "lst", "baseline", "z_score",
    "severity", "status", "ndvi_change",
]


def _make_site(
    site: str,
    land_cover: str,
    n_years: int = 10,
    lst_mean: float = 25.0,
    lst_std: float = 2.0,
    ndvi_mean: float = 0.4,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Build a full monthly time series (n_years × 12 months) for one site with
    realistic seasonal variation plus Gaussian noise.
    """
    if rng is None:
        rng = _RNG
    rows = []
    for year in range(2010, 2010 + n_years):
        for month in range(1, 13):
            seasonal = lst_std * np.sin((month - 1) * np.pi / 6)
            lst = lst_mean + seasonal + rng.normal(0, 0.5)
            ndvi = ndvi_mean + rng.normal(0, 0.02)
            rows.append(
                {
                    "date": pd.Timestamp(year, month, 1),
                    "year": year,
                    "month": month,
                    "lst": round(lst, 2),
                    "ndvi": round(ndvi, 4),
                    "site": site,
                    "land_cover": land_cover,
                }
            )
    return pd.DataFrame(rows)


def _inject_spike(
    df: pd.DataFrame, year: int, month: int, delta_celsius: float = 20.0
) -> pd.DataFrame:
    """Add an absolute *delta_celsius* to the LST of one specific row.

    Using an absolute value (not sigma × std) so the z-score is predictably
    large even with small group sizes — when n=10, a sigma-based spike inflates
    the group std and yields an effective z far below the requested sigma.
    """
    df = df.copy()
    mask = (df["year"] == year) & (df["month"] == month)
    df.loc[mask, "lst"] += delta_celsius
    return df


# ── compute_zscores ───────────────────────────────────────────────────────────


class TestComputeZscores:
    @pytest.fixture()
    def base_df(self):
        return _make_site("Forest_A", "Forest", n_years=10)

    def test_adds_required_columns(self, base_df):
        result = compute_zscores(base_df)
        for col in ("baseline", "baseline_std", "z_score"):
            assert col in result.columns, f"Missing column: {col}"

    def test_baseline_equals_group_mean(self, base_df):
        result = compute_zscores(base_df)
        # For a specific (site, month), baseline must equal the group mean
        for month in range(1, 13):
            group = base_df[(base_df["month"] == month)]["lst"]
            expected = group.mean()
            actual = result[result["month"] == month]["baseline"].iloc[0]
            assert abs(actual - expected) < 1e-6

    def test_z_score_of_mean_is_near_zero(self, base_df):
        result = compute_zscores(base_df)
        # The mean of all z-scores within each month should be ~0
        for month in range(1, 13):
            month_z = result[result["month"] == month]["z_score"].dropna()
            assert abs(month_z.mean()) < 0.5

    def test_original_rows_preserved(self, base_df):
        result = compute_zscores(base_df)
        assert len(result) == len(base_df)

    def test_nan_when_single_year(self):
        # Only 1 observation per (site, month) → std=NaN → z_score=NaN
        df = _make_site("Solo", "Forest", n_years=1)
        result = compute_zscores(df)
        assert result["z_score"].isna().all()

    def test_multisite_baselines_are_independent(self):
        df = pd.concat(
            [
                _make_site("Hot_Site", "Arid", n_years=5, lst_mean=35.0),
                _make_site("Cool_Site", "Forest", n_years=5, lst_mean=18.0),
            ],
            ignore_index=True,
        )
        result = compute_zscores(df)
        hot_baseline = result[result["site"] == "Hot_Site"]["baseline"].mean()
        cool_baseline = result[result["site"] == "Cool_Site"]["baseline"].mean()
        assert hot_baseline > cool_baseline


# ── detect_anomalies ──────────────────────────────────────────────────────────


class TestDetectAnomalies:
    @pytest.fixture()
    def site_df(self):
        # n_years=20 so the single spike row is a small fraction of each month's
        # group — preventing the spike from inflating the baseline std so much
        # that the z-score falls below the Critical threshold.
        return _make_site("Test_Site", "Forest", n_years=20)

    def test_known_spike_detected_as_critical(self, site_df):
        spiked = _inject_spike(site_df, year=2015, month=7, delta_celsius=20.0)
        anomalies = detect_anomalies(spiked)
        critical = anomalies[
            (anomalies["site"] == "Test_Site") & (anomalies["severity"] == "Critical")
        ]
        assert len(critical) >= 1, "Spike should appear as Critical anomaly"

    def test_clean_series_produces_no_anomalies(self):
        # Very small noise → z-scores always < 1.5
        df = pd.DataFrame(
            {
                "date": [pd.Timestamp(2010 + i // 12, i % 12 + 1, 1) for i in range(120)],
                "year": [2010 + i // 12 for i in range(120)],
                "month": [i % 12 + 1 for i in range(120)],
                # perfect seasonal pattern, zero noise
                "lst": [20.0 + 0.001 * (i % 12) for i in range(120)],
                "ndvi": [0.4] * 120,
                "site": ["Clean_Site"] * 120,
                "land_cover": ["Forest"] * 120,
            }
        )
        anomalies = detect_anomalies(df)
        assert len(anomalies) == 0, "Perfect series should yield no anomalies"

    def test_output_columns_match_anomalies_csv_schema(self, site_df):
        spiked = _inject_spike(site_df, year=2015, month=7, delta_celsius=20.0)
        anomalies = detect_anomalies(spiked)
        assert list(anomalies.columns) == ANOMALY_CSV_COLS

    def test_status_column_is_new(self, site_df):
        spiked = _inject_spike(site_df, year=2015, month=7, delta_celsius=20.0)
        anomalies = detect_anomalies(spiked)
        if len(anomalies):
            assert (anomalies["status"] == "New").all()

    def test_severity_values_are_valid(self, site_df):
        spiked = _inject_spike(site_df, year=2015, month=7, delta_celsius=20.0)
        anomalies = detect_anomalies(spiked)
        valid = {"Warning", "Severe", "Critical"}
        assert set(anomalies["severity"].unique()).issubset(valid)

    def test_custom_thresholds_lower_means_more_anomalies(self, site_df):
        default = detect_anomalies(site_df)
        strict = detect_anomalies(
            site_df, thresholds={"Warning": 0.5, "Severe": 1.0, "Critical": 1.5}
        )
        assert len(strict) >= len(default)

    def test_accepts_pre_scored_df(self, site_df):
        # If z_score already present, should not recompute baseline
        scored = compute_zscores(site_df)
        scored["baseline"] = 0.0  # intentionally wrong — should be used as-is
        scored["z_score"] = 99.0  # everything should be Critical
        anomalies = detect_anomalies(scored)
        assert (anomalies["severity"] == "Critical").all()

    def test_ndvi_change_is_diff_within_site(self, site_df):
        spiked = _inject_spike(site_df, year=2015, month=7, delta_celsius=20.0)
        anomalies = detect_anomalies(spiked)
        # ndvi_change should be a finite float for all rows except the first of the site
        non_null = anomalies["ndvi_change"].dropna()
        assert len(non_null) > 0

    def test_spike_z_score_above_threshold(self, site_df):
        spiked = _inject_spike(site_df, year=2015, month=7, delta_celsius=20.0)
        scored = compute_zscores(spiked)
        spike_z = scored[
            (scored["year"] == 2015) & (scored["month"] == 7)
        ]["z_score"].iloc[0]
        assert spike_z >= DEFAULT_THRESHOLDS["Critical"]


# ── fit_ndvi_lst ──────────────────────────────────────────────────────────────


class TestFitNdviLst:
    @pytest.fixture()
    def neg_corr_df(self):
        """Synthetic data with a clear negative NDVI → LST relationship."""
        rng = np.random.default_rng(0)
        n = 200
        ndvi = rng.uniform(0.1, 0.8, n)
        lst = 40.0 - 25.0 * ndvi + rng.normal(0, 1.5, n)
        return pd.DataFrame(
            {
                "lst": lst,
                "ndvi": ndvi,
                "land_cover": rng.choice(["Forest", "Arid", "Urban"], n).tolist(),
                "month": rng.integers(1, 13, n).tolist(),
            }
        )

    def test_ndvi_coefficient_is_negative(self, neg_corr_df):
        result = fit_ndvi_lst(neg_corr_df)
        assert result["simple"]["ndvi_coef"] < 0, (
            "NDVI coefficient should be negative (more vegetation → lower LST)"
        )

    def test_multi_ndvi_coefficient_is_negative(self, neg_corr_df):
        result = fit_ndvi_lst(neg_corr_df)
        assert result["multi"]["ndvi_coef"] < 0

    def test_simple_returns_required_keys(self, neg_corr_df):
        result = fit_ndvi_lst(neg_corr_df)
        for key in ("ndvi_coef", "intercept", "r2", "p_value", "n"):
            assert key in result["simple"], f"Missing key in simple: {key}"

    def test_multi_returns_required_keys(self, neg_corr_df):
        result = fit_ndvi_lst(neg_corr_df)
        for key in ("ndvi_coef", "intercept", "r2", "adj_r2", "p_values", "n"):
            assert key in result["multi"], f"Missing key in multi: {key}"

    def test_r2_between_zero_and_one(self, neg_corr_df):
        result = fit_ndvi_lst(neg_corr_df)
        assert 0.0 <= result["simple"]["r2"] <= 1.0
        assert 0.0 <= result["multi"]["r2"] <= 1.0

    def test_p_value_significant_for_strong_relationship(self, neg_corr_df):
        result = fit_ndvi_lst(neg_corr_df)
        assert result["simple"]["p_value"] < 0.05

    def test_n_matches_non_null_rows(self, neg_corr_df):
        result = fit_ndvi_lst(neg_corr_df)
        assert result["simple"]["n"] == len(neg_corr_df.dropna(subset=["lst", "ndvi"]))

    def test_nan_rows_dropped_gracefully(self, neg_corr_df):
        df_with_nans = neg_corr_df.copy()
        df_with_nans.loc[[0, 1, 2], "ndvi"] = float("nan")
        result = fit_ndvi_lst(df_with_nans)
        assert result["simple"]["n"] == len(neg_corr_df) - 3

    def test_raises_on_too_few_rows(self):
        tiny = pd.DataFrame(
            {"lst": [1.0, 2.0], "ndvi": [0.3, 0.4],
             "land_cover": ["Forest", "Forest"], "month": [1, 2]}
        )
        with pytest.raises(ValueError, match="at least 3"):
            fit_ndvi_lst(tiny)

    def test_multi_r2_geq_simple_r2(self, neg_corr_df):
        result = fit_ndvi_lst(neg_corr_df)
        assert result["multi"]["r2"] >= result["simple"]["r2"]


# ── compare_site_types ────────────────────────────────────────────────────────


class TestCompareSiteTypes:
    @pytest.fixture()
    def differentiated_df(self):
        """Three land-cover groups with clearly different mean LSTs."""
        rng = np.random.default_rng(7)
        groups = {"Forest": 22.0, "Arid": 38.0, "Urban": 30.0}
        rows = []
        for lc, mean in groups.items():
            lst = rng.normal(mean, 1.5, 60).tolist()
            rows += [{"lst": v, "land_cover": lc} for v in lst]
        return pd.DataFrame(rows)

    @pytest.fixture()
    def identical_df(self):
        """Three groups drawn from the same distribution — H0 should hold."""
        rng = np.random.default_rng(99)
        rows = []
        for lc in ["Forest", "Arid", "Urban"]:
            lst = rng.normal(25.0, 2.0, 50).tolist()
            rows += [{"lst": v, "land_cover": lc} for v in lst]
        return pd.DataFrame(rows)

    def test_anova_table_columns(self, differentiated_df):
        result = compare_site_types(differentiated_df)
        expected = {"F", "p_value", "df_between", "df_within"}
        assert expected.issubset(set(result["anova_table"].columns))

    def test_pairwise_columns(self, differentiated_df):
        result = compare_site_types(differentiated_df)
        expected = {"group_a", "group_b", "t_stat", "p_value", "significant"}
        assert expected.issubset(set(result["pairwise"].columns))

    def test_pairwise_has_correct_number_of_pairs(self, differentiated_df):
        result = compare_site_types(differentiated_df)
        n_groups = differentiated_df["land_cover"].nunique()
        expected_pairs = n_groups * (n_groups - 1) // 2
        assert len(result["pairwise"]) == expected_pairs

    def test_rejects_h0_for_differentiated_groups(self, differentiated_df):
        result = compare_site_types(differentiated_df)
        assert result["rejected_h0"] is True
        assert result["anova_table"]["p_value"].iloc[0] < 0.05

    def test_does_not_reject_h0_for_identical_groups(self, identical_df):
        result = compare_site_types(identical_df)
        # With identical distributions, ANOVA p-value should be large
        assert result["anova_table"]["p_value"].iloc[0] > 0.05

    def test_all_pairs_significant_for_differentiated_groups(self, differentiated_df):
        result = compare_site_types(differentiated_df)
        assert result["pairwise"]["significant"].all()

    def test_f_statistic_is_positive(self, differentiated_df):
        result = compare_site_types(differentiated_df)
        assert result["anova_table"]["F"].iloc[0] > 0

    def test_raises_with_only_one_group(self):
        df = pd.DataFrame({"lst": [1.0, 2.0, 3.0], "land_cover": ["Forest"] * 3})
        with pytest.raises(ValueError, match="at least two"):
            compare_site_types(df)

    def test_nan_lst_rows_excluded(self, differentiated_df):
        df_with_nans = differentiated_df.copy()
        df_with_nans.loc[0, "lst"] = float("nan")
        # Should not crash
        result = compare_site_types(df_with_nans)
        assert "rejected_h0" in result
