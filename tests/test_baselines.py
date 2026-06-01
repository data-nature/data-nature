"""
tests/test_baselines.py

Tests for src/data_nature/stats/baselines.py covering:
  1. Known synthetic series — exact mean/std values
  2. Recent-window edge case — first `min_years` rows produce NaN
  3. Schema validation — happy path and missing-column detection
  4. Multi-site, multi-month independence
  5. Single-site full round-trip (load → compute → shape)
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from data_nature.stats.baselines import (
    DATA_CONTRACT,
    compute_baselines,
    validate_schema,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(
    site: str = "Test_Site",
    month: int = 6,
    lsts: list[float] | None = None,
    ndvis: list[float] | None = None,
    start_year: int = 2000,
) -> pd.DataFrame:
    """Build a minimal site_monthly-schema DataFrame for one site/month."""
    if lsts is None:
        lsts = [20.0, 21.0, 22.0, 23.0, 24.0]
    if ndvis is None:
        ndvis = [0.5] * len(lsts)
    n = len(lsts)
    return pd.DataFrame(
        {
            "year": list(range(start_year, start_year + n)),
            "month": [month] * n,
            "site": [site] * n,
            "lst": lsts,
            "ndvi": ndvis,
            "z_score_lst": [None] * n,
            "z_score_ndvi": [None] * n,
            "delta": [None] * n,
            "is_anomaly": [False] * n,
        }
    )


# ---------------------------------------------------------------------------
# 1. Known synthetic series
# ---------------------------------------------------------------------------

class TestKnownSeries:
    """Verify exact numeric outputs against hand-computed values."""

    def test_first_valid_row_mean(self):
        """4th year baseline_mean == mean of first 3 years."""
        df = _make_df(lsts=[10.0, 20.0, 30.0, 99.0, 99.0], month=1)
        bl = compute_baselines(df, min_years=3)

        row = bl[(bl["year"] == 2003)]  # 4th year (index 3), prior = [10,20,30]
        assert len(row) == 1
        assert math.isclose(row.iloc[0]["baseline_mean"], 20.0, rel_tol=1e-9)

    def test_first_valid_row_std(self):
        """4th year baseline_std == std(10,20,30) with ddof=1 == 10.0."""
        df = _make_df(lsts=[10.0, 20.0, 30.0, 99.0, 99.0], month=1)
        bl = compute_baselines(df, min_years=3)

        row = bl[(bl["year"] == 2003)]
        expected_std = pd.Series([10.0, 20.0, 30.0]).std(ddof=1)
        assert math.isclose(row.iloc[0]["baseline_std"], expected_std, rel_tol=1e-9)

    def test_expanding_window_grows(self):
        """Each successive year's baseline_mean uses one more prior observation."""
        lsts = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        df = _make_df(lsts=lsts, month=3)
        bl = compute_baselines(df, min_years=1)

        # Year 2001 (i=1): prior=[10] → mean=10
        row_2001 = bl[bl["year"] == 2001].iloc[0]
        assert math.isclose(row_2001["baseline_mean"], 10.0)

        # Year 2004 (i=4): prior=[10,12,14,16] → mean=13
        row_2004 = bl[bl["year"] == 2004].iloc[0]
        assert math.isclose(row_2004["baseline_mean"], 13.0)

    def test_ndvi_baseline_computed(self):
        """NDVI baselines are computed independently from LST."""
        ndvis = [0.2, 0.4, 0.6, 0.8]
        df = _make_df(lsts=[10.0] * 4, ndvis=ndvis, month=5)
        bl = compute_baselines(df, min_years=3)

        row = bl[bl["year"] == 2003].iloc[0]
        expected_ndvi_mean = sum(ndvis[:3]) / 3  # 0.4
        assert math.isclose(row["ndvi_baseline_mean"], expected_ndvi_mean, rel_tol=1e-9)

    def test_baseline_count_correct(self):
        """baseline_count == number of prior-year observations."""
        df = _make_df(lsts=[1.0, 2.0, 3.0, 4.0, 5.0], month=7)
        bl = compute_baselines(df, min_years=1)

        for i, year in enumerate(range(2000, 2005)):
            count = bl[bl["year"] == year].iloc[0]["baseline_count"]
            assert count == i, f"Year {year}: expected count {i}, got {count}"


# ---------------------------------------------------------------------------
# 2. Recent-window / edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Rows with fewer prior years than min_years must be NaN."""

    def test_early_years_are_nan(self):
        """Years 0..min_years-1 should have NaN baseline_mean."""
        df = _make_df(lsts=[5.0] * 10, month=4)
        bl = compute_baselines(df, min_years=3)

        early = bl[bl["year"].isin([2000, 2001, 2002])]
        assert early["baseline_mean"].isna().all(), "Early years must be NaN"
        assert early["baseline_std"].isna().all()

    def test_first_valid_year_not_nan(self):
        """Year at index min_years should have a valid baseline."""
        df = _make_df(lsts=[5.0] * 10, month=4)
        bl = compute_baselines(df, min_years=3)

        row = bl[bl["year"] == 2003].iloc[0]
        assert not math.isnan(row["baseline_mean"])

    def test_single_prior_year_std_is_nan(self):
        """With only 1 prior year, std is undefined (NaN) even at min_years=1."""
        df = _make_df(lsts=[10.0, 20.0], month=2)
        bl = compute_baselines(df, min_years=1)

        row = bl[bl["year"] == 2001].iloc[0]
        assert not math.isnan(row["baseline_mean"])   # mean is computable
        assert math.isnan(row["baseline_std"])         # std needs ≥2 points

    def test_min_years_zero_gives_nan_only_first_row(self):
        """With min_years=0, only the very first row (0 prior obs) is NaN."""
        df = _make_df(lsts=[10.0, 20.0, 30.0], month=9)
        bl = compute_baselines(df, min_years=0)

        assert math.isnan(bl[bl["year"] == 2000].iloc[0]["baseline_mean"])
        assert not math.isnan(bl[bl["year"] == 2001].iloc[0]["baseline_mean"])

    def test_missing_required_column_raises(self):
        """compute_baselines raises ValueError if required column is absent."""
        df = _make_df()
        df = df.drop(columns=["ndvi"])
        with pytest.raises(ValueError, match="missing columns"):
            compute_baselines(df)


# ---------------------------------------------------------------------------
# 3. Schema validation
# ---------------------------------------------------------------------------

class TestSchemaValidation:

    def test_happy_path(self, tmp_path: Path):
        """All files present with correct columns → no exception, all in 'ok'."""
        for fname, cols in DATA_CONTRACT.items():
            (tmp_path / fname).write_text(",".join(cols) + "\n")

        report = validate_schema(tmp_path)
        assert set(report["ok"]) == set(DATA_CONTRACT.keys())
        assert report["missing_cols"] == {}
        assert report["not_found"] == []

    def test_missing_file_is_skipped(self, tmp_path: Path):
        """A file absent from the directory is recorded as 'not_found', not an error."""
        # Write only one file
        fname = "site_locations.csv"
        cols = DATA_CONTRACT[fname]
        (tmp_path / fname).write_text(",".join(cols) + "\n")

        report = validate_schema(tmp_path)
        assert fname in report["ok"]
        # Everything else is not_found, but no exception
        assert len(report["not_found"]) == len(DATA_CONTRACT) - 1

    def test_missing_column_raises(self, tmp_path: Path):
        """A present file missing a required column triggers ValueError."""
        fname = "site_monthly.csv"
        bad_cols = ["year", "month", "site"]  # missing lst, ndvi, etc.
        (tmp_path / fname).write_text(",".join(bad_cols) + "\n")

        with pytest.raises(ValueError, match="Schema validation failed"):
            validate_schema(tmp_path)

    def test_extra_columns_are_fine(self, tmp_path: Path):
        """Files with *extra* columns (superset) should still pass."""
        fname = "lst_history.csv"
        required = DATA_CONTRACT[fname]
        extra_cols = required + ["extra_col", "another_col"]
        (tmp_path / fname).write_text(",".join(extra_cols) + "\n")

        report = validate_schema(tmp_path)
        assert fname in report["ok"]


# ---------------------------------------------------------------------------
# 4. Multi-site, multi-month independence
# ---------------------------------------------------------------------------

class TestMultiSiteMultiMonth:

    def test_sites_are_independent(self):
        """Baselines for site A are unaffected by site B's values."""
        df_a = _make_df(site="Site_A", lsts=[10.0, 10.0, 10.0, 10.0], month=1)
        df_b = _make_df(site="Site_B", lsts=[100.0, 100.0, 100.0, 100.0], month=1)
        df = pd.concat([df_a, df_b], ignore_index=True)
        bl = compute_baselines(df, min_years=3)

        row_a = bl[(bl["site"] == "Site_A") & (bl["year"] == 2003)].iloc[0]
        row_b = bl[(bl["site"] == "Site_B") & (bl["year"] == 2003)].iloc[0]
        assert math.isclose(row_a["baseline_mean"], 10.0)
        assert math.isclose(row_b["baseline_mean"], 100.0)

    def test_months_are_independent(self):
        """Baseline for month 1 is unaffected by month 7 data."""
        df_jan = _make_df(site="S", lsts=[5.0, 5.0, 5.0, 5.0], month=1)
        df_jul = _make_df(site="S", lsts=[30.0, 30.0, 30.0, 30.0], month=7)
        df = pd.concat([df_jan, df_jul], ignore_index=True)
        bl = compute_baselines(df, min_years=3)

        row_jan = bl[(bl["month"] == 1) & (bl["year"] == 2003)].iloc[0]
        row_jul = bl[(bl["month"] == 7) & (bl["year"] == 2003)].iloc[0]
        assert math.isclose(row_jan["baseline_mean"], 5.0)
        assert math.isclose(row_jul["baseline_mean"], 30.0)

    def test_output_has_correct_row_count(self):
        """Output should have (n_sites × n_months × n_years) rows."""
        n_years = 5
        df = pd.concat(
            [
                _make_df(site=f"Site_{s}", month=m, lsts=[float(i) for i in range(n_years)])
                for s in range(3)
                for m in [1, 6, 12]
            ],
            ignore_index=True,
        )
        bl = compute_baselines(df)
        assert len(bl) == 3 * 3 * n_years  # 3 sites × 3 months × 5 years


# ---------------------------------------------------------------------------
# 5. Integration: real processed data (skipped if not available)
# ---------------------------------------------------------------------------

_PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

@pytest.mark.skipif(
    not (_PROCESSED / "site_monthly.csv").exists(),
    reason="Real processed data not available",
)
class TestRealData:

    def test_output_shape(self):
        """Real data produces one row per (site, month, year)."""
        df = pd.read_csv(_PROCESSED / "site_monthly.csv")
        bl = compute_baselines(df)

        expected_rows = len(df)  # one baseline row per source row
        assert len(bl) == expected_rows

    def test_all_sites_present(self):
        """All 8 sites appear in the baselines output."""
        df = pd.read_csv(_PROCESSED / "site_monthly.csv")
        bl = compute_baselines(df)
        assert bl["site"].nunique() == 8

    def test_no_future_data_leakage(self):
        """For any (site, month, year) row, baseline_count < year - 1999."""
        df = pd.read_csv(_PROCESSED / "site_monthly.csv")
        bl = compute_baselines(df, min_years=1).dropna(subset=["baseline_mean"])
        # baseline_count == number of prior years == year - first_year
        # Just check it never exceeds the row index within the sorted group
        for (site, month), grp in bl.groupby(["site", "month"]):
            grp = grp.sort_values("year").reset_index(drop=True)
            # After dropna the first row has baseline_count >= 1; track the
            # expected count as grp.iloc[0]["baseline_count"] + offset.
            for pos, (_, row) in enumerate(grp.iterrows()):
                expected_count = grp.iloc[0]["baseline_count"] + pos
                assert row["baseline_count"] == expected_count, (
                    f"Leakage detected: site={site}, month={month}, year={row['year']}, "
                    f"count={row['baseline_count']} but expected {expected_count}"
                )