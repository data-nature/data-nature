"""
Unit tests for src/data_nature/ingest/earth_engine.py.

All Google Earth Engine network calls are mocked — no live GEE connection
is required to run this suite.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

import data_nature.ingest.earth_engine as ee_mod
from data_nature.ingest.earth_engine import EEAuthError


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_ee_flag():
    """Reset the module-level initialisation flag before every test."""
    original = ee_mod._EE_INITIALIZED
    ee_mod._EE_INITIALIZED = False
    yield
    ee_mod._EE_INITIALIZED = original


@pytest.fixture()
def fake_sites_df():
    return pd.DataFrame(
        {
            "site": ["Carmel_Forest", "Nazareth_Urban"],
            "lat": [32.72, 32.70],
            "lng": [35.03, 35.30],
            "land_cover": ["Forest", "Urban"],
        }
    )


@pytest.fixture()
def ndvi_features():
    """Simulated ee.FeatureCollection.getInfo() payload for NDVI."""
    return {
        "features": [
            {"properties": {"date": "2020-01-01", "NDVI": 2500}},
            {"properties": {"date": "2020-01-17", "NDVI": 2600}},
            {"properties": {"date": "2020-02-02", "NDVI": 3000}},
            {"properties": {"date": "2020-02-18", "NDVI": None}},  # null — should be skipped
        ]
    }


# ── authenticate() ────────────────────────────────────────────────────────────


class TestAuthenticate:
    def test_service_account_path(self, monkeypatch):
        monkeypatch.setenv("GEE_SERVICE_ACCOUNT_EMAIL", "sa@proj.iam.gserviceaccount.com")
        monkeypatch.setenv("GEE_PRIVATE_KEY_PATH", "/path/to/key.json")
        monkeypatch.setenv("GEE_PROJECT_ID", "my-project")

        with (
            patch.object(ee_mod.ee, "ServiceAccountCredentials") as mock_creds,
            patch.object(ee_mod.ee, "Initialize") as mock_init,
        ):
            ee_mod.authenticate()

        mock_creds.assert_called_once_with("sa@proj.iam.gserviceaccount.com", "/path/to/key.json")
        mock_init.assert_called_once()
        assert ee_mod._EE_INITIALIZED is True

    def test_interactive_fallback_when_env_vars_absent(self, monkeypatch):
        monkeypatch.delenv("GEE_SERVICE_ACCOUNT_EMAIL", raising=False)
        monkeypatch.delenv("GEE_PRIVATE_KEY_PATH", raising=False)
        monkeypatch.delenv("GEE_PROJECT_ID", raising=False)

        with (
            # prevent load_dotenv() from reloading the real .env file
            patch("data_nature.ingest.earth_engine.load_dotenv"),
            patch.object(ee_mod.ee, "Authenticate") as mock_auth,
            patch.object(ee_mod.ee, "Initialize") as mock_init,
        ):
            ee_mod.authenticate()

        mock_auth.assert_called_once()
        mock_init.assert_called_once()
        assert ee_mod._EE_INITIALIZED is True

    def test_raises_ee_auth_error_on_failure(self, monkeypatch):
        monkeypatch.delenv("GEE_SERVICE_ACCOUNT_EMAIL", raising=False)
        monkeypatch.delenv("GEE_PRIVATE_KEY_PATH", raising=False)

        with patch.object(ee_mod.ee, "Authenticate", side_effect=Exception("no token")):
            with pytest.raises(EEAuthError, match="Earth Engine authentication failed"):
                ee_mod.authenticate()

        assert ee_mod._EE_INITIALIZED is False

    def test_error_message_contains_remediation_hints(self, monkeypatch):
        monkeypatch.delenv("GEE_SERVICE_ACCOUNT_EMAIL", raising=False)
        monkeypatch.delenv("GEE_PRIVATE_KEY_PATH", raising=False)

        with patch.object(ee_mod.ee, "Authenticate", side_effect=Exception("oops")):
            with pytest.raises(EEAuthError) as exc_info:
                ee_mod.authenticate()

        msg = str(exc_info.value)
        assert "GEE_SERVICE_ACCOUNT_EMAIL" in msg or "earthengine authenticate" in msg

    def test_idempotent_second_call(self):
        ee_mod._EE_INITIALIZED = True

        with patch.object(ee_mod.ee, "Initialize") as mock_init:
            ee_mod.authenticate()
            ee_mod.authenticate()

        mock_init.assert_not_called()


# ── _extract_ndvi_monthly() ───────────────────────────────────────────────────


class TestExtractNdviMonthly:
    def _make_point(self):
        point = MagicMock()
        return point

    def test_returns_monthly_means(self, ndvi_features):
        mock_collection = MagicMock()
        # wire the builder chain so .filterDate().filterBounds().select() all return the same mock
        mock_collection.filterDate.return_value = mock_collection
        mock_collection.filterBounds.return_value = mock_collection
        mock_collection.select.return_value = mock_collection
        mock_collection.map.return_value.getInfo.return_value = ndvi_features

        with patch.object(ee_mod.ee, "ImageCollection", return_value=mock_collection):
            df = ee_mod._extract_ndvi_monthly(self._make_point(), "2020-01-01", "2020-03-01")

        assert list(df.columns) == ["year", "month", "ndvi"]
        # Jan: mean of 2500 and 2600 scaled = (0.25 + 0.26) / 2
        jan = df[(df["year"] == 2020) & (df["month"] == 1)]["ndvi"].iloc[0]
        assert abs(jan - 0.255) < 1e-4
        # Feb: only one non-null value (3000 × 0.0001 = 0.3)
        feb = df[(df["year"] == 2020) & (df["month"] == 2)]["ndvi"].iloc[0]
        assert abs(feb - 0.3) < 1e-4

    def test_null_ndvi_values_are_skipped(self, ndvi_features):
        mock_collection = MagicMock()
        mock_collection.filterDate.return_value = mock_collection
        mock_collection.filterBounds.return_value = mock_collection
        mock_collection.select.return_value = mock_collection
        mock_collection.map.return_value.getInfo.return_value = ndvi_features

        with patch.object(ee_mod.ee, "ImageCollection", return_value=mock_collection):
            df = ee_mod._extract_ndvi_monthly(self._make_point(), "2020-01-01", "2020-03-01")

        # Feb row should exist with one valid value (the None entry is dropped)
        feb = df[(df["year"] == 2020) & (df["month"] == 2)]
        assert len(feb) == 1

    def test_empty_collection_returns_empty_df(self):
        mock_collection = MagicMock()
        mock_collection.map.return_value.getInfo.return_value = {"features": []}

        with patch.object(ee_mod.ee, "ImageCollection", return_value=mock_collection):
            df = ee_mod._extract_ndvi_monthly(self._make_point(), "2020-01-01", "2020-02-01")

        assert df.empty
        assert list(df.columns) == ["year", "month", "ndvi"]


# ── _extract_lst_monthly() ────────────────────────────────────────────────────


class TestExtractLstMonthly:
    def _make_point(self):
        return MagicMock()

    def _mock_collection(self, monthly_raw_values: dict[tuple, float | None]):
        """Build a mock collection where the monthly filterDate().mean().reduceRegion()
        chain returns values from *monthly_raw_values* keyed by (year, month).

        The first filterDate() call is the full-range builder chain — it must return
        the collection itself so .filterBounds().select() keep chaining correctly.
        Subsequent calls are the per-month filters inside the loop.
        """
        mock_col = MagicMock()
        mock_col.filterBounds.return_value = mock_col
        mock_col.select.return_value = mock_col

        call_count = [0]

        def filter_date_side_effect(start, end):
            call_count[0] += 1
            if call_count[0] == 1:
                # initial full-range filter — return self so the builder chain works
                return mock_col
            # monthly filter — return per-month result chain
            y, m, _ = start.split("-")
            val = monthly_raw_values.get((int(y), int(m)))
            get_mock = MagicMock()
            get_mock.getInfo.return_value = val
            reduce_mock = MagicMock()
            reduce_mock.get.return_value = get_mock
            mean_mock = MagicMock()
            mean_mock.reduceRegion.return_value = reduce_mock
            filter_result = MagicMock()
            filter_result.mean.return_value = mean_mock
            return filter_result

        mock_col.filterDate.side_effect = filter_date_side_effect
        return mock_col

    def test_converts_kelvin_to_celsius(self):
        # raw 14_500 → 14500 × 0.02 − 273.15 = 290 − 273.15 = 16.85 °C
        mock_col = self._mock_collection({(2020, 1): 14_500})

        with (
            patch.object(ee_mod.ee, "ImageCollection", return_value=mock_col),
            patch.object(ee_mod.ee, "Reducer"),  # prevents ee.Reducer.mean() from hitting the API
        ):
            df = ee_mod._extract_lst_monthly(self._make_point(), "2020-01-01", "2020-02-01")

        assert len(df) == 1
        assert abs(df.iloc[0]["lst"] - 16.85) < 0.01

    def test_null_months_are_skipped(self):
        mock_col = self._mock_collection({(2020, 1): None, (2020, 2): 14_500})

        with (
            patch.object(ee_mod.ee, "ImageCollection", return_value=mock_col),
            patch.object(ee_mod.ee, "Reducer"),
        ):
            df = ee_mod._extract_lst_monthly(self._make_point(), "2020-01-01", "2020-03-01")

        assert len(df) == 1
        assert df.iloc[0]["month"] == 2

    def test_failed_month_is_warned_and_skipped(self, caplog):
        mock_col = MagicMock()
        mock_col.filterBounds.return_value = mock_col
        mock_col.select.return_value = mock_col

        call_count = [0]

        def filter_date_side_effect(start, end):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_col  # full-range builder call — must not raise
            raise Exception("EE timeout")

        mock_col.filterDate.side_effect = filter_date_side_effect

        import logging

        with patch.object(ee_mod.ee, "ImageCollection", return_value=mock_col):
            with caplog.at_level(logging.WARNING, logger="data_nature.ingest.earth_engine"):
                df = ee_mod._extract_lst_monthly(self._make_point(), "2020-01-01", "2020-02-01")

        assert df.empty
        assert any("LST extraction skipped" in r.message for r in caplog.records)

    def test_empty_range_returns_empty_df(self):
        collection = MagicMock()

        with patch.object(ee_mod.ee, "ImageCollection", return_value=collection):
            # start == end → no periods
            df = ee_mod._extract_lst_monthly(self._make_point(), "2020-01-01", "2020-01-01")

        assert df.empty
        assert list(df.columns) == ["year", "month", "lst"]


# ── fetch_site_series() ───────────────────────────────────────────────────────


class TestFetchSiteSeries:
    @pytest.fixture()
    def mock_ndvi_df(self):
        return pd.DataFrame({"year": [2020, 2020], "month": [1, 2], "ndvi": [0.4, 0.5]})

    @pytest.fixture()
    def mock_lst_df(self):
        return pd.DataFrame({"year": [2020, 2020], "month": [1, 2], "lst": [16.0, 17.5]})

    def test_returns_tidy_dataframe(self, fake_sites_df, mock_ndvi_df, mock_lst_df):
        with (
            patch.object(ee_mod, "authenticate"),
            patch.object(ee_mod, "_load_sites", return_value=fake_sites_df),
            patch.object(ee_mod.ee, "Geometry") as mock_geom,
            patch.object(ee_mod, "_extract_ndvi_monthly", return_value=mock_ndvi_df),
            patch.object(ee_mod, "_extract_lst_monthly", return_value=mock_lst_df),
        ):
            df = ee_mod.fetch_site_series("Carmel_Forest", "2020-01-01", "2020-03-01")

        expected_cols = ["date", "year", "month", "lst", "ndvi", "site", "land_cover"]
        assert list(df.columns) == expected_cols

    def test_correct_site_and_land_cover_attached(self, fake_sites_df, mock_ndvi_df, mock_lst_df):
        with (
            patch.object(ee_mod, "authenticate"),
            patch.object(ee_mod, "_load_sites", return_value=fake_sites_df),
            patch.object(ee_mod.ee, "Geometry"),
            patch.object(ee_mod, "_extract_ndvi_monthly", return_value=mock_ndvi_df),
            patch.object(ee_mod, "_extract_lst_monthly", return_value=mock_lst_df),
        ):
            df = ee_mod.fetch_site_series("Carmel_Forest", "2020-01-01", "2020-03-01")

        assert (df["site"] == "Carmel_Forest").all()
        assert (df["land_cover"] == "Forest").all()

    def test_sorted_by_date(self, fake_sites_df, mock_ndvi_df, mock_lst_df):
        with (
            patch.object(ee_mod, "authenticate"),
            patch.object(ee_mod, "_load_sites", return_value=fake_sites_df),
            patch.object(ee_mod.ee, "Geometry"),
            patch.object(ee_mod, "_extract_ndvi_monthly", return_value=mock_ndvi_df),
            patch.object(ee_mod, "_extract_lst_monthly", return_value=mock_lst_df),
        ):
            df = ee_mod.fetch_site_series("Carmel_Forest", "2020-01-01", "2020-03-01")

        assert df["date"].is_monotonic_increasing

    def test_date_column_is_first_of_month(self, fake_sites_df, mock_ndvi_df, mock_lst_df):
        with (
            patch.object(ee_mod, "authenticate"),
            patch.object(ee_mod, "_load_sites", return_value=fake_sites_df),
            patch.object(ee_mod.ee, "Geometry"),
            patch.object(ee_mod, "_extract_ndvi_monthly", return_value=mock_ndvi_df),
            patch.object(ee_mod, "_extract_lst_monthly", return_value=mock_lst_df),
        ):
            df = ee_mod.fetch_site_series("Carmel_Forest", "2020-01-01", "2020-03-01")

        assert (df["date"].dt.day == 1).all()

    def test_raises_value_error_for_unknown_site(self, fake_sites_df):
        with (
            patch.object(ee_mod, "authenticate"),
            patch.object(ee_mod, "_load_sites", return_value=fake_sites_df),
        ):
            with pytest.raises(ValueError, match="not found in site_locations.csv"):
                ee_mod.fetch_site_series("NonExistent_Site")

    def test_value_error_lists_available_sites(self, fake_sites_df):
        with (
            patch.object(ee_mod, "authenticate"),
            patch.object(ee_mod, "_load_sites", return_value=fake_sites_df),
        ):
            with pytest.raises(ValueError) as exc_info:
                ee_mod.fetch_site_series("Ghost_Site")

        assert "Carmel_Forest" in str(exc_info.value)

    def test_ee_auth_error_propagates(self, fake_sites_df):
        with (
            patch.object(ee_mod, "authenticate", side_effect=EEAuthError("no auth")),
            patch.object(ee_mod, "_load_sites", return_value=fake_sites_df),
        ):
            with pytest.raises(EEAuthError):
                ee_mod.fetch_site_series("Carmel_Forest")

    def test_passes_start_end_to_extractors(self, fake_sites_df, mock_ndvi_df, mock_lst_df):
        with (
            patch.object(ee_mod, "authenticate"),
            patch.object(ee_mod, "_load_sites", return_value=fake_sites_df),
            patch.object(ee_mod.ee, "Geometry"),
            patch.object(ee_mod, "_extract_ndvi_monthly", return_value=mock_ndvi_df) as mock_n,
            patch.object(ee_mod, "_extract_lst_monthly", return_value=mock_lst_df) as mock_l,
        ):
            ee_mod.fetch_site_series("Carmel_Forest", "2010-06-01", "2015-01-01")

        _, n_start, n_end = mock_n.call_args[0]
        assert n_start == "2010-06-01"
        assert n_end == "2015-01-01"

        _, l_start, l_end = mock_l.call_args[0]
        assert l_start == "2010-06-01"
        assert l_end == "2015-01-01"


# ── run_pipeline() ────────────────────────────────────────────────────────────


class TestRunPipeline:
    def _make_site_df(self, n_rows=3):
        return pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=n_rows, freq="MS"),
                "year": [2020] * n_rows,
                "month": list(range(1, n_rows + 1)),
                "lst": [16.0] * n_rows,
                "ndvi": [0.4] * n_rows,
                "site": ["Carmel_Forest"] * n_rows,
                "land_cover": ["Forest"] * n_rows,
            }
        )

    def test_writes_one_csv_per_site(self, fake_sites_df, tmp_path):
        site_df = self._make_site_df()

        with (
            patch.object(ee_mod, "_load_sites", return_value=fake_sites_df),
            patch.object(ee_mod, "fetch_site_series", return_value=site_df),
        ):
            ee_mod.run_pipeline(output_dir=tmp_path)

        written = sorted(p.name for p in tmp_path.glob("*.csv"))
        assert written == ["Carmel_Forest.csv", "Nazareth_Urban.csv"]

    def test_csv_content_matches_dataframe(self, fake_sites_df, tmp_path):
        site_df = self._make_site_df()

        with (
            patch.object(ee_mod, "_load_sites", return_value=fake_sites_df),
            patch.object(ee_mod, "fetch_site_series", return_value=site_df),
        ):
            ee_mod.run_pipeline(output_dir=tmp_path)

        written = pd.read_csv(tmp_path / "Carmel_Forest.csv")
        assert list(written.columns) == list(site_df.columns)
        assert len(written) == len(site_df)

    def test_creates_output_directory(self, fake_sites_df, tmp_path):
        new_dir = tmp_path / "deep" / "nested"
        site_df = self._make_site_df()

        with (
            patch.object(ee_mod, "_load_sites", return_value=fake_sites_df),
            patch.object(ee_mod, "fetch_site_series", return_value=site_df),
        ):
            ee_mod.run_pipeline(output_dir=new_dir)

        assert new_dir.exists()

    def test_auth_error_aborts_pipeline(self, fake_sites_df, tmp_path):
        with (
            patch.object(ee_mod, "_load_sites", return_value=fake_sites_df),
            patch.object(
                ee_mod, "fetch_site_series", side_effect=EEAuthError("bad creds")
            ),
        ):
            with pytest.raises(EEAuthError):
                ee_mod.run_pipeline(output_dir=tmp_path)

        assert list(tmp_path.glob("*.csv")) == []

    def test_individual_site_failure_continues_to_next(self, fake_sites_df, tmp_path):
        site_df = self._make_site_df()
        call_count = 0

        def side_effect(site, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if site == "Carmel_Forest":
                raise RuntimeError("EE timeout")
            return site_df

        with (
            patch.object(ee_mod, "_load_sites", return_value=fake_sites_df),
            patch.object(ee_mod, "fetch_site_series", side_effect=side_effect),
        ):
            ee_mod.run_pipeline(output_dir=tmp_path)

        # Carmel_Forest failed — only Nazareth_Urban should be written
        written = [p.name for p in tmp_path.glob("*.csv")]
        assert "Nazareth_Urban.csv" in written
        assert "Carmel_Forest.csv" not in written

    def test_default_output_dir_is_data_raw(self, fake_sites_df):
        site_df = self._make_site_df()
        captured_paths: list[Path] = []

        original_to_csv = pd.DataFrame.to_csv

        def capturing_to_csv(self_df, path_or_buf, **kwargs):
            captured_paths.append(Path(path_or_buf))
            # Don't actually write; just record the path
            return None

        with (
            patch.object(ee_mod, "_load_sites", return_value=fake_sites_df),
            patch.object(ee_mod, "fetch_site_series", return_value=site_df),
            patch("pandas.DataFrame.to_csv", capturing_to_csv),
            patch("pathlib.Path.mkdir"),
        ):
            ee_mod.run_pipeline()

        assert all("data" in str(p) and "raw" in str(p) for p in captured_paths)
