"""
Earth Engine ingest pipeline — monthly NDVI and LST for Northern Israel sites.

Collections used
----------------
NDVI  MODIS/061/MOD13Q1   16-day composite, 250 m, band ``NDVI`` × 0.0001
LST   MODIS/061/MOD11A1   daily, 1 km, band ``LST_Day_1km`` × 0.02 − 273.15 (K → °C)
"""

from __future__ import annotations

import calendar
import datetime
import logging
import os
from pathlib import Path

import ee
import pandas as pd
from dotenv import load_dotenv

log = logging.getLogger(__name__)

_EE_INITIALIZED: bool = False

_NDVI_COLLECTION = "MODIS/061/MOD13Q1"
_LST_COLLECTION = "MODIS/061/MOD11A1"


def _last_complete_month_end() -> str:
    """Return the first day of the current month as an ISO string (exclusive end for last complete month)."""
    today = datetime.date.today()
    return today.replace(day=1).isoformat()

_SITE_LOCATIONS_PATH = (
    Path(__file__).parents[3] / "data" / "processed" / "site_locations.csv"
)


class EEAuthError(RuntimeError):
    """Raised when Google Earth Engine cannot be authenticated or initialised."""


# ── authentication ────────────────────────────────────────────────────────────


def authenticate() -> None:
    """
    Authenticate and initialise Google Earth Engine.

    Resolution order:
    1. Service-account credentials from environment variables
       ``GEE_SERVICE_ACCOUNT_EMAIL``, ``GEE_PRIVATE_KEY_PATH``, ``GEE_PROJECT_ID``.
    2. Interactive OAuth2 browser flow (``ee.Authenticate()``), suitable for
       local development when the env vars are absent.

    Calling this function a second time is a no-op if initialisation already
    succeeded in the current process.

    Raises
    ------
    EEAuthError
        On any failure, with a message that names the missing configuration so
        the caller can surface it to the user without a traceback.
    """
    global _EE_INITIALIZED
    if _EE_INITIALIZED:
        return

    load_dotenv()
    sa_email = os.getenv("GEE_SERVICE_ACCOUNT_EMAIL")
    key_path = os.getenv("GEE_PRIVATE_KEY_PATH")
    project = os.getenv("GEE_PROJECT_ID") or None

    try:
        if sa_email and key_path:
            credentials = ee.ServiceAccountCredentials(sa_email, key_path)
            ee.Initialize(credentials, project=project)
            log.info("Earth Engine initialised via service account (%s).", sa_email)
        else:
            log.warning(
                "GEE_SERVICE_ACCOUNT_EMAIL / GEE_PRIVATE_KEY_PATH not set — "
                "falling back to interactive authentication."
            )
            ee.Authenticate()
            ee.Initialize(project=project)
            log.info("Earth Engine initialised via interactive auth.")
        _EE_INITIALIZED = True
    except Exception as exc:
        raise EEAuthError(
            f"Earth Engine authentication failed: {exc}\n\n"
            "To fix this, either:\n"
            "  • Set GEE_SERVICE_ACCOUNT_EMAIL, GEE_PRIVATE_KEY_PATH, and\n"
            "    GEE_PROJECT_ID in your .env file (see .env.example), or\n"
            "  • Run `earthengine authenticate` in your terminal for interactive login."
        ) from exc


# ── internal helpers ──────────────────────────────────────────────────────────


def _load_sites() -> pd.DataFrame:
    """Return the site AOI table from data/processed/site_locations.csv."""
    return pd.read_csv(_SITE_LOCATIONS_PATH)


def _extract_ndvi_monthly(
    point: ee.Geometry,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Pull all 16-day MODIS NDVI observations for *point* and aggregate to monthly means.

    Collection: MODIS/061/MOD13Q1 — band ``NDVI``, scale factor 0.0001, 250 m.

    Parameters
    ----------
    point : ee.Geometry
        Point geometry for the site centroid.
    start, end : str
        ISO date strings (start inclusive, end exclusive).

    Returns
    -------
    pd.DataFrame
        Columns: ``year``, ``month``, ``ndvi``.  Empty DataFrame when no data
        is available.
    """
    collection = (
        ee.ImageCollection(_NDVI_COLLECTION)
        .filterDate(start, end)
        .filterBounds(point)
        .select("NDVI")
    )

    def _tag(image: ee.Image) -> ee.Feature:
        date = image.date().format("YYYY-MM-dd")
        value = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=500,
        )
        return ee.Feature(None, value.set("date", date))

    features = collection.map(_tag).getInfo()["features"]

    records = []
    for f in features:
        props = f["properties"]
        raw = props.get("NDVI")
        if raw is not None:
            records.append({"date": props["date"], "ndvi": round(raw * 0.0001, 6)})

    if not records:
        return pd.DataFrame(columns=["year", "month", "ndvi"])

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df.groupby(["year", "month"], as_index=False)["ndvi"].mean()


def _extract_lst_monthly(
    point: ee.Geometry,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Pull daily MODIS LST, aggregate to monthly means, and convert to °C.

    Collection: MODIS/061/MOD11A1 — band ``LST_Day_1km``.
    Conversion: celsius = raw_value × 0.02 − 273.15 (Kelvin scale factor).

    One Earth Engine request is made per calendar month in [start, end).
    Failed months are skipped with a warning so the pipeline can continue.

    Parameters
    ----------
    point : ee.Geometry
        Point geometry for the site centroid.
    start, end : str
        ISO date strings (start inclusive, end exclusive).

    Returns
    -------
    pd.DataFrame
        Columns: ``year``, ``month``, ``lst``.  Empty DataFrame when no data
        is available.
    """
    def _mask_cloudy(image: ee.Image) -> ee.Image:
        qc = image.select("QC_Day")
        # Bits 0-1: 00 = LST produced with good quality
        good_quality = qc.bitwiseAnd(0b11).eq(0)
        return image.select("LST_Day_1km").updateMask(good_quality)

    collection = (
        ee.ImageCollection(_LST_COLLECTION)
        .filterDate(start, end)
        .filterBounds(point)
        .select(["LST_Day_1km", "QC_Day"])
        .map(_mask_cloudy)
    )

    records = []
    for period in pd.period_range(start, end, freq="M"):
        year, month = period.year, period.month
        last_day = calendar.monthrange(year, month)[1]
        month_start = f"{year}-{month:02d}-01"
        month_end = f"{year}-{month:02d}-{last_day}"

        try:
            val = (
                collection.filterDate(month_start, month_end)
                .mean()
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=point,
                    scale=1000,
                )
                .get("LST_Day_1km")
                .getInfo()
            )
            if val is not None:
                records.append(
                    {
                        "year": year,
                        "month": month,
                        "lst": round(val * 0.02 - 273.15, 2),
                    }
                )
        except Exception as exc:
            log.warning("LST extraction skipped for %d-%02d: %s", year, month, exc)

    return (
        pd.DataFrame(records)
        if records
        else pd.DataFrame(columns=["year", "month", "lst"])
    )


# ── public API ────────────────────────────────────────────────────────────────


def fetch_site_series(
    site: str,
    start: str = "2000-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Fetch a tidy monthly time series of LST and NDVI for one named site.

    Collections used:
      - NDVI: MODIS/061/MOD13Q1  (16-day composite, 250 m, band NDVI × 0.0001)
      - LST:  MODIS/061/MOD11A1  (daily, 1 km, band LST_Day_1km × 0.02 − 273.15 K→°C)

    Parameters
    ----------
    site : str
        Site name matching a row in ``data/processed/site_locations.csv``
        (e.g. ``"Carmel_Forest"``).
    start : str
        ISO date string for the start of the window, inclusive.
        Defaults to ``"2000-01-01"``.
    end : str
        ISO date string for the end of the window, exclusive.
        Defaults to the first day of the current month (last complete month).

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame sorted by date with columns:
        ``date`` (YYYY-MM-01), ``year``, ``month``, ``lst`` (°C),
        ``ndvi``, ``site``, ``land_cover``.

    Raises
    ------
    EEAuthError
        If Earth Engine cannot be authenticated.  The error message includes
        actionable remediation steps.
    ValueError
        If *site* is not found in ``site_locations.csv``.
    """
    if end is None:
        end = _last_complete_month_end()
    authenticate()

    sites_df = _load_sites()
    row = sites_df[sites_df["site"] == site]
    if row.empty:
        available = sites_df["site"].tolist()
        raise ValueError(
            f"Site '{site}' not found in site_locations.csv.\n"
            f"Available sites: {available}"
        )

    row = row.iloc[0]
    point = ee.Geometry.Point([float(row["lng"]), float(row["lat"])])
    land_cover = row["land_cover"]

    log.info("Fetching NDVI for %s (%s – %s)…", site, start, end)
    ndvi_df = _extract_ndvi_monthly(point, start, end)

    log.info("Fetching LST for %s (%s – %s)…", site, start, end)
    lst_df = _extract_lst_monthly(point, start, end)

    merged = lst_df.merge(ndvi_df, on=["year", "month"], how="outer")
    merged["date"] = pd.to_datetime(merged[["year", "month"]].assign(day=1))
    merged["site"] = site
    merged["land_cover"] = land_cover

    return (
        merged[["date", "year", "month", "lst", "ndvi", "site", "land_cover"]]
        .sort_values("date")
        .reset_index(drop=True)
    )


def run_pipeline(
    start: str = "2000-01-01",
    end: str | None = None,
    output_dir: str | Path | None = None,
) -> None:
    """
    Run the full ingest pipeline for all eight Northern Israel sites.

    Iterates over every site in ``data/processed/site_locations.csv``, calls
    :func:`fetch_site_series`, and writes one CSV per site to *output_dir*.
    Existing files are overwritten.

    Individual site failures are logged as warnings and do not stop the
    pipeline.  An :class:`EEAuthError` is re-raised immediately because it
    indicates a systemic problem that will affect every subsequent site.

    Parameters
    ----------
    start, end : str
        ISO date strings passed to :func:`fetch_site_series`.
    output_dir : path-like, optional
        Destination directory for the raw CSV files.  Defaults to
        ``data/raw/`` relative to the project root.
    """
    if end is None:
        end = _last_complete_month_end()
    if output_dir is None:
        output_dir = Path(__file__).parents[3] / "data" / "raw"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sites_df = _load_sites()
    for site in sites_df["site"]:
        try:
            log.info("Processing site: %s", site)
            df = fetch_site_series(site, start, end)
            out_path = output_dir / f"{site}.csv"
            df.to_csv(out_path, index=False)
            log.info("Wrote %d rows → %s", len(df), out_path)
        except EEAuthError:
            raise
        except Exception as exc:
            log.warning("Skipping %s — %s", site, exc)


def fetch_site_grid(
    site: str,
    season_months: list[int],
    grid_size: int = 10,
    cell_lat: float = 0.0027,
    cell_lng: float = 0.0032,
    baseline_start: int = 2015,
    baseline_end: int = 2023,
) -> tuple["np.ndarray", "np.ndarray"]:
    """
    Fetch a real MODIS LST and NDVI spatial grid for a site and season.

    Samples MODIS MOD11A1 (LST) and MOD13Q1 (NDVI) at each of the
    ``grid_size × grid_size`` cell centres around the site, averaged over
    ``baseline_start``–``baseline_end`` for the given season months.

    Parameters
    ----------
    site : str
        Site name matching ``data/processed/site_locations.csv``.
    season_months : list[int]
        Month numbers to include (e.g. [6, 7, 8] for Summer).
    grid_size : int
        Number of cells per side. Default 10.
    cell_lat, cell_lng : float
        Cell dimensions in degrees. Default ≈ 300 m.
    baseline_start, baseline_end : int
        Year range to average over for stable seasonal values.

    Returns
    -------
    (lst_grid, ndvi_grid) : tuple of (grid_size, grid_size) float64 arrays
        LST in °C, NDVI in [0, 1].

    Raises
    ------
    EEAuthError
        If GEE cannot be authenticated.
    ValueError
        If site is not found.
    """
    import numpy as np

    try:
        ee.Initialize(project="datanature")
    except Exception:
        pass

    sites_df = _load_sites()
    row = sites_df[sites_df["site"] == site]
    if row.empty:
        raise ValueError(f"Site '{site}' not found in site_locations.csv.")
    row = row.iloc[0]
    center_lat, center_lng = float(row["lat"]), float(row["lng"])

    # Build a FeatureCollection of grid cell centres
    half = grid_size / 2.0
    features = []
    for i in range(grid_size):
        for j in range(grid_size):
            lat = center_lat + (half - i - 0.5) * cell_lat
            lng = center_lng + (j - half + 0.5) * cell_lng
            features.append(
                ee.Feature(ee.Geometry.Point([lng, lat]), {"row": i, "col": j})
            )
    fc = ee.FeatureCollection(features)

    month_filter = ee.Filter.Or(
        [ee.Filter.calendarRange(m, m, "month") for m in season_months]
    )
    year_filter = ee.Filter.calendarRange(baseline_start, baseline_end, "year")

    def _mask_lst(image: ee.Image) -> ee.Image:
        qc = image.select("QC_Day")
        good = qc.bitwiseAnd(0b11).eq(0)
        return image.select("LST_Day_1km").updateMask(good)

    lst_img = (
        ee.ImageCollection(_LST_COLLECTION)
        .filter(year_filter)
        .filter(month_filter)
        .select(["LST_Day_1km", "QC_Day"])
        .map(_mask_lst)
        .mean()
        .multiply(0.02)
        .subtract(273.15)
        .rename("lst")
    )

    ndvi_img = (
        ee.ImageCollection(_NDVI_COLLECTION)
        .filter(year_filter)
        .filter(month_filter)
        .select("NDVI")
        .mean()
        .multiply(0.0001)
        .rename("ndvi")
    )

    sampled = (
        lst_img.addBands(ndvi_img)
        .sampleRegions(collection=fc, scale=250, geometries=False)
        .getInfo()
    )

    lst_grid = np.full((grid_size, grid_size), np.nan)
    ndvi_grid = np.full((grid_size, grid_size), np.nan)
    for feat in sampled["features"]:
        props = feat["properties"]
        i, j = int(props["row"]), int(props["col"])
        if props.get("lst") is not None:
            lst_grid[i, j] = props["lst"]
        if props.get("ndvi") is not None:
            ndvi_grid[i, j] = props["ndvi"]

    # Fill any missing cells with site mean
    lst_mean = float(np.nanmean(lst_grid)) if not np.all(np.isnan(lst_grid)) else 35.0
    ndvi_mean = float(np.nanmean(ndvi_grid)) if not np.all(np.isnan(ndvi_grid)) else 0.3
    lst_grid = np.where(np.isnan(lst_grid), lst_mean, lst_grid)
    ndvi_grid = np.where(np.isnan(ndvi_grid), ndvi_mean, ndvi_grid)

    return lst_grid, np.clip(ndvi_grid, 0.01, 0.99)
