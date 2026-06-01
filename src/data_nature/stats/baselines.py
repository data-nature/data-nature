"""
baselines.py — Per-site, per-month rolling baseline computation.

Computes rolling mean, std, and count of LST (and NDVI) for each
(site, month) pair using all years up to but not including the
current year (leave-one-out expanding window).

Output schema (site_baselines.csv):
    site, month, baseline_mean, baseline_std, baseline_count,
    ndvi_baseline_mean, ndvi_baseline_std
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
MOCK_DIR = _REPO_ROOT / "data" / "mock"
BASELINES_PATH = PROCESSED_DIR / "site_baselines.csv"

# ---------------------------------------------------------------------------
# Data contract — frozen schemas from the sprint plan
# ---------------------------------------------------------------------------
DATA_CONTRACT: dict[str, list[str]] = {
    "site_monthly.csv": [
        "year", "month", "site", "lst", "ndvi",
        "z_score_lst", "z_score_ndvi", "delta", "is_anomaly",
    ],
    "lst_timeseries.csv": [
        "date", "site", "lst", "baseline_mean", "baseline_std",
    ],
    "lst_history.csv": ["date", "site", "lst"],
    "site_locations.csv": ["site", "lat", "lng", "land_cover"],
    "anomalies.csv": [
        "date", "site", "lst", "baseline", "z_score",
        "severity", "status", "ndvi_change",
    ],
    "lst_forecast.csv": [
        "date", "site", "model", "lst_forecast", "lst_low", "lst_high",
    ],
    "model_metrics.csv": ["site", "model", "mae", "rmse", "r2"],
}

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_schema(data_dir: Path | None = None) -> dict[str, list[str]]:
    """
    Validate every processed CSV against the frozen data contract.

    Checks only files that are present in *data_dir* (so missing optional
    files don't fail validation — they are simply skipped with a warning).

    Parameters
    ----------
    data_dir:
        Directory to check. Defaults to ``data/processed/``.

    Returns
    -------
    dict
        ``{"ok": [...filenames], "missing_cols": {filename: [col, ...]},
           "not_found": [...filenames]}``

    Raises
    ------
    ValueError
        If any present file is missing required columns.
    """
    data_dir = Path(data_dir) if data_dir else PROCESSED_DIR
    results: dict[str, list[str]] = {"ok": [], "missing_cols": {}, "not_found": []}

    for filename, required_cols in DATA_CONTRACT.items():
        path = data_dir / filename
        if not path.exists():
            logger.warning("Schema check: %s not found in %s — skipping.", filename, data_dir)
            results["not_found"].append(filename)
            continue

        df = pd.read_csv(path, nrows=0)  # header only
        actual_cols = list(df.columns)
        missing = [c for c in required_cols if c not in actual_cols]

        if missing:
            results["missing_cols"][filename] = missing
            logger.error(
                "Schema violation in %s — missing columns: %s", filename, missing
            )
        else:
            results["ok"].append(filename)
            logger.info("Schema OK: %s", filename)

    if results["missing_cols"]:
        raise ValueError(
            f"Schema validation failed. Missing columns: {results['missing_cols']}"
        )

    return results


# ---------------------------------------------------------------------------
# Baseline computation
# ---------------------------------------------------------------------------

def compute_baselines(
    df: pd.DataFrame,
    min_years: int = 3,
) -> pd.DataFrame:
    """
    Compute per-site, per-month rolling baselines from a site_monthly DataFrame.

    Uses an **expanding leave-one-out window**: for a given (site, month, year),
    the baseline is the mean/std of *all prior years only* — so the current year
    never contaminates its own baseline (essential for z-score validity).

    Parameters
    ----------
    df:
        DataFrame with at minimum columns: year, month, site, lst, ndvi.
        Typically ``data/processed/site_monthly.csv``.
    min_years:
        Minimum number of prior-year observations required before a baseline
        is considered valid.  Rows with fewer prior observations get NaN
        baseline values.

    Returns
    -------
    pd.DataFrame
        Columns: site, month, year, baseline_mean, baseline_std,
        baseline_count, ndvi_baseline_mean, ndvi_baseline_std

        One row per (site, month, year) — the baseline that would be used
        when assessing that year's observation.

    Notes
    -----
    * For the z-score engine (DN-A2), the relevant columns are
      ``baseline_mean`` and ``baseline_std`` at the (site, month) level.
    * ``baseline_count`` lets downstream code filter out low-data early years.
    """
    required = {"year", "month", "site", "lst", "ndvi"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"compute_baselines: input DataFrame missing columns: {missing}")

    df = df.copy()
    df = df.sort_values(["site", "month", "year"]).reset_index(drop=True)

    records = []

    for (site, month), group in df.groupby(["site", "month"], sort=True):
        group = group.sort_values("year").reset_index(drop=True)

        lst_vals = group["lst"].to_numpy()
        ndvi_vals = group["ndvi"].to_numpy()
        years = group["year"].to_numpy()

        for i, year in enumerate(years):
            # All years strictly before this one
            prior_lst = lst_vals[:i]
            prior_ndvi = ndvi_vals[:i]
            count = len(prior_lst)

            if count < min_years:
                b_mean = b_std = ndvi_mean = ndvi_std = float("nan")
            else:
                b_mean = float(prior_lst.mean())
                b_std = float(prior_lst.std(ddof=1)) if count > 1 else float("nan")
                ndvi_mean = float(prior_ndvi.mean())
                ndvi_std = float(prior_ndvi.std(ddof=1)) if count > 1 else float("nan")

            records.append(
                {
                    "site": site,
                    "month": int(month),
                    "year": int(year),
                    "baseline_mean": b_mean,
                    "baseline_std": b_std,
                    "baseline_count": count,
                    "ndvi_baseline_mean": ndvi_mean,
                    "ndvi_baseline_std": ndvi_std,
                }
            )

    result = pd.DataFrame(records)
    logger.info(
        "compute_baselines: produced %d rows for %d sites × %d months.",
        len(result),
        result["site"].nunique(),
        result["month"].nunique(),
    )
    return result


# ---------------------------------------------------------------------------
# Convenience: load real data (with mock fallback) and save baselines
# ---------------------------------------------------------------------------

def load_site_monthly(data_dir: Path | None = None) -> pd.DataFrame:
    """
    Load site_monthly.csv from *data_dir*, falling back to mock data.

    Parameters
    ----------
    data_dir:
        Directory to try first.  Defaults to ``data/processed/``.

    Returns
    -------
    pd.DataFrame
    """
    data_dir = Path(data_dir) if data_dir else PROCESSED_DIR
    path = data_dir / "site_monthly.csv"

    if path.exists():
        logger.info("Loading site_monthly from %s", path)
        return pd.read_csv(path)

    mock_path = MOCK_DIR / "site_monthly.csv"
    if mock_path.exists():
        logger.warning(
            "Real site_monthly.csv not found — falling back to mock data at %s", mock_path
        )
        return pd.read_csv(mock_path)

    raise FileNotFoundError(
        f"site_monthly.csv not found in {data_dir} or mock fallback {MOCK_DIR}."
    )


def build_and_save_baselines(
    data_dir: Path | None = None,
    output_path: Path | None = None,
    min_years: int = 3,
) -> pd.DataFrame:
    """
    End-to-end helper: load site_monthly, compute baselines, save CSV.

    Parameters
    ----------
    data_dir:
        Source directory for site_monthly.csv.
    output_path:
        Where to write site_baselines.csv.  Defaults to
        ``data/processed/site_baselines.csv``.
    min_years:
        Passed through to :func:`compute_baselines`.

    Returns
    -------
    pd.DataFrame
        The baselines DataFrame (also written to disk).
    """
    output_path = Path(output_path) if output_path else BASELINES_PATH
    df = load_site_monthly(data_dir)
    baselines = compute_baselines(df, min_years=min_years)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    baselines.to_csv(output_path, index=False)
    logger.info("Baselines saved to %s (%d rows).", output_path, len(baselines))
    return baselines


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # 1. Validate schema
    print("── Schema validation ──────────────────────────────")
    try:
        report = validate_schema()
        print(f"  OK       : {report['ok']}")
        print(f"  Not found: {report['not_found']}")
        print("  All present files pass schema validation.\n")
    except ValueError as exc:
        print(f"  FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

    # 2. Build + save baselines
    print("── Building baselines ─────────────────────────────")
    bl = build_and_save_baselines()
    print(bl.head(10).to_string(index=False))
    print(f"\n  Total rows : {len(bl)}")
    print(f"  Saved to   : {BASELINES_PATH}")
