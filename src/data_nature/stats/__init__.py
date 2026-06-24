from .anomaly import BASELINE_END, BASELINE_START, DEFAULT_THRESHOLDS, compute_zscores, detect_anomalies

try:
    from .regression import compare_site_types, fit_ndvi_lst
    _REGRESSION_AVAILABLE = True
except ImportError:
    _REGRESSION_AVAILABLE = False

__all__ = [
    "compute_zscores",
    "detect_anomalies",
    "DEFAULT_THRESHOLDS",
    "BASELINE_START",
    "BASELINE_END",
    "fit_ndvi_lst",
    "compare_site_types",
]
