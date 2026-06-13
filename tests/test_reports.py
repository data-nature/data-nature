"""
tests/test_reports.py — DN-B6 tests for the PDF report generator
=================================================================
All tests run on synthetic data — no processed CSVs required.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from data_nature.reports.generator import build_pdf, conclusions, make_chart_png


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _synthetic_anom(n: int = 20) -> pd.DataFrame:
    """Minimal synthetic anomalies DataFrame."""
    sites    = ["Site_A", "Site_B", "Site_C"]
    sevs     = ["Critical", "Severe", "Mild"]
    rows = []
    for i in range(n):
        rows.append({
            "date":        pd.Timestamp(f"2025-{(i % 12) + 1:02d}-01"),
            "site":        sites[i % 3],
            "lst":         30.0 + i * 0.5,
            "baseline":    28.0,
            "z_score":     2.0 + (i % 3) * 0.5,
            "severity":    sevs[i % 3],
            "status":      "New",
            "ndvi_change": -0.05 + i * 0.002,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def anom_df():
    return _synthetic_anom()


@pytest.fixture
def empty_df():
    return _synthetic_anom(0)


def _pdf_kwargs(df, **overrides):
    defaults = dict(
        rpt_type="🚨 Anomaly Report",
        sites=["Site_A", "Site_B"],
        d0=date(2025, 1, 1),
        d1=date(2025, 12, 31),
        researcher="Test Researcher",
        org="Test Org",
        notes="",
        df=df,
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# conclusions()
# ---------------------------------------------------------------------------

class TestConclusions:
    def test_empty_df_returns_no_anomaly_message(self, empty_df):
        result = conclusions(empty_df, ["Site_A"])
        assert "No thermal anomalies" in result

    def test_non_empty_includes_count(self, anom_df):
        result = conclusions(anom_df, ["Site_A", "Site_B"])
        assert str(len(anom_df)) in result

    def test_includes_site_count(self, anom_df):
        sites = ["Site_A", "Site_B"]
        result = conclusions(anom_df, sites)
        assert str(len(sites)) in result

    def test_mentions_critical_when_present(self, anom_df):
        result = conclusions(anom_df, ["Site_A"])
        assert "Critical" in result

    def test_returns_string(self, anom_df):
        assert isinstance(conclusions(anom_df, ["Site_A"]), str)

    def test_non_empty_result(self, anom_df):
        assert len(conclusions(anom_df, ["Site_A"])) > 0


# ---------------------------------------------------------------------------
# make_chart_png()
# ---------------------------------------------------------------------------

class TestMakeChartPng:
    def test_returns_bytes_io(self, anom_df):
        import io
        buf = make_chart_png(anom_df, ["Site_A", "Site_B"])
        assert isinstance(buf, io.BytesIO)

    def test_position_at_zero(self, anom_df):
        buf = make_chart_png(anom_df, ["Site_A"])
        assert buf.tell() == 0

    def test_non_empty_bytes(self, anom_df):
        buf = make_chart_png(anom_df, ["Site_A"])
        content = buf.read()
        assert len(content) > 0

    def test_is_valid_png(self, anom_df):
        buf = make_chart_png(anom_df, ["Site_A"])
        header = buf.read(8)
        assert header[:4] == b"\x89PNG", "Output is not a valid PNG"

    def test_works_with_empty_df(self, empty_df):
        """Empty DataFrame should still produce a valid PNG (no-data message)."""
        import io
        buf = make_chart_png(empty_df, ["Site_A"])
        assert isinstance(buf, io.BytesIO)
        buf.seek(0)
        assert buf.read(4) == b"\x89PNG"


# ---------------------------------------------------------------------------
# build_pdf()
# ---------------------------------------------------------------------------

class TestBuildPdf:
    def test_returns_bytes(self, anom_df):
        result = build_pdf(**_pdf_kwargs(anom_df))
        assert isinstance(result, bytes)

    def test_non_empty_output(self, anom_df):
        result = build_pdf(**_pdf_kwargs(anom_df))
        assert len(result) > 1000, "PDF is suspiciously small"

    def test_is_valid_pdf(self, anom_df):
        result = build_pdf(**_pdf_kwargs(anom_df))
        assert result[:4] == b"%PDF", "Output does not start with PDF magic bytes"

    def test_works_with_empty_anomalies(self, empty_df):
        """Should still produce a valid PDF when no anomalies match the filter."""
        result = build_pdf(**_pdf_kwargs(empty_df))
        assert result[:4] == b"%PDF"

    def test_works_with_researcher_notes(self, anom_df):
        result = build_pdf(**_pdf_kwargs(anom_df, notes="Important field note."))
        assert isinstance(result, bytes)
        assert result[:4] == b"%PDF"

    def test_all_report_types(self, anom_df):
        for rpt in ["📅 Weekly Summary", "📆 Monthly Summary",
                    "🚨 Anomaly Report", "🌱 Optimization Report"]:
            result = build_pdf(**_pdf_kwargs(anom_df, rpt_type=rpt))
            assert result[:4] == b"%PDF", f"Invalid PDF for report type: {rpt}"

    def test_single_site(self, anom_df):
        result = build_pdf(**_pdf_kwargs(anom_df, sites=["Site_A"]))
        assert result[:4] == b"%PDF"

    def test_many_sites(self, anom_df):
        result = build_pdf(**_pdf_kwargs(anom_df, sites=["Site_A", "Site_B", "Site_C"]))
        assert result[:4] == b"%PDF"

    def test_size_reasonable(self, anom_df):
        """PDF should be under 5 MB for typical report."""
        result = build_pdf(**_pdf_kwargs(anom_df))
        assert len(result) < 5 * 1024 * 1024, "PDF is unexpectedly large"