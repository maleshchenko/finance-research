"""
Unit tests for the PDF extraction layer.

These tests use in-memory fixtures — no real PDFs required.
"""

import pytest

from app.ingestion.extractor import (
    _detect_heading,
    _extract_metrics_from_table,
    _parse_numeric,
)


# ── _parse_numeric ────────────────────────────────────────────────────────────

class TestParseNumeric:
    def test_plain_integer(self):
        assert _parse_numeric("12345") == 12345.0

    def test_comma_separated(self):
        assert _parse_numeric("85,777") == 85777.0

    def test_negative_parentheses(self):
        assert _parse_numeric("(1,234)") == -1234.0

    def test_decimal(self):
        assert _parse_numeric("28.4") == pytest.approx(28.4)

    def test_millions_suffix(self):
        assert _parse_numeric("1.5M") == pytest.approx(1.5)

    def test_billions_suffix(self):
        # B → multiply by 1000 (stored in millions)
        assert _parse_numeric("2B") == pytest.approx(2000.0)

    def test_empty_string(self):
        assert _parse_numeric("") is None

    def test_non_numeric(self):
        assert _parse_numeric("N/A") is None

    def test_percent_stripped(self):
        assert _parse_numeric("28.4%") == pytest.approx(28.4)


# ── _extract_metrics_from_table ────────────────────────────────────────────────

class TestExtractMetrics:
    def _table(self, rows):
        return rows

    def test_recognises_revenue_row(self):
        table = [
            ["", "Q3 2024", "Q2 2024", "Q1 2024"],
            ["Total Revenue", "85,777", "90,753", "94,836"],
            ["Net Income", "21,448", "20,000", "19,500"],
        ]
        metrics = _extract_metrics_from_table(table, page_number=5)
        names = {m.metric_name for m in metrics}
        assert "revenue" in names
        assert "net_income" in names

    def test_first_numeric_column_only(self):
        table = [
            ["", "Q3 2024", "Q2 2024"],
            ["Total Revenue", "85,777", "90,753"],
        ]
        metrics = _extract_metrics_from_table(table, page_number=5)
        assert len(metrics) == 1
        assert metrics[0].value == 85777.0

    def test_empty_table_returns_empty(self):
        assert _extract_metrics_from_table([], page_number=1) == []

    def test_unrecognised_rows_skipped(self):
        table = [
            ["Headcount", "165,000", "162,000"],
            ["Unknown Metric", "999", "888"],
        ]
        metrics = _extract_metrics_from_table(table, page_number=3)
        assert metrics == []

    def test_operating_margin_percent_conversion(self):
        # Value expressed as decimal fraction → should be converted to percent
        table = [
            ["", "FY2024"],
            ["Operating Margin", "0.284"],
        ]
        metrics = _extract_metrics_from_table(table, page_number=10)
        margin = next((m for m in metrics if m.metric_name == "operating_margin"), None)
        assert margin is not None
        assert margin.value == pytest.approx(28.4)


# ── _detect_heading ───────────────────────────────────────────────────────────

class TestDetectHeading:
    def test_mda_heading(self):
        text = "ITEM 7. Management's Discussion and Analysis of Financial Condition"
        assert _detect_heading(text) is not None

    def test_risk_factors(self):
        text = "\nITEM 1A. Risk Factors\n"
        assert _detect_heading(text) is not None

    def test_no_heading(self):
        text = "Revenue grew 12% year-over-year driven by strong iPhone sales."
        assert _detect_heading(text) is None

    def test_results_of_operations(self):
        text = "Results of Operations\nThe following table summarises..."
        heading = _detect_heading(text)
        assert heading is not None
        assert "Results" in heading
