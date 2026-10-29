"""Unit tests for the `calculate` tool (no DB required)."""

import json

import pytest

from app.agent.tools import calculate


class TestCalculateTool:
    def _call(self, operation, values, labels=None):
        # The @tool decorator wraps the function; call the underlying function
        raw = calculate.invoke({"operation": operation, "values": values, "labels": labels})
        return json.loads(raw)

    def test_yoy_growth(self):
        result = self._call(
            "yoy_growth",
            [100.0, 112.0, 130.0],
            ["FY2022", "FY2023", "FY2024"],
        )
        changes = result["changes"]
        assert len(changes) == 2
        assert changes[0]["pct_change"] == pytest.approx(12.0)
        assert changes[1]["pct_change"] == pytest.approx(16.07, abs=0.1)

    def test_cagr(self):
        result = self._call("cagr", [100.0, 125.0, 156.25], ["Y1", "Y2", "Y3"])
        assert result["cagr_pct"] == pytest.approx(25.0, abs=0.01)
        assert result["periods"] == 2

    def test_margin(self):
        result = self._call("margin", [24_311.0, 85_777.0])
        assert result["margin_pct"] == pytest.approx(28.34, abs=0.1)

    def test_delta(self):
        result = self._call("delta", [80_000.0, 85_777.0], ["Q2 2024", "Q3 2024"])
        assert result["absolute_change"] == pytest.approx(5777.0)
        assert result["pct_change"] == pytest.approx(7.22, abs=0.1)
        assert result["from_label"] == "Q2 2024"

    def test_mean(self):
        result = self._call("mean", [10.0, 20.0, 30.0])
        assert result["mean"] == 20.0
        assert result["min"] == 10.0
        assert result["max"] == 30.0

    def test_median_odd(self):
        result = self._call("median", [1.0, 3.0, 5.0])
        assert result["median"] == 3.0

    def test_median_even(self):
        result = self._call("median", [1.0, 2.0, 3.0, 4.0])
        assert result["median"] == 2.5

    def test_zero_denominator_in_cagr(self):
        result = self._call("cagr", [0.0, 100.0])
        assert result["cagr_pct"] is None

    def test_empty_values(self):
        result = self._call("mean", [])
        assert "error" in result

    def test_unsupported_operation(self):
        result = self._call("nonsense", [1.0, 2.0])
        assert "error" in result
