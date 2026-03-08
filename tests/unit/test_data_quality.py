"""Tests for data quality validation layer."""

import numpy as np
import pandas as pd
import pytest

from finpredict.data.quality import DataQualityChecker


def _make_clean_data(n=100):
    """Create clean synthetic market data that passes all checks."""
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "SP500": 3000 + np.cumsum(rng.normal(0.5, 5, n)),
        "VIX": 20 + rng.normal(0, 2, n),
        "T10Y": 2.0 + rng.normal(0, 0.1, n),
        "T3M": 1.5 + rng.normal(0, 0.05, n),
    }, index=dates)


class TestDataQualityChecker:

    def test_clean_data_passes(self):
        checker = DataQualityChecker()
        data = _make_clean_data()
        warnings = checker.validate(data)
        # Clean data should have no warnings (except possibly range info for SP500)
        critical = [w for w in warnings if w["severity"] in ("warning", "error")]
        assert len(critical) == 0

    def test_staleness_detected(self):
        checker = DataQualityChecker()
        data = _make_clean_data(n=100)
        # Make VIX stale: set last 10 values to NaN
        data.loc[data.index[-10:], "VIX"] = np.nan
        warnings = checker.validate(data)
        stale = [w for w in warnings if w["check"] == "staleness" and w["column"] == "VIX"]
        assert len(stale) >= 1

    def test_range_violation_vix(self):
        checker = DataQualityChecker()
        data = _make_clean_data()
        # Set some VIX values out of range
        data.iloc[10, data.columns.get_loc("VIX")] = 95.0
        data.iloc[20, data.columns.get_loc("VIX")] = 3.0
        warnings = checker.validate(data)
        vix_range = [w for w in warnings if w["check"] == "range" and w["column"] == "VIX"]
        assert len(vix_range) >= 1

    def test_completeness_threshold(self):
        checker = DataQualityChecker()
        data = _make_clean_data(n=100)
        # Set 30% of T10Y to NaN (above 20% threshold)
        data.loc[data.index[:30], "T10Y"] = np.nan
        warnings = checker.validate(data)
        completeness = [w for w in warnings if w["check"] == "completeness" and w["column"] == "T10Y"]
        assert len(completeness) >= 1

    def test_consistency_jump(self):
        checker = DataQualityChecker()
        data = _make_clean_data()
        # Insert a 50% jump
        data.iloc[50, data.columns.get_loc("SP500")] = data.iloc[49, data.columns.get_loc("SP500")] * 1.6
        warnings = checker.validate(data)
        consistency = [w for w in warnings if w["check"] == "consistency"]
        assert len(consistency) >= 1

    def test_returns_list_of_dicts(self):
        checker = DataQualityChecker()
        data = _make_clean_data()
        warnings = checker.validate(data)
        assert isinstance(warnings, list)
        for w in warnings:
            assert isinstance(w, dict)
            assert "check" in w
            assert "message" in w
            assert "severity" in w
