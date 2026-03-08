"""Tests for global crash dataset and contagion features."""

import numpy as np
import pandas as pd
import pytest

from finpredict.data.global_crashes import (
    identify_global_crashes,
    compute_contagion_features,
)


def _make_index(name, start="2000-01-01", n=500, crash_at=200, crash_depth=0.30):
    """Create synthetic price series with optional crash."""
    dates = pd.bdate_range(start, periods=n)
    prices = np.ones(n) * 100.0
    # Gradual rise then crash
    for i in range(1, n):
        if crash_at <= i < crash_at + 60:
            prices[i] = prices[i - 1] * (1 - crash_depth / 60)
        else:
            prices[i] = prices[i - 1] * 1.001
    return pd.Series(prices, index=dates, name=name)


class TestIdentifyGlobalCrashes:

    def test_identifies_crash(self):
        """Synthetic drawdown should be detected."""
        idx = _make_index("TestIdx", crash_at=200, crash_depth=0.30)
        index_data = {"TestIdx": idx}
        crashes = identify_global_crashes(index_data, threshold=0.20)
        assert len(crashes) >= 1
        assert crashes.iloc[0]["market"] == "TestIdx"
        assert crashes.iloc[0]["max_dd"] < -0.20

    def test_no_crash_below_threshold(self):
        """Small drawdown should not trigger crash detection."""
        idx = _make_index("SmallDip", crash_at=200, crash_depth=0.08)
        crashes = identify_global_crashes({"SmallDip": idx}, threshold=0.20)
        assert len(crashes) == 0

    def test_multiple_markets(self):
        """Crashes across multiple markets are identified separately."""
        idx1 = _make_index("A", crash_at=100, crash_depth=0.25)
        idx2 = _make_index("B", crash_at=200, crash_depth=0.30)
        crashes = identify_global_crashes({"A": idx1, "B": idx2}, threshold=0.20)
        markets = crashes["market"].unique()
        assert "A" in markets
        assert "B" in markets


class TestContagionFeatures:

    def test_shape_and_columns(self):
        """Output should have 2 columns aligned to target index."""
        idx = _make_index("X")
        target = idx.index
        features = compute_contagion_features({"X": idx}, target)
        assert "n_markets_in_drawdown" in features.columns
        assert "global_drawdown_avg" in features.columns
        assert len(features) == len(target)

    def test_missing_indices_handled(self):
        """Empty dict should return zero-filled features."""
        target = pd.bdate_range("2020-01-01", periods=50)
        features = compute_contagion_features({}, target)
        assert len(features) == 50
        assert (features["n_markets_in_drawdown"] == 0).all()
        assert (features["global_drawdown_avg"] == 0).all()

    def test_n_markets_count(self):
        """Count should reflect how many markets are in drawdown."""
        # Create 3 markets: 2 crash, 1 doesn't
        idx1 = _make_index("A", crash_at=200, crash_depth=0.25)
        idx2 = _make_index("B", crash_at=200, crash_depth=0.25)
        idx3 = _make_index("C", crash_at=200, crash_depth=0.05)  # no crash
        index_data = {"A": idx1, "B": idx2, "C": idx3}
        target = idx1.index
        features = compute_contagion_features(index_data, target)
        # During crash period, at least 2 markets should be in drawdown
        max_count = features["n_markets_in_drawdown"].max()
        assert max_count >= 2

    def test_drawdown_avg_negative_during_crash(self):
        """Average drawdown should be negative when markets are crashing."""
        idx = _make_index("X", crash_at=100, crash_depth=0.35)
        target = idx.index
        features = compute_contagion_features({"X": idx}, target)
        min_avg = features["global_drawdown_avg"].min()
        assert min_avg < -0.10
