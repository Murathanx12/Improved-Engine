"""Tests for BUG 3 fix: removal of duplicate drawdown_from_peak feature."""

import numpy as np
import pandas as pd
import pytest

from finpredict.ml.features import build_feature_matrix


@pytest.fixture
def sample_data():
    """Create synthetic market data for feature building."""
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.bdate_range("2020-01-01", periods=n)
    sp500 = 3000 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))

    df = pd.DataFrame({
        "SP500": sp500,
        "VIX": rng.uniform(12, 35, n),
        "T10Y": rng.uniform(1.0, 4.0, n),
        "T3M": rng.uniform(0.5, 3.5, n),
        "T30Y": rng.uniform(2.0, 5.0, n),
        "HYG": rng.uniform(70, 90, n),
        "LQD": rng.uniform(100, 130, n),
        "Gold": rng.uniform(1500, 2000, n),
        "NASDAQ": rng.uniform(10000, 16000, n),
        "Russell": rng.uniform(1500, 2500, n),
    }, index=dates)
    return df


class TestDrawdownFromPeakRemoved:
    """Verify duplicate feature is removed."""

    def test_drawdown_from_peak_not_in_columns(self, sample_data):
        """drawdown_from_peak should no longer exist."""
        features = build_feature_matrix(sample_data)
        assert "drawdown_from_peak" not in features.columns

    def test_dist_52w_high_still_exists(self, sample_data):
        """dist_52w_high should still be present."""
        features = build_feature_matrix(sample_data)
        assert "dist_52w_high" in features.columns

    def test_dist_52w_high_values_valid(self, sample_data):
        """dist_52w_high should be <= 0 (drawdown from peak)."""
        features = build_feature_matrix(sample_data)
        valid = features["dist_52w_high"].dropna()
        assert (valid <= 0).all()

    def test_renamed_interaction_features_exist(self, sample_data):
        """Interaction features should use new names."""
        features = build_feature_matrix(sample_data)
        assert "vol_x_dist52w" in features.columns
        assert "dist52w_x_vix" in features.columns

    def test_old_interaction_names_removed(self, sample_data):
        """Old interaction feature names should not exist."""
        features = build_feature_matrix(sample_data)
        assert "vol_x_drawdown" not in features.columns
        assert "drawdown_x_vix" not in features.columns
        assert "skew_x_drawdown" not in features.columns
