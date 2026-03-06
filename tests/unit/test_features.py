"""Tests for features: BUG 3 fix, new predictive features (Task 1), and near-term signals (Task 2)."""

import numpy as np
import pandas as pd
import pytest

from finpredict.ml.features import build_feature_matrix, _frac_diff_ffd


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


@pytest.fixture
def sample_data_with_etfs(sample_data):
    """Sample data extended with TLT, UUP, GLD and VIX3M/SKEW."""
    rng = np.random.default_rng(99)
    n = len(sample_data)
    sample_data["TLT"] = rng.uniform(100, 160, n)
    sample_data["UUP"] = rng.uniform(24, 30, n)
    sample_data["GLD"] = rng.uniform(150, 200, n)
    sample_data["VIX3M"] = rng.uniform(14, 30, n)
    sample_data["SKEW"] = rng.uniform(110, 160, n)
    return sample_data


@pytest.fixture
def sample_fred_data(sample_data):
    """FRED data dict for build_feature_matrix fred_data parameter."""
    rng = np.random.default_rng(77)
    n = len(sample_data)
    idx = sample_data.index
    return {
        "hy_oas": pd.Series(rng.uniform(3.0, 8.0, n), index=idx),
        "ted_spread": pd.Series(rng.uniform(0.1, 1.5, n), index=idx),
        "tips_10y": pd.Series(rng.uniform(0.5, 2.5, n), index=idx),
        "margin_credit": pd.Series(np.linspace(500, 800, n) + rng.normal(0, 10, n), index=idx),
        "mfg_employment": pd.Series(np.linspace(12000, 13000, n) + rng.normal(0, 50, n), index=idx),
        "industrial_prod": pd.Series(np.linspace(100, 110, n) + rng.normal(0, 1, n), index=idx),
        "business_loans": pd.Series(np.linspace(2000, 2500, n) + rng.normal(0, 20, n), index=idx),
    }


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


class TestFractionalDifferentiation:
    """Tests for the FFD helper function."""

    def test_output_shape_matches_input(self):
        rng = np.random.default_rng(42)
        prices = pd.Series(np.cumprod(1 + rng.normal(0.0003, 0.01, 300)))
        result = _frac_diff_ffd(np.log(prices), d=0.4)
        assert len(result) == len(prices)

    def test_has_nan_warmup(self):
        prices = pd.Series(np.cumprod(1 + np.random.default_rng(1).normal(0, 0.01, 500)))
        result = _frac_diff_ffd(np.log(prices), d=0.4)
        assert result.isna().any(), "FFD should have NaN warmup period"
        assert not result.isna().all(), "FFD should produce some valid values"

    def test_d_zero_returns_original(self):
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _frac_diff_ffd(series, d=0.0)
        valid = result.dropna()
        np.testing.assert_allclose(valid.values, series.iloc[-len(valid):].values, atol=1e-6)


class TestERPFeatures:
    """Tests for Equity Risk Premium features (Section 11.1)."""

    def test_erp_exists(self, sample_data):
        features = build_feature_matrix(sample_data)
        assert "erp" in features.columns
        assert "erp_zscore" in features.columns
        assert "erp_below_1pct" in features.columns

    def test_erp_with_tips(self, sample_data, sample_fred_data):
        features = build_feature_matrix(sample_data, fred_data=sample_fred_data)
        assert "erp" in features.columns
        # With 500 data points, ERP needs 1260+ for rolling avg warmup,
        # so values will be 0 (filled NaN). Just verify column exists and
        # FRED tips_10y was used (feature count includes FRED columns).
        assert "fred_tips_10y" in features.columns

    def test_erp_below_1pct_is_binary(self, sample_data):
        features = build_feature_matrix(sample_data)
        vals = features["erp_below_1pct"].unique()
        assert set(vals).issubset({0.0, 1.0})

    def test_cape_proxy_renamed_to_price_to_avg_ratio(self):
        """B8: The feature should be named price_to_avg_ratio, not cape_proxy."""
        import inspect
        from finpredict.ml import features as feat_mod
        source = inspect.getsource(feat_mod.build_feature_matrix)
        assert "price_to_avg_ratio" in source, (
            "Should use 'price_to_avg_ratio' instead of misleading 'cape_proxy'"
        )
        assert "cape_proxy" not in source, (
            "Should NOT use 'cape_proxy' — it's not a real CAPE calculation"
        )


class TestFracDiffFeatures:
    """Tests for fractionally differentiated price features (Section 11.8)."""

    def test_frac_diff_exists(self, sample_data):
        features = build_feature_matrix(sample_data)
        assert "sp500_frac_diff" in features.columns
        assert "sp500_frac_diff_zscore" in features.columns


class TestRiskOffFeatures:
    """Tests for global risk-off flow features (Section 11.4)."""

    def test_risk_off_features_with_etfs(self, sample_data_with_etfs):
        features = build_feature_matrix(sample_data_with_etfs)
        assert "risk_off_binary_21d" in features.columns
        assert "risk_off_score_21d" in features.columns
        assert "risk_off_score_63d" in features.columns
        assert "tlt_ret_zscore" in features.columns

    def test_risk_off_binary_is_binary(self, sample_data_with_etfs):
        features = build_feature_matrix(sample_data_with_etfs)
        vals = features["risk_off_binary_21d"].unique()
        assert set(vals).issubset({0.0, 1.0})

    def test_no_risk_off_without_etfs(self, sample_data):
        features = build_feature_matrix(sample_data)
        assert "risk_off_binary_21d" not in features.columns


class TestCrossAssetCorrelation:
    """Tests for SPX-TLT correlation features (Section 11.7)."""

    def test_spx_tlt_corr_with_tlt(self, sample_data_with_etfs):
        features = build_feature_matrix(sample_data_with_etfs)
        assert "spx_tlt_corr_30d" in features.columns
        assert "spx_tlt_corr_63d" in features.columns
        assert "spx_tlt_corr_positive" in features.columns
        assert "spx_tlt_corr_zscore" in features.columns

    def test_spx_tlt_corr_positive_is_binary(self, sample_data_with_etfs):
        features = build_feature_matrix(sample_data_with_etfs)
        vals = features["spx_tlt_corr_positive"].unique()
        assert set(vals).issubset({0.0, 1.0})


class TestNearTermSignals:
    """Tests for Section 10 near-term stress signals."""

    def test_options_fear_index(self, sample_data_with_etfs):
        """VIX3M + SKEW → vix_backwardation → options_fear_index."""
        features = build_feature_matrix(sample_data_with_etfs)
        assert "options_fear_index" in features.columns
        assert "options_fear_5d_ma" in features.columns
        assert "options_fear_extreme" in features.columns

    def test_vix_backwardation_duration(self, sample_data_with_etfs):
        features = build_feature_matrix(sample_data_with_etfs)
        assert "vix_backwardation_duration" in features.columns
        assert "vix_backwardation_5d_pct" in features.columns
        assert "vix_backwardation_21d_pct" in features.columns

    def test_vix_ts_velocity_features(self, sample_data_with_etfs):
        """N2: VIX term structure velocity and persistence features."""
        features = build_feature_matrix(sample_data_with_etfs)
        assert "vix_ts_velocity_5d" in features.columns
        assert "vix_ts_velocity_21d" in features.columns

    def test_smart_money_proxy(self, sample_data):
        """Russell is in sample_data → small_large_ratio → smart_money_proxy."""
        features = build_feature_matrix(sample_data)
        assert "smart_money_proxy" in features.columns
        assert "smart_money_proxy_negative_streak" in features.columns

    def test_realized_vol_trend(self, sample_data):
        """vol_1w is always computed → realized_vol features should exist."""
        features = build_feature_matrix(sample_data)
        assert "realized_vol_trend_3w" in features.columns
        assert "realized_vol_rising_weeks" in features.columns
        assert "realized_vol_accel" in features.columns
        assert "vol_breakout" in features.columns


class TestHYVelocityFeatures:
    """Tests for HY spread velocity features (Section 10.2)."""

    def test_hy_velocity_with_fred(self, sample_data, sample_fred_data):
        features = build_feature_matrix(sample_data, fred_data=sample_fred_data)
        assert "hy_oas_chg_1w" in features.columns
        assert "hy_oas_chg_2w" in features.columns
        assert "hy_oas_chg_4w" in features.columns
        assert "hy_oas_accel" in features.columns
        assert "hy_oas_widening_streak" in features.columns

    def test_no_hy_velocity_without_fred(self, sample_data):
        features = build_feature_matrix(sample_data)
        assert "hy_oas_chg_1w" not in features.columns


class TestTEDSpreadVelocity:
    """Tests for TED spread velocity features (Section 10.3)."""

    def test_ted_velocity_with_fred(self, sample_data, sample_fred_data):
        features = build_feature_matrix(sample_data, fred_data=sample_fred_data)
        assert "ted_spread_chg_1w" in features.columns
        assert "ted_spread_velocity_zscore" in features.columns


class TestFREDFeatures:
    """Tests for FRED-dependent features (Sections 11.3, 11.5, 11.6)."""

    def test_margin_debt_features(self, sample_data, sample_fred_data):
        features = build_feature_matrix(sample_data, fred_data=sample_fred_data)
        assert "margin_debt_yoy" in features.columns
        assert "margin_debt_declining" in features.columns

    def test_mfg_employment_features(self, sample_data, sample_fred_data):
        features = build_feature_matrix(sample_data, fred_data=sample_fred_data)
        assert "mfg_employment_chg_3m" in features.columns
        assert "mfg_employment_declining_3m" in features.columns

    def test_industrial_prod_features(self, sample_data, sample_fred_data):
        features = build_feature_matrix(sample_data, fred_data=sample_fred_data)
        assert "industrial_prod_yoy" in features.columns

    def test_business_loans_features(self, sample_data, sample_fred_data):
        features = build_feature_matrix(sample_data, fred_data=sample_fred_data)
        assert "business_loans_yoy" in features.columns
        assert "business_loans_accel_3m" in features.columns
        assert "credit_crunch_signal" in features.columns

    def test_fred_monthly_data_shifted_for_publication_lag(self):
        """B9: Monthly FRED data must be shifted by ~21 trading days for publication lag."""
        import inspect
        from finpredict.ml import features as feat_mod
        source = inspect.getsource(feat_mod.build_feature_matrix)
        # Must detect monthly data and shift it
        assert "is_monthly" in source, "Should detect monthly FRED series"
        assert "shift(21)" in source, "Monthly FRED data should be shifted by 21 trading days"


class TestDynamicCrashThreshold:
    """M2: VIX-scaled crash threshold for regime-aware labeling."""

    def test_dynamic_threshold_signature(self):
        """build_target_crash should accept dynamic_vix parameter."""
        import inspect
        from finpredict.ml.features import build_target_crash
        sig = inspect.signature(build_target_crash)
        assert "dynamic_vix" in sig.parameters, "build_target_crash must have dynamic_vix param"

    def test_dynamic_threshold_config_exists(self):
        from finpredict.config import config as cfg
        dyn = cfg.get("ml", {}).get("dynamic_crash_threshold", {})
        assert "vix_long_run_avg" in dyn
        assert "min_threshold" in dyn
        assert "max_threshold" in dyn


class TestFeatureCount:
    """Verify total feature count exceeds target."""

    def test_base_feature_count_above_80(self, sample_data):
        features = build_feature_matrix(sample_data)
        assert features.shape[1] > 80, f"Expected >80 features, got {features.shape[1]}"

    def test_full_feature_count_above_100(self, sample_data_with_etfs):
        """With ETFs, feature count should be well above 100."""
        features = build_feature_matrix(sample_data_with_etfs)
        assert features.shape[1] > 100, f"Expected >100 features, got {features.shape[1]}"
