"""Tests for Cox Survival Model."""

import numpy as np
import pandas as pd
import pytest


class TestCrashSurvivalModel:

    def test_import(self):
        from finpredict.ml.survival_model import CrashSurvivalModel
        model = CrashSurvivalModel()
        assert hasattr(model, "train")
        assert hasattr(model, "predict_proba")
        assert hasattr(model, "is_trained")
        assert not model.is_trained

    def test_train_and_predict(self):
        """Train on synthetic data and verify predictions are in [0, 1]."""
        from finpredict.ml.survival_model import CrashSurvivalModel, _HAS_LIFELINES
        if not _HAS_LIFELINES:
            pytest.skip("lifelines not installed")

        rng = np.random.default_rng(42)
        n = 2000
        dates = pd.bdate_range("2000-01-01", periods=n)

        # Synthetic features
        features = pd.DataFrame({
            "vix_zscore": rng.normal(0, 1, n),
            "term_spread": rng.normal(1.5, 0.5, n),
            "credit_spread_proxy": rng.normal(0, 0.02, n),
            "mom_12m": rng.normal(0.08, 0.15, n),
            "vol_ratio_1m_12m": rng.normal(1.0, 0.3, n),
            "mom_6m": rng.normal(0.04, 0.10, n),
            "erp": rng.normal(0.03, 0.02, n),
            "sma_200d_dev": rng.normal(0, 0.05, n),
            "dist_52w_high": rng.uniform(-0.15, 0, n),
            "vol_1m": rng.uniform(0.005, 0.03, n),
        }, index=dates)

        # Synthetic price data with a couple of crashes
        prices = np.ones(n) * 3000.0
        for i in range(1, n):
            if 500 <= i < 560:
                prices[i] = prices[i - 1] * 0.993  # crash ~35%
            elif 1200 <= i < 1250:
                prices[i] = prices[i - 1] * 0.994  # crash ~26%
            else:
                prices[i] = prices[i - 1] * 1.0004
        data = pd.DataFrame({"SP500": prices}, index=dates)

        model = CrashSurvivalModel()
        result = model.train(
            features, {},
            train_end_idx=1500,
            min_train_samples=500,
            data=data,
        )

        assert result["success"] is True
        assert model.is_trained

        # Predict on last row
        probs = model.predict_proba(features.iloc[-1:], "12m")
        assert len(probs) == 1
        assert 0.0 <= probs[0] <= 1.0

    def test_untrained_raises(self):
        from finpredict.ml.survival_model import CrashSurvivalModel
        model = CrashSurvivalModel()
        with pytest.raises(RuntimeError):
            model.predict_proba(pd.DataFrame({"vix_zscore": [0.5]}), "12m")

    def test_multi_horizon(self):
        """3m probability should be <= 12m probability (monotonic in time)."""
        from finpredict.ml.survival_model import CrashSurvivalModel, _HAS_LIFELINES
        if not _HAS_LIFELINES:
            pytest.skip("lifelines not installed")

        rng = np.random.default_rng(42)
        n = 2000
        dates = pd.bdate_range("2000-01-01", periods=n)

        features = pd.DataFrame({
            "vix_zscore": rng.normal(0, 1, n),
            "term_spread": rng.normal(1.5, 0.5, n),
            "credit_spread_proxy": rng.normal(0, 0.02, n),
            "mom_12m": rng.normal(0.08, 0.15, n),
            "vol_ratio_1m_12m": rng.normal(1.0, 0.3, n),
            "mom_6m": rng.normal(0.04, 0.10, n),
            "erp": rng.normal(0.03, 0.02, n),
            "sma_200d_dev": rng.normal(0, 0.05, n),
            "dist_52w_high": rng.uniform(-0.15, 0, n),
            "vol_1m": rng.uniform(0.005, 0.03, n),
        }, index=dates)

        prices = np.ones(n) * 3000.0
        for i in range(1, n):
            if 500 <= i < 560:
                prices[i] = prices[i - 1] * 0.993
            else:
                prices[i] = prices[i - 1] * 1.0004
        data = pd.DataFrame({"SP500": prices}, index=dates)

        model = CrashSurvivalModel()
        model.train(features, {}, train_end_idx=1500, min_train_samples=500, data=data)

        row = features.iloc[-1:]
        p3m = model.predict_proba(row, "3m")[0]
        p12m = model.predict_proba(row, "12m")[0]
        # 3m crash prob should be <= 12m (monotonic survival function)
        assert p3m <= p12m + 0.01  # small tolerance for numerical issues

    def test_get_top_features(self):
        from finpredict.ml.survival_model import CrashSurvivalModel, _HAS_LIFELINES
        if not _HAS_LIFELINES:
            pytest.skip("lifelines not installed")

        rng = np.random.default_rng(42)
        n = 2000
        dates = pd.bdate_range("2000-01-01", periods=n)

        features = pd.DataFrame({
            "vix_zscore": rng.normal(0, 1, n),
            "term_spread": rng.normal(1.5, 0.5, n),
            "credit_spread_proxy": rng.normal(0, 0.02, n),
            "mom_12m": rng.normal(0.08, 0.15, n),
            "vol_ratio_1m_12m": rng.normal(1.0, 0.3, n),
            "mom_6m": rng.normal(0.04, 0.10, n),
            "erp": rng.normal(0.03, 0.02, n),
            "sma_200d_dev": rng.normal(0, 0.05, n),
            "dist_52w_high": rng.uniform(-0.15, 0, n),
            "vol_1m": rng.uniform(0.005, 0.03, n),
        }, index=dates)

        prices = np.ones(n) * 3000.0
        for i in range(1, n):
            if 500 <= i < 560:
                prices[i] = prices[i - 1] * 0.993
            else:
                prices[i] = prices[i - 1] * 1.0004
        data = pd.DataFrame({"SP500": prices}, index=dates)

        model = CrashSurvivalModel()
        model.train(features, {}, train_end_idx=1500, min_train_samples=500, data=data)

        top = model.get_top_features(3)
        assert isinstance(top, list)
        assert len(top) <= 3
        for name, coef in top:
            assert isinstance(name, str)
            assert isinstance(coef, float)
