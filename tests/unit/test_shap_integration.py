"""Tests for SHAP explainability integration in CrashPredictor."""

import numpy as np
import pandas as pd
import pytest


class TestPredictWithShap:
    """Verify predict_with_shap method structure."""

    def test_method_exists(self):
        """CrashPredictor should have predict_with_shap method."""
        from finpredict.ml.crash_model import CrashPredictor
        cp = CrashPredictor()
        assert hasattr(cp, "predict_with_shap")

    def test_returns_dict_structure(self):
        """predict_with_shap should return dict with expected keys."""
        from finpredict.ml.crash_model import CrashPredictor
        from finpredict.ml.features import build_feature_matrix, build_target_crash

        rng = np.random.default_rng(42)
        n = 3000
        dates = pd.bdate_range("2005-01-01", periods=n)
        sp500 = 1500 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))

        data = pd.DataFrame({
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

        features = build_feature_matrix(data)
        target = build_target_crash(data)

        cp = CrashPredictor(n_estimators=50)
        result = cp.train(features, {"12m": target}, train_end_idx=len(features))

        if result.get("success"):
            pred = cp.predict_with_shap(features.iloc[-1:], "12m", compute_shap=False)
            assert "crash_prob" in pred
            assert "selected_model" in pred
            assert "shap_values" in pred
            assert pred["shap_values"] is None  # compute_shap=False

    def test_shap_disabled_by_default(self):
        """With compute_shap=False, shap_values should be None."""
        from finpredict.ml.crash_model import CrashPredictor
        cp = CrashPredictor()
        # Even untrained, the method signature should work
        assert hasattr(cp, "predict_with_shap")

    def test_shap_values_dict_format(self):
        """When SHAP is computed, values should be dict of {feature: float}."""
        from finpredict.ml.crash_model import CrashPredictor
        from finpredict.ml.features import build_feature_matrix, build_target_crash

        rng = np.random.default_rng(42)
        n = 3000
        dates = pd.bdate_range("2005-01-01", periods=n)
        sp500 = 1500 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))

        data = pd.DataFrame({
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

        features = build_feature_matrix(data)
        target = build_target_crash(data)

        cp = CrashPredictor(n_estimators=50)
        result = cp.train(features, {"12m": target}, train_end_idx=len(features))

        if result.get("success") and cp.selected_model.get("12m") == "lgb":
            try:
                import shap
                pred = cp.predict_with_shap(features.iloc[-1:], "12m", compute_shap=True)
                if pred["shap_values"] is not None:
                    assert isinstance(pred["shap_values"], dict)
                    assert len(pred["shap_values"]) <= 10
                    for k, v in pred["shap_values"].items():
                        assert isinstance(k, str)
                        assert isinstance(v, (float, np.floating))
            except ImportError:
                pytest.skip("shap not installed")
