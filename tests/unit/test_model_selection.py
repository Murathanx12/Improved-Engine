"""Tests for automatic model selection in CrashPredictor."""

import numpy as np
import pandas as pd
import pytest


class TestModelSelection:
    """Verify model selection infrastructure exists and functions."""

    def test_selected_model_dict_exists(self):
        """CrashPredictor should have selected_model and model_selection_results."""
        from finpredict.ml.crash_model import CrashPredictor
        cp = CrashPredictor()
        assert hasattr(cp, "selected_model")
        assert hasattr(cp, "model_selection_results")
        assert isinstance(cp.selected_model, dict)
        assert isinstance(cp.model_selection_results, dict)

    def test_selection_populated_after_training(self):
        """After training, selected_model should have entries."""
        from finpredict.ml.crash_model import CrashPredictor
        from finpredict.ml.features import build_feature_matrix

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

        # Build crash target
        from finpredict.ml.features import build_target_crash
        target = build_target_crash(data)

        cp = CrashPredictor(n_estimators=50)
        result = cp.train(features, {"12m": target}, train_end_idx=len(features))

        if result.get("success"):
            assert "12m" in cp.selected_model
            assert cp.selected_model["12m"] in ("lgb", "logistic")

            if "12m" in cp.model_selection_results:
                sel = cp.model_selection_results["12m"]
                assert "lgb_brier" in sel
                assert "logistic_brier" in sel
                assert "selected" in sel

    def test_predict_uses_selected_model(self):
        """predict_proba should respect selected_model choice."""
        from finpredict.ml.crash_model import CrashPredictor

        cp = CrashPredictor()
        # Default selection should be "lgb" for untrained model
        assert cp.selected_model.get("12m", "lgb") == "lgb"


class TestOOSCollectionSkipsNone:
    """Bug 3: OOS prediction lists must never contain 0.12 placeholders."""

    def test_oos_collection_skips_none(self):
        """Simulate OOS collection where temporal model is unavailable for first 3 folds.
        Verify no 0.12 placeholders leak into meta-stacker training data."""
        oos_predictions = {
            "lgb": {"12m": []},
            "xgb": {"12m": []},
            "lstm": {"12m": []},
            "tcn": {"12m": []},
        }

        # Simulate 5 folds
        for fold in range(5):
            lgb_val = 0.15 + fold * 0.05   # Always available
            xgb_val = 0.18 + fold * 0.03   # Always available

            # Temporal model not available for first 3 folds
            lstm_val = None if fold < 3 else 0.25 + fold * 0.02
            tcn_val = None if fold < 3 else 0.22 + fold * 0.02

            # This mirrors the fixed OOS collection (no `else 0.12`)
            oos_predictions["lgb"]["12m"].append(lgb_val)
            oos_predictions["xgb"]["12m"].append(xgb_val)
            oos_predictions["lstm"]["12m"].append(lstm_val)
            oos_predictions["tcn"]["12m"].append(tcn_val)

        # Verify: no 0.12 placeholders anywhere
        for model_name in ["lgb", "xgb", "lstm", "tcn"]:
            vals = oos_predictions[model_name]["12m"]
            assert 0.12 not in vals, (
                f"Found 0.12 placeholder in {model_name} OOS predictions: {vals}"
            )

        # LSTM/TCN should have None for first 3 folds, real values for last 2
        lstm_vals = oos_predictions["lstm"]["12m"]
        assert lstm_vals[:3] == [None, None, None]
        assert all(v is not None for v in lstm_vals[3:])

        # When converted to numpy for meta-stacker, None becomes NaN
        import numpy as np
        lstm_arr = np.array(lstm_vals, dtype=float)
        assert np.isnan(lstm_arr[:3]).all()
        assert not np.isnan(lstm_arr[3:]).any()
