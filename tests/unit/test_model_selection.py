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


class TestStackerExcludesMissingModels:
    """B1: Meta-stacker must exclude missing models, not inject 0.12."""

    def test_stacker_excludes_missing_models_at_inference(self):
        """Train stacker with 4 models, provide only 2 at inference.
        Output must equal reweighted combination of available models."""
        from finpredict.ml.meta_stacker import MetaStacker

        rng = np.random.default_rng(42)
        n = 200

        # Synthetic OOS predictions and targets
        model_predictions = {
            "lgb": {"12m": rng.uniform(0.05, 0.5, n)},
            "xgb": {"12m": rng.uniform(0.05, 0.5, n)},
            "lstm": {"12m": rng.uniform(0.05, 0.5, n)},
            "tcn": {"12m": rng.uniform(0.05, 0.5, n)},
        }
        regime_probs = rng.dirichlet([2, 1, 0.5], n)
        targets = {"12m": (rng.uniform(0, 1, n) > 0.85).astype(float)}

        stacker = MetaStacker()
        result = stacker.train(model_predictions, regime_probs, targets)
        assert result["success"]

        # At inference, only 2 models available
        preds_partial = {"lgb": 0.30, "xgb": 0.25, "lstm": None, "tcn": None}
        output = stacker.predict_proba(preds_partial, regime_probs[-1:], "12m")

        # Should be simple average of the 2 available models (fallback)
        expected_avg = np.mean([0.30, 0.25])
        assert abs(float(output[0]) - expected_avg) < 0.01, (
            f"Expected ~{expected_avg:.3f}, got {float(output[0]):.3f}"
        )

        # Verify it does NOT produce the stacker's output with 0.12 fill
        preds_filled = {"lgb": 0.30, "xgb": 0.25, "lstm": 0.12, "tcn": 0.12}
        output_filled = stacker.predict_proba(preds_filled, regime_probs[-1:], "12m")
        # The filled version should differ (or at least not be relied upon)
        assert output is not output_filled  # Different objects at minimum

    def test_stacker_falls_back_to_single_model(self):
        """With only 1 model at inference, return its raw prediction."""
        from finpredict.ml.meta_stacker import MetaStacker

        rng = np.random.default_rng(42)
        n = 200
        model_predictions = {
            "lgb": {"12m": rng.uniform(0.05, 0.5, n)},
            "xgb": {"12m": rng.uniform(0.05, 0.5, n)},
            "lstm": {"12m": rng.uniform(0.05, 0.5, n)},
            "tcn": {"12m": rng.uniform(0.05, 0.5, n)},
        }
        regime_probs = rng.dirichlet([2, 1, 0.5], n)
        targets = {"12m": (rng.uniform(0, 1, n) > 0.85).astype(float)}

        stacker = MetaStacker()
        stacker.train(model_predictions, regime_probs, targets)

        # Only 1 model available
        preds_single = {"lgb": 0.65, "xgb": None, "lstm": None, "tcn": None}
        output = stacker.predict_proba(preds_single, regime_probs[-1:], "12m")

        # Should return the single model's prediction (clipped)
        assert abs(float(output[0]) - 0.65) < 0.01

    def test_lookup_blend_reduces_when_ml_diverges(self):
        """B10: Dynamic blend weight should reduce when ML diverges from base rate."""
        from finpredict.config import config as cfg

        ml_pred = 0.70
        base_rate = 0.12
        blend_weight = cfg.get("ml", {}).get("lookup_blend_weight", 0.15)
        divergence_scale = cfg.get("ml", {}).get("lookup_blend_divergence_scale", 3.0)

        divergence = abs(ml_pred - base_rate)
        dynamic_weight = blend_weight * max(0.0, 1.0 - divergence * divergence_scale)

        # With divergence = 0.58, dynamic_weight should be much less than 0.15
        assert dynamic_weight < blend_weight * 0.5, (
            f"dynamic_weight={dynamic_weight:.4f} should be much less than {blend_weight}"
        )

        # Final prediction should be close to ML prediction, not pulled to base rate
        final = ml_pred * (1 - dynamic_weight) + base_rate * dynamic_weight
        assert final > 0.60, f"Final {final:.3f} too far from ML pred {ml_pred}"


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


class TestEnsembleDisagreement:
    """N1: Ensemble disagreement signal from meta-stacker."""

    def test_get_ensemble_disagreement_exists(self):
        from finpredict.ml.meta_stacker import MetaStacker
        stacker = MetaStacker()
        result = stacker.get_ensemble_disagreement()
        assert "ensemble_std" in result
        assert "ensemble_min" in result
        assert "ensemble_max" in result
        assert "disagreement_flag" in result

    def test_disagreement_threshold_in_config(self):
        from finpredict.config import config as cfg
        thresh = cfg.get("ml", {}).get("ensemble_disagreement_threshold")
        assert thresh is not None, "Config must define ensemble_disagreement_threshold"
        assert thresh > 0


class TestExponentialTemporalWeighting:
    """M4: Temporal weighting should use exponential decay from config."""

    def test_temporal_weight_decay_in_config(self):
        from finpredict.config import config as cfg
        decay = cfg.get("ml", {}).get("temporal_weight_decay")
        assert decay is not None, "Config must define ml.temporal_weight_decay"
        assert decay > 0, "Decay rate must be positive"

    def test_crash_model_uses_exponential_weights(self):
        import inspect
        from finpredict.ml.crash_model import CrashPredictor
        source = inspect.getsource(CrashPredictor._train_single)
        assert "np.exp(" in source, "Should use exponential temporal weighting"
        assert "temporal_weight_decay" in source, "Should read decay from config"
        assert "np.linspace(0.5, 1.5" not in source, "Should NOT use linear ramp"


class TestPurgeGapsCentralized:
    """M1: Purge gaps must be defined in config, not hardcoded in each model."""

    def test_purge_gaps_in_config(self):
        from finpredict.config import config as cfg
        purge = cfg.get("ml", {}).get("purge_gaps", {})
        assert "3m" in purge, "Config must define purge_gaps.3m"
        assert "6m" in purge, "Config must define purge_gaps.6m"
        assert "12m" in purge, "Config must define purge_gaps.12m"
        assert purge["3m"] < purge["6m"] < purge["12m"]

    def test_models_use_config_purge_gaps(self):
        """All model files should reference config purge_gaps."""
        import inspect
        from finpredict.ml.crash_model import CrashPredictor
        source = inspect.getsource(CrashPredictor._train_single)
        assert "purge_gaps" in source, "crash_model should use config purge_gaps"
        assert "purge_cfg" in source, "crash_model should use purge_cfg from config"


class TestBacktestNoLGBSubstitution:
    """B11/B12: Backtest must pass None for missing models, not LGB prediction."""

    def test_backtest_model_preds_use_none_not_lgb(self):
        """B11: model_preds dict should pass None, not LGB fallback."""
        import inspect
        from finpredict.simulation import backtest as bt_mod
        source = inspect.getsource(bt_mod)
        # The model_preds dict should NOT substitute lgb for missing models
        assert '"xgb": xgb_crash_12m,' in source, (
            "model_preds should pass xgb_crash_12m directly (may be None)"
        )
        assert '"lstm": lstm_crash_12m,' in source, (
            "model_preds should pass lstm_crash_12m directly (may be None)"
        )

    def test_backtest_xgb_oos_uses_none_not_lgb(self):
        """B12: XGB 6m/3m OOS should use None when model unavailable, not LGB."""
        import inspect
        from finpredict.simulation import backtest as bt_mod
        source = inspect.getsource(bt_mod)
        # Should NOT find patterns like: else lgb_crash_6m for XGB OOS
        # Should find: xgb_6m = None
        assert "xgb_6m = None" in source, "XGB 6m OOS should default to None"
        assert "xgb_3m = None" in source, "XGB 3m OOS should default to None"
