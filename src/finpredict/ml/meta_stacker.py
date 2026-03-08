"""
Meta-Stacker — Regime-Aware Model Ensemble
=============================================

WHY THIS EXISTS:
    Different models excel in different market regimes:
    - LSTM/TCN capture temporal stress build-up → better in slow-onset bear markets
    - LightGBM/XGBoost read current conditions → better for sudden shocks
    - In bull regimes, snapshot features are sufficient (low noise)
    - In crisis regimes, temporal memory is critical (panic sequences)

    The meta-stacker learns HOW to weight these models based on the current
    regime. It's a logistic regression (few parameters = resistant to overfitting)
    that takes all model predictions + regime probabilities as input.

HOW IT WORKS:
    Meta-features for each sample:
        [lgb_pred, xgb_pred, lstm_pred, tcn_pred, p_bull, p_bear, p_crisis]

    The logistic stacker learns interaction effects:
    - coeff(lstm_pred * p_bear) > coeff(lgb_pred * p_bear) means LSTM
      is more trusted in bearish regimes
    - The model naturally learns the ensemble weight per regime

TRAINING:
    Trained on OUT-OF-SAMPLE predictions from the walk-forward backtest.
    This prevents the stacker from learning to overfit to in-sample behavior.
    Each model's prediction at time t was made without seeing any data after t.
"""

import logging
import numpy as np
from typing import Optional

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss

from finpredict.config import config as _cfg

logger = logging.getLogger(__name__)


class MetaStacker:
    """Regime-aware meta-learner that combines model predictions.

    Learns optimal weighting of LSTM, TCN, LightGBM, and XGBoost
    predictions conditioned on the current market regime.
    """

    MODEL_NAMES = ["lgb", "xgb", "lstm", "tcn", "cox"]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.stackers = {}     # {horizon: LogisticRegression}
        self.scalers = {}      # {horizon: StandardScaler}
        self.is_trained = False
        self._available_models = []
        self._model_weights = {}

    def train(
        self,
        model_predictions: dict,
        regime_probs: np.ndarray,
        targets: dict,
        train_end_idx: Optional[int] = None,
    ) -> dict:
        """Train the meta-stacker on out-of-sample model predictions.

        Args:
            model_predictions: Dict of {model_name: {horizon: np.ndarray}}
                Each array contains OOS predictions from the walk-forward backtest.
                Example: {"lgb": {"12m": array}, "xgb": {"12m": array}, ...}
            regime_probs: Array of shape (n_samples, 3) with [p_bull, p_bear, p_crisis].
            targets: Dict of {horizon: binary_array} actual outcomes.
            train_end_idx: Optional temporal cutoff.
        """
        # Determine which models are available (have predictions)
        self._available_models = [
            m for m in self.MODEL_NAMES
            if m in model_predictions and model_predictions[m]
        ]

        if len(self._available_models) < 2:
            return {"success": False, "reason": f"Need >=2 models, have {len(self._available_models)}"}

        # Determine available horizons
        sample_model = model_predictions[self._available_models[0]]
        horizons = list(sample_model.keys())

        results = {}
        for horizon in horizons:
            r = self._train_horizon(
                model_predictions, regime_probs, targets.get(horizon),
                horizon, train_end_idx,
            )
            results[horizon] = r

        if any(r.get("success") for r in results.values()):
            self.is_trained = True
            return {"success": True, "horizons": results}
        return {"success": False, "reason": "No horizon trained successfully"}

    def _train_horizon(
        self,
        model_predictions: dict,
        regime_probs: np.ndarray,
        target: np.ndarray,
        horizon: str,
        train_end_idx: Optional[int],
    ) -> dict:
        """Train meta-stacker for a single horizon."""
        if target is None:
            return {"success": False, "reason": "No target available"}

        # Build meta-feature matrix
        meta_features = []
        feature_names = []

        for model_name in self._available_models:
            preds = model_predictions[model_name].get(horizon)
            if preds is None:
                continue
            meta_features.append(np.asarray(preds, dtype=float))
            feature_names.append(f"pred_{model_name}")

        if not meta_features:
            return {"success": False, "reason": "No model predictions available"}

        # Add regime probabilities as features
        if regime_probs is not None and len(regime_probs) > 0:
            rp = np.asarray(regime_probs, dtype=float)
            if rp.ndim == 1:
                rp = rp.reshape(-1, 1)
            for i, name in enumerate(["p_bull", "p_bear", "p_crisis"]):
                if i < rp.shape[1]:
                    meta_features.append(rp[:, i])
                    feature_names.append(name)

        # Add interaction features (model pred * regime prob)
        if regime_probs is not None and regime_probs.ndim == 2:
            for model_name in self._available_models:
                preds = model_predictions[model_name].get(horizon)
                if preds is not None and regime_probs.shape[1] >= 2:
                    # Interaction with bear probability
                    meta_features.append(np.asarray(preds) * regime_probs[:len(preds), 1])
                    feature_names.append(f"{model_name}_x_bear")

        # Stack into matrix
        min_len = min(len(f) for f in meta_features)
        X = np.column_stack([f[:min_len] for f in meta_features])
        y = np.asarray(target[:min_len], dtype=float)

        # Remove NaN rows
        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid]
        y = y[valid]

        if len(X) < 50 or y.sum() < 2:
            return {"success": False, "reason": f"Insufficient data: {len(X)} rows, {y.sum()} positives"}

        # Train/val split
        if train_end_idx is not None:
            n = min(train_end_idx, len(X))
        else:
            n = len(X)

        split = int(n * 0.8)
        X_train, X_val = X[:split], X[split:n]
        y_train, y_val = y[:split], y[split:n]

        if y_train.sum() < 2 or len(np.unique(y_val)) < 2:
            # Use all data for training (no validation metrics)
            X_train, y_train = X[:n], y[:n]
            X_val, y_val = X_train, y_train

        # Standardize
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        # Logistic regression with L2 regularization
        lr = LogisticRegression(
            C=0.5,
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced',
            random_state=self.random_state,
        )
        lr.fit(X_train_s, y_train)

        self.stackers[horizon] = lr
        self.scalers[horizon] = scaler

        # Compute metrics
        val_probs = lr.predict_proba(X_val_s)[:, 1]
        val_brier = brier_score_loss(y_val, val_probs)

        # Extract model weights for interpretability
        coefs = dict(zip(feature_names, lr.coef_[0]))
        self._model_weights[horizon] = coefs

        print(f"  [META] Horizon {horizon}: Brier={val_brier:.4f}, "
              f"models={self._available_models}, "
              f"features={len(feature_names)}")

        return {
            "success": True,
            "val_brier": float(val_brier),
            "n_train": len(X_train),
            "feature_weights": {k: float(v) for k, v in coefs.items()},
        }

    def predict_proba(
        self,
        model_predictions: dict,
        regime_probs: np.ndarray = None,
        horizon: str = "12m",
    ) -> np.ndarray:
        """Predict ensemble crash probability.

        Args:
            model_predictions: {model_name: probability_value_or_array}
                Example: {"lgb": 0.35, "xgb": 0.30, "lstm": 0.45, "tcn": 0.40}
                Missing models should be passed as None — they are excluded
                from the combination rather than filled with a constant.
            regime_probs: [p_bull, p_bear, p_crisis] array.
            horizon: Prediction horizon.

        Returns:
            Calibrated ensemble crash probability.
        """
        ms_cfg = _cfg.get("ml", {}).get("meta_stacker", {})
        ms_cfg.get("min_models_required", 2)
        fallback_single = ms_cfg.get("fallback_to_best_single", True)

        # Determine which models from training are actually available now
        available_at_inference = [
            m for m in self._available_models
            if model_predictions.get(m) is not None
        ]

        # Collect only non-None predictions for simple averaging
        available_preds = [
            (k, v) for k, v in model_predictions.items()
            if k in self.MODEL_NAMES and v is not None
        ]

        if not available_preds:
            logger.warning("[WARN] No model predictions available for ensemble")
            base = _cfg.get("ml", {}).get("crash_base_rate_fallback", 0.12)
            return np.array([base])

        if len(available_preds) == 1 and fallback_single:
            # Single model: return its prediction directly
            return np.clip(np.array([float(available_preds[0][1])]), 0.02, 0.98)

        if not self.is_trained or horizon not in self.stackers:
            # Fallback: simple average of available predictions
            vals = [float(v) for _, v in available_preds]
            return np.clip(np.array([np.mean(vals)]), 0.02, 0.98)

        # If any model that was available during training is now missing,
        # the stacker's learned weights are invalid — fall back to simple
        # average of whatever models ARE available.
        if set(available_at_inference) != set(self._available_models):
            logger.warning(
                "[WARN] Models changed since training "
                f"(train={self._available_models}, now={available_at_inference}), "
                "falling back to simple average"
            )
            vals = [float(v) for _, v in available_preds]
            return np.clip(np.array([np.mean(vals)]), 0.02, 0.98)

        # Build meta-features (all training models guaranteed present)
        meta_features = []
        for model_name in self._available_models:
            pred = model_predictions[model_name]
            meta_features.append(float(pred) if np.isscalar(pred) else float(pred[0]))

        # Add regime probabilities
        if regime_probs is not None:
            rp = np.asarray(regime_probs, dtype=float).flatten()
            for i in range(min(3, len(rp))):
                meta_features.append(rp[i])

            # Add interaction features
            for model_name in self._available_models:
                pred = model_predictions[model_name]
                pred_val = float(pred) if np.isscalar(pred) else float(pred[0])
                if len(rp) >= 2:
                    meta_features.append(pred_val * rp[1])  # pred * p_bear
        else:
            # Fill with neutral regime probs
            for _ in range(3):
                meta_features.append(0.33)
            for model_name in self._available_models:
                pred = model_predictions[model_name]
                pred_val = float(pred) if np.isscalar(pred) else float(pred[0])
                meta_features.append(pred_val * 0.33)

        X = np.array(meta_features).reshape(1, -1)
        X = np.nan_to_num(X, nan=0.0)
        X_s = self.scalers[horizon].transform(X)
        prob = self.stackers[horizon].predict_proba(X_s)[:, 1]

        # Also compute ensemble disagreement stats
        pred_vals = [float(v) for _, v in available_preds]
        self._last_ensemble_std = float(np.std(pred_vals)) if len(pred_vals) > 1 else 0.0
        self._last_ensemble_min = float(np.min(pred_vals))
        self._last_ensemble_max = float(np.max(pred_vals))
        threshold = _cfg.get("ml", {}).get("ensemble_disagreement_threshold", 0.12)
        self._last_disagreement_flag = self._last_ensemble_std > threshold

        return np.clip(prob, 0.02, 0.98)

    def get_ensemble_disagreement(self) -> dict:
        """Return the most recent ensemble disagreement statistics.

        Returns:
            Dict with ensemble_std, ensemble_min, ensemble_max, disagreement_flag.
        """
        return {
            "ensemble_std": getattr(self, "_last_ensemble_std", 0.0),
            "ensemble_min": getattr(self, "_last_ensemble_min", 0.0),
            "ensemble_max": getattr(self, "_last_ensemble_max", 0.0),
            "disagreement_flag": getattr(self, "_last_disagreement_flag", False),
        }

    def get_model_weights(self, horizon: str = "12m") -> dict:
        """Return learned model weights for interpretability.

        Returns:
            Dict of {feature_name: coefficient} showing which models
            the stacker trusts most and how regime affects weighting.
        """
        return self._model_weights.get(horizon, {})


__all__ = ["MetaStacker"]
