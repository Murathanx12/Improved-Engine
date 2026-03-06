"""
XGBoost Crash Predictor — Ensemble Peer to LightGBM
=====================================================

Trained identically to the LightGBM CrashPredictor but uses a different
tree-building algorithm. Ensembling two GBMs of different architectures
consistently outperforms either alone because they make different errors:

- LightGBM: leaf-wise growth → deeper, sharper splits
- XGBoost: level-wise growth → more balanced, broader exploration

Both use Platt scaling calibration, temporal weighting, and purged
train/val splits to prevent data leakage.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

from finpredict.config import config as _cfg

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression as _PlattScaler
    from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss

    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False


if _HAS_XGBOOST:

    class XGBoostCrashPredictor:
        """
        Multi-horizon XGBoost crash probability estimator.

        Mirror of CrashPredictor (LightGBM) with identical training pipeline
        but XGBoost's level-wise tree building for ensemble diversity.
        """

        def __init__(self, n_estimators: int = 800, random_state: int = 42):
            self.n_estimators = n_estimators
            self.random_state = random_state
            self.models = {}
            self.calibrators = {}
            self.feature_names = None
            self.is_trained = False
            self._train_crash_rate = {}

        def train(
            self,
            features: pd.DataFrame,
            targets: dict | pd.Series,
            train_end_idx: Optional[int] = None,
            min_train_samples: int = 1260,
        ) -> dict:
            """Train XGBoost crash predictors on one or more horizons."""
            if isinstance(targets, pd.Series):
                targets = {"12m": targets}

            if train_end_idx is not None:
                X = features.iloc[:train_end_idx].copy()
                target_slices = {h: t.iloc[:train_end_idx].copy() for h, t in targets.items()}
            else:
                X = features.copy()
                target_slices = {h: t.copy() for h, t in targets.items()}

            primary_target = target_slices.get("12m", list(target_slices.values())[0])
            valid = primary_target.notna() & X.notna().any(axis=1)
            X_clean = X[valid]
            if len(X_clean) < min_train_samples:
                return {"success": False, "reason": f"Only {len(X_clean)} samples"}

            self.feature_names = list(X_clean.columns)

            results = {}
            for horizon, target in target_slices.items():
                y = target.iloc[:train_end_idx] if train_end_idx is not None else target.copy()
                valid_h = y.notna() & X.notna().any(axis=1)
                X_h = X[valid_h]
                y_h = y[valid_h].astype(int)

                if len(X_h) < min_train_samples or y_h.nunique() < 2:
                    continue

                self._train_crash_rate[horizon] = float(y_h.mean())
                r = self._train_single(X_h, y_h, horizon)
                results[horizon] = r

            if not self.models:
                return {"success": False, "reason": "No horizon trained successfully"}

            self.is_trained = True
            primary_result = results.get("12m", list(results.values())[0])
            return primary_result

        def _train_single(self, X: pd.DataFrame, y: pd.Series, horizon: str) -> dict:
            """Train a single horizon XGBoost model with temporal weighting."""
            pos_rate = float(y.mean())
            n_samples = len(X)

            # Exponential temporal weighting from config
            decay = _cfg.get("ml", {}).get("temporal_weight_decay", 0.0005)
            temporal_weights = np.exp(-decay * (n_samples - np.arange(n_samples)))
            crash_transitions = np.abs(y.diff().fillna(0).values)
            transition_weight = 1.0 + crash_transitions * 2.0
            sample_weights = temporal_weights * transition_weight

            # Purged split — gaps from config
            purge_cfg = _cfg.get("ml", {}).get("purge_gaps", {"3m": 70, "6m": 140, "12m": 265})
            gap_days = purge_cfg.get(horizon, purge_cfg.get("12m", 265))
            val_size = max(504, n_samples // 5)
            split_idx = n_samples - val_size - gap_days

            if split_idx < min(1260, n_samples // 2):
                split_idx = n_samples - val_size
                gap_days = 0

            train_X = X.iloc[:split_idx]
            train_y = y.iloc[:split_idx]
            train_w = sample_weights[:split_idx]
            val_X = X.iloc[split_idx + gap_days:]
            val_y = y.iloc[split_idx + gap_days:]

            if len(val_y) < 50 or val_y.nunique() < 2:
                split_idx = int(n_samples * 0.8)
                train_X = X.iloc[:split_idx]
                train_y = y.iloc[:split_idx]
                train_w = sample_weights[:split_idx]
                val_X = X.iloc[split_idx:]
                val_y = y.iloc[split_idx:]

            if train_y.nunique() < 2:
                return {"success": False, "reason": f"Training set has only class {train_y.unique()[0]}"}
            if val_y.nunique() < 2:
                eval_split = max(50, int(len(train_X) * 0.1))
                val_X = train_X.iloc[-eval_split:]
                val_y = train_y.iloc[-eval_split:]

            scale_pos = min((1 - pos_rate) / max(pos_rate, 0.01), 10.0)

            model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=7,
                learning_rate=0.008,
                min_child_weight=30,
                subsample=0.75,
                colsample_bytree=0.65,
                reg_alpha=0.05,
                reg_lambda=0.5,
                scale_pos_weight=scale_pos,
                random_state=self.random_state,
                eval_metric="logloss",
                verbosity=0,
                n_jobs=-1,
                tree_method="hist",
                early_stopping_rounds=100,
            )
            model.fit(
                train_X, train_y,
                sample_weight=train_w,
                eval_set=[(val_X, val_y)],
                verbose=False,
            )
            self.models[horizon] = model

            # Platt scaling calibration
            raw_probs = model.predict_proba(val_X)[:, 1]

            if raw_probs.std() < 1e-6:
                print(f"  [WARN] XGB zero variance in raw probs for {horizon}, skipping calibration")
            else:
                calibrator = _PlattScaler(C=1.0, solver='lbfgs', max_iter=1000)
                calibrator.fit(raw_probs.reshape(-1, 1), val_y.values)
                self.calibrators[horizon] = calibrator

                # Verify calibration preserved monotonicity
                test_scores = np.linspace(
                    max(raw_probs.min(), 1e-6), min(raw_probs.max(), 1 - 1e-6), 20
                )
                cal_test = calibrator.predict_proba(test_scores.reshape(-1, 1))[:, 1]
                corr = np.corrcoef(test_scores, cal_test)[0, 1]
                if np.isnan(corr) or corr < 0.5:
                    print(f"  [WARN] XGB calibration may be inverted for {horizon}, using raw probs")
                    self.calibrators.pop(horizon, None)

            # Compute metrics
            if horizon in self.calibrators:
                cal_probs = self.calibrators[horizon].predict_proba(raw_probs.reshape(-1, 1))[:, 1]
            else:
                cal_probs = raw_probs
            val_brier = brier_score_loss(val_y, cal_probs)

            try:
                val_auc = roc_auc_score(val_y, cal_probs)
            except ValueError:
                val_auc = 0.5

            return {
                "success": True,
                "horizon": horizon,
                "n_train": len(train_X),
                "n_val": len(val_X),
                "pos_rate": pos_rate,
                "val_brier": float(val_brier),
                "val_auc": float(val_auc),
                "pred_std": float(cal_probs.std()),
            }

        def predict_proba(self, features: pd.DataFrame, horizon: str = "12m") -> np.ndarray:
            """Predict calibrated crash probability."""
            if not self.is_trained:
                raise RuntimeError("XGBoost model not trained — call train() first")

            X = features[self.feature_names] if isinstance(features, pd.DataFrame) else features

            if horizon not in self.models:
                horizon = list(self.models.keys())[0]

            raw = self.models[horizon].predict_proba(X)[:, 1]

            if horizon in self.calibrators:
                calibrated = self.calibrators[horizon].predict_proba(raw.reshape(-1, 1))[:, 1]
            else:
                calibrated = raw

            return np.clip(calibrated, 0.02, 0.98)

        def predict_all_horizons(self, features: pd.DataFrame) -> dict:
            """Predict crash probability at all trained horizons."""
            results = {}
            for horizon in self.models:
                results[horizon] = self.predict_proba(features, horizon)
            return results

else:
    class XGBoostCrashPredictor:
        """Fallback when XGBoost is not installed."""

        def __init__(self, n_estimators: int = 800, random_state: int = 42):
            self.is_trained = False
            self.models = {}

        def train(self, features, targets, train_end_idx=None, min_train_samples=1260):
            return {"success": False, "reason": "XGBoost not installed"}

        def predict_proba(self, features, horizon: str = "12m"):
            base = _cfg.get("ml", {}).get("crash_base_rate_fallback", 0.12)
            logger.warning("[WARN] XGBoost not installed, returning base rate %.2f", base)
            return np.full(len(features), base)

        def predict_all_horizons(self, features):
            return {}


__all__ = ["XGBoostCrashPredictor"]
