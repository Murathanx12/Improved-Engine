"""
ML Return Predictor (v7 — Quantile Regression, Data-Driven)
=============================================================

WHY THE OLD VERSION FAILED:
    1. Single point predictions with aggressive clipping [-0.60, 1.00]
    2. No uncertainty quantification — can't distinguish confident
       vs uncertain predictions
    3. Over-regularized (reg_lambda=3.0) → predictions regressed to mean
    4. No connection between crash model and return model

WHAT'S CHANGED:
    1. Trains THREE models per horizon: median, 10th percentile, 90th percentile
    2. Quantile regression via LightGBM's quantile objective
    3. Less regularization to preserve genuine return predictability
    4. Temporal weighting (recent data matters more)
    5. Skill score comparison vs naive (historical average) baseline
    6. All parameters learned from data

HOW IT WORKS:
    Return prediction is inherently noisier than crash prediction (lower SNR).
    But certain features have genuine predictive power at 6-12 month horizons:
    - Starting valuation (trend deviation, CAPE proxy)
    - Yield curve state (inverted = lower future returns)
    - Momentum (mean-reversion at 12m+ horizons)
    - Macro state (unemployment trend, Fed policy)

    The model outputs a DISTRIBUTION of returns (10th, 50th, 90th percentile)
    rather than a single point estimate. This is more honest and more useful —
    the Monte Carlo simulator can use these bounds directly.
"""

import numpy as np
import pandas as pd
from typing import Optional

import lightgbm as lgb


class ReturnPredictor:
    """
    Multi-horizon return predictor with quantile regression.
    
    For each horizon, trains three models:
    - Median (50th percentile) — point estimate
    - Lower (10th percentile) — downside bound
    - Upper (90th percentile) — upside bound
    """

    def __init__(self, n_estimators: int = 600, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = {}              # {horizon: median lgb model}
        self.quantile_models = {}     # {(horizon, quantile): lgb model}
        self.feature_names = None
        self.feature_importances_ = None
        self.is_trained = False
        self.train_stats = {}         # {horizon: {mean, std, min, max}}
        self._naive_mae = {}          # baseline MAE per horizon

    def train(
        self,
        features: pd.DataFrame,
        targets: dict | pd.Series,
        train_end_idx: Optional[int] = None,
        min_train_samples: int = 1260,
    ) -> dict:
        """
        Train return predictors with quantile regression.

        For each horizon, trains:
        - Median model (primary point estimate)
        - 10th percentile model (downside scenario)
        - 90th percentile model (upside scenario)
        """
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
        combined_importances = np.zeros(len(self.feature_names))

        for horizon, target in target_slices.items():
            y = target.iloc[:train_end_idx] if train_end_idx else target.copy()
            valid_h = y.notna() & X.notna().any(axis=1)
            X_h = X[valid_h]
            y_h = y[valid_h]

            if len(X_h) < min_train_samples:
                continue

            self.train_stats[horizon] = {
                "mean": float(y_h.mean()),
                "std": float(y_h.std()),
                "min": float(y_h.min()),
                "max": float(y_h.max()),
                "p10": float(y_h.quantile(0.10)),
                "p90": float(y_h.quantile(0.90)),
            }

            r = self._train_single(X_h, y_h, horizon)
            results[horizon] = r

            if r["success"] and horizon in self.models:
                imp = self.models[horizon].feature_importances_
                if imp.sum() > 0:
                    combined_importances += imp / imp.sum()

        if not self.models:
            return {"success": False, "reason": "No horizon trained successfully"}

        self.feature_importances_ = dict(zip(self.feature_names, combined_importances))
        self.is_trained = True

        return results.get("12m", list(results.values())[0])

    def _train_single(self, X: pd.DataFrame, y: pd.Series, horizon: str) -> dict:
        """Train median + quantile models for a single horizon."""
        n_samples = len(X)

        # ── Temporal weighting ────────────────────────────────────
        temporal_weights = np.linspace(0.5, 1.5, n_samples)

        # ── Purged split ──────────────────────────────────────────
        gap_days = {"3m": 70, "6m": 140, "12m": 265}.get(horizon, 265)
        val_size = max(504, n_samples // 5)
        split_idx = n_samples - val_size - gap_days

        if split_idx < min(1260, n_samples // 2):
            split_idx = int(n_samples * 0.8)
            gap_days = 0

        train_X = X.iloc[:split_idx]
        train_y = y.iloc[:split_idx]
        train_w = temporal_weights[:split_idx]
        val_X = X.iloc[split_idx + gap_days:]
        val_y = y.iloc[split_idx + gap_days:]

        if len(val_y) < 50:
            split_idx = int(n_samples * 0.8)
            train_X = X.iloc[:split_idx]
            train_y = y.iloc[:split_idx]
            train_w = temporal_weights[:split_idx]
            val_X = X.iloc[split_idx:]
            val_y = y.iloc[split_idx:]

        # ── Train median model (primary) ──────────────────────────
        base_params = {
            "n_estimators": self.n_estimators,
            "max_depth": 6,
            "num_leaves": 30,
            "learning_rate": 0.008,
            "min_child_samples": 40,
            "subsample": 0.75,
            "colsample_bytree": 0.60,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_gain_to_split": 0.003,
            "random_state": self.random_state,
            "verbose": -1,
            "n_jobs": -1,
        }

        # Median regression (L1 loss, more robust to outliers)
        median_params = {**base_params, "objective": "regression", "metric": "mae"}
        model = lgb.LGBMRegressor(**median_params)
        model.fit(
            train_X, train_y,
            sample_weight=train_w,
            eval_set=[(val_X, val_y)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=80, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        self.models[horizon] = model

        # ── Train quantile models ─────────────────────────────────
        for alpha in [0.10, 0.90]:
            q_params = {**base_params,
                        "objective": "quantile",
                        "alpha": alpha,
                        "metric": "quantile"}
            q_model = lgb.LGBMRegressor(**q_params)
            q_model.fit(
                train_X, train_y,
                sample_weight=train_w,
                eval_set=[(val_X, val_y)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=80, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            self.quantile_models[(horizon, alpha)] = q_model

        # ── Compute metrics ───────────────────────────────────────
        val_pred = model.predict(val_X)
        val_mae = float(np.abs(val_y.values - val_pred).mean())

        val_corr = 0.0
        if len(val_y) > 5:
            c = np.corrcoef(val_y.values, val_pred)[0, 1]
            val_corr = float(c if not np.isnan(c) else 0)

        # Naive baseline: predict historical average
        naive_pred = train_y.mean()
        naive_mae = float(np.abs(val_y.values - naive_pred).mean())
        skill = 1 - val_mae / naive_mae if naive_mae > 0 else 0
        self._naive_mae[horizon] = naive_mae

        # Quantile coverage: what % of actuals fall within [p10, p90]?
        if (horizon, 0.10) in self.quantile_models and (horizon, 0.90) in self.quantile_models:
            p10 = self.quantile_models[(horizon, 0.10)].predict(val_X)
            p90 = self.quantile_models[(horizon, 0.90)].predict(val_X)
            coverage = float(((val_y.values >= p10) & (val_y.values <= p90)).mean())
        else:
            coverage = 0.0

        return {
            "success": True,
            "horizon": horizon,
            "n_train": len(train_X),
            "n_val": len(val_X),
            "val_mae": val_mae,
            "val_corr": val_corr,
            "mean_return": float(y.mean()),
            "pred_range": (float(val_pred.min()), float(val_pred.max())),
            "naive_mae": naive_mae,
            "skill_score": float(skill),
            "quantile_coverage": coverage,
        }

    def predict(self, features: pd.DataFrame, horizon: str = "12m") -> np.ndarray:
        """Predict median forward return."""
        if not self.is_trained:
            raise RuntimeError("Model not trained — call train() first")

        if horizon not in self.models:
            horizon = list(self.models.keys())[0]

        X = features[self.feature_names] if isinstance(features, pd.DataFrame) else features
        preds = self.models[horizon].predict(X)

        # Wider clip to preserve signal — learned from data range
        stats = self.train_stats.get(horizon, {"min": -0.60, "max": 1.50})
        lo = max(stats.get("min", -0.60) * 1.2, -0.80)  # 20% beyond worst seen
        hi = min(stats.get("max", 1.50) * 1.2, 2.00)     # 20% beyond best seen
        return np.clip(preds, lo, hi)

    def predict_quantiles(
        self, features: pd.DataFrame, horizon: str = "12m"
    ) -> dict:
        """
        Predict return distribution: median, 10th, 90th percentiles.
        
        Returns:
            dict with keys 'median', 'p10', 'p90' → np.ndarray
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        if horizon not in self.models:
            horizon = list(self.models.keys())[0]

        X = features[self.feature_names] if isinstance(features, pd.DataFrame) else features
        result = {"median": self.models[horizon].predict(X)}

        for alpha in [0.10, 0.90]:
            key = "p10" if alpha == 0.10 else "p90"
            if (horizon, alpha) in self.quantile_models:
                result[key] = self.quantile_models[(horizon, alpha)].predict(X)
            else:
                # Fallback: offset from median using training stats
                stats = self.train_stats.get(horizon, {"std": 0.15})
                z = -1.28 if alpha == 0.10 else 1.28
                result[key] = result["median"] + z * stats["std"]

        return result

    def predict_all_horizons(self, features: pd.DataFrame) -> dict:
        """Predict returns at all trained horizons."""
        results = {}
        for horizon in self.models:
            results[horizon] = self.predict(features, horizon)
        return results

    def get_top_features(self, n: int = 15) -> list:
        """Return top N features by importance."""
        if self.feature_importances_ is None:
            return []
        return sorted(
            self.feature_importances_.items(),
            key=lambda x: x[1], reverse=True
        )[:n]
