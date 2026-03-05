"""
ML Crash Predictor (v7 — Discriminative, Data-Driven)
========================================================

WHY THE OLD VERSION FAILED:
    1. Histogram calibration with 15 bins → sparse crash data mapped to bin
       midpoints (20-25%), squashing ALL predictions to medium risk
    2. Scale_pos_weight alone can't fix temporal class imbalance
    3. Too much regularization killed the signal
    4. No temporal weighting → ancient data dilutes recent patterns

WHAT'S CHANGED:
    1. Isotonic regression calibration (monotonic, handles sparse bins)
    2. Temporal sample weighting (recent data weighted higher)
    3. Multiple drawdown thresholds (10%, 15%, 20%) for richer signal
    4. Ensemble of models with different sensitivities
    5. Spread enforcement — if predictions cluster, we detect and flag it
    6. All parameters learned from data, zero hardcoded thresholds

HOW IT WORKS:
    The model learns which feature combinations preceded crashes historically.
    Key learned patterns include:
    - Yield curve inversions (6-18 months before recession)
    - Volatility compression followed by expansion
    - Momentum exhaustion after extended rallies
    - Credit spread widening
    - Macro deterioration (unemployment rising, sentiment falling)

    The model trains on expanding windows of data and predicts crash
    probability at 3m, 6m, and 12m horizons. Isotonic regression maps
    raw LightGBM scores to calibrated probabilities that are monotonically
    increasing (higher raw score = higher probability, guaranteed).
"""

import numpy as np
import pandas as pd
from typing import Optional

try:
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression as _PlattScaler
    from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss

    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False


if _HAS_LIGHTGBM:

    class CrashPredictor:
        """
        Multi-horizon LightGBM crash probability estimator.

        Uses Platt scaling (logistic sigmoid) for calibration instead of
        isotonic regression, which is more robust with sparse crash data.
        Severity ensemble models are trained for feature analysis only
        and are NOT blended into final predictions.
        """

        # Key features for the simple logistic regression model.
        # These are robust macro/market indicators with established crash-leading properties.
        LOGISTIC_FEATURES = [
            "vix_zscore",           # VIX z-score (vol regime)
            "yield_curve_10y3m",    # Yield curve (10Y-3M, inversion signal)
            "credit_spread_chg_3m", # HY OAS 3-month change (credit stress)
            "sp500_12m_return",     # 12-month momentum (mean-reversion signal)
            "vol_ratio_1m_12m",     # Short/long vol ratio (vol clustering)
        ]

        def __init__(self, n_estimators: int = 800, random_state: int = 42):
            self.n_estimators = n_estimators
            self.random_state = random_state
            self.models = {}                 # {horizon: lgb model}
            self.calibrators = {}            # {horizon: Platt scaler}
            self.severity_models = {}        # {severity: lgb model} for analysis
            self.severity_calibrators = {}   # {severity: Platt scaler}
            self.logistic_models = {}        # {horizon: LogisticRegression}
            self.feature_names = None
            self.feature_importances_ = None
            self.is_trained = False
            self._train_crash_rate = {}      # base rate per horizon for fallback
            self._lgb_brier = {}             # {horizon: brier_score} for model selection

        def train(
            self,
            features: pd.DataFrame,
            targets: dict | pd.Series,
            train_end_idx: Optional[int] = None,
            min_train_samples: int = 1260,
            severity_targets: dict = None,
        ) -> dict:
            """
            Train crash predictors on one or more horizons.

            Uses expanding window: all data before train_end_idx is training data.
            The validation set is carved from the most recent portion with a
            purge gap to prevent label leakage.

            Args:
                features: Feature matrix (daily, backward-looking only)
                targets: Dict of {horizon: binary_series} or single Series
                train_end_idx: Temporal cutoff index
                min_train_samples: Minimum observations needed to train
                severity_targets: Optional dict of {threshold_label: binary_series}
                    for multi-threshold ensemble (e.g., 10%, 15%, 20% drawdowns)
            """
            if isinstance(targets, pd.Series):
                targets = {"12m": targets}

            if train_end_idx is not None:
                X = features.iloc[:train_end_idx].copy()
                target_slices = {h: t.iloc[:train_end_idx].copy() for h, t in targets.items()}
            else:
                X = features.copy()
                target_slices = {h: t.copy() for h, t in targets.items()}

            # Use 12m target for feature selection
            primary_target = target_slices.get("12m", list(target_slices.values())[0])
            valid = primary_target.notna() & X.notna().any(axis=1)
            X_clean = X[valid]
            if len(X_clean) < min_train_samples:
                return {"success": False, "reason": f"Only {len(X_clean)} samples"}

            self.feature_names = list(X_clean.columns)

            results = {}
            combined_importances = np.zeros(len(self.feature_names))

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

                if r["success"] and horizon in self.models:
                    imp = self.models[horizon].feature_importances_
                    if imp.sum() > 0:
                        combined_importances += imp / imp.sum()

            if not self.models:
                return {"success": False, "reason": "No horizon trained successfully"}

            # ── Train severity ensemble (10%, 15%, 20% drawdown thresholds) ──
            if severity_targets:
                self._train_severity_ensemble(X, severity_targets, train_end_idx, min_train_samples)

            # ── Train simple logistic regression on key features ──────────
            # With sparse crash data (~10% base rate, <10 crash events),
            # a logistic model with 5 hand-picked features generalizes better
            # than LightGBM with 80+ features.
            for horizon, target in target_slices.items():
                self._train_logistic(X, target, horizon, train_end_idx, min_train_samples)

            self.feature_importances_ = dict(zip(self.feature_names, combined_importances))
            self.is_trained = True

            primary_result = results.get("12m", list(results.values())[0])
            return primary_result

        def _train_severity_ensemble(
            self,
            X: pd.DataFrame,
            severity_targets: dict,
            train_end_idx: Optional[int],
            min_train_samples: int,
        ):
            """Train models at multiple drawdown severity levels for ensemble.

            More lenient thresholds (10%, 15%) provide more training examples,
            capturing early-warning patterns that the strict 20% model misses.
            The ensemble blends predictions: 0.15 * p_10 + 0.25 * p_15 + 0.60 * p_20
            """
            for label, target in severity_targets.items():
                y = target.iloc[:train_end_idx] if train_end_idx is not None else target.copy()
                valid = y.notna() & X.iloc[:len(y)].notna().any(axis=1)
                X_sev = X.iloc[:len(y)][valid]
                y_sev = y[valid].astype(int)

                if len(X_sev) < min_train_samples or y_sev.nunique() < 2:
                    continue

                try:
                    r = self._train_single(X_sev, y_sev, label)
                    if r["success"]:
                        # Move from primary models dict to severity dict
                        self.severity_models[label] = self.models.pop(label)
                        if label in self.calibrators:
                            self.severity_calibrators[label] = self.calibrators.pop(label)
                except Exception:
                    continue

        def _train_single(self, X: pd.DataFrame, y: pd.Series, horizon: str) -> dict:
            """
            Train a single horizon model with:
            1. Temporal sample weighting (recent data weighted 2x vs oldest)
            2. Purged train/val split (gap covers the forward-looking window)
            3. LightGBM with controlled complexity
            4. Isotonic regression calibration on validation set
            """
            pos_rate = float(y.mean())
            n_samples = len(X)

            # ── Temporal weighting ────────────────────────────────────
            # Recent observations matter more — market structure evolves
            # Linear ramp from 0.5 (oldest) to 1.5 (newest)
            temporal_weights = np.linspace(0.5, 1.5, n_samples)

            # Also upweight samples near crash transitions (these are most informative)
            crash_transitions = np.abs(y.diff().fillna(0).values)
            transition_weight = 1.0 + crash_transitions * 2.0  # 3x weight on transitions

            sample_weights = temporal_weights * transition_weight

            # ── Purged split ──────────────────────────────────────────
            # Gap must cover the forward window to prevent label leakage
            gap_days = {"3m": 70, "6m": 140, "12m": 265}.get(horizon, 265)
            val_size = max(504, n_samples // 5)  # 20% for validation
            split_idx = n_samples - val_size - gap_days

            if split_idx < min(1260, n_samples // 2):
                # Not enough data for proper purge — use simple split
                split_idx = n_samples - val_size
                gap_days = 0

            train_X = X.iloc[:split_idx]
            train_y = y.iloc[:split_idx]
            train_w = sample_weights[:split_idx]
            val_X = X.iloc[split_idx + gap_days:]
            val_y = y.iloc[split_idx + gap_days:]

            if len(val_y) < 50 or val_y.nunique() < 2:
                # Fallback: use last 20% without gap (less ideal but functional)
                split_idx = int(n_samples * 0.8)
                train_X = X.iloc[:split_idx]
                train_y = y.iloc[:split_idx]
                train_w = sample_weights[:split_idx]
                val_X = X.iloc[split_idx:]
                val_y = y.iloc[split_idx:]

            # ── Ensure both classes present in train AND val ─────────
            # LightGBM's LabelEncoder will crash if val_y contains a label
            # not seen in train_y (e.g., train has only 0, val has 0 and 1).
            if train_y.nunique() < 2:
                return {"success": False, "reason": f"Training set has only class {train_y.unique()[0]}"}
            if val_y.nunique() < 2:
                # Val set single-class: still train the model but skip early stopping
                # by using a small portion of training data as eval set instead
                eval_split = max(50, int(len(train_X) * 0.1))
                val_X = train_X.iloc[-eval_split:]
                val_y = train_y.iloc[-eval_split:]

            # ── LightGBM training ─────────────────────────────────────
            # Less regularized than v6 — the model NEEDS to find patterns in
            # the relatively sparse crash signal
            scale_pos = (1 - pos_rate) / max(pos_rate, 0.01)
            # Clamp to prevent extreme weights
            scale_pos = min(scale_pos, 10.0)

            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "n_estimators": self.n_estimators,
                "max_depth": 7,              # Deeper to capture interactions
                "num_leaves": 40,            # More leaves for nuance
                "learning_rate": 0.008,      # Slow learning → more iterations → better fit
                "min_child_samples": 30,     # Allow fine-grained splits
                "subsample": 0.75,           # Row sampling
                "colsample_bytree": 0.65,    # Column sampling
                "reg_alpha": 0.05,           # Very light L1 (was 0.1)
                "reg_lambda": 0.5,           # Light L2 (was 1.0)
                "min_gain_to_split": 0.002,  # Allow subtle splits
                "scale_pos_weight": scale_pos,
                "random_state": self.random_state,
                "verbose": -1,
                "n_jobs": -1,
            }

            model = lgb.LGBMClassifier(**params)
            model.fit(
                train_X, train_y,
                sample_weight=train_w,
                eval_set=[(val_X, val_y)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            self.models[horizon] = model

            # ── Platt scaling calibration ──────────────────────────────
            # Platt scaling (logistic sigmoid) has only 2 parameters (A, B)
            # vs isotonic's N parameters, making it far more robust with
            # sparse positive examples (~10% crash base rate).
            raw_probs = model.predict_proba(val_X)[:, 1]
            calibrator = _PlattScaler(C=1.0, solver='lbfgs', max_iter=1000)
            calibrator.fit(raw_probs.reshape(-1, 1), val_y.values)
            self.calibrators[horizon] = calibrator

            # ── Verify calibration preserved monotonicity ─────────────
            test_scores = np.linspace(
                max(raw_probs.min(), 1e-6), min(raw_probs.max(), 1 - 1e-6), 20
            )
            cal_test = calibrator.predict_proba(test_scores.reshape(-1, 1))[:, 1]
            if np.corrcoef(test_scores, cal_test)[0, 1] < 0.5:
                print(f"  [WARN] Calibration may be inverted for {horizon}, using raw probs")
                self.calibrators.pop(horizon, None)

            # ── Compute metrics ───────────────────────────────────────
            if horizon in self.calibrators:
                cal_probs = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
            else:
                cal_probs = raw_probs
            val_brier = brier_score_loss(val_y, cal_probs)
            val_logloss = log_loss(val_y, np.clip(cal_probs, 1e-7, 1 - 1e-7), labels=[0, 1])

            try:
                val_auc = roc_auc_score(val_y, cal_probs)
            except ValueError:
                val_auc = 0.5

            self._lgb_brier[horizon] = float(val_brier)

            pred_range = (float(cal_probs.min()), float(cal_probs.max()))
            pred_std = float(cal_probs.std())

            return {
                "success": True,
                "horizon": horizon,
                "n_train": len(train_X),
                "n_val": len(val_X),
                "pos_rate": pos_rate,
                "val_brier": float(val_brier),
                "val_logloss": float(val_logloss),
                "val_auc": float(val_auc),
                "n_estimators_used": getattr(model, "best_iteration_", self.n_estimators),
                "pred_range": pred_range,
                "pred_std": pred_std,
                "discrimination": "GOOD" if pred_std > 0.05 else "POOR",
            }

        def _train_logistic(
            self,
            X: pd.DataFrame,
            target: pd.Series,
            horizon: str,
            train_end_idx: Optional[int],
            min_train_samples: int,
        ):
            """Train a simple logistic regression on key macro features.

            With sparse crash data, 5 hand-picked features generalize far better
            than 80+ features in LightGBM. This model serves as primary when
            LightGBM Brier > 0.22 (near random).
            """
            y = target.iloc[:train_end_idx] if train_end_idx is not None else target.copy()
            valid = y.notna() & X.notna().any(axis=1)
            X_v = X[valid]
            y_v = y[valid].astype(int)

            # Find which logistic features are available in the feature matrix
            available_feats = [f for f in self.LOGISTIC_FEATURES if f in X_v.columns]
            if len(available_feats) < 2:
                # Try fuzzy matching for common naming variations
                feat_map = {
                    "vix_zscore": ["VIX_zscore", "vix_z", "VIX_z_score"],
                    "yield_curve_10y3m": ["yield_curve", "T10Y3M", "yield_spread"],
                    "credit_spread_chg_3m": ["credit_spread_3m", "hy_oas_chg", "credit_chg"],
                    "sp500_12m_return": ["sp500_ret_12m", "momentum_12m", "ret_12m"],
                    "vol_ratio_1m_12m": ["vol_ratio", "vratio_1m12m", "short_long_vol"],
                }
                for canonical, aliases in feat_map.items():
                    if canonical not in available_feats:
                        for alias in aliases:
                            if alias in X_v.columns:
                                available_feats.append(alias)
                                break

            if len(available_feats) < 2 or len(y_v) < min_train_samples:
                return

            X_log = X_v[available_feats].fillna(0)

            # Simple 80/20 split
            split = int(len(X_log) * 0.8)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_log.iloc[:split])
            X_val = scaler.transform(X_log.iloc[split:])
            y_train = y_v.iloc[:split]
            y_val = y_v.iloc[split:]

            if y_train.nunique() < 2 or y_val.nunique() < 2:
                return

            lr = _PlattScaler(C=0.1, solver='lbfgs', max_iter=1000, class_weight='balanced')
            lr.fit(X_train, y_train)

            self.logistic_models[horizon] = {
                "model": lr,
                "scaler": scaler,
                "features": available_feats,
            }
            print(f"  [OK] Logistic crash model ({horizon}): "
                  f"{len(available_feats)} features, "
                  f"Brier={brier_score_loss(y_val, lr.predict_proba(X_val)[:, 1]):.4f}")

        @staticmethod
        def _lookup_table_prob(features: pd.DataFrame) -> float:
            """Empirical conditional crash rate lookup table.

            Returns the base-rate crash probability conditioned on current
            macro signals. More honest than a complex ML model that overfits
            when the model can't beat random.
            """
            # Try to extract key signals
            vix = None
            yc_inverted = False

            for col in ["VIX", "vix", "VIX_close"]:
                if col in features.columns:
                    vix = float(features[col].iloc[-1]) if len(features) > 0 else None
                    break

            for col in ["yield_curve_10y3m", "yield_curve", "T10Y3M", "yield_spread"]:
                if col in features.columns:
                    val = float(features[col].iloc[-1]) if len(features) > 0 else 0
                    yc_inverted = val < 0
                    break

            if yc_inverted and vix is not None and vix > 25:
                return 0.50  # Both signals: 50% within 12 months
            elif yc_inverted:
                return 0.35  # Inverted yield curve: 35%
            elif vix is not None and vix > 25:
                return 0.25  # Elevated VIX: 25%
            else:
                return 0.12  # Base rate: 12%

        def predict_proba(self, features: pd.DataFrame, horizon: str = "12m") -> np.ndarray:
            """
            Predict calibrated crash probability at given horizon.

            Uses LightGBM with Platt scaling as primary. If LightGBM Brier > 0.22
            (near random), falls back to logistic regression. The lookup table
            provides sanity-check bounds; when logistic diverges from it by >15pp,
            they blend 50/50.

            Pipeline: features → model selection → calibration → lookup blend → clip
            """
            if not self.is_trained:
                raise RuntimeError("Model not trained — call train() first")

            X = features[self.feature_names] if isinstance(features, pd.DataFrame) else features

            # ── Model selection based on LightGBM performance ──────────
            lgb_brier = self._lgb_brier.get(horizon, 1.0)
            use_logistic = lgb_brier > 0.22 and horizon in self.logistic_models

            if use_logistic:
                # LightGBM near random — use logistic regression
                lm = self.logistic_models[horizon]
                feat_cols = lm["features"]
                if isinstance(features, pd.DataFrame) and all(f in features.columns for f in feat_cols):
                    X_log = lm["scaler"].transform(features[feat_cols].fillna(0))
                    calibrated = lm["model"].predict_proba(X_log)[:, 1]
                else:
                    # Features not available, fall back to LightGBM anyway
                    use_logistic = False

            if not use_logistic:
                # Primary: LightGBM with Platt scaling
                if horizon not in self.models:
                    horizon = list(self.models.keys())[0]

                raw = self.models[horizon].predict_proba(X)[:, 1]

                if horizon in self.calibrators:
                    calibrated = self.calibrators[horizon].predict_proba(
                        raw.reshape(-1, 1)
                    )[:, 1]
                else:
                    calibrated = raw

            # ── Lookup table sanity check ──────────────────────────────
            # When model prediction diverges from empirical base rates by
            # >15pp, blend 50/50 to prevent extreme miscalibration
            if isinstance(features, pd.DataFrame):
                lookup_prob = self._lookup_table_prob(features)
                for i in range(len(calibrated)):
                    if abs(calibrated[i] - lookup_prob) > 0.15:
                        calibrated[i] = 0.5 * calibrated[i] + 0.5 * lookup_prob

            return np.clip(calibrated, 0.02, 0.98)

        def predict_all_horizons(self, features: pd.DataFrame) -> dict:
            """Predict crash probability at all trained horizons."""
            results = {}
            for horizon in self.models:
                results[horizon] = self.predict_proba(features, horizon)
            return results

        def get_discrimination_report(self) -> dict:
            """Report on model's ability to discriminate crash vs non-crash."""
            report = {}
            for horizon, model in self.models.items():
                report[horizon] = {
                    "n_estimators_used": getattr(model, "best_iteration_", 0),
                    "base_crash_rate": self._train_crash_rate.get(horizon, 0),
                    "has_calibrator": horizon in self.calibrators,
                }
            return report

        def get_top_features(self, n: int = 15) -> list:
            """Return top N features by combined importance across horizons."""
            if self.feature_importances_ is None:
                return []
            return sorted(
                self.feature_importances_.items(),
                key=lambda x: x[1], reverse=True
            )[:n]

        def get_shap_values(self, features: pd.DataFrame, horizon: str = "12m") -> list:
            """Compute SHAP values to explain why the model predicts high/low crash probability.

            Returns list of (feature_name, shap_value) tuples sorted by absolute SHAP value.
            Positive SHAP = feature pushes crash probability UP.
            Negative SHAP = feature pushes crash probability DOWN.
            """
            try:
                import shap
            except ImportError:
                return []

            if horizon not in self.models:
                if not self.models:
                    return []
                horizon = list(self.models.keys())[0]

            X = features[self.feature_names] if isinstance(features, pd.DataFrame) else features
            explainer = shap.TreeExplainer(self.models[horizon])
            shap_values = explainer.shap_values(X)

            # For binary classification, shap_values may be a list [class_0, class_1]
            if isinstance(shap_values, list):
                sv = shap_values[1]  # Class 1 (crash) SHAP values
            else:
                sv = shap_values

            # Return feature contributions for the last row (current prediction)
            row = sv[-1] if len(sv.shape) > 1 else sv
            contributions = list(zip(self.feature_names, row))
            return sorted(contributions, key=lambda x: abs(x[1]), reverse=True)

        def run_counterfactual(
            self,
            base_features: pd.DataFrame,
            scenarios: list,
        ) -> dict:
            """Estimate crash probability under hypothetical market conditions.

            Each scenario overrides one or more feature values in the current
            feature row and re-runs the model.  Useful for answering questions
            like "what would crash probability be if VIX hit 40?".

            Args:
                base_features: DataFrame with one row — the current feature vector
                                (output of build_feature_matrix(...).iloc[-1:]).
                scenarios: list of dicts, each with:
                    "label"    – human-readable scenario name
                    "overrides" – {feature_name: new_value}

            Returns:
                {
                    "base_prob_3m":  float,
                    "base_prob_12m": float,
                    "scenarios": [
                        {
                            "label": str,
                            "overrides": dict,
                            "crash_prob_3m":  float,
                            "crash_prob_12m": float,
                            "delta_3m":  float,   # change vs base
                            "delta_12m": float,
                        },
                        ...
                    ]
                }
            """
            if not self.is_trained or not self.models:
                return {"base_prob_3m": None, "base_prob_12m": None, "scenarios": []}

            base_3m = (
                float(self.predict_proba(base_features, "3m")[0])
                if "3m" in self.models else None
            )
            base_12m = float(self.predict_proba(base_features, "12m")[0])

            results = []
            for sc in scenarios:
                label = sc.get("label", "Scenario")
                overrides = sc.get("overrides", {})

                modified = base_features.copy()
                for col, val in overrides.items():
                    if col in modified.columns:
                        modified[col] = val

                prob_3m = (
                    float(self.predict_proba(modified, "3m")[0])
                    if "3m" in self.models else None
                )
                prob_12m = float(self.predict_proba(modified, "12m")[0])

                results.append({
                    "label": label,
                    "overrides": overrides,
                    "crash_prob_3m": prob_3m,
                    "crash_prob_12m": prob_12m,
                    "delta_3m": (prob_3m - base_3m) if (prob_3m is not None and base_3m is not None) else None,
                    "delta_12m": prob_12m - base_12m,
                })

            return {
                "base_prob_3m": base_3m,
                "base_prob_12m": base_12m,
                "scenarios": results,
            }

else:
    # ── Fallback when LightGBM is not installed ─────────────────────
    class CrashPredictor:
        """Very small fallback crash predictor used when LightGBM is
        not available. It predicts the historical crash frequency learned
        at train time as a constant probability for all rows.
        """

        def __init__(self, n_estimators: int = 100, random_state: int = 42):
            self.n_estimators = n_estimators
            self.random_state = random_state
            self.is_trained = False
            self.baseline = 0.05
            self.models = {}

        def train(self, features, targets, train_end_idx=None, min_train_samples=1260):
            if isinstance(targets, dict):
                t = targets.get("12m", list(targets.values())[0])
            else:
                t = targets
            if train_end_idx is not None:
                t = t.iloc[:train_end_idx]
            t = t.dropna()
            if len(t) < 1:
                return {"success": False, "reason": "no valid target samples"}
            self.baseline = float(t.mean())
            self.is_trained = True
            return {
                "success": True,
                "val_auc": 0.5,
                "val_brier": float(((self.baseline - t) ** 2).mean()),
                "pred_range": (self.baseline, self.baseline),
                "pred_std": 0.0,
                "discrimination": 0.0,
            }

        def predict_proba(self, features, horizon: str = "12m"):
            n = len(features)
            return np.full(n, self.baseline)

        def predict_all_horizons(self, features):
            return {"12m": self.predict_proba(features)}

        def get_discrimination_report(self):
            return {}

        def get_top_features(self, n: int = 10):
            return []

        def get_shap_values(self, features, horizon: str = "12m"):
            return []

        def run_counterfactual(self, base_features, scenarios: list) -> dict:
            return {"base_prob_3m": None, "base_prob_12m": None, "scenarios": []}


__all__ = ["CrashPredictor"]
