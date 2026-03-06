"""
Crash Timing Classifier — 3-Month Window Precision
====================================================

WHY THIS EXISTS:
    The engine predicts crash probability at 3m, 6m, and 12m horizons as
    independent binary classifiers. They don't combine to tell you WHEN
    within the next 12 months a crash is most likely. A model that says
    '65% chance in 12 months AND 30% in 3 months' actually implies the
    crash is most likely in months 4-12, but this is never made explicit.

    This module takes horizon probabilities and features, and outputs a
    conditional probability distribution over 3-month windows:
    Q1 (0-3m), Q2 (3-6m), Q3 (6-9m), Q4 (9-12m), Beyond 12m.

HOW IT WORKS:
    1. Training labels: For each historical date, we find which 3-month
       window the next crash actually fell in (or "beyond" if none in 12m).
    2. LightGBM multiclass classifier trained on features + horizon probs.
    3. Output probabilities are rescaled to sum to the overall crash_prob_12m.

OUTPUT:
    {"Q1_0_3m": 0.25, "Q2_3_6m": 0.15, "Q3_6_9m": 0.10,
     "Q4_9_12m": 0.05, "beyond_12m": 0.45}
    → "Highest crash risk: Q1 — next 3 months"
"""

import numpy as np
import pandas as pd
from typing import Optional

try:
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False


WINDOWS = ["Q1_0_3m", "Q2_3_6m", "Q3_6_9m", "Q4_9_12m", "beyond_12m"]

# Trading days per quarter
_Q_DAYS = {
    "Q1_0_3m": (0, 63),
    "Q2_3_6m": (63, 126),
    "Q3_6_9m": (126, 189),
    "Q4_9_12m": (189, 252),
}


def _build_timing_labels(data: pd.DataFrame, crash_threshold: float = -0.20) -> pd.Series:
    """For each date, determine which 3-month window the next crash falls in.

    Returns:
        Series of integer labels aligned with data.index:
            0 = Q1 (crash in 0-3 months)
            1 = Q2 (crash in 3-6 months)
            2 = Q3 (crash in 6-9 months)
            3 = Q4 (crash in 9-12 months)
            4 = beyond 12m (no crash in next 12 months)
    """
    prices = data["SP500"].values
    n = len(prices)
    labels = np.full(n, 4, dtype=int)  # Default: beyond 12m

    for i in range(n - 252):
        # Forward-looking: find when the first crash occurs
        peak = prices[i]
        found = False
        for j in range(i + 1, min(i + 253, n)):
            peak = max(peak, prices[j])
            drawdown = (prices[j] - peak) / peak
            if drawdown <= crash_threshold:
                # Crash occurred at day j - i from date i
                days_ahead = j - i
                if days_ahead <= 63:
                    labels[i] = 0  # Q1
                elif days_ahead <= 126:
                    labels[i] = 1  # Q2
                elif days_ahead <= 189:
                    labels[i] = 2  # Q3
                else:
                    labels[i] = 3  # Q4
                found = True
                break

        if not found:
            labels[i] = 4  # No crash in next 12 months

    return pd.Series(labels, index=data.index, name="crash_window")


if _HAS_LIGHTGBM:

    class CrashTimingClassifier:
        """Predicts WHEN within the next 12 months a crash is most likely.

        Given the overall crash probability from the ensemble, this
        distributes it across 3-month windows to give investors
        actionable timing information.
        """

        def __init__(self, n_estimators: int = 400, random_state: int = 42):
            self.n_estimators = n_estimators
            self.random_state = random_state
            self.model = None
            self.feature_names = None
            self.is_trained = False
            self._class_priors = np.full(5, 0.2)

        def train(
            self,
            features: pd.DataFrame,
            data: pd.DataFrame,
            train_end_idx: Optional[int] = None,
            min_train_samples: int = 1260,
            crash_threshold: float = -0.20,
        ) -> dict:
            """Train the crash timing classifier.

            Args:
                features: Feature matrix (same as crash model uses).
                data: Raw market data (needs SP500 column for label construction).
                train_end_idx: Temporal cutoff.
                min_train_samples: Minimum training samples.
                crash_threshold: Drawdown threshold for crash definition.
            """
            # Build timing labels
            labels = _build_timing_labels(data, crash_threshold)

            if train_end_idx is not None:
                X = features.iloc[:train_end_idx].copy()
                y = labels.iloc[:train_end_idx].copy()
            else:
                X = features.copy()
                y = labels.copy()

            # Align and clean
            valid = y.notna() & X.notna().any(axis=1)
            X = X[valid]
            y = y[valid]

            if len(X) < min_train_samples:
                return {"success": False, "reason": f"Only {len(X)} samples < {min_train_samples}"}

            self.feature_names = list(X.columns)

            # Store class priors for calibration
            for c in range(5):
                self._class_priors[c] = max((y == c).mean(), 0.01)

            # Temporal weighting
            n = len(X)
            sample_weights = np.linspace(0.5, 1.5, n)

            # Train/val split with purge gap (labels are forward-looking 12 months)
            gap_days = 265  # Purge gap covers 12-month forward label window
            val_size = max(504, n // 5)
            split = n - val_size - gap_days

            if split < min_train_samples // 2:
                # Not enough data for proper purge — reduce gap
                split = int(n * 0.75)
                gap_days = 0

            train_X = X.iloc[:split]
            train_y = y.iloc[:split]
            train_w = sample_weights[:split]
            val_X = X.iloc[split + gap_days:]
            val_y = y.iloc[split + gap_days:]

            if len(val_y) < 50 or val_y.nunique() < 2:
                # Fallback: simple split without purge
                split = int(n * 0.8)
                train_X = X.iloc[:split]
                train_y = y.iloc[:split]
                train_w = sample_weights[:split]
                val_X = X.iloc[split:]
                val_y = y.iloc[split:]

            if val_y.nunique() < 2:
                eval_split = int(len(train_X) * 0.9)
                val_X = train_X.iloc[eval_split:]
                val_y = train_y.iloc[eval_split:]

            model = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=5,
                n_estimators=self.n_estimators,
                max_depth=5,
                num_leaves=25,
                learning_rate=0.01,
                min_child_samples=50,
                subsample=0.75,
                colsample_bytree=0.60,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state,
                verbose=-1,
                n_jobs=-1,
            )
            model.fit(
                train_X, train_y,
                sample_weight=train_w,
                eval_set=[(val_X, val_y)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )

            self.model = model
            self.is_trained = True

            # Compute validation accuracy
            val_pred = model.predict(val_X)
            accuracy = float((val_pred == val_y.values).mean())
            print(f"  [TIMING] Crash timing model: accuracy={accuracy:.2%}, "
                  f"classes={val_y.value_counts().to_dict()}")

            return {
                "success": True,
                "accuracy": accuracy,
                "n_train": len(train_X),
                "class_distribution": {WINDOWS[c]: float(self._class_priors[c]) for c in range(5)},
            }

        def predict_window(
            self,
            features: pd.DataFrame,
            crash_prob_12m: float = None,
        ) -> dict:
            """Predict probability distribution over 3-month crash windows.

            Args:
                features: Current feature row(s).
                crash_prob_12m: Overall 12-month crash probability from ensemble.
                    If provided, the Q1-Q4 probabilities are rescaled to sum to this.

            Returns:
                Dict mapping window names to probabilities.
                Example: {"Q1_0_3m": 0.25, "Q2_3_6m": 0.15, ..., "beyond_12m": 0.45}
            """
            if not self.is_trained:
                # Return uniform distribution
                return {w: 0.20 for w in WINDOWS}

            X = features[self.feature_names] if isinstance(features, pd.DataFrame) else features
            raw_probs = self.model.predict_proba(X)

            # Take last row if multiple
            if raw_probs.ndim == 2:
                probs = raw_probs[-1]
            else:
                probs = raw_probs

            # Rescale to match overall crash probability
            if crash_prob_12m is not None:
                # Q1-Q4 should sum to crash_prob_12m
                # beyond_12m = 1 - crash_prob_12m
                crash_windows = probs[:4]
                crash_sum = crash_windows.sum()
                if crash_sum > 1e-6:
                    scaled_crash = crash_windows * (crash_prob_12m / crash_sum)
                else:
                    scaled_crash = np.full(4, crash_prob_12m / 4)

                probs = np.append(scaled_crash, 1.0 - crash_prob_12m)
                probs = np.clip(probs, 0.0, 1.0)

            return {WINDOWS[i]: float(probs[i]) for i in range(5)}

        def get_most_likely_window(self, window_probs: dict) -> tuple:
            """Return the window with highest crash probability.

            Args:
                window_probs: Output from predict_window().

            Returns:
                (window_name, probability) tuple.
            """
            # Only consider crash windows (not "beyond")
            crash_windows = {k: v for k, v in window_probs.items() if k != "beyond_12m"}
            if not crash_windows:
                return ("beyond_12m", window_probs.get("beyond_12m", 1.0))
            best = max(crash_windows, key=crash_windows.get)
            return (best, crash_windows[best])

else:

    class CrashTimingClassifier:
        """Fallback when LightGBM is not installed."""

        def __init__(self, n_estimators: int = 400, random_state: int = 42):
            self.is_trained = False

        def train(self, features, data, train_end_idx=None, **kwargs):
            return {"success": False, "reason": "LightGBM not installed"}

        def predict_window(self, features, crash_prob_12m=None):
            return {w: 0.20 for w in WINDOWS}

        def get_most_likely_window(self, window_probs):
            return ("beyond_12m", 0.80)


__all__ = ["CrashTimingClassifier", "WINDOWS"]
