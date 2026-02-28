"""Compatibility wrapper re-exporting the real CrashPredictor.

This module attempts to import the implementation in `finpredict.ml.ml`.
If the real implementation cannot be imported (missing LightGBM or other
dependencies), a small fallback `CrashPredictor` is provided so the
pipeline can still run for testing / demos.
"""

try:
    from .ml.crash_model import CrashPredictor  # type: ignore
except Exception:
    import numpy as np

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
            # Targets may be a dict of horizons or a Series
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

        def get_top_features(self, n: int = 10):
            return []

__all__ = ["CrashPredictor"]
