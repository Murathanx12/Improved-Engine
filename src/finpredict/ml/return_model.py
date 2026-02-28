"""Compatibility wrapper re-exporting the real ReturnPredictor.

This wrapper tries to import the full implementation. If LightGBM is
not available, a lightweight fallback `ReturnPredictor` is provided
which returns conservative, zero-centered predictions.
"""

try:
    from .ml.return_model import ReturnPredictor  # type: ignore
except Exception:
    import numpy as np

    class ReturnPredictor:
        """Fallback return predictor when LightGBM is missing.

        This fallback provides the same public methods used by the
        rest of the codebase: `train`, `predict`, `predict_quantiles`,
        `get_top_features` and `is_trained` attribute.
        """

        def __init__(self, n_estimators: int = 100, random_state: int = 42):
            self.n_estimators = n_estimators
            self.random_state = random_state
            self.models = {}
            self.feature_names = None
            self.feature_importances_ = None
            self.is_trained = False

        def train(self, features, targets, train_end_idx=None, min_train_samples=1260):
            # Accept either Series or dict of horizons
            if isinstance(targets, dict):
                t = targets.get("12m", list(targets.values())[0])
            else:
                t = targets
            if train_end_idx is not None:
                t = t.iloc[:train_end_idx]
            t = t.dropna()
            if len(t) < 1:
                return {"success": False, "reason": "no valid target samples"}
            # simple stats
            self.is_trained = True
            self.feature_names = list(features.columns) if hasattr(features, "columns") else None
            return {"success": True, "val_mae": float(np.abs(t).mean()), "val_corr": 0.0, "pred_range": (float(t.min()), float(t.max())), "skill_score": 0.0}

        def predict(self, features, horizon: str = "12m"):
            n = len(features)
            # return zeros (no predicted drift)
            return np.zeros(n)

        def predict_quantiles(self, features, horizon: str = "12m"):
            n = len(features)
            median = np.zeros(n)
            p10 = median - 0.10
            p90 = median + 0.10
            return {"median": median, "p10": p10, "p90": p90}

        def predict_all_horizons(self, features):
            return {"12m": self.predict(features)}

        def get_top_features(self, n: int = 15):
            return []

__all__ = ["ReturnPredictor"]
