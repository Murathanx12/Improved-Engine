"""
Cox Proportional Hazard Model for Crash Prediction
======================================================

Models crash timing as a survival analysis problem: for each date,
how many days until the next >=20% drawdown?

WHY THIS APPROACH:
    - Naturally handles censored observations (no crash within horizon)
    - Produces time-varying hazard rates (crash prob at 3m, 6m, 12m from
      a single model)
    - Cox PH is semi-parametric: learns feature effects without assuming
      a specific hazard distribution
    - Regularized (penalizer=0.1) to handle sparse crash events

FEATURES:
    Uses the same 10 LOGISTIC_FEATURES from CrashPredictor — these are
    domain-motivated, low-correlation indicators that work well with
    limited positive examples.

INTERFACE:
    Follows the same train/predict_proba pattern as CrashPredictor and
    XGBoostCrashPredictor for seamless integration into the backtest
    and meta-stacker pipeline.
"""

import logging
import numpy as np
import pandas as pd

from finpredict.config import config

logger = logging.getLogger(__name__)

# Same features as logistic crash model — proven domain-motivated set
COX_FEATURES = [
    "vix_zscore",
    "term_spread",
    "credit_spread_proxy",
    "mom_12m",
    "vol_ratio_1m_12m",
    "mom_6m",
    "erp",
    "sma_200d_dev",
    "dist_52w_high",
    "vol_1m",
]

HORIZON_DAYS = {
    "3m": 63,
    "6m": 126,
    "12m": 252,
}

try:
    from lifelines import CoxPHFitter
    from sklearn.preprocessing import StandardScaler

    _HAS_LIFELINES = True
except ImportError:
    _HAS_LIFELINES = False


def _build_survival_targets(
    data: pd.DataFrame,
    max_horizon: int = 252,
    threshold: float = -0.20,
) -> pd.DataFrame:
    """Build survival analysis targets from price data.

    For each date t, look forward up to max_horizon days.
    Find the first day where drawdown from t exceeds threshold.

    Returns:
        DataFrame with columns 'duration' and 'event', indexed like data.
    """
    prices = data["SP500"].values.astype(float)
    n = len(prices)
    durations = np.full(n, max_horizon, dtype=float)
    events = np.zeros(n, dtype=float)

    for i in range(n - 1):
        end = min(n, i + max_horizon + 1)
        window = prices[i:end]
        if len(window) <= 1:
            continue
        peak = np.maximum.accumulate(window)
        mask = peak > 0
        if not mask.any():
            continue
        dd = np.where(mask, (window - peak) / peak, 0.0)
        crash_indices = np.where(dd <= threshold)[0]
        if len(crash_indices) > 0:
            first_crash = crash_indices[0]
            if first_crash > 0:  # Don't count day 0
                durations[i] = float(first_crash)
                events[i] = 1.0

    return pd.DataFrame(
        {"duration": durations, "event": events},
        index=data.index,
    )


class CrashSurvivalModel:
    """Cox PH crash prediction model.

    Produces crash probabilities at multiple horizons from a single
    fitted model by evaluating the survival function S(t) at different
    time points: P(crash within t) = 1 - S(t).
    """

    def __init__(self, penalizer: float = 0.1):
        self.penalizer = penalizer
        self.is_trained = False
        self._model = None
        self._scaler = None
        self._fill_values = None
        self._available_features = []
        self._base_rate = config.get("ml", {}).get("crash_base_rate_fallback", 0.12)

    def train(
        self,
        features: pd.DataFrame,
        targets: dict,  # unused, kept for interface compat
        train_end_idx: int,
        min_train_samples: int = 1260,
        data: pd.DataFrame = None,
    ) -> dict:
        """Train the Cox PH model.

        Args:
            features: Feature matrix from build_feature_matrix.
            targets: Unused (survival targets built internally from data).
            train_end_idx: Temporal cutoff index.
            min_train_samples: Minimum training samples required.
            data: Raw market data (must contain SP500 for target construction).

        Returns:
            Dict with success status and validation metrics.
        """
        if not _HAS_LIFELINES:
            logger.warning("[COX] lifelines not installed — using fallback")
            return {"success": False, "reason": "lifelines not installed"}

        if data is None:
            return {"success": False, "reason": "data required for survival targets"}

        if train_end_idx < min_train_samples:
            return {
                "success": False,
                "reason": f"Need {min_train_samples} samples, have {train_end_idx}",
            }

        # Build survival targets from price data — only use data up to
        # train_end_idx + 252 so forward-looking labels don't peek beyond
        # the backtest cutoff into the validation/test period.
        target_data_end = min(train_end_idx + 252, len(data))
        surv = _build_survival_targets(
            data.iloc[:target_data_end],
            max_horizon=252,
            threshold=-config["risk"]["crash_threshold"],
        )

        # Select available features
        self._available_features = [f for f in COX_FEATURES if f in features.columns]
        if len(self._available_features) < 3:
            return {
                "success": False,
                "reason": f"Only {len(self._available_features)} features available",
            }

        # Only use targets up to train_end_idx - 252 for training:
        # observations near the cutoff have labels that look beyond it.
        purge_gap = config.get("ml", {}).get("purge_gaps", {}).get("12m", 265)
        train_end = train_end_idx - 252  # purge for forward-looking labels
        val_start = train_end + purge_gap
        if train_end < min_train_samples:
            train_end = int(train_end_idx * 0.7)
            val_start = train_end + purge_gap

        X_train = features[self._available_features].iloc[:train_end]
        surv_train = surv.iloc[:train_end]

        # Remove NaN rows
        valid = ~(X_train.isna().any(axis=1) | surv_train.isna().any(axis=1))
        # Also filter out zero-duration rows
        valid = valid & (surv_train["duration"] > 0)
        X_train = X_train[valid]
        surv_train = surv_train[valid]

        if len(X_train) < 100 or surv_train["event"].sum() < 3:
            return {
                "success": False,
                "reason": f"Insufficient data: {len(X_train)} rows, {surv_train['event'].sum():.0f} events",
            }

        # Store training medians for NaN filling at prediction time
        self._fill_values = X_train.median()

        # Scale features (Cox PH is sensitive to feature scales)
        self._scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self._scaler.fit_transform(X_train),
            columns=self._available_features,
            index=X_train.index,
        )

        # Build training DataFrame for lifelines
        train_df = pd.concat([X_scaled, surv_train], axis=1)

        try:
            self._model = CoxPHFitter(penalizer=self.penalizer)
            self._model.fit(
                train_df,
                duration_col="duration",
                event_col="event",
                show_progress=False,
            )
        except Exception as e:
            logger.warning(f"[COX] CoxPHFitter.fit failed: {e}")
            return {"success": False, "reason": str(e)}

        # Validation concordance
        val_concordance = None
        if val_start < len(features):
            val_end = min(train_end_idx, len(features))
            X_val = features[self._available_features].iloc[val_start:val_end]
            surv_val = surv.iloc[val_start:val_end]
            valid_val = ~(X_val.isna().any(axis=1) | surv_val.isna().any(axis=1))
            valid_val = valid_val & (surv_val["duration"] > 0)
            X_val = X_val[valid_val]
            surv_val = surv_val[valid_val]
            if len(X_val) >= 20:
                try:
                    X_val_s = pd.DataFrame(
                        self._scaler.transform(X_val),
                        columns=self._available_features,
                        index=X_val.index,
                    )
                    val_df = pd.concat([X_val_s, surv_val], axis=1)
                    val_concordance = self._model.score(val_df)
                except Exception:
                    pass

        self.is_trained = True
        result = {
            "success": True,
            "n_train": len(X_train),
            "n_events": int(surv_train["event"].sum()),
        }
        if val_concordance is not None:
            result["val_concordance"] = float(val_concordance)
            print(
                f"  [COX] Trained: {len(X_train)} samples, "
                f"{int(surv_train['event'].sum())} events, "
                f"concordance={val_concordance:.3f}"
            )
        else:
            print(
                f"  [COX] Trained: {len(X_train)} samples, "
                f"{int(surv_train['event'].sum())} events"
            )

        return result

    def predict_proba(
        self,
        features: pd.DataFrame,
        horizon: str = "12m",
    ) -> np.ndarray:
        """Predict crash probability at the given horizon.

        Args:
            features: Feature matrix (1 or more rows).
            horizon: "3m", "6m", or "12m".

        Returns:
            Array of crash probabilities, clipped to [0.02, 0.98].

        Raises:
            RuntimeError: If model has not been trained.
        """
        if not self.is_trained:
            raise RuntimeError("CrashSurvivalModel has not been trained")

        horizon_days = HORIZON_DAYS.get(horizon, 252)

        available = [f for f in self._available_features if f in features.columns]
        if len(available) < len(self._available_features):
            X = features.reindex(columns=self._available_features)
        else:
            X = features[self._available_features]

        # Fill NaN with training medians (not zero — zero is a false neutral)
        X = X.fillna(self._fill_values)

        X_scaled = pd.DataFrame(
            self._scaler.transform(X),
            columns=self._available_features,
            index=X.index,
        )

        try:
            surv_func = self._model.predict_survival_function(X_scaled)
            # surv_func is a DataFrame with time points as index.
            # Interpolate to get S(t) at the exact horizon day rather
            # than snapping to the nearest event time.
            times = surv_func.index.values.astype(float)
            if horizon_days <= times[0]:
                s_t = surv_func.iloc[0].values
            elif horizon_days >= times[-1]:
                s_t = surv_func.iloc[-1].values
            else:
                # Linear interpolation between bracketing event times
                right = np.searchsorted(times, horizon_days)
                left = right - 1
                t_lo, t_hi = times[left], times[right]
                frac = (horizon_days - t_lo) / (t_hi - t_lo) if t_hi != t_lo else 0.0
                s_t = surv_func.iloc[left].values * (1 - frac) + surv_func.iloc[right].values * frac
            crash_prob = 1.0 - s_t
        except Exception:
            crash_prob = np.full(len(X), self._base_rate)

        return np.clip(crash_prob, 0.02, 0.98)

    def get_top_features(self, n: int = 5) -> list[tuple]:
        """Return top features by absolute Cox coefficient magnitude.

        Returns:
            List of (feature_name, coefficient) tuples.
        """
        if not self.is_trained or self._model is None:
            return []
        coefs = self._model.params_
        sorted_feats = coefs.abs().sort_values(ascending=False)
        return [(name, float(coefs[name])) for name in sorted_feats.index[:n]]


class _CrashSurvivalFallback:
    """Fallback when lifelines is not installed. Returns base rate."""

    def __init__(self):
        self.is_trained = False
        self._base_rate = config.get("ml", {}).get("crash_base_rate_fallback", 0.12)

    def train(self, *args, **kwargs):
        return {"success": False, "reason": "lifelines not installed"}

    def predict_proba(self, features, horizon="12m"):
        if not self.is_trained:
            raise RuntimeError("CrashSurvivalModel has not been trained")
        return np.full(len(features), self._base_rate)

    def get_top_features(self, n=5):
        return []


# Export the right class based on availability
if not _HAS_LIFELINES:
    CrashSurvivalModel = _CrashSurvivalFallback  # type: ignore[misc]

__all__ = ["CrashSurvivalModel"]
