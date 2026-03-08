"""
Anomaly Detection — Isolation Forest + Bayesian Changepoint Detection
=====================================================================

WHY THIS EXISTS:
    When market conditions look like nothing in the training data, the ML
    models' predictions should NOT be trusted at face value. An Isolation
    Forest flags days that are statistical outliers — "regime unknown" —
    so the report can warn the user that the model is extrapolating.

    Bayesian Online Changepoint Detection (BOCPD) identifies the exact day
    when the statistical properties of the return series shift — detecting
    regime transitions in real time rather than in hindsight.

USAGE:
    detector = AnomalyDetector()
    detector.fit(feature_matrix_train)

    # At prediction time:
    anomaly_score = detector.score(current_features)
    is_anomalous = detector.is_anomalous(current_features)

    changepoints = detector.detect_changepoints(returns_series)
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    """Isolation Forest anomaly detector for market conditions.

    Flags days where the feature vector looks unlike anything in the
    training history. When anomaly_score < threshold, the model is
    extrapolating and predictions should be treated with lower confidence.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self._feature_names = None

    def fit(self, features: pd.DataFrame) -> dict:
        """Fit the Isolation Forest on historical feature data.

        Args:
            features: Feature matrix (all backward-looking features).

        Returns:
            dict with fit statistics.
        """
        self._feature_names = list(features.columns)
        X = features.values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)
        self.is_fitted = True

        scores = self.model.decision_function(X_scaled)
        return {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "anomaly_threshold": float(np.percentile(scores, self.contamination * 100)),
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std()),
        }

    def score(self, features: pd.DataFrame) -> np.ndarray:
        """Compute anomaly scores for given features.

        Lower scores = more anomalous. Negative = flagged as anomaly.

        Returns:
            Array of anomaly scores (one per row).
        """
        if not self.is_fitted:
            raise RuntimeError("AnomalyDetector not fitted — call fit() first")

        X = features[self._feature_names].values if isinstance(features, pd.DataFrame) else features
        X = np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)

    def is_anomalous(self, features: pd.DataFrame) -> np.ndarray:
        """Boolean array: True where current conditions are anomalous."""
        if not self.is_fitted:
            return np.zeros(len(features), dtype=bool)
        scores = self.score(features)
        return scores < 0  # Isolation Forest convention: negative = anomaly

    def anomaly_report(self, features: pd.DataFrame) -> dict:
        """Generate a human-readable anomaly assessment.

        Returns:
            dict with anomaly status, score, and confidence adjustment.
        """
        if not self.is_fitted:
            return {"status": "UNKNOWN", "score": 0.0, "confidence_factor": 1.0}

        scores = self.score(features)
        latest_score = float(scores[-1]) if len(scores) > 0 else 0.0
        is_anom = latest_score < 0

        if is_anom:
            # Scale confidence reduction by how extreme the anomaly is
            confidence_factor = max(0.3, min(1.0, 1.0 + latest_score))
            status = "ANOMALOUS"
        else:
            confidence_factor = 1.0
            status = "NORMAL"

        return {
            "status": status,
            "score": latest_score,
            "is_anomalous": is_anom,
            "confidence_factor": confidence_factor,
            "interpretation": (
                f"Current market conditions are {'UNLIKE' if is_anom else 'consistent with'} "
                f"historical patterns (score={latest_score:.3f}). "
                f"{'Model predictions may be unreliable — extrapolating beyond training data.' if is_anom else 'Model predictions should be normally reliable.'}"
            ),
        }


class BayesianChangepoint:
    """Bayesian Online Changepoint Detection (BOCPD).

    Detects when the statistical properties of the return series shift,
    identifying regime transitions in real time.

    Based on Adams & MacKay (2007) "Bayesian Online Changepoint Detection".
    Uses a simple Gaussian model with known variance for speed.
    """

    def __init__(self, hazard_rate: float = 1 / 252, mu_prior: float = 0.0, var_prior: float = 1.0):
        """
        Args:
            hazard_rate: Prior probability of changepoint at each step.
                1/252 = expect ~1 changepoint per year.
            mu_prior: Prior mean of the observation model.
            var_prior: Prior variance of the observation model.
        """
        self.hazard_rate = hazard_rate
        self.mu_prior = mu_prior
        self.var_prior = var_prior

    def detect(self, returns: pd.Series, window: int = 60) -> pd.DataFrame:
        """Run BOCPD on a return series.

        Args:
            returns: Daily log returns.
            window: Only compute over the last `window` days for efficiency.

        Returns:
            DataFrame with columns:
                - run_length_prob: Probability of being at each run length
                - changepoint_prob: Probability that a changepoint occurred
                - regime_age: Expected number of days since last changepoint
        """
        x = returns.dropna().values[-window:] if window else returns.dropna().values
        n = len(x)
        if n < 10:
            return pd.DataFrame(
                {
                    "changepoint_prob": np.zeros(len(returns)),
                    "regime_age": np.full(len(returns), float(len(returns))),
                },
                index=returns.index,
            )

        # Run length probabilities: R(t) = P(run_length = r at time t)
        # R is (n+1,) at each step; we track the growth probability
        max_run = n + 1
        R = np.zeros((n + 1, max_run))
        R[0, 0] = 1.0

        # Sufficient statistics for online Gaussian
        np.full(max_run, self.mu_prior)
        np.full(max_run, self.var_prior)
        counts = np.zeros(max_run)
        sums = np.zeros(max_run)
        sum_sq = np.zeros(max_run)

        changepoint_probs = np.zeros(n)
        regime_ages = np.zeros(n)

        h = self.hazard_rate

        for t in range(n):
            # Observation likelihood for each possible run length
            predprobs = np.zeros(t + 1)
            for r in range(t + 1):
                if counts[r] < 2:
                    # Use prior
                    predprobs[r] = _gaussian_pdf(x[t], self.mu_prior, self.var_prior)
                else:
                    mean = sums[r] / counts[r]
                    var = max(sum_sq[r] / counts[r] - mean**2, 1e-10) + self.var_prior / counts[r]
                    predprobs[r] = _gaussian_pdf(x[t], mean, var)

            # Growth probabilities
            R[t + 1, 1 : t + 2] = R[t, : t + 1] * predprobs * (1 - h)
            # Changepoint probability
            R[t + 1, 0] = np.sum(R[t, : t + 1] * predprobs * h)

            # Normalize
            evidence = R[t + 1, : t + 2].sum()
            if evidence > 0:
                R[t + 1, : t + 2] /= evidence

            changepoint_probs[t] = float(R[t + 1, 0])

            # Expected run length (regime age)
            run_lengths = np.arange(t + 2)
            regime_ages[t] = float(np.sum(run_lengths * R[t + 1, : t + 2]))

            # Update sufficient statistics
            new_counts = counts[: t + 1] + 1
            new_sums = sums[: t + 1] + x[t]
            new_sum_sq = sum_sq[: t + 1] + x[t] ** 2

            counts[1 : t + 2] = new_counts
            sums[1 : t + 2] = new_sums
            sum_sq[1 : t + 2] = new_sum_sq
            counts[0] = 0
            sums[0] = 0
            sum_sq[0] = 0

        # Align with original index
        result_index = returns.dropna().index[-n:] if window else returns.dropna().index
        result = pd.DataFrame(
            {
                "changepoint_prob": changepoint_probs,
                "regime_age": regime_ages,
            },
            index=result_index,
        )

        return result.reindex(returns.index).fillna(0)

    def recent_changepoint(
        self, returns: pd.Series, window: int = 60, threshold: float = 0.30
    ) -> dict:
        """Check if a recent changepoint was detected.

        Returns:
            dict with:
                - detected: bool
                - days_ago: int (days since last changepoint above threshold)
                - max_prob: float (highest changepoint probability in window)
        """
        result = self.detect(returns, window=window)
        cp = result["changepoint_prob"].dropna()

        if len(cp) == 0:
            return {"detected": False, "days_ago": window, "max_prob": 0.0}

        max_prob = float(cp.max())
        if max_prob >= threshold:
            days_ago = int(len(cp) - cp.values.argmax() - 1)
            return {"detected": True, "days_ago": days_ago, "max_prob": max_prob}
        return {"detected": False, "days_ago": window, "max_prob": max_prob}


def _gaussian_pdf(x: float, mu: float, var: float) -> float:
    """Gaussian probability density."""
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)


__all__ = ["AnomalyDetector", "BayesianChangepoint"]
