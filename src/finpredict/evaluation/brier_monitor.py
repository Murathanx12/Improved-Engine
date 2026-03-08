"""
Rolling Brier Score Monitor — Detect Model Degradation
========================================================

Reads the prediction log (logs/prediction_log.csv), compares past
predictions against actual market outcomes, and computes a rolling
Brier score. If recent predictions are significantly worse than the
historical baseline, flags degradation.

Usage from main.py:
    monitor = RollingBrierMonitor()
    result = monitor.evaluate_and_check(data)
    if result["degraded"]:
        print("WARNING: Model degradation detected")
"""

import pandas as pd
from pathlib import Path
from typing import Optional

from finpredict.config import PROJECT_ROOT


_LOG_FILE = PROJECT_ROOT / "logs" / "prediction_log.csv"
_ROLLING_FILE = PROJECT_ROOT / "logs" / "brier_rolling.csv"


class RollingBrierMonitor:
    """Monitor crash prediction accuracy over time using rolling Brier scores."""

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or _LOG_FILE

    def _load_log(self) -> pd.DataFrame:
        """Load prediction log CSV. Returns empty DataFrame if missing."""
        if not self.log_path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.log_path, parse_dates=["prediction_date"])
            return df
        except Exception:
            return pd.DataFrame()

    def evaluate_past_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Check past predictions against actual outcomes.

        For each logged prediction, computes whether a crash (>= 20% drawdown)
        actually occurred within the predicted horizon by looking at SP500 data.

        Args:
            data: Market DataFrame with SP500 column and DatetimeIndex.

        Returns:
            DataFrame with added columns: actual_crash_12m, brier_12m.
            Only rows with sufficient elapsed time are evaluated.
        """
        log = self._load_log()
        if log.empty or "crash_prob_12m" not in log.columns:
            return pd.DataFrame()

        sp500 = data["SP500"]
        results = []

        for _, row in log.iterrows():
            pred_date = pd.Timestamp(row["prediction_date"])
            horizon_end = pred_date + pd.Timedelta(days=365)

            # Only evaluate if enough time has elapsed
            if horizon_end > sp500.index.max():
                continue

            crash_prob = row.get("crash_prob_12m")
            if pd.isna(crash_prob):
                continue

            # Compute actual max drawdown over the 12m window
            mask = (sp500.index >= pred_date) & (sp500.index <= horizon_end)
            window = sp500[mask]
            if len(window) < 20:
                continue

            peak = window.expanding().max()
            drawdown = (window - peak) / peak
            max_dd = drawdown.min()
            actual_crash = 1.0 if max_dd <= -0.20 else 0.0

            brier = (crash_prob - actual_crash) ** 2

            results.append(
                {
                    "prediction_date": pred_date,
                    "crash_prob_12m": crash_prob,
                    "actual_crash_12m": actual_crash,
                    "max_drawdown_12m": float(max_dd),
                    "brier_12m": float(brier),
                }
            )

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)

    def compute_rolling_brier(self, evaluated: pd.DataFrame, window: int = 8) -> pd.Series:
        """Compute rolling Brier score over the last N evaluated predictions.

        Args:
            evaluated: Output of evaluate_past_predictions.
            window: Number of predictions in the rolling window.

        Returns:
            Series of rolling Brier scores.
        """
        if evaluated.empty or "brier_12m" not in evaluated.columns:
            return pd.Series(dtype=float)
        return evaluated["brier_12m"].rolling(window, min_periods=2).mean()

    def detect_degradation(
        self,
        evaluated: pd.DataFrame,
        threshold_pct: float = 0.15,
        recent_window: int = 4,
    ) -> dict:
        """Compare recent Brier score vs full history to detect degradation.

        Args:
            evaluated: Output of evaluate_past_predictions.
            threshold_pct: If recent Brier > baseline * (1 + threshold), flag.
            recent_window: Number of most recent predictions to compare.

        Returns:
            Dict with degraded, current_brier, baseline_brier, pct_change.
        """
        if evaluated.empty or len(evaluated) < recent_window:
            return {
                "degraded": False,
                "current_brier": None,
                "baseline_brier": None,
                "pct_change": None,
                "evaluated": len(evaluated) if not evaluated.empty else 0,
                "reason": "insufficient_data",
            }

        baseline_brier = float(evaluated["brier_12m"].mean())
        recent = evaluated.tail(recent_window)
        current_brier = float(recent["brier_12m"].mean())

        if baseline_brier <= 0:
            pct_change = 0.0
            degraded = False
        else:
            pct_change = (current_brier - baseline_brier) / baseline_brier
            degraded = pct_change > threshold_pct

        return {
            "degraded": degraded,
            "current_brier": current_brier,
            "baseline_brier": baseline_brier,
            "pct_change": pct_change,
            "evaluated": len(evaluated),
            "reason": "degraded" if degraded else "ok",
        }

    def generate_report(self, evaluated: pd.DataFrame) -> dict:
        """Save rolling Brier CSV and return summary."""
        if evaluated.empty:
            return {"saved": False, "evaluated": 0}

        rolling = self.compute_rolling_brier(evaluated)
        evaluated = evaluated.copy()
        evaluated["rolling_brier"] = rolling.values

        _ROLLING_FILE.parent.mkdir(parents=True, exist_ok=True)
        evaluated.to_csv(_ROLLING_FILE, index=False)

        return {
            "saved": True,
            "path": str(_ROLLING_FILE),
            "evaluated": len(evaluated),
            "mean_brier": float(evaluated["brier_12m"].mean()),
            "latest_rolling": float(rolling.iloc[-1])
            if not rolling.empty and pd.notna(rolling.iloc[-1])
            else None,
        }

    def evaluate_and_check(self, data: pd.DataFrame) -> dict:
        """Full pipeline: evaluate past predictions and check for degradation.

        Args:
            data: Market DataFrame with SP500 column.

        Returns:
            Dict with evaluated count, rolling_brier, degraded flag.
        """
        evaluated = self.evaluate_past_predictions(data)
        degradation = self.detect_degradation(evaluated)

        rolling_brier = None
        if not evaluated.empty:
            rolling = self.compute_rolling_brier(evaluated)
            if not rolling.empty and pd.notna(rolling.iloc[-1]):
                rolling_brier = float(rolling.iloc[-1])
            self.generate_report(evaluated)

        return {
            "evaluated": degradation["evaluated"],
            "rolling_brier": rolling_brier,
            "degraded": degradation["degraded"],
            "current_brier": degradation["current_brier"],
            "baseline_brier": degradation["baseline_brier"],
        }
