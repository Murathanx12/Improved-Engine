"""
Evaluation Metrics for Crash Probability Models
=================================================

Core metrics for evaluating calibrated probability predictions:
- Brier Score (BS): Mean squared error of probability predictions
- Brier Skill Score (BSS): Relative improvement over a baseline
- Reliability Diagram: Calibration curve with ECE
- ROC AUC over time: Rolling discrimination ability
- Prediction Spread Check: Detects underdispersed (collapsed) models

Reference:
    Brier, G. W. (1950). Verification of forecasts expressed in terms
    of probability. Monthly Weather Review, 78(1), 1–3.
"""

import numpy as np
import pandas as pd
from typing import Optional


def brier_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute the Brier Score (mean squared error of probability predictions).

    Args:
        y_pred: Predicted probabilities in [0, 1].
        y_true: Binary outcomes (0 or 1).

    Returns:
        Brier Score (lower is better; 0 = perfect, 0.25 = random coin flip).
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if len(y_pred) == 0:
        return float("nan")
    return float(np.mean((y_pred - y_true) ** 2))


def brier_skill_score(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    baseline: str = "climatology",
    vix_series: Optional[pd.Series] = None,
    spread_series: Optional[pd.Series] = None,
) -> float:
    """Compute the Brier Skill Score relative to a baseline.

    BSS = 1 - (BS_model / BS_baseline)
    Positive BSS means the model beats the baseline.

    Args:
        y_pred: Predicted probabilities in [0, 1].
        y_true: Binary outcomes (0 or 1).
        baseline: One of "climatology", "vix25", "yield_curve".
        vix_series: VIX values aligned with y_pred (required for "vix25").
        spread_series: 10Y-3M yield spread aligned with y_pred (required for "yield_curve").

    Returns:
        BSS (positive = model beats baseline, negative = model worse than baseline).
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    bs_model = brier_score(y_pred, y_true)

    if baseline == "climatology":
        base_rate = y_true.mean()
        bs_baseline = brier_score(np.full_like(y_true, base_rate), y_true)
    elif baseline == "vix25":
        if vix_series is None:
            raise ValueError("vix_series required for 'vix25' baseline")
        vix_arr = np.asarray(vix_series, dtype=float)
        baseline_pred = (vix_arr > 25).astype(float)
        bs_baseline = brier_score(baseline_pred, y_true)
    elif baseline == "yield_curve":
        if spread_series is None:
            raise ValueError("spread_series required for 'yield_curve' baseline")
        spread_arr = np.asarray(spread_series, dtype=float)
        baseline_pred = (spread_arr < 0).astype(float)
        bs_baseline = brier_score(baseline_pred, y_true)
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    if bs_baseline == 0:
        return 0.0
    return float(1.0 - bs_model / bs_baseline)


def reliability_diagram(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute reliability diagram data and Expected Calibration Error (ECE).

    Args:
        y_pred: Predicted probabilities in [0, 1].
        y_true: Binary outcomes (0 or 1).
        n_bins: Number of bins for the calibration curve.

    Returns:
        dict with keys:
            bin_centers: Center of each probability bin.
            bin_frequencies: Observed event frequency per bin.
            bin_counts: Number of samples per bin.
            calibration_error: Expected Calibration Error (ECE).
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = np.zeros(n_bins)
    bin_frequencies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    total = len(y_pred)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_pred >= lo) & (y_pred <= hi)
        else:
            mask = (y_pred >= lo) & (y_pred < hi)

        count = mask.sum()
        bin_counts[i] = count
        bin_centers[i] = (lo + hi) / 2

        if count > 0:
            bin_frequencies[i] = y_true[mask].mean()
            ece += abs(bin_frequencies[i] - bin_centers[i]) * count / total

    return {
        "bin_centers": bin_centers,
        "bin_frequencies": bin_frequencies,
        "bin_counts": bin_counts,
        "calibration_error": float(ece),
    }


def roc_auc_over_time(predictions_df: pd.DataFrame) -> pd.Series:
    """Compute rolling 2-year AUC from a predictions DataFrame.

    Args:
        predictions_df: DataFrame with columns: date, y_pred, y_true.

    Returns:
        pd.Series indexed by date with rolling AUC values.
    """
    from sklearn.metrics import roc_auc_score

    df = predictions_df.sort_values("date").copy()
    window_days = 252 * 2  # 2 years of trading days

    results = {}
    dates = df["date"].values

    for i in range(len(df)):
        cutoff = dates[i] - pd.Timedelta(days=window_days)
        window = df[(df["date"] >= cutoff) & (df["date"] <= dates[i])]

        if len(window) < 10:
            continue
        if window["y_true"].nunique() < 2:
            continue

        try:
            auc = roc_auc_score(window["y_true"], window["y_pred"])
            results[dates[i]] = auc
        except Exception:
            continue

    return pd.Series(results, name="rolling_auc")


def prediction_spread_check(y_pred: np.ndarray) -> dict:
    """Check prediction spread for underdispersion.

    Underdispersed predictions (std < 5%) indicate the model has collapsed
    to predicting near the base rate for everything — useless in practice.

    Args:
        y_pred: Predicted probabilities.

    Returns:
        dict with: mean, std, min, max, is_underdispersed.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    std = float(y_pred.std()) if len(y_pred) > 0 else 0.0
    return {
        "mean": float(y_pred.mean()) if len(y_pred) > 0 else 0.0,
        "std": std,
        "min": float(y_pred.min()) if len(y_pred) > 0 else 0.0,
        "max": float(y_pred.max()) if len(y_pred) > 0 else 0.0,
        "is_underdispersed": std < 0.05,
    }


# ═══════════════════════════════════════════════════════════════════════
# ADVANCED VALIDATION METRICS
# ═══════════════════════════════════════════════════════════════════════


def lead_time_accuracy(
    bt: pd.DataFrame,
    crash_periods: pd.DataFrame,
    prob_col: str = "ml_crash_12m",
    prob_threshold: float = 0.40,
) -> dict:
    """Measure how many days before each crash the model raised an alert.

    For each historical crash, finds the earliest date where crash_prob
    exceeded prob_threshold within the 12 months preceding the crash.

    Target: > 30 days average lead time.
    """
    if bt.empty or crash_periods.empty or prob_col not in bt.columns:
        return {"mean_lead_days": 0, "median_lead_days": 0, "per_crash": [],
                "n_crashes": 0, "n_detected": 0}

    bt_sorted = bt.sort_values("date").copy()
    lead_times = []
    per_crash = []

    for _, crash in crash_periods.iterrows():
        crash_start = pd.Timestamp(crash["start"])
        lookback_start = crash_start - pd.Timedelta(days=365)

        mask = (bt_sorted["date"] >= lookback_start) & (bt_sorted["date"] < crash_start)
        window = bt_sorted[mask]

        if window.empty:
            per_crash.append({"crash_start": str(crash_start.date()),
                              "detected": False, "lead_days": 0})
            continue

        alerts = window[window[prob_col] >= prob_threshold]
        if not alerts.empty:
            first_alert = alerts["date"].iloc[0]
            lead_days = (crash_start - pd.Timestamp(first_alert)).days
            lead_times.append(lead_days)
            per_crash.append({
                "crash_start": str(crash_start.date()),
                "detected": True,
                "lead_days": lead_days,
                "first_alert_prob": float(alerts[prob_col].iloc[0]),
            })
        else:
            per_crash.append({"crash_start": str(crash_start.date()),
                              "detected": False, "lead_days": 0})

    return {
        "mean_lead_days": float(np.mean(lead_times)) if lead_times else 0,
        "median_lead_days": float(np.median(lead_times)) if lead_times else 0,
        "per_crash": per_crash,
        "n_crashes": len(crash_periods),
        "n_detected": len(lead_times),
    }


def false_alarm_rate(
    bt: pd.DataFrame,
    prob_col: str = "ml_crash_12m",
    actual_col: str = "actual_crash_12m",
    alarm_threshold: float = 0.60,
    horizon_days: int = 63,
) -> dict:
    """Fraction of high-probability alerts where no crash occurred.

    Groups consecutive alarm dates into episodes, checks if a crash
    occurred within horizon_days of each episode start.

    Target: < 30%.
    """
    if bt.empty or prob_col not in bt.columns:
        return {"rate": 0.0, "n_alarms": 0, "n_false_alarms": 0}

    bt_sorted = bt.sort_values("date").copy()
    alarms = bt_sorted[bt_sorted[prob_col] >= alarm_threshold]

    if alarms.empty:
        return {"rate": 0.0, "n_alarms": 0, "n_false_alarms": 0}

    # Group into episodes (gap > horizon_days = new episode)
    episodes = []
    prev_date = None
    for _, row in alarms.iterrows():
        dt = pd.Timestamp(row["date"])
        if prev_date is None or (dt - prev_date).days > horizon_days:
            episodes.append(dt)
        prev_date = dt

    n_false = 0
    for ep_start in episodes:
        ep_end = ep_start + pd.Timedelta(days=horizon_days)
        future = bt_sorted[
            (bt_sorted["date"] >= ep_start) & (bt_sorted["date"] <= ep_end)
        ]
        if future.empty or not future[actual_col].any():
            n_false += 1

    n_episodes = len(episodes)
    rate = n_false / n_episodes if n_episodes > 0 else 0.0

    return {"rate": float(rate), "n_alarms": n_episodes, "n_false_alarms": n_false}


def missed_crash_rate(
    bt: pd.DataFrame,
    crash_periods: pd.DataFrame,
    prob_col: str = "ml_crash_12m",
    safe_threshold: float = 0.30,
    lead_days: int = 21,
) -> dict:
    """Fraction of crashes where model said 'safe' (prob < threshold) beforehand.

    Most dangerous failure mode. Checks crash_prob at lead_days before
    each crash start.

    Target: < 20%.
    """
    if bt.empty or crash_periods.empty or prob_col not in bt.columns:
        return {"rate": 0.0, "n_crashes": 0, "n_missed": 0}

    bt_sorted = bt.sort_values("date").copy()
    n_missed = 0
    n_evaluated = 0

    for _, crash in crash_periods.iterrows():
        crash_start = pd.Timestamp(crash["start"])
        check_date = crash_start - pd.Timedelta(days=lead_days)

        before = bt_sorted[bt_sorted["date"] <= check_date]
        if before.empty:
            continue

        prob = float(before.iloc[-1][prob_col])
        n_evaluated += 1
        if prob < safe_threshold:
            n_missed += 1

    rate = n_missed / n_evaluated if n_evaluated > 0 else 0.0
    return {"rate": float(rate), "n_crashes": n_evaluated, "n_missed": n_missed}


def regime_persistence_iou(
    hmm_regimes: pd.Series,
    crash_periods: pd.DataFrame,
    regime_label: str = "Bear",
) -> dict:
    """Intersection-over-Union between HMM bear regime and actual bear markets.

    Measures overlap between HMM's bear regime days and actual crash periods.

    Target: IoU > 0.70.
    """
    if hmm_regimes.empty or crash_periods.empty:
        return {"iou": 0.0, "hmm_bear_days": 0, "actual_bear_days": 0, "overlap_days": 0}

    all_dates = hmm_regimes.index
    hmm_bear = set(all_dates[hmm_regimes == regime_label])

    actual_bear = set()
    for _, crash in crash_periods.iterrows():
        start = pd.Timestamp(crash["start"])
        end = pd.Timestamp(crash["end"])
        period_dates = all_dates[(all_dates >= start) & (all_dates <= end)]
        actual_bear.update(period_dates)

    intersection = hmm_bear & actual_bear
    union = hmm_bear | actual_bear
    iou = len(intersection) / len(union) if len(union) > 0 else 0.0

    return {
        "iou": float(iou),
        "hmm_bear_days": len(hmm_bear),
        "actual_bear_days": len(actual_bear),
        "overlap_days": len(intersection),
    }


def directional_accuracy_vs_consensus(
    bt: pd.DataFrame,
    consensus_return: float,
    return_col: str = "ml_return_12m",
    actual_col: str = "actual_return_12m",
) -> dict:
    """When engine disagrees with consensus direction, who was right?

    Target: engine correct > 55% of divergences.
    """
    if bt.empty or return_col not in bt.columns or actual_col not in bt.columns:
        return {"engine_correct_rate": 0.0, "n_divergences": 0, "n_engine_correct": 0}

    n_divergences = 0
    n_engine_correct = 0

    for _, row in bt.iterrows():
        engine_pred = float(row[return_col])
        actual = float(row[actual_col])
        engine_bullish = engine_pred > 0
        consensus_bullish = consensus_return > 0

        if engine_bullish != consensus_bullish:
            n_divergences += 1
            if abs(engine_pred - actual) < abs(consensus_return - actual):
                n_engine_correct += 1

    rate = n_engine_correct / n_divergences if n_divergences > 0 else 0.0
    return {
        "engine_correct_rate": float(rate),
        "n_divergences": n_divergences,
        "n_engine_correct": n_engine_correct,
    }


def signal_max_drawdown(
    bt: pd.DataFrame,
    prob_col: str = "ml_crash_12m",
    price_col: str = "start_price",
    short_threshold: float = 0.50,
) -> dict:
    """Max drawdown if investor shorted SPX when crash_prob > threshold.

    Target: max DD < 25%.
    """
    if bt.empty or prob_col not in bt.columns or price_col not in bt.columns:
        return {"max_drawdown": 0.0, "total_return": 0.0, "sharpe": 0.0}

    bt_sorted = bt.sort_values("date").copy()
    cumulative = 1.0
    peak = 1.0
    max_dd = 0.0
    period_returns = []

    for i in range(len(bt_sorted) - 1):
        row = bt_sorted.iloc[i]
        next_row = bt_sorted.iloc[i + 1]

        price_now = float(row[price_col])
        price_next = float(next_row[price_col])
        if price_now <= 0:
            continue

        market_return = price_next / price_now - 1.0
        crash_prob = float(row[prob_col])

        strategy_return = -market_return if crash_prob > short_threshold else 0.0
        period_returns.append(strategy_return)
        cumulative *= (1 + strategy_return)
        peak = max(peak, cumulative)
        dd = (cumulative - peak) / peak if peak > 0 else 0.0
        max_dd = min(max_dd, dd)

    total_return = cumulative - 1.0
    if period_returns:
        arr = np.array(period_returns)
        # Annualize based on number of observations per year
        # bt_sorted has step_months spacing between rows; infer from data
        if len(bt_sorted) >= 2:
            avg_days = (bt_sorted["date"].iloc[-1] - bt_sorted["date"].iloc[0]).days / max(len(bt_sorted) - 1, 1)
            n_periods_per_year = max(1, 365.25 / avg_days)
        else:
            n_periods_per_year = 4  # default quarterly
        sharpe = (arr.mean() / arr.std() * np.sqrt(n_periods_per_year)) if arr.std() > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "max_drawdown": float(abs(max_dd)),
        "total_return": float(total_return),
        "sharpe": float(sharpe),
    }


def regime_conditional_bss(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    regimes: np.ndarray,
) -> dict:
    """M3: Compute BSS stratified by regime label.

    Args:
        y_pred: Predicted crash probabilities.
        y_true: Binary crash outcomes.
        regimes: Regime labels aligned with predictions (e.g., "Bull", "Bear", "Crisis").

    Returns:
        Dict with overall BSS and per-regime BSS/count.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    regimes = np.asarray(regimes)

    result = {"overall_bss": brier_skill_score(y_pred, y_true)}

    for regime in np.unique(regimes):
        mask = regimes == regime
        n = int(mask.sum())
        if n < 10 or len(np.unique(y_true[mask])) < 2:
            result[f"bss_{regime}"] = float("nan")
            result[f"n_{regime}"] = n
            continue
        result[f"bss_{regime}"] = brier_skill_score(y_pred[mask], y_true[mask])
        result[f"n_{regime}"] = n

    return result


class ConformalPredictor:
    """N3: Split conformal prediction for coverage-guaranteed crash intervals.

    Given calibration residuals, produces prediction intervals with
    finite-sample coverage guarantees (Vovk et al., 2005).

    Usage:
        cp = ConformalPredictor(target_coverage=0.90)
        cp.calibrate(cal_predictions, cal_true_labels)
        interval = cp.predict_interval(new_prediction)
        # interval = {"lower": 0.15, "upper": 0.45, "coverage": 0.90}
    """

    def __init__(self, target_coverage: float = 0.90):
        self.target_coverage = target_coverage
        self._quantile = None
        self._calibrated = False

    def calibrate(self, cal_pred: np.ndarray, cal_true: np.ndarray) -> None:
        """Compute conformal quantile from calibration residuals.

        Args:
            cal_pred: Calibration set predicted probabilities.
            cal_true: Calibration set true binary labels.
        """
        cal_pred = np.asarray(cal_pred, dtype=float)
        cal_true = np.asarray(cal_true, dtype=float)
        n = len(cal_pred)
        if n < 10:
            self._quantile = 0.5  # fallback
            self._calibrated = True
            return

        # Nonconformity scores: absolute residuals
        scores = np.abs(cal_pred - cal_true)

        # Conformal quantile with finite-sample correction
        q_level = np.ceil((n + 1) * self.target_coverage) / n
        q_level = min(q_level, 1.0)
        self._quantile = float(np.quantile(scores, q_level))
        self._calibrated = True

    def predict_interval(self, point_pred: float) -> dict:
        """Produce a prediction interval around a point prediction.

        Args:
            point_pred: The model's point probability prediction.

        Returns:
            Dict with lower, upper bounds and target coverage.
        """
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before predict_interval()")

        lower = max(0.0, point_pred - self._quantile)
        upper = min(1.0, point_pred + self._quantile)

        return {
            "lower": float(lower),
            "upper": float(upper),
            "point": float(point_pred),
            "coverage": self.target_coverage,
        }

    def predict_intervals_batch(self, predictions: np.ndarray) -> list:
        """Produce intervals for a batch of predictions."""
        return [self.predict_interval(float(p)) for p in np.asarray(predictions)]
