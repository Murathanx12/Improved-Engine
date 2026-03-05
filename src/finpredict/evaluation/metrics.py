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
