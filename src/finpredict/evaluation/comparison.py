"""
Baseline Model Comparisons
============================

Simple baseline crash probability models for benchmarking:
- VIX threshold: P(crash) = 1 if VIX > 25
- Yield curve inversion: P(crash) = 1 if 10Y-3M < 0
- Climatology: Always predict the historical base rate

These baselines represent the "bar" an ML model must clear.
A positive Brier Skill Score vs these baselines is the primary
result needed for an academic publication.
"""

import numpy as np
import pandas as pd

from finpredict.evaluation.metrics import (
    brier_score,
    brier_skill_score,
    reliability_diagram,
    prediction_spread_check,
)


def baseline_crash_prob_vix(
    vix_series: pd.Series,
    threshold: float = 25,
) -> pd.Series:
    """Simple binary baseline: P(crash) = 1.0 if VIX > threshold else 0.0.

    Args:
        vix_series: VIX values.
        threshold: VIX level above which crash is predicted.

    Returns:
        Binary crash probability series.
    """
    return (vix_series > threshold).astype(float)


def baseline_crash_prob_yield_curve(spread_series: pd.Series) -> pd.Series:
    """Simple binary baseline: P(crash) = 1.0 if 10Y-3M < 0 else 0.0.

    Args:
        spread_series: 10Y minus 3M treasury yield spread.

    Returns:
        Binary crash probability series.
    """
    return (spread_series < 0).astype(float)


def baseline_crash_prob_climatology(y_true: pd.Series) -> float:
    """Return the historical base rate of crashes in the training window.

    Args:
        y_true: Binary crash outcomes.

    Returns:
        Historical crash frequency (float between 0 and 1).
    """
    y = np.asarray(y_true, dtype=float)
    return float(y.mean()) if len(y) > 0 else 0.0


def compare_models(results: dict) -> pd.DataFrame:
    """Compare multiple models across standard evaluation metrics.

    Args:
        results: Dict of {model_name: {"y_pred": array, "y_true": array,
                 optional "vix_series": Series, "spread_series": Series}}.

    Returns:
        DataFrame with columns: model, brier_score, bss_vs_climatology,
        bss_vs_vix, bss_vs_yield_curve, auc, calibration_error,
        is_underdispersed.
    """
    from sklearn.metrics import roc_auc_score

    rows = []
    for name, data in results.items():
        y_pred = np.asarray(data["y_pred"], dtype=float)
        y_true = np.asarray(data["y_true"], dtype=float)

        bs = brier_score(y_pred, y_true)
        bss_clim = brier_skill_score(y_pred, y_true, baseline="climatology")

        # VIX baseline (if available)
        bss_vix = float("nan")
        if "vix_series" in data and data["vix_series"] is not None:
            bss_vix = brier_skill_score(
                y_pred, y_true, baseline="vix25",
                vix_series=data["vix_series"],
            )

        # Yield curve baseline (if available)
        bss_yc = float("nan")
        if "spread_series" in data and data["spread_series"] is not None:
            bss_yc = brier_skill_score(
                y_pred, y_true, baseline="yield_curve",
                spread_series=data["spread_series"],
            )

        # AUC
        try:
            n_unique = len(set(y_true))
            auc = float(roc_auc_score(y_true, y_pred)) if n_unique >= 2 else float("nan")
        except Exception:
            auc = float("nan")

        # Calibration
        rel = reliability_diagram(y_pred, y_true)
        spread = prediction_spread_check(y_pred)

        rows.append({
            "model": name,
            "brier_score": bs,
            "bss_vs_climatology": bss_clim,
            "bss_vs_vix": bss_vix,
            "bss_vs_yield_curve": bss_yc,
            "auc": auc,
            "calibration_error": rel["calibration_error"],
            "is_underdispersed": spread["is_underdispersed"],
        })

    return pd.DataFrame(rows)
