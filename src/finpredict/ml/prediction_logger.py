"""
Prediction Logger — CSV Snapshots of Every Engine Run
======================================================

Appends one row per engine run to logs/prediction_log.csv with:
- Timestamp and prediction date
- All crash probabilities (per-model and ensemble, all horizons)
- Expected returns (point + quantile range)
- Top feature values at prediction time
- Model selection info and regime state
- SHAP top drivers

This creates a historical record for tracking model drift, calibration
monitoring, and post-hoc analysis of prediction accuracy.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from finpredict.config import PROJECT_ROOT


_LOG_DIR = PROJECT_ROOT / "logs"
_LOG_FILE = _LOG_DIR / "prediction_log.csv"

# Features to snapshot — key indicators that drive crash predictions
_SNAPSHOT_FEATURES = [
    "vix_zscore",
    "term_spread",
    "credit_spread_proxy",
    "mom_12m",
    "mom_6m",
    "vol_ratio_1m_12m",
    "erp",
    "sma_200d_dev",
    "dist_52w_high",
    "vol_1m",
]

_COLUMNS = [
    "run_timestamp",
    "prediction_date",
    # Ensemble predictions
    "crash_prob_3m",
    "crash_prob_6m",
    "crash_prob_12m",
    "expected_return_12m",
    "return_p10_12m",
    "return_p90_12m",
    # Per-model breakdown
    "lgb_crash_12m",
    "xgb_crash_12m",
    "lstm_crash_12m",
    "tcn_crash_12m",
    # Model selection
    "selected_model_12m",
    # Regime
    "current_regime",
    "hmm_regime_probs",
    # Crash timing
    "crash_timing_top_window",
    "crash_timing_top_prob",
    # Feature snapshot
    *[f"feat_{f}" for f in _SNAPSHOT_FEATURES],
    # SHAP top 3 drivers
    "shap_top1_name",
    "shap_top1_value",
    "shap_top2_name",
    "shap_top2_value",
    "shap_top3_name",
    "shap_top3_value",
]


def log_prediction(
    current_features: pd.DataFrame,
    ml_crash_prob: Optional[float] = None,
    ml_crash_3m: Optional[float] = None,
    ml_crash_6m: Optional[float] = None,
    ml_predicted_return: Optional[float] = None,
    ml_return_p10: Optional[float] = None,
    ml_return_p90: Optional[float] = None,
    lgb_crash_12m: Optional[float] = None,
    xgb_crash_12m: Optional[float] = None,
    lstm_crash_12m: Optional[float] = None,
    tcn_crash_12m: Optional[float] = None,
    selected_model: Optional[str] = None,
    current_regime: Optional[str] = None,
    hmm_regime_probs: Optional[np.ndarray] = None,
    crash_timing_results: Optional[dict] = None,
    shap_contributions: Optional[list] = None,
    crash_timing_model=None,
) -> Path:
    """Append a prediction snapshot row to the CSV log.

    Args:
        current_features: Feature DataFrame (uses last row for snapshot).
        All other args: prediction outputs from the engine pipeline.

    Returns:
        Path to the log file.
    """
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    row = current_features.iloc[-1]
    prediction_date = str(row.name.date()) if hasattr(row.name, "date") else str(row.name)

    # Feature snapshot
    feat_values = {}
    for f in _SNAPSHOT_FEATURES:
        val = float(row[f]) if f in row.index and pd.notna(row.get(f)) else None
        feat_values[f"feat_{f}"] = val

    # Crash timing top window
    timing_window = None
    timing_prob = None
    if crash_timing_results and crash_timing_model:
        try:
            timing_window, timing_prob = crash_timing_model.get_most_likely_window(
                crash_timing_results
            )
        except Exception:
            pass

    # SHAP top 3
    shap_entries = {}
    if shap_contributions:
        for i, (name, val) in enumerate(shap_contributions[:3], 1):
            shap_entries[f"shap_top{i}_name"] = name
            shap_entries[f"shap_top{i}_value"] = f"{val:.4f}"

    # HMM probs as string
    hmm_str = None
    if hmm_regime_probs is not None:
        try:
            hmm_str = ",".join(f"{p:.3f}" for p in hmm_regime_probs)
        except Exception:
            pass

    record = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prediction_date": prediction_date,
        "crash_prob_3m": ml_crash_3m,
        "crash_prob_6m": ml_crash_6m,
        "crash_prob_12m": ml_crash_prob,
        "expected_return_12m": ml_predicted_return,
        "return_p10_12m": ml_return_p10,
        "return_p90_12m": ml_return_p90,
        "lgb_crash_12m": lgb_crash_12m,
        "xgb_crash_12m": xgb_crash_12m,
        "lstm_crash_12m": lstm_crash_12m,
        "tcn_crash_12m": tcn_crash_12m,
        "selected_model_12m": selected_model,
        "current_regime": current_regime,
        "hmm_regime_probs": hmm_str,
        "crash_timing_top_window": timing_window,
        "crash_timing_top_prob": timing_prob,
        **feat_values,
        **shap_entries,
    }

    # Write header if file doesn't exist
    write_header = not _LOG_FILE.exists()

    with open(_LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(record)

    return _LOG_FILE
