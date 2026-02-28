"""
ML-First Walk-Forward Backtest (v7 — Discrimination-Focused)
===============================================================

WHY THE OLD VERSION FAILED:
    1. ML models trained but predictions never used (old main.py ran instead)
    2. Crash calibration bins showed 0/102/0 split — zero discrimination
    3. MC crash probabilities (not ML) were being reported as "predictions"
    4. No tracking of prediction SPREAD over time

WHAT'S CHANGED:
    1. ML predictions are the PRIMARY output — MC is secondary
    2. Tracks prediction spread: if std < 5%, model is useless
    3. Proper discrimination metrics: AUC, calibration slope, reliability diagram
    4. Retrains every 3 months with expanding window (more freshness)
    5. Returns ML models in .attrs for use by main pipeline
    6. Quantile regression evaluation (p10/p90 coverage)

HOW IT WORKS:
    Walk-forward backtesting ensures ZERO data leakage:
    
    For each prediction date t (every 3 months from 2000 to present):
        1. Build features using ONLY data up to date t
        2. Train ML models on ALL data before t (expanding window)
        3. Make predictions for the next 3m, 6m, 12m
        4. Record predictions
        5. After simulation ends, compare predictions to actual outcomes
    
    This is the honest test: can the model predict the future using
    only information available at the time of prediction?
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional

from finpredict.config import config, get_institutional_return
from finpredict.simulation.monte_carlo import simulate_paths
from finpredict.ml.features import (
    build_feature_matrix, build_target_crash, build_target_return,
    build_target_crash_multi, build_target_return_multi,
)
from finpredict.ml.crash_model import CrashPredictor
from finpredict.ml.return_model import ReturnPredictor


def run_backtest(
    data: pd.DataFrame,
    crash_freq: float,
    fred_data: dict = None,
) -> pd.DataFrame:
    """
    ML-First Walk-Forward Backtest.

    At each prediction point:
        1. Build features from historical data (including FRED time series)
        2. Train/update ML models on all prior data (expanding window)
        3. Record ML predictions (crash prob + expected return + quantiles)
        4. Run Monte Carlo for uncertainty bands
        5. Compare to actual outcomes

    The KEY metrics are:
        - Crash model: Brier score, AUC, prediction range, calibration
        - Return model: MAE, correlation, skill score, quantile coverage
        - MC model: MAPE, band coverage, directional accuracy

    Returns:
        pd.DataFrame with per-prediction results and metrics in .attrs
    """
    print("[MODULE 6] Running ML walk-forward backtest (2000 → present)...")

    bt_cfg = config["backtest"]
    risk_cfg = config["risk"]
    backtest_start = pd.Timestamp(config["data"]["backtest_start"])
    forward_days = bt_cfg["forward_days"]

    pred_dates = pd.date_range(
        backtest_start,
        data.index[-1] - pd.Timedelta(days=forward_days + 30),
        freq=f'{bt_cfg["step_months"]}MS',
    )

    # ── Pre-compute features and targets ──────────────────────────
    print("  [ML] Building feature matrix with FRED time series...")
    features = build_feature_matrix(data, fred_data=fred_data)
    n_features = len(features.columns)
    print(f"  [ML] {n_features} features built")

    # Multi-horizon targets
    crash_targets = build_target_crash_multi(data, threshold=-risk_cfg["crash_threshold"])
    return_targets = build_target_return_multi(data)

    # ── Initialize ML models ──────────────────────────────────────
    crash_model = CrashPredictor(n_estimators=800)
    return_model = ReturnPredictor(n_estimators=600)

    last_train_idx = -9999
    retrain_interval = 63         # Retrain every ~3 months (was 126 in v6)
    min_train_samples = 1260      # Need 5+ years of data

    results = []
    ml_train_log = []

    for pred_date in tqdm(pred_dates, desc="  Backtesting"):
        idx = data.index.get_indexer([pred_date], method="nearest")[0]

        if idx < max(min_train_samples, 504 + 252):
            continue

        # ── Retrain ML models periodically ────────────────────────
        if idx - last_train_idx >= retrain_interval:
            crash_result = crash_model.train(
                features, crash_targets,
                train_end_idx=idx,
                min_train_samples=min_train_samples,
            )
            return_result = return_model.train(
                features, return_targets,
                train_end_idx=idx,
                min_train_samples=min_train_samples,
            )

            if crash_result.get("success") and return_result.get("success"):
                last_train_idx = idx
                ml_train_log.append({
                    "date": pred_date,
                    "crash_val_auc": crash_result.get("val_auc", 0),
                    "crash_val_brier": crash_result.get("val_brier", 0),
                    "crash_pred_range": crash_result.get("pred_range", (0, 0)),
                    "crash_pred_std": crash_result.get("pred_std", 0),
                    "crash_discrimination": crash_result.get("discrimination", "UNKNOWN"),
                    "return_val_mae": return_result.get("val_mae", 0),
                    "return_val_corr": return_result.get("val_corr", 0),
                    "return_pred_range": return_result.get("pred_range", (0, 0)),
                    "return_skill": return_result.get("skill_score", 0),
                    "return_quantile_cov": return_result.get("quantile_coverage", 0),
                })

        if not crash_model.is_trained or not return_model.is_trained:
            continue

        # ── Get ML predictions ────────────────────────────────────
        current_features = features.iloc[idx:idx+1]
        try:
            ml_crash_12m = float(crash_model.predict_proba(current_features, "12m")[0])
            ml_crash_6m = float(crash_model.predict_proba(current_features, "6m")[0]) \
                if "6m" in crash_model.models else None
            ml_crash_3m = float(crash_model.predict_proba(current_features, "3m")[0]) \
                if "3m" in crash_model.models else None

            ml_return_12m = float(return_model.predict(current_features, "12m")[0])
            ml_return_6m = float(return_model.predict(current_features, "6m")[0]) \
                if "6m" in return_model.models else None
            ml_return_3m = float(return_model.predict(current_features, "3m")[0]) \
                if "3m" in return_model.models else None

            # Quantile predictions for uncertainty
            quantiles = return_model.predict_quantiles(current_features, "12m")
            ml_return_p10 = float(quantiles.get("p10", [ml_return_12m - 0.15])[0])
            ml_return_p90 = float(quantiles.get("p90", [ml_return_12m + 0.15])[0])
        except Exception:
            continue

        # ── Monte Carlo for uncertainty bands ─────────────────────
        lookback = bt_cfg["lookback_years"] * 252
        hist = data.iloc[max(0, idx - lookback):idx]
        if len(hist) < 252:
            continue

        returns = hist["SP500"].pct_change().dropna()
        log_returns = np.log(1 + returns)
        hist_mu = log_returns.mean() * 252
        sigma = returns.std() * np.sqrt(252)
        start_price = float(data["SP500"].iloc[idx])
        risk = float(data["Risk_Score"].iloc[idx]) if "Risk_Score" in data.columns else 0

        base_scenario = {"drift_adj": 0, "vol_mult": 1.0, "crash_mult": 1.0}
        paths = simulate_paths(
            start_price, hist_mu, sigma, forward_days,
            bt_cfg["simulations_per_point"], crash_freq, risk, base_scenario,
            ml_crash_prob=ml_crash_12m,
            ml_predicted_return=ml_return_12m,
            ml_return_p10=ml_return_p10,
            ml_return_p90=ml_return_p90,
        )

        pred_mean = float(paths[-1].mean())
        pred_p95 = float(np.percentile(paths[-1], 95))
        pred_p05 = float(np.percentile(paths[-1], 5))

        # ── Actual outcomes ───────────────────────────────────────
        actual_idx_12m = min(idx + 252, len(data) - 1)
        actual_idx_6m = min(idx + 126, len(data) - 1)
        actual_idx_3m = min(idx + 63, len(data) - 1)

        actual_price_12m = float(data["SP500"].iloc[actual_idx_12m])
        actual_return_12m = actual_price_12m / start_price - 1.0

        # Check for actual crashes
        def _check_crash(start_idx, end_idx, thresh=-risk_cfg["crash_threshold"]):
            fwd = data["SP500"].iloc[start_idx:end_idx + 1]
            pk = fwd.expanding().max()
            dd = ((fwd - pk) / pk).min()
            return dd <= thresh

        actual_crash_12m = _check_crash(idx, actual_idx_12m)
        actual_crash_6m = _check_crash(idx, actual_idx_6m)
        actual_crash_3m = _check_crash(idx, actual_idx_3m)

        # MC crash probability (secondary)
        sim_peak = np.maximum.accumulate(paths, axis=0)
        sim_dd = (paths - sim_peak) / sim_peak
        sim_crash = float((sim_dd.min(axis=0) <= -risk_cfg["crash_threshold"]).mean())

        error = pred_mean - actual_price_12m
        pct_error = (error / actual_price_12m) * 100
        within_bounds = (actual_price_12m >= pred_p05) and (actual_price_12m <= pred_p95)

        results.append({
            "date": pred_date,
            "start_price": start_price,
            "actual_price": actual_price_12m,
            "pred_mean": pred_mean,
            "pred_p95": pred_p95,
            "pred_p05": pred_p05,
            "pct_error": pct_error,
            "within_bounds": within_bounds,
            # ML crash predictions (PRIMARY)
            "ml_crash_12m": ml_crash_12m,
            "ml_crash_6m": ml_crash_6m,
            "ml_crash_3m": ml_crash_3m,
            # ML return predictions (PRIMARY)
            "ml_return_12m": ml_return_12m,
            "ml_return_6m": ml_return_6m,
            "ml_return_3m": ml_return_3m,
            "ml_return_p10": ml_return_p10,
            "ml_return_p90": ml_return_p90,
            # Actual outcomes
            "actual_crash_12m": actual_crash_12m,
            "actual_crash_6m": actual_crash_6m,
            "actual_crash_3m": actual_crash_3m,
            "actual_return_12m": actual_return_12m,
            "actual_return_6m": float(data["SP500"].iloc[actual_idx_6m]) / start_price - 1,
            "actual_return_3m": float(data["SP500"].iloc[actual_idx_3m]) / start_price - 1,
            # MC secondary
            "sim_crash_prob": sim_crash,
            # Context
            "risk_score": risk,
            "direction_correct": (pred_mean > start_price) == (actual_price_12m > start_price),
        })

    bt = pd.DataFrame(results)

    if len(bt) > 0:
        _print_results(bt, crash_model, return_model, ml_train_log)
        bt.attrs.update(_compute_metrics(bt, crash_model, return_model, ml_train_log))

    return bt


# ═══════════════════════════════════════════════════════════════════════
# RESULTS PRINTING & METRICS
# ═══════════════════════════════════════════════════════════════════════

def _print_results(bt, crash_model, return_model, ml_train_log):
    """Print comprehensive ML evaluation results with discrimination focus."""
    mape = np.abs(bt["pct_error"]).mean()
    coverage = bt["within_bounds"].mean() * 100
    direction = bt["direction_correct"].mean() * 100

    print(f"\n  [RESULTS] ML Walk-Forward Backtest:")
    print(f"  Predictions:          {len(bt)}")
    print(f"  MC MAPE:              {mape:.1f}%")
    print(f"  90% Band Coverage:    {coverage:.1f}%")
    print(f"  Directional Acc:      {direction:.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # ML CRASH MODEL EVALUATION
    # ═══════════════════════════════════════════════════════════════
    bt_c = bt.copy()
    bt_c["actual_crash_int"] = bt_c["actual_crash_12m"].astype(float)
    ml_brier = float(((bt_c["ml_crash_12m"] - bt_c["actual_crash_int"]) ** 2).mean())
    crash_range = (bt["ml_crash_12m"].min(), bt["ml_crash_12m"].max())
    crash_std = bt["ml_crash_12m"].std()
    crash_mean = bt["ml_crash_12m"].mean()

    try:
        from sklearn.metrics import roc_auc_score
        crash_auc = roc_auc_score(bt_c["actual_crash_int"], bt["ml_crash_12m"])
    except Exception:
        crash_auc = 0.5

    print(f"\n  ═══ ML CRASH MODEL (12-month) ═══")
    print(f"  Brier Score:          {ml_brier:.4f} (0.25 = random, lower = better)")
    print(f"  AUC:                  {crash_auc:.3f} (0.5 = random, higher = better)")
    print(f"  Prediction Range:     [{crash_range[0]*100:.1f}%, {crash_range[1]*100:.1f}%]")
    print(f"  Prediction Std Dev:   {crash_std*100:.1f}%")
    print(f"  Prediction Mean:      {crash_mean*100:.1f}%")

    # DISCRIMINATION CHECK — this is the key metric
    if crash_std < 0.05:
        print(f"  ⚠ WARNING: Low discrimination (std={crash_std*100:.1f}% < 5%)")
        print(f"    Model predictions are clustered — limited predictive value")
    else:
        print(f"  ✓ Good discrimination (std={crash_std*100:.1f}% > 5%)")

    # Fine-grained calibration
    bins = [(0, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.50), (0.50, 1.0)]
    print(f"  Crash Calibration:")
    for lo, hi in bins:
        mask = (bt["ml_crash_12m"] >= lo) & (bt["ml_crash_12m"] < hi)
        subset = bt[mask]
        if len(subset) > 0:
            actual_rate = float(subset["actual_crash_12m"].mean() * 100)
            pred_avg = float(subset["ml_crash_12m"].mean() * 100)
            print(f"    [{lo*100:.0f}-{hi*100:.0f}%]: predicted avg {pred_avg:.1f}%, "
                  f"actual {actual_rate:.1f}% (n={len(subset)})")
        else:
            print(f"    [{lo*100:.0f}-{hi*100:.0f}%]: no predictions")

    # Coarse summary
    low = bt[bt["ml_crash_12m"] < 0.15]
    med = bt[(bt["ml_crash_12m"] >= 0.15) & (bt["ml_crash_12m"] < 0.40)]
    high = bt[bt["ml_crash_12m"] >= 0.40]
    cal_low = float(low["actual_crash_12m"].mean() * 100) if len(low) > 0 else 0
    cal_med = float(med["actual_crash_12m"].mean() * 100) if len(med) > 0 else 0
    cal_high = float(high["actual_crash_12m"].mean() * 100) if len(high) > 0 else 0
    print(f"  Summary: Low<15%: {cal_low:.0f}% actual (n={len(low)}) | "
          f"Med 15-40%: {cal_med:.0f}% actual (n={len(med)}) | "
          f"High>40%: {cal_high:.0f}% actual (n={len(high)})")

    # ═══════════════════════════════════════════════════════════════
    # ML RETURN MODEL EVALUATION
    # ═══════════════════════════════════════════════════════════════
    ml_return_mae = float(np.abs(bt["actual_return_12m"] - bt["ml_return_12m"]).mean())
    corr_val = bt[["actual_return_12m", "ml_return_12m"]].corr().iloc[0, 1]
    ml_return_corr = float(corr_val if not np.isnan(corr_val) else 0)
    ret_range = (bt["ml_return_12m"].min(), bt["ml_return_12m"].max())

    # Skill score vs naive
    naive_pred = bt["actual_return_12m"].mean()
    naive_mae = float(np.abs(bt["actual_return_12m"] - naive_pred).mean())
    skill = 1 - ml_return_mae / naive_mae if naive_mae > 0 else 0

    # Quantile coverage
    if "ml_return_p10" in bt.columns and "ml_return_p90" in bt.columns:
        q_mask = bt["ml_return_p10"].notna() & bt["ml_return_p90"].notna()
        if q_mask.sum() > 0:
            bt_q = bt[q_mask]
            q_coverage = float(((bt_q["actual_return_12m"] >= bt_q["ml_return_p10"]) &
                               (bt_q["actual_return_12m"] <= bt_q["ml_return_p90"])).mean())
        else:
            q_coverage = 0
    else:
        q_coverage = 0

    print(f"\n  ═══ ML RETURN MODEL (12-month) ═══")
    print(f"  Return MAE:           {ml_return_mae*100:.1f}%")
    print(f"  Return Correlation:   {ml_return_corr:.3f}")
    print(f"  Prediction Range:     [{ret_range[0]*100:.1f}%, {ret_range[1]*100:.1f}%]")
    print(f"  Naive MAE:            {naive_mae*100:.1f}%")
    print(f"  Skill Score:          {skill:.3f} (>0 = better than naive)")
    if q_coverage > 0:
        print(f"  Quantile Coverage:    {q_coverage*100:.1f}% (target: 80%)")

    # ═══════════════════════════════════════════════════════════════
    # TRAINING LOG
    # ═══════════════════════════════════════════════════════════════
    if ml_train_log:
        last = ml_train_log[-1]
        print(f"\n  [ML TRAINING]")
        print(f"  Models retrained:     {len(ml_train_log)} times")
        print(f"  Last Crash AUC:       {last.get('crash_val_auc', 0):.3f}")
        print(f"  Last Return Corr:     {last.get('return_val_corr', 0):.3f}")
        print(f"  Last Return Skill:    {last.get('return_skill', 0):.3f}")
        disc = last.get('crash_discrimination', 'UNKNOWN')
        print(f"  Discrimination:       {disc}")

    # Top features
    if crash_model.is_trained:
        top = crash_model.get_top_features(10)
        if top:
            print(f"\n  [TOP CRASH PREDICTORS]")
            for feat, imp in top:
                print(f"    {feat}: {imp:.1f}")


def _compute_metrics(bt, crash_model, return_model, ml_train_log):
    """Compute all metrics for DataFrame .attrs."""
    bt_c = bt.copy()
    bt_c["actual_crash_int"] = bt_c["actual_crash_12m"].astype(float)

    mape = float(np.abs(bt["pct_error"]).mean())
    coverage = float(bt["within_bounds"].mean() * 100)
    direction = float(bt["direction_correct"].mean() * 100)
    ml_brier = float(((bt_c["ml_crash_12m"] - bt_c["actual_crash_int"]) ** 2).mean())
    ml_return_mae = float(np.abs(bt["actual_return_12m"] - bt["ml_return_12m"]).mean())

    corr_val = bt[["actual_return_12m", "ml_return_12m"]].corr().iloc[0, 1]
    ml_return_corr = float(corr_val if not np.isnan(corr_val) else 0)

    try:
        from sklearn.metrics import roc_auc_score
        crash_auc = float(roc_auc_score(bt_c["actual_crash_int"], bt["ml_crash_12m"]))
    except Exception:
        crash_auc = 0.5

    low = bt[bt["ml_crash_12m"] < 0.15]
    med = bt[(bt["ml_crash_12m"] >= 0.15) & (bt["ml_crash_12m"] < 0.40)]
    high = bt[bt["ml_crash_12m"] >= 0.40]

    # Naive baseline
    naive_mae = float(np.abs(bt["actual_return_12m"] - bt["actual_return_12m"].mean()).mean())
    skill = 1 - ml_return_mae / naive_mae if naive_mae > 0 else 0

    return {
        "mape": mape,
        "coverage": coverage,
        "direction": direction,
        "brier_score": ml_brier,
        "crash_auc": crash_auc,
        "ml_return_mae": ml_return_mae,
        "ml_return_corr": ml_return_corr,
        "ml_return_skill": float(skill),
        "cal_low": float(low["actual_crash_12m"].mean() * 100) if len(low) > 0 else 0,
        "cal_med": float(med["actual_crash_12m"].mean() * 100) if len(med) > 0 else 0,
        "cal_high": float(high["actual_crash_12m"].mean() * 100) if len(high) > 0 else 0,
        "n_low": len(low), "n_med": len(med), "n_high": len(high),
        "ml_train_log": ml_train_log,
        "crash_model": crash_model,
        "return_model": return_model,
        "crash_pred_range": (float(bt["ml_crash_12m"].min()), float(bt["ml_crash_12m"].max())),
        "crash_pred_std": float(bt["ml_crash_12m"].std()),
        "return_pred_range": (float(bt["ml_return_12m"].min()), float(bt["ml_return_12m"].max())),
    }
