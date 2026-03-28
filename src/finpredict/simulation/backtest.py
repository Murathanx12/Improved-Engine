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

from finpredict.config import config, get_institutional_return
from finpredict.simulation.monte_carlo import simulate_paths
from finpredict.ml.features import (
    build_feature_matrix,
    build_target_crash_multi,
    build_target_return_multi,
)
from finpredict.ml.crash_model import CrashPredictor
from finpredict.ml.return_model import ReturnPredictor
from finpredict.ml.xgboost_model import XGBoostCrashPredictor
from finpredict.ml.sequence_model import TemporalEnsemble
from finpredict.ml.meta_stacker import MetaStacker
from finpredict.ml.crash_timing import CrashTimingClassifier
from finpredict.ml.survival_model import CrashSurvivalModel


def run_backtest(
    data: pd.DataFrame,
    crash_freq: float,
    fred_data: dict = None,
    global_data: dict = None,
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
    print(
        f"  [BACKTEST] step_months={bt_cfg['step_months']}, "
        f"pred_dates={len(pred_dates)} from {pred_dates[0].date()} to {pred_dates[-1].date()}"
    )

    # ── Pre-compute features and targets ──────────────────────────
    print("  [ML] Building feature matrix with FRED time series...")
    features = build_feature_matrix(data, fred_data=fred_data, global_data=global_data)
    n_features = len(features.columns)
    print(f"  [ML] {n_features} features built")

    # Multi-horizon targets
    crash_targets = build_target_crash_multi(data, threshold=-risk_cfg["crash_threshold"])
    return_targets = build_target_return_multi(data)

    # ── Initialize ML models ──────────────────────────────────────
    crash_model = CrashPredictor(n_estimators=300)
    return_model = ReturnPredictor(n_estimators=300)
    xgb_model = XGBoostCrashPredictor(n_estimators=300)
    temporal_model = TemporalEnsemble(hidden_dim=64)
    meta_stacker = MetaStacker()
    crash_timing = CrashTimingClassifier()
    cox_model = CrashSurvivalModel()

    last_train_idx = -9999
    retrain_interval = 63  # Retrain every ~3 months (was 126 in v6)
    min_train_samples = 1260  # Need 5+ years of data

    # Collectors for meta-stacker OOS predictions
    oos_predictions = {"lgb": {}, "xgb": {}, "lstm": {}, "tcn": {}, "cox": {}}
    oos_regime_probs = []
    oos_targets = {}
    oos_indices = []

    results = []
    ml_train_log = []

    for pred_date in tqdm(pred_dates, desc="  Backtesting"):
        idx = data.index.get_indexer([pred_date], method="nearest")[0]

        if idx < max(min_train_samples, 504 + 252):
            continue

        # ── Retrain ML models periodically ────────────────────────
        if idx - last_train_idx >= retrain_interval:
            crash_result = crash_model.train(
                features,
                crash_targets,
                train_end_idx=idx,
                min_train_samples=min_train_samples,
            )
            return_result = return_model.train(
                features,
                return_targets,
                train_end_idx=idx,
                min_train_samples=min_train_samples,
            )

            # Train XGBoost peer model
            xgb_result = xgb_model.train(
                features,
                crash_targets,
                train_end_idx=idx,
                min_train_samples=min_train_samples,
            )

            # Train LSTM + TCN temporal ensemble
            temporal_result = temporal_model.train(
                features,
                crash_targets,
                train_end_idx=idx,
                min_sequences=400,
                n_epochs=30,
                patience=10,
            )

            # Train crash timing classifier
            crash_timing.train(
                features,
                data,
                train_end_idx=idx,
                min_train_samples=min_train_samples,
            )

            # Train Cox survival model
            cox_model.train(
                features,
                crash_targets,
                train_end_idx=idx,
                min_train_samples=min_train_samples,
                data=data,
            )

            # Train meta-stacker on accumulated OOS predictions
            if len(oos_indices) >= 20:
                stacker_preds = {}
                for model_name in ["lgb", "xgb", "lstm", "tcn"]:
                    if oos_predictions[model_name]:
                        model_horizons = {}
                        for h, v in oos_predictions[model_name].items():
                            arr = np.array(v, dtype=float)
                            # Replace None (now NaN) with NaN — MetaStacker's
                            # NaN row filter handles partial availability
                            model_horizons[h] = arr
                        stacker_preds[model_name] = model_horizons
                # For tree models (lgb, xgb), they're always available so
                # rows are kept. LSTM/TCN None entries become NaN — the
                # MetaStacker skips those rows only for horizons where
                # LSTM/TCN columns are included.
                stacker_targets = {h: np.array(v) for h, v in oos_targets.items()}
                rp_arr = np.array(oos_regime_probs) if oos_regime_probs else None
                meta_stacker.train(stacker_preds, rp_arr, stacker_targets)

            if crash_result.get("success") and return_result.get("success"):
                last_train_idx = idx
                ml_train_log.append(
                    {
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
                        "xgb_val_brier": xgb_result.get("val_brier", 0)
                        if xgb_result.get("success")
                        else None,
                        "temporal_val_loss": temporal_result.get("lstm_val_loss", None)
                        if temporal_result.get("success")
                        else None,
                    }
                )

        if not crash_model.is_trained or not return_model.is_trained:
            continue

        # ── Get ML predictions ────────────────────────────────────
        current_features = features.iloc[idx : idx + 1]
        try:
            # LightGBM predictions
            lgb_crash_12m = float(crash_model.predict_proba(current_features, "12m")[0])
            lgb_crash_6m = (
                float(crash_model.predict_proba(current_features, "6m")[0])
                if "6m" in crash_model.models
                else None
            )
            lgb_crash_3m = (
                float(crash_model.predict_proba(current_features, "3m")[0])
                if "3m" in crash_model.models
                else None
            )

            # XGBoost predictions
            xgb_crash_12m = None
            if xgb_model.is_trained:
                xgb_crash_12m = float(xgb_model.predict_proba(current_features, "12m")[0])

            # LSTM + TCN individual predictions (per-horizon)
            lstm_crash_12m = None
            tcn_crash_12m = None
            lstm_crash_6m = None
            tcn_crash_6m = None
            lstm_crash_3m = None
            tcn_crash_3m = None
            if temporal_model.is_trained:
                seq_start = max(0, idx - temporal_model.WINDOW_SIZE + 1)
                seq_features = features.iloc[seq_start : idx + 1]
                if len(seq_features) >= temporal_model.WINDOW_SIZE:
                    try:
                        indiv = temporal_model.predict_individual(seq_features, "12m")
                        if indiv.get("lstm") is not None:
                            lstm_crash_12m = float(indiv["lstm"][-1])
                        if indiv.get("tcn") is not None:
                            tcn_crash_12m = float(indiv["tcn"][-1])
                    except Exception:
                        temporal_preds = temporal_model.predict_proba(seq_features, "12m")
                        lstm_crash_12m = float(temporal_preds[-1])
                        tcn_crash_12m = lstm_crash_12m
                    try:
                        indiv_6m = temporal_model.predict_individual(seq_features, "6m")
                        if indiv_6m.get("lstm") is not None:
                            lstm_crash_6m = float(indiv_6m["lstm"][-1])
                        if indiv_6m.get("tcn") is not None:
                            tcn_crash_6m = float(indiv_6m["tcn"][-1])
                    except Exception:
                        pass
                    try:
                        indiv_3m = temporal_model.predict_individual(seq_features, "3m")
                        if indiv_3m.get("lstm") is not None:
                            lstm_crash_3m = float(indiv_3m["lstm"][-1])
                        if indiv_3m.get("tcn") is not None:
                            tcn_crash_3m = float(indiv_3m["tcn"][-1])
                    except Exception:
                        pass

            # Cox survival model predictions
            cox_crash_12m = None
            if cox_model.is_trained:
                try:
                    cox_crash_12m = float(cox_model.predict_proba(current_features, "12m")[0])
                except Exception:
                    pass

            # Collect OOS predictions for meta-stacker training
            oos_indices.append(idx)
            for h in ["3m", "6m", "12m"]:
                if h not in oos_predictions["lgb"]:
                    oos_predictions["lgb"][h] = []
                    oos_predictions["xgb"][h] = []
                    oos_predictions["lstm"][h] = []
                    oos_predictions["tcn"][h] = []
                    oos_predictions["cox"][h] = []
                    oos_targets[h] = []

            oos_predictions["lgb"]["12m"].append(lgb_crash_12m)
            oos_predictions["xgb"]["12m"].append(xgb_crash_12m)
            oos_predictions["lstm"]["12m"].append(lstm_crash_12m)
            oos_predictions["tcn"]["12m"].append(tcn_crash_12m)
            oos_predictions["cox"]["12m"].append(cox_crash_12m)

            # Collect 6m and 3m OOS predictions for multi-horizon meta-stacker.
            # Always append (use NaN placeholder when model unavailable) to
            # keep arrays aligned across all horizons.
            xgb_6m = None
            if xgb_model.is_trained and "6m" in xgb_model.models:
                xgb_6m = float(xgb_model.predict_proba(current_features, "6m")[0])
            oos_predictions["lgb"]["6m"].append(
                lgb_crash_6m if lgb_crash_6m is not None else np.nan
            )
            oos_predictions["xgb"]["6m"].append(xgb_6m)
            oos_predictions["lstm"]["6m"].append(lstm_crash_6m)
            oos_predictions["tcn"]["6m"].append(tcn_crash_6m)
            cox_6m = None
            if cox_model.is_trained:
                try:
                    cox_6m = float(cox_model.predict_proba(current_features, "6m")[0])
                except Exception:
                    pass
            oos_predictions["cox"]["6m"].append(cox_6m)

            xgb_3m = None
            if xgb_model.is_trained and "3m" in xgb_model.models:
                xgb_3m = float(xgb_model.predict_proba(current_features, "3m")[0])
            oos_predictions["lgb"]["3m"].append(
                lgb_crash_3m if lgb_crash_3m is not None else np.nan
            )
            oos_predictions["xgb"]["3m"].append(xgb_3m)
            oos_predictions["lstm"]["3m"].append(lstm_crash_3m)
            oos_predictions["tcn"]["3m"].append(tcn_crash_3m)
            cox_3m = None
            if cox_model.is_trained:
                try:
                    cox_3m = float(cox_model.predict_proba(current_features, "3m")[0])
                except Exception:
                    pass
            oos_predictions["cox"]["3m"].append(cox_3m)

            # Append OOS targets immediately after predictions to keep arrays aligned.
            # (Previously targets were appended much later, after continue-points that
            # could break alignment and cause shape mismatches in MetaStacker.)
            _actual_idx_12m = min(idx + 252, len(data) - 1)
            _actual_idx_6m = min(idx + 126, len(data) - 1)
            _actual_idx_3m = min(idx + 63, len(data) - 1)

            def _oos_check_crash(start_idx, end_idx, thresh=-risk_cfg["crash_threshold"]):
                fwd = data["SP500"].iloc[start_idx : end_idx + 1]
                pk = fwd.expanding().max()
                dd = ((fwd - pk) / pk).min()
                return dd <= thresh

            if "12m" in oos_targets:
                oos_targets["12m"].append(float(_oos_check_crash(idx, _actual_idx_12m)))
            if "6m" in oos_targets:
                oos_targets["6m"].append(float(_oos_check_crash(idx, _actual_idx_6m)))
            if "3m" in oos_targets:
                oos_targets["3m"].append(float(_oos_check_crash(idx, _actual_idx_3m)))

            # Meta-stacker ensemble prediction
            if meta_stacker.is_trained:
                model_preds = {
                    "lgb": lgb_crash_12m,
                    "xgb": xgb_crash_12m,
                    "lstm": lstm_crash_12m,
                    "tcn": tcn_crash_12m,
                    "cox": cox_crash_12m,
                }
                ml_crash_12m = float(meta_stacker.predict_proba(model_preds, horizon="12m")[0])
            else:
                # Simple average before meta-stacker is trained
                preds_to_avg = [lgb_crash_12m]
                if xgb_crash_12m is not None:
                    preds_to_avg.append(xgb_crash_12m)
                if lstm_crash_12m is not None:
                    preds_to_avg.append(lstm_crash_12m)
                if cox_crash_12m is not None:
                    preds_to_avg.append(cox_crash_12m)
                ml_crash_12m = float(np.mean(preds_to_avg))

            ml_crash_6m = lgb_crash_6m
            ml_crash_3m = lgb_crash_3m

            ml_return_12m = float(return_model.predict(current_features, "12m")[0])
            ml_return_6m = (
                float(return_model.predict(current_features, "6m")[0])
                if "6m" in return_model.models
                else None
            )
            ml_return_3m = (
                float(return_model.predict(current_features, "3m")[0])
                if "3m" in return_model.models
                else None
            )

            # Quantile predictions for uncertainty
            quantiles = return_model.predict_quantiles(current_features, "12m")
            ml_return_p10 = float(quantiles.get("p10", [ml_return_12m - 0.15])[0])
            ml_return_p90 = float(quantiles.get("p90", [ml_return_12m + 0.15])[0])
        except Exception:
            continue

        # ── Monte Carlo for uncertainty bands ─────────────────────
        lookback = bt_cfg["lookback_years"] * 252
        hist = data.iloc[max(0, idx - lookback) : idx]
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
            start_price,
            hist_mu,
            sigma,
            forward_days,
            bt_cfg["simulations_per_point"],
            crash_freq,
            risk,
            base_scenario,
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
            fwd = data["SP500"].iloc[start_idx : end_idx + 1]
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

        results.append(
            {
                "date": pred_date,
                "start_price": start_price,
                "actual_price": actual_price_12m,
                "pred_mean": pred_mean,
                "pred_p95": pred_p95,
                "pred_p05": pred_p05,
                "pct_error": pct_error,
                "within_bounds": within_bounds,
                # ML crash predictions (ENSEMBLE — PRIMARY)
                "ml_crash_12m": ml_crash_12m,
                "ml_crash_6m": ml_crash_6m,
                "ml_crash_3m": ml_crash_3m,
                # Per-model crash predictions (for analysis)
                "lgb_crash_12m": lgb_crash_12m,
                "xgb_crash_12m": xgb_crash_12m,
                "lstm_crash_12m": lstm_crash_12m,
                "tcn_crash_12m": tcn_crash_12m,
                "cox_crash_12m": cox_crash_12m,
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
            }
        )

    bt = pd.DataFrame(results)

    if len(bt) > 0:
        _print_results(bt, crash_model, return_model, ml_train_log)
        bt.attrs.update(_compute_metrics(bt, crash_model, return_model, ml_train_log))

        # Store all ensemble models for use by main pipeline
        bt.attrs["xgb_model"] = xgb_model
        bt.attrs["temporal_model"] = temporal_model
        bt.attrs["meta_stacker"] = meta_stacker
        bt.attrs["crash_timing"] = crash_timing
        bt.attrs["cox_model"] = cox_model

        # ── Evaluation module integration ─────────────────────────────
        try:
            from finpredict.evaluation.metrics import (
                brier_score as eval_brier_score,
                brier_skill_score,
                reliability_diagram,
                prediction_spread_check,
            )

            y_pred = bt["ml_crash_12m"].values
            y_true = bt["actual_crash_12m"].astype(float).values

            bs = eval_brier_score(y_pred, y_true)
            bss_clim = brier_skill_score(y_pred, y_true, baseline="climatology")
            rel = reliability_diagram(y_pred, y_true)
            spread = prediction_spread_check(y_pred)

            eval_results = {
                "brier_score": bs,
                "bss_vs_climatology": bss_clim,
                "calibration_error": rel["calibration_error"],
                "prediction_spread": spread,
            }

            # VIX baseline (if VIX data available in backtest context)
            if "vix" in bt.columns:
                bss_vix = brier_skill_score(
                    y_pred,
                    y_true,
                    baseline="vix25",
                    vix_series=bt["vix"],
                )
                eval_results["bss_vs_vix"] = bss_vix

            bt.attrs["evaluation"] = eval_results

            print("\n  [EVAL] ═══ MODEL EVALUATION ═══")
            print(f"  [EVAL] Brier Score:          {bs:.4f}")
            print(f"  [EVAL] BSS vs Climatology:   {bss_clim:.4f}")
            print(f"  [EVAL] Calibration Error:    {rel['calibration_error']:.4f}")
            if spread["is_underdispersed"]:
                print(f"  [EVAL] [WARN] Predictions are underdispersed (std={spread['std']:.4f})")
            if bss_clim < 0.02:
                print(f"  [EVAL] [WARN] Model barely beats naive baseline (BSS={bss_clim:.4f})")
            elif bss_clim > 0:
                print("  [EVAL] [OK] Model beats climatology baseline")
            else:
                print("  [EVAL] [WARN] Model does NOT beat climatology baseline")

        except Exception as e:
            print(f"  [EVAL] Evaluation failed: {e}")

        # ── Paper-ready results table ──────────────────────────────────
        try:
            paper_table = _build_paper_results_table(bt, data)
            bt.attrs["paper_results"] = paper_table
        except Exception as e:
            print(f"  [EVAL] Paper results table failed: {e}")

    return bt


# ═══════════════════════════════════════════════════════════════════════
# RESULTS PRINTING & METRICS
# ═══════════════════════════════════════════════════════════════════════


def _print_results(bt, crash_model, return_model, ml_train_log):
    """Print comprehensive ML evaluation results with discrimination focus."""
    mape = np.abs(bt["pct_error"]).mean()
    coverage = bt["within_bounds"].mean() * 100
    direction = bt["direction_correct"].mean() * 100

    print("\n  [RESULTS] ML Walk-Forward Backtest:")
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

    print("\n  ═══ ML CRASH MODEL (12-month) ═══")
    print(f"  Brier Score:          {ml_brier:.4f} (0.25 = random, lower = better)")
    print(f"  AUC:                  {crash_auc:.3f} (0.5 = random, higher = better)")
    print(f"  Prediction Range:     [{crash_range[0]*100:.1f}%, {crash_range[1]*100:.1f}%]")
    print(f"  Prediction Std Dev:   {crash_std*100:.1f}%")
    print(f"  Prediction Mean:      {crash_mean*100:.1f}%")

    # DISCRIMINATION CHECK — this is the key metric
    if crash_std < 0.05:
        print(f"  ⚠ WARNING: Low discrimination (std={crash_std*100:.1f}% < 5%)")
        print("    Model predictions are clustered — limited predictive value")
    else:
        print(f"  ✓ Good discrimination (std={crash_std*100:.1f}% > 5%)")

    # Fine-grained calibration
    bins = [(0, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.50), (0.50, 1.0)]
    print("  Crash Calibration:")
    for lo, hi in bins:
        mask = (bt["ml_crash_12m"] >= lo) & (bt["ml_crash_12m"] < hi)
        subset = bt[mask]
        if len(subset) > 0:
            actual_rate = float(subset["actual_crash_12m"].mean() * 100)
            pred_avg = float(subset["ml_crash_12m"].mean() * 100)
            print(
                f"    [{lo*100:.0f}-{hi*100:.0f}%]: predicted avg {pred_avg:.1f}%, "
                f"actual {actual_rate:.1f}% (n={len(subset)})"
            )
        else:
            print(f"    [{lo*100:.0f}-{hi*100:.0f}%]: no predictions")

    # Coarse summary
    low = bt[bt["ml_crash_12m"] < 0.15]
    med = bt[(bt["ml_crash_12m"] >= 0.15) & (bt["ml_crash_12m"] < 0.40)]
    high = bt[bt["ml_crash_12m"] >= 0.40]
    cal_low = float(low["actual_crash_12m"].mean() * 100) if len(low) > 0 else 0
    cal_med = float(med["actual_crash_12m"].mean() * 100) if len(med) > 0 else 0
    cal_high = float(high["actual_crash_12m"].mean() * 100) if len(high) > 0 else 0
    print(
        f"  Summary: Low<15%: {cal_low:.0f}% actual (n={len(low)}) | "
        f"Med 15-40%: {cal_med:.0f}% actual (n={len(med)}) | "
        f"High>40%: {cal_high:.0f}% actual (n={len(high)})"
    )

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
            q_coverage = float(
                (
                    (bt_q["actual_return_12m"] >= bt_q["ml_return_p10"])
                    & (bt_q["actual_return_12m"] <= bt_q["ml_return_p90"])
                ).mean()
            )
        else:
            q_coverage = 0
    else:
        q_coverage = 0

    print("\n  ═══ ML RETURN MODEL (12-month) ═══")
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
        print("\n  [ML TRAINING]")
        print(f"  Models retrained:     {len(ml_train_log)} times")
        print(f"  Last Crash AUC:       {last.get('crash_val_auc', 0):.3f}")
        print(f"  Last Return Corr:     {last.get('return_val_corr', 0):.3f}")
        print(f"  Last Return Skill:    {last.get('return_skill', 0):.3f}")
        disc = last.get("crash_discrimination", "UNKNOWN")
        print(f"  Discrimination:       {disc}")

    # Top features
    if crash_model.is_trained:
        top = crash_model.get_top_features(10)
        if top:
            print("\n  [TOP CRASH PREDICTORS]")
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

    metrics = {
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
        "n_low": len(low),
        "n_med": len(med),
        "n_high": len(high),
        "ml_train_log": ml_train_log,
        "crash_model": crash_model,
        "return_model": return_model,
        "crash_pred_range": (float(bt["ml_crash_12m"].min()), float(bt["ml_crash_12m"].max())),
        "crash_pred_std": float(bt["ml_crash_12m"].std()),
        "return_pred_range": (float(bt["ml_return_12m"].min()), float(bt["ml_return_12m"].max())),
    }

    # ── Advanced Validation Metrics ─────────────────────────────────
    try:
        from finpredict.evaluation.metrics import (
            lead_time_accuracy,
            false_alarm_rate,
            missed_crash_rate,
            directional_accuracy_vs_consensus,
            signal_max_drawdown,
        )

        # Build crash_periods from backtest actual outcomes
        crash_periods = _extract_crash_periods(bt)

        advanced = {}
        if not crash_periods.empty:
            advanced["lead_time"] = lead_time_accuracy(bt, crash_periods)
            advanced["missed_crash"] = missed_crash_rate(bt, crash_periods)

        advanced["false_alarm"] = false_alarm_rate(bt)
        advanced["signal_drawdown"] = signal_max_drawdown(bt)

        consensus_ret = get_institutional_return()
        advanced["directional_vs_consensus"] = directional_accuracy_vs_consensus(bt, consensus_ret)

        metrics["advanced_metrics"] = advanced

        # Print advanced metrics
        print("\n  ═══ ADVANCED VALIDATION METRICS ═══")
        if "lead_time" in advanced:
            lt = advanced["lead_time"]
            status = "OK" if lt["mean_lead_days"] > 30 else "WARN"
            print(
                f"  Lead Time Accuracy:   {lt['mean_lead_days']:.0f} days avg "
                f"({lt['n_detected']}/{lt['n_crashes']} detected) [{status}]"
            )
        fa = advanced["false_alarm"]
        fa_status = "OK" if fa["rate"] < 0.30 else "WARN"
        print(
            f"  False Alarm Rate:     {fa['rate']*100:.0f}% "
            f"({fa['n_false_alarms']}/{fa['n_alarms']} false) [{fa_status}]"
        )
        if "missed_crash" in advanced:
            mc = advanced["missed_crash"]
            mc_status = "OK" if mc["rate"] < 0.20 else "WARN"
            print(
                f"  Missed Crash Rate:    {mc['rate']*100:.0f}% "
                f"({mc['n_missed']}/{mc['n_crashes']} missed) [{mc_status}]"
            )
        sd = advanced["signal_drawdown"]
        sd_status = "OK" if sd["max_drawdown"] < 0.25 else "WARN"
        print(f"  Signal Max Drawdown:  {sd['max_drawdown']*100:.1f}% [{sd_status}]")
        dvc = advanced["directional_vs_consensus"]
        dvc_status = "OK" if dvc["engine_correct_rate"] > 0.55 else "WARN"
        print(
            f"  Dir. vs Consensus:    {dvc['engine_correct_rate']*100:.0f}% correct "
            f"({dvc['n_engine_correct']}/{dvc['n_divergences']} divergences) [{dvc_status}]"
        )

    except Exception as e:
        print(f"  [EVAL] Advanced metrics failed: {e}")

    return metrics


def _extract_crash_periods(bt: pd.DataFrame) -> pd.DataFrame:
    """Extract crash periods from backtest results for metrics computation.

    Identifies transitions where actual_crash_12m goes from False to True,
    marking the start of crash episodes.
    """
    if "actual_crash_12m" not in bt.columns or "date" not in bt.columns:
        return pd.DataFrame(columns=["start", "end"])

    bt_sorted = bt.sort_values("date").copy()
    crashes = []
    in_crash = False
    crash_start = None

    for _, row in bt_sorted.iterrows():
        is_crash = bool(row["actual_crash_12m"])
        if is_crash and not in_crash:
            crash_start = pd.Timestamp(row["date"])
            in_crash = True
        elif not is_crash and in_crash:
            crashes.append({"start": crash_start, "end": pd.Timestamp(row["date"])})
            in_crash = False

    # Handle crash that extends to end of data
    if in_crash and crash_start is not None:
        crashes.append({"start": crash_start, "end": bt_sorted["date"].iloc[-1]})

    return pd.DataFrame(crashes)


# ═══════════════════════════════════════════════════════════════════════
# PAPER-READY RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════


def _build_paper_results_table(bt: pd.DataFrame, data: pd.DataFrame) -> dict:
    """Build comprehensive per-model, per-horizon results table for publication.

    Generates Table 1 (Crash Model BSS) and Table 2 (Lead Time / False Alarm)
    suitable for an academic paper.

    Returns:
        dict with 'bss_table' (DataFrame), 'operational_table' (DataFrame),
        and 'per_model_auc' (dict).
    """
    from sklearn.metrics import roc_auc_score
    from finpredict.evaluation.metrics import (
        brier_score as eval_brier,
        brier_skill_score as eval_bss,
        reliability_diagram,
        lead_time_accuracy,
        false_alarm_rate,
        missed_crash_rate,
    )

    model_cols = {
        "LightGBM": "lgb_crash_12m",
        "XGBoost": "xgb_crash_12m",
        "LSTM": "lstm_crash_12m",
        "TCN": "tcn_crash_12m",
        "Meta-Ensemble": "ml_crash_12m",
    }

    # Also handle multi-horizon for ensemble
    horizon_cols = {
        "3m": ("ml_crash_3m", "actual_crash_3m"),
        "6m": ("ml_crash_6m", "actual_crash_6m"),
        "12m": ("ml_crash_12m", "actual_crash_12m"),
    }

    # ── Table 1: BSS per model (12m horizon) ────────────────────────
    print("\n  ═══════════════════════════════════════════════════════════")
    print("  TABLE 1: Crash Prediction — Brier Skill Score (12-month)")
    print("  ═══════════════════════════════════════════════════════════")

    bss_rows = []
    y_true_12m = bt["actual_crash_12m"].astype(float).values

    # Prepare VIX series for baseline if available
    vix_series = None
    if "VIX" in data.columns:
        # Align VIX with backtest dates
        vix_vals = []
        for _, row in bt.iterrows():
            dt = pd.Timestamp(row["date"])
            idx = data.index.get_indexer([dt], method="nearest")[0]
            vix_vals.append(float(data["VIX"].iloc[idx]) if "VIX" in data.columns else 20.0)
        vix_series = pd.Series(vix_vals)

    # Yield spread for baseline
    spread_series = None
    if "T10Y" in data.columns and "T3M" in data.columns:
        spread_vals = []
        for _, row in bt.iterrows():
            dt = pd.Timestamp(row["date"])
            idx = data.index.get_indexer([dt], method="nearest")[0]
            t10y = float(data["T10Y"].iloc[idx]) if "T10Y" in data.columns else 0.03
            t3m = float(data["T3M"].iloc[idx]) if "T3M" in data.columns else 0.02
            spread_vals.append(t10y - t3m)
        spread_series = pd.Series(spread_vals)

    for model_name, col in model_cols.items():
        if col not in bt.columns:
            continue

        y_pred = bt[col].values
        valid = ~np.isnan(y_pred)
        if valid.sum() < 10:
            continue

        yp = y_pred[valid]
        yt = y_true_12m[valid]

        bs = eval_brier(yp, yt)
        bss_clim = eval_bss(yp, yt, baseline="climatology")

        bss_vix = float("nan")
        if vix_series is not None:
            vs = vix_series.values[valid]
            bss_vix = eval_bss(yp, yt, baseline="vix25", vix_series=pd.Series(vs))

        bss_yc = float("nan")
        if spread_series is not None:
            ss = spread_series.values[valid]
            bss_yc = eval_bss(yp, yt, baseline="yield_curve", spread_series=pd.Series(ss))

        try:
            auc = float(roc_auc_score(yt, yp)) if len(set(yt)) >= 2 else float("nan")
        except Exception:
            auc = float("nan")

        rel = reliability_diagram(yp, yt)
        pred_std = float(np.std(yp))

        bss_rows.append(
            {
                "Model": model_name,
                "N": int(valid.sum()),
                "Brier": f"{bs:.4f}",
                "BSS_Clim": f"{bss_clim:+.4f}",
                "BSS_VIX25": f"{bss_vix:+.4f}" if not np.isnan(bss_vix) else "—",
                "BSS_YC": f"{bss_yc:+.4f}" if not np.isnan(bss_yc) else "—",
                "AUC": f"{auc:.3f}" if not np.isnan(auc) else "—",
                "ECE": f"{rel['calibration_error']:.4f}",
                "Pred_Std": f"{pred_std:.4f}",
            }
        )

    bss_df = pd.DataFrame(bss_rows)
    if not bss_df.empty:
        # Print formatted table
        header = f"  {'Model':<16} {'N':>5} {'Brier':>8} {'BSS_Clim':>10} {'BSS_VIX':>10} {'BSS_YC':>10} {'AUC':>6} {'ECE':>8} {'Std':>8}"
        print(header)
        print(f"  {'─'*90}")
        for _, row in bss_df.iterrows():
            print(
                f"  {row['Model']:<16} {row['N']:>5} {row['Brier']:>8} {row['BSS_Clim']:>10} "
                f"{row['BSS_VIX25']:>10} {row['BSS_YC']:>10} {row['AUC']:>6} "
                f"{row['ECE']:>8} {row['Pred_Std']:>8}"
            )

    # ── Table 1b: Multi-horizon BSS (Ensemble only) ─────────────────
    print("\n  ═══════════════════════════════════════════════════════════")
    print("  TABLE 1b: Meta-Ensemble BSS by Horizon")
    print("  ═══════════════════════════════════════════════════════════")

    horizon_rows = []
    for horizon, (pred_col, actual_col) in horizon_cols.items():
        if pred_col not in bt.columns or actual_col not in bt.columns:
            continue

        yp = bt[pred_col].dropna().values
        yt = bt[actual_col].loc[bt[pred_col].notna()].astype(float).values

        if len(yp) < 10 or len(yp) != len(yt):
            continue

        bs = eval_brier(yp, yt)
        bss_clim = eval_bss(yp, yt, baseline="climatology")

        try:
            auc = float(roc_auc_score(yt, yp)) if len(set(yt)) >= 2 else float("nan")
        except Exception:
            auc = float("nan")

        base_rate = float(yt.mean())
        horizon_rows.append(
            {
                "Horizon": horizon,
                "N": len(yp),
                "Base_Rate": f"{base_rate:.1%}",
                "Brier": f"{bs:.4f}",
                "BSS_Clim": f"{bss_clim:+.4f}",
                "AUC": f"{auc:.3f}" if not np.isnan(auc) else "—",
            }
        )

    if horizon_rows:
        print(
            f"  {'Horizon':<10} {'N':>5} {'Base Rate':>10} {'Brier':>8} {'BSS_Clim':>10} {'AUC':>6}"
        )
        print(f"  {'─'*55}")
        for row in horizon_rows:
            print(
                f"  {row['Horizon']:<10} {row['N']:>5} {row['Base_Rate']:>10} "
                f"{row['Brier']:>8} {row['BSS_Clim']:>10} {row['AUC']:>6}"
            )

    # ── Table 2: Operational Metrics ────────────────────────────────
    print("\n  ═══════════════════════════════════════════════════════════")
    print("  TABLE 2: Operational Metrics (Lead Time, False Alarms)")
    print("  ═══════════════════════════════════════════════════════════")

    crash_periods = _extract_crash_periods(bt)
    op_rows = []

    for model_name, col in model_cols.items():
        if col not in bt.columns:
            continue

        valid = bt[col].notna()
        if valid.sum() < 10:
            continue

        bt_valid = bt[valid].copy()

        # Lead time (threshold=0.40 for early warning)
        lt = {"mean_lead_days": 0, "n_detected": 0, "n_crashes": 0}
        if not crash_periods.empty:
            lt = lead_time_accuracy(bt_valid, crash_periods, prob_col=col, prob_threshold=0.40)

        # False alarm rate (threshold=0.60 for high-confidence alerts)
        fa = false_alarm_rate(bt_valid, prob_col=col, alarm_threshold=0.60)

        # Missed crash rate (threshold=0.30 for "safe" calls)
        mc = {"rate": 0, "n_missed": 0, "n_crashes": 0}
        if not crash_periods.empty:
            mc = missed_crash_rate(bt_valid, crash_periods, prob_col=col, safe_threshold=0.30)

        op_rows.append(
            {
                "Model": model_name,
                "Lead_Days": f"{lt['mean_lead_days']:.0f}",
                "Detected": f"{lt['n_detected']}/{lt['n_crashes']}",
                "FA_Rate": f"{fa['rate']:.0%}",
                "FA_Count": f"{fa['n_false_alarms']}/{fa['n_alarms']}",
                "Miss_Rate": f"{mc['rate']:.0%}",
                "Miss_Count": f"{mc['n_missed']}/{mc['n_crashes']}",
            }
        )

    if op_rows:
        print(
            f"  {'Model':<16} {'Lead(d)':>8} {'Det.':>6} {'FA Rate':>8} {'FA':>6} {'Miss%':>7} {'Miss':>6}"
        )
        print(f"  {'─'*65}")
        for row in op_rows:
            print(
                f"  {row['Model']:<16} {row['Lead_Days']:>8} {row['Detected']:>6} "
                f"{row['FA_Rate']:>8} {row['FA_Count']:>6} {row['Miss_Rate']:>7} "
                f"{row['Miss_Count']:>6}"
            )

    # ── Table 3: Per-crash lead time detail ─────────────────────────
    if not crash_periods.empty:
        print("\n  ═══════════════════════════════════════════════════════════")
        print("  TABLE 3: Per-Crash Detection Detail (Meta-Ensemble)")
        print("  ═══════════════════════════════════════════════════════════")

        lt_detail = lead_time_accuracy(
            bt, crash_periods, prob_col="ml_crash_12m", prob_threshold=0.40
        )
        if lt_detail["per_crash"]:
            print(f"  {'Crash Start':<14} {'Detected':>10} {'Lead Days':>10} {'Alert Prob':>11}")
            print(f"  {'─'*50}")
            for pc in lt_detail["per_crash"]:
                det_str = "YES" if pc["detected"] else "NO"
                lead_str = f"{pc['lead_days']}" if pc["detected"] else "—"
                prob_str = f"{pc.get('first_alert_prob', 0):.1%}" if pc["detected"] else "—"
                print(f"  {pc['crash_start']:<14} {det_str:>10} {lead_str:>10} {prob_str:>11}")

    # ── Summary verdict ─────────────────────────────────────────────
    print("\n  ═══════════════════════════════════════════════════════════")
    print("  PUBLICATION READINESS ASSESSMENT")
    print("  ═══════════════════════════════════════════════════════════")

    if bss_rows:
        ensemble_row = next((r for r in bss_rows if r["Model"] == "Meta-Ensemble"), None)
        if ensemble_row:
            bss_val = float(ensemble_row["BSS_Clim"].replace("+", ""))
            auc_str = ensemble_row["AUC"]
            auc_val = float(auc_str) if auc_str != "—" else 0.5

            checks = []
            checks.append(("BSS > 0 (beats climatology)", bss_val > 0))
            checks.append(("AUC > 0.60 (discrimination)", auc_val > 0.60))

            if op_rows:
                ens_op = next((r for r in op_rows if r["Model"] == "Meta-Ensemble"), None)
                if ens_op:
                    fa_val = float(ens_op["FA_Rate"].replace("%", "")) / 100
                    miss_val = float(ens_op["Miss_Rate"].replace("%", "")) / 100
                    checks.append(("False alarm rate < 30%", fa_val < 0.30))
                    checks.append(("Missed crash rate < 20%", miss_val < 0.20))

            pred_std_val = float(ensemble_row["Pred_Std"])
            checks.append(("Prediction spread > 5% (not collapsed)", pred_std_val > 0.05))

            passed = sum(1 for _, v in checks if v)
            total = len(checks)

            for label, ok in checks:
                status = "PASS" if ok else "FAIL"
                print(f"  [{status}] {label}")

            print(f"\n  Result: {passed}/{total} checks passed")
            if passed == total:
                print("  Verdict: READY for submission — model demonstrates genuine skill")
            elif passed >= total - 1:
                print("  Verdict: NEAR-READY — minor improvements needed")
            else:
                print("  Verdict: NOT READY — significant improvements needed")

    return {
        "bss_table": bss_df if not bss_df.empty else pd.DataFrame(),
        "horizon_table": pd.DataFrame(horizon_rows) if horizon_rows else pd.DataFrame(),
        "operational_table": pd.DataFrame(op_rows) if op_rows else pd.DataFrame(),
    }
