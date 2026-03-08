"""
Market Prediction Engine v7 — ML-First Pipeline
===================================================

ARCHITECTURE:
    1. Fetch data (Yahoo Finance + FRED historical time series)
    2. Fit statistical models (GARCH volatility, HMM regimes)
    3. Build 80+ ML features from market + macro data
    4. Walk-forward ML backtest (expanding window, zero data leakage)
    5. Generate current ML predictions (crash probability + expected returns)
    6. ML-conditioned Monte Carlo (uncertainty quantification)
    7. Factor-based sector analysis (differentiated by beta, momentum, etc.)
    8. Individual stock analysis
    9. Generate PDF report

KEY PRINCIPLE: ML predictions are PRIMARY. Monte Carlo is SECONDARY.
    The ML models learn from 35 years of data to read the current
    market state. Monte Carlo just quantifies uncertainty around those
    learned predictions. No hardcoded parameters anywhere.

DEPENDENCIES:
    pip install lightgbm scikit-learn numpy pandas scipy yfinance
    pip install fredapi pyyaml python-dotenv tqdm arch hmmlearn
    pip install reportlab pyarrow matplotlib
"""

import sys
import traceback

import numpy as np
import pandas as pd

from finpredict.config import config, PROJECT_ROOT, get_forecast_days
from finpredict.data import cached_fetch_all_data
from finpredict.data.fred_fetcher import (
    fetch_fred_data,
    get_recession_probability,
    get_macro_features,
)
from finpredict.risk import build_risk_score, detect_regimes, identify_crashes
from finpredict.models.garch import fit_garch
from finpredict.models.hmm_regimes import fit_hmm_regimes, get_regime_probs
from finpredict.simulation import run_monte_carlo, compute_valuation_penalty, run_stress_tests
from finpredict.simulation.backtest import run_backtest
from finpredict.models.sectors import analyze_sectors
from finpredict.models.stocks import select_stocks_from_sectors, analyze_stocks
from finpredict.reporting.pdf_report import generate_report
from finpredict.ml.features import (
    build_feature_matrix,
)
from finpredict.ml.anomaly_detector import AnomalyDetector, BayesianChangepoint
from finpredict.ml.prediction_logger import log_prediction
from finpredict.intelligence import fetch_gdelt_data, compute_event_score
from finpredict.intelligence.event_scorer import adjust_crash_probability

from datetime import datetime


def main():
    """Complete v7 ML-First engine pipeline."""
    try:
        # Ensure UTF-8 output encoding without replacing the stdout object.
        # The TextIOWrapper replacement broke output capture in IDEs (VS, VS Code).
        # reconfigure() is the safe alternative; ignore errors in environments
        # that don't support it (e.g. some CI runners).
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

        np.random.seed(42)
        sim_cfg = config["simulation"]

        print("=" * 90)
        print("  MARKET PREDICTION ENGINE v7.0 — ML-FIRST")
        print("  LightGBM | GARCH Volatility | HMM Regimes | FRED Macro (Time-Varying)")
        print("  Isotonic Calibration | Quantile Regression | Factor-Based Sectors")
        print("=" * 90)
        print(f"  Date:        {datetime.now().strftime('%B %d, %Y %I:%M %p')}")
        print(f"  Horizon:     {sim_cfg['forecast_years']} years")
        print(f"  Simulations: {sim_cfg['num_simulations']:,}")
        print(f"  Backtest:    {config['data']['backtest_start']} → present")
        print("=" * 90 + "\n")

        # ══════════════════════════════════════════════════════════
        # 1. FETCH MARKET DATA
        # ══════════════════════════════════════════════════════════
        print("[MODULE 1] Fetching market data from Yahoo Finance...")
        data, sector_data = cached_fetch_all_data()

        # ── Data quality validation ───────────────────────────────
        try:
            from finpredict.data.quality import DataQualityChecker

            dq_warnings = DataQualityChecker().validate(data)
            if dq_warnings:
                print(f"  [DQ] {len(dq_warnings)} data quality warning(s):")
                for w in dq_warnings:
                    print(f"    [{w['severity'].upper()}] {w['message']}")
            else:
                print("  [DQ] All data quality checks passed")
            print()
        except Exception as e:
            print(f"  [DQ] Data quality check failed: {e} — continuing\n")

        # ══════════════════════════════════════════════════════════
        # 2. FETCH FRED TIME SERIES (full history for ML training)
        # ══════════════════════════════════════════════════════════
        fred_data = fetch_fred_data()
        # Fetch global index data for contagion features
        global_data = {}
        try:
            from finpredict.data.global_crashes import cached_fetch_global_indices

            global_data = cached_fetch_global_indices()
        except Exception as e:
            print(f"  [GLOBAL] Global indices unavailable: {e}")

        recession_prob = None
        fred_features = {}
        if fred_data:
            recession_prob = get_recession_probability(fred_data)
            fred_features = get_macro_features(fred_data)
            n_fred = sum(1 for v in fred_data.values() if len(v) > 0)
            print(f"  [FRED] {n_fred} time series loaded for ML training")
            print(f"  [FRED] Recession probability: {recession_prob*100:.1f}%")
            for k, v in fred_features.items():
                print(f"  [FRED] {k}: {v:.2f}")
            print()

        # ══════════════════════════════════════════════════════════
        # 3. GJR-GARCH VOLATILITY MODEL
        # ══════════════════════════════════════════════════════════
        print("[MODULE 2] Fitting GARCH volatility model...")
        data["Daily_Returns"] = data["SP500"].pct_change()
        returns = data["Daily_Returns"].dropna()
        garch_result = fit_garch(returns)
        garch_vol = garch_result.current_vol if garch_result.success else None

        # Extract GARCH persistence for MC volatility dynamics
        garch_persistence = None
        garch_xi = None
        garch_rho_leverage = None
        if garch_result.success:
            # Persistence = alpha + gamma/2 + beta (for GJR-GARCH)
            garch_persistence = garch_result.alpha + garch_result.gamma / 2 + garch_result.beta
            print(f"  [GARCH] Conditional vol: {garch_vol*100:.1f}%")
            print(f"  [GARCH] Persistence: {garch_persistence:.4f}")

            # Derive vol-of-vol (xi) and leverage correlation (rho) from GARCH
            garch_cfg = config.get("simulation", {}).get("garch_derived_params", {})
            alpha = garch_result.alpha
            gamma = garch_result.gamma

            # rho_leverage: derived from GJR-GARCH asymmetry parameter
            rho_raw = -np.sqrt(gamma / (2 * alpha + gamma + 1e-8))
            garch_rho_leverage = float(
                np.clip(
                    rho_raw,
                    garch_cfg.get("rho_leverage_min", -0.95),
                    garch_cfg.get("rho_leverage_max", -0.30),
                )
            )

            # xi: coefficient of variation of GARCH conditional volatility
            # conditional_volatility from arch is in % units (returns scaled ×100)
            if garch_result.model_fit is not None:
                cond_vol = garch_result.model_fit.conditional_volatility / 100
                xi_raw = float(cond_vol.std() / cond_vol.mean()) if cond_vol.mean() > 0 else 0.06
                garch_xi = float(
                    np.clip(
                        xi_raw,
                        garch_cfg.get("xi_min", 0.02),
                        garch_cfg.get("xi_max", 0.15),
                    )
                )

            print(f"  [GARCH] Derived rho_leverage: {garch_rho_leverage:.3f}")
            if garch_xi is not None:
                print(f"  [GARCH] Derived xi (vol-of-vol): {garch_xi:.4f}")

        # ══════════════════════════════════════════════════════════
        # 4. HMM REGIME DETECTION
        # ══════════════════════════════════════════════════════════
        print("[MODULE 3] Detecting market regimes (HMM)...")
        hmm_result = fit_hmm_regimes(data)
        get_regime_probs(hmm_result)

        # ══════════════════════════════════════════════════════════
        # 5. RISK SCORE + REGIME LABELS
        # ══════════════════════════════════════════════════════════
        print("[MODULE 4] Computing risk score...")
        data["Risk_Score"] = build_risk_score(data)
        data["Regime"], rule_regime = detect_regimes(data)

        if hmm_result.success:
            current_regime = hmm_result.current_regime
            data["Regime"] = hmm_result.regime_labels
            print(f"  [REGIME] Using HMM: {current_regime} " f"(rule-based: {rule_regime})")
        else:
            current_regime = rule_regime
            print(f"  [REGIME] Using rule-based: {current_regime}")

        current_price = float(data["SP500"].iloc[-1])
        current_risk = float(data["Risk_Score"].iloc[-1])

        # Consistency check: if HMM says Bear/Crisis but risk score is low,
        # the HMM state labels may have swapped due to random seed. Downgrade.
        if pd.notna(current_risk) and current_regime in ("Bear", "Crisis") and current_risk < 0.5:
            print(
                f"  [WARN] HMM regime '{current_regime}' contradicts "
                f"low risk score ({current_risk:.2f}σ), downgrading to Neutral"
            )
            current_regime = "Neutral"
        current_vix = float(data["VIX"].iloc[-1]) if "VIX" in data.columns else 20.0
        yield_curve = (
            float(data["T10Y"].iloc[-1] - data["T3M"].iloc[-1])
            if "T10Y" in data.columns and "T3M" in data.columns
            else 0.5
        )

        # ══════════════════════════════════════════════════════════
        # 6. HISTORICAL CRASH ANALYSIS
        # ══════════════════════════════════════════════════════════
        print("[MODULE 5] Analyzing crash history...")
        crash_df, crash_freq = identify_crashes(data)

        # ══════════════════════════════════════════════════════════
        # 7. VALUATION CONSTRAINT
        # ══════════════════════════════════════════════════════════
        val_penalty, cape_value = compute_valuation_penalty(data)
        cape_str = f" (CAPE: {cape_value:.1f})" if cape_value else " (trend proxy)"
        print(f"  [VALUATION] Penalty: {val_penalty*100:+.2f}%/yr{cape_str}\n")

        # ══════════════════════════════════════════════════════════
        # 8. ML WALK-FORWARD BACKTEST (the core)
        # ══════════════════════════════════════════════════════════
        # This trains crash + return models on expanding windows
        # and evaluates their prediction quality over 25 years
        bt_results = run_backtest(data, crash_freq, fred_data=fred_data, global_data=global_data)

        # ══════════════════════════════════════════════════════════
        # 8b. ROLLING BRIER MONITOR (degradation detection)
        # ══════════════════════════════════════════════════════════
        try:
            from finpredict.evaluation.brier_monitor import RollingBrierMonitor

            monitor = RollingBrierMonitor()
            degradation = monitor.evaluate_and_check(data)
            if degradation["evaluated"] > 0:
                print(f"  [MONITOR] Evaluated {degradation['evaluated']} past predictions")
                if degradation["rolling_brier"] is not None:
                    print(f"  [MONITOR] Rolling Brier: {degradation['rolling_brier']:.4f}")
                if degradation["degraded"]:
                    print("  [MONITOR] WARNING: Model degradation detected!")
            else:
                print("  [MONITOR] No past predictions to evaluate yet")
        except Exception as e:
            print(f"  [MONITOR] Brier monitor unavailable: {e}")

        # ══════════════════════════════════════════════════════════
        # 9. CURRENT ML PREDICTIONS (PRIMARY OUTPUT)
        # ══════════════════════════════════════════════════════════
        ml_crash_prob = None
        ml_predicted_return = None
        ml_crash_3m = None
        ml_crash_6m = None
        ml_return_3m = None
        ml_return_6m = None
        ml_return_p10 = None
        ml_return_p90 = None
        shap_contributions = None
        counterfactual_results = None
        crash_timing_results = None

        crash_model = bt_results.attrs.get("crash_model")
        return_model = bt_results.attrs.get("return_model")
        xgb_model = bt_results.attrs.get("xgb_model")
        temporal_model = bt_results.attrs.get("temporal_model")
        meta_stacker = bt_results.attrs.get("meta_stacker")
        cox_model = bt_results.attrs.get("cox_model")
        crash_timing_model = bt_results.attrs.get("crash_timing")

        if crash_model and crash_model.is_trained and return_model and return_model.is_trained:
            current_features = build_feature_matrix(
                data, fred_data=fred_data, global_data=global_data
            )
            current_row = current_features.iloc[-1:]

            # Multi-horizon crash predictions (with SHAP for 12m)
            shap_result = crash_model.predict_with_shap(
                current_row,
                "12m",
                compute_shap=True,
            )
            lgb_crash_12m = float(shap_result["crash_prob"][0])
            shap_values = shap_result.get("shap_values")
            if "6m" in crash_model.models:
                ml_crash_6m = float(crash_model.predict_proba(current_row, "6m")[0])
            if "3m" in crash_model.models:
                ml_crash_3m = float(crash_model.predict_proba(current_row, "3m")[0])

            # XGBoost prediction
            xgb_crash_12m = None
            if xgb_model and xgb_model.is_trained:
                xgb_crash_12m = float(xgb_model.predict_proba(current_row, "12m")[0])

            # LSTM + TCN temporal predictions (individual)
            lstm_crash_12m = None
            tcn_crash_12m = None
            if temporal_model and temporal_model.is_trained:
                seq_start = max(0, len(current_features) - temporal_model.WINDOW_SIZE)
                seq_features = current_features.iloc[seq_start:]
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

            # Cox survival model prediction
            cox_crash_12m = None
            if cox_model and cox_model.is_trained:
                try:
                    cox_crash_12m = float(cox_model.predict_proba(current_row, "12m")[0])
                except Exception:
                    pass

            # Meta-stacker ensemble
            if meta_stacker and meta_stacker.is_trained:
                model_preds = {
                    "lgb": lgb_crash_12m,
                    "xgb": xgb_crash_12m,
                    "lstm": lstm_crash_12m,
                    "tcn": tcn_crash_12m,
                    "cox": cox_crash_12m,
                }
                regime_probs_arr = hmm_result.regime_probs if hmm_result.success else None
                ml_crash_prob = float(
                    meta_stacker.predict_proba(
                        model_preds,
                        regime_probs_arr,
                        horizon="12m",
                    )[0]
                )
            else:
                # Simple average of available models
                preds_to_avg = [lgb_crash_12m]
                if xgb_crash_12m is not None:
                    preds_to_avg.append(xgb_crash_12m)
                if lstm_crash_12m is not None:
                    preds_to_avg.append(lstm_crash_12m)
                if cox_crash_12m is not None:
                    preds_to_avg.append(cox_crash_12m)
                ml_crash_prob = float(np.mean(preds_to_avg))

            # Crash timing — which 3-month window is highest risk
            if crash_timing_model and crash_timing_model.is_trained:
                crash_timing_results = crash_timing_model.predict_window(
                    current_row,
                    crash_prob_12m=ml_crash_prob,
                )

            # Multi-horizon return predictions
            ml_predicted_return = float(return_model.predict(current_row, "12m")[0])
            if "6m" in return_model.models:
                ml_return_6m = float(return_model.predict(current_row, "6m")[0])
            if "3m" in return_model.models:
                ml_return_3m = float(return_model.predict(current_row, "3m")[0])

            # Quantile predictions for uncertainty
            quantiles = return_model.predict_quantiles(current_row, "12m")
            ml_return_p10 = float(quantiles.get("p10", [ml_predicted_return - 0.15])[0])
            ml_return_p90 = float(quantiles.get("p90", [ml_predicted_return + 0.15])[0])

            print(f"\n{'='*60}")
            print("  ML ENSEMBLE PREDICTIONS — Current Market State")
            print(f"{'='*60}")
            if ml_crash_3m is not None:
                print(f"  3-Month Crash Prob:   {ml_crash_3m*100:.1f}%")
            if ml_crash_6m is not None:
                print(f"  6-Month Crash Prob:   {ml_crash_6m*100:.1f}%")
            print(f"  12-Month Crash Prob:  {ml_crash_prob*100:.1f}% (ensemble)")
            print()

            # Per-model breakdown
            print("  Per-Model 12M Crash:")
            print(f"    LightGBM:           {lgb_crash_12m*100:.1f}%")
            if xgb_crash_12m is not None:
                print(f"    XGBoost:            {xgb_crash_12m*100:.1f}%")
            if lstm_crash_12m is not None:
                print(f"    LSTM:               {lstm_crash_12m*100:.1f}%")
            if tcn_crash_12m is not None:
                print(f"    TCN:                {tcn_crash_12m*100:.1f}%")
            if cox_crash_12m is not None:
                print(f"    Cox PH:             {cox_crash_12m*100:.1f}%")
            if meta_stacker and meta_stacker.is_trained:
                weights = meta_stacker.get_model_weights("12m")
                if weights:
                    print(
                        f"    Meta-stacker weights: {', '.join(f'{k}={v:.2f}' for k, v in list(weights.items())[:4])}"
                    )
            print()

            # Crash timing
            if crash_timing_results:
                from finpredict.ml.crash_timing import WINDOWS

                print("  Crash Timing (3-month windows):")
                for w in WINDOWS:
                    prob = crash_timing_results.get(w, 0)
                    bar = "#" * int(prob * 40)
                    label = (
                        w.replace("_", " ")
                        .replace("0 3m", "0-3m")
                        .replace("3 6m", "3-6m")
                        .replace("6 9m", "6-9m")
                        .replace("9 12m", "9-12m")
                    )
                    print(f"    {label:14s} {prob*100:5.1f}%  {bar}")
                best_window, best_prob = crash_timing_model.get_most_likely_window(
                    crash_timing_results
                )
                if best_prob > 0.10:
                    print(f"    >> Highest crash risk: {best_window} ({best_prob*100:.1f}%)")
                print()

            if ml_return_3m is not None:
                print(f"  3-Month Expected:     {ml_return_3m*100:+.1f}%")
            if ml_return_6m is not None:
                print(f"  6-Month Expected:     {ml_return_6m*100:+.1f}%")
            print(f"  12-Month Expected:    {ml_predicted_return*100:+.1f}%")
            print(f"  12-Month Range:       [{ml_return_p10*100:+.1f}%, {ml_return_p90*100:+.1f}%]")

            # Top crash signals (gain-based importance)
            top_crash = crash_model.get_top_features(5)
            if top_crash:
                print("\n  Top Crash Signals:")
                for feat, imp in top_crash:
                    val = current_features[feat].iloc[-1]
                    print(f"    {feat} = {val:.4f} (importance: {imp:.1f})")

            # SHAP explanations (why the model is predicting this crash probability)
            # Use shap_values from predict_with_shap (already computed above)
            if shap_values:
                shap_contributions = sorted(
                    shap_values.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )
                print("\n  SHAP Crash Drivers (current prediction):")
                for feat, sv in shap_contributions[:7]:
                    direction = "UP" if sv > 0 else "DOWN"
                    if feat in current_features.columns:
                        val = current_features[feat].iloc[-1]
                        print(f"    {feat} = {val:.4f} → pushes crash prob {direction} ({sv:+.4f})")
                    else:
                        print(f"    {feat} → pushes crash prob {direction} ({sv:+.4f})")
            else:
                shap_contributions = None

            # Counterfactual / what-if sensitivity analysis
            _COUNTERFACTUAL_SCENARIOS = [
                {"label": "VIX spikes to 40", "overrides": {"vix": 40.0}},
                {"label": "Yield curve inverts -1%", "overrides": {"term_spread": -1.0}},
                {
                    "label": "VIX 40 + inverted curve",
                    "overrides": {"vix": 40.0, "term_spread": -1.0},
                },
                {"label": "VIX falls to 15 (calm)", "overrides": {"vix": 15.0}},
            ]
            counterfactual_results = crash_model.run_counterfactual(
                current_row, _COUNTERFACTUAL_SCENARIOS
            )
            if counterfactual_results.get("scenarios"):
                print("\n  What-If Sensitivity:")
                for sc in counterfactual_results["scenarios"]:
                    p12 = sc.get("crash_prob_12m", 0)
                    d12 = sc.get("delta_12m", 0)
                    print(f"    {sc['label']}: 12M crash = {p12*100:.1f}% ({d12*100:+.1f}%)")
        else:
            print("\n[WARN] ML models not available — using statistical defaults")

        # ══════════════════════════════════════════════════════════
        # 9a. ANOMALY DETECTION & CHANGEPOINT ANALYSIS
        # ══════════════════════════════════════════════════════════
        anomaly_report = None
        try:
            print("\n[MODULE 6a] Running anomaly detection...")
            if crash_model and crash_model.is_trained:
                anomaly_det = AnomalyDetector()
                anomaly_det.fit(current_features.iloc[:-1])
                anomaly_report = anomaly_det.anomaly_report(current_row)
                print(
                    f"  [ANOMALY] Status: {anomaly_report['status']} "
                    f"(score={anomaly_report['score']:.3f})"
                )
                if anomaly_report["is_anomalous"]:
                    print(f"  [ANOMALY] {anomaly_report['interpretation']}")
                    # Reduce confidence in ML predictions when extrapolating
                    if ml_crash_prob is not None:
                        conf_factor = anomaly_report["confidence_factor"]
                        base_rate = crash_model._train_crash_rate.get(
                            "12m",
                            config.get("ml", {}).get("crash_base_rate_fallback", 0.12),
                        )
                        ml_above_base = ml_crash_prob > base_rate
                        anomaly_signals_risk = conf_factor < 0.7
                        if ml_above_base and anomaly_signals_risk:
                            # ML predicts elevated crash risk AND anomaly confirms
                            # stress — preserve the ML signal, don't dampen
                            print(
                                f"  [ANOMALY] Crash prob kept at {ml_crash_prob*100:.1f}% "
                                f"(anomaly confirms stress, conf={conf_factor:.2f})"
                            )
                        else:
                            adjusted = ml_crash_prob * conf_factor + base_rate * (1 - conf_factor)
                            print(
                                f"  [ANOMALY] Crash prob adjusted: {ml_crash_prob*100:.1f}% "
                                f"-> {adjusted*100:.1f}% (confidence factor: {conf_factor:.2f})"
                            )
                            ml_crash_prob = adjusted

            # Changepoint detection on recent returns
            cp_detector = BayesianChangepoint(hazard_rate=1 / 252)
            recent_returns = data["Daily_Returns"].dropna()
            cp_report = cp_detector.recent_changepoint(recent_returns, window=90)
            if cp_report["detected"]:
                print(
                    f"  [CHANGEPOINT] Regime shift detected {cp_report['days_ago']} days ago "
                    f"(prob={cp_report['max_prob']:.2f})"
                )
            else:
                print(
                    f"  [CHANGEPOINT] No recent regime shift detected "
                    f"(max_prob={cp_report['max_prob']:.2f})"
                )
        except Exception as e:
            print(f"  [ANOMALY] Detection failed: {e} — continuing without")

        # ══════════════════════════════════════════════════════════
        # 9b. OSINT INTELLIGENCE LAYER (event-driven risk)
        # ══════════════════════════════════════════════════════════
        event_score_result = None
        try:
            print("\n[MODULE 6b] Fetching OSINT intelligence (GDELT)...")
            gdelt_data = fetch_gdelt_data()
            if gdelt_data.get("success"):
                event_score_result = compute_event_score(gdelt_data, fred_features)
                es = event_score_result["event_score"]
                print(f"  [OSINT] Event Score: {es:.2f}")
                print(f"  [OSINT] {event_score_result['interpretation']}")
                for comp, val in event_score_result["components"].items():
                    print(f"    {comp}: {val:.2f}")

                # Adjust ML crash probability with event-driven intelligence
                if ml_crash_prob is not None:
                    original_prob = ml_crash_prob
                    ml_crash_prob = adjust_crash_probability(ml_crash_prob, event_score_result)
                    if abs(ml_crash_prob - original_prob) > 0.01:
                        print(
                            f"  [OSINT] Crash prob adjusted: {original_prob*100:.1f}% → {ml_crash_prob*100:.1f}%"
                        )
            else:
                print("  [OSINT] GDELT unavailable — using ML predictions only")
        except Exception as e:
            print(f"  [OSINT] Intelligence layer error: {e} — continuing without")

        # ══════════════════════════════════════════════════════════
        # 9c. VALIDATION LAYER (cross-check engine outputs)
        # ══════════════════════════════════════════════════════════
        regime_validation = None
        external_validation = None
        try:
            print("\n[MODULE 6c] Running validation checks...")
            from finpredict.validation import validate_regime, validate_external
            from finpredict.data.alternative_fetchers import (
                fetch_aaii_sentiment,
                fetch_naaim_exposure,
                fetch_imf_gdp_forecast,
                fetch_fed_funds_futures,
            )

            regime_validation = validate_regime(data, current_regime, hmm_result)
            print(
                f"  [VALIDATION] Regime: {regime_validation.regime} — "
                f"{'CONFIRMED' if regime_validation.confirmed else 'UNCONFIRMED'}"
            )
            print(f"  [VALIDATION] Confidence: {regime_validation.confidence}")
            for note in regime_validation.notes:
                print(f"    {note}")

            # Fetch alternative data sources (each handles its own errors)
            alt_data = {}
            try:
                alt_data["imf"] = fetch_imf_gdp_forecast()
            except Exception:
                alt_data["imf"] = None
            try:
                alt_data["aaii"] = fetch_aaii_sentiment()
            except Exception:
                alt_data["aaii"] = None
            try:
                alt_data["naaim"] = fetch_naaim_exposure()
            except Exception:
                alt_data["naaim"] = None
            try:
                alt_data["fed_futures"] = fetch_fed_funds_futures()
            except Exception:
                alt_data["fed_futures"] = None

            external_validation = validate_external(
                fred_data,
                ml_crash_prob,
                current_regime,
                data=data,
                alt_data=alt_data,
            )
            print(
                f"  [VALIDATION] External agreement: " f"{external_validation.engine_agreement:.0%}"
            )
            print(
                f"  [VALIDATION] Consensus direction: " f"{external_validation.consensus_direction}"
            )
            for alert in external_validation.divergence_alerts:
                print(f"    [DIVERGENCE] {alert}")
        except Exception as e:
            print(f"  [VALIDATION] Validation layer error: {e} — continuing without")

        # ══════════════════════════════════════════════════════════
        # 9d. LOG PREDICTION SNAPSHOT
        # ══════════════════════════════════════════════════════════
        try:
            selected_model_name = None
            if crash_model and hasattr(crash_model, "selected_model"):
                selected_model_name = crash_model.selected_model.get("12m")
            hmm_probs_for_log = hmm_result.regime_probs if hmm_result.success else None
            log_path = log_prediction(
                current_features=current_features,
                ml_crash_prob=ml_crash_prob,
                ml_crash_3m=ml_crash_3m,
                ml_crash_6m=ml_crash_6m,
                ml_predicted_return=ml_predicted_return,
                ml_return_p10=ml_return_p10,
                ml_return_p90=ml_return_p90,
                lgb_crash_12m=lgb_crash_12m,
                xgb_crash_12m=xgb_crash_12m,
                lstm_crash_12m=lstm_crash_12m,
                tcn_crash_12m=tcn_crash_12m,
                selected_model=selected_model_name,
                current_regime=current_regime,
                hmm_regime_probs=hmm_probs_for_log,
                crash_timing_results=crash_timing_results,
                shap_contributions=shap_contributions,
                crash_timing_model=crash_timing_model,
            )
            print(f"\n[LOG] Prediction snapshot saved to {log_path}")
        except Exception as e:
            print(f"\n[LOG] Prediction logging failed: {e} — continuing")

        # ══════════════════════════════════════════════════════════
        # 10. ML-CONDITIONED MONTE CARLO (secondary)
        # ══════════════════════════════════════════════════════════
        print("\n[MODULE 7] Running ML-conditioned Monte Carlo...")
        # Extract HMM regime data for MC drift tilt
        hmm_means = hmm_result.state_means if hmm_result.success else None
        hmm_probs_arr = hmm_result.regime_probs if hmm_result.success else None
        hmm_vols = hmm_result.state_vols if hmm_result.success else None

        # Compute historical residuals for block bootstrap
        historical_residuals = data["SP500"].pct_change().dropna().values[-252 * 10 :]
        if len(historical_residuals) < 504:
            print("  [WARN] Insufficient history for block bootstrap (<2yr); using i.i.d. sampling")
            historical_residuals = None

        mc_results = run_monte_carlo(
            current_price,
            current_regime,
            current_risk,
            crash_freq,
            current_vix,
            yield_curve,
            val_penalty,
            garch_vol=garch_vol,
            garch_persistence=garch_persistence,
            recession_prob=recession_prob,
            ml_crash_prob=ml_crash_prob,
            ml_predicted_return=ml_predicted_return,
            ml_return_p10=ml_return_p10,
            ml_return_p90=ml_return_p90,
            hmm_state_means=hmm_means,
            hmm_regime_probs=hmm_probs_arr,
            hmm_state_vols=hmm_vols,
            historical_residuals=historical_residuals,
            seed=42,
            xi=garch_xi,
            rho_leverage=garch_rho_leverage,
        )

        # ══════════════════════════════════════════════════════════
        # 10b. HISTORICAL STRESS TESTS
        # ══════════════════════════════════════════════════════════
        stress_results = run_stress_tests(current_price)
        print(f"\n[MODULE 7b] Historical stress tests from S&P ${current_price:,.0f}:")
        for crisis, info in stress_results.items():
            print(f"  {crisis}: trough ${info['trough_price']:,.0f} ({info['drop_pct']*100:.1f}%)")

        # ══════════════════════════════════════════════════════════
        # 11. SECTOR ANALYSIS (factor-based differentiation)
        # ══════════════════════════════════════════════════════════
        forecast_days = get_forecast_days()
        sector_results = analyze_sectors(
            data,
            sector_data,
            forecast_days,
            ml_predicted_return=ml_predicted_return,
            ml_crash_prob=ml_crash_prob,
            ml_return_p10=ml_return_p10,
            ml_return_p90=ml_return_p90,
            garch_vol=garch_vol,
        )

        # ══════════════════════════════════════════════════════════
        # 12. STOCK ANALYSIS
        # ══════════════════════════════════════════════════════════
        rf_rate = float(data["T3M"].dropna().iloc[-1]) if "T3M" in data.columns else 0.04
        rf_rate = max(0.0, min(rf_rate, 0.10))

        n_stocks = config.get("stocks", {}).get("screener_count", 20)
        watchlist = select_stocks_from_sectors(sector_results, n_stocks=n_stocks)
        print(
            f"  [DATA-DRIVEN] Selected {len(watchlist)} stocks from top sectors: "
            f"{', '.join(watchlist[:8])}{'...' if len(watchlist) > 8 else ''}"
        )
        stock_results = analyze_stocks(
            tickers=watchlist,
            forecast_days=forecast_days,
            risk_free_rate=rf_rate,
            ml_predicted_return=ml_predicted_return,
            ml_crash_prob=ml_crash_prob,
        )

        # ══════════════════════════════════════════════════════════
        # 13. GENERATE PDF REPORT
        # ══════════════════════════════════════════════════════════
        print("\n[MODULE 8] Generating PDF report...")
        report_cfg = config.get("reporting", {})
        output_dir = PROJECT_ROOT / report_cfg.get("output_dir", "reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = "Market_Prediction_v7_ML_Report.pdf"
        output_path = str(output_dir / filename)

        generate_report(
            data,
            mc_results,
            bt_results,
            sector_results,
            stock_results,
            current_price,
            current_regime,
            current_risk,
            crash_freq,
            output_path,
            shap_contributions=shap_contributions,
            counterfactual_results=counterfactual_results,
            stress_results=stress_results,
            fred_data=fred_data,
            regime_validation=regime_validation,
            external_validation=external_validation,
            crash_timing_results=crash_timing_results,
        )

        # ══════════════════════════════════════════════════════════
        # SUMMARY
        # ══════════════════════════════════════════════════════════
        print("\n" + "=" * 90)
        print("  ENGINE V7.0 ML-FIRST — ANALYSIS COMPLETE")
        print("=" * 90)
        print(f"  S&P 500:            ${current_price:,.2f}")
        print(
            f"  Regime:             {current_regime}"
            f" ({'HMM' if hmm_result.success else 'rule-based'})"
        )
        print(f"  Risk Score:         {current_risk:.2f}σ")
        if garch_vol is not None:
            print(f"  GARCH Vol:          {garch_vol*100:.1f}% annualized")
            if garch_persistence is not None:
                print(f"  GARCH Persistence:  {garch_persistence:.4f}")
        if recession_prob is not None:
            print(f"  Recession Prob:     {recession_prob*100:.1f}% (FRED)")
        print()

        if ml_crash_prob is not None:
            print("  ═══ ML PREDICTIONS (PRIMARY) ═══")
            print(f"  ML Crash (12m):     {ml_crash_prob*100:.1f}%")
            if ml_crash_6m is not None:
                print(f"  ML Crash (6m):      {ml_crash_6m*100:.1f}%")
            if ml_crash_3m is not None:
                print(f"  ML Crash (3m):      {ml_crash_3m*100:.1f}%")
            print(f"  ML Return (12m):    {ml_predicted_return*100:+.1f}%")
            print(f"  ML Return Range:    [{ml_return_p10*100:+.1f}%, {ml_return_p90*100:+.1f}%]")
        print()

        print("  ═══ MONTE CARLO (UNCERTAINTY BANDS) ═══")
        print(
            f"  5Y Projection:      ${mc_results['final_mean']:,.0f} "
            f"({mc_results['total_return_pct']:+.1f}%)"
        )
        print(f"  Annualized:         {mc_results['annual_return_pct']:.1f}%")
        if ml_crash_prob is not None:
            print(f"  ML Crash (12m):     {ml_crash_prob*100:.1f}%")
        print(f"  MC Crash (12m):     {mc_results['crash_prob_1y']:.1f}% (conditioned on ML)")
        if (
            ml_crash_prob is not None
            and abs(ml_crash_prob * 100 - mc_results["crash_prob_1y"]) > 10
        ):
            print(
                f"  [WARN] ML and MC crash probs diverge by "
                f"{abs(ml_crash_prob * 100 - mc_results['crash_prob_1y']):.0f}pp — calibration concern"
            )
        print(f"  5Y Crash Prob (MC): {mc_results['crash_prob_5y']:.1f}%")
        print(
            f"  1Y CVaR (95%):      {mc_results.get('cvar_95_1y_pct', mc_results['cvar_95_pct']):.1f}%"
        )
        print(
            f"  5Y CVaR (95%):      {mc_results.get('cvar_95_5y_pct', mc_results['cvar_95_pct']):.1f}%"
        )
        print()

        if len(bt_results) > 0:
            print("  ═══ BACKTEST VALIDATION ═══")
            print(f"  MC MAPE:            {bt_results.attrs.get('mape', 0):.1f}%")
            print(f"  ML Crash Brier:     {bt_results.attrs.get('brier_score', 0):.4f}")
            print(f"  ML Crash AUC:       {bt_results.attrs.get('crash_auc', 0):.3f}")
            print(f"  ML Return Corr:     {bt_results.attrs.get('ml_return_corr', 0):.3f}")
            print(f"  ML Return Skill:    {bt_results.attrs.get('ml_return_skill', 0):.3f}")
            cr = bt_results.attrs.get("crash_pred_range", (0, 0))
            print(f"  Crash Pred Range:   [{cr[0]*100:.1f}%, {cr[1]*100:.1f}%]")
            cs = bt_results.attrs.get("crash_pred_std", 0)
            disc = "GOOD" if cs > 0.05 else "POOR"
            print(f"  Crash Pred Std:     {cs*100:.1f}% ({disc})")
        print(f"\n  Report: {output_path}")
        print("=" * 90 + "\n")

        return {
            "data": data,
            "mc_results": mc_results,
            "bt_results": bt_results,
            "sector_results": sector_results,
            "stock_results": stock_results,
            "crash_freq": crash_freq,
            "garch_result": garch_result,
            "hmm_result": hmm_result,
            "fred_data": fred_data,
            "crash_model": crash_model,
            "return_model": return_model,
            "ml_crash_prob": ml_crash_prob,
            "ml_predicted_return": ml_predicted_return,
            "ml_crash_3m": ml_crash_3m,
            "ml_crash_6m": ml_crash_6m,
            "ml_return_p10": ml_return_p10,
            "ml_return_p90": ml_return_p90,
        }

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    results = main()
