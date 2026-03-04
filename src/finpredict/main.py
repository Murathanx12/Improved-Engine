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
    fetch_fred_data, get_recession_probability, get_macro_features,
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
    build_feature_matrix, build_target_crash, build_target_return,
    build_target_crash_multi, build_target_return_multi,
)
from finpredict.ml.crash_model import CrashPredictor
from finpredict.ml.return_model import ReturnPredictor
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

        # ══════════════════════════════════════════════════════════
        # 2. FETCH FRED TIME SERIES (full history for ML training)
        # ══════════════════════════════════════════════════════════
        fred_data = fetch_fred_data()
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
        if garch_result.success:
            # Persistence = alpha + gamma/2 + beta (for GJR-GARCH)
            garch_persistence = (
                garch_result.alpha
                + garch_result.gamma / 2
                + garch_result.beta
            )
            print(f"  [GARCH] Conditional vol: {garch_vol*100:.1f}%")
            print(f"  [GARCH] Persistence: {garch_persistence:.4f}")

        # ══════════════════════════════════════════════════════════
        # 4. HMM REGIME DETECTION
        # ══════════════════════════════════════════════════════════
        print("[MODULE 3] Detecting market regimes (HMM)...")
        hmm_result = fit_hmm_regimes(data)
        hmm_probs = get_regime_probs(hmm_result)

        # ══════════════════════════════════════════════════════════
        # 5. RISK SCORE + REGIME LABELS
        # ══════════════════════════════════════════════════════════
        print("[MODULE 4] Computing risk score...")
        data["Risk_Score"] = build_risk_score(data)
        data["Regime"], rule_regime = detect_regimes(data)

        if hmm_result.success:
            current_regime = hmm_result.current_regime
            data["Regime"] = hmm_result.regime_labels
            print(f"  [REGIME] Using HMM: {current_regime} "
                  f"(rule-based: {rule_regime})")
        else:
            current_regime = rule_regime
            print(f"  [REGIME] Using rule-based: {current_regime}")

        current_price = float(data["SP500"].iloc[-1])
        current_risk = float(data["Risk_Score"].iloc[-1])
        current_vix = (
            float(data["VIX"].iloc[-1]) if "VIX" in data.columns else 20.0
        )
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
        bt_results = run_backtest(data, crash_freq, fred_data=fred_data)

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

        crash_model = bt_results.attrs.get("crash_model")
        return_model = bt_results.attrs.get("return_model")

        if (crash_model and crash_model.is_trained
                and return_model and return_model.is_trained):
            current_features = build_feature_matrix(data, fred_data=fred_data)
            current_row = current_features.iloc[-1:]

            # Multi-horizon crash predictions
            ml_crash_prob = float(crash_model.predict_proba(current_row, "12m")[0])
            if "6m" in crash_model.models:
                ml_crash_6m = float(crash_model.predict_proba(current_row, "6m")[0])
            if "3m" in crash_model.models:
                ml_crash_3m = float(crash_model.predict_proba(current_row, "3m")[0])

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
            print(f"  ML PREDICTIONS — Current Market State")
            print(f"{'='*60}")
            if ml_crash_3m is not None:
                print(f"  3-Month Crash Prob:   {ml_crash_3m*100:.1f}%")
            if ml_crash_6m is not None:
                print(f"  6-Month Crash Prob:   {ml_crash_6m*100:.1f}%")
            print(f"  12-Month Crash Prob:  {ml_crash_prob*100:.1f}%")
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
                print(f"\n  Top Crash Signals:")
                for feat, imp in top_crash:
                    val = current_features[feat].iloc[-1]
                    print(f"    {feat} = {val:.4f} (importance: {imp:.1f})")

            # SHAP explanations (why the model is predicting this crash probability)
            shap_contributions = crash_model.get_shap_values(current_row, "12m")
            if shap_contributions:
                print(f"\n  SHAP Crash Drivers (current prediction):")
                for feat, sv in shap_contributions[:7]:
                    direction = "UP" if sv > 0 else "DOWN"
                    val = current_features[feat].iloc[-1]
                    print(f"    {feat} = {val:.4f} → pushes crash prob {direction} ({sv:+.4f})")

            # Counterfactual / what-if sensitivity analysis
            _COUNTERFACTUAL_SCENARIOS = [
                {"label": "VIX spikes to 40",         "overrides": {"vix": 40.0}},
                {"label": "Yield curve inverts -1%",   "overrides": {"term_spread": -1.0}},
                {"label": "VIX 40 + inverted curve",   "overrides": {"vix": 40.0, "term_spread": -1.0}},
                {"label": "VIX falls to 15 (calm)",    "overrides": {"vix": 15.0}},
            ]
            counterfactual_results = crash_model.run_counterfactual(
                current_row, _COUNTERFACTUAL_SCENARIOS
            )
            if counterfactual_results.get("scenarios"):
                print(f"\n  What-If Sensitivity:")
                for sc in counterfactual_results["scenarios"]:
                    p12 = sc.get("crash_prob_12m", 0)
                    d12 = sc.get("delta_12m", 0)
                    print(f"    {sc['label']}: 12M crash = {p12*100:.1f}% ({d12*100:+.1f}%)")
        else:
            print("\n[WARN] ML models not available — using statistical defaults")

        # ══════════════════════════════════════════════════════════
        # 9b. OSINT INTELLIGENCE LAYER (event-driven risk)
        # ══════════════════════════════════════════════════════════
        event_score_result = None
        try:
            print(f"\n[MODULE 6b] Fetching OSINT intelligence (GDELT)...")
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
                        print(f"  [OSINT] Crash prob adjusted: {original_prob*100:.1f}% → {ml_crash_prob*100:.1f}%")
            else:
                print(f"  [OSINT] GDELT unavailable — using ML predictions only")
        except Exception as e:
            print(f"  [OSINT] Intelligence layer error: {e} — continuing without")

        # ══════════════════════════════════════════════════════════
        # 10. ML-CONDITIONED MONTE CARLO (secondary)
        # ══════════════════════════════════════════════════════════
        print(f"\n[MODULE 7] Running ML-conditioned Monte Carlo...")
        # Extract HMM regime data for MC drift tilt
        hmm_means = hmm_result.state_means if hmm_result.success else None
        hmm_probs_arr = hmm_result.regime_probs if hmm_result.success else None
        hmm_vols = hmm_result.state_vols if hmm_result.success else None

        mc_results = run_monte_carlo(
            current_price, current_regime, current_risk,
            crash_freq, current_vix, yield_curve, val_penalty,
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
            data, sector_data, forecast_days,
            ml_predicted_return=ml_predicted_return,
            ml_crash_prob=ml_crash_prob,
            ml_return_p10=ml_return_p10,
            ml_return_p90=ml_return_p90,
            garch_vol=garch_vol,
        )

        # ══════════════════════════════════════════════════════════
        # 12. STOCK ANALYSIS
        # ══════════════════════════════════════════════════════════
        rf_rate = (
            float(data["T3M"].dropna().iloc[-1])
            if "T3M" in data.columns else 0.04
        )
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
        print(f"\n[MODULE 8] Generating PDF report...")
        report_cfg = config.get("reporting", {})
        output_dir = PROJECT_ROOT / report_cfg.get("output_dir", "reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = "Market_Prediction_v7_ML_Report.pdf"
        output_path = str(output_dir / filename)

        generate_report(
            data, mc_results, bt_results, sector_results, stock_results,
            current_price, current_regime, current_risk, crash_freq, output_path,
            shap_contributions=shap_contributions,
            counterfactual_results=counterfactual_results,
            stress_results=stress_results,
            fred_data=fred_data,
        )

        # ══════════════════════════════════════════════════════════
        # SUMMARY
        # ══════════════════════════════════════════════════════════
        print("\n" + "=" * 90)
        print("  ENGINE V7.0 ML-FIRST — ANALYSIS COMPLETE")
        print("=" * 90)
        print(f"  S&P 500:            ${current_price:,.2f}")
        print(f"  Regime:             {current_regime}"
              f" ({'HMM' if hmm_result.success else 'rule-based'})")
        print(f"  Risk Score:         {current_risk:.2f}σ")
        if garch_vol is not None:
            print(f"  GARCH Vol:          {garch_vol*100:.1f}% annualized")
            if garch_persistence is not None:
                print(f"  GARCH Persistence:  {garch_persistence:.4f}")
        if recession_prob is not None:
            print(f"  Recession Prob:     {recession_prob*100:.1f}% (FRED)")
        print()

        if ml_crash_prob is not None:
            print(f"  ═══ ML PREDICTIONS (PRIMARY) ═══")
            print(f"  ML Crash (12m):     {ml_crash_prob*100:.1f}%")
            if ml_crash_6m is not None:
                print(f"  ML Crash (6m):      {ml_crash_6m*100:.1f}%")
            if ml_crash_3m is not None:
                print(f"  ML Crash (3m):      {ml_crash_3m*100:.1f}%")
            print(f"  ML Return (12m):    {ml_predicted_return*100:+.1f}%")
            print(f"  ML Return Range:    [{ml_return_p10*100:+.1f}%, {ml_return_p90*100:+.1f}%]")
        print()

        print(f"  ═══ MONTE CARLO (UNCERTAINTY BANDS) ═══")
        print(f"  5Y Projection:      ${mc_results['final_mean']:,.0f} "
              f"({mc_results['total_return_pct']:+.1f}%)")
        print(f"  Annualized:         {mc_results['annual_return_pct']:.1f}%")
        print(f"  1Y Crash Prob (MC): {mc_results['crash_prob_1y']:.1f}%")
        print(f"  5Y Crash Prob (MC): {mc_results['crash_prob_5y']:.1f}%")
        print(f"  CVaR (95%):         {mc_results['cvar_95_pct']:.1f}%")
        print()

        if len(bt_results) > 0:
            print(f"  ═══ BACKTEST VALIDATION ═══")
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
