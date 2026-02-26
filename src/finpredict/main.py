"""
Market Prediction Engine v4.5 — Main Entry Point (v2)
=======================================================

Pipeline (upgraded):
    1. Fetch market data (Yahoo Finance) + FRED macro indicators
    2. Fit GJR-GARCH for time-varying volatility
    3. Fit HMM for probabilistic regime detection
    4. Compute risk score + identify historical crashes
    5. Compute FRED recession probability
    6. Compute valuation penalty
    7. Walk-forward backtest (with GARCH vol + calibration fix)
    8. Monte Carlo projection (GARCH + HMM + FRED conditioned)
    9. Sector and stock analysis
   10. Generate PDF report

Usage:
    python -m finpredict.main
"""

import sys
import io
import traceback

import numpy as np
import pandas as pd

from finpredict.config import config, PROJECT_ROOT, get_forecast_days
from finpredict.data import cached_fetch_all_data
from finpredict.data.fred_fetcher import fetch_fred_data, get_recession_probability, get_macro_features
from finpredict.risk import build_risk_score, detect_regimes, identify_crashes
from finpredict.models.garch import fit_garch
from finpredict.models.hmm_regimes import fit_hmm_regimes, get_regime_probs
from finpredict.simulation import run_monte_carlo, compute_valuation_penalty
from finpredict.simulation.backtest import run_backtest
from finpredict.models.sectors import analyze_sectors
from finpredict.models.stocks import select_stocks_from_sectors, analyze_stocks
from finpredict.reporting.pdf_report import generate_report

from datetime import datetime


def main():
    """Complete V4.5 engine pipeline (v2 — GARCH + HMM + FRED)."""
    try:
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

        np.random.seed(42)
        sim_cfg = config["simulation"]

        print("=" * 90)
        print("  MARKET PREDICTION ENGINE v4.5 — CORE ENGINE (v2)")
        print("  Calibrated | GARCH Volatility | HMM Regimes | FRED Macro")
        print("=" * 90)
        print(f"  Date:        {datetime.now().strftime('%B %d, %Y %I:%M %p')}")
        print(f"  Horizon:     {sim_cfg['forecast_years']} years")
        print(f"  Simulations: {sim_cfg['num_simulations']:,}")
        print(f"  Backtest:    {config['data']['backtest_start']} → present")
        print("=" * 90 + "\n")

        # ── 1. Fetch market data ─────────────────────────────────────────
        data, sector_data = cached_fetch_all_data()

        # ── 1b. Fetch FRED macro data ────────────────────────────────────
        fred_data = fetch_fred_data()
        recession_prob = None
        if fred_data:
            recession_prob = get_recession_probability(fred_data)
            macro = get_macro_features(fred_data)
            print(f"  [FRED] Recession probability: {recession_prob*100:.1f}%")
            for k, v in macro.items():
                print(f"  [FRED] {k}: {v:.2f}")
            print()

        # ── 2. GJR-GARCH volatility ─────────────────────────────────────
        data["Daily_Returns"] = data["SP500"].pct_change()
        returns = data["Daily_Returns"].dropna()
        garch_result = fit_garch(returns)
        garch_vol = garch_result.current_vol if garch_result.success else None

        # ── 3. HMM regime detection ──────────────────────────────────────
        hmm_result = fit_hmm_regimes(data)
        hmm_probs = get_regime_probs(hmm_result)

        # ── 4. Risk score + rule-based regimes (as backup/complement) ────
        data["Risk_Score"] = build_risk_score(data)
        data["Regime"], rule_regime = detect_regimes(data)

        # Use HMM regime if available, fall back to rule-based
        if hmm_result.success:
            current_regime = hmm_result.current_regime
            data["Regime"] = hmm_result.regime_labels
            print(f"  [REGIME] Using HMM: {current_regime} "
                  f"(rule-based would say: {rule_regime})")
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

        # ── 5. Historical crash analysis ─────────────────────────────────
        crash_df, crash_freq = identify_crashes(data)

        # ── 6. Valuation constraint ──────────────────────────────────────
        val_penalty, cape_value = compute_valuation_penalty(data)
        cape_str = f" (CAPE: {cape_value:.1f})" if cape_value else " (trend proxy)"
        print(f"[VALUATION] Penalty: {val_penalty*100:+.2f}%/yr{cape_str}\n")

        # ── 7. Walk-forward backtest ─────────────────────────────────────
        bt_results = run_backtest(data, crash_freq)

        # ── 8. Monte Carlo projection (conditioned on GARCH + HMM + FRED)
        mc_results = run_monte_carlo(
            current_price, current_regime, current_risk,
            crash_freq, current_vix, yield_curve, val_penalty,
            garch_vol=garch_vol,
            recession_prob=recession_prob,
        )

        # ── 9. Sector analysis ───────────────────────────────────────────
        forecast_days = get_forecast_days()
        sector_results = analyze_sectors(data, sector_data, forecast_days)

        # ── 10. Stock analysis ───────────────────────────────────────────
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
        )

        # ── 11. Generate PDF report ──────────────────────────────────────
        report_cfg = config.get("reporting", {})
        output_dir = PROJECT_ROOT / report_cfg.get("output_dir", "reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = report_cfg.get(
            "filename_template", "Market_Prediction_v{version}_Report.pdf"
        )
        filename = filename.format(version=report_cfg.get("version", "4.5"))
        output_path = str(output_dir / filename)

        generate_report(
            data, mc_results, bt_results, sector_results, stock_results,
            current_price, current_regime, current_risk, crash_freq, output_path,
        )

        # ── Summary ──────────────────────────────────────────────────────
        print("=" * 90)
        print("  ENGINE V4.5 (v2) — ANALYSIS COMPLETE")
        print("=" * 90)
        print(f"  S&P 500:          ${current_price:,.2f}")
        print(f"  Regime:           {current_regime}"
              f" ({'HMM' if hmm_result.success else 'rule-based'})")
        print(f"  Risk Score:       {current_risk:.2f}σ")
        if garch_vol:
            print(f"  GARCH Vol:        {garch_vol*100:.1f}% annualized")
        if recession_prob is not None:
            print(f"  Recession Prob:   {recession_prob*100:.1f}% (FRED)")
        print(f"  5Y Projection:    ${mc_results['final_mean']:,.0f} "
              f"({mc_results['total_return_pct']:+.1f}%)")
        print(f"  1Y Crash Prob:    {mc_results['crash_prob_1y']:.1f}%")
        print(f"  5Y Crash Prob:    {mc_results['crash_prob_5y']:.1f}%")
        print(f"  CVaR (95%):       {mc_results['cvar_95_pct']:.1f}%")
        if len(bt_results) > 0:
            print(f"  Backtest MAPE:    {bt_results.attrs.get('mape', 0):.1f}%")
            print(f"  Brier Score:      {bt_results.attrs.get('brier_score', 0):.4f}")
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
        }

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    results = main()
