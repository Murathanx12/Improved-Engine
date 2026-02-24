"""
Market Prediction Engine v4.5 — Main Entry Point
==================================================

Orchestrates the full pipeline:
    1. Fetch market data (Module 1)
    2. Compute risk score (Module 3) and regimes (Module 2)
    3. Identify historical crashes (Module 4)
    4. Compute valuation penalty
    5. Walk-forward backtest (Module 6)
    6. Monte Carlo projection (Module 5)
    7. Sector analysis (Module 8)
    8. Individual stock analysis (Module 8b)
    9. Generate PDF report (Module 10)

Usage:
    python -m finpredict.main
"""

import sys
import io
import os
import traceback

import numpy as np
import pandas as pd

from finpredict.config import config, PROJECT_ROOT, get_forecast_days
from finpredict.data import cached_fetch_all_data
from finpredict.risk import build_risk_score, detect_regimes, identify_crashes
from finpredict.simulation import run_monte_carlo, compute_valuation_penalty
from finpredict.simulation.backtest import run_backtest
from finpredict.models.sectors import analyze_sectors
from finpredict.models.stocks import select_stocks_from_sectors, analyze_stocks
from finpredict.reporting.pdf_report import generate_report

from datetime import datetime


def main():
    """Complete V4.5 engine pipeline."""
    try:
        # Ensure UTF-8 output
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

        # Reproducibility seed
        np.random.seed(42)

        sim_cfg = config["simulation"]

        print("=" * 90)
        print("  MARKET PREDICTION ENGINE v4.5 — CORE ENGINE")
        print("  Engine-First Reset | Crash Probability Focus")
        print("=" * 90)
        print(f"  Date:        {datetime.now().strftime('%B %d, %Y %I:%M %p')}")
        print(f"  Horizon:     {sim_cfg['forecast_years']} years")
        print(f"  Simulations: {sim_cfg['num_simulations']:,}")
        print(f"  Backtest:    {config['data']['backtest_start']} → present")
        print("=" * 90 + "\n")

        # 1. Fetch data (with caching)
        data, sector_data = cached_fetch_all_data()

        # 2. Compute features
        # Risk score first, then regimes — regime detection uses Risk_Score
        # as a leading indicator to detect transitions earlier
        data["Daily_Returns"] = data["SP500"].pct_change()
        data["Risk_Score"] = build_risk_score(data)
        data["Regime"], current_regime = detect_regimes(data)

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

        # 3. Historical crash analysis
        crash_df, crash_freq = identify_crashes(data)

        # 4. Valuation constraint
        val_penalty, cape_value = compute_valuation_penalty(data)
        cape_str = f" (CAPE: {cape_value:.1f})" if cape_value else " (trend proxy)"
        print(f"[VALUATION] Penalty: {val_penalty*100:+.2f}%/yr{cape_str}\n")

        # 5. Walk-forward backtest
        bt_results = run_backtest(data, crash_freq)

        # 6. Monte Carlo projection
        mc_results = run_monte_carlo(
            current_price, current_regime, current_risk,
            crash_freq, current_vix, yield_curve, val_penalty,
        )

        # 7. Sector analysis
        forecast_days = get_forecast_days()
        sector_results = analyze_sectors(data, sector_data, forecast_days)

        # 8. Individual stock analysis (data-driven from top sectors)
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

        # 9. Generate PDF report
        report_cfg = config.get("reporting", {})
        output_dir = PROJECT_ROOT / report_cfg.get("output_dir", "reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = report_cfg.get("filename_template", "Market_Prediction_v{version}_Report.pdf")
        filename = filename.format(version=report_cfg.get("version", "4.5"))
        output_path = str(output_dir / filename)

        generate_report(
            data, mc_results, bt_results, sector_results, stock_results,
            current_price, current_regime, current_risk, crash_freq, output_path,
        )

        # Summary
        print("=" * 90)
        print("  ENGINE V4.5 — ANALYSIS COMPLETE")
        print("=" * 90)
        print(f"  S&P 500:          ${current_price:,.2f}")
        print(f"  Regime:           {current_regime}")
        print(f"  Risk Score:       {current_risk:.2f}σ")
        print(f"  5Y Projection:    ${mc_results['final_mean']:,.0f} "
              f"({mc_results['total_return_pct']:+.1f}%)")
        print(f"  1Y Crash Prob:    {mc_results['crash_prob_1y']:.1f}%")
        print(f"  5Y Crash Prob:    {mc_results['crash_prob_5y']:.1f}%")
        print(f"  CVaR (95%):       {mc_results['cvar_95_pct']:.1f}%")
        if len(bt_results) > 0:
            print(f"  Backtest MAPE:    {bt_results.attrs.get('mape', 0):.1f}%")
            print(f"  Backtest Coverage:{bt_results.attrs.get('coverage', 0):.1f}%")
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
        }

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    results = main()
