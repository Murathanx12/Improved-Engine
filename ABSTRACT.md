# Aegis Finance Engine v7 — Project Abstract

## What It Does

A quantitative market prediction engine that forecasts S&P 500 crash probabilities and expected returns over 3-month, 6-month, and 12-month horizons. It combines five ML models (LightGBM, XGBoost, LSTM, TCN, Cox survival) in a regime-aware meta-stacker ensemble, feeds predictions into a jump-diffusion Monte Carlo simulation, and outputs a 21-page PDF report with scenario analysis, sector projections, and risk metrics.

## What Was Achieved

- **Full pipeline operational:** Data ingestion (Yahoo Finance + 22 FRED macro series + 6 global indices), feature engineering (80+ backward-looking features), walk-forward backtesting (51 prediction points, 2000-present), Monte Carlo simulation (2000 paths, 5-year horizon), and automated PDF report generation.
- **Five-model ensemble:** LightGBM and XGBoost for snapshot features, LSTM and TCN for temporal sequences, Cox proportional hazards for survival analysis — combined via a logistic meta-stacker that learns regime-dependent weightings.
- **Zero data leakage:** Expanding-window backtest with purge gaps (70-265 trading days per horizon), temporal train/val splits, and publication-lag-shifted FRED data.
- **19 bugs fixed across 4 development phases**, covering prediction corruption, methodology errors, and dead code.

## What the v7.0 Report Showed

The engine's first complete run (March 2026) produced:
- **12-month crash probability:** 19.3% (reasonable given VIX ~25, no active recession)
- **Expected annual return:** +8.7% (ML prediction, pre-simulation)
- **5-year Monte Carlo annualized return:** -1.5% (significantly below +3-6% institutional consensus)
- **Brier Score:** 0.193 (Brier Skill Score: -0.39 vs climatology baseline of 0.25)

The -1.5% annualized result revealed a **systematic bearish bias of 4-8% annually** vs institutional consensus.

## Root Causes Identified

1. **Missing jump-diffusion compensator (Bug 20, ~1.4-2.7%/yr):** The Merton (1976) drift adjustment term -λk was computed but never applied to the simulation loop. Negative jumps mechanically reduced E[S(T)] below the risk-neutral expectation without the compensating positive drift.

2. **Bearish-skewed scenario weights (Bug 21, ~1-2%/yr):** Base scenario probabilities allocated 87.5% weight to negative/neutral outcomes. Historical base rates show ~70% of calendar years are positive. Rebalanced to ~65% positive/neutral.

3. **Stale institutional benchmarks (Bug 22):** Several 2024-era forecasts were being used to anchor the consensus return. Updated to 2026 published values (consensus ~5.7% → ~5.9%).

4. **Lagging indicator dominance:** Unemployment z-score was the top SHAP feature (+0.20), but unemployment is a lagging indicator — it peaks after recessions start. Added initial jobless claims (ICSA, leading by 6-9 months) and Chicago Fed NFCI (leading by 3-6 months) as Bug 23.

5. **Backtest hyperparameter inflation (Bug 24):** Walk-forward backtest was training models with 2-3x the intended estimators/dimensions, increasing computation without improving accuracy.

## Limitations

- **Crash model discrimination is weak:** BSS of -0.39 means the ensemble performs worse than always predicting the historical crash base rate. The models capture aggregate risk levels but cannot reliably distinguish "crash imminent" from "elevated risk." This is partly fundamental — crashes are rare, non-stationary events — and partly a feature engineering gap.

- **Feature set lacks true leading indicators:** The 80+ features are dominated by concurrent and lagging signals (price-derived momentum, unemployment, CPI). The ICSA/NFCI additions help but are insufficient. Ideally the feature set would include: credit default swap spreads, options skew term structure, high-frequency order flow imbalance, and corporate earnings revision breadth.

- **Monte Carlo inherits ML bias:** The simulation uses ML-predicted returns as drift input. If the ML models are systematically bearish (due to feature composition), the entire 2000-path simulation is anchored to a biased starting point. The institutional consensus anchor helps but doesn't fully correct this.

- **Single-market focus:** S&P 500 only. Cross-market contagion features exist (6 global indices) but only as inputs, not as separate prediction targets. No bond/commodity/FX forecasting.

- **PDF-only output:** The 21-page report is comprehensive but static. No interactive dashboard, no API endpoint, no real-time monitoring. A Streamlit dashboard or REST API would make the predictions more actionable.

- **No live trading integration:** Predictions are point-in-time snapshots. There is no position sizing, no portfolio construction, no execution layer. The engine is analytical, not operational.

## What Can Be Improved

### High Impact (next iteration)
1. **Feature engineering:** Replace/supplement lagging indicators with leading signals. Priority: NFCI (added), yield curve dynamics (slope + acceleration), credit spreads momentum, options market implied distributions.
2. **Calibration:** Implement isotonic regression or Venn-ABERS calibration on the meta-stacker output. Current Platt scaling on individual models doesn't guarantee ensemble calibration.
3. **Interactive output:** Streamlit dashboard with scenario sliders, model contribution waterfall charts, and real-time data refresh.

### Medium Impact
4. **Regime-conditional evaluation:** Compute Brier scores separately for bull/bear/crisis regimes. Aggregate BSS masks potential strength in crisis detection.
5. **Ensemble diversity:** The 5 models use the same feature matrix. Adding fundamentally different signal sources (NLP sentiment from earnings calls, options-implied distributions) would increase ensemble value.
6. **Temporal cross-validation:** Replace simple expanding window with purged k-fold CV to increase the number of evaluation points and reduce variance in skill estimates.

### Lower Impact
7. **Multi-asset extension:** Extend to bonds (TLT, HYG), commodities (gold, oil), and international equities for portfolio-level risk assessment.
8. **Execution layer:** Position sizing based on Kelly criterion or risk parity, with crash probability as the primary signal.
9. **Monitoring:** Automated Brier score degradation alerts (foundation exists in `brier_monitor.py`), data quality dashboards, model drift detection.
