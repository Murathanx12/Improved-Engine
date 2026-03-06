# Architecture & Methodology

This document explains the FinPredict engine's architecture, data flow, ML methodology, and statistical models in detail.

---

## Table of Contents

- [System Diagram](#system-diagram)
- [Pipeline Overview](#pipeline-overview)
- [Module Details](#module-details)
  - [1. Data Layer](#1-data-layer)
  - [2. Statistical Models](#2-statistical-models)
  - [3. Feature Engineering](#3-feature-engineering)
  - [4. ML Ensemble](#4-ml-ensemble)
  - [5. Walk-Forward Backtesting](#5-walk-forward-backtesting)
  - [6. Monte Carlo Simulation](#6-monte-carlo-simulation)
  - [7. Risk & Regime Detection](#7-risk--regime-detection)
  - [8. Intelligence Layer](#8-intelligence-layer)
  - [9. Validation Layer](#9-validation-layer)
  - [10. Reporting](#10-reporting)
- [Key Design Principles](#key-design-principles)
- [Configuration Architecture](#configuration-architecture)
- [Data Flow Diagram](#data-flow-diagram)

---

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FINPREDICT ENGINE v7.0 — PIPELINE                      │
└─────────────────────────────────────────────────────────────────────────────┘

 ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 │ Yahoo Finance│   │   FRED API   │   │   Finnhub    │   │   GDELT      │
 │  (Prices,    │   │  (Macro      │   │  (Sentiment, │   │  (Geopolitical│
 │   VIX, ETFs) │   │   Indicators)│   │   Quotes)    │   │   Events)    │
 └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
        │                  │                   │                   │
        └──────────┬───────┴───────────┬───────┘                   │
                   │                   │                           │
                   ▼                   ▼                           ▼
        ┌─────────────────┐  ┌─────────────────┐       ┌──────────────────┐
        │   DATA CACHE    │  │  DATA CACHE     │       │ OSINT LAYER      │
        │  (Parquet)      │  │  (FRED Series)  │       │ (Event Scoring)  │
        └────────┬────────┘  └────────┬────────┘       └────────┬─────────┘
                 │                    │                          │
                 └────────┬───────────┘                          │
                          │                                      │
                          ▼                                      │
        ┌─────────────────────────────────┐                      │
        │      STATISTICAL MODELS         │                      │
        │  ┌───────────┐ ┌─────────────┐  │                      │
        │  │ GJR-GARCH │ │ HMM Regimes │  │                      │
        │  │(Volatility)│ │(Bull/Bear/  │  │                      │
        │  │           │ │ Neutral/    │  │                      │
        │  │           │ │ Crisis)     │  │                      │
        │  └─────┬─────┘ └──────┬──────┘  │                      │
        └────────┼──────────────┼─────────┘                      │
                 │              │                                 │
                 ▼              ▼                                 │
        ┌─────────────────────────────────┐                      │
        │      FEATURE ENGINEERING        │                      │
        │  (80+ features from market +    │                      │
        │   macro data, FRED time series, │                      │
        │   GARCH outputs, HMM states)    │                      │
        └────────────────┬────────────────┘                      │
                         │                                       │
                         ▼                                       │
        ┌─────────────────────────────────────────────┐          │
        │           ML ENSEMBLE                       │          │
        │                                             │          │
        │  ┌──────────┐ ┌──────────┐ ┌─────────────┐ │          │
        │  │ LightGBM │ │ XGBoost  │ │ LSTM + TCN  │ │          │
        │  │ (Crash + │ │ (Crash)  │ │ (Temporal   │ │          │
        │  │  Return)  │ │          │ │  Ensemble)  │ │          │
        │  └────┬─────┘ └────┬─────┘ └──────┬──────┘ │          │
        │       │             │              │        │          │
        │       └──────┬──────┴──────────────┘        │          │
        │              │                              │          │
        │              ▼                              │          │
        │       ┌─────────────┐                       │          │
        │       │ MetaStacker │ (Learned ensemble     │          │
        │       │  (Ridge)    │  weights per model)   │          │
        │       └──────┬──────┘                       │          │
        │              │                              │          │
        │   ┌──────────┼──────────────┐               │          │
        │   ▼          ▼              ▼               │          │
        │ Crash     Expected      Quantile            │          │
        │ Prob      Return       [P10, P90]           │          │
        │ (3m/6m/   (3m/6m/                          │          │
        │  12m)      12m)                             │          │
        └──────┬──────────────────────────────────────┘          │
               │                                                 │
               │◄────────────────────────────────────────────────┘
               │        (Event score adjusts crash prob)
               ▼
        ┌─────────────────────────────────┐
        │  ANOMALY + CHANGEPOINT          │
        │  DETECTION                      │
        │  (Adjust confidence when        │
        │   market is in novel state)     │
        └────────────────┬────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────────────┐
        │  ML-CONDITIONED MONTE CARLO                 │
        │                                             │
        │  10,000 paths x 5 years                     │
        │  • Jump-diffusion with Student-t tails      │
        │  • 8 scenarios (dynamic probability)        │
        │  • GARCH vol dynamics + HMM regime blending │
        │  • Block bootstrap (preserves clustering)   │
        │  • ML crash prob conditions jump intensity   │
        │  • Institutional return anchoring           │
        │  • Valuation constraint (CAPE penalty)      │
        └────────────────┬────────────────────────────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
        ┌──────────┐ ┌────────┐ ┌──────────┐
        │  Sector  │ │ Stock  │ │Historical│
        │ Analysis │ │Analysis│ │ Stress   │
        │(11 ETFs) │ │(Top 20)│ │  Tests   │
        └────┬─────┘ └───┬────┘ └────┬─────┘
             │            │           │
             └──────┬─────┴───────────┘
                    │
                    ▼
        ┌─────────────────────────────────┐
        │   VALIDATION LAYER              │
        │  • Regime cross-check (SMA,     │
        │    breadth, HMM vs rules)       │
        │  • External sources (LEI, SLOOS,│
        │    AAII, NAAIM, IMF GDP)        │
        └────────────────┬────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────┐
        │       PDF REPORT                │
        │  (Charts, tables, SHAP,         │
        │   crash timing, counterfactuals)│
        └─────────────────────────────────┘
```

---

## Pipeline Overview

The engine runs as a sequential pipeline defined in `src/finpredict/main.py`. Each module feeds data forward to the next:

| Step | Module | Source File | Purpose |
|------|--------|-------------|---------|
| 1 | Data Fetch | `data/fetchers.py`, `data/cache.py` | Download and cache market data |
| 2 | FRED Fetch | `data/fred_fetcher.py` | Macro economic time series |
| 3 | GARCH | `models/garch.py` | Conditional volatility estimation |
| 4 | HMM Regimes | `models/hmm_regimes.py` | Market regime classification |
| 5 | Risk Score | `risk/scoring.py` | 9-factor composite risk indicator |
| 6 | Crash History | `risk/crashes.py` | Historical crash identification |
| 7 | Valuation | `simulation/valuation.py` | CAPE-based penalty calculation |
| 8 | Backtest | `simulation/backtest.py` | Walk-forward ML training + evaluation |
| 9 | ML Predictions | `ml/*.py` | Current crash prob + return estimates |
| 9a | Anomaly Detection | `ml/anomaly_detector.py` | Out-of-distribution detection |
| 9b | OSINT | `intelligence/gdelt_fetcher.py` | Geopolitical event adjustment |
| 9c | Validation | `validation/*.py` | Cross-check predictions |
| 10 | Monte Carlo | `simulation/scenarios.py` | 5-year projection simulation |
| 11 | Sector Analysis | `models/sectors.py` | Factor-based sector ranking |
| 12 | Stock Analysis | `models/stocks.py` | Individual stock screening |
| 13 | Report | `reporting/pdf_report.py` | PDF generation |

---

## Module Details

### 1. Data Layer

**Files:** `data/fetchers.py`, `data/cache.py`, `data/fred_fetcher.py`, `data/alternative_fetchers.py`

The data layer fetches market data from multiple sources and caches results locally as Parquet files to avoid redundant API calls.

**Market data (Yahoo Finance):**
- S&P 500, VIX, Treasury yields (3M, 10Y, 30Y)
- Credit ETFs (HYG, LQD), Gold, NASDAQ, Russell 2000
- 11 sector ETFs (XLK, XLV, XLF, etc.)
- Mega-cap tickers for market concentration (HHI) calculation

**Macro data (FRED):**
- Yield spread (T10Y3M) -- recession predictor
- Sahm Rule indicator
- Unemployment, CPI, Fed Funds rate
- Consumer sentiment (UMich)
- Credit spreads (HY OAS, IG OAS)
- Leading Economic Index (LEI)
- Senior Loan Officer Survey (SLOOS)

**Cache behavior:**
- Default TTL: 1 hour (configurable in `engine_config.yaml`)
- Format: Parquet (fast, compressed)
- Location: `data_cache/` directory
- Force refresh: `make refresh`

### 2. Statistical Models

#### GJR-GARCH Volatility (`models/garch.py`)

The GJR-GARCH(1,1) model captures **volatility clustering** and **leverage effect** (bad news increases volatility more than good news):

```
σ²(t) = ω + α·ε²(t-1) + γ·ε²(t-1)·I(ε<0) + β·σ²(t-1)
```

Where `γ > 0` captures the asymmetric leverage effect. The model provides:
- Current conditional volatility
- Persistence parameter (α + γ/2 + β)
- Derived parameters for Monte Carlo: leverage correlation (ρ) and vol-of-vol (ξ)

#### Hidden Markov Model (`models/hmm_regimes.py`)

A Gaussian HMM with 4 hidden states detects market regimes:
- **Bull**: High returns, low volatility
- **Neutral**: Moderate returns, moderate volatility
- **Bear**: Negative returns, elevated volatility
- **Crisis**: Extreme negative returns, very high volatility

The HMM outputs regime probabilities that feed into Monte Carlo drift/volatility blending.

### 3. Feature Engineering

**File:** `ml/features.py`

Builds 80+ features from market and macro data, including:

- **Price features**: Returns (1d, 5d, 21d, 63d, 252d), drawdown, distance from highs
- **Volatility features**: Rolling vol (21d, 63d), GARCH vol, VIX, VIX term structure
- **Momentum features**: RSI, moving average ratios, rate of change
- **Credit features**: HY/IG spread changes, credit stress indicators
- **Macro features**: Yield curve slope, FRED indicators (aligned to daily frequency)
- **Regime features**: HMM state probabilities, risk score
- **Market breadth**: NASDAQ/S&P ratio, Russell/S&P divergence, sector breadth
- **Concentration**: Herfindahl-Hirschman Index (HHI) from mega-cap weights

### 4. ML Ensemble

The engine uses four model families, combined via a MetaStacker:

| Model | File | Type | Strengths |
|-------|------|------|-----------|
| LightGBM | `ml/crash_model.py`, `ml/return_model.py` | Gradient boosting | Fast, handles missing values, feature importance |
| XGBoost | `ml/xgboost_model.py` | Gradient boosting | Regularization, second model for ensemble diversity |
| LSTM | `ml/sequence_model.py` | Recurrent neural net | Temporal patterns, long-range dependencies |
| TCN | `ml/sequence_model.py` | Temporal conv net | Parallel computation, dilated receptive field |

**MetaStacker** (`ml/meta_stacker.py`): A Ridge regression that learns optimal weights for each model's predictions, conditioned on HMM regime probabilities. Falls back to simple averaging if fewer than 2 models are available.

**Crash Timing** (`ml/crash_timing.py`): Predicts which 3-month window (0-3m, 3-6m, 6-9m, 9-12m) is most likely for a crash, given that a crash will happen.

**Targets:**
- **Crash target**: Binary -- did the market fall ≥20% within the next N months?
- **Return target**: Continuous -- what was the N-month forward return?
- **Multi-horizon**: 3m, 6m, 12m targets built with configurable purge gaps to prevent data leakage

### 5. Walk-Forward Backtesting

**File:** `simulation/backtest.py`

The backtest uses an **expanding window** approach:

```
Training window:         [1990 ──────────── T]
Purge gap:                                   [T ── T+gap]
Prediction point:                                        T+gap
Forward evaluation:                                      [T+gap ── T+gap+252d]

Next iteration:          [1990 ──────────────── T+3mo]
                                                         [gap]  T+3mo+gap
```

Key settings (from `engine_config.yaml`):
- Step size: 3 months between prediction points
- Lookback: 10 years minimum history
- Forward horizon: 252 trading days (1 year)
- Purge gaps: 70/140/265 days for 3m/6m/12m horizons

**Evaluation metrics:**
- Brier Score (crash probability calibration)
- AUC-ROC (crash discrimination)
- Return correlation and skill score
- MAPE for Monte Carlo forecasts

### 6. Monte Carlo Simulation

**File:** `simulation/scenarios.py`

The Monte Carlo engine runs 10,000 paths over 5 years using **ML-conditioned jump-diffusion**:

```
dS/S = μ·dt + σ·dW + J·dN
```

Where:
- **μ (drift)** = blend of institutional consensus + ML predicted return + HMM regime drift + valuation penalty
- **σ (volatility)** = GARCH-derived with stochastic vol-of-vol (ξ) and leverage (ρ)
- **J (jumps)** = sudden crash events, intensity scaled by ML crash probability
- **dN** = Poisson process with Student-t jump sizes (fat tails)
- **dW** = block-bootstrapped innovations (preserves volatility clustering)

**8 scenarios** (Base Case, AI Boom, Soft Landing, Correction, Stagflation, Recession, AI Bubble, Geopolitical Crisis) each have base probabilities that are dynamically adjusted based on:
- Current regime (HMM)
- VIX level
- Risk score
- Yield curve slope
- ML crash probability

### 7. Risk & Regime Detection

**Files:** `risk/scoring.py`, `risk/regimes.py`, `risk/crashes.py`

**9-Factor Composite Risk Score:**

| Factor | Weight | Description |
|--------|--------|-------------|
| VIX | 2.0 | Fear gauge (z-score) |
| Yield Curve | 1.8 | 10Y-3M spread (inversion = recession) |
| Credit Spread | 1.9 | HYG/LQD stress |
| Long Yield Vol | 1.0 | Bond market disruption |
| Momentum Exhaustion | 1.5 | >2σ moves from mean |
| Short-Term Vol | 1.3 | 20-day rolling vol regime |
| Gold/Stock Ratio | 1.2 | Flight to safety |
| Market Breadth | 1.0 | NASDAQ/S&P leadership |
| Small Cap Divergence | 1.1 | Russell 2000 vs S&P 500 |

### 8. Intelligence Layer

**Files:** `intelligence/gdelt_fetcher.py`, `intelligence/event_scorer.py`

Fetches geopolitical event data from GDELT and computes an event score that adjusts ML crash probabilities. This captures tail risks (military conflict, sanctions, political instability) that may not appear in market data yet.

### 9. Validation Layer

**Files:** `validation/regime_validator.py`, `validation/external_validator.py`

Cross-checks engine outputs against external sources:
- **Regime validation**: Confirms HMM regime with 200-day SMA, sector breadth, and rule-based fallback
- **External validation**: Compares predictions against LEI, SLOOS, AAII sentiment, NAAIM exposure, IMF GDP forecasts

### 10. Reporting

**File:** `reporting/pdf_report.py`

Generates a professional PDF report including:
- Crash probability curves and timing analysis
- S&P 500 5-year projection with confidence bands
- Scenario analysis with probability breakdown
- SHAP explanations for crash predictions
- Counterfactual ("what-if") sensitivity analysis
- Sector and stock rankings
- Backtest validation metrics
- Historical stress test results

---

## Key Design Principles

1. **ML-first**: Machine learning predictions are the primary output. Monte Carlo is secondary and serves to quantify uncertainty around ML estimates.

2. **Zero data leakage**: Walk-forward backtesting with configurable purge gaps ensures no future information leaks into training data.

3. **Configuration-driven**: All parameters live in `engine_config.yaml`. No magic numbers are hardcoded in source files.

4. **Fail-safe degradation**: Each module handles its own errors. If FRED is unavailable, the engine continues with market data only. If ML models fail to train, statistical defaults are used.

5. **Institutional anchoring**: Monte Carlo drift is anchored to consensus returns from Vanguard, BlackRock, Goldman Sachs, etc. -- not raw historical averages.

---

## Configuration Architecture

```
engine_config.yaml           .env (gitignored)
┌──────────────────┐         ┌──────────────────┐
│ data:            │         │ FRED_API_KEY=...  │
│   tickers: ...   │         │ FINNHUB_API_KEY=..│
│   fred_series: ..│         │ FMP_API_KEY=...   │
│ simulation: ...  │         └──────────────────┘
│ risk: ...        │                  │
│ scenarios: ...   │                  │
│ ml: ...          │                  │
└───────┬──────────┘                  │
        │                             │
        └──────────┬──────────────────┘
                   │
                   ▼
         config/settings.py
         ┌─────────────────┐
         │ config (dict)   │  ← YAML parameters
         │ api_keys (obj)  │  ← .env secrets
         │ PROJECT_ROOT    │  ← Auto-detected path
         └─────────────────┘
```

The `config` dict and `api_keys` object are module-level singletons loaded once at import time. All modules access them via:

```python
from finpredict.config import config, api_keys
```

---

## Data Flow Diagram

```
Yahoo Finance + FRED + Finnhub
            │
            ▼
     ┌──────────────┐
     │  Raw Data     │  S&P 500, VIX, yields, ETFs, FRED series
     └──────┬───────┘
            │
     ┌──────▼───────┐
     │  Features     │  80+ engineered features (returns, vol, macro, regime)
     └──────┬───────┘
            │
     ┌──────▼───────┐     ┌────────────┐
     │  ML Models    │────▶│  Backtest   │  Walk-forward validation
     │  (Train)      │     │  Metrics    │  (Brier, AUC, Corr)
     └──────┬───────┘     └────────────┘
            │
     ┌──────▼───────┐
     │  ML Predict   │  Crash prob (3m/6m/12m) + Expected return + Quantiles
     └──────┬───────┘
            │
     ┌──────▼───────┐
     │  Monte Carlo  │  10K paths x 5yr, conditioned on ML outputs
     └──────┬───────┘
            │
     ┌──────▼───────┐
     │  Sector +     │  Factor-based analysis, stock screening
     │  Stock        │
     └──────┬───────┘
            │
     ┌──────▼───────┐
     │  PDF Report   │  charts, tables, SHAP, counterfactuals
     └──────────────┘
```
