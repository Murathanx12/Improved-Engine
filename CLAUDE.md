# CLAUDE.md — Aegis Finance Engine v7

## Project Overview

This is the **Aegis Finance Engine v7**, a market prediction engine that uses ML-first crash probability prediction, Monte Carlo simulation, and multi-factor analysis to forecast S&P 500 behavior over 1-month to 5-year horizons.

**Primary goal:** Predict market crashes (≥20% drawdowns) with meaningful discrimination.  
**Secondary goal:** Market movement forecasting, sector and individual stock analysis.  
**Success criteria:** Brier score < 0.18, BSS vs climatology > 0.10, crash prediction range spanning 5%–60%.

**Location:** `C:\Users\mrthn\market-prediction-engine`  
**Platform:** Windows 11, Python 3.12, virtual environment at `.venv\`  
**Activation:** `.venv\Scripts\activate`

## How to Run

```bash
# Activate environment
.venv\Scripts\activate

# Run full engine pipeline (generates PDF report in reports/)
python -m finpredict

# Run tests
python -m pytest tests/ -v

# Run tests with coverage
python -m pytest tests/ --cov=src/finpredict --cov-report=term-missing

# Lint
ruff check src/

# Auto-fix lint issues
ruff check --fix src/
```

## Architecture

```
src/finpredict/
├── main.py                    # Pipeline orchestrator — start here
├── config/settings.py         # YAML loader + API key manager
├── data/
│   ├── fetchers.py            # Yahoo Finance data acquisition
│   ├── fred_fetcher.py        # FRED macro data (22 series)
│   ├── alternative_fetchers.py # AAII, NAAIM, IMF, Fed futures
│   └── cache.py               # Parquet-based data caching
├── ml/
│   ├── features.py            # 80+ backward-looking ML features
│   ├── crash_model.py         # LightGBM + Logistic crash predictor
│   ├── return_model.py        # LightGBM quantile regression
│   ├── xgboost_model.py       # XGBoost peer crash model
│   ├── sequence_model.py      # LSTM + TCN temporal ensemble
│   ├── meta_stacker.py        # Regime-aware model combiner
│   ├── crash_timing.py        # 3-month window crash timing
│   ├── anomaly_detector.py    # Isolation Forest + Bayesian changepoint
│   └── prediction_logger.py   # CSV prediction snapshots (logs/prediction_log.csv)
├── models/
│   ├── garch.py               # GJR-GARCH(1,1) volatility
│   ├── hmm_regimes.py         # 3-state HMM regime detection
│   ├── sectors.py             # Factor-based sector analysis
│   └── stocks.py              # Individual stock projections
├── simulation/
│   ├── monte_carlo.py         # Jump-diffusion MC with O-U vol dynamics
│   ├── backtest.py            # Walk-forward expanding-window validation
│   ├── scenarios.py           # Dynamic scenario probability weighting
│   ├── valuation.py           # CAPE/trend valuation constraint
│   └── stress_test.py         # Historical crisis stress tests
├── risk/
│   ├── scoring.py             # 9-factor composite risk score
│   ├── regimes.py             # Rule-based regime detection (fallback)
│   └── crashes.py             # Historical crash identification
├── evaluation/
│   ├── metrics.py             # Brier, BSS, reliability diagrams, ECE
│   └── comparison.py          # Baseline model comparisons
├── validation/
│   ├── regime_validator.py    # Cross-check regime labels
│   └── external_validator.py  # External data source validation
├── intelligence/
│   ├── gdelt_fetcher.py       # GDELT geopolitical event data
│   └── event_scorer.py        # Event-driven risk scoring
├── reporting/pdf_report.py    # Multi-page PDF generation
└── utils/charts.py            # Matplotlib chart generation
```

## Configuration

**All parameters live in `engine_config.yaml` — NEVER hardcode values in Python.**

- Simulation: forecast_years, num_simulations, jump parameters, volatility bounds
- ML: purge gaps, temporal decay, model selection, lookup table blend settings
- Risk: crash threshold (20%), indicator weights, regime thresholds
- Scenarios: 8 scenario definitions with returns, volatilities, crash multipliers
- API keys: `.env` file (gitignored) — FRED_API_KEY, FINNHUB_API_KEY, FMP_API_KEY

## Key Principles

1. **ML predictions are PRIMARY, Monte Carlo is SECONDARY.** ML models learn from 35 years of data. MC quantifies uncertainty around ML predictions.
2. **No hardcoded parameters.** Everything is data-driven or config-driven. If you see a magic number, extract it to engine_config.yaml.
3. **No future data leakage.** All features must be strictly backward-looking. Walk-forward backtest uses expanding windows with purge gaps.
4. **Graceful degradation.** If any data source, model, or library fails, the engine must continue with fallbacks. Never crash on missing data.
5. **LightGBM handles NaN natively.** Do NOT fill NaN with 0 in the feature matrix. Only replace inf with NaN, then ffill. Leave remaining NaN for models to handle.
6. **Simple models for rare events.** Crash prediction has ~7 positive examples in 35 years. Logistic regression with 10 features outperforms LightGBM with 80+ features on this problem. Prefer simple models unless complex ones demonstrably beat them by >0.01 Brier.

## Critical Rules When Modifying Code

### DO
- Read engine_config.yaml before modifying any parameter
- Run `python -m pytest tests/ -v` after every change
- Use `ruff check src/` to catch issues before committing
- Write descriptive commit messages: `fix(critical): describe what and why`
- Add unit tests for any new prediction logic
- Use `np.random.default_rng(seed)` for reproducibility (not `np.random.seed()`)
- Handle missing optional dependencies with try/except ImportError patterns

### DO NOT
- Use `fillna(0)` on the feature matrix — LightGBM handles NaN
- Subtract val_penalty from ML predictions — ML already captures valuation via features
- Add lookup table overrides that replace ML signal — the lookup blend is disabled (divergence_threshold=1.0 in config)
- Train severity ensemble models without using them in predict_proba()
- Label markets with -2% to -5% annual returns as "Bear" — that's Neutral (bear threshold is -0.05)
- Use `np.random.seed()` (legacy API) — use `np.random.default_rng(seed)` instead
- Hardcode paths — use `PROJECT_ROOT` from config/settings.py
- Break the fallback chain — every external dependency must have a graceful fallback

## Known Bugs to Fix (Priority Order)

### CRITICAL — Fix Before Any Other Work

**Bug 1 — Lookup Table Override (FIXED via config)**
- File: `engine_config.yaml` → `ml.lookup_table_blend.divergence_threshold`
- Must be `1.0` (disabled). If set to `0.15`, it destroys ML predictions by blending with crude heuristic.

**Bug 2 — Severity Ensemble Dead Code**
- File: `ml/crash_model.py`
- `_train_severity_ensemble()` trains models at 10%/15%/20% thresholds but `predict_proba()` never reads from `self.severity_models`. Either implement the blend or remove the dead code.

**Bug 3 — Double Valuation Penalty**
- File: `simulation/monte_carlo.py`, `run_monte_carlo()`
- Line: `base_annual_return = ml_predicted_return - val_penalty`
- ML features already capture valuation (sma_200d_dev, dist_52w_high, erp). Subtracting val_penalty again double-counts. When ml_predicted_return is not None, use it directly without val_penalty.

**Bug 4 — fillna(0) Corrupts Features**
- File: `ml/features.py`, `build_feature_matrix()`, final cleanup
- Line: `df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)`
- Remove `.fillna(0)`. LightGBM handles NaN natively. Zero-filling binary indicators and credit spreads corrupts the signal.

**Bug 5 — Bear Regime Threshold Too Aggressive**
- File: `engine_config.yaml` → `risk.regimes`
- `bear_return_threshold` should be `-0.05` (not `-0.10`), `neutral_return_threshold` should be `-0.02` (not `0.00`). Otherwise flat markets trigger false bearish tilts.

### HIGH — Fix After Critical Bugs

**Bug 6 — Sector MC Inconsistency**
- File: `models/sectors.py`
- `_normalize_to_index()` runs AFTER per-sector MC. Path statistics don't match normalized returns. Run normalization first, then MC with normalized drift. Increase n_sims from 500 to 2000.

**Bug 7 — LightGBM Over-Parameterized for Crash Prediction**
- File: `ml/crash_model.py`
- 800 estimators + 80 features for 7 crash events is over-parameterized. Make logistic regression the default unless LightGBM beats it by >0.01 Brier. Expand LOGISTIC_FEATURES to 10 domain-motivated indicators.

**Bug 8 — Anomaly Adjustment Direction**
- File: `main.py`, anomaly detection block (~line 260)
- Adjustment always shrinks toward base rate regardless of direction. If ML predicts high crash prob correctly during stress, anomaly detector dampens it. Only shrink when anomaly direction is ambiguous.

## Testing

193 tests across 11 test files. All must pass before committing.

Key test files:
- `test_engine.py` — Config, risk, regimes, crashes, simulation, GARCH, HMM, FRED
- `test_features.py` — Feature engineering, FRED features, dynamic thresholds
- `test_evaluation.py` — Brier score, BSS, baselines, reliability diagrams
- `test_model_selection.py` — LGB vs logistic selection, meta-stacker, purge gaps
- `test_sequence_model.py` — XGBoost, LSTM, TCN, CrashTiming, MetaStacker
- `test_validation.py` — Regime validation, external validation, composite confidence

When adding new modules, add corresponding tests in `tests/unit/`.

## Pipeline Flow (main.py)

```
1. Fetch market data (Yahoo Finance + cache) → SP500, VIX, Treasuries, sectors
2. Fetch FRED macro data → 22 time series (yield spread, unemployment, CPI, etc.)
3. Fit GJR-GARCH → conditional volatility, persistence, leverage params
4. Fit HMM → 3-state regime detection (Bull/Bear/Crisis with probabilities)
5. Compute risk score → 9-factor composite z-score
6. Identify historical crashes → crash frequency baseline
7. Compute valuation penalty → CAPE or trend deviation
8. Walk-forward ML backtest → train/evaluate crash + return models
9. Current ML predictions → crash prob (3m/6m/12m), expected returns, SHAP
10. Anomaly detection → Isolation Forest + Bayesian changepoint
11. OSINT intelligence → GDELT event scoring + crash prob adjustment
12. Validation layer → regime cross-check, external data agreement
13. ML-conditioned Monte Carlo → 10,000 paths, scenario-weighted
14. Sector analysis → factor model (CAPM + momentum + mean reversion)
15. Stock analysis → top stocks from top sectors
16. PDF report generation
```

## Roadmap (What to Build Next)

### Phase 2 — ML Model Improvement (COMPLETED)
- ~~Make logistic regression primary crash model~~ (done: 10 features, auto-selection, n_estimators=300)
- ~~Add prediction logging (CSV with feature snapshots)~~ (done: ml/prediction_logger.py → logs/prediction_log.csv)
- ~~Fix anomaly detector direction~~ (done: preserves ML signal when anomaly confirms stress)

### Phase 3 — New Capabilities
- Implement survival/hazard crash model (Cox PH via lifelines)
- Add cross-market crash dataset (FTSE, Nikkei, DAX, HSI, STOXX)
- Add data quality validation layer

### Phase 4 — Production Hardening
- Add core unit tests for predict_proba, feature matrix, MC
- Add pre-commit hooks (ruff + pytest)
- Add rolling Brier score monitor for degradation detection

### Phase 5 — Advanced Features
- LLM news integration pipeline
- Portfolio construction module (crash prob → position sizing)
- Signal decay analysis and prediction half-life tracking

## Dependencies

Core: numpy, pandas, scipy, matplotlib, yfinance, fredapi, hmmlearn, arch, scikit-learn, lightgbm, xgboost, torch, shap, reportlab
Dev: pytest, pytest-cov, ruff, pre-commit, lifelines
Config: pyyaml, python-dotenv, pydantic

## Commit Message Convention

```
fix(critical): description     — for bugs that corrupt predictions
fix(high): description         — for methodology issues
fix(medium): description       — for design improvements
feat: description              — for new capabilities
test: description              — for new or fixed tests
chore: description             — for tooling, config, cleanup
```
