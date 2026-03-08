# CLAUDE.md — Aegis Finance Engine v7

## Project

Market crash prediction engine. S&P 500 focus. ML-first: predictions drive Monte Carlo, not the other way around.

**Location:** `C:\Users\mrthn\market-prediction-engine`  
**Python:** 3.12, venv at `.venv\`  
**Config:** `engine_config.yaml` (all parameters — never hardcode in Python)  
**Keys:** `.env` (gitignored) — FRED_API_KEY, FINNHUB_API_KEY, FMP_API_KEY

## Commands

```bash
.venv\Scripts\activate
python -m finpredict          # Full pipeline → PDF report in reports/
python -m pytest tests/ -v    # 211+ tests must pass
ruff check src/               # Lint check
ruff check --fix src/         # Auto-fix lint
```

## Architecture

```
src/finpredict/
├── main.py                         # Pipeline orchestrator
├── config/settings.py              # YAML + API key loader
├── data/
│   ├── fetchers.py                 # Yahoo Finance
│   ├── fred_fetcher.py             # FRED macro (22 series)
│   ├── global_crashes.py           # 6 global indices + contagion features
│   ├── quality.py                  # Data quality validation
│   └── cache.py                    # Parquet caching
├── ml/
│   ├── features.py                 # 80+ backward-looking features
│   ├── crash_model.py              # LightGBM + Logistic crash predictor
│   ├── return_model.py             # LightGBM quantile regression
│   ├── xgboost_model.py            # XGBoost peer crash model
│   ├── sequence_model.py           # LSTM + TCN temporal ensemble
│   ├── meta_stacker.py             # Regime-aware model combiner (5 models: lgb, xgb, lstm, tcn, cox)
│   ├── survival_model.py           # Cox PH crash model (survival analysis)
│   ├── crash_timing.py             # 3-month window timing
│   ├── anomaly_detector.py         # Isolation Forest + Bayesian changepoint
│   └── prediction_logger.py        # CSV audit trail (logs/prediction_log.csv)
├── models/
│   ├── garch.py                    # GJR-GARCH(1,1)
│   ├── hmm_regimes.py              # 3-state HMM
│   ├── sectors.py                  # Factor model (2-pass: normalize then simulate)
│   └── stocks.py                   # Individual stock projections
├── simulation/
│   ├── monte_carlo.py              # Jump-diffusion MC
│   ├── backtest.py                 # Walk-forward expanding-window validation
│   ├── scenarios.py                # 8 scenario weighting
│   ├── valuation.py                # CAPE constraint
│   └── stress_test.py              # Historical crisis tests
├── evaluation/
│   ├── metrics.py                  # Brier, BSS, reliability diagrams
│   ├── comparison.py               # Baseline comparisons
│   └── brier_monitor.py            # Rolling Brier degradation monitor
├── risk/                           # scoring.py, regimes.py, crashes.py
├── validation/                     # regime_validator.py, external_validator.py
├── intelligence/                   # gdelt_fetcher.py, event_scorer.py
├── reporting/pdf_report.py         # PDF generation
└── utils/charts.py                 # Matplotlib charts
```

## Rules

### DO
- Put all parameters in `engine_config.yaml`
- Run `pytest` after every change
- Use `np.random.default_rng(seed)` for reproducibility
- Add unit tests for new prediction logic
- Handle missing libraries with `try/except ImportError` + fallback class

### DO NOT
- Use `fillna(0)` on the feature matrix — LightGBM handles NaN natively
- Subtract `val_penalty` from `ml_predicted_return` — ML features already capture valuation
- Use `np.random.seed()` (legacy API)
- Hardcode file paths — use `PROJECT_ROOT` from config

## Completed Fixes (For Reference)

- Bug 1: Lookup table override disabled (divergence_threshold: 1.0)
- Bug 2: Severity ensemble dead code removed (saves ~30% training time)
- Bug 3: Double valuation penalty removed from ML path
- Bug 4: fillna(0) removed from feature matrix
- Bug 5: Bear regime thresholds widened (neutral: -0.02, bear: -0.05)
- Bug 6: Sector MC uses 2-pass normalize-then-simulate, 2000 sims
- Bug 7: LightGBM reduced to 300 estimators, logistic expanded to 10 features
- Bug 8: Anomaly detector preserves ML signal when anomaly confirms stress
- Bug 9: Model selection margin — logistic preferred unless LGB beats by >0.01 Brier
- Bug 10: Logistic fillna(0) replaced with training medians
- Bug 11: Cox survival targets respect temporal boundaries (no data leakage)
- Bug 12: Cox fillna(0) replaced with training medians
- Bug 13: Cox survival function uses interpolation instead of nearest-neighbor
- Bug 14: Feature matrix ffill() limited to 5 rows (was unbounded)
- Bug 15: FRED publication lag shift applied before reindex (prevents forward-look)
- Bug 16: Legacy np.random.seed() replaced with np.random.default_rng()
- Bug 17: OOS prediction arrays always aligned across horizons (meta-stacker fix)
- Bug 18: GARCH mean reversion uses persistence parameter (was hardcoded 0.99/0.01)
- Bug 19: Dead code removed (features.py, monte_carlo.py)

## Healthy Output Ranges

When the engine is working correctly, the summary should show:
- Crash predictions spanning 5%-55% (not clustered at 20-25%)
- Crash prediction std > 8% (not < 5%)
- Per-model breakdown shows 5 models: LGB, XGB, LSTM, TCN, Cox
- Sector returns differentiated (20-80% range, not uniform 48-51%)
- Brier Score < 0.22 (random = 0.25, target < 0.18)
- 3m crash < 6m crash < 12m crash (monotonic)

## Commit Convention

```
fix(critical): description     # bugs that corrupt predictions
fix(high): description         # methodology issues
feat: description              # new capabilities
test: description              # new or fixed tests
chore: description             # tooling, config, cleanup
```