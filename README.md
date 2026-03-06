# FinPredict - Market Prediction Engine v7.0

> ML-first crash probability analysis and 5-year S&P 500 projection engine using LightGBM, GJR-GARCH, HMM regime detection, and jump-diffusion Monte Carlo simulation with institutional anchoring.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Setup (Local - Visual Studio Code)](#setup-local---visual-studio-code)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Open in VS Code](#2-open-in-vs-code)
  - [3. Create a Python Virtual Environment](#3-create-a-python-virtual-environment)
  - [4. Install Dependencies](#4-install-dependencies)
  - [5. Configure API Keys](#5-configure-api-keys)
  - [6. Verify Setup](#6-verify-setup)
  - [7. Run the Engine](#7-run-the-engine)
- [Configuration](#configuration)
- [Key Outputs](#key-outputs)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Common Commands](#common-commands)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

FinPredict is an ML-first financial prediction engine that combines multiple machine learning models (LightGBM, XGBoost, LSTM, TCN) with traditional quantitative finance methods (GARCH volatility, HMM regimes, Monte Carlo simulation) to produce:

- **Crash probability estimates** at 3-month, 6-month, 1 year, and 5 year horizons
- **Expected return predictions** with uncertainty quantification (quantile regression)
- **5-year S&P 500 projections** via ML-conditioned Monte Carlo
- **Sector and individual stock analysis** using factor-based differentiation
- A **professional PDF report** summarizing all findings

The ML models are primary -- Monte Carlo simulation is secondary, serving only to quantify uncertainty around the learned predictions.

## Features

- **Multi-model ML ensemble**: LightGBM, XGBoost, LSTM, TCN with MetaStacker
- **80+ engineered features** from market data, macro indicators, and FRED time series
- **Walk-forward backtesting** (2000-present) with zero data leakage
- **GJR-GARCH** volatility modeling with leverage effect
- **Hidden Markov Model** regime detection (Bull / Neutral / Bear / Crisis)
- **8 economic scenarios** with dynamic probability adjustment
- **SHAP explainability** -- see exactly which signals drive crash predictions
- **Anomaly detection** and Bayesian changepoint analysis
- **OSINT intelligence layer** (GDELT geopolitical events)
- **External validation** cross-checking against FRED, AAII sentiment, NAAIM exposure

## Architecture

For a detailed architecture breakdown with diagrams, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

```
src/finpredict/
├── config/           # Settings loader (engine_config.yaml + .env)
├── data/             # Market data fetching, FRED, caching
├── ml/               # ML models (LightGBM, XGBoost, LSTM, TCN, MetaStacker)
├── models/           # GARCH, HMM regimes, sector/stock analysis
├── risk/             # Risk scoring, regime detection, crash identification
├── simulation/       # Monte Carlo, scenarios, backtesting, stress tests
├── intelligence/     # GDELT event scoring, OSINT layer
├── validation/       # Regime + external validation cross-checks
├── evaluation/       # Metrics and model comparison
├── utils/            # Charts and visualization helpers
├── reporting/        # PDF report generation
└── main.py           # Pipeline orchestrator (runs all modules in sequence)
```

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python**  | >= 3.11 | 3.12 recommended |
| **pip**     | latest  | Comes with Python |
| **Git**     | any     | For cloning the repo |
| **VS Code** | latest  | Recommended editor (optional) |
| **Make**    | any     | For `make` shortcuts (optional -- see below) |

> **Windows users:** `make` is not installed by default. You can either:
> - Install it via [Chocolatey](https://chocolatey.org/): `choco install make`
> - Install it via [Scoop](https://scoop.sh/): `scoop install make`
> - Or skip `make` entirely and use the raw `python` commands shown in [Common Commands](#common-commands)

You will also need **free API keys** from:

| Service | Sign Up | Required? |
|---------|---------|-----------|
| FRED    | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) | Yes |
| Finnhub | [finnhub.io/register](https://finnhub.io/register) | Yes |
| FMP     | [financialmodelingprep.com](https://site.financialmodelingprep.com) | Yes |

---

## Setup (Local - Visual Studio Code)

### 1. Clone the Repository

Open a terminal (PowerShell on Windows, Terminal on macOS/Linux):

```bash
git clone https://github.com/Murathanx12/Improved-Engine.git
cd Improved-Engine
```

### 2. Open in VS Code

```bash
code .
```

Or open VS Code manually and use **File > Open Folder** to select the `Improved-Engine` directory.

VS Code should automatically detect the workspace settings and recommend extensions. Accept the prompt to install recommended extensions.

### 3. Create a Python Virtual Environment

Open the VS Code integrated terminal (`Ctrl+`` ` or **Terminal > New Terminal**):

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> If you get an execution policy error, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

VS Code should detect the virtual environment. If prompted to select a Python interpreter, choose the one inside `.venv`.

### 4. Install Dependencies

```bash
# Install the package in editable mode (includes all dependencies)
pip install -e ".[dev]"
```

This installs all required packages listed in `pyproject.toml`, including:
- numpy, pandas, scipy, matplotlib (core scientific stack)
- yfinance, fredapi, finnhub-python (data fetchers)
- lightgbm, xgboost, torch, scikit-learn (ML models)
- arch, hmmlearn (volatility and regime models)
- reportlab (PDF generation)
- pytest, ruff (development tools)

> **Note:** PyTorch (`torch`) may take a while to install. If you have an NVIDIA GPU and want CUDA support, see [pytorch.org](https://pytorch.org/get-started/locally/) for platform-specific install commands.

### 5. Configure API Keys

```bash
# Copy the example environment file
cp .env.example .env
```

On Windows (PowerShell):
```powershell
Copy-Item .env.example .env
```

Open `.env` in VS Code and replace the placeholder values with your actual API keys:

```env
FRED_API_KEY=your_actual_fred_key_here
FINNHUB_API_KEY=your_actual_finnhub_key_here
FMP_API_KEY=your_actual_fmp_key_here
```

> `.env` is listed in `.gitignore` -- your keys will never be committed.

### 6. Verify Setup

```bash
python test_setup.py
```

This checks:
- All Python packages are importable
- `engine_config.yaml` loads correctly
- API keys are set in `.env`
- FRED, Finnhub, and Yahoo Finance connections work

If all tests pass, you'll see: `Setup complete. All systems go.`

### 7. Run the Engine

```bash
python -m finpredict.main
```

Or use the Makefile shortcut:

```bash
make run
```

The engine takes approximately 5-15 minutes depending on your hardware and network speed. It will:
1. Fetch market data from Yahoo Finance and FRED
2. Fit GARCH and HMM models
3. Train ML models via walk-forward backtesting
4. Generate Monte Carlo projections
5. Analyze sectors and stocks
6. Output a PDF report in the `reports/` directory

---

## Configuration

The engine uses two configuration files:

| File | Contains | Committed? |
|------|----------|-----------|
| `engine_config.yaml` | All engine parameters (simulation, risk, scenarios, etc.) | Yes |
| `.env` | API keys only | No (gitignored) |

Key sections in `engine_config.yaml`:

- **`data`** -- Tickers, FRED series, date ranges
- **`ml`** -- Crash base rates, ensemble settings, purge gaps
- **`simulation`** -- Monte Carlo paths, jump-diffusion parameters, mean reversion
- **`backtest`** -- Walk-forward settings (step size, lookback, forward horizon)
- **`risk`** -- Crash thresholds, indicator weights, regime detection
- **`scenarios`** -- 8 economic scenarios with base probabilities
- **`institutional_benchmarks`** -- Vanguard, BlackRock, Goldman, etc. consensus returns
- **`reporting`** -- Output directory, chart style, PDF settings

---

## Key Outputs

| Output | Description |
|--------|-------------|
| **Crash probability curves** | 3-month, 6-month, and 12-month crash probability estimates |
| **Expected return predictions** | Point estimate + [P10, P90] quantile range |
| **5-year S&P 500 projection** | ML-conditioned Monte Carlo with 90% confidence interval |
| **8-scenario Monte Carlo** | Dynamic probability adjustment based on current regime |
| **Walk-forward backtest** | 2000-present with Brier score, AUC, and return correlation |
| **Sector analysis** | Factor-based (beta, momentum, quality) differentiation |
| **Stock analysis** | Top picks from best-ranked sectors with DCF + Sharpe |
| **PDF report** | Professional report saved to `reports/` directory |

---

## Data Sources

| Source | Data | Cost | Rate Limit |
|--------|------|------|------------|
| Yahoo Finance | Prices, VIX, Treasuries, sector ETFs | Free | Generous |
| FRED | Yield curve, recession indicators, CPI, unemployment | Free | 120 req/min |
| Finnhub | Quotes, fundamentals, sentiment | Free | 60 req/min |
| FMP | Analyst targets, financials | Free tier (250/day) | 250 req/day |
| Shiller/Yale | CAPE ratio (Shiller P/E) | Free | N/A |
| GDELT | Geopolitical event data (OSINT) | Free | N/A |

---

## Project Structure

```
Improved-Engine/
├── .env.example            # API keys template (copy to .env)
├── .gitignore              # Git ignore rules
├── .vscode/                # VS Code workspace settings
│   ├── extensions.json     # Recommended extensions
│   └── settings.json       # Python, linting, formatting
├── docs/
│   └── ARCHITECTURE.md     # Detailed architecture + methodology docs
├── engine_config.yaml      # All engine parameters (safe to commit)
├── Makefile                # Shortcuts: make run, make test, etc.
├── notebooks/              # Jupyter notebooks for exploration
├── pyproject.toml          # Package metadata + dependencies
├── README.md               # This file
├── reports/                # Generated PDF reports (gitignored)
├── requirements.txt        # Flat dependency list (alternative to pyproject.toml)
├── src/
│   └── finpredict/         # Main package
│       ├── __init__.py
│       ├── __main__.py     # Entry point for python -m finpredict
│       ├── main.py         # Pipeline orchestrator
│       ├── config/         # Configuration loader
│       ├── data/           # Data fetching + caching
│       ├── ml/             # Machine learning models
│       ├── models/         # Statistical models (GARCH, HMM, sectors, stocks)
│       ├── risk/           # Risk scoring + regime detection
│       ├── simulation/     # Monte Carlo + backtesting
│       ├── intelligence/   # OSINT / GDELT layer
│       ├── validation/     # Cross-validation checks
│       ├── evaluation/     # Metrics + comparison
│       ├── utils/          # Charting helpers
│       └── reporting/      # PDF report generator
├── test_setup.py           # Setup verification script
└── tests/                  # Unit + integration tests
    ├── unit/
    └── integration/
```

---

## Common Commands

All commands assume the virtual environment is activated. You can use either the `make` shortcuts or the raw `python` commands.

| Task | With Make | Without Make (raw command) |
|------|-----------|--------------------------|
| **Run the engine** | `make run` | `python -m finpredict.main` |
| **First-time setup** | `make setup` | `pip install -e ".[dev]" && python test_setup.py` |
| **Install dependencies** | `make install` | `pip install -e ".[dev]"` |
| **Verify setup** | `make check` | `python test_setup.py` |
| **Run tests** | `make test` | `pytest tests/ -v --tb=short` |
| **Run tests + coverage** | `make coverage` | `pytest tests/ -v --cov=finpredict --cov-report=term-missing` |
| **Lint code** | `make lint` | `ruff check src/ tests/` |
| **Clean caches** | `make clean` | _(manually delete `__pycache__`, `.pytest_cache`, `data_cache/*.parquet`)_ |
| **Force data refresh** | `make refresh` | `python -c "from finpredict.data import cached_fetch_all_data; cached_fetch_all_data(force_refresh=True)"` |

> **Note:** `make run` sets `PYTHONPATH=src` automatically, so it works even without `pip install -e .`. If you run `python -m finpredict.main` directly, you must either install the package first or set `PYTHONPATH=src` yourself.

### Running Tests

```bash
# All tests
pytest tests/ -v

# Only unit tests
pytest tests/unit/ -v

# With coverage
pytest tests/ -v --cov=finpredict --cov-report=term-missing

# A specific test file
pytest tests/unit/test_engine.py -v
```

### Linting

```bash
ruff check src/ tests/
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'finpredict'`

Either install the package in editable mode:
```bash
pip install -e ".[dev]"
```

Or use `make run` which sets `PYTHONPATH` automatically (no install needed):
```bash
make run
```

### PowerShell execution policy error on Windows

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### PyTorch installation fails or is too slow

Install the CPU-only version (much smaller download):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

### FRED API key error

Make sure `.env` exists and contains a valid `FRED_API_KEY`. Get a free key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html).

### `engine_config.yaml not found`

Run commands from the project root directory (`Improved-Engine/`), not from inside `src/`.

### Cached data is stale

```bash
make refresh
# or
make clean
```

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development guidelines, coding standards, and how to submit changes.

---

## License

This project is for educational and research purposes.
