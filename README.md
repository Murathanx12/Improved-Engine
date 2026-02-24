# Market Prediction Engine v4.5

Crash probability analysis and 5-year market projection engine using jump-diffusion Monte Carlo simulation with institutional anchoring.

## Quick Start

```bash
# Clone and enter project
git clone https://github.com/Murathanx12/Improved-Engine.git
cd Improved-Engine

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install (editable mode)
pip install -e ".[dev]"

# Copy and fill in API keys
cp .env.example .env
# Edit .env with your FRED, Finnhub, FMP keys

# Verify setup
python test_setup.py

# Run the engine
make run
```

## Architecture

```
src/finpredict/
├── config/           # Settings from engine_config.yaml + .env
├── data/             # Market data fetching and caching (Module 1)
├── risk/             # Risk scoring, regimes, crash analysis (Modules 2-4)
├── simulation/       # Monte Carlo engine, scenarios, backtest (Modules 5-6)
├── models/           # Sector and stock analysis (Modules 7-8)
├── utils/            # Charts and visualization (Module 9)
├── reporting/        # PDF report generation (Module 10)
└── main.py           # Pipeline orchestrator
```

**Configuration split:**
- `.env` — API keys only (gitignored)
- `engine_config.yaml` — All engine parameters (safe to commit)

## Key Outputs

- **Crash probability curves** (3mo → 5yr horizons)
- **5-year S&P 500 projection** with 90% confidence interval
- **8-scenario Monte Carlo** with dynamic probability adjustment
- **Walk-forward backtest** (2000→present, zero data leakage)
- **Sector and stock analysis** with institutional benchmarking
- **Professional PDF report**

## Data Sources

| Source | What | Cost |
|--------|------|------|
| Yahoo Finance | Prices, VIX, Treasuries | Free |
| FRED | Yield curve, macro indicators | Free |
| Finnhub | Quotes, fundamentals, sentiment | Free |
| FMP | Analyst targets, financials | Free tier |
| Shiller/Yale | CAPE ratio | Free |
