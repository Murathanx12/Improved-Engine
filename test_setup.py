"""
Setup Verification Script
==========================

Verifies that:
    1. All packages import correctly
    2. engine_config.yaml loads properly
    3. .env API keys are accessible
    4. API connections work (FRED, Finnhub, yfinance)
    5. Modular package structure resolves correctly

Usage:
    python test_setup.py
"""

import sys


def test_packages():
    print("Testing package imports...")
    import numpy, pandas, yfinance, fredapi, finnhub
    import scipy, matplotlib, hmmlearn, arch, sklearn
    import pydantic, yaml, tqdm, reportlab
    print("  [OK] All core packages imported")


def test_config():
    print("\nTesting config system...")
    from finpredict.config import config, api_keys, PROJECT_ROOT

    assert "data" in config, "Missing 'data' section in engine_config.yaml"
    assert "simulation" in config, "Missing 'simulation' section"
    assert "risk" in config, "Missing 'risk' section"
    assert "scenarios" in config, "Missing 'scenarios' section"
    print(f"  [OK] engine_config.yaml loaded ({len(config)} sections)")
    print(f"  [OK] Project root: {PROJECT_ROOT}")

    assert api_keys.has("fred"), "FRED_API_KEY not set in .env"
    assert api_keys.has("finnhub"), "FINNHUB_API_KEY not set in .env"
    print(f"  [OK] API keys loaded (FRED: {'set' if api_keys.has('fred') else 'MISSING'}, "
          f"Finnhub: {'set' if api_keys.has('finnhub') else 'MISSING'}, "
          f"FMP: {'set' if api_keys.has('fmp') else 'MISSING'})")


def test_config_values():
    print("\nTesting config values...")
    from finpredict.config import config, get_institutional_return, get_forecast_days

    sim = config["simulation"]
    assert sim["num_simulations"] > 0, "num_simulations must be positive"
    assert sim["forecast_years"] > 0, "forecast_years must be positive"
    print(f"  [OK] Simulations: {sim['num_simulations']:,}")
    print(f"  [OK] Forecast: {sim['forecast_years']} years")

    inst_ret = get_institutional_return()
    assert 0.01 < inst_ret < 0.20, f"Institutional return out of range: {inst_ret}"
    print(f"  [OK] Institutional consensus return: {inst_ret*100:.1f}%")

    forecast_days = get_forecast_days()
    assert forecast_days > 0
    print(f"  [OK] Forecast days: {forecast_days}")


def test_modular_imports():
    print("\nTesting modular imports...")
    from finpredict.data import fetch_all_data, fetch_safe, cached_fetch_all_data
    from finpredict.risk import build_risk_score, detect_regimes, identify_crashes
    from finpredict.simulation import run_monte_carlo, compute_valuation_penalty
    from finpredict.simulation.backtest import run_backtest
    from finpredict.simulation.scenarios import build_scenarios
    from finpredict.models import analyze_sectors, analyze_stocks, select_stocks_from_sectors
    from finpredict.utils import fig_to_image, chart_backtest, chart_projection
    from finpredict.reporting import generate_report
    print("  [OK] All modular imports resolved")


def test_fred_api():
    print("\nTesting FRED API...")
    import fredapi
    from finpredict.config import api_keys

    fred = fredapi.Fred(api_key=api_keys.fred)
    vix = fred.get_series("VIXCLS", limit=5)
    print(f"  [OK] VIX: {vix.iloc[-1]:.2f}")
    spread = fred.get_series("T10Y3M", limit=5)
    print(f"  [OK] Yield Spread: {spread.iloc[-1]:.2f}%")
    sahm = fred.get_series("SAHMREALTIME", limit=5)
    print(f"  [OK] Sahm Rule: {sahm.iloc[-1]:.2f}")


def test_finnhub_api():
    print("\nTesting Finnhub API...")
    import finnhub
    from finpredict.config import api_keys

    client = finnhub.Client(api_key=api_keys.finnhub)
    quote = client.quote("SPY")
    print(f"  [OK] SPY: ${quote['c']:.2f}")


def test_yfinance():
    print("\nTesting yfinance...")
    import yfinance as yf

    spy = yf.download("SPY", period="5d", progress=False)
    close = spy["Close"].iloc[-1]
    if hasattr(close, "iloc"):
        close = close.iloc[0]
    print(f"  [OK] SPY last close: ${float(close):.2f}")


def main():
    tests = [
        ("Packages", test_packages),
        ("Config System", test_config),
        ("Config Values", test_config_values),
        ("Modular Imports", test_modular_imports),
        ("FRED API", test_fred_api),
        ("Finnhub API", test_finnhub_api),
        ("yfinance", test_yfinance),
    ]

    failed = []
    for name, test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed.append(name)

    print("\n" + "=" * 60)
    if failed:
        print(f"  SETUP INCOMPLETE — {len(failed)} test(s) failed: {', '.join(failed)}")
        print("=" * 60)
        sys.exit(1)
    else:
        print("  ✅ Setup complete. All systems go.")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
