"""
Module 8: Sector Analysis
==========================

Runs sector-level Monte Carlo projections for the 11 GICS sector ETFs.
Returns are capped at config max_5y_return for realism.

Usage:
    from finpredict.models.sectors import analyze_sectors

    sector_results = analyze_sectors(data, sector_data, forecast_days)
"""

import numpy as np
import pandas as pd

from finpredict.config import config
from finpredict.simulation.monte_carlo import simulate_paths


def analyze_sectors(
    data: pd.DataFrame,
    sector_data: dict[str, pd.Series],
    forecast_days: int,
) -> dict:
    """
    Module 8 Entry Point: Run sector-level projections.

    Args:
        data: Master DataFrame (for risk-free rate extraction)
        sector_data: dict of sector name → price series
        forecast_days: Simulation horizon in trading days

    Returns:
        dict of sector name → {current, expected_return, median_return, volatility, sharpe}
    """
    print("[MODULE 8] Analyzing sector performance...")

    max_5y_return = config["simulation"]["max_5y_return"]

    # Risk-free rate from data
    if "T3M" in data.columns:
        rf = float(data["T3M"].dropna().iloc[-1])
        rf = max(0.0, min(rf, 0.10))
    else:
        rf = 0.04

    results = {}

    for name, series in sector_data.items():
        series = series.dropna()
        if len(series) < 504:
            continue

        returns = series.pct_change().dropna()
        log_returns = np.log(1 + returns)
        mu = log_returns.mean() * 252
        sigma = min(returns.std() * np.sqrt(252), 0.80)
        current = float(series.iloc[-1])

        # Quick Monte Carlo (2000 paths)
        paths = simulate_paths(current, mu, sigma, forecast_days, 2000)

        # Apply return cap
        final_prices = paths[-1]
        max_price = current * (1 + max_5y_return)
        final_prices = np.minimum(final_prices, max_price)

        exp_ret = float(np.mean(final_prices) / current - 1) * 100
        med_ret = float(np.median(final_prices) / current - 1) * 100

        results[name] = {
            "current": current,
            "expected_return": exp_ret,
            "median_return": med_ret,
            "volatility": sigma * 100,
            "sharpe": (mu - rf) / sigma if sigma > 0 else 0,
        }
        print(f"  [OK] {name}: {exp_ret:+.1f}% expected")

    print()
    return results
