"""
Historical Stress Test Module
==============================

Deterministic scenario analysis: applies the exact peak-to-trough drawdowns
from major historical crises to the current S&P 500 level.

This is intentionally separate from the probabilistic Monte Carlo engine.
It answers the question: "If 2008 happened again from today's price, where
would the S&P 500 trough, and how long would recovery take?"

Data sourced from Shiller/Yale, NBER, and Bloomberg.

Usage:
    from finpredict.simulation.stress_test import run_stress_tests

    results = run_stress_tests(current_price=5_700)
"""

from __future__ import annotations

# Historical crisis parameters
# peak_to_trough: maximum drawdown from peak (negative fraction)
# duration_days: calendar days from peak to trough
# recovery_days: calendar days from trough back to prior peak (approx.)
HISTORICAL_CRISES: dict[str, dict] = {
    "2008 Financial Crisis": {
        "peak_to_trough": -0.5647,
        "duration_days": 356,
        "recovery_days": 1474,
        "description": "Lehman collapse, housing bust, global credit freeze",
        "peak_year": 2007,
    },
    "2020 COVID Crash": {
        "peak_to_trough": -0.3386,
        "duration_days": 33,
        "recovery_days": 148,
        "description": "Pandemic-driven liquidity shock; fastest crash and recovery on record",
        "peak_year": 2020,
    },
    "2000 Dot-Com Bust": {
        "peak_to_trough": -0.4912,
        "duration_days": 929,
        "recovery_days": 2520,
        "description": "Tech bubble collapse; slow grinding bear market through 9/11",
        "peak_year": 2000,
    },
    "1987 Black Monday": {
        "peak_to_trough": -0.3353,
        "duration_days": 101,
        "recovery_days": 612,
        "description": "Portfolio insurance cascade; single worst trading day in history",
        "peak_year": 1987,
    },
    "1973-74 Oil Crisis": {
        "peak_to_trough": -0.4795,
        "duration_days": 630,
        "recovery_days": 2190,
        "description": "OPEC embargo, stagflation, Watergate political uncertainty",
        "peak_year": 1973,
    },
}


def run_stress_tests(current_price: float) -> dict:
    """Apply historical peak-to-trough drawdowns to the current S&P 500 price.

    Args:
        current_price: current S&P 500 index level

    Returns:
        dict mapping crisis name → {
            trough_price: float,    S&P level at trough
            drop_pct: float,        fraction drop (negative, e.g. -0.56)
            drop_points: float,     index points lost
            duration_days: int,     days from peak to trough
            recovery_days: int,     days from trough back to prior peak
            description: str,       brief narrative
            peak_year: int,         year of the prior peak
        }
    """
    results = {}
    for name, params in HISTORICAL_CRISES.items():
        drop = params["peak_to_trough"]
        trough = current_price * (1.0 + drop)
        results[name] = {
            "trough_price": round(trough, 1),
            "drop_pct": drop,
            "drop_points": round(current_price * drop, 1),
            "duration_days": params["duration_days"],
            "recovery_days": params["recovery_days"],
            "description": params["description"],
            "peak_year": params["peak_year"],
        }
    return results
