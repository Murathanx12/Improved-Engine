"""Simulation — Monte Carlo engine, scenarios, backtesting, valuation."""

from finpredict.simulation.monte_carlo import run_monte_carlo, simulate_paths
from finpredict.simulation.scenarios import build_scenarios
from finpredict.simulation.valuation import compute_valuation_penalty
from finpredict.simulation.backtest import run_backtest
from finpredict.simulation.stress_test import run_stress_tests

__all__ = [
    "run_monte_carlo", "simulate_paths", "build_scenarios",
    "compute_valuation_penalty", "run_backtest", "run_stress_tests",
]
