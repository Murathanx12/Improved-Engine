"""Shared pytest fixtures for all tests."""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def rng():
    """Reproducible random number generator for tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def sample_returns(rng):
    """Generate 1000 daily returns resembling S&P 500."""
    return rng.normal(loc=0.0003, scale=0.012, size=1000)


@pytest.fixture
def sample_market_data(rng):
    """Generate a minimal market DataFrame for testing modules."""
    n = 1000
    dates = pd.bdate_range("2020-01-01", periods=n)

    # Simulate S&P 500 price series
    returns = rng.normal(loc=0.0003, scale=0.012, size=n)
    prices = 3000 * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        "SP500": prices,
        "VIX": rng.normal(20, 5, n).clip(10, 80),
        "T10Y": rng.normal(0.035, 0.005, n).clip(0.01, 0.08),
        "T3M": rng.normal(0.02, 0.003, n).clip(0.001, 0.06),
        "T30Y": rng.normal(0.04, 0.005, n).clip(0.015, 0.08),
        "HYG": 80 + rng.normal(0, 2, n).cumsum() * 0.01,
        "LQD": 110 + rng.normal(0, 1, n).cumsum() * 0.01,
        "Gold": 1800 + rng.normal(0, 10, n).cumsum() * 0.1,
        "NASDAQ": 10000 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n))),
        "Russell": 1800 * np.exp(np.cumsum(rng.normal(0.0002, 0.014, n))),
    }, index=dates)

    return data
