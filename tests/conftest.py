"""Shared pytest fixtures for all tests."""
import pytest
import numpy as np


@pytest.fixture
def rng():
    """Reproducible random number generator for tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def sample_returns(rng):
    """Generate 1000 daily returns resembling S&P 500."""
    return rng.normal(loc=0.0003, scale=0.012, size=1000)