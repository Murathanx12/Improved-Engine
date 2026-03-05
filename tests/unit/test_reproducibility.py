"""Tests for BUG 2 fix: RNG seed reproducibility.

Verifies that simulate_paths produces identical results when given the
same seed, and different results with different seeds.
"""

import numpy as np
import pytest

from finpredict.simulation.monte_carlo import simulate_paths


def _run_sim(seed):
    """Helper to run a small simulation with a given seed."""
    return simulate_paths(
        start_price=100.0,
        historical_mu=0.08,
        historical_sigma=0.16,
        days=63,
        n_sims=50,
        crash_freq=0.07,
        risk_score=0.5,
        scenario={"drift_adj": 0.0, "vol_mult": 1.0, "crash_mult": 1.0},
        seed=seed,
    )


class TestReproducibility:
    """Verify seeded simulation is deterministic."""

    def test_same_seed_identical_output(self):
        """Two runs with the same seed produce bit-identical results."""
        result1 = _run_sim(seed=42)
        result2 = _run_sim(seed=42)
        np.testing.assert_array_equal(result1, result2)

    def test_different_seed_different_output(self):
        """Two runs with different seeds produce different results."""
        result1 = _run_sim(seed=42)
        result2 = _run_sim(seed=99)
        assert not np.array_equal(result1, result2)

    def test_none_seed_still_works(self):
        """Passing seed=None (default) should not crash."""
        result = _run_sim(seed=None)
        assert result.shape == (64, 50)  # days+1, n_sims
        assert np.all(result[0, :] == 100.0)  # starts at start_price

    def test_output_shape_with_seed(self):
        """Seeded simulation should have correct shape."""
        result = _run_sim(seed=42)
        assert result.shape == (64, 50)  # days+1, n_sims
