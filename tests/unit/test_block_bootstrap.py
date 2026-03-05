"""Tests for BUG 1 fix: vectorized block bootstrap residuals.

Compares the vectorized implementation against the original pure-Python
loop to ensure identical output, and benchmarks the speedup.
"""

import time
import numpy as np
import pytest


def _generate_block_bootstrap_residuals_old(
    historical_returns: np.ndarray,
    days: int,
    n_sims: int,
    block_size: int = 21,
    rng=None,
) -> np.ndarray:
    """Original (pre-fix) implementation with Python inner loop."""
    if rng is None:
        rng = np.random.default_rng()

    n_hist = len(historical_returns)
    max_start = n_hist - block_size
    if max_start < 1:
        return rng.standard_normal(size=(days, n_sims))

    mu = historical_returns.mean()
    sigma = historical_returns.std()
    if sigma < 1e-10:
        return rng.standard_normal(size=(days, n_sims))
    standardized = (historical_returns - mu) / sigma

    n_blocks_needed = (days + block_size - 1) // block_size
    residuals = np.zeros((days, n_sims))

    starts = rng.integers(0, max_start, size=(n_blocks_needed, n_sims))
    for b in range(n_blocks_needed):
        row_start = b * block_size
        row_end = min(row_start + block_size, days)
        actual_len = row_end - row_start
        for sim in range(n_sims):
            residuals[row_start:row_end, sim] = standardized[
                starts[b, sim]:starts[b, sim] + actual_len
            ]

    return residuals


def _generate_block_bootstrap_residuals_new(
    historical_returns: np.ndarray,
    days: int,
    n_sims: int,
    block_size: int = 21,
    rng=None,
) -> np.ndarray:
    """Vectorized implementation (the fix)."""
    if rng is None:
        rng = np.random.default_rng()

    n_hist = len(historical_returns)
    max_start = n_hist - block_size
    if max_start < 1:
        return rng.standard_normal(size=(days, n_sims))

    mu = historical_returns.mean()
    sigma = historical_returns.std()
    if sigma < 1e-10:
        return rng.standard_normal(size=(days, n_sims))
    standardized = (historical_returns - mu) / sigma

    n_blocks_needed = (days + block_size - 1) // block_size
    residuals = np.zeros((days, n_sims))

    starts = rng.integers(0, max_start, size=(n_blocks_needed, n_sims))
    for b in range(n_blocks_needed):
        row_start = b * block_size
        row_end = min(row_start + block_size, days)
        actual_len = row_end - row_start
        idx = starts[b, :, np.newaxis] + np.arange(actual_len)
        residuals[row_start:row_end, :] = standardized[idx].T

    return residuals


class TestBlockBootstrapCorrectness:
    """Verify vectorized output matches original on identical inputs."""

    def test_outputs_match_exactly(self):
        """Old and new implementations produce identical results."""
        rng_old = np.random.default_rng(seed=42)
        rng_new = np.random.default_rng(seed=42)

        hist = np.random.default_rng(0).normal(0.0003, 0.012, size=2520)

        old = _generate_block_bootstrap_residuals_old(
            hist, days=252, n_sims=500, block_size=21, rng=rng_old,
        )
        new = _generate_block_bootstrap_residuals_new(
            hist, days=252, n_sims=500, block_size=21, rng=rng_new,
        )

        np.testing.assert_allclose(old, new, atol=1e-10)

    def test_outputs_match_non_divisible_days(self):
        """Correctness when days is not a multiple of block_size."""
        rng_old = np.random.default_rng(seed=99)
        rng_new = np.random.default_rng(seed=99)

        hist = np.random.default_rng(1).normal(0.0003, 0.012, size=2520)

        old = _generate_block_bootstrap_residuals_old(
            hist, days=260, n_sims=200, block_size=21, rng=rng_old,
        )
        new = _generate_block_bootstrap_residuals_new(
            hist, days=260, n_sims=200, block_size=21, rng=rng_new,
        )

        np.testing.assert_allclose(old, new, atol=1e-10)

    def test_output_shape(self):
        """Verify output shape is (days, n_sims)."""
        rng = np.random.default_rng(seed=42)
        hist = np.random.default_rng(0).normal(0, 0.01, size=1000)
        result = _generate_block_bootstrap_residuals_new(
            hist, days=252, n_sims=100, block_size=21, rng=rng,
        )
        assert result.shape == (252, 100)

    def test_standardized_output(self):
        """Residuals should be approximately zero-mean, unit-variance."""
        rng = np.random.default_rng(seed=42)
        hist = np.random.default_rng(0).normal(0.0003, 0.012, size=5000)
        result = _generate_block_bootstrap_residuals_new(
            hist, days=1000, n_sims=500, block_size=21, rng=rng,
        )
        assert abs(result.mean()) < 0.05
        assert abs(result.std() - 1.0) < 0.1


class TestBlockBootstrapBenchmark:
    """Benchmark old vs new implementation."""

    def test_vectorized_is_faster(self):
        """New implementation should be significantly faster."""
        hist = np.random.default_rng(0).normal(0.0003, 0.012, size=2520)

        # Warm up
        rng = np.random.default_rng(seed=0)
        _generate_block_bootstrap_residuals_new(
            hist, days=252, n_sims=100, block_size=21, rng=rng,
        )

        # Benchmark old
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed=42)
        _generate_block_bootstrap_residuals_old(
            hist, days=252, n_sims=1000, block_size=21, rng=rng,
        )
        old_time = time.perf_counter() - t0

        # Benchmark new
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed=42)
        _generate_block_bootstrap_residuals_new(
            hist, days=252, n_sims=1000, block_size=21, rng=rng,
        )
        new_time = time.perf_counter() - t0

        speedup = old_time / new_time if new_time > 0 else float("inf")
        print(f"\n  [BENCHMARK] Block bootstrap (n_sims=1000, days=252):")
        print(f"    Old (Python loop): {old_time:.4f}s")
        print(f"    New (vectorized):  {new_time:.4f}s")
        print(f"    Speedup:           {speedup:.1f}x")

        # Should be at least 3x faster (typically 5-200x depending on environment)
        assert speedup > 3, f"Expected >3x speedup, got {speedup:.1f}x"
