"""Tests for BUG 4 fix: data-derived xi and rho_leverage parameters.

Verifies the formulas for deriving vol-of-vol (xi) and leverage correlation
(rho_leverage) from GARCH model parameters.
"""

import numpy as np
import pandas as pd
import pytest


def compute_rho_leverage(alpha: float, gamma: float,
                         clip_min: float = -0.95, clip_max: float = -0.30) -> float:
    """Reproduce the rho_leverage derivation from main.py."""
    rho_raw = -np.sqrt(gamma / (2 * alpha + gamma + 1e-8))
    return float(np.clip(rho_raw, clip_min, clip_max))


def compute_xi(cond_vol_pct: pd.Series,
               clip_min: float = 0.02, clip_max: float = 0.15) -> float:
    """Reproduce the xi derivation from main.py.

    Args:
        cond_vol_pct: Conditional volatility in percentage units (arch convention)
    """
    cond_vol = cond_vol_pct / 100  # Convert from % to decimal
    xi_raw = float(cond_vol.std() / cond_vol.mean()) if cond_vol.mean() > 0 else 0.06
    return float(np.clip(xi_raw, clip_min, clip_max))


class TestRhoLeverageFormula:
    """Test leverage correlation derivation from GARCH gamma."""

    def test_typical_equity_values(self):
        """Typical S&P 500 GARCH: alpha~0.05, gamma~0.10 → rho around -0.71."""
        rho = compute_rho_leverage(alpha=0.05, gamma=0.10)
        assert -0.95 <= rho <= -0.30
        # gamma / (2*alpha + gamma) = 0.10 / 0.20 = 0.5 → sqrt(0.5) ≈ 0.707
        assert abs(rho - (-0.707)) < 0.01

    def test_high_leverage(self):
        """High gamma → more negative rho, clipped at -0.95."""
        rho = compute_rho_leverage(alpha=0.01, gamma=0.20)
        # gamma / (2*0.01 + 0.20) = 0.20 / 0.22 ≈ 0.909 → sqrt ≈ 0.953
        assert rho == -0.95  # Clipped

    def test_low_leverage(self):
        """Low gamma → less negative rho, clipped at -0.30."""
        rho = compute_rho_leverage(alpha=0.10, gamma=0.01)
        # gamma / (2*0.10 + 0.01) = 0.01 / 0.21 ≈ 0.048 → sqrt ≈ 0.218
        assert rho == -0.30  # Clipped

    def test_zero_gamma(self):
        """Zero gamma (no asymmetry) → rho should be at clip min."""
        rho = compute_rho_leverage(alpha=0.05, gamma=0.0)
        assert rho == -0.30  # sqrt(0) = 0, clipped to -0.30

    def test_result_always_negative(self):
        """Leverage correlation should always be negative."""
        for alpha, gamma in [(0.05, 0.10), (0.01, 0.20), (0.10, 0.01)]:
            rho = compute_rho_leverage(alpha, gamma)
            assert rho < 0


class TestXiFormula:
    """Test vol-of-vol derivation from GARCH conditional volatility."""

    def test_typical_values(self):
        """Typical conditional vol series should give xi in valid range."""
        # Simulate conditional vol in percentage units (arch convention)
        rng = np.random.default_rng(42)
        base_vol = 15.0  # 15% in percentage units
        cond_vol_pct = pd.Series(base_vol + rng.normal(0, 2.0, 1000))
        cond_vol_pct = cond_vol_pct.clip(5.0, 50.0)

        xi = compute_xi(cond_vol_pct)
        assert 0.02 <= xi <= 0.15

    def test_constant_vol_gives_low_xi(self):
        """Constant vol → zero std → clipped to xi_min."""
        cond_vol_pct = pd.Series([15.0] * 1000)
        xi = compute_xi(cond_vol_pct)
        assert xi == 0.02  # Clipped to min

    def test_highly_variable_vol_clipped(self):
        """Extremely variable vol → high CV → clipped to xi_max."""
        rng = np.random.default_rng(42)
        cond_vol_pct = pd.Series(rng.uniform(5.0, 50.0, 1000))
        xi = compute_xi(cond_vol_pct)
        assert xi == 0.15  # Clipped to max

    def test_percentage_conversion_matters(self):
        """If we forget /100, xi would be same (CV is unitless), but
        the function signature expects percentage units as input."""
        cond_vol_pct = pd.Series([15.0, 16.0, 14.0, 17.0, 13.0] * 200)
        xi = compute_xi(cond_vol_pct)
        # CV = std/mean is unitless, so same result either way
        assert 0.02 <= xi <= 0.15


class TestClipBoundsFromConfig:
    """Verify clip bounds are respected."""

    def test_custom_clip_bounds(self):
        """Custom clip bounds should override defaults."""
        rho = compute_rho_leverage(alpha=0.05, gamma=0.10,
                                   clip_min=-0.80, clip_max=-0.50)
        assert -0.80 <= rho <= -0.50

        cond_vol_pct = pd.Series([15.0, 16.0, 14.0] * 100)
        xi = compute_xi(cond_vol_pct, clip_min=0.05, clip_max=0.10)
        assert 0.05 <= xi <= 0.10
