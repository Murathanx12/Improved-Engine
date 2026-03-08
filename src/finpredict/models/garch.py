"""
GJR-GARCH Volatility Model
============================

Replaces constant volatility (σ) with time-varying conditional volatility.

GJR-GARCH adds an asymmetric leverage term: negative returns boost volatility
more than positive returns — exactly the dynamic observed during selloffs.
For S&P 500, this leverage effect is the strongest across all asset classes.

Uses skewed Student-t distribution for innovations to capture fat tails
and negative skew in equity returns.

Usage:
    from finpredict.models.garch import fit_garch, forecast_volatility

    garch_result = fit_garch(returns)
    vol_path = forecast_volatility(garch_result, horizon=252, n_sims=1000)
"""

import numpy as np
import pandas as pd
from typing import NamedTuple


class GARCHResult(NamedTuple):
    """Results from fitting a GJR-GARCH model."""

    omega: float  # Constant in variance equation
    alpha: float  # ARCH effect (reaction to shocks)
    gamma: float  # Leverage effect (asymmetry)
    beta: float  # GARCH effect (persistence)
    current_vol: float  # Current annualized volatility
    nu: float  # Degrees of freedom (Student-t)
    model_fit: object  # The full arch model result (for forecasting)
    success: bool  # Whether fit succeeded


def fit_garch(
    returns: pd.Series,
    min_obs: int = 500,
) -> GARCHResult:
    """
    Fit a GJR-GARCH(1,1) model with skewed Student-t innovations.

    Args:
        returns: Daily return series (decimal, not percentage)
        min_obs: Minimum observations required

    Returns:
        GARCHResult with fitted parameters and current volatility
    """
    try:
        from arch import arch_model
    except ImportError:
        print("  [WARN] arch library not available, falling back to constant vol")
        return _fallback_result(returns)

    clean = returns.dropna()
    if len(clean) < min_obs:
        print(f"  [WARN] Only {len(clean)} returns, need {min_obs} for GARCH")
        return _fallback_result(returns)

    try:
        # Scale to percentage for numerical stability (arch convention)
        scaled = clean * 100

        # GJR-GARCH(1,1) with skewed Student-t innovations
        # p=1: one GARCH lag, o=1: one leverage lag, q=1: one ARCH lag
        am = arch_model(
            scaled,
            vol="Garch",
            p=1,
            o=1,
            q=1,
            dist="skewt",
            mean="Constant",
        )
        res = am.fit(disp="off", show_warning=False)

        omega = float(res.params.get("omega", 0.01))
        alpha = float(res.params.get("alpha[1]", 0.05))
        gamma = float(res.params.get("gamma[1]", 0.05))
        beta = float(res.params.get("beta[1]", 0.90))
        nu = float(res.params.get("nu", 8.0))

        # Current conditional volatility (annualized)
        cond_vol = float(res.conditional_volatility.iloc[-1])
        current_vol_annual = cond_vol / 100 * np.sqrt(252)

        # Sanity check
        if not (0.05 < current_vol_annual < 1.5):
            print(f"  [WARN] GARCH vol {current_vol_annual:.2f} out of range, using fallback")
            return _fallback_result(returns)

        print("  [GARCH] Fitted GJR-GARCH(1,1,1)")
        print(f"  [GARCH] α={alpha:.4f}, γ={gamma:.4f} (leverage), β={beta:.4f}")
        print(f"  [GARCH] Current conditional vol: {current_vol_annual*100:.1f}% annualized")
        print(f"  [GARCH] Student-t df: {nu:.1f}")

        return GARCHResult(
            omega=omega,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            current_vol=current_vol_annual,
            nu=nu,
            model_fit=res,
            success=True,
        )

    except Exception as e:
        print(f"  [WARN] GARCH fit failed: {e}, using constant vol")
        return _fallback_result(returns)


def forecast_volatility(
    garch: GARCHResult,
    horizon: int = 252,
    n_sims: int = 1000,
) -> np.ndarray:
    """
    Simulate forward volatility paths from fitted GARCH model.

    Args:
        garch: Fitted GARCH result
        horizon: Days to forecast
        n_sims: Number of volatility paths

    Returns:
        np.ndarray of shape (horizon, n_sims) — daily volatilities (decimal)
    """
    if not garch.success or garch.model_fit is None:
        # Return constant vol paths
        return np.full((horizon, n_sims), garch.current_vol / np.sqrt(252))

    try:
        # Use arch's simulation forecasting
        res = garch.model_fit
        fcast = res.forecast(
            horizon=min(horizon, 252),  # arch has horizon limits
            method="simulation",
            simulations=n_sims,
        )
        # variance is in percentage-squared units
        var_paths = fcast.simulations.variances.values  # shape: (1, horizon, n_sims)
        if var_paths.ndim == 3:
            var_paths = var_paths[0]  # Remove singleton first dim

        # Convert from % variance to decimal daily vol
        vol_paths = np.sqrt(var_paths) / 100

        # If we need more than arch can forecast, extend with persistence
        if horizon > vol_paths.shape[0]:
            extended = np.zeros((horizon, n_sims))
            extended[: vol_paths.shape[0]] = vol_paths
            # Long-run variance for extension
            persistence = garch.alpha + garch.gamma / 2 + garch.beta
            lr_var = garch.omega / (1 - persistence) if persistence < 1 else garch.omega * 100
            lr_vol = np.sqrt(lr_var) / 100
            for t in range(vol_paths.shape[0], horizon):
                # Mean-revert toward long-run vol
                extended[t] = extended[t - 1] * 0.99 + lr_vol * 0.01
            vol_paths = extended

        return vol_paths

    except Exception:
        return np.full((horizon, n_sims), garch.current_vol / np.sqrt(252))


def _fallback_result(returns: pd.Series) -> GARCHResult:
    """Create a fallback result using simple historical volatility."""
    vol = float(returns.dropna().std() * np.sqrt(252))
    return GARCHResult(
        omega=0.01,
        alpha=0.05,
        gamma=0.05,
        beta=0.90,
        current_vol=vol,
        nu=8.0,
        model_fit=None,
        success=False,
    )
