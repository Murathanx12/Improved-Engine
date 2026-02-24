"""
Module 5: Monte Carlo Simulation Engine
=========================================

Generates thousands of possible future price paths using jump-diffusion.

How the jump-diffusion process works:
    At each time step t:
        S(t) = S(t-1) * exp(drift + diffusion) * (1 + jump)

    Where:
        drift     = (μ - 0.5σ²) * dt
        diffusion = σ * √dt * ε    (ε ~ Student-t with df=8 for fat tails)
        jump      = J * Bernoulli(p)  (J ~ Normal(-0.10, 0.05))

Why jump-diffusion (not plain GBM):
    Standard GBM assumes normally distributed returns. In reality, markets
    have FAT TAILS — extreme moves happen far more often than normal predicts.
    The jump component adds occasional large negative shocks.

Institutional anchoring:
    Expected return is calibrated to consensus of major firms (Vanguard,
    Schwab, BlackRock, etc.) rather than raw historical data.

Usage:
    from finpredict.simulation.monte_carlo import run_monte_carlo

    mc_results = run_monte_carlo(current_price, regime, risk_score, ...)
"""

import numpy as np
from scipy import stats

from finpredict.config import config, get_forecast_days
from finpredict.simulation.scenarios import build_scenarios
from finpredict.simulation.valuation import compute_valuation_penalty


# ── Path Simulation ──────────────────────────────────────────────────────────────────

def simulate_paths(
    start_price: float,
    annual_return: float,
    annual_vol: float,
    days: int,
    n_sims: int,
    crash_rate: float | None = None,
    risk_level: float = 0.0,
    scenario_params: dict | None = None,
) -> np.ndarray:
    """
    Jump-diffusion Monte Carlo path simulation.

    Args:
        start_price: Starting price level
        annual_return: Expected annual drift (geometric/log)
        annual_vol: Annual volatility (σ)
        days: Trading days to simulate
        n_sims: Number of paths
        crash_rate: Annual crash rate from historical data
        risk_level: Current composite risk score
        scenario_params: Dict with 'crash_mult' etc.

    Returns:
        np.ndarray of shape (days+1, n_sims)
    """
    sim_cfg = config["simulation"]
    jd = sim_cfg["jump_diffusion"]

    params = scenario_params or {}
    mu = annual_return + params.get("drift_adj", 0)
    sigma = annual_vol * params.get("vol_mult", 1.0)
    crash_mult = params.get("crash_mult", 1.0)

    trading_days = sim_cfg["trading_days_per_year"]
    dt = 1.0 / trading_days

    # Risk-adjusted crash rate
    base_jump_rate = jd["annual_rate"]
    if crash_rate is not None:
        historical_sudden = crash_rate * 0.30  # ~30% of crashes are sudden
        base_jump_rate = max(base_jump_rate, historical_sudden)

    # Bidirectional risk scaling: low risk reduces, high risk increases
    risk_factor = 1.0 + 0.20 * risk_level
    risk_factor = max(0.5, min(2.0, risk_factor))
    adj_jump_rate = min(0.25, base_jump_rate * crash_mult * risk_factor)
    daily_jump_prob = adj_jump_rate / trading_days

    df_param = jd["t_degrees_of_freedom"]

    # Pre-generate all random variables (vectorized)
    t_shocks = stats.t.rvs(df=df_param, size=(days, n_sims))
    t_shocks /= np.sqrt(df_param / (df_param - 2))  # Normalize variance to 1

    jump_mask = np.random.random((days, n_sims)) < daily_jump_prob
    jump_sizes = (
        np.random.normal(jd["mean"], jd["std"], (days, n_sims)) * jump_mask
    )

    paths = np.zeros((days + 1, n_sims))
    paths[0] = start_price

    for t in range(1, days + 1):
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * t_shocks[t - 1]
        paths[t] = paths[t - 1] * np.exp(drift + diffusion) * (1 + jump_sizes[t - 1])
        paths[t] = np.maximum(paths[t], start_price * 0.02)  # Floor at 2%

    return paths


# ── Full Monte Carlo Pipeline ────────────────────────────────────────────────────────

def run_monte_carlo(
    current_price: float,
    regime: str,
    risk_score: float,
    crash_freq: float,
    vix_level: float = 20.0,
    yield_curve: float = 0.5,
    valuation_penalty: float = 0.0,
) -> dict:
    """
    Module 5 Entry Point: Run full multi-scenario Monte Carlo simulation.

    Args:
        current_price: Current S&P 500 price
        regime: Current market regime
        risk_score: Current composite risk score
        crash_freq: Historical annual crash frequency
        vix_level: Current VIX
        yield_curve: Current 10Y-3M spread
        valuation_penalty: Annual return drag from CAPE/trend

    Returns:
        dict with paths, statistics, crash probabilities, risk metrics
    """
    print("[MODULE 5] Running Monte Carlo simulation...")

    sim_cfg = config["simulation"]
    risk_cfg = config["risk"]

    # Build dynamically-adjusted scenarios
    scenarios = build_scenarios(regime, risk_score, vix_level, yield_curve, valuation_penalty)

    forecast_days = get_forecast_days()
    all_paths_list = []
    scenario_results = {}

    for name, params in scenarios.items():
        n_sims = max(200, int(sim_cfg["num_simulations"] * params["probability"]))
        sp = {"drift_adj": 0, "vol_mult": 1.0, "crash_mult": params["crash_multiplier"]}

        paths = simulate_paths(
            current_price, params["return"], params["volatility"],
            forecast_days, n_sims, crash_freq, risk_score, sp,
        )
        all_paths_list.append(paths)

        scenario_results[name] = {
            "probability": params["probability"],
            "return": params["return"],
            "volatility": params["volatility"],
            "description": params["description"],
            "mean_final": float(np.mean(paths[-1])),
            "total_return": float(np.mean(paths[-1]) / current_price - 1) * 100,
            "n_sims": n_sims,
        }
        print(
            f"  [OK] {name} ({params['probability']*100:.0f}%): "
            f"{n_sims:,} sims → ${np.mean(paths[-1]):,.0f}"
        )

    # Combine all paths
    all_paths = np.concatenate(all_paths_list, axis=1)

    # Statistics
    final = all_paths[-1]
    mean_path = all_paths.mean(axis=1)
    median_path = np.median(all_paths, axis=1)
    p05 = np.percentile(all_paths, 5, axis=1)
    p25 = np.percentile(all_paths, 25, axis=1)
    p75 = np.percentile(all_paths, 75, axis=1)
    p95 = np.percentile(all_paths, 95, axis=1)

    # Crash probabilities by horizon (peak-to-trough, matches identify_crashes)
    crash_threshold = risk_cfg["crash_threshold"]
    horizons = {
        "3mo": 63, "6mo": 126, "12mo": 252, "18mo": 378,
        "24mo": 504, "36mo": 756, "48mo": 1008, "60mo": 1260,
    }
    crash_probs = {}
    for label, d in horizons.items():
        if d < all_paths.shape[0]:
            window_paths = all_paths[:d + 1]
            running_peak = np.maximum.accumulate(window_paths, axis=0)
            drawdowns = (window_paths - running_peak) / running_peak
            max_dd_per_sim = drawdowns.min(axis=0)
            crash_probs[label] = float(np.mean(max_dd_per_sim <= -crash_threshold)) * 100

    # Risk metrics
    var_95_val = np.percentile(final, 5)
    cvar_vals = final[final <= var_95_val]
    cvar_95 = float(np.mean(cvar_vals)) if len(cvar_vals) > 0 else var_95_val

    # Max drawdown (peak-to-trough)
    running_peak_all = np.maximum.accumulate(all_paths, axis=0)
    all_drawdowns = (all_paths - running_peak_all) / running_peak_all
    max_dd_per_sim = all_drawdowns.min(axis=0)
    avg_max_dd = float(np.mean(max_dd_per_sim))

    forecast_years = sim_cfg["forecast_years"]

    results = {
        "all_paths": all_paths,
        "mean_path": mean_path,
        "median_path": median_path,
        "p05": p05, "p25": p25, "p75": p75, "p95": p95,
        "final_mean": float(np.mean(final)),
        "final_median": float(np.median(final)),
        "total_return_pct": float(np.mean(final) / current_price - 1) * 100,
        "annual_return_pct": float(
            ((np.mean(final) / current_price) ** (1 / forecast_years) - 1) * 100
        ),
        "scenarios": scenario_results,
        "crash_probs": crash_probs,
        "cvar_95_pct": float((cvar_95 / current_price - 1) * 100),
        "max_drawdown_pct": float(avg_max_dd * 100),
        "crash_prob_1y": crash_probs.get("12mo", 0),
        "crash_prob_5y": crash_probs.get("60mo", 0),
    }

    print(f"\n  [OK] Mean {forecast_years}Y target: ${results['final_mean']:,.0f} "
          f"({results['total_return_pct']:+.1f}%)")
    print(f"  [OK] 90% CI: ${p05[-1]:,.0f} — ${p95[-1]:,.0f}")
    print(f"  [OK] 1Y crash probability: {results['crash_prob_1y']:.1f}%")
    print(f"  [OK] 5Y crash probability: {results['crash_prob_5y']:.1f}%")
    print(f"  [OK] CVaR (95%): {results['cvar_95_pct']:.1f}%")
    print(f"  [OK] Avg max drawdown: {results['max_drawdown_pct']:.1f}%\n")

    return results
