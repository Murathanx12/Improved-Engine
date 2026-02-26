"""
Module 5: Monte Carlo Simulation Engine (v2 — Calibrated)
===========================================================

Major changes from v1:
    1. CALIBRATION FIX: Risk score now has 3x stronger impact on crash
       probability, giving real discrimination between calm/stressed markets.
    2. GJR-GARCH integration: Uses time-varying volatility when available.
    3. Reduced base jump rate to match ~20-25% 1Y crash probability in
       normal conditions (was producing ~50%).

Usage:
    from finpredict.simulation.monte_carlo import run_monte_carlo, simulate_paths
"""

import numpy as np
from scipy import stats

from finpredict.config import config, get_forecast_days
from finpredict.simulation.scenarios import build_scenarios


def simulate_paths(
    start_price: float,
    annual_return: float,
    annual_vol: float,
    days: int,
    n_sims: int,
    crash_rate: float | None = None,
    risk_level: float = 0.0,
    scenario_params: dict | None = None,
    garch_vol: float | None = None,
) -> np.ndarray:
    """
    Jump-diffusion Monte Carlo path simulation (calibrated v2).

    CALIBRATION FIX:
        Risk score scaling increased from ±20% to ±50%.
        At risk=-1σ (calm):   factor=0.50 → crash rate halved
        At risk= 0σ (normal): factor=1.00 → baseline
        At risk=+2σ (stress): factor=2.00 → crash rate doubled

    Args:
        start_price: Starting price level
        annual_return: Expected annual drift (log)
        annual_vol: Annualized volatility — fallback if no GARCH
        days: Trading days to simulate
        n_sims: Number of paths
        crash_rate: Historical annual crash frequency
        risk_level: Current composite risk score (z-score)
        scenario_params: Dict with 'crash_mult', 'vol_mult', 'drift_adj'
        garch_vol: GARCH conditional volatility (annualized), overrides annual_vol

    Returns:
        np.ndarray of shape (days+1, n_sims)
    """
    sim_cfg = config["simulation"]
    jd = sim_cfg["jump_diffusion"]

    params = scenario_params or {}
    mu = annual_return + params.get("drift_adj", 0)
    crash_mult = params.get("crash_mult", 1.0)

    # Use GARCH vol when available, otherwise historical
    sigma = garch_vol if garch_vol is not None else annual_vol
    sigma *= params.get("vol_mult", 1.0)

    trading_days = sim_cfg["trading_days_per_year"]
    dt = 1.0 / trading_days

    # ── CALIBRATION FIX: Stronger risk conditioning ──────────────────────
    #
    # Old (v1): risk_factor = 1.0 + 0.20 * risk_level  → ±20%, no real signal
    # New (v2): risk_factor = 1.0 + 0.50 * risk_level  → ±50%, real discrimination
    #
    # This is THE fix for the inverted calibration:
    #   Low  risk (<15% predicted): 86% actual crashes  ← WRONG
    #   High risk (>40% predicted):  7% actual crashes  ← WRONG
    #
    # The problem: with ±20% scaling, a -1σ risk score barely reduces
    # crash rate (factor=0.80), so "low risk" predictions still had tons
    # of crashes. Meanwhile "high risk" was also barely different (factor=1.20).
    #
    base_jump_rate = jd["annual_rate"]
    if crash_rate is not None:
        historical_sudden = crash_rate * 0.30
        base_jump_rate = max(base_jump_rate, historical_sudden)

    risk_factor = 1.0 + 0.50 * risk_level
    risk_factor = max(0.25, min(3.0, risk_factor))
    adj_jump_rate = min(0.30, base_jump_rate * crash_mult * risk_factor)
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
        paths[t] = np.maximum(paths[t], start_price * 0.02)

    return paths


def run_monte_carlo(
    current_price: float,
    regime: str,
    risk_score: float,
    crash_freq: float,
    vix_level: float = 20.0,
    yield_curve: float = 0.5,
    valuation_penalty: float = 0.0,
    garch_vol: float | None = None,
    recession_prob: float | None = None,
) -> dict:
    """
    Full multi-scenario Monte Carlo simulation.

    New in v2:
        - garch_vol: GARCH conditional volatility replaces constant σ
        - recession_prob: FRED-based recession probability for scenario weights

    Returns:
        dict with paths, statistics, crash probabilities, risk metrics
    """
    print("[MODULE 5] Running Monte Carlo simulation...")

    sim_cfg = config["simulation"]
    risk_cfg = config["risk"]

    scenarios = build_scenarios(
        regime, risk_score, vix_level, yield_curve, valuation_penalty,
        recession_prob=recession_prob,
    )

    forecast_days = get_forecast_days()
    all_paths_list = []
    scenario_results = {}

    for name, params in scenarios.items():
        n_sims = max(200, int(sim_cfg["num_simulations"] * params["probability"]))
        sp = {"drift_adj": 0, "vol_mult": 1.0, "crash_mult": params["crash_multiplier"]}

        paths = simulate_paths(
            current_price, params["return"], params["volatility"],
            forecast_days, n_sims, crash_freq, risk_score, sp,
            garch_vol=garch_vol,
        )
        all_paths_list.append(paths)

        scenario_results[name] = {
            "probability": params["probability"],
            "return": params["return"],
            "volatility": params["volatility"],
            "description": params["description"],
            "category": params.get("category", "neutral"),
            "mean_final": float(np.mean(paths[-1])),
            "total_return": float(np.mean(paths[-1]) / current_price - 1) * 100,
            "n_sims": n_sims,
        }
        print(
            f"  [OK] {name} ({params['probability']*100:.0f}%): "
            f"{n_sims:,} sims → ${np.mean(paths[-1]):,.0f}"
        )

    all_paths = np.concatenate(all_paths_list, axis=1)

    # Statistics
    final = all_paths[-1]
    mean_path = all_paths.mean(axis=1)
    median_path = np.median(all_paths, axis=1)
    p05 = np.percentile(all_paths, 5, axis=1)
    p25 = np.percentile(all_paths, 25, axis=1)
    p75 = np.percentile(all_paths, 75, axis=1)
    p95 = np.percentile(all_paths, 95, axis=1)

    # Crash probabilities by horizon (peak-to-trough drawdown)
    crash_threshold = risk_cfg["crash_threshold"]
    horizons = {
        "3mo": 63, "6mo": 126, "12mo": 252, "18mo": 378,
        "24mo": 504, "36mo": 756, "48mo": 1008, "60mo": 1260,
    }
    crash_probs = {}
    for label, d in horizons.items():
        if d < all_paths.shape[0]:
            window = all_paths[:d + 1]
            peak = np.maximum.accumulate(window, axis=0)
            dd = (window - peak) / peak
            max_dd = dd.min(axis=0)
            crash_probs[label] = float(np.mean(max_dd <= -crash_threshold)) * 100

    # Risk metrics
    var_95_val = np.percentile(final, 5)
    cvar_vals = final[final <= var_95_val]
    cvar_95 = float(np.mean(cvar_vals)) if len(cvar_vals) > 0 else var_95_val

    peak_all = np.maximum.accumulate(all_paths, axis=0)
    dd_all = (all_paths - peak_all) / peak_all
    max_dd_per_sim = dd_all.min(axis=0)
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

    vol_src = f"GARCH {garch_vol*100:.0f}%" if garch_vol else "historical"
    print(f"\n  [OK] Volatility source: {vol_src}")
    print(f"  [OK] Mean {forecast_years}Y target: ${results['final_mean']:,.0f} "
          f"({results['total_return_pct']:+.1f}%)")
    print(f"  [OK] 90% CI: ${p05[-1]:,.0f} — ${p95[-1]:,.0f}")
    print(f"  [OK] 1Y crash probability: {results['crash_prob_1y']:.1f}%")
    print(f"  [OK] 5Y crash probability: {results['crash_prob_5y']:.1f}%")
    print(f"  [OK] CVaR (95%): {results['cvar_95_pct']:.1f}%")
    print(f"  [OK] Avg max drawdown: {results['max_drawdown_pct']:.1f}%\n")

    return results
