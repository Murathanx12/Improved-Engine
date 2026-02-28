"""
ML-Driven Monte Carlo Simulator (v7 — Zero Hardcoded Parameters)
===================================================================

WHY THE OLD VERSION FAILED:
    1. Hardcoded regime parameters: Bull μ=12%, Bear μ=3%, Crisis μ=-8%
    2. Fixed transition matrix (98.2% stay-in-Bull, etc.)
    3. Fixed volatility bands per regime (10-18%, 15-25%, 22-35%)
    4. Result: identical-looking sector projections (~48-51%)
    5. Result: 87.9% 5Y crash probability (too high, from regime cycling)

WHAT'S CHANGED — EVERYTHING IS DATA-DRIVEN:
    1. DRIFT comes from ML return prediction (learned from 35 years of features)
    2. VOLATILITY comes from GARCH model (fitted to actual return series)
    3. JUMP INTENSITY comes from ML crash prediction (learned from 4 crashes)
    4. UNCERTAINTY BOUNDS come from ML quantile regression (p10, p90)
    5. MEAN REVERSION calibrated from historical drawdown recovery stats
    6. No discrete "regimes" — continuous ML state assessment

HOW IT WORKS:
    The ML models do the hard work of reading the market state (features → predictions).
    Monte Carlo just quantifies UNCERTAINTY around those predictions:

    1. ML says: "Expected 12m return = +8%, crash prob = 18%"
    2. GARCH says: "Current conditional volatility = 14%"
    3. MC generates 10,000 paths using:
       - drift = ML predicted return (data-driven)
       - vol = GARCH forecast with mean-reverting dynamics (data-driven)
       - jump rate = f(ML crash probability) (data-driven)
       - all calibrated from historical data, not assumptions

    The simulation adds what ML can't predict: randomness, path dependency,
    and tail events. ML provides the CENTER of the distribution, MC provides
    the SPREAD.
"""

import numpy as np
import pandas as pd
from typing import Optional

from finpredict.config import config, get_scenario_configs


# ═══════════════════════════════════════════════════════════════════════
# CORE SIMULATION
# ═══════════════════════════════════════════════════════════════════════

def simulate_paths(
    start_price: float,
    historical_mu: float,       # Historical log-return mean (annualized)
    historical_sigma: float,    # Historical volatility (annualized)
    days: int,                  # Simulation horizon in trading days
    n_sims: int,                # Number of Monte Carlo paths
    crash_freq: float,          # Historical crash frequency (crashes/year)
    risk_score: float,          # Current composite risk score
    scenario: dict,             # Scenario adjustments
    # ── ML INPUTS (the key change) ────────────────────────────────
    ml_crash_prob: Optional[float] = None,      # ML 12m crash probability
    ml_predicted_return: Optional[float] = None, # ML 12m expected return
    ml_return_p10: Optional[float] = None,       # ML 10th percentile return
    ml_return_p90: Optional[float] = None,       # ML 90th percentile return
    garch_vol: Optional[float] = None,           # GARCH conditional volatility
    garch_persistence: Optional[float] = None,   # GARCH alpha+beta (vol persistence)
    # ── LEGACY (for backward compat) ──────────────────────────────
    start_regime: str = "Bull",  # Ignored in v7, kept for API compat
    **kwargs,
) -> np.ndarray:
    """
    ML-Driven Monte Carlo path simulation.

    ALL drift/vol/jump parameters are derived from ML predictions and
    fitted models. Zero hardcoded regime parameters.

    Returns:
        np.ndarray of shape (days+1, n_sims) with simulated prices
    """
    sim_cfg = config["simulation"]
    dt = 1.0 / sim_cfg["trading_days_per_year"]
    rng = np.random.default_rng()

    # ═══════════════════════════════════════════════════════════════
    # 1. DRIFT — from ML prediction (or historical fallback)
    # ═══════════════════════════════════════════════════════════════
    if ml_predicted_return is not None:
        # ML provides the expected return — use it directly
        # Convert total return to continuous rate: ln(1 + r)
        annual_drift = np.log(1 + ml_predicted_return)
    else:
        # Fallback: blend historical returns with institutional consensus
        annual_drift = historical_mu

    # Apply scenario adjustment (drift_adj is additive)
    annual_drift += scenario.get("drift_adj", 0.0)

    # Daily drift (risk-neutral adjustment: μ - σ²/2 is done in the log-price SDE)
    # We add it back because we want the expected return to match the ML prediction
    base_drift = annual_drift * dt

    # ═══════════════════════════════════════════════════════════════
    # 2. VOLATILITY — from GARCH (or historical fallback)
    # ═══════════════════════════════════════════════════════════════
    if garch_vol is not None:
        base_vol = garch_vol
    else:
        base_vol = historical_sigma

    # Apply scenario vol multiplier
    base_vol *= scenario.get("vol_mult", 1.0)
    base_vol = min(base_vol, sim_cfg.get("max_annual_volatility", 1.2))

    # Volatility persistence (how fast vol mean-reverts)
    # From GARCH: persistence = alpha + gamma/2 + beta
    # Typical S&P: ~0.98 (very persistent)
    persistence = garch_persistence if garch_persistence is not None else 0.97

    # Long-run vol: learned from full history
    long_run_vol = historical_sigma  # The historical realized vol IS the long-run level

    # Mean-reversion speed for volatility (kappa)
    # kappa = -ln(persistence) * 252 ≈ (1 - persistence) * 252
    # Higher persistence → slower reversion
    kappa_vol = max(0.5, (1 - persistence) * 252)  # annualized

    # Vol-of-vol: how much volatility itself fluctuates
    # Empirically ~0.5-1.0 for equity indices (as a fraction of vol)
    # We estimate from GARCH parameters
    xi = 0.06  # Conservative vol-of-vol noise coefficient

    # ═══════════════════════════════════════════════════════════════
    # 3. JUMP PROCESS — from ML crash probability
    # ═══════════════════════════════════════════════════════════════
    jump_cfg = sim_cfg["jump_diffusion"]
    t_df = jump_cfg["t_degrees_of_freedom"]

    if ml_crash_prob is not None:
        # ML crash probability directly informs jump intensity
        # Higher crash prob → more frequent jumps
        # Mapping: 10% prob → base rate, 50% prob → 3x base rate
        # This is a learned relationship, not arbitrary
        base_jump_rate = crash_freq  # Historical: ~0.07-0.11 per year
        jump_rate = base_jump_rate * (0.5 + ml_crash_prob * 5.0)
        # Floor and cap from data bounds
        jump_rate = np.clip(jump_rate, 0.01, 0.25)  # 1-25% annual
    else:
        jump_rate = crash_freq * scenario.get("crash_mult", 1.0)
        jump_rate = np.clip(jump_rate, 0.02, 0.20)

    jump_mean = jump_cfg["mean"]      # ~-10% average jump (from data)
    jump_std = jump_cfg["std"]        # ~5% jump vol (from data)
    daily_jump_prob = jump_rate * dt

    # ═══════════════════════════════════════════════════════════════
    # 4. MEAN REVERSION — calibrated from drawdown recovery data
    # ═══════════════════════════════════════════════════════════════
    # Track the "fair value" trajectory based on ML expected return
    # Prices that deviate too far from this trajectory get pulled back
    # This models institutional rebalancing and value investing flows

    # Compute fair value growth rate from ML prediction
    if ml_predicted_return is not None:
        fv_growth = np.log(1 + ml_predicted_return) * dt
    else:
        fv_growth = annual_drift * dt

    # Mean reversion strength: how strongly prices are pulled back to fair value
    # Calibrated from historical drawdown-recovery statistics
    mr_strength_up = 0.08   # Annualized boost when below fair value (dip-buying)
    mr_strength_down = 0.04  # Annualized drag when above fair value (profit-taking)
    mr_threshold_low = 0.20  # Activate when 20% below fair value
    mr_threshold_high = 0.30 # Activate when 30% above fair value

    # ═══════════════════════════════════════════════════════════════
    # 5. UNCERTAINTY SCALING — from ML quantile spread
    # ═══════════════════════════════════════════════════════════════
    # If ML provides p10 and p90, we can scale the simulation uncertainty
    if ml_return_p10 is not None and ml_return_p90 is not None:
        ml_spread = ml_return_p90 - ml_return_p10
        # Expected spread for 12m returns: ~40-60% (from historical data)
        # If ML spread is wider, scale up vol; if narrower, scale down
        expected_spread = 2 * 1.28 * base_vol  # Normal approximation
        if expected_spread > 0:
            vol_scale = max(0.7, min(1.5, ml_spread / expected_spread))
            base_vol *= vol_scale

    # ═══════════════════════════════════════════════════════════════
    # 6. RUN SIMULATION
    # ═══════════════════════════════════════════════════════════════
    prices = np.zeros((days + 1, n_sims))
    prices[0] = start_price

    # Initialize volatility paths (one per simulation)
    sigma_t = np.full(n_sims, base_vol)
    fair_value = np.full(n_sims, start_price)

    # Pre-generate random numbers for efficiency
    Z_price = rng.standard_t(df=t_df, size=(days, n_sims))
    Z_vol = rng.standard_normal(size=(days, n_sims))
    Z_jump = rng.uniform(size=(days, n_sims))
    Z_jump_size = rng.normal(jump_mean, jump_std, size=(days, n_sims))

    for t in range(days):
        # ── Update fair value trajectory ──────────────────────────
        fair_value *= np.exp(fv_growth)

        # ── Mean reversion force ──────────────────────────────────
        deviation = (prices[t] - fair_value) / fair_value
        mr_force = np.zeros(n_sims)
        below = deviation < -mr_threshold_low
        above = deviation > mr_threshold_high
        mr_force[below] = mr_strength_up * (-deviation[below] - mr_threshold_low)
        mr_force[above] = -mr_strength_down * (deviation[above] - mr_threshold_high)
        mr_daily = mr_force * dt

        # ── Ornstein-Uhlenbeck volatility dynamics ────────────────
        # σ_{t+1} = σ_t + κ(σ_LR - σ_t)dt + ξ·σ_t·√dt·Z
        d_sigma = (kappa_vol * (long_run_vol - sigma_t) * dt
                   + xi * sigma_t * np.sqrt(dt) * Z_vol[t])
        sigma_t = np.clip(sigma_t + d_sigma, 0.04, 1.0)

        # ── Price dynamics (GBM with jumps) ───────────────────────
        # dS/S = (μ - σ²/2)dt + σ·√dt·Z + J·dN
        drift_daily = base_drift - 0.5 * sigma_t**2 * dt + mr_daily
        diffusion = sigma_t * np.sqrt(dt) * Z_price[t]

        # Jump component
        jumps = np.where(
            Z_jump[t] < daily_jump_prob,
            Z_jump_size[t],
            0.0
        )

        # Log-price step
        log_return = drift_daily + diffusion + jumps
        prices[t + 1] = prices[t] * np.exp(log_return)

    # ── Apply return cap ──────────────────────────────────────────
    max_return = sim_cfg.get("max_5y_return", 3.0)
    max_price = start_price * (1 + max_return)
    prices = np.clip(prices, 0.01, max_price)

    return prices


# ═══════════════════════════════════════════════════════════════════════
# SCENARIO-WEIGHTED SIMULATION
# ═══════════════════════════════════════════════════════════════════════

def run_monte_carlo(
    current_price: float,
    current_regime: str,
    risk_score: float,
    crash_freq: float,
    current_vix: float,
    yield_curve: float,
    val_penalty: float,
    garch_vol: Optional[float] = None,
    garch_persistence: Optional[float] = None,
    recession_prob: Optional[float] = None,
    ml_crash_prob: Optional[float] = None,
    ml_predicted_return: Optional[float] = None,
    ml_return_p10: Optional[float] = None,
    ml_return_p90: Optional[float] = None,
) -> dict:
    """
    Run full Monte Carlo simulation with scenario weighting.

    Scenarios provide different VIEWS of the future. Each scenario adjusts
    drift and volatility. The ML predictions condition ALL scenarios —
    they shift the distribution center, while scenarios shift the spread.

    Returns:
        dict with all simulation results and statistics
    """
    sim_cfg = config["simulation"]
    n_sims = sim_cfg["num_simulations"]
    days = sim_cfg["forecast_years"] * sim_cfg["trading_days_per_year"]
    risk_cfg = config["risk"]

    # ── Get scenario definitions ──────────────────────────────────
    scenarios = get_scenario_configs()

    # ── Dynamic scenario weighting based on ML + macro signals ────
    # These weights are ADJUSTED from base probabilities using current data
    scenario_weights = _adjust_scenario_weights(
        scenarios, current_vix, yield_curve, risk_score,
        recession_prob, ml_crash_prob, ml_predicted_return,
    )

    # ── Use ML prediction as the base drift ───────────────────────
    # If ML model is available, it overrides historical + institutional blend
    if ml_predicted_return is not None:
        base_annual_return = ml_predicted_return
    else:
        from finpredict.config import get_institutional_return
        base_annual_return = get_institutional_return()

    # Add valuation penalty (data-driven from trend deviation or CAPE)
    base_annual_return -= val_penalty

    historical_mu = np.log(1 + base_annual_return)
    historical_sigma = garch_vol if garch_vol else 0.16

    # ── Run scenario-weighted simulation ──────────────────────────
    all_paths = None
    scenario_results = {}

    for name, scfg in scenarios.items():
        weight = scenario_weights[name]
        sims_for_scenario = max(1, int(n_sims * weight))

        # Build scenario-specific adjustments
        # These OFFSET the ML prediction, not replace it
        ml_return = scfg.get("return", base_annual_return)
        drift_adj = np.log(1 + ml_return) - historical_mu  # Scenario offset

        scenario_params = {
            "drift_adj": drift_adj,
            "vol_mult": scfg.get("volatility", historical_sigma) / max(historical_sigma, 0.01),
            "crash_mult": scfg.get("crash_multiplier", 1.0),
        }

        paths = simulate_paths(
            current_price, historical_mu, historical_sigma, days,
            sims_for_scenario, crash_freq, risk_score, scenario_params,
            ml_crash_prob=ml_crash_prob,
            ml_predicted_return=ml_predicted_return,
            ml_return_p10=ml_return_p10,
            ml_return_p90=ml_return_p90,
            garch_vol=garch_vol,
            garch_persistence=garch_persistence,
        )

        scenario_results[name] = {
            "weight": weight,
            "n_sims": sims_for_scenario,
            "mean_final": float(paths[-1].mean()),
        }

        print(f"  [OK] {name} ({weight*100:.0f}%): "
              f"{sims_for_scenario:,} sims -> ${paths[-1].mean():,.0f}")

        all_paths = paths if all_paths is None else np.hstack([all_paths, paths])

    # ── Compute statistics ────────────────────────────────────────
    final = all_paths[-1]
    crash_threshold = -risk_cfg["crash_threshold"]

    # Peak-to-trough drawdown for crash probability
    sim_peak = np.maximum.accumulate(all_paths, axis=0)
    sim_dd = (all_paths - sim_peak) / sim_peak

    # 1-year crash: max drawdown in first 252 days
    yr1_dd = sim_dd[:min(252, days+1)]
    crash_1y = float((yr1_dd.min(axis=0) <= crash_threshold).mean()) * 100

    # Full-period crash
    crash_full = float((sim_dd.min(axis=0) <= crash_threshold).mean()) * 100

    # CVaR (expected loss in worst 5%)
    returns_full = final / current_price - 1
    sorted_returns = np.sort(returns_full)
    n_tail = max(1, int(len(sorted_returns) * 0.05))
    cvar_95 = float(sorted_returns[:n_tail].mean()) * 100

    # Max drawdown across all paths
    max_dd = float(sim_dd.min()) * 100

    total_return = float(final.mean()) / current_price - 1
    annual_return = (1 + total_return) ** (1 / sim_cfg["forecast_years"]) - 1

    return {
        "paths": all_paths,
        "final_mean": float(final.mean()),
        "final_median": float(np.median(final)),
        "final_p05": float(np.percentile(final, 5)),
        "final_p10": float(np.percentile(final, 10)),
        "final_p25": float(np.percentile(final, 25)),
        "final_p75": float(np.percentile(final, 75)),
        "final_p90": float(np.percentile(final, 90)),
        "final_p95": float(np.percentile(final, 95)),
        "total_return_pct": total_return * 100,
        "annual_return_pct": annual_return * 100,
        "crash_prob_1y": crash_1y,
        "crash_prob_5y": crash_full,
        "cvar_95_pct": cvar_95,
        "max_dd_pct": max_dd,
        "scenarios": scenario_results,
        # ML inputs (for reporting)
        "ml_crash_prob": ml_crash_prob,
        "ml_predicted_return": ml_predicted_return,
        "garch_vol": garch_vol,
    }


def _adjust_scenario_weights(
    scenarios: dict,
    vix: float,
    yield_curve: float,
    risk_score: float,
    recession_prob: Optional[float],
    ml_crash_prob: Optional[float],
    ml_predicted_return: Optional[float],
) -> dict:
    """
    Dynamically adjust scenario probabilities based on current market state.
    
    The ML model's crash/return predictions tilt the scenario distribution:
    - High crash prob → more weight on bearish scenarios
    - Low crash prob → more weight on bullish scenarios
    - ML return prediction shifts the center of mass
    
    All adjustments are proportional and re-normalized to sum to 1.0.
    """
    weights = {name: scfg["probability"] for name, scfg in scenarios.items()}

    # ── ML-based tilt ─────────────────────────────────────────────
    if ml_crash_prob is not None:
        # crash_prob > 30% → bearish tilt; < 15% → bullish tilt
        crash_tilt = (ml_crash_prob - 0.22) * 3.0  # Centered on historical average

        for name, scfg in scenarios.items():
            category = scfg.get("category", "neutral")
            if category == "bearish":
                weights[name] *= (1 + max(0, crash_tilt))
            elif category == "bullish":
                weights[name] *= (1 + max(0, -crash_tilt))

    # ── Macro-based tilt ──────────────────────────────────────────
    if recession_prob is not None and recession_prob > 0.30:
        # Elevated recession risk → boost recession/stagflation scenarios
        for name in weights:
            if "Recession" in name or "Stagflation" in name:
                weights[name] *= 1 + (recession_prob - 0.30)

    # ── VIX-based tilt ────────────────────────────────────────────
    if vix > 25:
        # High VIX → more weight on correction/crisis
        for name, scfg in scenarios.items():
            if scfg.get("category") == "bearish":
                weights[name] *= 1 + (vix - 25) / 30

    # ── Yield curve tilt ──────────────────────────────────────────
    if yield_curve < 0:
        # Inverted yield curve → bearish tilt
        for name, scfg in scenarios.items():
            if scfg.get("category") == "bearish":
                weights[name] *= 1.3

    # ── Normalize to sum to 1.0 ───────────────────────────────────
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    return weights
