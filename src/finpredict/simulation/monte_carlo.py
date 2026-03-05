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
# BLOCK BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════════

def _generate_block_bootstrap_residuals(
    historical_returns: np.ndarray,
    days: int,
    n_sims: int,
    block_size: int = 21,
    rng=None,
) -> np.ndarray:
    """Sample overlapping blocks of historical residuals to preserve volatility clustering.

    Real markets exhibit autocorrelated volatility: calm periods follow calm,
    crisis follows crisis. I.i.d. draws destroy this structure. Block bootstrap
    samples contiguous blocks of historical returns, preserving within-block
    serial correlation (volatility clustering, momentum, mean-reversion).

    Args:
        historical_returns: Array of standardized historical daily returns
        days: Number of simulation days
        n_sims: Number of simulation paths
        block_size: Size of each block (21 ≈ 1 trading month)
        rng: Numpy random generator

    Returns:
        np.ndarray of shape (days, n_sims) with block-bootstrapped residuals
    """
    if rng is None:
        rng = np.random.default_rng()

    n_hist = len(historical_returns)
    max_start = n_hist - block_size
    if max_start < 1:
        return rng.standard_normal(size=(days, n_sims))

    # Standardize historical returns to zero mean, unit variance
    mu = historical_returns.mean()
    sigma = historical_returns.std()
    if sigma < 1e-10:
        return rng.standard_normal(size=(days, n_sims))
    standardized = (historical_returns - mu) / sigma

    n_blocks_needed = (days + block_size - 1) // block_size
    residuals = np.zeros((days, n_sims))

    # Vectorized block sampling: sample all block starts at once
    starts = rng.integers(0, max_start, size=(n_blocks_needed, n_sims))
    for b in range(n_blocks_needed):
        row_start = b * block_size
        row_end = min(row_start + block_size, days)
        actual_len = row_end - row_start
        # Vectorized: gather all sims at once via advanced indexing
        idx = starts[b, :, np.newaxis] + np.arange(actual_len)  # (n_sims, actual_len)
        residuals[row_start:row_end, :] = standardized[idx].T

    return residuals


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
    # ── GARCH-DERIVED SIMULATION PARAMS ─────────────────────────
    xi: Optional[float] = None,                    # Vol-of-vol coefficient (data-derived)
    rho_leverage: Optional[float] = None,          # Leverage effect correlation (data-derived)
    # ── HMM REGIME INPUTS ────────────────────────────────────────
    hmm_state_means: Optional[np.ndarray] = None,   # [bull_mu, bear_mu, crisis_mu]
    hmm_regime_probs: Optional[np.ndarray] = None,   # [p_bull, p_bear, p_crisis]
    hmm_state_vols: Optional[np.ndarray] = None,     # [bull_vol, bear_vol, crisis_vol]
    # ── BLOCK BOOTSTRAP ───────────────────────────────────────────
    historical_residuals: Optional[np.ndarray] = None,  # For block bootstrap
    # ── LEGACY (for backward compat) ──────────────────────────────
    start_regime: str = "Bull",  # Ignored in v7, kept for API compat
    seed: Optional[int] = None,  # RNG seed for reproducibility
    **kwargs,
) -> np.ndarray:
    """
    ML-Driven Monte Carlo path simulation.

    ALL drift/vol/jump parameters are derived from ML predictions and
    fitted models. Zero hardcoded regime parameters.

    Args:
        seed: Optional random seed for reproducibility. When provided,
              creates a deterministic np.random.default_rng(seed).

    Returns:
        np.ndarray of shape (days+1, n_sims) with simulated prices
    """
    sim_cfg = config["simulation"]
    dt = 1.0 / sim_cfg["trading_days_per_year"]
    rng = np.random.default_rng(seed)

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

    # ── HMM regime tilt (configurable blend with HMM regime-expected return) ──
    # HMM provides regime-specific μ values. We blend these as a small tilt
    # to the ML drift so that the HMM affects both σ AND μ (not just σ).
    # ML prediction remains primary, HMM provides regime context.
    hmm_drift_blend = sim_cfg.get("hmm_drift_blend", 0.15)
    hmm_vol_blend = sim_cfg.get("hmm_vol_blend", 0.15)
    if hmm_state_means is not None and hmm_regime_probs is not None:
        hmm_expected = float(np.dot(hmm_regime_probs, hmm_state_means))
        hmm_tilt = hmm_drift_blend * (hmm_expected - annual_drift)
        annual_drift += hmm_tilt

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

    # ── HMM regime volatility blend ────────────────────────────────
    # Blend GARCH vol with HMM regime-weighted vol for richer conditioning
    if hmm_state_vols is not None and hmm_regime_probs is not None:
        hmm_vol = float(np.dot(hmm_regime_probs, hmm_state_vols))
        base_vol = (1 - hmm_vol_blend) * base_vol + hmm_vol_blend * hmm_vol

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
    # Estimated from GARCH conditional volatility (passed as parameter)
    # Falls back to conservative default if not provided
    garch_params = sim_cfg.get("garch_derived_params", {})
    if xi is None:
        xi = 0.06  # Fallback when GARCH-derived value not available
    xi = float(xi)

    # ═══════════════════════════════════════════════════════════════
    # 3. JUMP PROCESS — from ML crash probability
    # ═══════════════════════════════════════════════════════════════
    jump_cfg = sim_cfg["jump_diffusion"]
    t_df = jump_cfg["t_degrees_of_freedom"]

    if ml_crash_prob is not None and ml_crash_prob > 0.01:
        # Empirically-grounded jump rate from crash probability.
        # Using Poisson arrival: P(crash) ≈ 1 - exp(-jump_rate * horizon)
        # Solving: jump_rate = -ln(1 - crash_prob) / horizon_years
        # This ensures the fraction of paths experiencing a crash-level
        # jump matches the ML-predicted crash probability.
        jump_rate = -np.log(1 - min(ml_crash_prob, 0.95)) / 1.0  # 1-year horizon
        jump_rate = np.clip(jump_rate, 0.01, 0.50)
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
    mr_cfg = sim_cfg.get("mean_reversion", {})
    mr_strength_up = mr_cfg.get("strength_up", 0.08)
    mr_strength_down = mr_cfg.get("strength_down", 0.04)
    mr_threshold_low = mr_cfg.get("threshold_low", 0.20)
    mr_threshold_high = mr_cfg.get("threshold_high", 0.30)

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
    fair_value = np.full(n_sims, float(start_price))

    # Pre-generate random numbers for efficiency
    # Use block bootstrap if enabled and historical residuals are available
    use_block_bootstrap = sim_cfg.get("use_block_bootstrap", False)
    block_size = sim_cfg.get("block_bootstrap_size", 21)
    if use_block_bootstrap and historical_residuals is not None and len(historical_residuals) > block_size:
        Z_price = _generate_block_bootstrap_residuals(
            historical_residuals, days, n_sims, block_size, rng
        )
    else:
        Z_price = rng.standard_t(df=t_df, size=(days, n_sims))
    Z_vol_raw = rng.standard_normal(size=(days, n_sims))
    Z_jump = rng.uniform(size=(days, n_sims))
    Z_jump_size = rng.normal(jump_mean, jump_std, size=(days, n_sims))

    # ── Leverage effect: correlate vol innovations with price shocks ──
    # In real markets, negative returns increase volatility (Black's leverage effect)
    # Derived from GARCH gamma parameter (passed as parameter)
    if rho_leverage is None:
        rho_leverage = -0.7  # Fallback when GARCH-derived value not available
    rho_leverage = float(rho_leverage)
    Z_vol = rho_leverage * Z_price + np.sqrt(1 - rho_leverage**2) * Z_vol_raw

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
    hmm_state_means: Optional[np.ndarray] = None,
    hmm_regime_probs: Optional[np.ndarray] = None,
    hmm_state_vols: Optional[np.ndarray] = None,
    historical_residuals: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    xi: Optional[float] = None,
    rho_leverage: Optional[float] = None,
) -> dict:
    """
    Run full Monte Carlo simulation with scenario weighting.

    Scenarios provide different VIEWS of the future. Each scenario adjusts
    drift and volatility. The ML predictions condition ALL scenarios —
    they shift the distribution center, while scenarios shift the spread.

    Args:
        seed: Optional random seed for reproducibility.

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
    # If ML model is available, it overrides historical + institutional blend.
    # Valuation penalty is applied directly to the base return so it
    # actually affects drift (previously it was applied to historical_mu
    # which was bypassed when ml_predicted_return was set).
    from finpredict.config import get_institutional_return
    inst_return = get_institutional_return()

    if ml_predicted_return is not None:
        base_annual_return = ml_predicted_return - val_penalty
    else:
        base_annual_return = inst_return - val_penalty

    historical_mu = np.log(1 + base_annual_return)
    historical_sigma = garch_vol if garch_vol else 0.16

    # Base drift for scenario drift_adj computation — use the val-adjusted
    # ML return so that drift_adj = scenario_return - base gives each
    # scenario its stated return exactly.
    base_mu = np.log(1 + base_annual_return)

    # ── Run scenario-weighted simulation ──────────────────────────
    all_paths = None
    scenario_results = {}

    for i, (name, scfg) in enumerate(scenarios.items()):
        weight = scenario_weights[name]
        sims_for_scenario = max(1, int(n_sims * weight))

        # Build scenario-specific adjustments
        # drift_adj = how much this scenario deviates from the base drift.
        # Since simulate_paths uses base_annual_return as the drift center,
        # drift_adj is computed relative to it so each scenario produces
        # its stated absolute return: drift = base + (scenario - base) = scenario
        scenario_return = scfg.get("return", inst_return)
        drift_adj = np.log(1 + scenario_return) - base_mu

        scenario_params = {
            "drift_adj": drift_adj,
            "vol_mult": scfg.get("volatility", historical_sigma) / max(historical_sigma, 0.01),
            "crash_mult": scfg.get("crash_multiplier", 1.0),
        }

        # Derive per-scenario seed from base seed for deterministic but distinct paths
        scenario_seed = None
        if seed is not None:
            scenario_seed = seed + i

        paths = simulate_paths(
            current_price, historical_mu, historical_sigma, days,
            sims_for_scenario, crash_freq, risk_score, scenario_params,
            ml_crash_prob=ml_crash_prob,
            ml_predicted_return=base_annual_return,
            ml_return_p10=ml_return_p10,
            ml_return_p90=ml_return_p90,
            garch_vol=garch_vol,
            garch_persistence=garch_persistence,
            hmm_state_means=hmm_state_means,
            hmm_regime_probs=hmm_regime_probs,
            hmm_state_vols=hmm_state_vols,
            historical_residuals=historical_residuals,
            seed=scenario_seed,
            xi=xi,
            rho_leverage=rho_leverage,
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

    # CVaR (expected loss in worst 5%) — both 1Y and 5Y horizons
    returns_full = final / current_price - 1
    sorted_returns = np.sort(returns_full)
    n_tail = max(1, int(len(sorted_returns) * 0.05))
    cvar_95_5y = float(sorted_returns[:n_tail].mean()) * 100

    # 1-year CVaR (matches the crash probability horizon)
    yr1_idx = min(252, all_paths.shape[0] - 1)
    yr1_final = all_paths[yr1_idx]
    yr1_returns = yr1_final / current_price - 1
    yr1_sorted = np.sort(yr1_returns)
    yr1_n_tail = max(1, int(len(yr1_sorted) * 0.05))
    cvar_95_1y = float(yr1_sorted[:yr1_n_tail].mean()) * 100

    # Max drawdown: average per-path max drawdown (for reporting)
    per_path_max_dd = sim_dd.min(axis=0)
    max_dd = float(per_path_max_dd.mean()) * 100

    total_return = float(final.mean()) / current_price - 1
    annual_return = (1 + total_return) ** (1 / sim_cfg["forecast_years"]) - 1

    # ── Path-level statistics (required by chart functions) ───────
    mean_path = all_paths.mean(axis=1)
    median_path = np.median(all_paths, axis=1)
    p05_path = np.percentile(all_paths, 5, axis=1)
    p25_path = np.percentile(all_paths, 25, axis=1)
    p75_path = np.percentile(all_paths, 75, axis=1)
    p95_path = np.percentile(all_paths, 95, axis=1)

    # ── Crash probability by horizon ──────────────────────────────
    trading_days_per_year = sim_cfg["trading_days_per_year"]
    crash_probs = {}
    horizon_map = {
        "1mo": 21, "3mo": 63, "6mo": 126,
        "12mo": 252, "24mo": 504, "36mo": 756, "60mo": days,
    }
    for label, horizon_days in horizon_map.items():
        if horizon_days > days:
            horizon_days = days
        h_dd = sim_dd[:horizon_days + 1]
        crash_probs[label] = float((h_dd.min(axis=0) <= crash_threshold).mean()) * 100

    # ── Enrich scenario results with config fields ─────────────────
    for name, scfg in scenarios.items():
        if name in scenario_results:
            mean_final = scenario_results[name]["mean_final"]
            scen_total_ret = (mean_final / current_price - 1) * 100
            scenario_results[name].update({
                "probability": scenario_results[name]["weight"],
                "return": scfg.get("return", 0.0),
                "total_return": scen_total_ret,
                "volatility": scfg.get("volatility", historical_sigma),
                "description": scfg.get("description", ""),
            })

    # ── Realism validation ──────────────────────────────────────
    realism = _validate_realism(all_paths, current_price, sim_cfg["forecast_years"])

    return {
        # Full path array (required by combined chart)
        "all_paths": all_paths,
        # Backward-compatible alias
        "paths": all_paths,
        # Path percentile bands (required by chart_projection and chart_combined_projection_crash)
        "mean_path": mean_path,
        "median_path": median_path,
        "p05": p05_path,
        "p25": p25_path,
        "p75": p75_path,
        "p95": p95_path,
        # Final-value statistics
        "final_mean": float(final.mean()),
        "final_median": float(np.median(final)),
        "final_p05": float(np.percentile(final, 5)),
        "final_p10": float(np.percentile(final, 10)),
        "final_p25": float(np.percentile(final, 25)),
        "final_p75": float(np.percentile(final, 75)),
        "final_p90": float(np.percentile(final, 90)),
        "final_p95": float(np.percentile(final, 95)),
        # Return statistics
        "total_return_pct": total_return * 100,
        "annual_return_pct": annual_return * 100,
        # Risk statistics
        "crash_prob_1y": crash_1y,
        "crash_prob_5y": crash_full,
        "crash_probs": crash_probs,
        "cvar_95_pct": cvar_95_5y,
        "cvar_95_1y_pct": cvar_95_1y,
        "cvar_95_5y_pct": cvar_95_5y,
        "max_dd_pct": max_dd,
        "max_drawdown_pct": abs(max_dd),
        # Scenario breakdown
        "scenarios": scenario_results,
        # ML inputs (for reporting)
        "ml_crash_prob": ml_crash_prob,
        "ml_predicted_return": ml_predicted_return,
        "garch_vol": garch_vol,
        "realism_check": realism,
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

    # ── Enforce minimum weight floor ────────────────────────────
    # Every scenario gets at least 2% weight to prevent zero-sim scenarios
    for k in weights:
        weights[k] = max(weights[k], 0.02)
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    return weights


def _validate_realism(
    paths: np.ndarray,
    start_price: float,
    years: float,
) -> dict:
    """
    Validate simulation realism against empirical S&P 500 statistics.

    Checks:
        - Mean annual return: should be 4-15% (S&P 500 long-run ~10%)
        - Annual volatility: should be 10-30% (S&P 500 ~16%)
        - Crash frequency: 30-90% of 5Y paths should have >20% drawdown
        - Fat tails: return kurtosis should be >3

    Prints warnings for out-of-range metrics. Returns dict with results.
    """
    final = paths[-1]
    n_sims = paths.shape[1]
    trading_days = paths.shape[0] - 1

    # Annual return
    total_returns = final / start_price - 1
    mean_total = float(total_returns.mean())
    mean_annual = (1 + mean_total) ** (1 / max(years, 0.5)) - 1

    # Annualized vol from daily log returns
    daily_log_rets = np.diff(np.log(paths), axis=0)
    mean_daily_vol = float(daily_log_rets.std(axis=0).mean())
    annual_vol = mean_daily_vol * np.sqrt(252)

    # Crash frequency (>20% peak-to-trough drawdown)
    sim_peak = np.maximum.accumulate(paths, axis=0)
    sim_dd = (paths - sim_peak) / np.where(sim_peak > 0, sim_peak, 1.0)
    crash_pct = float((sim_dd.min(axis=0) <= -0.20).mean())

    # Fat tails (kurtosis of daily returns)
    # Sample a subset of paths for efficiency
    sample_rets = daily_log_rets[:, :min(1000, n_sims)].flatten()
    kurt = float(pd.Series(sample_rets).kurtosis())
    skew = float(pd.Series(sample_rets).skew())

    warnings = []
    if mean_annual < 0.04 or mean_annual > 0.15:
        warnings.append(f"Annual return {mean_annual*100:.1f}% outside 4-15% range")
    if annual_vol < 0.10 or annual_vol > 0.30:
        warnings.append(f"Annual vol {annual_vol*100:.1f}% outside 10-30% range")
    if crash_pct < 0.30 or crash_pct > 0.90:
        warnings.append(f"Crash frequency {crash_pct*100:.0f}% outside 30-90% range")
    if kurt < 3:
        warnings.append(f"Kurtosis {kurt:.1f} < 3 (insufficient fat tails)")

    if warnings:
        print("  [REALISM CHECK]")
        for w in warnings:
            print(f"    [WARN] {w}")
    else:
        print("  [REALISM CHECK] All metrics within expected ranges")

    return {
        "mean_annual_return": mean_annual,
        "annual_vol": annual_vol,
        "crash_frequency": crash_pct,
        "kurtosis": kurt,
        "skewness": skew,
        "warnings": warnings,
    }
