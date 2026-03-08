"""
Scenario Management
====================

Defines the multi-scenario framework and dynamically adjusts probabilities
based on current market conditions (regime, risk score, VIX, yield curve).

Scenarios are defined in engine_config.yaml and loaded at runtime.
The base probabilities are then adjusted using market conditions.

Usage:
    from finpredict.simulation.scenarios import build_scenarios

    scenarios = build_scenarios(regime, risk_score, vix_level, yield_curve)
"""

from finpredict.config import get_scenario_configs


def build_scenarios(
    regime: str,
    risk_score: float,
    vix_level: float = 20.0,
    yield_curve: float = 0.5,
    valuation_penalty: float = 0.0,
    recession_prob: float | None = None,
) -> dict:
    """
    Build scenario dict with dynamically adjusted probabilities.

    Loads base scenarios from engine_config.yaml, applies valuation penalty
    to returns, then adjusts probabilities based on current conditions.

    New in v2: recession_prob from FRED indicators shifts bearish scenarios.

    Args:
        regime: Current market regime ("Bull", "Bear", "Volatile", "Crisis")
        risk_score: Current composite risk score (z-score)
        vix_level: Current VIX level
        yield_curve: Current 10Y-3M yield spread (negative = inverted)
        valuation_penalty: Annual return adjustment from CAPE/trend analysis
        recession_prob: FRED-derived recession probability (0-1), None to skip

    Returns:
        dict of scenario name → {probability, return, volatility, crash_mult, ...}
    """
    scenarios = get_scenario_configs()

    # Apply valuation penalty to all scenario returns
    if abs(valuation_penalty) > 0.001:
        for s in scenarios.values():
            s["return"] = s["return"] + valuation_penalty

    # Dynamic probability adjustment
    scenarios = _adjust_probabilities(
        scenarios,
        regime,
        risk_score,
        vix_level,
        yield_curve,
        recession_prob,
    )

    return scenarios


def _adjust_probabilities(
    scenarios: dict,
    regime: str,
    risk_score: float,
    vix_level: float,
    yield_curve: float,
    recession_prob: float | None = None,
) -> dict:
    """
    DEPRECATED: Scenario weight adjustment is now consolidated in
    monte_carlo.py:_adjust_scenario_weights() which uses ML + macro signals.
    This function is retained for backward compatibility with build_scenarios()
    but is NOT used in the main simulation pipeline.

    Groups scenarios by their 'category' field (bullish/neutral/bearish)
    and shifts probability mass based on regime, risk, VIX, yield curve,
    and FRED recession probability.
    """
    s = {k: dict(v) for k, v in scenarios.items()}

    # Group by category
    bullish = [k for k, v in s.items() if v.get("category") == "bullish"]
    bearish = [k for k, v in s.items() if v.get("category") == "bearish"]
    [k for k, v in s.items() if v.get("category") == "neutral"]

    # ── FRED recession probability (strongest signal) ────────────────────
    if recession_prob is not None and recession_prob > 0.30:
        # High recession probability from FRED macro indicators
        boost = 1.0 + (recession_prob - 0.30) * 2.0  # 30%→1.0x, 60%→1.6x, 90%→2.2x
        for name in bearish:
            s[name]["probability"] *= boost
        for name in bullish:
            s[name]["probability"] *= max(0.3, 1.0 - (recession_prob - 0.30))

    # Risk score adjustments
    if risk_score > 2.0:  # High stress
        for name in bearish:
            s[name]["probability"] *= 1.4
        for name in bullish:
            s[name]["probability"] *= 0.6
    elif risk_score > 1.0:  # Elevated stress
        for name in bearish:
            s[name]["probability"] *= 1.2
        for name in bullish:
            s[name]["probability"] *= 0.8
    elif risk_score < -0.5:  # Low stress
        for name in bullish:
            s[name]["probability"] *= 1.3
        for name in bearish:
            s[name]["probability"] *= 0.7

    # VIX adjustments
    if vix_level > 30:
        if "Geopolitical Crisis" in s:
            s["Geopolitical Crisis"]["probability"] *= 1.4
        if "Recession" in s:
            s["Recession"]["probability"] *= 1.3
    elif vix_level < 15:
        for name in bullish:
            s[name]["probability"] *= 1.15

    # Yield curve inversion → recession signal
    if yield_curve < 0:
        if "Recession" in s:
            s["Recession"]["probability"] *= 1.5
        if "Stagflation" in s:
            s["Stagflation"]["probability"] *= 1.3
        for name in bullish:
            s[name]["probability"] *= 0.7

    # Regime adjustments (HMM or rule-based)
    regime_lower = regime.lower() if isinstance(regime, str) else "bull"
    if regime_lower == "bull":
        for name in bullish:
            s[name]["probability"] *= 1.15
        if "Base Case" in s:
            s["Base Case"]["probability"] *= 1.1
    elif regime_lower == "neutral":
        # Neutral regime: no strong adjustments
        pass
    elif regime_lower in ("bear", "crisis"):
        for name in bearish:
            s[name]["probability"] *= 1.2
        if "Market Correction" in s:
            s["Market Correction"]["probability"] *= 1.2
    elif regime_lower == "volatile":
        if "Market Correction" in s:
            s["Market Correction"]["probability"] *= 1.3
        if "AI Bubble Collapse" in s:
            s["AI Bubble Collapse"]["probability"] *= 1.2

    # Renormalize with FLOOR — no scenario drops below 2%
    MIN_PROB = 0.02
    for v in s.values():
        v["probability"] = max(v["probability"], MIN_PROB)
    total = sum(v["probability"] for v in s.values())
    for v in s.values():
        v["probability"] /= total

    return s
