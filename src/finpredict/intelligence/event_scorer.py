"""
Event Risk Scorer — Composite Exogenous Risk Assessment
=========================================================

Combines signals from multiple intelligence sources into a single
event risk score that modifies the engine's crash probability.

Signal sources:
    1. GDELT news tone (40% weight): deteriorating tone = stress
    2. GDELT volume spikes (30% weight): unusual activity = developing crisis
    3. FRED GPR Index (30% weight): geopolitical risk from newspaper coverage

The event score is designed to:
    - Be ADDITIVE to the ML crash probability (not replace it)
    - Activate primarily during CONVERGENCE (multiple signals elevated)
    - Have minimal impact during normal conditions (avoid false alarms)

Integration points:
    - Crash probability: crash_prob_adjusted = base_prob * (1 + event_adjustment)
    - Scenario weights: boost geopolitical crisis scenario when score is high
    - MC simulation: inject additional jump probability during elevated risk
"""

import numpy as np


def compute_event_score(
    gdelt_data: dict,
    fred_macro: dict = None,
) -> dict:
    """Compute composite event risk score from intelligence signals.

    Args:
        gdelt_data: Output from fetch_gdelt_data()
        fred_macro: Optional dict with FRED macro features (may include 'gpr_world')

    Returns:
        dict with:
            - event_score: float [0, 1], composite risk score
            - crash_adjustment: float, multiplicative adjustment to crash probability
            - regime_override: str or None, force regime if score is extreme
            - components: dict of individual signal scores
            - interpretation: str, human-readable risk assessment
    """
    components = {}

    # ── 1. News Tone Signal (40% weight) ──────────────────────────
    tone_score = 0.0
    if gdelt_data.get("success"):
        avg_tone = gdelt_data.get("avg_tone", 0.0)
        tone_trend = gdelt_data.get("tone_trend", 0.0)

        # GDELT tone ranges roughly from -10 (very negative) to +10 (very positive)
        # Financial news typically ranges -3 to +3
        # Map to 0-1 score where 1 = extreme negative
        tone_score = np.clip((-avg_tone + 1) / 4, 0, 1)  # Centered, scaled

        # Deteriorating trend amplifies the signal
        if tone_trend < -0.5:
            tone_score = min(1.0, tone_score * 1.3)

    components["tone"] = float(tone_score)

    # ── 2. Volume Spike Signal (30% weight) ───────────────────────
    volume_score = 0.0
    if gdelt_data.get("success"):
        volume_z = gdelt_data.get("volume_zscore", 0.0)
        # z > 2 = unusual volume (something happening)
        # z > 3 = extreme volume (crisis developing)
        volume_score = np.clip((volume_z - 1) / 3, 0, 1)

    components["volume"] = float(volume_score)

    # ── 3. Geopolitical Risk Signal (30% weight) ──────────────────
    gpr_score = 0.0
    if fred_macro:
        gpr_value = fred_macro.get("gpr_world", 0)
        if isinstance(gpr_value, (int, float)) and gpr_value > 0:
            # GPR index: historical average ~100, spikes to 200-400 during crises
            gpr_score = np.clip((gpr_value - 100) / 200, 0, 1)

    components["gpr"] = float(gpr_score)

    # ── Composite score (weighted average) ────────────────────────
    weights = {"tone": 0.40, "volume": 0.30, "gpr": 0.30}
    event_score = sum(components[k] * weights[k] for k in weights)
    event_score = float(np.clip(event_score, 0, 1))

    # ── Convergence bonus: when multiple signals fire simultaneously ──
    # If 2+ signals are elevated (>0.4), boost the combined score
    n_elevated = sum(1 for v in components.values() if v > 0.4)
    if n_elevated >= 2:
        convergence_boost = 0.15 * (n_elevated - 1)
        event_score = min(1.0, event_score + convergence_boost)

    # ── Derive crash adjustment ────────────────────────────────────
    # Low scores (0-0.2): no adjustment
    # Medium scores (0.2-0.5): moderate boost to crash probability
    # High scores (0.5-1.0): significant boost
    if event_score < 0.2:
        crash_adjustment = 0.0
    elif event_score < 0.5:
        crash_adjustment = (event_score - 0.2) * 1.5  # Up to +0.45
    else:
        crash_adjustment = 0.45 + (event_score - 0.5) * 2.0  # Up to +1.45

    # ── Regime override (only for extreme scores) ──────────────────
    regime_override = None
    if event_score > 0.75:
        regime_override = "Crisis"

    # ── Interpretation ─────────────────────────────────────────────
    if event_score < 0.15:
        interpretation = "Low exogenous risk — normal conditions"
    elif event_score < 0.35:
        interpretation = "Moderate exogenous risk — heightened monitoring"
    elif event_score < 0.55:
        interpretation = "Elevated exogenous risk — multiple stress signals active"
    elif event_score < 0.75:
        interpretation = "High exogenous risk — significant event-driven stress"
    else:
        interpretation = "CRITICAL exogenous risk — crisis-level signal convergence"

    return {
        "event_score": event_score,
        "crash_adjustment": float(crash_adjustment),
        "regime_override": regime_override,
        "components": components,
        "interpretation": interpretation,
    }


def adjust_crash_probability(
    base_crash_prob: float,
    event_result: dict,
    max_adjustment: float = 0.30,
) -> float:
    """Apply event-based adjustment to ML crash probability.

    The adjustment is capped to prevent the event layer from completely
    overriding the ML model. The ML model remains the primary predictor;
    the event layer provides early warning for exogenous shocks.

    Args:
        base_crash_prob: ML-predicted crash probability [0, 1]
        event_result: Output from compute_event_score()
        max_adjustment: Maximum additive adjustment to crash probability

    Returns:
        Adjusted crash probability, clipped to [0.02, 0.98]
    """
    raw_adj = event_result.get("crash_adjustment", 0.0) * base_crash_prob
    capped_adj = min(raw_adj, max_adjustment)
    adjusted = base_crash_prob + capped_adj
    return float(np.clip(adjusted, 0.02, 0.98))
