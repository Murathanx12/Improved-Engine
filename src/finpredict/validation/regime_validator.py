"""Regime Validation — confirms or challenges HMM regime calls.

Every BEAR/Crisis regime call is automatically cross-checked against:
    1. Price structure (SPX vs 200-day SMA)
    2. Market breadth (sector advance/decline)
    3. Institutional consensus (expected returns from major firms)

A BEAR call is only marked "CONFIRMED" when multiple independent signals agree.
This prevents false alarms from HMM label swaps or transient volatility spikes.

Usage:
    from finpredict.validation.regime_validator import validate_regime

    result = validate_regime(data, "Bear", hmm_result)
    print(result.confirmed, result.confidence, result.notes)
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from finpredict.config import config


@dataclass
class RegimeValidation:
    """Result of regime cross-validation checks."""
    regime: str
    confirmed: bool
    price_confirmed: bool
    breadth_confirmed: bool
    consensus_aligned: bool
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    notes: list = field(default_factory=list)


def validate_regime(
    data: pd.DataFrame,
    current_regime: str,
    hmm_result=None,
) -> RegimeValidation:
    """Run three automatic confirmation checks on the current regime call.

    For BULL regimes, checks are inverted (confirm bullish conditions).
    For BEAR/Crisis, all three checks must agree for HIGH confidence.

    Args:
        data: Market DataFrame with SP500, Sector_* columns
        current_regime: Current regime string ("Bull", "Bear", "Crisis", etc.)
        hmm_result: Optional HMM result object (for future use)

    Returns:
        RegimeValidation with confirmation status and notes
    """
    notes = []
    is_bearish = current_regime in ("Bear", "Crisis", "Volatile")

    # ── Check 1: 200-day SMA Price Confirmation ────────────────────
    price_confirmed = _check_price_structure(data, is_bearish, notes)

    # ── Check 2: Market Breadth Confirmation ───────────────────────
    breadth_confirmed = _check_breadth(data, is_bearish, notes)

    # ── Check 3: Institutional Consensus Alignment ─────────────────
    consensus_aligned = _check_consensus(current_regime, notes)

    # ── Determine Overall Confidence ───────────────────────────────
    checks_passed = sum([price_confirmed, breadth_confirmed, consensus_aligned])

    if is_bearish:
        confirmed = checks_passed >= 2
        if checks_passed == 3:
            confidence = "HIGH"
        elif checks_passed == 2:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
            notes.append(
                f"BEAR regime called but only {checks_passed}/3 checks confirm — "
                "treat with caution"
            )
    else:
        # For BULL/Neutral, invert: confirmed if NOT contradicted
        confirmed = checks_passed >= 1
        confidence = "HIGH" if checks_passed >= 2 else "MEDIUM" if checks_passed == 1 else "LOW"

    return RegimeValidation(
        regime=current_regime,
        confirmed=confirmed,
        price_confirmed=price_confirmed,
        breadth_confirmed=breadth_confirmed,
        consensus_aligned=consensus_aligned,
        confidence=confidence,
        notes=notes,
    )


def _check_price_structure(
    data: pd.DataFrame,
    is_bearish: bool,
    notes: list,
) -> bool:
    """Check 1: Is price below (bearish) or above (bullish) the 200-day SMA?"""
    val_cfg = config.get("validation", {}).get("regime_confirmation", {})
    sma_period = val_cfg.get("sma_period", 200)

    sp = data["SP500"]
    if len(sp) < sma_period:
        notes.append("Insufficient data for 200d SMA check")
        return False

    sma_200 = sp.rolling(sma_period).mean()
    current_price = float(sp.iloc[-1])
    current_sma = float(sma_200.iloc[-1])

    if np.isnan(current_sma):
        notes.append("200d SMA not available")
        return False

    pct_from_sma = (current_price - current_sma) / current_sma * 100

    if is_bearish:
        confirmed = current_price < current_sma
        if confirmed:
            notes.append(
                f"Price BELOW 200d SMA ({pct_from_sma:+.1f}%) — "
                "bear signal confirmed by price structure"
            )
        else:
            notes.append(
                f"Price ABOVE 200d SMA ({pct_from_sma:+.1f}%) — "
                "bear signal NOT confirmed by price structure"
            )
    else:
        confirmed = current_price > current_sma
        if confirmed:
            notes.append(
                f"Price ABOVE 200d SMA ({pct_from_sma:+.1f}%) — "
                "bullish structure intact"
            )
        else:
            notes.append(
                f"Price BELOW 200d SMA ({pct_from_sma:+.1f}%) — "
                "bullish call contradicted by price structure"
            )

    return confirmed


def _check_breadth(
    data: pd.DataFrame,
    is_bearish: bool,
    notes: list,
) -> bool:
    """Check 2: Are the majority of sectors declining (bearish) or advancing (bullish)?"""
    val_cfg = config.get("validation", {}).get("regime_confirmation", {})
    breadth_period = val_cfg.get("breadth_period", 21)
    min_declining = val_cfg.get("min_sectors_declining", 6)

    sector_cols = [c for c in data.columns if c.startswith("Sector_")]
    if len(sector_cols) < 3:
        notes.append("Insufficient sector data for breadth check")
        return False

    # Compute sector returns over breadth_period
    n_declining = 0
    n_total = 0
    for col in sector_cols:
        series = data[col].dropna()
        if len(series) < breadth_period + 1:
            continue
        ret = float(series.iloc[-1] / series.iloc[-breadth_period] - 1)
        n_total += 1
        if ret < 0:
            n_declining += 1

    if n_total == 0:
        notes.append("No sector returns available for breadth check")
        return False

    n_advancing = n_total - n_declining

    if is_bearish:
        confirmed = n_declining >= min_declining
        if confirmed:
            notes.append(
                f"Breadth: {n_declining}/{n_total} sectors declining — "
                "broad-based weakness confirms bear"
            )
        else:
            notes.append(
                f"Breadth: only {n_declining}/{n_total} sectors declining — "
                "breadth divergence, bear signal not confirmed by internals"
            )
    else:
        confirmed = n_advancing > n_declining
        if confirmed:
            notes.append(
                f"Breadth: {n_advancing}/{n_total} sectors advancing — "
                "breadth supports bullish call"
            )
        else:
            notes.append(
                f"Breadth: {n_declining}/{n_total} sectors declining — "
                "narrow leadership, bullish call may be fragile"
            )

    return confirmed


def _check_consensus(
    current_regime: str,
    notes: list,
) -> bool:
    """Check 3: Does institutional consensus align with the regime call?"""
    benchmarks = config.get("institutional_benchmarks", {})
    if not benchmarks:
        notes.append("No institutional benchmarks available for consensus check")
        return False

    # Compute weighted mean expected annual return from institutions
    returns = []
    for name, info in benchmarks.items():
        if isinstance(info, dict) and "annual" in info:
            returns.append(float(info["annual"]))

    if not returns:
        notes.append("No valid institutional return forecasts found")
        return False

    consensus_return = np.mean(returns)
    is_bearish = current_regime in ("Bear", "Crisis", "Volatile")

    if is_bearish:
        # Consensus < 3% = aligned with bearish view
        confirmed = consensus_return < 0.03
        if confirmed:
            notes.append(
                f"Consensus annual return {consensus_return*100:.1f}% < 3% — "
                "institutions also cautious, bear aligned"
            )
        else:
            notes.append(
                f"Consensus annual return {consensus_return*100:.1f}% — "
                f"institutions expect positive returns, diverges from BEAR call"
            )
    else:
        # Consensus > 3% = aligned with bullish view
        confirmed = consensus_return >= 0.03
        if confirmed:
            notes.append(
                f"Consensus annual return {consensus_return*100:.1f}% — "
                "institutions bullish, aligned with engine"
            )
        else:
            notes.append(
                f"Consensus annual return {consensus_return*100:.1f}% < 3% — "
                "institutions cautious, diverges from bullish call"
            )

    return confirmed
