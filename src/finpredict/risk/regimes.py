"""
Module 2: Market Regime Detection
==================================

Classifies each trading day into Bull/Bear/Volatile using a rolling window
of returns + volatility PLUS leading indicators (VIX, risk score) to detect
regime transitions earlier than pure price-based classification.

Current approach: Rule-based with leading indicator overlay.
Future (Phase II-C): HMM-based regime detection using hmmlearn.

Usage:
    from finpredict.risk.regimes import detect_regimes

    data['Regime'], current_regime = detect_regimes(data)
"""

import numpy as np
import pandas as pd

from finpredict.config import config


def detect_regimes(data: pd.DataFrame, window: int = 252) -> tuple[pd.Series, str]:
    """
    Module 2 Entry Point: Classify each day into Bull/Bear/Volatile.

    Uses rolling returns + volatility with leading indicator overlays
    (VIX, risk score) for early transition detection.

    Args:
        data: DataFrame with 'SP500' column (and optionally 'VIX', 'Risk_Score')
        window: Rolling window size (default 252 = 1 year)

    Returns:
        regimes: pd.Series of regime labels
        current: Current regime string
    """
    print("[MODULE 2] Detecting market regimes...")

    thresholds = config["risk"]["regimes"]
    returns = data["SP500"].pct_change()
    # Keep the same index as `returns` so label-based slicing stays aligned.
    # dropna() would shorten the series and cause off-by-one errors when using
    # positional .iloc on both series simultaneously.
    log_returns = np.log(1 + returns).replace([np.inf, -np.inf], np.nan)
    regimes = pd.Series(index=data.index, dtype=str, data="")

    has_vix = "VIX" in data.columns
    has_risk = "Risk_Score" in data.columns

    for i in range(window, len(returns)):
        # Slice by label so both series cover exactly the same calendar window
        date_window = returns.index[max(0, i - window):i]
        w = log_returns.loc[date_window].dropna()
        if len(w) < 60:
            continue

        ann_ret = w.mean() * 252    # Geometric annualized return
        ann_vol = returns.loc[date_window].std() * np.sqrt(252)

        # Base classification (price-based)
        if ann_vol > thresholds["high_vol_threshold"]:
            base_regime = "Volatile"
        elif ann_ret > thresholds["bull_return_threshold"]:
            base_regime = "Bull"
        else:
            base_regime = "Bear"

        # Leading indicator overlay: detect regime TRANSITIONS early
        vix_now = (
            float(data["VIX"].iloc[i])
            if has_vix and pd.notna(data["VIX"].iloc[i])
            else None
        )
        risk_now = (
            float(data["Risk_Score"].iloc[i])
            if has_risk and pd.notna(data["Risk_Score"].iloc[i])
            else None
        )

        # Override: Bull → Volatile if stress signals are flashing
        if base_regime == "Bull":
            stress_signals = 0
            if vix_now is not None and vix_now > thresholds["vix_stress_threshold"]:
                stress_signals += 1
            if risk_now is not None and risk_now > thresholds["risk_stress_threshold"]:
                stress_signals += 1
            if stress_signals >= 2:
                base_regime = "Volatile"

        # Override: Bear → Bull if stress signals are very low (recovery detection)
        elif base_regime == "Bear":
            if (vix_now is not None and vix_now < thresholds["vix_calm_threshold"]
                    and risk_now is not None
                    and risk_now < thresholds["risk_calm_threshold"]):
                if ann_vol < 0.20:
                    base_regime = "Bull"

        regimes.iloc[i] = base_regime

    current = regimes.iloc[-1] if regimes.iloc[-1] else "Unknown"
    print(f"  [OK] Current regime: {current}\n")
    return regimes, current
