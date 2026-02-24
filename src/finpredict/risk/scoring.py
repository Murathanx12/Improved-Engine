"""
Module 3: Composite Risk Scoring
=================================

Builds a single risk score from 9 market indicators. This score drives
crash probability estimation and scenario probability adjustment.

Indicators (with weights from engine_config.yaml):
    1. VIX z-score              — Fear gauge
    2. Yield curve inversion    — Recession predictor (10Y - 3M)
    3. Credit spread stress     — Corporate debt fear (LQD/HYG)
    4. Long yield volatility    — Bond market disruption
    5. Momentum exhaustion      — Extreme moves (>2σ from mean)
    6. Short-term vol regime    — 20-day rolling vol z-score
    7. Gold/stock ratio change  — Flight to safety signal
    8. Market breadth            — Narrow leadership (NASDAQ/S&P)
    9. Small cap divergence     — Russell 2000 vs S&P 500

Output:
    A z-score clipped to [-4, +4]. Higher = more risk.
    Values > 2.0 historically preceded crashes within 6-12 months.

Usage:
    from finpredict.risk.scoring import build_risk_score

    data['Risk_Score'] = build_risk_score(data)
"""

import numpy as np
import pandas as pd

from finpredict.config import config


# ── Helpers ──────────────────────────────────────────────────────────────────────────

def rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """Calculate rolling z-score, clipped to [-5, +5]."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    z = (series - mean) / std
    return z.clip(-5, 5)


# ── Main Risk Score Builder ──────────────────────────────────────────────────────────

def build_risk_score(data: pd.DataFrame) -> pd.Series:
    """
    Module 3 Entry Point: Compute 9-factor composite risk indicator.

    Args:
        data: Master DataFrame with market data columns

    Returns:
        pd.Series — the composite risk score (z-score, clipped [-4, +4])
    """
    print("[MODULE 3] Computing 9-factor composite risk score...")

    weights_cfg = config["risk"]["indicator_weights"]
    signals, weights = [], []

    # 1. VIX (Fear Index)
    if "VIX" in data.columns:
        signals.append(rolling_zscore(data["VIX"], 252))
        weights.append(weights_cfg["vix"])

    # 2. Yield curve inversion (10Y - 3M) — NY Fed's preferred recession indicator
    if "T10Y" in data.columns and "T3M" in data.columns:
        curve = data["T10Y"] - data["T3M"]
        signals.append(-rolling_zscore(curve, 252))  # Negative = inverted = risky
        weights.append(weights_cfg["yield_curve"])

    # 3. Credit spread (LQD/HYG ratio)
    if "HYG" in data.columns and "LQD" in data.columns:
        spread = data["LQD"] / data["HYG"]
        signals.append(rolling_zscore(spread, 252))
        weights.append(weights_cfg["credit_spread"])

    # 4. Long yield volatility (30Y rate change)
    if "T30Y" in data.columns:
        yld_chg = data["T30Y"].pct_change(60)
        signals.append(rolling_zscore(yld_chg, 252).abs())
        weights.append(weights_cfg["long_yield_vol"])

    # 5. Momentum exhaustion
    ret_60d = data["SP500"].pct_change(60)
    mom_z = rolling_zscore(ret_60d, 252)
    signals.append(mom_z.apply(lambda x: max(0, abs(x) - 2.0)))
    weights.append(weights_cfg["momentum_exhaustion"])

    # 6. Short-term vol regime (20d rolling vol)
    daily_ret = data["SP500"].pct_change()
    vol_20d = daily_ret.rolling(20).std() * np.sqrt(252)
    signals.append(rolling_zscore(vol_20d, 252))
    weights.append(weights_cfg["short_term_vol"])

    # 7. Gold/Stock ratio
    if "Gold" in data.columns:
        ratio = data["Gold"] / data["SP500"]
        signals.append(rolling_zscore(ratio.pct_change(60), 252))
        weights.append(weights_cfg["gold_stock_ratio"])

    # 8. Market breadth (NASDAQ / S&P 500)
    if "NASDAQ" in data.columns:
        breadth = data["NASDAQ"] / data["SP500"]
        signals.append(rolling_zscore(breadth.pct_change(60), 252).abs())
        weights.append(weights_cfg["market_breadth"])

    # 9. Small cap divergence (Russell / S&P 500)
    if "Russell" in data.columns:
        divergence = data["Russell"] / data["SP500"]
        signals.append(-rolling_zscore(divergence.pct_change(60), 252))
        weights.append(weights_cfg["small_cap_divergence"])

    if not signals:
        return pd.Series(0, index=data.index)

    total_w = sum(weights)
    composite = sum(s * w for s, w in zip(signals, weights)) / total_w
    score = composite.clip(-4, 4)

    current = score.iloc[-1]
    print(f"  [OK] Current risk score: {current:.2f}σ")
    print(f"  [OK] Indicators used: {len(signals)}/9\n")

    return score
