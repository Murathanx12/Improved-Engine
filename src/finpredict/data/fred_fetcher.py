"""
FRED Data Integration
======================

Fetches macroeconomic indicators from the Federal Reserve Economic Data API.
These are used for dynamic scenario weighting and crash probability estimation.

Key series:
    - T10Y3M: 10Y-3M Treasury spread (recession predictor)
    - SAHMREALTIME: Sahm Rule recession indicator
    - RECPROUSM156N: Chauvet-Piger recession probability
    - VIXCLS: VIX (longer history than Yahoo)
    - UNRATE: Unemployment rate
    - CPIAUCSL: Consumer Price Index
    - FEDFUNDS: Federal Funds Rate
    - UMCSENT: Consumer Sentiment

Usage:
    from finpredict.data.fred_fetcher import fetch_fred_data

    fred_data = fetch_fred_data()
"""

import pandas as pd
from typing import Optional

from finpredict.config import config, api_keys


def fetch_fred_data() -> dict[str, pd.Series]:
    """
    Fetch all configured FRED series.

    Returns:
        dict mapping series name → pd.Series of values

    Requires FRED_API_KEY in .env
    """
    if not api_keys.has("fred"):
        print("  [WARN] FRED_API_KEY not set, skipping FRED data")
        return {}

    try:
        from fredapi import Fred
    except ImportError:
        print("  [WARN] fredapi not installed, skipping FRED data")
        return {}

    print("[FRED] Fetching macroeconomic indicators...")
    fred = Fred(api_key=api_keys.fred)

    series_ids = config["data"]["fred_series"]
    results = {}

    for name, series_id in series_ids.items():
        try:
            data = fred.get_series(series_id)
            if data is not None and len(data) > 0:
                results[name] = data.dropna()
                latest = float(data.dropna().iloc[-1])
                print(f"  [OK] {name} ({series_id}): {latest:.2f}")
        except Exception as e:
            print(f"  [WARN] Failed to fetch {name} ({series_id}): {e}")

    print(f"  [OK] {len(results)}/{len(series_ids)} FRED series loaded\n")
    return results


def get_recession_probability(fred_data: dict) -> float:
    """
    Compute a blended recession probability from FRED indicators.

    Uses three signals:
        1. Yield curve spread (T10Y3M) — inverted = recession warning
        2. Sahm Rule — triggers at +0.50 percentage points
        3. Chauvet-Piger smoothed probability — direct model output

    Returns:
        float between 0 and 1 representing recession probability
    """
    signals = []

    # 1. Yield curve: inverted = high recession risk
    if "yield_spread" in fred_data:
        spread = float(fred_data["yield_spread"].iloc[-1])
        # Logistic mapping: spread of 0 → 50%, spread of -1 → 85%, spread of +2 → 10%
        yield_prob = 1 / (1 + _exp_safe(2.0 * spread))
        signals.append(("yield_curve", yield_prob, 0.35))

    # 2. Sahm Rule: triggers at 0.50+
    if "sahm_rule" in fred_data:
        sahm = float(fred_data["sahm_rule"].iloc[-1])
        # Hard trigger at 0.50, gradual ramp from 0 to 0.50
        if sahm >= 0.50:
            sahm_prob = 0.90
        else:
            sahm_prob = sahm / 0.50 * 0.50  # Linear ramp to 50%
        signals.append(("sahm_rule", sahm_prob, 0.30))

    # 3. Chauvet-Piger model probability (already 0-100)
    if "recession_prob" in fred_data:
        cp = float(fred_data["recession_prob"].iloc[-1])
        signals.append(("chauvet_piger", cp / 100.0, 0.35))

    if not signals:
        return 0.15  # Default: 15% base rate

    # Weighted average
    total_weight = sum(w for _, _, w in signals)
    blended = sum(prob * w for _, prob, w in signals) / total_weight

    return float(min(0.95, max(0.02, blended)))


def get_macro_features(fred_data: dict) -> dict:
    """
    Extract current macro features for scenario weighting.

    Returns dict with standardized macro indicators.
    """
    features = {}

    if "yield_spread" in fred_data:
        features["yield_spread"] = float(fred_data["yield_spread"].iloc[-1])

    if "sahm_rule" in fred_data:
        features["sahm_rule"] = float(fred_data["sahm_rule"].iloc[-1])

    if "unemployment" in fred_data:
        features["unemployment"] = float(fred_data["unemployment"].iloc[-1])

    if "fed_funds" in fred_data:
        features["fed_funds"] = float(fred_data["fed_funds"].iloc[-1])

    if "consumer_sentiment" in fred_data:
        features["consumer_sentiment"] = float(fred_data["consumer_sentiment"].iloc[-1])

    if "vix_fred" in fred_data:
        features["vix"] = float(fred_data["vix_fred"].iloc[-1])

    return features


# Actually use proper math.exp
import math

def _exp_safe(x: float) -> float:
    """Safe exponential that avoids overflow."""
    return math.exp(min(700, max(-700, x)))
