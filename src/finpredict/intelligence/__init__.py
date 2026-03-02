"""
OSINT Intelligence Layer — Event-Driven Risk Signals
======================================================

This module provides exogenous risk signals that the price-based ML models
cannot see: geopolitical events, news sentiment shifts, and crisis indicators.

Architecture:
    Layer 1 (existing): Market data → GARCH, HMM, ML → price-based risk
    Layer 2 (existing): FRED macro → recession probability, credit stress
    Layer 3 (THIS):     OSINT/news → geopolitical risk, sentiment cascades

The key insight: crashes are NOT predicted by price patterns alone.
They are predicted by SIGNAL CONVERGENCE — when multiple independent
indicators (credit stress + geopolitical tension + sentiment collapse)
all deteriorate simultaneously.

Data Sources:
    - GDELT (Global Database of Events, Language, Tone): Free, updated every 15 min
    - FRED GPR Index: Geopolitical risk from newspaper word counts (already in config)
    - News tone aggregation: Average sentiment from global news coverage

Usage:
    from finpredict.intelligence import fetch_gdelt_data, compute_event_score

    gdelt_data = fetch_gdelt_data()
    event_score = compute_event_score(gdelt_data, fred_macro_features)
"""

from finpredict.intelligence.gdelt_fetcher import fetch_gdelt_data
from finpredict.intelligence.event_scorer import compute_event_score

__all__ = ["fetch_gdelt_data", "compute_event_score"]
