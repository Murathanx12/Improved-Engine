"""
GDELT Data Fetcher — Global News Tone & Event Monitoring
==========================================================

GDELT (Global Database of Events, Language, and Tone) is the world's largest
open news database. It monitors news from virtually every country in 100+
languages and updates every 15 minutes.

For crash prediction, we extract:
    1. Average news tone (Goldstein Scale): negative tone = geopolitical stress
    2. News volume spikes: sudden increases indicate developing crises
    3. Conflict event counts: military actions, protests, diplomatic crises

The GDELT API is completely free and requires no API key.
Endpoint: https://api.gdeltproject.org/api/v2/doc/doc

Historical note on signal value:
    - Before 2008: Financial news tone deteriorated 3-4 weeks before Lehman
    - Before COVID: News volume spiked 2 weeks before market crash
    - Before Ukraine: Geopolitical event counts spiked months before invasion
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


def fetch_gdelt_data(
    lookback_days: int = 90,
    query: str = "economy OR financial OR market OR recession OR crisis",
) -> dict:
    """Fetch recent news tone and volume data from GDELT API.

    Args:
        lookback_days: How many days of history to fetch (max ~90 for free API)
        query: Search query to filter relevant news articles

    Returns:
        dict with keys:
            - avg_tone: float, average news tone (negative = pessimistic)
            - tone_trend: float, tone change over period (negative = deteriorating)
            - volume_zscore: float, current news volume vs historical (>2 = spike)
            - conflict_score: float, normalized conflict event intensity
            - raw_data: dict of raw API response data
            - success: bool
    """
    if not _HAS_REQUESTS:
        return _empty_result("requests library not installed")

    try:
        # GDELT DOC API — fetch tone timeline for financial/crisis news
        tone_data = _fetch_tone_timeline(query, lookback_days)
        volume_data = _fetch_volume_timeline(query, lookback_days)

        if not tone_data["success"] or not volume_data["success"]:
            return _empty_result("GDELT API returned no data")

        # Compute aggregate metrics
        tones = tone_data["daily_tones"]
        volumes = volume_data["daily_volumes"]

        avg_tone = float(np.mean(tones[-7:])) if len(tones) >= 7 else 0.0
        tone_30d = float(np.mean(tones[-30:])) if len(tones) >= 30 else avg_tone
        tone_trend = avg_tone - tone_30d  # Negative = deteriorating

        # Volume z-score: is current volume unusual?
        if len(volumes) > 14:
            vol_mean = np.mean(volumes[:-7])
            vol_std = np.std(volumes[:-7])
            current_vol = np.mean(volumes[-7:])
            volume_zscore = (current_vol - vol_mean) / max(vol_std, 1.0)
        else:
            volume_zscore = 0.0

        # Conflict event score: normalize recent conflict news volume vs baseline
        conflict_data = _fetch_conflict_timeline(lookback_days)
        conflict_score = _normalize_conflict(conflict_data)

        return {
            "avg_tone": avg_tone,
            "tone_trend": tone_trend,
            "volume_zscore": float(volume_zscore),
            "conflict_score": float(conflict_score),
            "raw_data": {
                "tones": tones[-30:] if len(tones) > 30 else tones,
                "volumes": volumes[-30:] if len(volumes) > 30 else volumes,
                "conflict_counts": conflict_data.get("daily_counts", [])[-30:],
            },
            "success": True,
        }

    except Exception as e:
        return _empty_result(f"GDELT fetch failed: {e}")


def _fetch_tone_timeline(query: str, days: int) -> dict:
    """Fetch daily average tone from GDELT DOC API."""
    end_date = datetime.now().strftime("%Y%m%d%H%M%S")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d%H%M%S")

    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "timelinetone",
        "timelinesmooth": 5,
        "startdatetime": start_date,
        "enddatetime": end_date,
        "format": "json",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return {"success": False, "daily_tones": []}

        data = resp.json()
        timeline = data.get("timeline", [])
        if not timeline:
            return {"success": False, "daily_tones": []}

        # Extract tone values from timeline series
        series = timeline[0].get("data", []) if timeline else []
        tones = [float(entry.get("value", 0)) for entry in series]

        return {"success": True, "daily_tones": tones}
    except Exception:
        return {"success": False, "daily_tones": []}


def _fetch_volume_timeline(query: str, days: int) -> dict:
    """Fetch daily article volume from GDELT DOC API."""
    end_date = datetime.now().strftime("%Y%m%d%H%M%S")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d%H%M%S")

    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "timelinevolraw",
        "startdatetime": start_date,
        "enddatetime": end_date,
        "format": "json",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return {"success": False, "daily_volumes": []}

        data = resp.json()
        timeline = data.get("timeline", [])
        if not timeline:
            return {"success": False, "daily_volumes": []}

        series = timeline[0].get("data", []) if timeline else []
        volumes = [float(entry.get("value", 0)) for entry in series]

        return {"success": True, "daily_volumes": volumes}
    except Exception:
        return {"success": False, "daily_volumes": []}


def _fetch_conflict_timeline(days: int) -> dict:
    """Fetch daily conflict/war/crisis article counts from GDELT DOC API."""
    end_date = datetime.now().strftime("%Y%m%d%H%M%S")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d%H%M%S")

    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": "conflict OR war OR military OR geopolitical OR sanctions OR crisis",
        "mode": "timelinevolraw",
        "startdatetime": start_date,
        "enddatetime": end_date,
        "format": "json",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return {"success": False, "daily_counts": []}

        data = resp.json()
        timeline = data.get("timeline", [])
        if not timeline:
            return {"success": False, "daily_counts": []}

        series = timeline[0].get("data", []) if timeline else []
        counts = [float(entry.get("value", 0)) for entry in series]
        return {"success": True, "daily_counts": counts}
    except Exception:
        return {"success": False, "daily_counts": []}


def _normalize_conflict(conflict_data: dict) -> float:
    """Map raw conflict article counts to a [0, 1] score.

    Compares the most recent 7-day average against the 90-day baseline.
    Returns 0.0 when recent conflict coverage is at or below the baseline,
    and approaches 1.0 when it is several standard deviations above.
    Falls back to 0.0 when data is unavailable.
    """
    if not conflict_data.get("success"):
        return 0.0

    counts = conflict_data.get("daily_counts", [])
    if len(counts) < 14:
        return 0.0

    recent = np.mean(counts[-7:])
    baseline = np.mean(counts[:-7])
    std = np.std(counts[:-7])

    if std < 1.0:
        return 0.0

    # Sigmoid centred at 0 z-scores → 0.5, at +2 z-scores → ~0.88
    z = (recent - baseline) / std
    score = 1.0 / (1.0 + np.exp(-z))
    # Shift so that z=0 (no anomaly) maps to 0 rather than 0.5
    score = max(0.0, (score - 0.5) * 2.0)
    return float(min(score, 1.0))


def _empty_result(reason: str = "") -> dict:
    """Return empty result when GDELT data is unavailable."""
    return {
        "avg_tone": 0.0,
        "tone_trend": 0.0,
        "volume_zscore": 0.0,
        "conflict_score": 0.0,
        "raw_data": {},
        "success": False,
        "reason": reason,
    }
