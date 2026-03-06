"""External Validation — cross-checks engine predictions against independent sources.

Compares the engine's bull/bear call and crash probability against:
    1. Conference Board Leading Economic Index (LEI) — via FRED
    2. Senior Loan Officer Survey (SLOOS) — via FRED
    3. Fed Funds Rate direction — via FRED
    4. Consumer Sentiment (UMCSENT) — via FRED (proxy for AAII)
    5. IMF GDP Forecasts — via IMF Data API

The engine should be able to disagree with consensus and be right.
But when it IS a significant outlier, that divergence is flagged
explicitly so the user knows they are acting against consensus.

Usage:
    from finpredict.validation.external_validator import validate_external

    result = validate_external(fred_data, ml_crash_prob, "Bear")
    print(result.engine_agreement, result.divergence_alerts)
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

import numpy as np
import pandas as pd

from finpredict.config import config


@dataclass
class ExternalValidation:
    """Result of external validation checks."""
    lei_signal: str = "UNKNOWN"          # "EXPANSION" / "WARNING" / "RECESSION"
    sloos_signal: str = "UNKNOWN"        # "EASING" / "NEUTRAL" / "TIGHTENING"
    fed_signal: str = "UNKNOWN"          # "HAWKISH" / "NEUTRAL" / "DOVISH"
    sentiment_signal: str = "UNKNOWN"    # "EXTREME_FEAR" / "FEAR" / "NEUTRAL" / "GREED"
    imf_signal: str = "UNKNOWN"          # "GROWTH" / "SLOWDOWN" / "CONTRACTION"
    fed_futures_signal: str = "UNKNOWN"  # "CUT_EXPECTED" / "HOLD" / "HIKE_EXPECTED"
    consensus_direction: str = "UNKNOWN" # "BULLISH" / "NEUTRAL" / "BEARISH"
    engine_agreement: float = 0.0        # 0-1: fraction of signals that agree
    divergence_alerts: list = field(default_factory=list)


def validate_external(
    fred_data: dict,
    ml_crash_prob: Optional[float],
    current_regime: str,
    alt_data: Optional[dict] = None,
) -> ExternalValidation:
    """Cross-check engine outputs against external data sources.

    Args:
        fred_data: Dict of FRED time series (from fred_fetcher)
        ml_crash_prob: Engine's current 12m crash probability
        current_regime: Engine's current regime call
        alt_data: Optional dict with keys "aaii", "naaim", "imf"

    Returns:
        ExternalValidation with per-source signals and agreement score
    """
    if alt_data is None:
        alt_data = {}

    val_cfg = config.get("validation", {}).get("external_sources", {})
    result = ExternalValidation()
    is_engine_bearish = current_regime in ("Bear", "Crisis", "Volatile") or (
        ml_crash_prob is not None and ml_crash_prob > 0.50
    )

    signals_agree = 0
    signals_total = 0

    # ── 1. Conference Board LEI ─────────────────────────────────────
    if fred_data and "lei" in fred_data:
        result.lei_signal = _assess_lei(fred_data["lei"], val_cfg)
        signals_total += 1
        lei_bearish = result.lei_signal in ("WARNING", "RECESSION")
        if lei_bearish == is_engine_bearish:
            signals_agree += 1
        elif result.lei_signal == "RECESSION" and not is_engine_bearish:
            result.divergence_alerts.append(
                f"LEI signals RECESSION but engine says {current_regime} — "
                "engine may be underestimating risk"
            )
        elif result.lei_signal == "EXPANSION" and is_engine_bearish:
            result.divergence_alerts.append(
                f"LEI signals EXPANSION but engine says {current_regime} — "
                "leading indicators do not confirm bearish view"
            )

    # ── 2. SLOOS (Senior Loan Officer Survey) ──────────────────────
    if fred_data and "sloos_ci" in fred_data:
        result.sloos_signal = _assess_sloos(fred_data["sloos_ci"], val_cfg)
        signals_total += 1
        sloos_bearish = result.sloos_signal == "TIGHTENING"
        if sloos_bearish == is_engine_bearish:
            signals_agree += 1
        elif result.sloos_signal == "TIGHTENING" and not is_engine_bearish:
            result.divergence_alerts.append(
                "Banks tightening lending (SLOOS) — "
                "credit conditions deteriorating, engine may be too optimistic"
            )

    # ── 3. Fed Funds Rate Direction ────────────────────────────────
    if fred_data and "fed_funds" in fred_data:
        result.fed_signal = _assess_fed(fred_data["fed_funds"])
        signals_total += 1
        # Rising rates = headwind for stocks (bearish-aligned)
        # But also: cutting rates aggressively can mean crisis response
        fed_bearish = result.fed_signal == "HAWKISH"
        if fed_bearish == is_engine_bearish:
            signals_agree += 1
        elif result.fed_signal == "DOVISH" and is_engine_bearish:
            result.divergence_alerts.append(
                "Fed cutting rates (dovish) — "
                "monetary policy turning supportive, may conflict with bearish call"
            )

    # ── 4. Consumer Sentiment (proxy for retail sentiment) ─────────
    if fred_data and "consumer_sentiment" in fred_data:
        result.sentiment_signal = _assess_sentiment(
            fred_data["consumer_sentiment"], val_cfg
        )
        signals_total += 1
        # Extreme fear is contrarian-BULLISH (bearish sentiment = bullish signal)
        sentiment_bearish = result.sentiment_signal in ("NEUTRAL", "GREED")
        if sentiment_bearish == is_engine_bearish:
            signals_agree += 1
        elif result.sentiment_signal == "EXTREME_FEAR" and is_engine_bearish:
            result.divergence_alerts.append(
                f"Consumer sentiment at EXTREME FEAR — "
                "historically contrarian-bullish, engine's bearish call "
                "may be at a sentiment bottom"
            )

    # ── 5. IMF GDP Forecast ────────────────────────────────────────
    imf_data = alt_data.get("imf")
    if imf_data:
        result.imf_signal = _assess_imf(imf_data, val_cfg)
        signals_total += 1
        imf_bearish = result.imf_signal in ("SLOWDOWN", "CONTRACTION")
        if imf_bearish == is_engine_bearish:
            signals_agree += 1
        elif result.imf_signal == "CONTRACTION" and not is_engine_bearish:
            result.divergence_alerts.append(
                "IMF projects GDP contraction — engine may be too optimistic"
            )

    # ── 6. Fed Funds Futures (market-implied rate expectations) ────
    fed_futures_data = alt_data.get("fed_futures")
    if fed_futures_data:
        result.fed_futures_signal = fed_futures_data.get("direction", "UNKNOWN")
        signals_total += 1
        # Rate cuts expected = dovish = near-term bullish for equities
        # Rate hikes expected = hawkish = headwind for equities
        futures_bearish = result.fed_futures_signal == "HIKE_EXPECTED"
        if futures_bearish == is_engine_bearish:
            signals_agree += 1
        elif result.fed_futures_signal == "CUT_EXPECTED" and is_engine_bearish:
            result.divergence_alerts.append(
                f"Fed futures imply rate CUTS (prob={fed_futures_data.get('rate_cut_probability', 0):.0%}) — "
                "monetary easing expected, may conflict with bearish call"
            )
        elif result.fed_futures_signal == "HIKE_EXPECTED" and not is_engine_bearish:
            result.divergence_alerts.append(
                "Fed futures imply rate HIKES — "
                "monetary tightening expected, engine may be too optimistic"
            )

    # ── Compute Agreement Score ────────────────────────────────────
    if signals_total > 0:
        result.engine_agreement = signals_agree / signals_total
    else:
        result.engine_agreement = 0.0

    # Determine consensus direction
    bearish_count = sum(1 for s in [
        result.lei_signal in ("WARNING", "RECESSION"),
        result.sloos_signal == "TIGHTENING",
        result.fed_signal == "HAWKISH",
        result.sentiment_signal in ("NEUTRAL", "GREED"),  # Not fearful = complacent
        result.imf_signal in ("SLOWDOWN", "CONTRACTION"),
        result.fed_futures_signal == "HIKE_EXPECTED",
    ] if s)
    if bearish_count >= 4:
        result.consensus_direction = "BEARISH"
    elif bearish_count <= 1:
        result.consensus_direction = "BULLISH"
    else:
        result.consensus_direction = "NEUTRAL"

    return result


def _assess_lei(lei_series: pd.Series, cfg: dict) -> str:
    """Assess Leading Economic Index signal.

    Consecutive monthly declines in LEI reliably precede recessions.
    ≥3 months = WARNING, ≥6 months = RECESSION signal.
    """
    s = pd.Series(lei_series).dropna()
    if len(s) < 7:
        return "UNKNOWN"

    warning_threshold = cfg.get("lei_decline_warning", 3)
    recession_threshold = cfg.get("lei_decline_recession", 6)

    # Count consecutive monthly declines (LEI is monthly)
    monthly = s.resample("MS").last().dropna()
    if len(monthly) < 2:
        return "UNKNOWN"

    consecutive_declines = 0
    for i in range(len(monthly) - 1, 0, -1):
        if monthly.iloc[i] < monthly.iloc[i - 1]:
            consecutive_declines += 1
        else:
            break

    if consecutive_declines >= recession_threshold:
        return "RECESSION"
    elif consecutive_declines >= warning_threshold:
        return "WARNING"
    else:
        return "EXPANSION"


def _assess_sloos(sloos_series: pd.Series, cfg: dict) -> str:
    """Assess Senior Loan Officer Survey signal.

    Net percentage of banks tightening standards.
    > threshold = TIGHTENING, < -threshold = EASING, else NEUTRAL.
    """
    s = pd.Series(sloos_series).dropna()
    if len(s) == 0:
        return "UNKNOWN"

    latest = float(s.iloc[-1])
    threshold = cfg.get("sloos_tightening_threshold", 20)

    if latest > threshold:
        return "TIGHTENING"
    elif latest < -threshold:
        return "EASING"
    else:
        return "NEUTRAL"


def _assess_fed(fed_funds_series: pd.Series) -> str:
    """Assess Fed Funds Rate direction.

    Compare current rate to 12 months ago.
    Rising = HAWKISH, flat = NEUTRAL, falling = DOVISH.
    """
    s = pd.Series(fed_funds_series).dropna()
    if len(s) < 252:
        return "UNKNOWN"

    current = float(s.iloc[-1])
    year_ago = float(s.iloc[-252]) if len(s) >= 252 else float(s.iloc[0])

    change = current - year_ago
    if change > 0.25:
        return "HAWKISH"
    elif change < -0.25:
        return "DOVISH"
    else:
        return "NEUTRAL"


def _assess_sentiment(sentiment_series: pd.Series, cfg: dict) -> str:
    """Assess consumer sentiment signal.

    UMCSENT < extreme_low = EXTREME_FEAR (historically contrarian-bullish).
    """
    s = pd.Series(sentiment_series).dropna()
    if len(s) == 0:
        return "UNKNOWN"

    latest = float(s.iloc[-1])
    extreme_low = cfg.get("sentiment_extreme_low", 60)

    if latest < extreme_low:
        return "EXTREME_FEAR"
    elif latest < 80:
        return "FEAR"
    elif latest < 100:
        return "NEUTRAL"
    else:
        return "GREED"


def _assess_imf(imf_data: dict, cfg: dict) -> str:
    """Assess IMF GDP growth forecast signal.

    GDP < warning threshold = CONTRACTION concern.
    """
    if not imf_data:
        return "UNKNOWN"

    current_year = datetime.now().year
    # Look for current or next year forecast
    gdp = imf_data.get(current_year) or imf_data.get(current_year + 1)
    if gdp is None:
        # Try the latest available year
        latest_year = max(imf_data.keys()) if imf_data else None
        if latest_year:
            gdp = imf_data[latest_year]

    if gdp is None:
        return "UNKNOWN"

    warning = cfg.get("imf_gdp_warning", 1.5)

    if gdp < 0:
        return "CONTRACTION"
    elif gdp < warning:
        return "SLOWDOWN"
    else:
        return "GROWTH"
