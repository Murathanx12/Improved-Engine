"""Tests for the validation layer (regime + external validators)."""

import numpy as np
import pandas as pd
import pytest

from finpredict.validation.regime_validator import validate_regime, RegimeValidation
from finpredict.validation.external_validator import validate_external, ExternalValidation


@pytest.fixture
def market_data_bull():
    """Synthetic data where price is above 200d SMA and most sectors are up."""
    rng = np.random.default_rng(42)
    n = 300
    dates = pd.bdate_range("2023-01-01", periods=n)
    # Steadily rising price → above 200d SMA
    sp500 = 4000 * np.cumprod(1 + np.full(n, 0.0005) + rng.normal(0, 0.005, n))
    df = pd.DataFrame({"SP500": sp500}, index=dates)
    # Add rising sectors
    for i, name in enumerate(["Technology", "Healthcare", "Financials",
                               "Energy", "Consumer Disc.", "Consumer Staples",
                               "Industrials", "Utilities", "Real Estate",
                               "Materials", "Communications"]):
        df[f"Sector_{name}"] = 100 * np.cumprod(1 + rng.normal(0.0003, 0.008, n))
    return df


@pytest.fixture
def market_data_bear():
    """Synthetic data where price is below 200d SMA and most sectors are down."""
    rng = np.random.default_rng(42)
    n = 300
    dates = pd.bdate_range("2023-01-01", periods=n)
    # Price rises then crashes
    returns = np.concatenate([
        np.full(200, 0.0005) + rng.normal(0, 0.005, 200),
        np.full(100, -0.003) + rng.normal(0, 0.01, 100),
    ])
    sp500 = 4000 * np.cumprod(1 + returns)
    df = pd.DataFrame({"SP500": sp500}, index=dates)
    # Add declining sectors
    for i, name in enumerate(["Technology", "Healthcare", "Financials",
                               "Energy", "Consumer Disc.", "Consumer Staples",
                               "Industrials", "Utilities", "Real Estate",
                               "Materials", "Communications"]):
        sector_ret = np.concatenate([
            rng.normal(0.0003, 0.008, 200),
            rng.normal(-0.003, 0.01, 100),
        ])
        df[f"Sector_{name}"] = 100 * np.cumprod(1 + sector_ret)
    return df


class TestRegimeValidation:
    def test_bull_regime_confirmed(self, market_data_bull):
        result = validate_regime(market_data_bull, "Bull", None)
        assert isinstance(result, RegimeValidation)
        assert result.regime == "Bull"
        assert result.price_confirmed is True  # Price above 200d SMA
        assert len(result.notes) > 0

    def test_bear_regime_unconfirmed_in_bull_market(self, market_data_bull):
        result = validate_regime(market_data_bull, "Bear", None)
        assert result.regime == "Bear"
        assert result.price_confirmed is False  # Price is above SMA
        assert result.confidence in ("LOW", "MEDIUM")

    def test_bear_regime_confirmed_in_bear_market(self, market_data_bear):
        result = validate_regime(market_data_bear, "Bear", None)
        assert result.regime == "Bear"
        # Price should be below 200d SMA after the crash
        assert result.price_confirmed is True

    def test_neutral_regime(self, market_data_bull):
        result = validate_regime(market_data_bull, "Neutral", None)
        assert result.regime == "Neutral"
        assert result.confirmed in (True, False)

    def test_insufficient_data(self):
        """Short data should not crash, just note insufficient data."""
        short_data = pd.DataFrame(
            {"SP500": [100, 101, 102]},
            index=pd.bdate_range("2023-01-01", periods=3),
        )
        result = validate_regime(short_data, "Bull", None)
        assert isinstance(result, RegimeValidation)
        assert any("Insufficient" in n for n in result.notes)


class TestExternalValidation:
    def test_with_empty_fred(self):
        result = validate_external({}, 0.3, "Bull")
        assert isinstance(result, ExternalValidation)
        assert result.engine_agreement == 0.0

    def test_lei_warning(self):
        """LEI declining for 4 months should trigger WARNING."""
        dates = pd.date_range("2023-01-01", periods=120, freq="D")
        # Monthly declining values
        lei = pd.Series(np.linspace(100, 95, 120), index=dates)
        fred_data = {"lei": lei}
        result = validate_external(fred_data, 0.5, "Bear")
        assert result.lei_signal in ("WARNING", "RECESSION")

    def test_lei_expansion(self):
        """LEI rising should show EXPANSION."""
        dates = pd.date_range("2023-01-01", periods=120, freq="D")
        lei = pd.Series(np.linspace(95, 105, 120), index=dates)
        fred_data = {"lei": lei}
        result = validate_external(fred_data, 0.2, "Bull")
        assert result.lei_signal == "EXPANSION"

    def test_fed_hawkish(self):
        """Fed funds rising by >0.25 should be HAWKISH."""
        dates = pd.date_range("2022-01-01", periods=500, freq="D")
        # Rising from 2% to 5%
        ff = pd.Series(np.linspace(2.0, 5.0, 500), index=dates)
        fred_data = {"fed_funds": ff}
        result = validate_external(fred_data, 0.3, "Neutral")
        assert result.fed_signal == "HAWKISH"

    def test_sentiment_extreme_fear(self):
        """UMCSENT < 60 should be EXTREME_FEAR."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        sentiment = pd.Series([55.0] * 10, index=dates)
        fred_data = {"consumer_sentiment": sentiment}
        result = validate_external(fred_data, 0.5, "Bear")
        assert result.sentiment_signal == "EXTREME_FEAR"

    def test_divergence_alerts_generated(self):
        """When LEI says recession but engine says bull, alert should fire."""
        dates = pd.date_range("2022-01-01", periods=365, freq="D")
        lei = pd.Series(np.linspace(105, 90, 365), index=dates)
        fred_data = {"lei": lei}
        result = validate_external(fred_data, 0.1, "Bull")
        assert len(result.divergence_alerts) > 0

    def test_imf_contraction(self):
        """GDP < 0 should show CONTRACTION."""
        from datetime import datetime
        current_year = datetime.now().year
        result = validate_external(
            {}, 0.5, "Bear",
            alt_data={"imf": {current_year: -0.5, current_year + 1: -0.3}},
        )
        assert result.imf_signal == "CONTRACTION"

    def test_imf_growth(self):
        """GDP > 1.5 should show GROWTH."""
        result = validate_external(
            {}, 0.2, "Bull",
            alt_data={"imf": {2025: 2.5, 2026: 2.8}},
        )
        assert result.imf_signal == "GROWTH"
