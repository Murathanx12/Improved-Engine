"""Unit tests for core engine modules."""

import numpy as np
import pandas as pd
import pytest


class TestConfig:
    """Tests for configuration loading."""

    def test_config_loads(self):
        from finpredict.config import config
        assert "data" in config
        assert "simulation" in config
        assert "risk" in config
        assert "scenarios" in config

    def test_institutional_return_in_range(self):
        from finpredict.config import get_institutional_return
        ret = get_institutional_return()
        assert 0.01 < ret < 0.20, f"Institutional return {ret} out of reasonable range"

    def test_forecast_days_positive(self):
        from finpredict.config import get_forecast_days
        days = get_forecast_days()
        assert days > 0

    def test_scenario_configs_sum_to_one(self):
        from finpredict.config import get_scenario_configs
        scenarios = get_scenario_configs()
        total = sum(s["probability"] for s in scenarios.values())
        assert abs(total - 1.0) < 0.01, f"Scenario probabilities sum to {total}, expected ~1.0"

    def test_all_scenarios_have_required_fields(self):
        from finpredict.config import get_scenario_configs
        required = {"probability", "return", "volatility", "crash_multiplier", "description"}
        for name, s in get_scenario_configs().items():
            for field in required:
                assert field in s, f"Scenario '{name}' missing field '{field}'"


class TestRiskScoring:
    """Tests for the 9-factor composite risk score."""

    def test_risk_score_shape(self, sample_market_data):
        from finpredict.risk.scoring import build_risk_score
        score = build_risk_score(sample_market_data)
        assert len(score) == len(sample_market_data)

    def test_risk_score_bounded(self, sample_market_data):
        from finpredict.risk.scoring import build_risk_score
        score = build_risk_score(sample_market_data)
        valid = score.dropna()
        assert valid.min() >= -4.0
        assert valid.max() <= 4.0

    def test_rolling_zscore_properties(self):
        from finpredict.risk.scoring import rolling_zscore
        series = pd.Series(np.random.randn(500))
        z = rolling_zscore(series, window=100)
        valid = z.dropna()
        assert valid.min() >= -5.0
        assert valid.max() <= 5.0


class TestRegimeDetection:
    """Tests for market regime classification."""

    def test_regimes_return_series_and_string(self, sample_market_data):
        from finpredict.risk.regimes import detect_regimes
        sample_market_data["Risk_Score"] = 0.0  # Add required column
        regimes, current = detect_regimes(sample_market_data)
        assert isinstance(regimes, pd.Series)
        assert isinstance(current, str)
        assert current in ("Bull", "Bear", "Volatile", "Unknown")

    def test_regimes_length_matches_data(self, sample_market_data):
        from finpredict.risk.regimes import detect_regimes
        sample_market_data["Risk_Score"] = 0.0
        regimes, _ = detect_regimes(sample_market_data)
        assert len(regimes) == len(sample_market_data)


class TestCrashIdentification:
    """Tests for historical crash detection."""

    def test_crash_detection_returns_dataframe(self, sample_market_data):
        from finpredict.risk.crashes import identify_crashes
        crash_df, freq = identify_crashes(sample_market_data)
        assert isinstance(crash_df, pd.DataFrame)
        assert isinstance(freq, float)
        assert freq >= 0

    def test_crash_frequency_reasonable(self, sample_market_data):
        from finpredict.risk.crashes import identify_crashes
        _, freq = identify_crashes(sample_market_data)
        # For synthetic data, frequency should be between 0 and 1
        assert 0 <= freq <= 1.0


class TestSimulation:
    """Tests for Monte Carlo path simulation."""

    def test_simulate_paths_shape(self):
        from finpredict.simulation.monte_carlo import simulate_paths
        paths = simulate_paths(
            start_price=5000, annual_return=0.06, annual_vol=0.18,
            days=252, n_sims=100,
        )
        assert paths.shape == (253, 100)  # days+1 rows, n_sims columns

    def test_simulate_paths_start_price(self):
        from finpredict.simulation.monte_carlo import simulate_paths
        paths = simulate_paths(
            start_price=5000, annual_return=0.06, annual_vol=0.18,
            days=252, n_sims=100,
        )
        assert np.all(paths[0] == 5000)

    def test_simulate_paths_positive(self):
        from finpredict.simulation.monte_carlo import simulate_paths
        paths = simulate_paths(
            start_price=5000, annual_return=0.06, annual_vol=0.18,
            days=252, n_sims=100,
        )
        assert np.all(paths > 0), "All prices must be positive"

    def test_simulate_paths_with_crash_rate(self):
        from finpredict.simulation.monte_carlo import simulate_paths
        paths = simulate_paths(
            start_price=5000, annual_return=0.06, annual_vol=0.18,
            days=252, n_sims=500, crash_rate=0.15, risk_level=1.5,
        )
        assert paths.shape[1] == 500


class TestScenarios:
    """Tests for scenario building and probability adjustment."""

    def test_build_scenarios_probabilities_sum_to_one(self):
        from finpredict.simulation.scenarios import build_scenarios
        scenarios = build_scenarios("Bull", 0.5, 20.0, 0.5)
        total = sum(s["probability"] for s in scenarios.values())
        assert abs(total - 1.0) < 0.001

    def test_high_risk_shifts_to_bearish(self):
        from finpredict.simulation.scenarios import build_scenarios
        low_risk = build_scenarios("Bull", -1.0, 15.0, 1.0)
        high_risk = build_scenarios("Volatile", 3.0, 35.0, -0.5)

        # Sum bearish probabilities
        bearish_low = sum(
            s["probability"] for s in low_risk.values()
            if s.get("category") == "bearish"
        )
        bearish_high = sum(
            s["probability"] for s in high_risk.values()
            if s.get("category") == "bearish"
        )
        assert bearish_high > bearish_low, "High risk should increase bearish probability"
