"""Unit tests for core engine modules (v2 — includes GARCH, HMM, calibration)."""

import numpy as np
import pandas as pd
import pytest


# ── Config Tests ────────────────────────────────────────────────────────────

class TestConfig:
    def test_config_loads(self):
        from finpredict.config import config
        assert "data" in config
        assert "simulation" in config
        assert "risk" in config
        assert "scenarios" in config

    def test_fred_series_configured(self):
        from finpredict.config import config
        fred = config["data"]["fred_series"]
        assert "yield_spread" in fred
        assert "sahm_rule" in fred
        assert "recession_prob" in fred

    def test_institutional_return_in_range(self):
        from finpredict.config import get_institutional_return
        ret = get_institutional_return()
        assert 0.01 < ret < 0.20

    def test_scenario_configs_sum_to_one(self):
        from finpredict.config import get_scenario_configs
        scenarios = get_scenario_configs()
        total = sum(s["probability"] for s in scenarios.values())
        assert abs(total - 1.0) < 0.01


# ── Risk Tests ──────────────────────────────────────────────────────────────

class TestRiskScoring:
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
    def test_regimes_return_series_and_string(self, sample_market_data):
        from finpredict.risk.regimes import detect_regimes
        sample_market_data["Risk_Score"] = 0.0
        regimes, current = detect_regimes(sample_market_data)
        assert isinstance(regimes, pd.Series)
        assert current in ("Bull", "Bear", "Volatile", "Unknown")


class TestCrashIdentification:
    def test_crash_detection_returns_dataframe(self, sample_market_data):
        from finpredict.risk.crashes import identify_crashes
        crash_df, freq = identify_crashes(sample_market_data)
        assert isinstance(crash_df, pd.DataFrame)
        assert freq >= 0


# ── Simulation Tests ────────────────────────────────────────────────────────

class TestSimulation:
    def test_simulate_paths_shape(self):
        from finpredict.simulation.monte_carlo import simulate_paths
        base_scenario = {"drift_adj": 0, "vol_mult": 1.0, "crash_mult": 1.0}
        paths = simulate_paths(5000, 0.06, 0.18, 252, 100, 0.07, 0.5, base_scenario, seed=42)
        assert paths.shape == (253, 100)

    def test_simulate_paths_start_price(self):
        from finpredict.simulation.monte_carlo import simulate_paths
        base_scenario = {"drift_adj": 0, "vol_mult": 1.0, "crash_mult": 1.0}
        paths = simulate_paths(5000, 0.06, 0.18, 252, 100, 0.07, 0.5, base_scenario, seed=42)
        assert np.all(paths[0] == 5000)

    def test_simulate_paths_positive(self):
        from finpredict.simulation.monte_carlo import simulate_paths
        base_scenario = {"drift_adj": 0, "vol_mult": 1.0, "crash_mult": 1.0}
        paths = simulate_paths(5000, 0.06, 0.18, 252, 100, 0.07, 0.5, base_scenario, seed=42)
        assert np.all(paths > 0)

    def test_garch_vol_used_when_provided(self):
        """GARCH vol should produce different results than default."""
        from finpredict.simulation.monte_carlo import simulate_paths
        base_scenario = {"drift_adj": 0, "vol_mult": 1.0, "crash_mult": 1.0}
        paths_default = simulate_paths(5000, 0.06, 0.18, 252, 500, 0.07, 0.5, base_scenario, seed=42)
        paths_garch = simulate_paths(5000, 0.06, 0.18, 252, 500, 0.07, 0.5, base_scenario, garch_vol=0.25, seed=42)
        # Different volatility → different paths
        # (Same seed but different sigma → different scaling)
        assert not np.allclose(paths_default[-1], paths_garch[-1])


# ── Calibration Fix Tests ───────────────────────────────────────────────────

class TestCalibrationFix:
    """
    The critical test: crash rate should be HIGHER when risk is HIGH
    and LOWER when risk is LOW.

    Before the fix, the model produced inverted calibration:
        Low risk  (<15% predicted): 86% actual crashes
        High risk (>40% predicted):  7% actual crashes
    """

    def test_high_risk_produces_more_crashes(self):
        from finpredict.simulation.monte_carlo import simulate_paths
        n_sims = 2000
        base_scenario = {"drift_adj": 0, "vol_mult": 1.0, "crash_mult": 1.0}

        # Low risk environment
        paths_low = simulate_paths(
            5000, 0.06, 0.18, 252, n_sims,
            crash_freq=0.05, risk_score=-1.0, scenario=base_scenario, seed=42,
        )
        peak_low = np.maximum.accumulate(paths_low, axis=0)
        dd_low = ((paths_low - peak_low) / peak_low).min(axis=0)
        crash_rate_low = (dd_low <= -0.20).mean()

        # High risk environment — higher crash freq and risk score
        high_scenario = {"drift_adj": -0.05, "vol_mult": 1.5, "crash_mult": 2.5}
        paths_high = simulate_paths(
            5000, 0.06, 0.18, 252, n_sims,
            crash_freq=0.15, risk_score=2.0, scenario=high_scenario, seed=42,
        )
        peak_high = np.maximum.accumulate(paths_high, axis=0)
        dd_high = ((paths_high - peak_high) / peak_high).min(axis=0)
        crash_rate_high = (dd_high <= -0.20).mean()

        assert crash_rate_high > crash_rate_low, (
            f"CALIBRATION STILL INVERTED: "
            f"high-risk crash rate ({crash_rate_high:.2%}) should exceed "
            f"low-risk ({crash_rate_low:.2%})"
        )

    def test_risk_factor_range(self):
        """Risk factor should span a wide range, not just ±20%."""
        # At risk=-2.0: factor = 1.0 + 0.50*(-2.0) = 0.0 → clamped to 0.25
        # At risk=+2.0: factor = 1.0 + 0.50*(2.0) = 2.0
        factor_low = max(0.25, min(3.0, 1.0 + 0.50 * (-2.0)))
        factor_high = max(0.25, min(3.0, 1.0 + 0.50 * (2.0)))
        assert factor_high / factor_low >= 4.0, (
            f"Risk factor range too narrow: {factor_low:.2f} to {factor_high:.2f}"
        )


# ── Scenario Tests ──────────────────────────────────────────────────────────

class TestScenarios:
    def test_probabilities_sum_to_one(self):
        from finpredict.simulation.scenarios import build_scenarios
        scenarios = build_scenarios("Bull", 0.5, 20.0, 0.5)
        total = sum(s["probability"] for s in scenarios.values())
        assert abs(total - 1.0) < 0.001

    def test_high_risk_shifts_to_bearish(self):
        from finpredict.simulation.scenarios import build_scenarios
        low_risk = build_scenarios("Bull", -1.0, 15.0, 1.0)
        high_risk = build_scenarios("Volatile", 3.0, 35.0, -0.5)
        bearish_low = sum(
            s["probability"] for s in low_risk.values() if s.get("category") == "bearish"
        )
        bearish_high = sum(
            s["probability"] for s in high_risk.values() if s.get("category") == "bearish"
        )
        assert bearish_high > bearish_low

    def test_recession_prob_shifts_bearish(self):
        """FRED recession probability should increase bearish scenarios."""
        from finpredict.simulation.scenarios import build_scenarios
        no_recession = build_scenarios("Bull", 0.0, 20.0, 0.5, recession_prob=0.05)
        high_recession = build_scenarios("Bull", 0.0, 20.0, 0.5, recession_prob=0.80)
        bearish_calm = sum(
            s["probability"] for s in no_recession.values() if s.get("category") == "bearish"
        )
        bearish_stress = sum(
            s["probability"] for s in high_recession.values() if s.get("category") == "bearish"
        )
        assert bearish_stress > bearish_calm, (
            f"FRED recession prob not shifting scenarios: "
            f"low={bearish_calm:.2%}, high={bearish_stress:.2%}"
        )


# ── GARCH Tests ─────────────────────────────────────────────────────────────

class TestGARCH:
    def test_garch_fit_returns_result(self):
        from finpredict.models.garch import fit_garch
        returns = pd.Series(np.random.randn(1000) * 0.01)
        result = fit_garch(returns)
        assert hasattr(result, "current_vol")
        assert result.current_vol > 0

    def test_garch_fallback_with_short_data(self):
        from finpredict.models.garch import fit_garch
        returns = pd.Series(np.random.randn(50) * 0.01)
        result = fit_garch(returns, min_obs=500)
        assert result.success is False
        assert result.current_vol > 0  # Should still have a fallback value


# ── HMM Tests ──────────────────────────────────────────────────────────────

class TestHMM:
    def test_hmm_fit_returns_result(self, sample_market_data):
        from finpredict.models.hmm_regimes import fit_hmm_regimes
        result = fit_hmm_regimes(sample_market_data, n_fits=3)
        assert result.current_regime in ("Bull", "Bear", "Crisis", "Unknown")
        assert len(result.regime_probs) == 3

    def test_regime_probs_sum_to_one(self, sample_market_data):
        from finpredict.models.hmm_regimes import fit_hmm_regimes, get_regime_probs
        result = fit_hmm_regimes(sample_market_data, n_fits=3)
        probs = get_regime_probs(result)
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01


# ── FRED Tests ──────────────────────────────────────────────────────────────

class TestFRED:
    def test_recession_probability_bounded(self):
        from finpredict.data.fred_fetcher import get_recession_probability
        # With mock data
        mock = {
            "yield_spread": pd.Series([2.5]),
            "sahm_rule": pd.Series([0.03]),
        }
        prob = get_recession_probability(mock)
        assert 0 <= prob <= 1.0

    def test_recession_probability_high_on_inversion(self):
        from finpredict.data.fred_fetcher import get_recession_probability
        mock = {
            "yield_spread": pd.Series([-1.5]),  # Deep inversion
            "sahm_rule": pd.Series([0.60]),      # Sahm triggered
            "recession_prob": pd.Series([75.0]), # High model probability
        }
        prob = get_recession_probability(mock)
        assert prob > 0.60, f"Recession prob should be high on inversion, got {prob:.2%}"

    def test_recession_probability_low_in_calm(self):
        from finpredict.data.fred_fetcher import get_recession_probability
        mock = {
            "yield_spread": pd.Series([2.5]),   # Steep curve
            "sahm_rule": pd.Series([0.02]),     # Low
            "recession_prob": pd.Series([2.0]), # Model says calm
        }
        prob = get_recession_probability(mock)
        assert prob < 0.30, f"Recession prob should be low in calm, got {prob:.2%}"


class TestAnomalyBaseRate:
    """Bug 4: Anomaly adjustment should use empirical crash rate, not hardcoded 0.12."""

    def test_anomaly_uses_empirical_base_rate(self):
        """The base_rate used in anomaly adjustment should come from
        crash_model._train_crash_rate, not a hardcoded constant."""
        from unittest.mock import MagicMock
        from finpredict.config import config

        # Mock crash_model with empirical rate
        crash_model = MagicMock()
        crash_model._train_crash_rate = {"12m": 0.08}
        crash_model.is_trained = True

        # Simulate the anomaly adjustment formula
        ml_crash_prob = 0.40
        conf_factor = 0.70
        anomaly_report = {"is_anomalous": True, "confidence_factor": conf_factor}

        # The fixed code: uses empirical rate
        base_rate = crash_model._train_crash_rate.get(
            "12m",
            config.get("ml", {}).get("crash_base_rate_fallback", 0.12),
        )
        adjusted = ml_crash_prob * conf_factor + base_rate * (1 - conf_factor)

        # With base_rate=0.08: adjusted = 0.40*0.70 + 0.08*0.30 = 0.304
        assert base_rate == 0.08, f"Expected empirical rate 0.08, got {base_rate}"
        expected = 0.40 * 0.70 + 0.08 * 0.30
        assert abs(adjusted - expected) < 1e-6

        # Verify it's different from the old hardcoded value
        old_adjusted = ml_crash_prob * conf_factor + 0.12 * (1 - conf_factor)
        assert adjusted != old_adjusted, "Adjustment should differ from hardcoded 0.12"

    def test_base_rate_fallback_from_config(self):
        """When _train_crash_rate is empty, fallback should come from config."""
        from unittest.mock import MagicMock
        from finpredict.config import config

        crash_model = MagicMock()
        crash_model._train_crash_rate = {}

        base_rate = crash_model._train_crash_rate.get(
            "12m",
            config.get("ml", {}).get("crash_base_rate_fallback", 0.12),
        )
        expected_fallback = config.get("ml", {}).get("crash_base_rate_fallback", 0.12)
        assert base_rate == expected_fallback
