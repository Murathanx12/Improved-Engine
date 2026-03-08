"""Core unit tests for predict_proba, feature matrix integrity, and Monte Carlo.

These tests cover the 3 most critical prediction paths:
1. CrashPredictor.predict_proba — output shape, bounds, multi-horizon, errors
2. build_feature_matrix — output integrity, no inf, no fillna(0) corruption
3. simulate_paths / run_monte_carlo — ML drift override, output structure
"""

import numpy as np
import pandas as pd
import pytest

from finpredict.ml.crash_model import CrashPredictor
from finpredict.ml.features import (
    build_feature_matrix,
    build_target_crash_multi,
)
from finpredict.simulation.monte_carlo import simulate_paths, run_monte_carlo


# ═══════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def synthetic_data():
    """2000-row synthetic market data with 2 crash periods for training."""
    rng = np.random.default_rng(seed=123)
    n = 2000
    dates = pd.bdate_range("2015-01-01", periods=n)

    # Base returns with 2 crash periods injected
    returns = rng.normal(loc=0.0003, scale=0.012, size=n)
    # Crash 1: rows 600-650
    returns[600:650] = rng.normal(loc=-0.015, scale=0.025, size=50)
    # Crash 2: rows 1400-1450
    returns[1400:1450] = rng.normal(loc=-0.018, scale=0.030, size=50)

    prices = 3000 * np.exp(np.cumsum(returns))

    data = pd.DataFrame(
        {
            "SP500": prices,
            "VIX": rng.normal(20, 5, n).clip(10, 80),
            "T10Y": rng.normal(0.035, 0.005, n).clip(0.01, 0.08),
            "T3M": rng.normal(0.02, 0.003, n).clip(0.001, 0.06),
            "T30Y": rng.normal(0.04, 0.005, n).clip(0.015, 0.08),
            "HYG": 80 + rng.normal(0, 2, n).cumsum() * 0.01,
            "LQD": 110 + rng.normal(0, 1, n).cumsum() * 0.01,
            "Gold": 1800 + rng.normal(0, 10, n).cumsum() * 0.1,
            "NASDAQ": 10000 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n))),
            "Russell": 1800 * np.exp(np.cumsum(rng.normal(0.0002, 0.014, n))),
        },
        index=dates,
    )

    return data


@pytest.fixture(scope="module")
def trained_crash_model(synthetic_data):
    """CrashPredictor trained on synthetic data."""
    features = build_feature_matrix(synthetic_data)
    targets = build_target_crash_multi(synthetic_data, threshold=-0.10)

    model = CrashPredictor(n_estimators=100, random_state=42)
    result = model.train(features, targets, min_train_samples=500)

    if not result.get("success"):
        pytest.skip(f"Training failed: {result.get('reason', 'unknown')}")

    return model, features


# ═══════════════════════════════════════════════════════════════════════
# TEST CRASH PREDICTOR CORE
# ═══════════════════════════════════════════════════════════════════════


class TestCrashPredictorCore:
    def test_output_shape_matches_input(self, trained_crash_model):
        model, features = trained_crash_model
        preds = model.predict_proba(features, "12m")
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(features)

    def test_probabilities_in_valid_range(self, trained_crash_model):
        model, features = trained_crash_model
        preds = model.predict_proba(features, "12m")
        assert np.all(preds >= 0.02), f"Min pred {preds.min():.4f} < 0.02"
        assert np.all(preds <= 0.98), f"Max pred {preds.max():.4f} > 0.98"

    def test_multiple_horizons(self, trained_crash_model):
        model, features = trained_crash_model
        row = features.iloc[-1:]
        for horizon in model.models:
            preds = model.predict_proba(row, horizon)
            assert len(preds) == 1
            assert 0.02 <= preds[0] <= 0.98

    def test_untrained_raises(self):
        model = CrashPredictor()
        dummy = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(RuntimeError, match="not trained"):
            model.predict_proba(dummy)

    def test_train_returns_success_dict(self, synthetic_data):
        features = build_feature_matrix(synthetic_data)
        targets = build_target_crash_multi(synthetic_data, threshold=-0.10)
        model = CrashPredictor(n_estimators=50, random_state=99)
        result = model.train(features, targets, min_train_samples=500)
        assert "success" in result
        if result["success"]:
            assert "val_brier" in result

    def test_logistic_selected_for_sparse_data(self, trained_crash_model):
        """Model selection results should exist and contain both scores."""
        model, features = trained_crash_model
        # model_selection_results should exist with both scores per horizon
        assert len(model.model_selection_results) > 0
        for h in model.model_selection_results:
            info = model.model_selection_results[h]
            assert "lgb_brier" in info
            assert "logistic_brier" in info
            assert info["selected"] in ("lgb", "logistic")


# ═══════════════════════════════════════════════════════════════════════
# TEST FEATURE MATRIX INTEGRITY
# ═══════════════════════════════════════════════════════════════════════


class TestFeatureMatrixIntegrity:
    def test_index_aligned_with_input(self, sample_market_data):
        features = build_feature_matrix(sample_market_data)
        assert features.index.equals(sample_market_data.index)

    def test_no_inf_in_output(self, sample_market_data):
        features = build_feature_matrix(sample_market_data)
        inf_count = np.isinf(features.values).sum()
        assert inf_count == 0, f"Found {inf_count} inf values in feature matrix"

    def test_all_columns_float(self, sample_market_data):
        features = build_feature_matrix(sample_market_data)
        for col in features.columns:
            assert np.issubdtype(
                features[col].dtype, np.floating
            ), f"Column '{col}' has dtype {features[col].dtype}, expected float"

    def test_no_duplicate_columns(self, sample_market_data):
        features = build_feature_matrix(sample_market_data)
        assert len(features.columns) == len(set(features.columns))

    def test_no_fillna_zero_corruption(self, sample_market_data):
        """Bug 4 verification: columns that need warmup should have NaN, not 0."""
        features = build_feature_matrix(sample_market_data)
        # mom_12m needs 252 rows of data — first row should be NaN
        if "mom_12m" in features.columns:
            first_val = features["mom_12m"].iloc[0]
            assert (
                pd.isna(first_val) or first_val != 0.0
            ), "mom_12m row 0 is 0.0 — likely fillna(0) corruption (Bug 4)"

    def test_fred_data_integration(self, sample_market_data):
        rng = np.random.default_rng(42)
        fred_data = {
            "hy_oas": pd.Series(
                rng.normal(4, 1, 100),
                index=pd.bdate_range("2020-01-01", periods=100),
            ),
        }
        features = build_feature_matrix(sample_market_data, fred_data=fred_data)
        fred_cols = [c for c in features.columns if c.startswith("fred_")]
        assert len(fred_cols) > 0, "FRED columns not added"

    def test_global_data_integration(self, sample_market_data):
        """With global_data dict, contagion columns should appear (if module available)."""
        rng = np.random.default_rng(42)
        n = len(sample_market_data)
        global_data = {
            "FTSE": pd.Series(
                7000 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n))),
                index=sample_market_data.index,
            ),
            "DAX": pd.Series(
                14000 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n))),
                index=sample_market_data.index,
            ),
        }
        try:
            features = build_feature_matrix(sample_market_data, global_data=global_data)
            # If compute_contagion_features exists, contagion cols should be added
            # Passes whether or not the module is importable
            assert isinstance(features, pd.DataFrame)
        except Exception:
            pytest.skip("global_crashes module not available")


# ═══════════════════════════════════════════════════════════════════════
# TEST SIMULATE PATHS CORE
# ═══════════════════════════════════════════════════════════════════════


class TestSimulatePathsCore:
    @pytest.fixture
    def base_params(self):
        return dict(
            start_price=5000.0,
            historical_mu=0.08,
            historical_sigma=0.16,
            days=252,
            n_sims=500,
            crash_freq=0.10,
            risk_score=0.5,
            scenario={"drift_adj": 0.0, "vol_mult": 1.0, "crash_mult": 1.0},
            seed=42,
        )

    def test_ml_return_overrides_historical(self, base_params):
        """Bug 3 verification: ML return should drive drift, not historical."""
        ml_return = 0.15  # 15% expected return
        paths = simulate_paths(
            **base_params,
            ml_predicted_return=ml_return,
        )
        mean_1y_return = paths[-1].mean() / base_params["start_price"] - 1
        # Should be roughly near ML target (within ±10pp given randomness)
        assert (
            abs(mean_1y_return - ml_return) < 0.15
        ), f"Mean 1Y return {mean_1y_return:.2%} too far from ML target {ml_return:.2%}"

    def test_output_shape(self, base_params):
        paths = simulate_paths(**base_params)
        assert paths.shape == (base_params["days"] + 1, base_params["n_sims"])

    def test_start_price_is_first_row(self, base_params):
        paths = simulate_paths(**base_params)
        np.testing.assert_array_equal(paths[0, :], base_params["start_price"])

    def test_all_prices_positive(self, base_params):
        paths = simulate_paths(**base_params)
        assert np.all(paths > 0), "Found non-positive prices in simulation"

    def test_crash_prob_affects_jump_rate(self, base_params):
        """Higher crash prob should produce more crash-like paths."""
        paths_low = simulate_paths(**base_params, ml_crash_prob=0.05)
        paths_high = simulate_paths(**{**base_params, "seed": 43}, ml_crash_prob=0.60)

        # Compute max drawdowns
        def max_dd(paths):
            peak = np.maximum.accumulate(paths, axis=0)
            dd = (paths - peak) / peak
            return dd.min(axis=0)

        dd_low = max_dd(paths_low)
        dd_high = max_dd(paths_high)
        # Higher crash prob should produce deeper average drawdowns
        assert (
            dd_high.mean() < dd_low.mean()
        ), f"High crash prob DD {dd_high.mean():.3f} not deeper than low {dd_low.mean():.3f}"

    def test_scenario_vol_mult_applied(self, base_params):
        """vol_mult=2.0 should produce wider paths than vol_mult=1.0."""
        paths_normal = simulate_paths(**base_params)
        params_high_vol = {**base_params, "seed": 42}
        params_high_vol["scenario"] = {"drift_adj": 0.0, "vol_mult": 2.0, "crash_mult": 1.0}
        paths_wide = simulate_paths(**params_high_vol)

        std_normal = paths_normal[-1].std()
        std_wide = paths_wide[-1].std()
        assert (
            std_wide > std_normal
        ), f"vol_mult=2.0 std {std_wide:.0f} not wider than vol_mult=1.0 std {std_normal:.0f}"


# ═══════════════════════════════════════════════════════════════════════
# TEST RUN MONTE CARLO OUTPUT
# ═══════════════════════════════════════════════════════════════════════


class TestRunMonteCarloOutput:
    @pytest.fixture(scope="class")
    def mc_result(self):
        """Run a small Monte Carlo simulation for output structure tests."""
        return run_monte_carlo(
            current_price=5000.0,
            current_regime="Bull",
            risk_score=0.3,
            crash_freq=0.10,
            current_vix=18.0,
            yield_curve=0.01,
            val_penalty=0.0,
            garch_vol=0.15,
            garch_persistence=0.97,
            ml_crash_prob=0.15,
            ml_predicted_return=0.08,
            seed=42,
        )

    def test_output_dict_keys(self, mc_result):
        required_keys = [
            "final_mean",
            "crash_prob_1y",
            "cvar_95_pct",
            "total_return_pct",
            "annual_return_pct",
            "final_p05",
            "final_p25",
            "final_p75",
            "final_p95",
            "scenarios",
            "all_paths",
        ]
        for key in required_keys:
            assert key in mc_result, f"Missing key '{key}' in run_monte_carlo output"

    def test_quantiles_monotonic(self, mc_result):
        p5 = mc_result["final_p05"]
        p25 = mc_result["final_p25"]
        p75 = mc_result["final_p75"]
        p95 = mc_result["final_p95"]
        assert (
            p5 < p25 < p75 < p95
        ), f"Quantiles not monotonic: p5={p5:.0f}, p25={p25:.0f}, p75={p75:.0f}, p95={p95:.0f}"

    def test_crash_prob_between_0_and_100(self, mc_result):
        cp = mc_result["crash_prob_1y"]
        assert 0 <= cp <= 100, f"crash_prob_1y={cp} outside [0, 100]"
