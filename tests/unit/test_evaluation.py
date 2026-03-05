"""Tests for the evaluation module (metrics.py + comparison.py)."""

import numpy as np
import pandas as pd
import pytest

from finpredict.evaluation.metrics import (
    brier_score,
    brier_skill_score,
    reliability_diagram,
    roc_auc_over_time,
    prediction_spread_check,
)
from finpredict.evaluation.comparison import (
    baseline_crash_prob_vix,
    baseline_crash_prob_yield_curve,
    baseline_crash_prob_climatology,
    compare_models,
)


class TestBrierScore:
    def test_perfect_prediction(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1.0, 0.0, 1.0, 0.0])
        assert brier_score(y_pred, y_true) == 0.0

    def test_worst_prediction(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.0, 1.0, 0.0, 1.0])
        assert brier_score(y_pred, y_true) == 1.0

    def test_coin_flip(self):
        y_true = np.array([1, 0])
        y_pred = np.array([0.5, 0.5])
        assert abs(brier_score(y_pred, y_true) - 0.25) < 1e-10

    def test_empty_input(self):
        assert np.isnan(brier_score(np.array([]), np.array([])))


class TestBrierSkillScore:
    def test_perfect_model_beats_climatology(self):
        y_true = np.array([1, 0, 1, 0, 0, 0, 0, 0, 1, 0])
        y_pred = y_true.astype(float)
        bss = brier_skill_score(y_pred, y_true, baseline="climatology")
        assert bss == 1.0

    def test_climatology_baseline_is_zero(self):
        """Predicting the base rate gives BSS = 0."""
        y_true = np.array([1, 0, 0, 0, 1, 0, 0, 0])
        base_rate = y_true.mean()
        y_pred = np.full_like(y_true, base_rate, dtype=float)
        bss = brier_skill_score(y_pred, y_true, baseline="climatology")
        assert abs(bss) < 1e-10

    def test_vix_baseline(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.8, 0.2, 0.9, 0.1])
        vix = pd.Series([30, 15, 28, 12])
        bss = brier_skill_score(y_pred, y_true, baseline="vix25", vix_series=vix)
        assert isinstance(bss, float)

    def test_yield_curve_baseline(self):
        y_true = np.array([1, 0, 0, 1])
        y_pred = np.array([0.7, 0.3, 0.2, 0.8])
        spread = pd.Series([-0.5, 1.0, 0.5, -0.3])
        bss = brier_skill_score(y_pred, y_true, baseline="yield_curve",
                                spread_series=spread)
        assert isinstance(bss, float)

    def test_unknown_baseline_raises(self):
        with pytest.raises(ValueError, match="Unknown baseline"):
            brier_skill_score(np.array([0.5]), np.array([1]), baseline="magic")


class TestReliabilityDiagram:
    def test_output_keys(self):
        y_pred = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        y_true = np.array([0, 0, 1, 1, 1])
        result = reliability_diagram(y_pred, y_true)
        assert "bin_centers" in result
        assert "bin_frequencies" in result
        assert "bin_counts" in result
        assert "calibration_error" in result

    def test_perfect_calibration_low_ece(self):
        """Well-calibrated predictions should have low ECE."""
        rng = np.random.default_rng(42)
        n = 10000
        y_pred = rng.uniform(0, 1, n)
        y_true = (rng.uniform(0, 1, n) < y_pred).astype(float)
        result = reliability_diagram(y_pred, y_true, n_bins=10)
        assert result["calibration_error"] < 0.05

    def test_bin_counts_sum_to_total(self):
        y_pred = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        y_true = np.array([0, 0, 1, 1, 1])
        result = reliability_diagram(y_pred, y_true)
        assert result["bin_counts"].sum() == 5


class TestRocAucOverTime:
    def test_returns_series(self):
        dates = pd.date_range("2010-01-01", periods=100, freq="3MS")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "date": dates,
            "y_pred": rng.uniform(0, 1, 100),
            "y_true": rng.choice([0, 1], 100, p=[0.7, 0.3]),
        })
        result = roc_auc_over_time(df)
        assert isinstance(result, pd.Series)


class TestPredictionSpreadCheck:
    def test_underdispersed(self):
        y_pred = np.full(100, 0.12)
        result = prediction_spread_check(y_pred)
        assert result["is_underdispersed"] is True
        assert result["std"] < 0.05

    def test_well_dispersed(self):
        y_pred = np.array([0.05, 0.15, 0.25, 0.45, 0.65, 0.85])
        result = prediction_spread_check(y_pred)
        assert result["is_underdispersed"] is False

    def test_keys(self):
        result = prediction_spread_check(np.array([0.5]))
        assert set(result.keys()) == {"mean", "std", "min", "max", "is_underdispersed"}


class TestBaselineModels:
    def test_vix_baseline(self):
        vix = pd.Series([30, 15, 28, 12, 26])
        result = baseline_crash_prob_vix(vix)
        expected = pd.Series([1.0, 0.0, 1.0, 0.0, 1.0])
        pd.testing.assert_series_equal(result, expected)

    def test_yield_curve_baseline(self):
        spread = pd.Series([-0.5, 1.0, 0.5, -0.3])
        result = baseline_crash_prob_yield_curve(spread)
        expected = pd.Series([1.0, 0.0, 0.0, 1.0])
        pd.testing.assert_series_equal(result, expected)

    def test_climatology_baseline(self):
        y_true = pd.Series([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        rate = baseline_crash_prob_climatology(y_true)
        assert rate == 0.2


class TestCompareModels:
    def test_compare_returns_dataframe(self):
        rng = np.random.default_rng(42)
        y_true = rng.choice([0, 1], 100, p=[0.8, 0.2])
        results = {
            "model_a": {"y_pred": rng.uniform(0, 1, 100), "y_true": y_true},
            "model_b": {"y_pred": rng.uniform(0, 1, 100), "y_true": y_true},
        }
        df = compare_models(results)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "brier_score" in df.columns
        assert "bss_vs_climatology" in df.columns
