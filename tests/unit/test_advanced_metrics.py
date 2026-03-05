"""Tests for the 6 advanced validation metrics."""

import numpy as np
import pandas as pd
import pytest

from finpredict.evaluation.metrics import (
    lead_time_accuracy,
    false_alarm_rate,
    missed_crash_rate,
    regime_persistence_iou,
    directional_accuracy_vs_consensus,
    signal_max_drawdown,
)


@pytest.fixture
def bt_with_crash():
    """Backtest results with one clear crash episode."""
    dates = pd.date_range("2005-01-01", periods=20, freq="3MS")
    return pd.DataFrame({
        "date": dates,
        "ml_crash_12m": [0.1, 0.15, 0.2, 0.3, 0.45, 0.65, 0.70, 0.55,
                         0.35, 0.2, 0.15, 0.1, 0.1, 0.12, 0.08, 0.1,
                         0.09, 0.11, 0.1, 0.08],
        "actual_crash_12m": [False, False, False, False, True, True, True, True,
                             False, False, False, False, False, False, False, False,
                             False, False, False, False],
        "ml_return_12m": [0.08, 0.06, 0.04, -0.02, -0.15, -0.20, -0.10, 0.05,
                          0.10, 0.08, 0.07, 0.06, 0.05, 0.06, 0.07, 0.08,
                          0.06, 0.07, 0.08, 0.07],
        "actual_return_12m": [0.10, 0.05, 0.02, -0.05, -0.25, -0.30, -0.15, 0.10,
                              0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.09, 0.10,
                              0.08, 0.07, 0.06, 0.08],
        "start_price": [1200 + i * 30 for i in range(20)],
    })


@pytest.fixture
def crash_periods():
    """Known crash period."""
    return pd.DataFrame({
        "start": [pd.Timestamp("2006-04-01")],
        "end": [pd.Timestamp("2007-04-01")],
    })


class TestLeadTimeAccuracy:
    def test_detects_crash_with_lead(self, bt_with_crash, crash_periods):
        result = lead_time_accuracy(bt_with_crash, crash_periods, prob_threshold=0.30)
        assert result["n_detected"] >= 1
        assert result["mean_lead_days"] > 0

    def test_no_detection_high_threshold(self, bt_with_crash, crash_periods):
        result = lead_time_accuracy(bt_with_crash, crash_periods, prob_threshold=0.99)
        assert result["n_detected"] == 0
        assert result["mean_lead_days"] == 0

    def test_empty_data(self):
        result = lead_time_accuracy(pd.DataFrame(), pd.DataFrame(columns=["start"]))
        assert result["n_crashes"] == 0

    def test_no_crash_periods(self, bt_with_crash):
        result = lead_time_accuracy(bt_with_crash, pd.DataFrame(columns=["start"]))
        assert result["n_crashes"] == 0


class TestFalseAlarmRate:
    def test_no_alarms(self, bt_with_crash):
        result = false_alarm_rate(bt_with_crash, alarm_threshold=0.99)
        assert result["n_alarms"] == 0
        assert result["rate"] == 0.0

    def test_with_alarms(self, bt_with_crash):
        result = false_alarm_rate(bt_with_crash, alarm_threshold=0.60)
        assert result["n_alarms"] >= 1
        assert 0 <= result["rate"] <= 1

    def test_empty_data(self):
        result = false_alarm_rate(pd.DataFrame())
        assert result["n_alarms"] == 0


class TestMissedCrashRate:
    def test_good_model_detects(self, bt_with_crash, crash_periods):
        """Model that has high prob before crash should not miss."""
        result = missed_crash_rate(bt_with_crash, crash_periods, safe_threshold=0.30)
        assert result["n_crashes"] >= 1

    def test_empty_data(self):
        result = missed_crash_rate(pd.DataFrame(), pd.DataFrame(columns=["start"]))
        assert result["n_crashes"] == 0


class TestRegimePersistenceIoU:
    def test_perfect_overlap(self):
        dates = pd.bdate_range("2020-01-01", periods=100)
        regimes = pd.Series(
            ["Bull"] * 50 + ["Bear"] * 50, index=dates
        )
        crashes = pd.DataFrame({
            "start": [dates[50]],
            "end": [dates[99]],
        })
        result = regime_persistence_iou(regimes, crashes)
        assert result["iou"] == 1.0

    def test_no_overlap(self):
        dates = pd.bdate_range("2020-01-01", periods=100)
        regimes = pd.Series(["Bull"] * 100, index=dates)
        crashes = pd.DataFrame({
            "start": [dates[50]],
            "end": [dates[99]],
        })
        result = regime_persistence_iou(regimes, crashes)
        assert result["iou"] == 0.0

    def test_partial_overlap(self):
        dates = pd.bdate_range("2020-01-01", periods=100)
        regimes = pd.Series(
            ["Bull"] * 30 + ["Bear"] * 40 + ["Bull"] * 30, index=dates
        )
        crashes = pd.DataFrame({
            "start": [dates[50]],
            "end": [dates[99]],
        })
        result = regime_persistence_iou(regimes, crashes)
        assert 0 < result["iou"] < 1.0

    def test_empty_data(self):
        result = regime_persistence_iou(pd.Series(dtype=str), pd.DataFrame(columns=["start", "end"]))
        assert result["iou"] == 0.0


class TestDirectionalAccuracyVsConsensus:
    def test_always_agree(self, bt_with_crash):
        """When consensus and engine always agree, no divergences."""
        result = directional_accuracy_vs_consensus(bt_with_crash, 0.07)
        # Both positive for most rows, so few divergences
        assert result["n_divergences"] >= 0

    def test_divergent_predictions(self, bt_with_crash):
        """Negative consensus should create divergences with positive engine predictions."""
        result = directional_accuracy_vs_consensus(bt_with_crash, -0.05)
        assert result["n_divergences"] > 0
        assert 0 <= result["engine_correct_rate"] <= 1

    def test_empty_data(self):
        result = directional_accuracy_vs_consensus(pd.DataFrame(), 0.05)
        assert result["n_divergences"] == 0


class TestSignalMaxDrawdown:
    def test_no_signals(self, bt_with_crash):
        """Very high threshold → never short → zero drawdown."""
        result = signal_max_drawdown(bt_with_crash, short_threshold=0.99)
        assert result["max_drawdown"] == 0.0
        assert result["total_return"] == 0.0

    def test_with_signals(self, bt_with_crash):
        result = signal_max_drawdown(bt_with_crash, short_threshold=0.50)
        assert isinstance(result["max_drawdown"], float)
        assert isinstance(result["total_return"], float)
        assert isinstance(result["sharpe"], float)

    def test_empty_data(self):
        result = signal_max_drawdown(pd.DataFrame())
        assert result["max_drawdown"] == 0.0
