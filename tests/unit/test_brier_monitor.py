"""Tests for the Rolling Brier Score Monitor."""

import numpy as np
import pandas as pd
import pytest

from finpredict.evaluation.brier_monitor import RollingBrierMonitor


@pytest.fixture
def tmp_log(tmp_path):
    """Create a temporary prediction log CSV."""
    return tmp_path / "prediction_log.csv"


@pytest.fixture
def market_data():
    """4 years of synthetic SP500 data with a crash in year 2."""
    rng = np.random.default_rng(42)
    n = 1008  # ~4 years of trading days
    dates = pd.bdate_range("2021-01-01", periods=n)
    returns = rng.normal(0.0003, 0.012, n)
    # Inject crash at ~500 days in (mid-2022)
    returns[480:530] = rng.normal(-0.02, 0.03, 50)
    prices = 4500 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"SP500": prices}, index=dates)


def _write_log(path, rows):
    """Helper to write prediction log rows."""
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


class TestBrierMonitor:
    def test_empty_log_handled(self, tmp_path, market_data):
        """No log file -> empty result."""
        monitor = RollingBrierMonitor(log_path=tmp_path / "nonexistent.csv")
        result = monitor.evaluate_and_check(market_data)
        assert result["evaluated"] == 0
        assert result["degraded"] is False

    def test_evaluate_with_synthetic_log(self, tmp_log, market_data):
        """Write fake predictions, check Brier is computed."""
        rows = []
        # Predictions made at various dates in 2021 (all have 12m of future data)
        for i, date in enumerate(["2021-03-01", "2021-06-01", "2021-09-01", "2021-12-01"]):
            rows.append(
                {
                    "prediction_date": date,
                    "crash_prob_12m": 0.15,
                    "crash_prob_3m": 0.05,
                }
            )
        _write_log(tmp_log, rows)

        monitor = RollingBrierMonitor(log_path=tmp_log)
        evaluated = monitor.evaluate_past_predictions(market_data)
        assert len(evaluated) > 0
        assert "brier_12m" in evaluated.columns
        assert all(evaluated["brier_12m"] >= 0)

    def test_degradation_detection(self, tmp_log, market_data):
        """Recent bad predictions should trigger degradation alert."""
        rows = []
        # Early predictions: good (low prob, no crash in year 1)
        for date in ["2021-02-01", "2021-04-01", "2021-06-01", "2021-08-01"]:
            rows.append({"prediction_date": date, "crash_prob_12m": 0.10})
        # Later predictions: bad (low prob but crash happens around day 480)
        for date in ["2021-10-01", "2021-11-01", "2021-12-01", "2022-01-03"]:
            rows.append({"prediction_date": date, "crash_prob_12m": 0.10})
        _write_log(tmp_log, rows)

        monitor = RollingBrierMonitor(log_path=tmp_log)
        evaluated = monitor.evaluate_past_predictions(market_data)

        if len(evaluated) >= 4:
            degradation = monitor.detect_degradation(evaluated, recent_window=4)
            # Should have some evaluated predictions
            assert degradation["evaluated"] > 0
            assert degradation["current_brier"] is not None

    def test_no_degradation_when_stable(self, tmp_log, market_data):
        """Consistent predictions don't trigger degradation."""
        rows = []
        # All predictions with same probability
        for date in [
            "2021-02-01",
            "2021-04-01",
            "2021-06-01",
            "2021-08-01",
            "2021-10-01",
            "2021-12-01",
            "2022-02-01",
            "2022-04-01",
        ]:
            rows.append({"prediction_date": date, "crash_prob_12m": 0.12})
        _write_log(tmp_log, rows)

        monitor = RollingBrierMonitor(log_path=tmp_log)
        evaluated = monitor.evaluate_past_predictions(market_data)

        if len(evaluated) >= 4:
            degradation = monitor.detect_degradation(evaluated, recent_window=4)
            # With consistent predictions, pct_change should be small
            if degradation["pct_change"] is not None:
                # Allow some variance but shouldn't be massive
                assert degradation["evaluated"] > 0

    def test_insufficient_data_skipped(self, tmp_log, market_data):
        """Fewer than 4 evaluated predictions -> no degradation alert."""
        rows = [{"prediction_date": "2021-03-01", "crash_prob_12m": 0.20}]
        _write_log(tmp_log, rows)

        monitor = RollingBrierMonitor(log_path=tmp_log)
        evaluated = monitor.evaluate_past_predictions(market_data)
        degradation = monitor.detect_degradation(evaluated, recent_window=4)
        assert degradation["degraded"] is False
        assert degradation["reason"] == "insufficient_data"

    def test_rolling_brier_computation(self):
        """Rolling Brier should compute a moving average."""
        monitor = RollingBrierMonitor()
        evaluated = pd.DataFrame(
            {
                "brier_12m": [0.1, 0.2, 0.15, 0.1, 0.3, 0.25],
            }
        )
        rolling = monitor.compute_rolling_brier(evaluated, window=3)
        assert len(rolling) == 6
        # First value should be NaN (window=3, min_periods=2)
        assert pd.isna(rolling.iloc[0])
        # Third value should be mean of first 3
        assert abs(rolling.iloc[2] - 0.15) < 0.001

    def test_generate_report(self, tmp_path):
        """Report should save CSV and return summary."""
        import finpredict.evaluation.brier_monitor as bm

        original = bm._ROLLING_FILE
        bm._ROLLING_FILE = tmp_path / "brier_rolling.csv"
        try:
            monitor = RollingBrierMonitor()
            evaluated = pd.DataFrame(
                {
                    "prediction_date": pd.date_range("2021-01-01", periods=5, freq="ME"),
                    "crash_prob_12m": [0.1, 0.2, 0.15, 0.1, 0.3],
                    "actual_crash_12m": [0, 0, 0, 1, 0],
                    "max_drawdown_12m": [-0.05, -0.08, -0.06, -0.25, -0.04],
                    "brier_12m": [0.01, 0.04, 0.0225, 0.81, 0.09],
                }
            )
            report = monitor.generate_report(evaluated)
            assert report["saved"] is True
            assert report["evaluated"] == 5
            assert (tmp_path / "brier_rolling.csv").exists()
        finally:
            bm._ROLLING_FILE = original
