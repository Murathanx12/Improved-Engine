"""
Module 6: Walk-Forward Backtest
================================

Honest out-of-sample validation using walk-forward methodology.
At each prediction point, uses ONLY data available up to that date.

Key metrics:
    - MAPE: Mean Absolute Percentage Error
    - 90% Band Coverage: How often actual falls within prediction interval
    - Directional Accuracy: % of correct up/down calls
    - Brier Score: Probability calibration quality (0 = perfect, 0.25 = random)
    - Crash Calibration: Binned predicted vs actual crash rates

Usage:
    from finpredict.simulation.backtest import run_backtest

    bt_results = run_backtest(data, crash_freq)
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from finpredict.config import config, get_institutional_return
from finpredict.simulation.monte_carlo import simulate_paths


def run_backtest(data: pd.DataFrame, crash_freq: float) -> pd.DataFrame:
    """
    Module 6 Entry Point: Walk-forward out-of-sample backtest.

    Args:
        data: Master DataFrame with SP500, Daily_Returns, Risk_Score columns
        crash_freq: Historical crash frequency (annual)

    Returns:
        pd.DataFrame with prediction results (with metrics in .attrs)
    """
    print("[MODULE 6] Running walk-forward backtest (2000 → present)...")

    bt_cfg = config["backtest"]
    risk_cfg = config["risk"]

    backtest_start = pd.Timestamp(config["data"]["backtest_start"])
    forward_days = bt_cfg["forward_days"]

    # Generate prediction dates every N months
    pred_dates = pd.date_range(
        backtest_start,
        data.index[-1] - pd.Timedelta(days=forward_days + 30),
        freq=f'{bt_cfg["step_months"]}MS',
    )

    results = []

    for pred_date in tqdm(pred_dates, desc="  Backtesting"):
        idx = data.index.get_indexer([pred_date], method="nearest")[0]
        if idx < 252 * 5:  # Need at least 5 years of history
            continue

        # CRITICAL: Only use data UP TO this date (no future leakage)
        lookback = bt_cfg["lookback_years"] * 252
        hist = data.iloc[max(0, idx - lookback):idx]
        if len(hist) < 252:
            continue

        returns = hist["SP500"].pct_change().dropna()
        log_returns = np.log(1 + returns)
        hist_mu = log_returns.mean() * 252  # Geometric annualized drift

        # Blend historical + institutional anchor (same as production model)
        inst_mu = get_institutional_return()
        mu = 0.5 * hist_mu + 0.5 * inst_mu

        sigma = returns.std() * np.sqrt(252)
        start_price = float(data["SP500"].iloc[idx])
        risk = float(data["Risk_Score"].iloc[idx]) if "Risk_Score" in data.columns else 0

        # Run simulation
        base = {"drift_adj": 0, "vol_mult": 1.0, "crash_mult": 1.0}
        paths = simulate_paths(
            start_price, mu, sigma, forward_days,
            bt_cfg["simulations_per_point"], crash_freq, risk, base,
        )

        pred_mean = float(paths[-1].mean())
        pred_median = float(np.median(paths[-1]))
        pred_p95 = float(np.percentile(paths[-1], 95))
        pred_p05 = float(np.percentile(paths[-1], 5))

        # Actual outcome
        actual_idx = min(idx + forward_days, len(data) - 1)
        actual_price = float(data["SP500"].iloc[actual_idx])

        # Actual crash check
        fwd_prices = data["SP500"].iloc[idx:actual_idx + 1]
        fwd_peak = fwd_prices.expanding().max()
        fwd_dd = ((fwd_prices - fwd_peak) / fwd_peak).min()
        actual_crash = fwd_dd <= -risk_cfg["crash_threshold"]

        # Model's crash probability (peak-to-trough)
        sim_peak = np.maximum.accumulate(paths, axis=0)
        sim_dd = (paths - sim_peak) / sim_peak
        sim_max_dd = sim_dd.min(axis=0)
        pred_crash_prob = float((sim_max_dd <= -risk_cfg["crash_threshold"]).mean())

        error = pred_mean - actual_price
        pct_error = (error / actual_price) * 100
        within_bounds = (actual_price >= pred_p05) and (actual_price <= pred_p95)

        results.append({
            "date": pred_date,
            "actual_price": actual_price,
            "pred_mean": pred_mean,
            "pred_p95": pred_p95,
            "pred_p05": pred_p05,
            "pct_error": pct_error,
            "within_bounds": within_bounds,
            "actual_crash": actual_crash,
            "pred_crash_prob": pred_crash_prob,
            "risk_score": risk,
            "direction_correct": (pred_mean > start_price) == (actual_price > start_price),
        })

    bt = pd.DataFrame(results)

    if len(bt) > 0:
        mape = np.abs(bt["pct_error"]).mean()
        coverage = bt["within_bounds"].mean() * 100
        direction = bt["direction_correct"].mean() * 100

        # Brier score
        bt["actual_crash_int"] = bt["actual_crash"].astype(float)
        brier_score = float(((bt["pred_crash_prob"] - bt["actual_crash_int"]) ** 2).mean())

        # Calibration bins
        low_risk = bt[bt["pred_crash_prob"] < 0.15]
        med_risk = bt[(bt["pred_crash_prob"] >= 0.15) & (bt["pred_crash_prob"] < 0.40)]
        high_risk = bt[bt["pred_crash_prob"] >= 0.40]

        cal_low = float(low_risk["actual_crash"].mean() * 100) if len(low_risk) > 0 else 0
        cal_med = float(med_risk["actual_crash"].mean() * 100) if len(med_risk) > 0 else 0
        cal_high = float(high_risk["actual_crash"].mean() * 100) if len(high_risk) > 0 else 0

        print(f"\n  [RESULTS] Walk-Forward Backtest:")
        print(f"  Predictions:        {len(bt)}")
        print(f"  MAPE:               {mape:.1f}%")
        print(f"  90% Band Coverage:  {coverage:.1f}%")
        print(f"  Directional Acc:    {direction:.1f}%")
        print(f"  Crash Calibration:")
        print(f"    Low risk  (<15%): {cal_low:.0f}% actual ({len(low_risk)} predictions)")
        print(f"    Med risk  (15-40%): {cal_med:.0f}% actual ({len(med_risk)} predictions)")
        print(f"    High risk (>40%): {cal_high:.0f}% actual ({len(high_risk)} predictions)")
        print(f"  Brier Score:        {brier_score:.4f}\n")

        bt.attrs.update({
            "mape": mape, "coverage": coverage, "direction": direction,
            "brier_score": brier_score,
            "cal_low": cal_low, "cal_med": cal_med, "cal_high": cal_high,
            "n_low": len(low_risk), "n_med": len(med_risk), "n_high": len(high_risk),
        })

    return bt
