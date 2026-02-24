"""
Module 4: Historical Crash Analysis
=====================================

Identifies all historical crashes (≥20% peak-to-trough drawdown) and
computes base rates for crash probability estimation.

How it works:
    1. Compute running peak and drawdown for entire S&P 500 history
    2. Mark ENTRY into crash when drawdown exceeds -20% from peak
    3. Mark EXIT when price recovers to within 5% of prior peak
    4. Record each crash with dates, depth, duration

Why the exit condition matters:
    The standard convention (recover to within 5% of peak) prevents
    double-counting in extended bear markets (2000-2002, 2007-2009)
    where partial recoveries followed by further drops would otherwise
    register as multiple crashes.

    Historical benchmark: ~15 bear markets since 1928, roughly 1 per 6-7 years.

Usage:
    from finpredict.risk.crashes import identify_crashes

    crash_df, crash_freq = identify_crashes(data)
"""

import pandas as pd

from finpredict.config import config


def identify_crashes(
    data: pd.DataFrame,
    threshold: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """
    Identify all crash periods in the S&P 500 history.

    Args:
        data: DataFrame with 'SP500' column
        threshold: Drawdown threshold for crash definition.
                   Defaults to config risk.crash_threshold (0.20).

    Returns:
        crash_df: DataFrame of crash events with start, end, max_dd, duration
        crash_freq: Annual crash frequency (crashes per year)
    """
    print("[MODULE 4] Analyzing historical crash patterns...")

    if threshold is None:
        threshold = config["risk"]["crash_threshold"]

    severe_threshold = config["risk"]["severe_threshold"]
    exit_recovery = config["risk"]["crash_exit_recovery"]

    peak = data["SP500"].expanding().max()
    drawdown = (data["SP500"] - peak) / peak

    crashes = []
    in_crash = False
    crash_start = None
    crash_min = 0.0

    for i in range(len(data)):
        dd = drawdown.iloc[i]
        if dd <= -threshold and not in_crash:
            in_crash = True
            crash_start = data.index[i]
            crash_min = dd
        elif in_crash:
            crash_min = min(crash_min, dd)
            # Standard exit: recover to within exit_recovery of prior peak
            if dd > -exit_recovery:
                crashes.append({
                    "start": crash_start,
                    "end": data.index[i],
                    "max_dd": crash_min,
                    "duration_days": (data.index[i] - crash_start).days,
                    "severity": "Severe" if crash_min <= -severe_threshold else "Moderate",
                })
                in_crash = False

    crash_df = pd.DataFrame(crashes)
    years = (data.index[-1] - data.index[0]).days / 365.25
    freq = len(crashes) / years if years > 0 else 0.05

    print(f"  [OK] {len(crashes)} crashes detected since {data.index[0].year}")
    if freq > 0:
        print(f"  [OK] Frequency: once every {1/freq:.1f} years")
    print()

    return crash_df, freq
