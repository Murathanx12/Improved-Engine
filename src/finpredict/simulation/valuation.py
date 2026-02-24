"""
Valuation Constraint
=====================

Applies mean-reversion drag to drift when valuations are stretched.

Two methods (uses best available):
    1. CAPE ratio (preferred): Shiller CAPE from Yale/yfinance
       Research shows CAPE explains ~40% of subsequent 10-year returns.
    2. Trend deviation (fallback): Compare price to long-run expected level

Usage:
    from finpredict.simulation.valuation import compute_valuation_penalty

    penalty, cape_value = compute_valuation_penalty(data)
"""

import numpy as np
import pandas as pd
import yfinance as yf

from finpredict.config import config


def _fetch_cape_ratio() -> float | None:
    """
    Fetch the current Shiller CAPE (Cyclically Adjusted P/E) ratio.

    Tries multiple sources with graceful fallback:
        1. Shiller's Yale Excel dataset
        2. yfinance trailing P/E as proxy
    """
    # Method 1: Shiller's own dataset
    try:
        url = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
        df = pd.read_excel(url, sheet_name="Data", skiprows=7, usecols=[0, 1, 10])
        df.columns = ["Date", "Price", "CAPE"]
        df = df.dropna(subset=["CAPE"])
        cape = float(df["CAPE"].iloc[-1])
        if 5 < cape < 100:
            return cape
    except Exception:
        pass

    # Method 2: yfinance trailing P/E as rough proxy
    try:
        sp_info = yf.Ticker("^GSPC").info
        pe = sp_info.get("trailingPE", None)
        if pe and 5 < pe < 100:
            return pe * 1.0
    except Exception:
        pass

    return None


def compute_valuation_penalty(data: pd.DataFrame) -> tuple[float, float | None]:
    """
    Compute valuation-based return penalty/boost.

    Args:
        data: DataFrame with 'SP500' column

    Returns:
        penalty: Annual return adjustment (negative = drag, positive = boost)
        cape_value: Current CAPE ratio if available, else None
    """
    val_cfg = config["simulation"]["valuation"]
    cape_avg = val_cfg["cape_long_run_average"]
    penalty_factor = val_cfg["cape_penalty_factor"]

    # Method 1: CAPE ratio
    cape = _fetch_cape_ratio()
    if cape is not None:
        deviation = (cape / cape_avg) - 1
        penalty = -deviation * penalty_factor
        penalty = float(np.clip(penalty, -0.03, 0.03))
        print(f"  [CAPE] Current CAPE ratio: {cape:.1f} (avg: {cape_avg:.0f})")
        print(f"  [CAPE] Valuation penalty: {penalty*100:+.2f}%/yr")
        return penalty, cape

    # Method 2: Trend deviation fallback
    print("  [WARN] CAPE data unavailable — using trend deviation proxy")
    sp = data["SP500"].dropna()
    if len(sp) < 252 * 15:
        return 0.0, None

    anchor_years = min(20, len(sp) / 252)
    anchor_idx = max(0, len(sp) - int(anchor_years * 252))
    anchor_price = float(sp.iloc[anchor_idx])
    current_price = float(sp.iloc[-1])

    long_run_nominal = val_cfg["long_run_real_return"] + val_cfg["inflation_target"]
    expected_price = anchor_price * (1 + long_run_nominal) ** anchor_years

    deviation = (current_price / expected_price) - 1
    penalty = float(np.clip(-deviation * penalty_factor, -0.03, 0.03))

    return penalty, None
