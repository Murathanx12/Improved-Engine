"""Feature & target builders used by the ML pipeline.

This is a minimal, robust implementation sufficient for the backtest
and main pipeline to run. It builds simple technical/macro features
and multi-horizon targets for returns and crash (max drawdown).
"""
from typing import Dict
import numpy as np
import pandas as pd


def build_feature_matrix(data: pd.DataFrame, fred_data: dict = None) -> pd.DataFrame:
    """Build a simple feature matrix from market `data` and optional `fred_data`.

    Returns a DataFrame aligned with `data.index`.
    """
    df = pd.DataFrame(index=data.index)

    # Core price-derived features
    df["SP500"] = data["SP500"]
    df["daily_ret"] = data["SP500"].pct_change()
    df["log_ret"] = np.log(1 + df["daily_ret"]).replace([np.inf, -np.inf], np.nan)

    # Momentum features: 1m, 3m, 12m (using trading days ~21,63,252)
    df["mom_1m"] = data["SP500"].pct_change(21)
    df["mom_3m"] = data["SP500"].pct_change(63)
    df["mom_12m"] = data["SP500"].pct_change(252)

    # Volatility features
    df["vol_1m"] = df["log_ret"].rolling(21).std()
    df["vol_3m"] = df["log_ret"].rolling(63).std()

    # Simple moving averages and deviations
    df["sma_3m"] = data["SP500"].rolling(63).mean()
    df["sma_12m"] = data["SP500"].rolling(252).mean()
    df["sma_3m_dev"] = (data["SP500"] - df["sma_3m"]) / df["sma_3m"]

    # Risk/recession proxies if available
    if "VIX" in data.columns:
        df["VIX"] = data["VIX"].fillna(method="ffill")
    if "T10Y" in data.columns and "T3M" in data.columns:
        df["term_spread"] = data["T10Y"] - data["T3M"]

    # Include any simple aggregated fred features if provided
    if fred_data:
        for k, series in fred_data.items():
            try:
                s = pd.Series(series).astype(float)
                s.index = pd.to_datetime(s.index)
                s = s.reindex(df.index).fillna(method="ffill")
                df[f"fred_{k}"] = s
            except Exception:
                # ignore malformed series
                continue

    # Final housekeeping
    df = df.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(0)
    return df


def _forward_return(series: pd.Series, days: int) -> pd.Series:
    return series.shift(-days) / series - 1.0


def build_target_return(data: pd.DataFrame, horizon_days: int = 252) -> pd.Series:
    """Return series over `horizon_days` forward (e.g., 252 days = 12m)."""
    return _forward_return(data["SP500"], horizon_days)


def build_target_return_multi(data: pd.DataFrame) -> Dict[str, pd.Series]:
    return {
        "12m": build_target_return(data, horizon_days=252),
        "6m": build_target_return(data, horizon_days=126),
        "3m": build_target_return(data, horizon_days=63),
    }


def _forward_max_drawdown(prices: pd.Series, days: int) -> pd.Series:
    """Compute forward-looking maximum drawdown over next `days` days.

    Returns the minimum drawdown (most negative) observed in the forward window.
    """
    n = len(prices)
    out = pd.Series(index=prices.index, dtype=float)
    for i in range(n):
        end = min(n, i + days + 1)
        fwd = prices.iloc[i:end]
        if len(fwd) <= 1:
            out.iat[i] = np.nan
            continue
        peak = fwd.cummax()
        dd = (fwd - peak) / peak
        out.iat[i] = dd.min()
    return out


def build_target_crash(data: pd.DataFrame, threshold: float = -0.2, horizon_days: int = 252) -> pd.Series:
    """Boolean series indicating if a crash (drawdown <= threshold) occurs within horizon."""
    mdd = _forward_max_drawdown(data["SP500"], horizon_days)
    return mdd <= threshold


def build_target_crash_multi(data: pd.DataFrame, threshold: float = -0.2) -> Dict[str, pd.Series]:
    return {
        "12m": build_target_crash(data, threshold=threshold, horizon_days=252),
        "6m": build_target_crash(data, threshold=threshold, horizon_days=126),
        "3m": build_target_crash(data, threshold=threshold, horizon_days=63),
    }
