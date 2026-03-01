"""Feature & target builders used by the ML pipeline.

Builds 80+ backward-looking ML features from market data and optional
FRED macro time series. Features span these categories:

    1. Price momentum & returns (multiple horizons)
    2. Volatility (realized, ratios, higher moments)
    3. Trend & technical (SMA, EMA, RSI, MACD, Bollinger)
    4. Fixed income & macro (yields, spreads, VIX dynamics)
    5. Tail risk (drawdowns, downside measures, CVaR)
    6. Cross-asset (gold/equity, bond/equity, breadth)
    7. Interaction features (vol×mom, vix×spread, etc.)
    8. FRED macro time series (if provided)

ALL features are strictly backward-looking — no future data leakage.
"""
from typing import Dict
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
# MAIN FEATURE BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_feature_matrix(data: pd.DataFrame, fred_data: dict = None) -> pd.DataFrame:
    """Build 80+ backward-looking features from market data and optional FRED macro.

    Args:
        data: DataFrame with columns SP500, VIX, T10Y, T3M, Gold, NASDAQ,
              Russell, HYG, LQD, etc. (missing columns handled gracefully)
        fred_data: Optional dict of FRED time series

    Returns:
        DataFrame aligned with data.index, all features backward-looking
    """
    df = pd.DataFrame(index=data.index)
    sp = data["SP500"]

    # ═══════════════════════════════════════════════════════════════
    # 1. CORE RETURNS
    # ═══════════════════════════════════════════════════════════════
    df["daily_ret"] = sp.pct_change()
    df["log_ret"] = np.log(1 + df["daily_ret"]).replace([np.inf, -np.inf], np.nan)

    # ═══════════════════════════════════════════════════════════════
    # 2. PRICE MOMENTUM (multiple horizons)
    # ═══════════════════════════════════════════════════════════════
    for days, name in [(5, "1w"), (10, "2w"), (21, "1m"), (42, "2m"),
                       (63, "3m"), (126, "6m"), (252, "12m")]:
        df[f"mom_{name}"] = sp.pct_change(days)

    # Distance from 52-week high and low
    high_252 = sp.rolling(252).max()
    low_252 = sp.rolling(252).min()
    df["dist_52w_high"] = (sp - high_252) / high_252  # Always <= 0
    df["dist_52w_low"] = (sp - low_252) / low_252     # Always >= 0

    # Current drawdown from rolling peak
    df["drawdown_from_peak"] = (sp - high_252) / high_252

    # ═══════════════════════════════════════════════════════════════
    # 3. VOLATILITY (realized, ratios, higher moments)
    # ═══════════════════════════════════════════════════════════════
    log_ret = df["log_ret"]
    for days, name in [(5, "1w"), (10, "2w"), (21, "1m"), (63, "3m"),
                       (126, "6m"), (252, "12m")]:
        df[f"vol_{name}"] = log_ret.rolling(days).std()

    # Volatility ratios — detect regime shifts
    df["vol_ratio_1m_3m"] = df["vol_1m"] / df["vol_3m"].replace(0, np.nan)
    df["vol_ratio_1m_12m"] = df["vol_1m"] / df["vol_12m"].replace(0, np.nan)
    df["vol_ratio_1w_1m"] = df["vol_1w"] / df["vol_1m"].replace(0, np.nan)

    # Vol-of-vol — second order volatility
    df["vol_of_vol"] = df["vol_1m"].rolling(63).std()

    # Higher moments
    df["realized_skew"] = log_ret.rolling(63).skew()
    df["realized_kurt"] = log_ret.rolling(63).apply(
        lambda x: pd.Series(x).kurtosis(), raw=False
    )

    # Worst single day in recent windows
    df["max_daily_loss_21d"] = log_ret.rolling(21).min()
    df["max_daily_loss_63d"] = log_ret.rolling(63).min()

    # Vol z-score: is current vol unusually high/low relative to its own history?
    vol_12m_mean = df["vol_1m"].rolling(252).mean()
    vol_12m_std = df["vol_1m"].rolling(252).std()
    df["vol_zscore"] = (df["vol_1m"] - vol_12m_mean) / vol_12m_std.replace(0, np.nan)

    # ═══════════════════════════════════════════════════════════════
    # 4. TREND & TECHNICAL INDICATORS
    # ═══════════════════════════════════════════════════════════════
    # Simple moving averages & deviations from trend
    for days, name in [(50, "50d"), (100, "100d"), (200, "200d")]:
        sma = sp.rolling(days).mean()
        df[f"sma_{name}_dev"] = (sp - sma) / sma

    # Golden/death cross: SMA50 vs SMA200
    sma_50 = sp.rolling(50).mean()
    sma_200 = sp.rolling(200).mean()
    df["golden_cross"] = (sma_50 > sma_200).astype(float)

    # Exponential moving averages for MACD
    ema_12 = sp.ewm(span=12, adjust=False).mean()
    ema_26 = sp.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_signal"] = (macd_line - macd_signal) / sp  # Normalized

    # RSI (14-day)
    delta = sp.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14d"] = 100 - (100 / (1 + rs))
    df["rsi_14d_norm"] = df["rsi_14d"] / 100  # Normalize to [0, 1]

    # Bollinger Band position (where price sits in the band)
    bb_mid = sp.rolling(20).mean()
    bb_std = sp.rolling(20).std()
    df["bollinger_pos"] = (sp - bb_mid) / (2 * bb_std).replace(0, np.nan)

    # Trend strength: momentum relative to volatility (like a signal-to-noise)
    df["trend_strength_3m"] = df["mom_3m"] / df["vol_3m"].replace(0, np.nan)
    df["trend_strength_12m"] = df["mom_12m"] / df["vol_12m"].replace(0, np.nan)

    # ═══════════════════════════════════════════════════════════════
    # 5. FIXED INCOME & VIX DYNAMICS
    # ═══════════════════════════════════════════════════════════════
    if "VIX" in data.columns:
        vix = data["VIX"].ffill()
        df["vix"] = vix
        df["vix_change_1m"] = vix.pct_change(21)
        df["vix_change_3m"] = vix.pct_change(63)
        vix_mean = vix.rolling(252).mean()
        vix_std = vix.rolling(252).std()
        df["vix_zscore"] = (vix - vix_mean) / vix_std.replace(0, np.nan)
        # VIX term structure proxy: VIX vs realized vol
        realized_vol_annual = df["vol_1m"] * np.sqrt(252) * 100  # Match VIX scale
        df["vix_term_structure"] = (vix - realized_vol_annual) / vix.replace(0, np.nan)

    if "T10Y" in data.columns:
        df["yield_10y"] = data["T10Y"]
        df["yield_10y_change_1m"] = data["T10Y"].diff(21)
        df["yield_10y_change_3m"] = data["T10Y"].diff(63)

    if "T3M" in data.columns:
        df["yield_3m"] = data["T3M"]

    if "T10Y" in data.columns and "T3M" in data.columns:
        spread = data["T10Y"] - data["T3M"]
        df["term_spread"] = spread
        df["term_spread_change_1m"] = spread.diff(21)
        df["term_spread_change_3m"] = spread.diff(63)
        df["yield_curve_inverted"] = (spread < 0).astype(float)

    if "T30Y" in data.columns and "T10Y" in data.columns:
        df["long_short_spread"] = data["T30Y"] - data["T10Y"]

    # Credit spread proxies
    if "HYG" in data.columns and "LQD" in data.columns:
        credit_ratio = data["HYG"] / data["LQD"].replace(0, np.nan)
        df["credit_spread_proxy"] = credit_ratio.pct_change(21)
        df["credit_spread_level"] = credit_ratio

    # ═══════════════════════════════════════════════════════════════
    # 6. TAIL RISK FEATURES
    # ═══════════════════════════════════════════════════════════════
    # Rolling max drawdown (backward-looking)
    for days, name in [(63, "3m"), (252, "12m")]:
        rolling_max = sp.rolling(days).max()
        dd = (sp - rolling_max) / rolling_max
        df[f"max_drawdown_{name}"] = dd.rolling(days).min()

    # Lower partial moment (downside semi-variance)
    neg_ret = log_ret.clip(upper=0)
    df["lower_partial_moment"] = neg_ret.rolling(63).apply(
        lambda x: np.sqrt((x**2).mean()), raw=True
    )

    # Historical CVaR (5th percentile expected shortfall)
    df["cvar_5pct_63d"] = log_ret.rolling(63).apply(
        lambda x: x[x <= np.percentile(x, 5)].mean() if len(x) > 5 else np.nan,
        raw=True,
    )

    # Fraction of negative days
    df["neg_day_ratio_21d"] = (log_ret < 0).rolling(21).mean()
    df["neg_day_ratio_63d"] = (log_ret < 0).rolling(63).mean()

    # Consecutive down days (current streak)
    is_down = (log_ret < 0).astype(float)
    # Use cumsum trick to get current streak length
    not_down = (is_down == 0).astype(float)
    group = not_down.cumsum()
    df["down_streak"] = is_down.groupby(group).cumsum()

    # ═══════════════════════════════════════════════════════════════
    # 7. CROSS-ASSET FEATURES
    # ═══════════════════════════════════════════════════════════════
    sp_ret = df["daily_ret"]

    if "Gold" in data.columns:
        gold_ret = data["Gold"].pct_change()
        df["gold_equity_ratio"] = data["Gold"] / sp
        df["gold_equity_ratio_change_3m"] = df["gold_equity_ratio"].pct_change(63)
        df["gold_equity_corr_63d"] = sp_ret.rolling(63).corr(gold_ret)

    if "NASDAQ" in data.columns:
        nasdaq_ret = data["NASDAQ"].pct_change()
        df["sp_nasdaq_ratio"] = sp / data["NASDAQ"].replace(0, np.nan)
        df["sp_nasdaq_corr_63d"] = sp_ret.rolling(63).corr(nasdaq_ret)

    if "Russell" in data.columns:
        russell_ret = data["Russell"].pct_change()
        df["small_large_ratio"] = data["Russell"] / sp
        df["small_large_change_3m"] = df["small_large_ratio"].pct_change(63)

    # Market breadth: dispersion across sectors
    sector_cols = [c for c in data.columns if c.startswith("Sector_")]
    if len(sector_cols) >= 3:
        sector_returns = data[sector_cols].pct_change()
        df["sector_dispersion"] = sector_returns.std(axis=1)
        df["sector_dispersion_63d"] = df["sector_dispersion"].rolling(63).mean()

    # Bond-equity correlation (if T10Y available as price proxy)
    if "T10Y" in data.columns:
        yield_change = data["T10Y"].diff()
        df["bond_equity_corr_63d"] = sp_ret.rolling(63).corr(yield_change)

    # ═══════════════════════════════════════════════════════════════
    # 8. INTERACTION FEATURES
    # ═══════════════════════════════════════════════════════════════
    # Volatility × momentum (high vol + negative momentum = danger)
    df["vol_x_mom_3m"] = df["vol_1m"] * df["mom_3m"]
    df["vol_x_mom_12m"] = df["vol_1m"] * df["mom_12m"]

    # VIX × yield curve (high VIX + inverted curve = recession signal)
    if "vix" in df.columns and "term_spread" in df.columns:
        df["vix_x_spread"] = df["vix"] * df["term_spread"]

    # Vol × drawdown (high vol during drawdown = panic)
    df["vol_x_drawdown"] = df["vol_1m"] * df["drawdown_from_peak"]

    # Momentum × RSI (overbought/oversold confirmation)
    if "rsi_14d_norm" in df.columns:
        df["mom_x_rsi"] = df["mom_3m"] * df["rsi_14d_norm"]

    # Vol regime × trend (vol expansion during trend break)
    df["vol_ratio_x_trend"] = df["vol_ratio_1m_3m"] * df["sma_50d_dev"]

    # Drawdown × VIX (market stress compound indicator)
    if "vix" in df.columns:
        df["drawdown_x_vix"] = df["drawdown_from_peak"] * df["vix"]
        df["vix_x_mom"] = df["vix"] * df["mom_1m"]

    # Yield spread × vol (tightening + high vol = stress)
    if "term_spread" in df.columns:
        df["spread_x_vol"] = df["term_spread"] * df["vol_1m"]

    # ═══════════════════════════════════════════════════════════════
    # 9. FRED MACRO FEATURES (time-varying, if provided)
    # ═══════════════════════════════════════════════════════════════
    if fred_data:
        for k, series in fred_data.items():
            try:
                s = pd.Series(series).astype(float)
                s.index = pd.to_datetime(s.index)
                s = s.reindex(df.index).ffill()
                df[f"fred_{k}"] = s
            except Exception:
                continue

    # ═══════════════════════════════════════════════════════════════
    # 10. FINAL CLEANUP
    # ═══════════════════════════════════════════════════════════════
    # Drop the raw price column (SP500 itself is not a feature)
    # Keep only derived features
    df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    return df


# ═══════════════════════════════════════════════════════════════════════
# TARGET BUILDERS
# ═══════════════════════════════════════════════════════════════════════

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

    Uses vectorized numpy approach for performance.
    Returns the minimum drawdown (most negative) observed in the forward window.
    """
    vals = prices.values.astype(float)
    n = len(vals)
    out = np.full(n, np.nan)

    # Vectorized approach: precompute cumulative max for all windows
    for i in range(n - 1):
        end = min(n, i + days + 1)
        window = vals[i:end]
        if len(window) <= 1:
            continue
        peak = np.maximum.accumulate(window)
        dd = (window - peak) / np.where(peak > 0, peak, 1.0)
        out[i] = dd.min()

    return pd.Series(out, index=prices.index)


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
