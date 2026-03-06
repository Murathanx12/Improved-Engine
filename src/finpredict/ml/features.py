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
# HELPER: Fractional Differentiation (Lopez de Prado, Ch. 5)
# ═══════════════════════════════════════════════════════════════════════

def _frac_diff_ffd(series: pd.Series, d: float = 0.4, threshold: float = 1e-4) -> pd.Series:
    """Fixed-width window fractional differentiation.

    Preserves memory in price series while achieving stationarity.
    Better than raw log-returns for ML features because it retains
    long-range dependence that differencing destroys.

    Reference: "Advances in Financial Machine Learning" (2018), Chapter 5.

    Args:
        series: Price series (typically log prices)
        d: Fractional differentiation order (0 < d < 1). d=0.4 is a good
           balance between stationarity and memory preservation.
        threshold: Weight cutoff — stop when |w_k| < threshold.

    Returns:
        Fractionally differentiated series (same index, NaN for warmup period)
    """
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    weights = np.array(weights[::-1])
    width = len(weights)

    vals = series.values.astype(float)
    out = np.full(len(vals), np.nan)
    for i in range(width - 1, len(vals)):
        if not np.any(np.isnan(vals[i - width + 1: i + 1])):
            out[i] = np.dot(weights, vals[i - width + 1: i + 1])
    return pd.Series(out, index=series.index)


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

    # ── VIX Term Structure (options market signals) ──────────────
    # VIX/VIX3M > 1.0 = backwardation = short-term fear exceeds long-term
    # This has preceded nearly every major crash since 2008
    if "VIX" in data.columns and "VIX3M" in data.columns:
        vix3m = data["VIX3M"].ffill()
        df["vix_term_structure_ratio"] = data["VIX"].ffill() / vix3m.replace(0, np.nan)
        df["vix_backwardation"] = (df["vix_term_structure_ratio"] > 1.0).astype(float)
        # N2: VIX term structure velocity and persistence
        df["vix_ts_velocity_5d"] = df["vix_term_structure_ratio"].diff(5)
        df["vix_ts_velocity_21d"] = df["vix_term_structure_ratio"].diff(21)
        # Duration of backwardation (consecutive days)
        backw = df["vix_backwardation"]
        groups = (backw != backw.shift()).cumsum()
        df["vix_backwardation_duration"] = backw.groupby(groups).cumsum()

    # SKEW Index: measures institutional demand for tail-risk hedging
    # SKEW > 145 = elevated crash protection buying
    if "SKEW" in data.columns:
        skew_data = data["SKEW"].ffill()
        df["skew_index"] = skew_data
        skew_mean = skew_data.rolling(252).mean()
        skew_std = skew_data.rolling(252).std()
        df["skew_zscore"] = (skew_data - skew_mean) / skew_std.replace(0, np.nan)
        df["skew_elevated"] = (skew_data > 145).astype(float)

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
    df["vol_x_dist52w"] = df["vol_1m"] * df["dist_52w_high"]

    # Momentum × RSI (overbought/oversold confirmation)
    if "rsi_14d_norm" in df.columns:
        df["mom_x_rsi"] = df["mom_3m"] * df["rsi_14d_norm"]

    # Vol regime × trend (vol expansion during trend break)
    df["vol_ratio_x_trend"] = df["vol_ratio_1m_3m"] * df["sma_50d_dev"]

    # Drawdown × VIX (market stress compound indicator)
    if "vix" in df.columns:
        df["dist52w_x_vix"] = df["dist_52w_high"] * df["vix"]
        df["vix_x_mom"] = df["vix"] * df["mom_1m"]

    # Yield spread × vol (tightening + high vol = stress)
    if "term_spread" in df.columns:
        df["spread_x_vol"] = df["term_spread"] * df["vol_1m"]

    # VIX term structure × momentum (backwardation + negative momentum = high crash risk)
    if "vix_term_structure_ratio" in df.columns:
        df["vix_ts_x_mom"] = df["vix_term_structure_ratio"] * df["mom_1m"]

    # SKEW × drawdown (elevated tail hedging during drawdown = institutional panic)
    if "skew_zscore" in df.columns:
        df["skew_x_dist52w"] = df["skew_zscore"] * df["dist_52w_high"]

    # ═══════════════════════════════════════════════════════════════
    # 9. FRED MACRO FEATURES (time-varying, if provided)
    # ═══════════════════════════════════════════════════════════════
    if fred_data:
        # Collect all new columns in a dict then concat once to avoid fragmentation
        fred_cols: Dict[str, pd.Series] = {}
        for k, series in fred_data.items():
            try:
                s = pd.Series(series).astype(float)
                s.index = pd.to_datetime(s.index)
                # For monthly FRED data, shift by ~21 trading days to
                # account for publication lag (data not available until
                # weeks after the reference period). Daily series (e.g.
                # VIX, yields) have minimal lag and are not shifted.
                if len(s) > 10:
                    median_gap = s.index.to_series().diff().median()
                    is_monthly = median_gap > pd.Timedelta(days=15)
                else:
                    is_monthly = False
                s = s.reindex(df.index)
                if is_monthly:
                    s = s.shift(21)  # 21 trading days ≈ 1 month lag
                s = s.ffill()
                col = f"fred_{k}"
                fred_cols[col] = s
                # ── Rate-of-change derivatives (more predictive than raw levels) ──
                fred_cols[f"{col}_chg_3m"] = s.pct_change(63)
                fred_cols[f"{col}_chg_12m"] = s.pct_change(252)
                col_mean = s.rolling(252).mean()
                col_std = s.rolling(252).std()
                fred_cols[f"{col}_zscore"] = (s - col_mean) / col_std.replace(0, np.nan)
            except Exception:
                continue
        if fred_cols:
            df = pd.concat([df, pd.DataFrame(fred_cols, index=df.index)], axis=1)

    # ── FRED interaction features ─────────────────────────────────
    # Credit OAS × vol (widening spreads + rising vol = systemic stress)
    interaction_cols: Dict[str, pd.Series] = {}
    if "fred_hy_oas" in df.columns:
        interaction_cols["hy_oas_x_vol"] = df["fred_hy_oas"] * df["vol_1m"]
    # Geopolitical risk × momentum (rising GPR + falling market = crisis)
    if "fred_gpr_world" in df.columns:
        interaction_cols["gpr_x_mom"] = df["fred_gpr_world"] * df["mom_1m"]
    if interaction_cols:
        df = pd.concat([df, pd.DataFrame(interaction_cols, index=df.index)], axis=1)

    # ═══════════════════════════════════════════════════════════════
    # 10. NEAR-TERM STRESS SIGNALS (0-90 day precision)
    # ═══════════════════════════════════════════════════════════════
    # These signals specifically predict near-term (0-90 day) stress,
    # not 12-month horizon. They add 3-month crash timing precision.

    # 10.1 Options Fear Index (put/call ratio proxy)
    # Composite from existing VIX, SKEW, and VIX term structure
    fear_components = []
    if "vix_zscore" in df.columns:
        fear_components.append(df["vix_zscore"] * 0.4)
    if "skew_zscore" in df.columns:
        fear_components.append(df["skew_zscore"] * 0.3)
    if "vix_backwardation" in df.columns:
        fear_components.append(df["vix_backwardation"] * 0.6)  # Scale binary up
    if fear_components:
        df["options_fear_index"] = sum(fear_components)
        df["options_fear_5d_ma"] = df["options_fear_index"].rolling(5).mean()
        ofi_mean = df["options_fear_index"].rolling(252).mean()
        ofi_std = df["options_fear_index"].rolling(252).std()
        df["options_fear_extreme"] = (
            (df["options_fear_index"] - ofi_mean) / ofi_std.replace(0, np.nan) > 2.0
        ).astype(float)

    # 10.2 HY Spread Change Velocity
    if "fred_hy_oas" in df.columns:
        hy = df["fred_hy_oas"]
        df["hy_oas_chg_1w"] = hy.diff(5)
        df["hy_oas_chg_2w"] = hy.diff(10)
        df["hy_oas_chg_4w"] = hy.diff(21)
        df["hy_oas_accel"] = df["hy_oas_chg_1w"] - df["hy_oas_chg_1w"].shift(5)
        widening = (hy.diff() > 0).astype(float)
        not_widening = (widening == 0).astype(float)
        hy_groups = not_widening.cumsum()
        df["hy_oas_widening_streak"] = widening.groupby(hy_groups).cumsum()

    # 10.3 TED Spread Velocity
    if "fred_ted_spread" in df.columns:
        ted = df["fred_ted_spread"]
        df["ted_spread_chg_1w"] = ted.diff(5)
        df["ted_spread_chg_2w"] = ted.diff(10)
        df["ted_spread_chg_4w"] = ted.diff(21)
        ted_chg_mean = df["ted_spread_chg_1w"].rolling(252).mean()
        ted_chg_std = df["ted_spread_chg_1w"].rolling(252).std()
        df["ted_spread_velocity_zscore"] = (
            (df["ted_spread_chg_1w"] - ted_chg_mean) / ted_chg_std.replace(0, np.nan)
        )

    # 10.4 VIX Backwardation Duration & Intensity
    if "vix_backwardation" in df.columns:
        vb = df["vix_backwardation"]
        not_bw = (vb == 0).astype(float)
        bw_groups = not_bw.cumsum()
        df["vix_backwardation_duration"] = vb.groupby(bw_groups).cumsum()
        if "vix_term_structure_ratio" in df.columns:
            df["vix_backwardation_intensity"] = (
                df["vix_term_structure_ratio"] - 1.0
            ).clip(lower=0)
        df["vix_backwardation_5d_pct"] = vb.rolling(5).mean()
        df["vix_backwardation_21d_pct"] = vb.rolling(21).mean()

    # 10.5 Smart Money Proxy (insider selling alternative)
    if "small_large_ratio" in df.columns:
        slr = df["small_large_ratio"]
        slr_21d = slr.pct_change(21)
        slr_63d_trend = slr.pct_change(63)
        df["smart_money_proxy"] = slr_21d - slr_63d_trend
        neg_proxy = (df["smart_money_proxy"] < 0).astype(float)
        not_neg = (neg_proxy == 0).astype(float)
        neg_groups = not_neg.cumsum()
        df["smart_money_proxy_negative_streak"] = neg_proxy.groupby(neg_groups).cumsum()

    # 10.6 Realized Vol Trend
    if "vol_1w" in df.columns:
        vol_weekly = df["vol_1w"]
        df["realized_vol_trend_3w"] = vol_weekly.rolling(15).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 15 else np.nan,
            raw=True,
        )
        vol_rising = (vol_weekly.diff(5) > 0).astype(float)
        not_rising = (vol_rising == 0).astype(float)
        rise_groups = not_rising.cumsum()
        df["realized_vol_rising_weeks"] = vol_rising.groupby(rise_groups).cumsum()
        vol_chg1 = vol_weekly.diff(5)
        vol_chg2 = vol_weekly.diff(5).shift(5)
        df["realized_vol_accel"] = vol_chg1 - vol_chg2
        if "vol_3m" in df.columns:
            breakout = (vol_weekly > df["vol_3m"] * 1.5).astype(float)
            df["vol_breakout"] = breakout.rolling(3).min()

    # ═══════════════════════════════════════════════════════════════
    # 11. NEW PREDICTIVE FEATURES (from open-source research)
    # ═══════════════════════════════════════════════════════════════

    # 11.1 Equity Risk Premium (ERP)
    # price_to_avg_ratio = price / 10-year average price (NOT true CAPE = P/E10)
    # The inverse is used as a rough valuation signal: higher ratio = more expensive
    sp_10yr_avg = sp.rolling(2520, min_periods=1260).mean()
    price_to_avg_ratio = sp / sp_10yr_avg.replace(0, np.nan)
    inverse_valuation_ratio = 1.0 / price_to_avg_ratio.replace(0, np.nan)

    if "fred_tips_10y" in df.columns:
        real_yield = df["fred_tips_10y"] / 100
        df["erp"] = inverse_valuation_ratio - real_yield
    elif "fred_cpi" in df.columns and "yield_10y" in df.columns:
        inflation_proxy = df["fred_cpi"].pct_change(252)
        real_yield_proxy = df["yield_10y"] - inflation_proxy
        df["erp"] = inverse_valuation_ratio - real_yield_proxy
    else:
        df["erp"] = inverse_valuation_ratio - 0.02

    erp_mean = df["erp"].rolling(252).mean()
    erp_std = df["erp"].rolling(252).std()
    df["erp_zscore"] = (df["erp"] - erp_mean) / erp_std.replace(0, np.nan)
    df["erp_below_1pct"] = (df["erp"] < 0.01).astype(float)

    # 11.2 Market Concentration (HHI proxy)
    mega_cols = [c for c in data.columns if c.startswith("MegaCap_")]
    if len(mega_cols) >= 5:
        mega_total = data[mega_cols].sum(axis=1)
        sp_implied = sp * 500
        df["concentration_top7"] = mega_total / sp_implied.replace(0, np.nan)
        df["concentration_change_3m"] = df["concentration_top7"].pct_change(63)
        conc_mean = df["concentration_top7"].rolling(252).mean()
        conc_std = df["concentration_top7"].rolling(252).std()
        df["concentration_zscore"] = (
            (df["concentration_top7"] - conc_mean) / conc_std.replace(0, np.nan)
        )
    elif "sector_dispersion_63d" in df.columns:
        disp = df["sector_dispersion_63d"]
        disp_mean = disp.rolling(252).mean()
        disp_std = disp.rolling(252).std()
        df["concentration_proxy"] = -(disp - disp_mean) / disp_std.replace(0, np.nan)

    # 11.3 Margin Debt YoY Change
    if "fred_margin_credit" in df.columns:
        mc = df["fred_margin_credit"]
        df["margin_debt_yoy"] = mc.pct_change(252)
        df["margin_debt_declining"] = (df["margin_debt_yoy"] < 0).astype(float)

    # 11.4 Global Risk-Off Flows (TLT + UUP + GLD)
    risk_off_cols: Dict[str, pd.Series] = {}
    tlt_in = "TLT" in data.columns
    uup_in = "UUP" in data.columns
    gld_in = "GLD" in data.columns
    if tlt_in and uup_in and gld_in:
        tlt_ret_21d = data["TLT"].pct_change(21)
        uup_ret_21d = data["UUP"].pct_change(21)
        gld_ret_21d = data["GLD"].pct_change(21)
        risk_off_cols["risk_off_binary_21d"] = (
            (tlt_ret_21d > 0) & (uup_ret_21d > 0) & (gld_ret_21d > 0)
        ).astype(float)
        for rname, rret in [("tlt", tlt_ret_21d), ("uup", uup_ret_21d), ("gld", gld_ret_21d)]:
            rm = rret.rolling(252).mean()
            rs = rret.rolling(252).std()
            risk_off_cols[f"{rname}_ret_zscore"] = (rret - rm) / rs.replace(0, np.nan)
        risk_off_cols["risk_off_score_21d"] = (
            risk_off_cols["tlt_ret_zscore"]
            + risk_off_cols["uup_ret_zscore"]
            + risk_off_cols["gld_ret_zscore"]
        )
        # 63d version
        tlt63 = data["TLT"].pct_change(63)
        uup63 = data["UUP"].pct_change(63)
        gld63 = data["GLD"].pct_change(63)
        ro_63d_parts = []
        for rname, rret in [("tlt", tlt63), ("uup", uup63), ("gld", gld63)]:
            rm = rret.rolling(252).mean()
            rs = rret.rolling(252).std()
            ro_63d_parts.append((rret - rm) / rs.replace(0, np.nan))
        risk_off_cols["risk_off_score_63d"] = sum(ro_63d_parts)
    if risk_off_cols:
        df = pd.concat([df, pd.DataFrame(risk_off_cols, index=df.index)], axis=1)

    # 11.5 ISM Manufacturing PMI Proxy
    if "fred_mfg_employment" in df.columns:
        me = df["fred_mfg_employment"]
        df["mfg_employment_chg_3m"] = me.pct_change(63)
        df["mfg_employment_declining_3m"] = (me.diff(63) < 0).astype(float)
    if "fred_industrial_prod" in df.columns:
        ip = df["fred_industrial_prod"]
        df["industrial_prod_yoy"] = ip.pct_change(252)

    # 11.6 Corporate Bond Issuance / Business Lending
    if "fred_business_loans" in df.columns:
        bl = df["fred_business_loans"]
        df["business_loans_yoy"] = bl.pct_change(252)
        df["business_loans_accel_3m"] = df["business_loans_yoy"].diff(63)
        df["credit_crunch_signal"] = (df["business_loans_yoy"] < 0).astype(float)

    # 11.7 Cross-Asset Correlation Spike (SPX-TLT)
    if "TLT" in data.columns:
        tlt_daily_ret = data["TLT"].pct_change()
        df["spx_tlt_corr_30d"] = sp_ret.rolling(30).corr(tlt_daily_ret)
        df["spx_tlt_corr_63d"] = sp_ret.rolling(63).corr(tlt_daily_ret)
        df["spx_tlt_corr_positive"] = (df["spx_tlt_corr_30d"] > 0).astype(float)
        corr_mean = df["spx_tlt_corr_63d"].rolling(252).mean()
        corr_std = df["spx_tlt_corr_63d"].rolling(252).std()
        df["spx_tlt_corr_zscore"] = (
            (df["spx_tlt_corr_63d"] - corr_mean) / corr_std.replace(0, np.nan)
        )

    # 11.8 Fractionally Differentiated Price (Lopez de Prado)
    df["sp500_frac_diff"] = _frac_diff_ffd(np.log(sp), d=0.4)
    fd_mean = df["sp500_frac_diff"].rolling(252).mean()
    fd_std = df["sp500_frac_diff"].rolling(252).std()
    df["sp500_frac_diff_zscore"] = (
        (df["sp500_frac_diff"] - fd_mean) / fd_std.replace(0, np.nan)
    )

    # ═══════════════════════════════════════════════════════════════
    # 12. FINAL CLEANUP
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

    Optimized: processes in chunks to avoid O(n*window) Python-level loop.
    Returns the minimum drawdown (most negative) observed in the forward window.
    """
    vals = prices.values.astype(float)
    n = len(vals)
    out = np.full(n, np.nan)

    # Build a matrix of forward windows using stride tricks for speed
    # For very large arrays, process in manageable chunks
    chunk_size = 2000
    for chunk_start in range(0, n - 1, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n - 1)
        for i in range(chunk_start, chunk_end):
            end = min(n, i + days + 1)
            window = vals[i:end]
            if len(window) <= 1:
                continue
            peak = np.maximum.accumulate(window)
            mask = peak > 0
            dd_min = 0.0
            if mask.any():
                dd = np.where(mask, (window - peak) / peak, 0.0)
                dd_min = dd.min()
            out[i] = dd_min

    return pd.Series(out, index=prices.index)


def build_target_crash(
    data: pd.DataFrame,
    threshold: float = -0.2,
    horizon_days: int = 252,
    dynamic_vix: bool = False,
) -> pd.Series:
    """Boolean series indicating if a crash (drawdown <= threshold) occurs within horizon.

    If dynamic_vix=True and VIX is available, scale threshold by VIX regime:
        effective_threshold = base * (vix_long_run / vix_current)
    Low-vol regimes get a tighter threshold; high-vol regimes get a looser one.
    """
    from finpredict.config import config as _cfg

    mdd = _forward_max_drawdown(data["SP500"], horizon_days)

    dyn_cfg = _cfg.get("ml", {}).get("dynamic_crash_threshold", {})
    if dynamic_vix and dyn_cfg.get("enabled", False) and "VIX" in data.columns:
        vix_avg = dyn_cfg.get("vix_long_run_avg", 20.0)
        min_thresh = -dyn_cfg.get("min_threshold", 0.15)
        max_thresh = -dyn_cfg.get("max_threshold", 0.30)
        vix = data["VIX"].rolling(21).mean().ffill().bfill()
        scale = vix_avg / vix.clip(lower=10.0)
        dynamic_threshold = (threshold * scale).clip(upper=min_thresh, lower=max_thresh)
        return mdd <= dynamic_threshold

    return mdd <= threshold


def build_target_crash_multi(data: pd.DataFrame, threshold: float = -0.2) -> Dict[str, pd.Series]:
    return {
        "12m": build_target_crash(data, threshold=threshold, horizon_days=252),
        "6m": build_target_crash(data, threshold=threshold, horizon_days=126),
        "3m": build_target_crash(data, threshold=threshold, horizon_days=63),
    }


def build_target_crash_ensemble(
    data: pd.DataFrame,
    thresholds: list = None,
    horizon_days: int = 252,
) -> Dict[str, pd.Series]:
    """Build crash targets at multiple drawdown thresholds for ensemble training.

    Training on multiple thresholds (10%, 15%, 20%) gives more positive examples:
    - 10% corrections happen ~every 2 years (more training data)
    - 15% drawdowns happen ~every 4 years
    - 20% crashes happen ~every 9 years (sparse but most important)

    Returns:
        dict mapping threshold label to binary pd.Series
    """
    if thresholds is None:
        thresholds = [-0.10, -0.15, -0.20]
    mdd = _forward_max_drawdown(data["SP500"], horizon_days)
    return {f"thresh_{abs(t)*100:.0f}pct": mdd <= t for t in thresholds}
