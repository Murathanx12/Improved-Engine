"""
Global Market Crash Dataset & Contagion Features
===================================================

Fetches international equity indices and computes cross-market
contagion features for the ML pipeline.

Indices:
    FTSE 100, Nikkei 225, DAX, Hang Seng, Euro Stoxx 50, EEM

Features:
    - n_markets_in_drawdown: count of indices >10% off expanding peak
    - global_drawdown_avg: mean current drawdown across all indices

All features are backward-looking. Forward-filled to handle partial
availability across markets with different trading calendars.
"""

import time

import pandas as pd

from finpredict.config import config, PROJECT_ROOT


# Default global index tickers
GLOBAL_TICKERS = {
    "FTSE": "^FTSE",
    "Nikkei": "^N225",
    "DAX": "^GDAXI",
    "HangSeng": "^HSI",
    "EuroStoxx50": "^STOXX50E",
    "EEM": "EEM",
}


def fetch_global_indices(
    start: str = None,
    tickers: dict = None,
) -> dict[str, pd.Series]:
    """Fetch global equity index price series via yfinance.

    Args:
        start: Start date string (defaults to config training_start).
        tickers: Dict of {name: ticker}. Defaults to GLOBAL_TICKERS.

    Returns:
        Dict of {name: price_series} for each successfully fetched index.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  [GLOBAL] yfinance not available — skipping global indices")
        return {}

    if start is None:
        start = config["data"]["training_start"]
    if tickers is None:
        cfg_tickers = config.get("global_markets", {}).get("tickers", {})
        tickers = cfg_tickers if cfg_tickers else GLOBAL_TICKERS

    results = {}
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
            if df is not None and len(df) > 100:
                close = df["Close"]
                if hasattr(close, "columns"):
                    close = close.iloc[:, 0]
                results[name] = close.dropna()
        except Exception:
            continue

    if results:
        print(f"  [GLOBAL] Fetched {len(results)} global indices: {', '.join(results.keys())}")
    else:
        print("  [GLOBAL] No global indices fetched — contagion features unavailable")

    return results


def cached_fetch_global_indices(force_refresh: bool = False) -> dict[str, pd.Series]:
    """Fetch global indices with parquet caching."""
    cache_path = PROJECT_ROOT / config["cache"]["directory"]
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / "global_indices.parquet"
    ttl = config["cache"]["ttl_hours"]

    if not force_refresh and cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < ttl:
            try:
                df = pd.read_parquet(cache_file)
                return {col: df[col].dropna() for col in df.columns}
            except Exception:
                pass

    data = fetch_global_indices()
    if data:
        try:
            df = pd.DataFrame(data)
            df.to_parquet(cache_file)
        except Exception:
            pass
    return data


def identify_global_crashes(
    index_data: dict[str, pd.Series],
    threshold: float = 0.20,
) -> pd.DataFrame:
    """Identify crash periods across global indices.

    Uses expanding-max drawdown logic (same as risk/crashes.py).

    Args:
        index_data: Dict of {name: price_series}.
        threshold: Drawdown threshold (default 20%).

    Returns:
        DataFrame with columns: market, start, end, max_dd, duration_days
    """
    crashes = []
    exit_recovery = config.get("risk", {}).get("crash_exit_recovery", 0.05)

    for market, prices in index_data.items():
        if len(prices) < 50:
            continue
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak

        in_crash = False
        crash_start = None
        crash_min = 0.0

        for i in range(len(prices)):
            dd = drawdown.iloc[i]
            if dd <= -threshold and not in_crash:
                in_crash = True
                crash_start = prices.index[i]
                crash_min = dd
            elif in_crash:
                crash_min = min(crash_min, dd)
                if dd > -exit_recovery:
                    crashes.append({
                        "market": market,
                        "start": crash_start,
                        "end": prices.index[i],
                        "max_dd": crash_min,
                        "duration_days": (prices.index[i] - crash_start).days,
                    })
                    in_crash = False

        if in_crash:
            crashes.append({
                "market": market,
                "start": crash_start,
                "end": prices.index[-1],
                "max_dd": crash_min,
                "duration_days": (prices.index[-1] - crash_start).days,
            })

    return pd.DataFrame(crashes)


def get_global_crash_count(
    index_data: dict[str, pd.Series],
    threshold: float = 0.20,
) -> pd.Series:
    """Compute per-date count of global indices currently in a crash.

    Args:
        index_data: Dict of {name: price_series}.
        threshold: Drawdown threshold.

    Returns:
        Series indexed by date with count of markets in crash.
    """
    crash_df = identify_global_crashes(index_data, threshold)
    if crash_df.empty:
        return pd.Series(dtype=float)

    # Get union of all dates
    all_dates = pd.DatetimeIndex([])
    for prices in index_data.values():
        all_dates = all_dates.union(prices.index)
    all_dates = all_dates.sort_values()

    count = pd.Series(0, index=all_dates, dtype=int)
    for _, row in crash_df.iterrows():
        mask = (count.index >= row["start"]) & (count.index <= row["end"])
        count[mask] += 1

    return count


def compute_contagion_features(
    index_data: dict[str, pd.Series],
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute cross-market contagion features.

    Two features:
        - n_markets_in_drawdown: count of indices >10% off expanding peak
        - global_drawdown_avg: mean current drawdown across all indices

    All features are backward-looking, forward-filled, and aligned to
    the target index.

    Args:
        index_data: Dict of {name: price_series}.
        target_index: DatetimeIndex to align features to.

    Returns:
        DataFrame with 2 columns aligned to target_index.
    """
    if not index_data:
        return pd.DataFrame(
            {"n_markets_in_drawdown": 0.0, "global_drawdown_avg": 0.0},
            index=target_index,
        )

    drawdown_threshold = -0.10  # 10% off peak = "in drawdown"

    # Compute current drawdown for each index
    drawdowns = {}
    for name, prices in index_data.items():
        if len(prices) < 50:
            continue
        peak = prices.expanding().max()
        dd = (prices - peak) / peak
        drawdowns[name] = dd

    if not drawdowns:
        return pd.DataFrame(
            {"n_markets_in_drawdown": 0.0, "global_drawdown_avg": 0.0},
            index=target_index,
        )

    # Align all drawdowns to a common index, forward-fill
    dd_df = pd.DataFrame(drawdowns)
    dd_df = dd_df.reindex(target_index).ffill()

    # Count markets in drawdown (>10% off peak)
    n_in_dd = (dd_df < drawdown_threshold).sum(axis=1).astype(float)

    # Mean drawdown across all available indices
    avg_dd = dd_df.mean(axis=1)

    return pd.DataFrame({
        "n_markets_in_drawdown": n_in_dd,
        "global_drawdown_avg": avg_dd,
    }, index=target_index)
