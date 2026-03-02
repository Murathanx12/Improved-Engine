"""
Module 1: Data Layer — Market Data Fetching
============================================

Fetches all market data from Yahoo Finance and builds the master DataFrame.
This is the single source of truth for the engine — no hardcoded prices.

Data Sources:
    - S&P 500 (^GSPC): Core market index
    - VIX (^VIX): Volatility/fear index
    - Treasuries (^TNX, ^IRX, ^TYX): Yield curve components
    - HYG/LQD: Credit spread proxies
    - Gold (GC=F): Safe haven / risk indicator
    - NASDAQ, Russell 2000: Breadth indicators
    - 11 Sector ETFs: Sector-level analysis

Each ticker is fetched with safe error handling. If a source fails,
the engine degrades gracefully rather than crashing.

Usage:
    from finpredict.data.fetchers import fetch_all_data

    data, sector_data = fetch_all_data()
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional

from finpredict.config import config


# ── Safe Ticker Fetch ────────────────────────────────────────────────────────────────

def fetch_safe(ticker: str, start: str, end: str, name: str = "") -> Optional[pd.Series]:
    """
    Safely download a single ticker's closing prices from Yahoo Finance.

    Args:
        ticker: Yahoo Finance ticker symbol
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        name: Human-readable name for logging

    Returns:
        pd.Series of closing prices (forward-filled), or None if fetch fails.
    """
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            print(f"  [WARN] No data for {name or ticker}")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            series = df["Close"].iloc[:, 0]
        else:
            series = df["Close"]
        return series.ffill()
    except Exception as e:
        print(f"  [WARN] Failed to fetch {name or ticker}: {e}")
        return None


# ── Treasury Yield Processing ────────────────────────────────────────────────────────

def _add_treasury(data: pd.DataFrame, ticker: str, column: str,
                  start: str, end: str, name: str) -> pd.DataFrame:
    """
    Fetch a Treasury yield and convert from percentage points to decimal.

    Yahoo Finance reports yields as percentage points (e.g., 4.09 = 4.09%).
    We divide by 100 to get decimal form (0.0409).

    BUG HISTORY: V3 divided by 10 instead of 100, giving 40.9% yields
    which caused negative Sharpe ratios. Fixed in V4.5.
    """
    series = fetch_safe(ticker, start, end, name)
    if series is not None:
        data[column] = series / 100  # Percentage points → decimal
    return data


# ── Main Data Fetching ───────────────────────────────────────────────────────────────

def fetch_all_data() -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """
    Module 1 Entry Point: Fetch all market data and build master DataFrame.

    Returns:
        data: pd.DataFrame with all market series, indexed by date
        sector_data: dict mapping sector names to their price series
    """
    print("[MODULE 1] Fetching market data from Yahoo Finance...")

    tickers = config["data"]["tickers"]
    start = config["data"]["training_start"]
    end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    # ── Core Index ───────────────────────────────────────────────────────────────
    sp500 = fetch_safe(tickers["index"], start, end, "S&P 500")
    if sp500 is None:
        raise RuntimeError("FATAL: Cannot fetch S&P 500 data. Check internet connection.")

    data = pd.DataFrame({"SP500": sp500})

    # ── Volatility ───────────────────────────────────────────────────────────────
    vix = fetch_safe(tickers["vix"], "1990-01-02", end, "VIX")
    if vix is not None:
        data["VIX"] = vix

    # ── Treasuries ───────────────────────────────────────────────────────────────
    # Note: ^IRX = 13-week T-Bill yield, labeled T3M. The 10Y-3M spread is the
    # NY Fed's preferred recession predictor (their probit model uses this exact spread).
    data = _add_treasury(data, tickers["treasury_10y"], "T10Y", start, end, "10Y Treasury")
    data = _add_treasury(data, tickers["treasury_3m"], "T3M", "1976-01-02", end, "13-Week T-Bill")
    data = _add_treasury(data, tickers["treasury_30y"], "T30Y", "1977-02-18", end, "30Y Treasury")

    # ── Credit Spreads ───────────────────────────────────────────────────────────
    hyg = fetch_safe(tickers["high_yield"], "2007-04-04", end, "High Yield Bonds")
    if hyg is not None:
        data["HYG"] = hyg
    lqd = fetch_safe(tickers["inv_grade"], "2002-07-22", end, "Inv Grade Bonds")
    if lqd is not None:
        data["LQD"] = lqd

    # ── Alternative Assets ───────────────────────────────────────────────────────
    # Gold is used in risk score factor #7 (Gold/Stock ratio → flight to safety)
    gold = fetch_safe(tickers["gold"], "2000-01-01", end, "Gold")
    if gold is not None:
        data["Gold"] = gold

    # ── Breadth Indicators ───────────────────────────────────────────────────────
    nasdaq = fetch_safe(tickers["nasdaq"], "1971-02-05", end, "NASDAQ")
    if nasdaq is not None:
        data["NASDAQ"] = nasdaq
    russell = fetch_safe(tickers["russell"], "1987-09-10", end, "Russell 2000")
    if russell is not None:
        data["Russell"] = russell

    # ── Options Market Signals ─────────────────────────────────────────────────
    # VIX term structure: VIX/VIX3M > 1.0 = backwardation = imminent risk priced
    if "vix3m" in tickers:
        vix3m = fetch_safe(tickers["vix3m"], "2008-01-03", end, "VIX3M (90-day)")
        if vix3m is not None:
            data["VIX3M"] = vix3m

    # SKEW Index: measures tail-risk hedging demand (high SKEW = institutions buying crash protection)
    if "skew" in tickers:
        skew = fetch_safe(tickers["skew"], "1990-01-02", end, "CBOE SKEW")
        if skew is not None:
            data["SKEW"] = skew

    # ── Sector ETFs ──────────────────────────────────────────────────────────────
    sector_tickers = config["data"]["sectors"]
    sector_start = config["data"]["sector_start"]
    sector_data = {}

    for name, tick in sector_tickers.items():
        s_data = fetch_safe(tick, sector_start, end, name)
        if s_data is not None:
            data[f"Sector_{name}"] = s_data
            sector_data[name] = s_data

    # ── Clean Up ─────────────────────────────────────────────────────────────────
    data = data.ffill().bfill()

    print(f"  [OK] {len(data):,} daily observations")
    print(f"  [OK] {len(data.columns)} data series")
    print(f"  [OK] Range: {data.index[0].date()} to {data.index[-1].date()}\n")

    return data, sector_data
