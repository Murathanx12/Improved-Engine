"""
Module 8b: Individual Stock Analysis
======================================

Analyzes individual stocks using fundamental-aware Monte Carlo:
    1. Fetch price history + fundamentals from Yahoo Finance
    2. Estimate drift moderated by market cap CAGR caps + analyst targets
    3. Run jump-diffusion Monte Carlo
    4. Report expected return, risk metrics, probability of loss

Market Cap Tier CAGR Caps:
    Mega (>$200B):  8-15%  — mature, can't double easily
    Large ($10-200B): 10-20% — moderate growth ceiling
    Mid ($2-10B):   12-25% — higher growth potential
    Small (<$2B):   15-30% — high risk, high ceiling

Usage:
    from finpredict.models.stocks import analyze_stocks, select_stocks_from_sectors

    watchlist = select_stocks_from_sectors(sector_results, n_stocks=20)
    stock_results = analyze_stocks(tickers=watchlist, forecast_days=1260)
"""

import numpy as np
import yfinance as yf

from finpredict.config import config
from finpredict.simulation.monte_carlo import simulate_paths


# CAGR caps by market cap tier [min_cagr, max_cagr]
STOCK_CAGR_CAPS = {
    "mega":  (0.04, 0.15),
    "large": (0.05, 0.20),
    "mid":   (0.06, 0.25),
    "small": (0.08, 0.30),
}

DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
    "TSLA", "JPM", "JNJ", "V", "UNH", "XOM",
]

SECTOR_STOCK_MAP = {
    "Technology":       ["AAPL", "MSFT", "NVDA", "AVGO", "CRM", "PLTR", "NOW", "AMD"],
    "Healthcare":       ["UNH", "LLY", "JNJ", "ISRG", "VRTX", "DXCM", "GEHC"],
    "Financials":       ["JPM", "V", "MA", "GS", "BLK", "COIN", "SQ"],
    "Energy":           ["XOM", "CVX", "SLB", "OKE", "FSLR", "ENPH"],
    "Consumer Disc.":   ["AMZN", "TSLA", "HD", "NKE", "BKNG", "ABNB"],
    "Industrials":      ["CAT", "GE", "RTX", "UBER", "AXON", "TT"],
    "Communications":   ["META", "GOOGL", "NFLX", "DIS", "RBLX", "SPOT"],
    "Consumer Staples": ["COST", "PG", "KO", "WMT", "MNST"],
    "Materials":        ["LIN", "FCX", "NEM", "VMC"],
    "Utilities":        ["NEE", "VST", "CEG", "SO"],
    "Real Estate":      ["PLD", "AMT", "EQIX", "O"],
}


def select_stocks_from_sectors(sector_results: dict, n_stocks: int = 20) -> list[str]:
    """
    Data-driven stock selection from top-performing sectors.

    Ranks sectors by expected return, allocates more picks to top sectors.
    """
    if not sector_results:
        return DEFAULT_WATCHLIST[:n_stocks]

    ranked = sorted(sector_results.items(), key=lambda x: x[1]["expected_return"], reverse=True)
    selected = []

    for i, (sector_name, _info) in enumerate(ranked):
        if sector_name not in SECTOR_STOCK_MAP:
            continue
        pool = SECTOR_STOCK_MAP[sector_name]
        picks = min(3, len(pool)) if i < 3 else min(2, len(pool)) if i < 7 else 1
        selected.extend(pool[:picks])
        if len(selected) >= n_stocks:
            break

    seen = set()
    return [t for t in selected if not (t in seen or seen.add(t))][:n_stocks]


def _get_cap_tier(market_cap) -> str:
    """Classify stock by market cap tier."""
    if market_cap is None or market_cap <= 0:
        return "large"
    b = market_cap / 1e9
    if b > 200: return "mega"
    elif b > 10: return "large"
    elif b > 2: return "mid"
    else: return "small"


def analyze_stock(
    ticker: str,
    forecast_days: int,
    risk_free_rate: float = 0.04,
) -> dict | None:
    """Analyze a single stock with fundamental-aware Monte Carlo."""
    max_5y_return = config["simulation"]["max_5y_return"]

    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        hist = stock.history(period="5y")
        if hist.empty or len(hist) < 252:
            print(f"    [WARN] {ticker}: Insufficient price history")
            return None

        prices = hist["Close"]
        current_price = float(prices.iloc[-1])

        # Fundamentals
        market_cap = info.get("marketCap", None)
        cap_tier = _get_cap_tier(market_cap)
        beta = info.get("beta", 1.0)
        if beta is None or beta <= 0:
            beta = 1.0
        analyst_target = info.get("targetMeanPrice", None)
        company_name = info.get("shortName", ticker)
        sector = info.get("sector", "Unknown")
        pe_ratio = info.get("trailingPE", None)

        # Historical drift + vol
        returns = prices.pct_change().dropna()
        log_returns = np.log(1 + returns)
        hist_mu = log_returns.mean() * 252
        hist_sigma = returns.std() * np.sqrt(252)

        # Cap to tier-appropriate CAGR
        min_cagr, max_cagr = STOCK_CAGR_CAPS[cap_tier]
        capped_mu = np.clip(hist_mu, min_cagr, max_cagr)

        # Analyst target moderation
        if analyst_target is not None and analyst_target > 0:
            analyst_1y_return = (analyst_target / current_price) - 1
            analyst_annual = np.clip(analyst_1y_return, -0.30, max_cagr)
            blended_mu = 0.60 * capped_mu + 0.40 * analyst_annual
        else:
            blended_mu = capped_mu

        final_mu = np.clip(blended_mu, min_cagr * 0.5, max_cagr)
        final_sigma = np.clip(hist_sigma, 0.15, 0.80)

        # Monte Carlo
        paths = simulate_paths(
            current_price, final_mu, final_sigma,
            forecast_days, 3000, crash_rate=0.07, risk_level=0.0,
        )

        final_prices = np.minimum(paths[-1], current_price * (1 + max_5y_return))
        exp_return = float(np.mean(final_prices) / current_price - 1) * 100
        med_return = float(np.median(final_prices) / current_price - 1) * 100
        p05 = float(np.percentile(final_prices, 5))
        p95 = float(np.percentile(final_prices, 95))
        prob_loss = float(np.mean(final_prices < current_price)) * 100

        running_peak = np.maximum.accumulate(paths, axis=0)
        drawdowns = (paths - running_peak) / running_peak
        avg_max_dd = float(np.mean(drawdowns.min(axis=0))) * 100

        sharpe = (final_mu - risk_free_rate) / final_sigma if final_sigma > 0 else 0

        return {
            "ticker": ticker, "name": company_name, "sector": sector,
            "current_price": current_price, "market_cap": market_cap,
            "cap_tier": cap_tier, "beta": beta, "pe_ratio": pe_ratio,
            "analyst_target": analyst_target,
            "hist_drift": hist_mu * 100, "capped_drift": final_mu * 100,
            "volatility": final_sigma * 100,
            "expected_return": exp_return, "median_return": med_return,
            "p05_price": p05, "p95_price": p95,
            "prob_loss_5y": prob_loss, "avg_max_drawdown": avg_max_dd,
            "sharpe": sharpe,
        }

    except Exception as e:
        print(f"    [WARN] {ticker}: Analysis failed — {e}")
        return None


def analyze_stocks(
    tickers: list[str] | None = None,
    forecast_days: int = 1260,
    risk_free_rate: float = 0.04,
) -> dict:
    """Module 8b Entry Point: Analyze a portfolio of individual stocks."""
    if tickers is None:
        tickers = DEFAULT_WATCHLIST

    print(f"[MODULE 8b] Analyzing {len(tickers)} individual stocks...")
    results = {}

    for ticker in tickers:
        result = analyze_stock(ticker, forecast_days, risk_free_rate)
        if result is not None:
            results[ticker] = result
            print(f"    [OK] {ticker} ({result['cap_tier']}): "
                  f"{result['expected_return']:+.1f}% expected")

    print(f"  [OK] {len(results)}/{len(tickers)} stocks analyzed\n")
    return results
