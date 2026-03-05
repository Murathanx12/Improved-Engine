"""Alternative data fetchers for validation and enhanced features.

Each fetcher wraps its network call in try/except and returns None on failure.
The main pipeline NEVER crashes due to alternative data unavailability.

Data sources:
    - AAII Sentiment Survey (weekly, free XLS download)
    - NAAIM Exposure Index (weekly, free web scrape)
    - IMF World Economic Outlook GDP forecasts (twice-yearly, free JSON API)
    - CBOE Put/Call Ratio (via yfinance SPY options chain)
    - Insider Selling (placeholder — requires SEC EDGAR or paid API)

Usage:
    from finpredict.data.alternative_fetchers import (
        fetch_aaii_sentiment, fetch_naaim_exposure, fetch_imf_gdp_forecast,
        fetch_put_call_ratio,
    )

    aaii = fetch_aaii_sentiment()  # DataFrame or None
"""

# ═══════════════════════════════════════════════════════════════════════
# PAID UPGRADE OPTIONS (for future reference):
#
# - CBOE Options Data:       ~$100/mo via CBOE DataShop — direct put/call ratio
# - OpenInsider Pro:         ~$25/mo  — clean insider transaction data
# - Quandl ZACKS/INSIDER:   ~$50/mo  — institutional-quality insider data
# - AAII Premium API:        ~$30/mo  — structured sentiment data with history
# - ISM Manufacturing PMI:  ~$500/yr — direct ISM subscription
# - Refinitiv/LSEG:         ~$300/mo — analyst consensus, institutional flows
# ═══════════════════════════════════════════════════════════════════════

import io
from typing import Optional

import pandas as pd


def fetch_put_call_ratio() -> Optional[pd.Series]:
    """Fetch CBOE equity put/call ratio from SPY options chain.

    Primary: yfinance SPY options chain (sum put OI / sum call OI) across
    the front two expiry dates.
    Fallback: Returns None — features.py uses composite proxy instead.

    Returns:
        Series with single value (current put/call ratio), or None.
    """
    try:
        import yfinance as yf
        spy = yf.Ticker("SPY")
        expirations = spy.options
        if not expirations or len(expirations) < 2:
            print("  [WARN] SPY options data unavailable — skipping put/call ratio")
            return None

        total_put_oi = 0
        total_call_oi = 0
        for exp in expirations[:2]:  # Front two expiries
            chain = spy.option_chain(exp)
            total_put_oi += chain.puts["openInterest"].sum()
            total_call_oi += chain.calls["openInterest"].sum()

        if total_call_oi == 0:
            return None

        ratio = total_put_oi / total_call_oi
        return pd.Series([ratio], index=[pd.Timestamp.now().normalize()],
                         name="put_call_ratio")
    except Exception as e:
        print(f"  [WARN] Put/call ratio fetch failed: {e}")
        return None


def fetch_aaii_sentiment() -> Optional[pd.DataFrame]:
    """Fetch AAII weekly sentiment survey.

    Source: https://www.aaii.com/files/surveys/sentiment.xls
    Returns: DataFrame with columns [date, bullish, neutral, bearish], or None.
    """
    url = "https://www.aaii.com/files/surveys/sentiment.xls"
    try:
        import requests
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (finpredict research engine)"
        })
        if resp.status_code != 200:
            print(f"  [WARN] AAII data unavailable (HTTP {resp.status_code}) — skipping sentiment feature")
            return None

        df = pd.read_excel(io.BytesIO(resp.content), sheet_name=0)

        # Find columns containing sentiment data
        cols = df.columns.tolist()
        date_col = None
        bull_col = None
        bear_col = None
        neutral_col = None

        for c in cols:
            cl = str(c).lower().strip()
            if "date" in cl or "reported" in cl:
                date_col = c
            elif "bull" in cl and "spread" not in cl:
                bull_col = c
            elif "bear" in cl and "spread" not in cl:
                bear_col = c
            elif "neutral" in cl:
                neutral_col = c

        if date_col is None or bull_col is None or bear_col is None:
            print("  [WARN] AAII data format changed — skipping sentiment feature")
            return None

        result = pd.DataFrame({
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "bullish": pd.to_numeric(df[bull_col], errors="coerce"),
            "bearish": pd.to_numeric(df[bear_col], errors="coerce"),
            "neutral": pd.to_numeric(df.get(neutral_col, 0), errors="coerce") if neutral_col else 0,
        }).dropna(subset=["date", "bullish", "bearish"])

        result = result.set_index("date").sort_index()
        print(f"  [OK] AAII sentiment: {len(result)} weeks loaded")
        return result

    except ImportError:
        print("  [WARN] requests/openpyxl not installed — skipping AAII sentiment")
        return None
    except Exception as e:
        print(f"  [WARN] AAII sentiment fetch failed: {e}")
        return None


def fetch_naaim_exposure() -> Optional[pd.Series]:
    """Fetch NAAIM Exposure Index (active manager equity exposure).

    Source: https://www.naaim.org/programs/naaim-exposure-index/
    Returns: Series of weekly exposure values (0-200%), or None.
    """
    try:
        import requests
        url = "https://www.naaim.org/programs/naaim-exposure-index/"
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (finpredict research engine)"
        })
        if resp.status_code != 200:
            print(f"  [WARN] NAAIM data unavailable (HTTP {resp.status_code}) — skipping")
            return None

        # Try to parse tables from the HTML page
        tables = pd.read_html(io.StringIO(resp.text))
        if not tables:
            print("  [WARN] NAAIM page has no tables — skipping")
            return None

        # Find the table with exposure data
        for tbl in tables:
            cols = [str(c).lower() for c in tbl.columns]
            if any("exposure" in c or "mean" in c for c in cols):
                # Try to extract date and exposure columns
                date_col = None
                exp_col = None
                for i, c in enumerate(cols):
                    if "date" in c:
                        date_col = tbl.columns[i]
                    elif "mean" in c or "exposure" in c:
                        exp_col = tbl.columns[i]

                if date_col and exp_col:
                    series = pd.Series(
                        pd.to_numeric(tbl[exp_col], errors="coerce").values,
                        index=pd.to_datetime(tbl[date_col], errors="coerce"),
                        name="naaim_exposure",
                    ).dropna()
                    if len(series) > 0:
                        print(f"  [OK] NAAIM exposure: {len(series)} weeks loaded")
                        return series

        print("  [WARN] NAAIM data format not recognized — skipping")
        return None
    except ImportError:
        print("  [WARN] requests/lxml not installed — skipping NAAIM exposure")
        return None
    except Exception as e:
        print(f"  [WARN] NAAIM exposure fetch failed: {e}")
        return None


def fetch_imf_gdp_forecast(country: str = "USA") -> Optional[dict]:
    """Fetch IMF World Economic Outlook GDP growth forecast.

    Source: IMF Data Mapper API (free JSON endpoint).
    Returns: dict with year → GDP growth forecast (%), or None.
    """
    try:
        import requests
        url = f"https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH/@WEO/{country}"
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            print(f"  [WARN] IMF data unavailable (HTTP {resp.status_code}) — skipping")
            return None

        data = resp.json()
        values = data.get("values", {}).get("NGDP_RPCH", {}).get(country, {})
        if not values:
            print("  [WARN] IMF GDP data empty — skipping")
            return None

        result = {int(yr): float(val) for yr, val in values.items()}
        latest_year = max(result.keys())
        print(f"  [OK] IMF GDP forecast: {latest_year} = {result[latest_year]:.1f}%")
        return result

    except ImportError:
        print("  [WARN] requests not installed — skipping IMF GDP forecast")
        return None
    except Exception as e:
        print(f"  [WARN] IMF GDP forecast fetch failed: {e}")
        return None


def fetch_insider_selling_ratio() -> None:
    """Placeholder for insider selling data.

    TODO: Implement via SEC EDGAR Form 4 API (free, rate-limited) or
    Quandl ZACKS/INSIDER (~$50/mo for institutional-quality data).

    SEC EDGAR approach:
        1. Fetch recent Form 4 filings from EDGAR full-text search
        2. Parse XML for transaction type (P=purchase, S=sale)
        3. Aggregate sell/buy ratio over trailing 4 weeks
        4. Normalize against historical distribution

    This is complex due to rate limiting (10 req/sec) and XML parsing.
    """
    raise NotImplementedError(
        "Insider selling data requires SEC EDGAR scraping or paid API. "
        "See data/alternative_fetchers.py for implementation notes."
    )
