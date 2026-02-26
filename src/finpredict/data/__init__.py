"""Data — fetching, caching, and FRED macro integration."""

from finpredict.data.fetchers import fetch_all_data, fetch_safe
from finpredict.data.cache import cached_fetch_all_data
from finpredict.data.fred_fetcher import fetch_fred_data, get_recession_probability

__all__ = [
    "fetch_all_data", "fetch_safe", "cached_fetch_all_data",
    "fetch_fred_data", "get_recession_probability",
]
