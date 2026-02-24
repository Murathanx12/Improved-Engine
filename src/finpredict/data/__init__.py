"""Data layer — fetching, caching, and processing market data."""

from finpredict.data.fetchers import fetch_all_data, fetch_safe
from finpredict.data.cache import cached_fetch_all_data

__all__ = ["fetch_all_data", "fetch_safe", "cached_fetch_all_data"]
