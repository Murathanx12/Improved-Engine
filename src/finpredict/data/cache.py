"""
Data Cache Layer
================

Wraps data fetchers with a file-based cache (Parquet format).
Subsequent runs skip API calls if cache is fresh (under TTL).

Usage:
    from finpredict.data.cache import cached_fetch_all_data

    data, sector_data = cached_fetch_all_data()  # Uses cache if fresh
    data, sector_data = cached_fetch_all_data(force_refresh=True)  # Force re-fetch
"""

import time
from pathlib import Path

import pandas as pd

from finpredict.config import config, PROJECT_ROOT
from finpredict.data.fetchers import fetch_all_data


def _cache_dir() -> Path:
    """Get or create cache directory."""
    cache_path = PROJECT_ROOT / config["cache"]["directory"]
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def _cache_age_hours(filepath: Path) -> float:
    """Return age of a file in hours. Returns inf if file doesn't exist."""
    if not filepath.exists():
        return float("inf")
    age_seconds = time.time() - filepath.stat().st_mtime
    return age_seconds / 3600


def cached_fetch_all_data(
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """
    Fetch market data with caching.

    Checks cache first. If cache exists and is under TTL, loads from disk.
    Otherwise fetches fresh data and updates cache.

    Args:
        force_refresh: If True, ignore cache and fetch fresh data.

    Returns:
        Same as fetch_all_data(): (data DataFrame, sector_data dict)
    """
    cache_path = _cache_dir()
    data_file = cache_path / "market_data.parquet"
    sectors_file = cache_path / "sector_data.parquet"
    ttl = config["cache"]["ttl_hours"]

    # Check cache freshness
    if not force_refresh and _cache_age_hours(data_file) < ttl:
        print(f"[CACHE] Loading from cache (age: {_cache_age_hours(data_file):.1f}h, TTL: {ttl}h)")
        sector_data = {}
        try:
            data = pd.read_parquet(data_file)
            if sectors_file.exists():
                sectors_df = pd.read_parquet(sectors_file)
                sector_data = {col: sectors_df[col].dropna() for col in sectors_df.columns}
        except Exception:
            # fallback to pickle if parquet absent or failing
            try:
                data = pd.read_pickle(cache_path / "market_data.pkl")
                if (cache_path / "sector_data.pkl").exists():
                    sectors_df = pd.read_pickle(cache_path / "sector_data.pkl")
                    sector_data = {col: sectors_df[col].dropna() for col in sectors_df.columns}
            except Exception:
                print("[CACHE] cache exists but could not be read; refreshing")
                raise
        print(f"  [OK] {len(data):,} rows, {len(data.columns)} columns\n")
        return data, sector_data

    # Fetch fresh data
    data, sector_data = fetch_all_data()

    # Save to cache. Parquet is preferred, but may not be available.
    try:
        data.to_parquet(data_file)
        if sector_data:
            sectors_df = pd.DataFrame(sector_data)
            sectors_df.to_parquet(sectors_file)
        print(f"[CACHE] Saved to {cache_path} (parquet)\n")
    except Exception as e:  # pragma: no cover - fallback path
        print(f"[CACHE] warning: unable to write parquet ({e}), falling back to pickle/csv")
        try:
            data.to_pickle(cache_path / "market_data.pkl")
            if sector_data:
                sectors_df = pd.DataFrame(sector_data)
                sectors_df.to_pickle(cache_path / "sector_data.pkl")
            print(f"[CACHE] Saved to {cache_path} (pickle)\n")
        except Exception as e2:
            print(f"[CACHE] error writing pickle cache: {e2}")
    return data, sector_data
