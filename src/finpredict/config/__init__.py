"""Configuration package — single import for all config needs."""

from finpredict.config.settings import (
    config,
    api_keys,
    get_institutional_return,
    get_forecast_days,
    get_scenario_configs,
    PROJECT_ROOT,
)

__all__ = [
    "config",
    "api_keys",
    "get_institutional_return",
    "get_forecast_days",
    "get_scenario_configs",
    "PROJECT_ROOT",
]
