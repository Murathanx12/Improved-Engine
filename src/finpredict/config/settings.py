"""
Configuration loader for the Market Prediction Engine.

Two sources:
  - .env           → API keys (secrets, gitignored)
  - engine_config.yaml → Engine parameters (safe to commit)

Usage:
    from finpredict.config.settings import config, api_keys

    # Access engine parameters
    config['simulation']['num_simulations']
    config['risk']['crash_threshold']

    # Access API keys
    api_keys.fred
    api_keys.finnhub
"""

import os
from pathlib import Path
from dataclasses import dataclass

import yaml
from dotenv import load_dotenv


# ── Locate project root (walk up until we find engine_config.yaml) ──────────────────


def _find_project_root() -> Path:
    """Find project root by looking for engine_config.yaml."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "engine_config.yaml").exists():
            return parent
    # Fallback: assume 3 levels up from this file (src/finpredict/config/settings.py)
    return Path(__file__).resolve().parents[3]


PROJECT_ROOT = _find_project_root()


# ── Load .env ───────────────────────────────────────────────────────────────────────

load_dotenv(PROJECT_ROOT / ".env")


# ── API Keys (typed access) ─────────────────────────────────────────────────────────


@dataclass
class APIKeys:
    """All API keys, loaded from .env file."""

    fred: str = ""
    finnhub: str = ""
    fmp: str = ""
    sec_edgar_user_agent: str = ""
    alpha_vantage: str = ""
    polygon: str = ""
    sendgrid: str = ""
    twilio_sid: str = ""
    twilio_token: str = ""
    twilio_phone: str = ""

    @classmethod
    def from_env(cls) -> "APIKeys":
        return cls(
            fred=os.getenv("FRED_API_KEY", ""),
            finnhub=os.getenv("FINNHUB_API_KEY", ""),
            fmp=os.getenv("FMP_API_KEY", ""),
            sec_edgar_user_agent=os.getenv("SEC_EDGAR_USER_AGENT", ""),
            alpha_vantage=os.getenv("ALPHA_VANTAGE_API_KEY", ""),
            polygon=os.getenv("POLYGON_API_KEY", ""),
            sendgrid=os.getenv("SENDGRID_API_KEY", ""),
            twilio_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
            twilio_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
            twilio_phone=os.getenv("TWILIO_PHONE_NUMBER", ""),
        )

    def has(self, key: str) -> bool:
        """Check if a key is set and not a placeholder."""
        val = getattr(self, key, "")
        return bool(val) and val != "" and "placeholder" not in val.lower()


# ── Load engine_config.yaml ─────────────────────────────────────────────────────────


def _load_config() -> dict:
    """Load engine configuration from YAML file."""
    config_path = PROJECT_ROOT / "engine_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"engine_config.yaml not found at {config_path}. "
            f"Project root detected as: {PROJECT_ROOT}"
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Module-level singletons (import these) ──────────────────────────────────────────

config: dict = _load_config()
api_keys: APIKeys = APIKeys.from_env()


# ── Convenience accessors ───────────────────────────────────────────────────────────


def get_institutional_return() -> float:
    """Compute consensus institutional expected return, adjusted for horizon."""
    benchmarks = config["institutional_benchmarks"]
    adj = benchmarks.get("horizon_adjustment", 1.05)
    returns = [v["annual"] for k, v in benchmarks.items() if isinstance(v, dict) and "annual" in v]
    return float(sum(returns) / len(returns)) * adj


def get_forecast_days() -> int:
    """Total trading days for the projection horizon."""
    sim = config["simulation"]
    return sim["forecast_years"] * sim["trading_days_per_year"]


def get_scenario_configs() -> dict:
    """Return scenario definitions with resolved returns."""
    inst_return = get_institutional_return()
    scenarios = {}
    for name, params in config["scenarios"].items():
        s = dict(params)  # copy
        if s.get("absolute_return") is not None:
            s["return"] = s["absolute_return"]
        elif s.get("return_multiplier") is not None:
            s["return"] = inst_return * s["return_multiplier"]
        else:
            s["return"] = inst_return
        s["probability"] = s.pop("base_probability")
        scenarios[name] = s
    return scenarios
