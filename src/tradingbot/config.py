from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ExchangeConfig(BaseModel):
    name: str = "upbit"
    rate_limit_per_sec: int = 10


class TradingConfig(BaseModel):
    symbols: list[str] = ["BTC/KRW"]
    timeframe: str = "1h"
    initial_balance: float = 1_000_000  # KRW


class RiskConfig(BaseModel):
    max_position_size_pct: float = 0.1
    max_open_positions: int = 3
    max_drawdown_pct: float = 0.20
    default_stop_loss_pct: float = 0.02
    risk_per_trade_pct: float = 0.01


class BacktestConfig(BaseModel):
    fee_rate: float = 0.0005
    slippage_pct: float = 0.001
    start_date: str | None = None
    end_date: str | None = None


class AppConfig(BaseModel):
    exchange: ExchangeConfig = ExchangeConfig()
    trading: TradingConfig = TradingConfig()
    risk: RiskConfig = RiskConfig()
    backtest: BacktestConfig = BacktestConfig()


class EnvSettings(BaseSettings):
    upbit_access_key: str = ""
    upbit_secret_key: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file."""
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    config_dir: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Load configuration from YAML files with optional overrides.

    Loads default.yaml first, then merges backtest.yaml if present.
    """
    if config_dir is None:
        config_dir = Path("config")

    data: dict[str, Any] = {}

    default_path = config_dir / "default.yaml"
    data = deep_merge(data, load_yaml_config(default_path))

    backtest_path = config_dir / "backtest.yaml"
    if backtest_path.exists():
        data = deep_merge(data, load_yaml_config(backtest_path))

    if overrides:
        data = deep_merge(data, overrides)

    return AppConfig(**data)


def load_env() -> EnvSettings:
    """Load environment variables / .env file."""
    return EnvSettings()
