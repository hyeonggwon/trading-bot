from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings

VALID_TIMEFRAMES = frozenset({"1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"})


class _StrictModel(BaseModel):
    """Config base: reject unknown keys instead of silently dropping them.

    Pydantic's default (extra="ignore") turns a typo'd YAML key — e.g.
    ``max_drawdown_pcnt:`` — into the silent application of the built-in
    default. On real-money risk settings that must fail loudly at load time.
    """

    model_config = ConfigDict(extra="forbid")


class ExchangeConfig(_StrictModel):
    name: str = "upbit"
    rate_limit_per_sec: Annotated[int, Field(ge=1)] = 10


class TradingConfig(_StrictModel):
    symbols: list[str] = ["BTC/KRW"]
    timeframe: str = "1h"
    initial_balance: Annotated[float, Field(gt=0)] = 1_000_000  # KRW

    @field_validator("timeframe")
    @classmethod
    def _validate_timeframe(cls, v: str) -> str:
        if v not in VALID_TIMEFRAMES:
            raise ValueError(f"invalid timeframe '{v}'; expected one of {sorted(VALID_TIMEFRAMES)}")
        return v


class RiskConfig(_StrictModel):
    max_position_size_pct: Annotated[float, Field(gt=0, le=1.0)] = 0.1
    max_open_positions: Annotated[int, Field(ge=1)] = 3
    max_drawdown_pct: Annotated[float, Field(gt=0, le=1.0)] = 0.20
    default_stop_loss_pct: Annotated[float, Field(gt=0, le=1.0)] = 0.02
    default_take_profit_pct: Annotated[float, Field(gt=0, le=1.0)] | None = None
    risk_per_trade_pct: Annotated[float, Field(gt=0, le=1.0)] = 0.01


class PyramidingConfig(_StrictModel):
    """Signal-triggered adds to an already-open position.

    Disabled by default — a held symbol is then never re-evaluated for entry,
    exactly as before pyramiding existed. Enabled, each add still has to clear
    the free-cash floor, which spends itself down and stops the sequence.
    """

    enabled: bool = False
    min_add_cash_pct: Annotated[float, Field(ge=0, le=1.0)] = 0.05


class BacktestConfig(_StrictModel):
    fee_rate: Annotated[float, Field(ge=0, le=1.0)] = 0.0005
    slippage_pct: Annotated[float, Field(ge=0, le=1.0)] = 0.001
    start_date: str | None = None
    end_date: str | None = None


class AppConfig(_StrictModel):
    exchange: ExchangeConfig = ExchangeConfig()
    trading: TradingConfig = TradingConfig()
    risk: RiskConfig = RiskConfig()
    pyramiding: PyramidingConfig = PyramidingConfig()
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


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
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

    default_path = config_dir / "default.yaml"
    data = load_yaml_config(default_path)

    backtest_path = config_dir / "backtest.yaml"
    if backtest_path.exists():
        data = deep_merge(data, load_yaml_config(backtest_path))

    if overrides:
        data = deep_merge(data, overrides)

    return AppConfig(**data)
