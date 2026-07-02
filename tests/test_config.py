"""Config field-constraint and timeframe-whitelist validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tradingbot.config import (
    VALID_TIMEFRAMES,
    AppConfig,
    RiskConfig,
    TradingConfig,
    load_config,
)
from tradingbot.live.engine import TIMEFRAME_SECONDS

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_shipped_default_config_loads_without_error() -> None:
    """The shipped config/default.yaml (max_position_size_pct=1.0) must stay valid."""
    cfg = load_config(config_dir=REPO_ROOT / "config")
    assert 0 < cfg.risk.max_position_size_pct <= 1.0


def test_full_position_size_boundary_allowed() -> None:
    """1.0 (100%) is intentional for the one-symbol-per-container model."""
    assert RiskConfig(max_position_size_pct=1.0).max_position_size_pct == 1.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_position_size_pct", 1.5),
        ("max_position_size_pct", 0.0),
        ("max_drawdown_pct", 1.5),
        ("default_stop_loss_pct", 0.0),
        ("risk_per_trade_pct", 1.5),
        ("max_open_positions", 0),
    ],
)
def test_risk_config_rejects_out_of_range(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        RiskConfig(**{field: value})


def test_trading_config_rejects_nonpositive_balance() -> None:
    with pytest.raises(ValidationError):
        TradingConfig(initial_balance=0)


def test_invalid_timeframe_rejected() -> None:
    with pytest.raises(ValidationError):
        TradingConfig(timeframe="4hr")


def test_valid_timeframe_accepted() -> None:
    assert TradingConfig(timeframe="4h").timeframe == "4h"


def test_appconfig_validates_nested_risk() -> None:
    with pytest.raises(ValidationError):
        AppConfig(risk={"max_position_size_pct": 2.0})


def test_valid_timeframes_in_sync_with_engine() -> None:
    """config's whitelist is the single source of truth mirror of engine TIMEFRAME_SECONDS."""
    assert VALID_TIMEFRAMES == set(TIMEFRAME_SECONDS.keys())


def test_unknown_yaml_key_rejected(tmp_path: Path) -> None:
    """오타 난 YAML 키는 조용히 증발하지 않고 로드 시점에 실패해야 한다.

    extra="ignore"(pydantic 기본)였다면 max_drawdown_pcnt 오타가 무시되고
    내장 기본값 0.20 이 소리 없이 적용된다 — 리스크 설정에서는 치명적.
    """
    (tmp_path / "default.yaml").write_text("risk:\n  max_drawdown_pcnt: 0.5\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="max_drawdown_pcnt"):
        load_config(config_dir=tmp_path)
