"""Tests for additional strategies (MACD momentum, Bollinger breakout)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingbot.backtest.engine import BacktestEngine
from tradingbot.config import AppConfig, BacktestConfig, RiskConfig, TradingConfig
from tradingbot.strategy.base import StrategyParams
from tradingbot.strategy.examples.bollinger_breakout import BollingerBreakoutStrategy
from tradingbot.strategy.examples.macd_momentum import MacdMomentumStrategy


def _make_cyclic_data(n: int = 300) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    t = np.linspace(0, 6 * np.pi, n)
    base = 50_000_000 + 5_000_000 * np.sin(t)
    noise = np.random.normal(0, 200_000, n)
    close = base + noise
    high = close + np.abs(np.random.normal(500_000, 200_000, n))
    low = close - np.abs(np.random.normal(500_000, 200_000, n))
    open_ = close + np.random.normal(0, 300_000, n)
    volume = np.random.uniform(100, 1000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _make_config() -> AppConfig:
    return AppConfig(
        trading=TradingConfig(symbols=["BTC/KRW"], timeframe="1h", initial_balance=10_000_000),
        risk=RiskConfig(
            max_position_size_pct=0.5, max_open_positions=1,
            max_drawdown_pct=0.30, default_stop_loss_pct=0.05, risk_per_trade_pct=0.02,
        ),
        backtest=BacktestConfig(fee_rate=0.0005, slippage_pct=0.001),
    )


class TestMacdMomentum:
    def test_backtest_runs(self):
        df = _make_cyclic_data(300)
        config = _make_config()
        strategy = MacdMomentumStrategy(StrategyParams({"fast": 8, "slow": 20, "signal": 7}))

        engine = BacktestEngine(strategy=strategy, config=config)
        report = engine.run({"BTC/KRW": df})

        assert report.final_balance > 0
        assert len(report.equity_curve) > 0

    def test_generates_trades(self):
        df = _make_cyclic_data(300)
        config = _make_config()
        strategy = MacdMomentumStrategy(StrategyParams({"fast": 8, "slow": 20, "signal": 7}))

        engine = BacktestEngine(strategy=strategy, config=config)
        report = engine.run({"BTC/KRW": df})

        assert report.total_trades >= 1, "MACD should generate trades on cyclic data"

    def test_param_space(self):
        space = MacdMomentumStrategy.param_space()
        assert "fast" in space
        assert "slow" in space
        assert "signal" in space


class TestBollingerBreakout:
    def test_backtest_runs(self):
        df = _make_cyclic_data(300)
        config = _make_config()
        strategy = BollingerBreakoutStrategy(StrategyParams({"period": 20, "std": 2.0}))

        engine = BacktestEngine(strategy=strategy, config=config)
        report = engine.run({"BTC/KRW": df})

        assert report.final_balance > 0
        assert len(report.equity_curve) > 0

    def test_generates_trades(self):
        df = _make_cyclic_data(300)
        config = _make_config()
        # Use tighter bands to trigger more breakouts
        strategy = BollingerBreakoutStrategy(StrategyParams({"period": 15, "std": 1.5}))

        engine = BacktestEngine(strategy=strategy, config=config)
        report = engine.run({"BTC/KRW": df})

        assert report.total_trades >= 1, "Bollinger should generate trades with tight bands"

    def test_param_space(self):
        space = BollingerBreakoutStrategy.param_space()
        assert "period" in space
        assert "std" in space
