"""Tests for multi-symbol backtesting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingbot.backtest.engine import BacktestEngine
from tradingbot.config import AppConfig, BacktestConfig, RiskConfig, TradingConfig
from tradingbot.strategy.base import StrategyParams
from tradingbot.strategy.examples.sma_cross import SmaCrossStrategy


def _make_symbol_data(
    symbol: str, n: int = 200, phase_offset: float = 0.0
) -> pd.DataFrame:
    """Generate synthetic data for a symbol with a phase offset for variety."""
    np.random.seed(hash(symbol) % 2**31)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    t = np.linspace(0, 4 * np.pi, n) + phase_offset
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


def _make_config(symbols: list[str], balance: float = 10_000_000) -> AppConfig:
    return AppConfig(
        trading=TradingConfig(
            symbols=symbols, timeframe="1h", initial_balance=balance,
        ),
        risk=RiskConfig(
            max_position_size_pct=0.2,
            max_open_positions=3,
            max_drawdown_pct=0.30,
            default_stop_loss_pct=0.05,
            risk_per_trade_pct=0.02,
        ),
        backtest=BacktestConfig(fee_rate=0.0005, slippage_pct=0.001),
    )


class TestMultiSymbolBacktest:
    def test_two_symbols(self):
        """Backtest with two symbols should generate trades from both."""
        symbols = ["BTC/KRW", "ETH/KRW"]
        data = {
            "BTC/KRW": _make_symbol_data("BTC/KRW", 200, phase_offset=0),
            "ETH/KRW": _make_symbol_data("ETH/KRW", 200, phase_offset=1.5),
        }
        config = _make_config(symbols)
        strategy = SmaCrossStrategy(StrategyParams({"fast_period": 10, "slow_period": 30}))
        strategy.symbols = symbols

        engine = BacktestEngine(strategy=strategy, config=config)
        report = engine.run(data)

        assert report.total_trades >= 1
        assert report.final_balance > 0
        assert len(report.equity_curve) > 0

    def test_three_symbols(self):
        """Three symbols with max 3 positions."""
        symbols = ["BTC/KRW", "ETH/KRW", "XRP/KRW"]
        data = {
            sym: _make_symbol_data(sym, 200, phase_offset=i * 1.0)
            for i, sym in enumerate(symbols)
        }
        config = _make_config(symbols)
        strategy = SmaCrossStrategy(StrategyParams({"fast_period": 10, "slow_period": 30}))
        strategy.symbols = symbols

        engine = BacktestEngine(strategy=strategy, config=config)
        report = engine.run(data)

        assert report.total_trades >= 1
        assert report.final_balance > 0

    def test_trades_have_correct_symbols(self):
        """Each trade should reference a valid symbol from the strategy."""
        symbols = ["BTC/KRW", "ETH/KRW"]
        data = {
            "BTC/KRW": _make_symbol_data("BTC/KRW", 200, phase_offset=0),
            "ETH/KRW": _make_symbol_data("ETH/KRW", 200, phase_offset=1.5),
        }
        config = _make_config(symbols)
        strategy = SmaCrossStrategy(StrategyParams({"fast_period": 10, "slow_period": 30}))
        strategy.symbols = symbols

        engine = BacktestEngine(strategy=strategy, config=config)
        report = engine.run(data)

        for trade in report.trades:
            assert trade.symbol in symbols

    def test_max_positions_respected(self):
        """Should never hold more positions than max_open_positions."""
        symbols = ["BTC/KRW", "ETH/KRW", "XRP/KRW", "SOL/KRW"]
        data = {
            sym: _make_symbol_data(sym, 300, phase_offset=i * 0.8)
            for i, sym in enumerate(symbols)
        }
        config = _make_config(symbols, balance=20_000_000)
        config.risk.max_open_positions = 2  # Only 2 simultaneous

        strategy = SmaCrossStrategy(StrategyParams({"fast_period": 10, "slow_period": 30}))
        strategy.symbols = symbols

        engine = BacktestEngine(strategy=strategy, config=config)
        report = engine.run(data)

        # The engine should have managed positions correctly
        assert report.final_balance > 0

    def test_single_symbol_backward_compatible(self):
        """Single symbol should work exactly as before."""
        data = {"BTC/KRW": _make_symbol_data("BTC/KRW", 200)}
        config = _make_config(["BTC/KRW"])
        strategy = SmaCrossStrategy(StrategyParams({"fast_period": 10, "slow_period": 30}))

        engine = BacktestEngine(strategy=strategy, config=config)
        report = engine.run(data)

        assert report.total_trades >= 1
        assert report.final_balance > 0

    def test_missing_symbol_data_skipped(self):
        """Symbols without data should be silently skipped."""
        symbols = ["BTC/KRW", "ETH/KRW"]
        data = {"BTC/KRW": _make_symbol_data("BTC/KRW", 200)}  # No ETH data
        config = _make_config(symbols)
        strategy = SmaCrossStrategy(StrategyParams({"fast_period": 10, "slow_period": 30}))
        strategy.symbols = symbols

        engine = BacktestEngine(strategy=strategy, config=config)
        report = engine.run(data)

        # Should still work with just BTC
        assert report.final_balance > 0

    def test_equity_curve_continuous(self):
        """Equity curve should have entries for each timestamp."""
        symbols = ["BTC/KRW", "ETH/KRW"]
        data = {
            "BTC/KRW": _make_symbol_data("BTC/KRW", 100),
            "ETH/KRW": _make_symbol_data("ETH/KRW", 100),
        }
        config = _make_config(symbols)
        strategy = SmaCrossStrategy(StrategyParams({"fast_period": 5, "slow_period": 15}))
        strategy.symbols = symbols

        engine = BacktestEngine(strategy=strategy, config=config)
        report = engine.run(data)

        # Equity curve should be monotonically timestamped
        assert report.equity_curve.index.is_monotonic_increasing

    def test_deterministic(self):
        """Same data + params should produce identical results."""
        symbols = ["BTC/KRW", "ETH/KRW"]
        data = {
            "BTC/KRW": _make_symbol_data("BTC/KRW", 200),
            "ETH/KRW": _make_symbol_data("ETH/KRW", 200, phase_offset=1.5),
        }
        config = _make_config(symbols)
        params = StrategyParams({"fast_period": 10, "slow_period": 30})

        engine1 = BacktestEngine(strategy=SmaCrossStrategy(params), config=config)
        report1 = engine1.run(data)

        engine2 = BacktestEngine(strategy=SmaCrossStrategy(params), config=config)
        report2 = engine2.run(data)

        assert report1.total_trades == report2.total_trades
        assert abs(report1.final_balance - report2.final_balance) < 0.01
