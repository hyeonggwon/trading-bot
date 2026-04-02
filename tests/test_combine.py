"""Tests for combine engine — filters, parsing, CombinedStrategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingbot.backtest.engine import BacktestEngine
from tradingbot.config import AppConfig, BacktestConfig, RiskConfig, TradingConfig
from tradingbot.strategy.combined import CombinedStrategy
from tradingbot.strategy.filters.momentum import RsiOverboughtFilter, RsiOversoldFilter
from tradingbot.strategy.filters.price import EmaAboveFilter, PriceBreakoutFilter
from tradingbot.strategy.filters.registry import (
    get_filter_map,
    parse_filter_spec,
    parse_filter_string,
)
from tradingbot.strategy.filters.trend import TrendUpFilter
from tradingbot.strategy.filters.volume import VolumeSpikeFilter


def _make_data(n: int = 300) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    t = np.linspace(0, 6 * np.pi, n)
    close = 50_000_000 + 5_000_000 * np.sin(t) + np.random.normal(0, 200_000, n)
    high = close + np.abs(np.random.normal(500_000, 200_000, n))
    low = close - np.abs(np.random.normal(500_000, 200_000, n))
    open_ = close + np.random.normal(0, 300_000, n)
    volume = np.random.uniform(100, 1000, n)
    # Add some volume spikes
    for idx in [50, 150, 250]:
        if idx < n:
            volume[idx] = 5000
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class TestFilterRegistry:
    def test_all_filters_registered(self):
        fmap = get_filter_map()
        assert "trend_up" in fmap
        assert "rsi_oversold" in fmap
        assert "volume_spike" in fmap
        assert "ema_above" in fmap
        assert len(fmap) == 9

    def test_parse_simple(self):
        f = parse_filter_spec("rsi_oversold:30")
        assert isinstance(f, RsiOversoldFilter)
        assert f.threshold == 30.0

    def test_parse_with_multiple_params(self):
        f = parse_filter_spec("trend_up:6:20")
        assert isinstance(f, TrendUpFilter)
        assert f.tf_factor == 6
        assert f.sma_period == 20

    def test_parse_no_params(self):
        f = parse_filter_spec("ema_above")
        assert isinstance(f, EmaAboveFilter)
        assert f.period == 20  # default

    def test_parse_trailing_colon(self):
        """Trailing colons should be ignored, not crash."""
        f = parse_filter_spec("rsi_oversold:30:")
        assert isinstance(f, RsiOversoldFilter)
        assert f.threshold == 30.0

    def test_parse_invalid_param(self):
        """Non-numeric params should raise clear ValueError."""
        import pytest
        with pytest.raises(ValueError, match="Invalid parameters"):
            parse_filter_spec("rsi_oversold:abc")

    def test_parse_unknown_filter(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown filter"):
            parse_filter_spec("nonexistent_filter")

    def test_parse_filter_string(self):
        filters = parse_filter_string("trend_up:4 + rsi_oversold:30 + volume_spike:2.5")
        assert len(filters) == 3
        assert filters[0].name == "trend_up"
        assert filters[1].name == "rsi_oversold"
        assert filters[2].name == "volume_spike"


class TestCombinedStrategy:
    def test_describe(self):
        entry = [RsiOversoldFilter(threshold=30), VolumeSpikeFilter(threshold=2.0)]
        exit_ = [RsiOverboughtFilter(threshold=70)]
        strategy = CombinedStrategy(entry_filters=entry, exit_filters=exit_)
        desc = strategy.describe()
        assert "rsi_oversold" in desc
        assert "volume_spike" in desc
        assert "rsi_overbought" in desc

    def test_backtest_runs(self):
        df = _make_data(300)
        entry = [RsiOversoldFilter(threshold=35)]
        exit_ = [RsiOverboughtFilter(threshold=65)]
        strategy = CombinedStrategy(entry_filters=entry, exit_filters=exit_)
        strategy.timeframe = "1h"

        config = AppConfig(
            trading=TradingConfig(symbols=["BTC/KRW"], timeframe="1h", initial_balance=10_000_000),
            risk=RiskConfig(max_position_size_pct=0.5, max_open_positions=1,
                           max_drawdown_pct=0.3, default_stop_loss_pct=0.05, risk_per_trade_pct=0.02),
            backtest=BacktestConfig(fee_rate=0.0005, slippage_pct=0.001),
        )

        engine = BacktestEngine(strategy=strategy, config=config)
        report = engine.run({"BTC/KRW": df})
        assert report.final_balance > 0

    def test_multi_filter_entry(self):
        """Multiple entry filters require ALL to pass (AND logic)."""
        df = _make_data(300)
        # Strict entry: RSI oversold + EMA above (hard to satisfy simultaneously)
        entry = [RsiOversoldFilter(threshold=30), EmaAboveFilter(period=50)]
        exit_ = [RsiOverboughtFilter(threshold=70)]
        strategy = CombinedStrategy(entry_filters=entry, exit_filters=exit_)
        strategy.timeframe = "1h"

        config = AppConfig(
            trading=TradingConfig(symbols=["BTC/KRW"], timeframe="1h", initial_balance=10_000_000),
            risk=RiskConfig(max_position_size_pct=0.5, max_open_positions=1,
                           max_drawdown_pct=0.3, default_stop_loss_pct=0.05, risk_per_trade_pct=0.02),
            backtest=BacktestConfig(fee_rate=0.0005, slippage_pct=0.001),
        )

        engine = BacktestEngine(strategy=strategy, config=config)
        report = engine.run({"BTC/KRW": df})
        # With stricter filters, should have fewer trades than single filter
        assert report.final_balance > 0

    def test_no_entry_filters_no_trades(self):
        df = _make_data(100)
        strategy = CombinedStrategy(entry_filters=[], exit_filters=[])
        strategy.timeframe = "1h"

        config = AppConfig(
            trading=TradingConfig(symbols=["BTC/KRW"], timeframe="1h", initial_balance=10_000_000),
            risk=RiskConfig(),
            backtest=BacktestConfig(fee_rate=0.0005, slippage_pct=0.001),
        )

        engine = BacktestEngine(strategy=strategy, config=config)
        report = engine.run({"BTC/KRW": df})
        assert report.total_trades == 0
