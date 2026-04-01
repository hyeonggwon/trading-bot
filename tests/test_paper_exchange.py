"""Tests for paper exchange and state management."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from tradingbot.core.enums import OrderSide, OrderStatus, OrderType, PositionSide
from tradingbot.core.models import Position
from tradingbot.exchange.base import BaseExchange
from tradingbot.exchange.paper import PaperExchange
from tradingbot.live.state import StateManager


class MockDataFeed(BaseExchange):
    """Minimal mock exchange for testing paper trading."""

    def __init__(self, price: float = 50_000_000):
        self._price = price

    async def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=100):
        dates = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        return pd.DataFrame({
            "open": [self._price] * 10,
            "high": [self._price * 1.01] * 10,
            "low": [self._price * 0.99] * 10,
            "close": [self._price] * 10,
            "volume": [100] * 10,
        }, index=dates)

    async def fetch_ticker(self, symbol):
        return {
            "last": self._price,
            "bid": self._price * 0.999,
            "ask": self._price * 1.001,
            "volume": 100,
            "timestamp": datetime.now(timezone.utc),
        }

    async def create_order(self, symbol, side, order_type, quantity, price=None):
        raise NotImplementedError("Mock does not create orders")

    async def fetch_order(self, order_id, symbol):
        raise NotImplementedError("Mock does not fetch orders")

    async def cancel_order(self, order_id, symbol):
        return False

    async def get_balance(self):
        return {"KRW": 1_000_000}

    async def get_open_orders(self, symbol=None):
        return []

    async def close(self):
        pass


class TestPaperExchange:
    @pytest.fixture
    def paper(self):
        feed = MockDataFeed(price=50_000_000)
        return PaperExchange(
            data_feed=feed,
            initial_balance=10_000_000,
            fee_rate=0.0005,
            slippage_pct=0.001,
        )

    def test_initial_balance(self, paper):
        assert paper.cash == 10_000_000
        assert paper.holdings == {}

    @pytest.mark.asyncio
    async def test_buy_market_order(self, paper):
        # Fetch ticker to set last price
        await paper.fetch_ticker("BTC/KRW")

        order = await paper.create_order(
            symbol="BTC/KRW",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001,
        )

        assert order.status == OrderStatus.FILLED
        assert order.filled_price is not None
        assert order.fee > 0
        assert paper.cash < 10_000_000
        assert "BTC" in paper.holdings
        assert paper.holdings["BTC"] == 0.001

    @pytest.mark.asyncio
    async def test_sell_market_order(self, paper):
        await paper.fetch_ticker("BTC/KRW")

        # Buy first
        await paper.create_order("BTC/KRW", OrderSide.BUY, OrderType.MARKET, 0.001)
        cash_after_buy = paper.cash

        # Then sell
        order = await paper.create_order("BTC/KRW", OrderSide.SELL, OrderType.MARKET, 0.001)

        assert order.status == OrderStatus.FILLED
        assert paper.cash > cash_after_buy
        assert "BTC" not in paper.holdings  # All sold

    @pytest.mark.asyncio
    async def test_insufficient_cash(self, paper):
        await paper.fetch_ticker("BTC/KRW")

        # Try to buy more than we can afford (10M KRW can buy ~0.0002 BTC at 50M)
        order = await paper.create_order("BTC/KRW", OrderSide.BUY, OrderType.MARKET, 1.0)

        assert order.status == OrderStatus.FILLED
        # Quantity should be reduced to fit budget
        assert order.quantity < 1.0
        assert paper.cash < 1  # Almost all cash used (may have epsilon)

    @pytest.mark.asyncio
    async def test_sell_more_than_held(self, paper):
        await paper.fetch_ticker("BTC/KRW")

        # Buy small amount
        await paper.create_order("BTC/KRW", OrderSide.BUY, OrderType.MARKET, 0.001)

        # Try to sell more than held
        order = await paper.create_order("BTC/KRW", OrderSide.SELL, OrderType.MARKET, 1.0)

        assert order.status == OrderStatus.FILLED
        assert order.quantity == 0.001  # Capped at holdings

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_delegates(self, paper):
        df = await paper.fetch_ohlcv("BTC/KRW", "1h", limit=10)
        assert len(df) == 10
        assert "close" in df.columns

    @pytest.mark.asyncio
    async def test_equity_calculation(self, paper):
        await paper.fetch_ticker("BTC/KRW")
        await paper.create_order("BTC/KRW", OrderSide.BUY, OrderType.MARKET, 0.001)

        equity = paper.equity()
        # Equity should be close to initial balance (minus slippage/fees)
        assert equity > 9_900_000
        assert equity < 10_100_000

    @pytest.mark.asyncio
    async def test_get_balance(self, paper):
        await paper.fetch_ticker("BTC/KRW")
        await paper.create_order("BTC/KRW", OrderSide.BUY, OrderType.MARKET, 0.001)

        balance = await paper.get_balance()
        assert "KRW" in balance
        assert "BTC" in balance


class TestStateManager:
    def test_save_and_load(self, tmp_path):
        state_file = tmp_path / "state.json"
        state = StateManager(state_file)

        pos = Position(
            symbol="BTC/KRW",
            side=PositionSide.LONG,
            size=0.001,
            entry_price=50_000_000,
            entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            stop_loss=49_000_000,
        )
        state.positions["BTC/KRW"] = pos
        state.record_equity(10_000_000)
        state.save()

        # Load into fresh state
        state2 = StateManager(state_file)
        state2.load()

        assert "BTC/KRW" in state2.positions
        loaded_pos = state2.positions["BTC/KRW"]
        assert loaded_pos.symbol == "BTC/KRW"
        assert loaded_pos.size == 0.001
        assert loaded_pos.entry_price == 50_000_000
        assert loaded_pos.stop_loss == 49_000_000
        assert len(state2.equity_history) == 1

    def test_load_nonexistent(self, tmp_path):
        state = StateManager(tmp_path / "nope.json")
        state.load()  # Should not raise
        assert state.positions == {}

    def test_load_corrupt_file(self, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text("not json{{{")

        state = StateManager(state_file)
        state.load()  # Should not raise, starts fresh
        assert state.positions == {}

    def test_clear(self, tmp_path):
        state_file = tmp_path / "state.json"
        state = StateManager(state_file)
        state.positions["BTC/KRW"] = Position(
            "BTC/KRW", PositionSide.LONG, 0.001, 50_000_000,
            datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        state.save()
        assert state_file.exists()

        state.clear()
        assert state.positions == {}
        assert not state_file.exists()

    def test_equity_history_capped(self, tmp_path):
        state_file = tmp_path / "state.json"
        state = StateManager(state_file)

        for i in range(1500):
            state.record_equity(10_000_000 + i)

        state.save()

        # Load back — should only keep last 1000
        state2 = StateManager(state_file)
        state2.load()
        assert len(state2.equity_history) == 1000
