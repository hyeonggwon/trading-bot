"""Tests for Phase 5: live trading components (OrderManager, TradeValidator)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from tradingbot.core.enums import OrderSide, OrderStatus, OrderType
from tradingbot.core.models import Order
from tradingbot.exchange.base import BaseExchange
from tradingbot.live.order_manager import OrderManager
from tradingbot.risk.validators import TradeValidator


# --- Mock exchange for testing ---

class InstantFillExchange(BaseExchange):
    """Exchange that fills orders instantly."""

    async def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=100):
        import pandas as pd
        return pd.DataFrame()

    async def fetch_ticker(self, symbol):
        return {"last": 50_000_000, "bid": 49_999_000, "ask": 50_001_000, "volume": 100, "timestamp": datetime.now(timezone.utc)}

    async def create_order(self, symbol, side, order_type, quantity, price=None):
        return Order(
            id="instant-001",
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.FILLED,
            created_at=datetime.now(timezone.utc),
            filled_at=datetime.now(timezone.utc),
            filled_price=50_000_000,
            fee=25_000,
        )

    async def fetch_order(self, order_id, symbol):
        return Order(
            id=order_id, symbol=symbol, side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=0.001,
            status=OrderStatus.FILLED,
            filled_price=50_000_000, fee=25_000,
        )

    async def cancel_order(self, order_id, symbol):
        return True

    async def get_balance(self):
        return {"KRW": 1_000_000}

    async def get_open_orders(self, symbol=None):
        return []

    async def close(self):
        pass


class DelayedFillExchange(BaseExchange):
    """Exchange that requires polling to confirm fill."""

    def __init__(self):
        self._poll_count = 0

    async def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=100):
        import pandas as pd
        return pd.DataFrame()

    async def fetch_ticker(self, symbol):
        return {"last": 50_000_000, "bid": 49_999_000, "ask": 50_001_000, "volume": 100, "timestamp": datetime.now(timezone.utc)}

    async def create_order(self, symbol, side, order_type, quantity, price=None):
        self._poll_count = 0
        return Order(
            id="delayed-001", symbol=symbol, side=side,
            order_type=order_type, quantity=quantity,
            status=OrderStatus.PENDING,
            created_at=datetime.now(timezone.utc),
        )

    async def fetch_order(self, order_id, symbol):
        self._poll_count += 1
        if self._poll_count >= 2:  # Fill after 2 polls
            return Order(
                id=order_id, symbol=symbol, side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=0.001,
                status=OrderStatus.FILLED,
                filled_price=50_000_000, fee=25_000,
                filled_at=datetime.now(timezone.utc),
            )
        return Order(
            id=order_id, symbol=symbol, side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=0.001,
            status=OrderStatus.PENDING,
        )

    async def cancel_order(self, order_id, symbol):
        return True

    async def get_balance(self):
        return {"KRW": 1_000_000}

    async def get_open_orders(self, symbol=None):
        return []

    async def close(self):
        pass


# --- OrderManager tests ---

class TestOrderManager:
    @pytest.mark.asyncio
    async def test_instant_fill(self):
        """Order that fills immediately should return filled order."""
        mgr = OrderManager(InstantFillExchange(), timeout_seconds=10)
        order = await mgr.submit_and_wait(
            "BTC/KRW", OrderSide.BUY, OrderType.MARKET, 0.001
        )
        assert order.status == OrderStatus.FILLED
        assert order.filled_price == 50_000_000
        assert mgr.active_order_count == 0

    @pytest.mark.asyncio
    async def test_delayed_fill(self):
        """Order that needs polling should eventually fill."""
        mgr = OrderManager(DelayedFillExchange(), timeout_seconds=30)
        order = await mgr.submit_and_wait(
            "BTC/KRW", OrderSide.BUY, OrderType.MARKET, 0.001
        )
        assert order.status == OrderStatus.FILLED
        assert mgr.active_order_count == 0

    @pytest.mark.asyncio
    async def test_cancel_all(self):
        """cancel_all should cancel open orders."""
        mgr = OrderManager(InstantFillExchange())
        cancelled = await mgr.cancel_all("BTC/KRW")
        assert cancelled == 0  # No open orders in mock


# --- TradeValidator tests ---

class TestTradeValidator:
    def test_order_size_within_limit(self):
        v = TradeValidator(max_order_value_krw=500_000)
        assert v.validate_order_size(0.01, 50_000_000) is True  # 500K = limit

    def test_order_size_exceeds_limit(self):
        v = TradeValidator(max_order_value_krw=500_000)
        assert v.validate_order_size(0.02, 50_000_000) is False  # 1M > 500K

    def test_cooldown_first_order(self):
        v = TradeValidator(order_cooldown_seconds=10)
        assert v.validate_cooldown() is True

    def test_cooldown_too_fast(self):
        v = TradeValidator(order_cooldown_seconds=10)
        v.record_order()
        assert v.validate_cooldown() is False

    def test_daily_loss_within_limit(self):
        v = TradeValidator(daily_loss_limit_krw=200_000)
        v.record_trade_pnl(-100_000)
        assert v.validate_daily_loss() is True

    def test_daily_loss_exceeded(self):
        v = TradeValidator(daily_loss_limit_krw=200_000)
        v.record_trade_pnl(-250_000)
        assert v.validate_daily_loss() is False

    def test_validate_all_passes(self):
        v = TradeValidator(max_order_value_krw=1_000_000, daily_loss_limit_krw=500_000)
        assert v.validate_all(0.001, 50_000_000) is True

    def test_validate_all_fails_size(self):
        v = TradeValidator(max_order_value_krw=100_000)
        assert v.validate_all(0.01, 50_000_000) is False

    def test_daily_reset(self):
        """PnL should reset on new day."""
        v = TradeValidator(daily_loss_limit_krw=200_000)
        v.record_trade_pnl(-250_000)
        assert v.validate_daily_loss() is False

        # Simulate day change by resetting the date
        v._daily_reset_date = None
        assert v.validate_daily_loss() is True  # Reset on "new day"
