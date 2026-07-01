"""Tests for Phase 5: live trading components (OrderManager, TradeValidator)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timezone

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


class PartialFillLimitExchange(BaseExchange):
    """LIMIT order partially fills then times out; the remaining quantity is
    re-submitted as MARKET. Used to verify cumulative fill tracking."""

    LIMIT_PRICE = 50_000_000.0
    MARKET_PRICE = 50_200_000.0
    TOTAL_QTY = 1.0
    FILLED_QTY = 0.4  # filled at the limit price before the timeout
    LIMIT_FEE = 10_000.0
    MARKET_FEE = 18_000.0

    def __init__(self):
        self._cancelled = False

    async def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=100):
        import pandas as pd
        return pd.DataFrame()

    async def fetch_ticker(self, symbol):
        return {"last": self.MARKET_PRICE, "bid": self.MARKET_PRICE,
                "ask": self.MARKET_PRICE, "volume": 100,
                "timestamp": datetime.now(UTC)}

    async def create_order(self, symbol, side, order_type, quantity, price=None):
        if order_type == OrderType.MARKET:
            # The re-order for the unfilled remainder.
            return Order(
                id="market-002", symbol=symbol, side=side,
                order_type=OrderType.MARKET, quantity=quantity,
                status=OrderStatus.FILLED,
                filled_price=self.MARKET_PRICE, fee=self.MARKET_FEE,
                created_at=datetime.now(UTC),
                filled_at=datetime.now(UTC),
            )
        # The initial LIMIT — stays pending so it times out.
        return Order(
            id="limit-001", symbol=symbol, side=side,
            order_type=OrderType.LIMIT, quantity=quantity, price=price,
            status=OrderStatus.PENDING,
            created_at=datetime.now(UTC),
        )

    async def fetch_order(self, order_id, symbol):
        if self._cancelled:
            # Post-cancel: reports the partially-filled portion (CCXT maps the
            # filled base amount onto Order.quantity).
            return Order(
                id=order_id, symbol=symbol, side=OrderSide.BUY,
                order_type=OrderType.LIMIT, quantity=self.FILLED_QTY,
                status=OrderStatus.CANCELLED,
                filled_price=self.LIMIT_PRICE, fee=self.LIMIT_FEE,
            )
        # During polling: still pending → forces the timeout path.
        return Order(
            id=order_id, symbol=symbol, side=OrderSide.BUY,
            order_type=OrderType.LIMIT, quantity=self.TOTAL_QTY,
            status=OrderStatus.PENDING,
        )

    async def cancel_order(self, order_id, symbol):
        self._cancelled = True
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

    @pytest.mark.asyncio
    async def test_limit_timeout_partial_fill_tracks_cumulative(self, monkeypatch):
        """A partially-filled LIMIT that times out must return the CUMULATIVE
        fill, not just the market re-order of the remainder.

        Old behavior returned only the re-ordered remaining quantity, silently
        discarding the portion already filled at the limit price — so the caller
        opened a position smaller (and mispriced) than what was actually held.
        """
        from tradingbot.live import order_manager as om

        monkeypatch.setattr(om, "POLL_INTERVAL_SECONDS", 0.01)
        ex = PartialFillLimitExchange()
        mgr = OrderManager(ex, timeout_seconds=1)

        order = await mgr.submit_and_wait(
            "BTC/KRW", OrderSide.BUY, OrderType.LIMIT,
            ex.TOTAL_QTY, price=ex.LIMIT_PRICE,
        )

        remaining = ex.TOTAL_QTY - ex.FILLED_QTY
        expected_price = (
            ex.LIMIT_PRICE * ex.FILLED_QTY + ex.MARKET_PRICE * remaining
        ) / ex.TOTAL_QTY

        # Cumulative quantity, not just the re-ordered remainder (0.6).
        assert order.quantity == pytest.approx(ex.TOTAL_QTY)
        # Quantity-weighted average fill price across both fills.
        assert order.filled_price == pytest.approx(expected_price)
        # Fees from both the partial limit fill and the market re-order.
        assert order.fee == pytest.approx(ex.LIMIT_FEE + ex.MARKET_FEE)


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

    def test_daily_state_survives_restart(self):
        """Restarting must not zero a daily loss the bot already booked.

        Without restore_daily_state(), a process restart resets _daily_pnl to 0
        and lets the bot keep trading past a daily-loss limit it had breached.
        """
        v = TradeValidator(daily_loss_limit_krw=200_000)
        v.record_trade_pnl(-250_000)
        assert v.validate_daily_loss() is False  # breached

        daily_pnl, reset_date = v.daily_state()
        assert daily_pnl == -250_000
        assert reset_date is not None

        # Fresh validator (simulating a restart) restores the persisted state.
        restored = TradeValidator(daily_loss_limit_krw=200_000)
        restored.restore_daily_state(daily_pnl, reset_date)
        assert restored.daily_state() == (-250_000, reset_date)
        # The breach is still in force after the "restart".
        assert restored.validate_daily_loss() is False
