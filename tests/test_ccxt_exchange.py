"""Tests for CcxtExchange order handling."""

from __future__ import annotations

import ccxt.async_support as ccxt_async
import pytest

from tradingbot.core.enums import OrderSide, OrderType
from tradingbot.exchange.ccxt_exchange import CcxtExchange


class TestCreateOrderNoRetry:
    @pytest.mark.asyncio
    async def test_create_order_not_retried_on_network_error(self):
        """Order submission must be attempted exactly once.

        create_order is non-idempotent: a NetworkError raised after the
        exchange accepted the order (lost response) would, on retry, place a
        duplicate order. The error must surface instead of being retried.
        """
        ex = CcxtExchange()
        calls = {"n": 0}

        async def _boom(*args, **kwargs):
            calls["n"] += 1
            raise ccxt_async.NetworkError("lost response")

        ex._exchange.create_order = _boom
        try:
            with pytest.raises(ccxt_async.NetworkError):
                await ex.create_order("BTC/KRW", OrderSide.BUY, OrderType.MARKET, 0.001)
            assert calls["n"] == 1
        finally:
            await ex.close()
