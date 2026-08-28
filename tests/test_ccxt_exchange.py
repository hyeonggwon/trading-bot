"""Tests for CcxtExchange order handling."""

from __future__ import annotations

import ccxt.async_support as ccxt_async
import pytest

from tradingbot.core.enums import OrderSide, OrderStatus, OrderType
from tradingbot.exchange.ccxt_exchange import CcxtExchange


class TestUpbitMarketBuyCost:
    """Upbit market buys are quote-denominated (ord_type='price').

    ccxt raises InvalidOrder for a market buy given only a base amount and no
    price/cost, which silently broke every live long entry. We must convert the
    base quantity into a KRW cost so the order actually reaches the exchange.
    """

    @pytest.mark.asyncio
    async def test_market_buy_sends_quote_cost(self):
        ex = CcxtExchange()
        captured: dict = {}

        async def _capture(symbol, ccxt_type, ccxt_side, amount, price, params):
            captured.update(
                type=ccxt_type,
                side=ccxt_side,
                amount=amount,
                price=price,
                params=params,
            )
            return {"id": "abc123"}

        ex._exchange.create_order = _capture
        try:
            order = await ex.create_order(
                "BTC/KRW", OrderSide.BUY, OrderType.MARKET, 0.002, 50_000_000
            )
        finally:
            await ex.close()

        assert captured["type"] == "market"
        assert captured["side"] == "buy"
        # cost = base quantity × reference price
        assert captured["params"]["cost"] == pytest.approx(0.002 * 50_000_000)
        assert order.status == OrderStatus.PENDING

    @pytest.mark.asyncio
    async def test_market_buy_without_price_rejected_before_exchange(self):
        ex = CcxtExchange()
        reached = {"n": 0}

        async def _spy(*args, **kwargs):
            reached["n"] += 1
            return {"id": "x"}

        ex._exchange.create_order = _spy
        try:
            with pytest.raises(ValueError, match="reference price"):
                await ex.create_order("BTC/KRW", OrderSide.BUY, OrderType.MARKET, 0.002)
            assert reached["n"] == 0  # rejected in our layer, never sent
        finally:
            await ex.close()

    @pytest.mark.asyncio
    async def test_market_sell_needs_no_cost(self):
        ex = CcxtExchange()
        captured: dict = {}

        async def _capture(symbol, ccxt_type, ccxt_side, amount, price, params):
            captured.update(side=ccxt_side, params=params)
            return {"id": "s1"}

        ex._exchange.create_order = _capture
        try:
            # No price required for a base-volume market sell.
            await ex.create_order("BTC/KRW", OrderSide.SELL, OrderType.MARKET, 0.002)
        finally:
            await ex.close()

        assert captured["side"] == "sell"
        assert "cost" not in captured["params"]


class TestFetchOrderFillQuantity:
    """The reported quantity must be what actually executed.

    A cancelled order that filled nothing carries filled=0.0. Treating that
    falsy value as "unknown" and falling back to the requested amount reports a
    zero-fill cancel as a full execution — which the live engine would then book
    as an open position and the order manager would skip re-ordering.
    """

    async def _fetch_with(self, raw: dict):
        ex = CcxtExchange()

        async def _stub(order_id, symbol):
            return {"id": order_id, "status": "canceled", **raw}

        ex._exchange.fetch_order = _stub
        try:
            return await ex.fetch_order("o1", "BTC/KRW")
        finally:
            await ex.close()

    @pytest.mark.asyncio
    async def test_zero_fill_reports_zero(self):
        order = await self._fetch_with({"filled": 0.0, "amount": 2.0})
        assert order.quantity == 0.0

    @pytest.mark.asyncio
    async def test_partial_fill_reports_filled_amount(self):
        order = await self._fetch_with({"filled": 1.5, "amount": 2.0})
        assert order.quantity == pytest.approx(1.5)

    @pytest.mark.asyncio
    async def test_missing_filled_falls_back_to_amount(self):
        order = await self._fetch_with({"filled": None, "amount": 2.0})
        assert order.quantity == pytest.approx(2.0)


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
                # Market buys carry a reference price (Upbit quote-cost model);
                # the error must still surface on the single attempt.
                await ex.create_order("BTC/KRW", OrderSide.BUY, OrderType.MARKET, 0.001, 50_000_000)
            assert calls["n"] == 1
        finally:
            await ex.close()


class TestTotalBalance:
    """Sizing spends free cash, but reconciliation must see locked amounts too:
    a coin reserved by a resting order is still held, and reading it as unheld
    drops the position as a phantom."""

    @pytest.mark.asyncio
    async def test_total_includes_locked_free_does_not(self):
        ex = CcxtExchange()

        async def _stub():
            return {
                "free": {"KRW": 1_000_000.0, "BTC": 0.0},
                "total": {"KRW": 1_000_000.0, "BTC": 0.05},
            }

        ex._exchange.fetch_balance = _stub
        try:
            free = await ex.get_balance()
            total = await ex.get_total_balance()
        finally:
            await ex.close()

        assert "BTC" not in free  # every unit is locked in an open order
        assert total["BTC"] == pytest.approx(0.05)
        assert total["KRW"] == pytest.approx(1_000_000.0)
