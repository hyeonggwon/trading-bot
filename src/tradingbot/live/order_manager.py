"""Order lifecycle management for live trading.

Tracks orders from creation to fill/cancel. Handles:
- Order status polling until fill or timeout
- Timeout cancellation and market re-order
- Fill confirmation and position update
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import structlog

from tradingbot.core.enums import OrderSide, OrderStatus, OrderType
from tradingbot.core.models import Order
from tradingbot.exchange.base import BaseExchange

logger = structlog.get_logger()

DEFAULT_ORDER_TIMEOUT_SECONDS = 60
POLL_INTERVAL_SECONDS = 2


class OrderManager:
    """Manages order lifecycle for live trading."""

    def __init__(
        self,
        exchange: BaseExchange,
        timeout_seconds: int = DEFAULT_ORDER_TIMEOUT_SECONDS,
    ):
        self.exchange = exchange
        self.timeout_seconds = timeout_seconds
        self._active_orders: dict[str, Order] = {}  # order_id -> Order

    async def submit_and_wait(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float | None = None,
    ) -> Order:
        """Submit an order and wait until it is filled or times out.

        For market orders: polls until filled (usually instant on Upbit).
        For limit orders: polls until filled or timeout, then cancels and
        re-submits as market order.

        Returns the final Order object with fill details.
        """
        order = await self.exchange.create_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
        )
        self._active_orders[order.id] = order

        logger.info(
            "order_submitted",
            order_id=order.id,
            symbol=symbol,
            side=side.value,
            type=order_type.value,
            quantity=f"{quantity:.8f}",
        )

        # If already filled (paper exchange fills instantly)
        if order.status == OrderStatus.FILLED:
            self._active_orders.pop(order.id, None)
            return order

        # Poll for fill
        filled_order = await self._poll_until_filled(order)

        if filled_order.status == OrderStatus.FILLED:
            self._active_orders.pop(order.id, None)
            return filled_order

        # Timeout — cancel and re-submit as market if it was a limit order
        if order_type == OrderType.LIMIT:
            logger.warning(
                "order_timeout_resubmit",
                order_id=order.id,
                symbol=symbol,
                timeout=self.timeout_seconds,
            )
            try:
                await self.exchange.cancel_order(order.id, symbol)
            except Exception as e:
                logger.error("cancel_failed", order_id=order.id, error=str(e))
            self._active_orders.pop(order.id, None)

            # Check for partial fill — only re-order the REMAINING quantity
            try:
                final_state = await self.exchange.fetch_order(order.id, symbol)
                # quantity field holds filled amount (from CCXT's "filled" field)
                # regardless of whether status is FILLED or CANCELLED (partial fill)
                filled_qty = final_state.quantity
            except Exception:
                filled_qty = 0

            remaining = quantity - filled_qty
            if remaining <= 0:
                # Fully filled during cancel race — return the filled order
                return final_state if filled_qty > 0 else filled_order

            # Re-submit the remaining quantity as a market order.
            market_order = await self.submit_and_wait(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=remaining,
            )

            # Nothing filled on the original limit — the market order alone
            # represents the whole position.
            if filled_qty <= 0:
                return market_order

            # Combine the already-filled limit portion with the market
            # re-order so the returned Order reflects the CUMULATIVE fill:
            # total quantity, quantity-weighted fill price, and summed fee.
            # Returning only the re-order would silently drop the portion
            # already filled at the limit price, under-sizing the position.
            total_qty = filled_qty + market_order.quantity
            limit_price = final_state.filled_price or price or 0.0
            market_price = market_order.filled_price or limit_price
            weighted_price = (
                limit_price * filled_qty + market_price * market_order.quantity
            ) / total_qty
            return Order(
                id=market_order.id,
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=total_qty,
                price=price,
                status=market_order.status,
                created_at=order.created_at,
                filled_at=market_order.filled_at,
                filled_price=weighted_price,
                fee=(final_state.fee or 0.0) + (market_order.fee or 0.0),
            )

        # Market order timeout (unusual) — return last polled state
        self._active_orders.pop(order.id, None)
        return filled_order

    async def _poll_until_filled(self, order: Order) -> Order:
        """Poll order status until filled or timeout. Returns last known state."""
        elapsed = 0.0
        last_known = order

        while elapsed < self.timeout_seconds:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            elapsed += POLL_INTERVAL_SECONDS

            try:
                updated = await self.exchange.fetch_order(order.id, order.symbol)
                last_known = updated
            except Exception as e:
                logger.warning("order_poll_error", order_id=order.id, error=str(e))
                continue

            if updated.status == OrderStatus.FILLED:
                logger.info(
                    "order_filled",
                    order_id=order.id,
                    price=updated.filled_price,
                    fee=updated.fee,
                )
                return updated

            if updated.status == OrderStatus.CANCELLED:
                logger.info("order_cancelled_externally", order_id=order.id)
                return updated

        logger.warning("order_poll_timeout", order_id=order.id, elapsed=elapsed)
        return last_known

    async def cancel_all(self, symbol: str) -> int:
        """Cancel all active orders for a symbol. Returns count cancelled."""
        cancelled = 0
        open_orders = await self.exchange.get_open_orders(symbol)
        for order in open_orders:
            success = await self.exchange.cancel_order(order.id, symbol)
            if success:
                cancelled += 1
                logger.info("order_cancelled", order_id=order.id, symbol=symbol)
        self._active_orders = {
            oid: o for oid, o in self._active_orders.items()
            if o.symbol != symbol
        }
        return cancelled

    @property
    def active_order_count(self) -> int:
        return len(self._active_orders)
