"""Paper trading exchange.

Simulates order execution using real-time market data from a data feed
exchange (e.g., CcxtExchange). Tracks balances, positions, and orders
in memory with optional state persistence.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pandas as pd
import structlog

from tradingbot.config import BacktestConfig
from tradingbot.core.enums import OrderSide, OrderStatus, OrderType
from tradingbot.core.models import Order
from tradingbot.exchange.base import BaseExchange

logger = structlog.get_logger()


class PaperExchange(BaseExchange):
    """Simulated exchange for paper trading.

    Uses a real exchange as a data feed for prices, but executes
    orders locally with simulated fills.
    """

    def __init__(
        self,
        data_feed: BaseExchange,
        initial_balance: float = 1_000_000,
        fee_rate: float = 0.0005,
        slippage_pct: float = 0.001,
    ):
        self._feed = data_feed
        self._fee_rate = fee_rate
        self._slippage_pct = slippage_pct

        # Portfolio state
        self._cash: float = initial_balance
        self._holdings: dict[str, float] = {}  # symbol base currency -> quantity
        self._open_orders: list[Order] = []
        self._filled_orders: list[Order] = []
        self._last_prices: dict[str, float] = {}

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def holdings(self) -> dict[str, float]:
        return dict(self._holdings)

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: int | None = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Delegate to data feed."""
        return await self._feed.fetch_ohlcv(symbol, timeframe, since, limit)

    async def fetch_ticker(self, symbol: str) -> dict:
        """Delegate to data feed and cache last price."""
        ticker = await self._feed.fetch_ticker(symbol)
        self._last_prices[symbol] = ticker["last"]
        return ticker

    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float | None = None,
    ) -> Order:
        """Simulate order creation and immediate market fill."""
        order = Order(
            id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.PENDING,
            created_at=datetime.now(timezone.utc),
        )

        if order_type == OrderType.MARKET:
            await self._fill_market_order(order)
        else:
            # Limit orders stay pending until price is reached
            self._open_orders.append(order)
            logger.info("paper_limit_order_placed", order_id=order.id, symbol=symbol, price=price)

        return order

    async def _fill_market_order(self, order: Order) -> None:
        """Fill a market order at current price with slippage."""
        last_price = self._last_prices.get(order.symbol)
        if last_price is None:
            ticker = await self.fetch_ticker(order.symbol)
            last_price = ticker["last"]

        # Apply slippage
        if order.side == OrderSide.BUY:
            fill_price = last_price * (1 + self._slippage_pct)
        else:
            fill_price = last_price * (1 - self._slippage_pct)

        fee = fill_price * order.quantity * self._fee_rate

        # Execute
        base_currency = order.symbol.split("/")[0]  # e.g., "BTC" from "BTC/KRW"

        if order.side == OrderSide.BUY:
            cost = fill_price * order.quantity + fee
            if cost > self._cash:
                # Reduce quantity to fit budget
                max_qty = self._cash / (fill_price * (1 + self._fee_rate))
                order.quantity = max_qty
                fee = fill_price * order.quantity * self._fee_rate
                cost = fill_price * order.quantity + fee

            if order.quantity <= 0:
                logger.warning("paper_order_rejected_no_cash", symbol=order.symbol)
                order.status = OrderStatus.CANCELLED
                return

            self._cash = max(0.0, self._cash - cost)
            self._holdings[base_currency] = self._holdings.get(base_currency, 0) + order.quantity

        else:  # SELL
            held = self._holdings.get(base_currency, 0)
            sell_qty = min(order.quantity, held)
            if sell_qty <= 0:
                logger.warning("paper_order_rejected_no_holdings", symbol=order.symbol)
                order.status = OrderStatus.CANCELLED
                return

            order.quantity = sell_qty
            fee = fill_price * sell_qty * self._fee_rate  # recalculate for actual quantity
            revenue = fill_price * sell_qty - fee
            self._cash += revenue
            self._holdings[base_currency] = held - sell_qty
            if self._holdings[base_currency] <= 0:
                del self._holdings[base_currency]

        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now(timezone.utc)
        order.filled_price = fill_price
        order.fee = fee
        self._filled_orders.append(order)
        # Cap filled orders history to prevent unbounded memory growth
        if len(self._filled_orders) > 10000:
            self._filled_orders = self._filled_orders[-5000:]

        logger.info(
            "paper_order_filled",
            order_id=order.id,
            side=order.side.value,
            symbol=order.symbol,
            quantity=f"{order.quantity:.8f}",
            price=f"{fill_price:,.0f}",
            fee=f"{fee:,.0f}",
        )

    async def fetch_order(self, order_id: str, symbol: str) -> Order:
        for order in self._filled_orders:
            if order.id == order_id:
                return order
        for order in self._open_orders:
            if order.id == order_id:
                return order
        raise ValueError(f"Order not found: {order_id}")

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        for i, order in enumerate(self._open_orders):
            if order.id == order_id:
                order.status = OrderStatus.CANCELLED
                self._open_orders.pop(i)
                return True
        return False

    async def get_balance(self) -> dict[str, float]:
        result: dict[str, float] = {"KRW": self._cash}
        for currency, qty in self._holdings.items():
            if qty > 0:
                result[currency] = qty
        return result

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        if symbol is None:
            return list(self._open_orders)
        return [o for o in self._open_orders if o.symbol == symbol]

    def equity(self) -> float:
        """Calculate total equity = cash + holdings value at last known prices."""
        total = self._cash
        for base_currency, qty in self._holdings.items():
            # Find a symbol with this base currency
            for sym, price in self._last_prices.items():
                if sym.startswith(base_currency + "/"):
                    total += price * qty
                    break
        return total

    async def close(self) -> None:
        await self._feed.close()
