"""Order fill simulator for backtesting.

Simulates realistic order execution including slippage and fees.
Market orders fill at the open of the next candle (conservative assumption).
"""

from __future__ import annotations

from dataclasses import dataclass

from tradingbot.config import BacktestConfig
from tradingbot.core.enums import OrderSide, OrderStatus, OrderType
from tradingbot.core.models import Candle, Order


@dataclass
class FillResult:
    """Result of attempting to fill an order."""

    filled: bool
    fill_price: float = 0.0
    fee: float = 0.0


class OrderSimulator:
    """Simulates order execution against candle data."""

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def simulate_fill(self, order: Order, candle: Candle) -> FillResult:
        """Attempt to fill an order using the given candle.

        Market orders: fill at candle open + slippage
        Limit orders: fill if candle's range crosses the limit price
        """
        if order.order_type == OrderType.MARKET:
            return self._fill_market(order, candle)
        elif order.order_type == OrderType.LIMIT:
            return self._fill_limit(order, candle)
        return FillResult(filled=False)

    def _fill_market(self, order: Order, candle: Candle) -> FillResult:
        """Fill market order at candle open with slippage."""
        base_price = candle.open

        # Apply slippage: buy gets worse (higher) price, sell gets worse (lower) price
        if order.side == OrderSide.BUY:
            fill_price = base_price * (1 + self.config.slippage_pct)
        else:
            fill_price = base_price * (1 - self.config.slippage_pct)

        fee = fill_price * order.quantity * self.config.fee_rate
        return FillResult(filled=True, fill_price=fill_price, fee=fee)

    def _fill_limit(self, order: Order, candle: Candle) -> FillResult:
        """Fill limit order if candle range crosses the limit price."""
        assert order.price is not None

        filled = False
        if order.side == OrderSide.BUY and candle.low <= order.price:
            filled = True
        elif order.side == OrderSide.SELL and candle.high >= order.price:
            filled = True

        if not filled:
            return FillResult(filled=False)

        fill_price = order.price
        fee = fill_price * order.quantity * self.config.fee_rate
        return FillResult(filled=True, fill_price=fill_price, fee=fee)

    def check_stop_loss(
        self, stop_loss_price: float, candle: Candle, quantity: float
    ) -> FillResult | None:
        """Check if a stop loss was triggered during this candle.

        Returns a FillResult if triggered, None otherwise.
        """
        if candle.low <= stop_loss_price:
            # Stop loss triggered — fill at stop price with slippage
            fill_price = stop_loss_price * (1 - self.config.slippage_pct)
            fee = fill_price * quantity * self.config.fee_rate
            return FillResult(filled=True, fill_price=fill_price, fee=fee)
        return None
