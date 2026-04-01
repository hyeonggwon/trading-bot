"""Abstract exchange interface.

Both CcxtExchange (real) and PaperExchange (simulated) implement this
interface, ensuring the live engine can swap between them seamlessly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from tradingbot.core.enums import OrderSide, OrderType
from tradingbot.core.models import Order


class BaseExchange(ABC):
    """Abstract base for all exchange implementations."""

    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: int | None = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch OHLCV candle data.

        Returns DataFrame with DatetimeIndex and columns: open, high, low, close, volume.
        """
        ...

    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> dict:
        """Fetch current ticker (last price, bid, ask, volume).

        Returns dict with keys: last, bid, ask, volume, timestamp.
        """
        ...

    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float | None = None,
    ) -> Order:
        """Create a new order on the exchange."""
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order. Returns True if successful."""
        ...

    @abstractmethod
    async def get_balance(self) -> dict[str, float]:
        """Get account balances.

        Returns dict mapping currency to available balance.
        E.g., {"KRW": 1000000, "BTC": 0.01}
        """
        ...

    @abstractmethod
    async def fetch_order(self, order_id: str, symbol: str) -> Order:
        """Fetch the current state of an order by ID."""
        ...

    @abstractmethod
    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Get list of open (unfilled) orders."""
        ...

    async def close(self) -> None:
        """Cleanup resources (connections, sessions)."""
        pass
