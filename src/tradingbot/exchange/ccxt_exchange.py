"""CCXT-based exchange implementation for Upbit.

Wraps the CCXT async client with rate limiting, retries, and error mapping.
Phase 4: read-only (fetch_ohlcv, fetch_ticker) + order creation for paper trading data feed.
Phase 5: full order management.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import ccxt.async_support as ccxt_async
import pandas as pd
import structlog

from tradingbot.config import EnvSettings, ExchangeConfig
from tradingbot.core.enums import OrderSide, OrderStatus, OrderType
from tradingbot.core.models import Order
from tradingbot.exchange.base import BaseExchange

logger = structlog.get_logger()

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0


class CcxtExchange(BaseExchange):
    """Real exchange connection via CCXT."""

    def __init__(
        self,
        exchange_config: ExchangeConfig | None = None,
        env: EnvSettings | None = None,
    ):
        config = exchange_config or ExchangeConfig()
        env = env or EnvSettings()

        exchange_class = getattr(ccxt_async, config.name)
        options: dict = {"enableRateLimit": True}

        if env.upbit_access_key and env.upbit_secret_key:
            options["apiKey"] = env.upbit_access_key
            options["secret"] = env.upbit_secret_key

        self._exchange: ccxt_async.Exchange = exchange_class(options)
        self._rate_limit_per_sec = config.rate_limit_per_sec

    async def _retry(self, coro_func, *args, **kwargs):
        """Execute with exponential backoff retry."""
        for attempt in range(MAX_RETRIES):
            try:
                return await coro_func(*args, **kwargs)
            except (ccxt_async.NetworkError, ccxt_async.ExchangeNotAvailable, ccxt_async.DDoSProtection) as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "exchange_retry",
                    error=str(e),
                    attempt=attempt + 1,
                    wait=wait,
                )
                await asyncio.sleep(wait)

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: int | None = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        ohlcv = await self._retry(
            self._exchange.fetch_ohlcv,
            symbol, timeframe, since, limit,
        )

        if not ohlcv:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        df = df.astype(float)
        return df

    async def fetch_ticker(self, symbol: str) -> dict:
        ticker = await self._retry(self._exchange.fetch_ticker, symbol)
        return {
            "last": ticker.get("last", 0),
            "bid": ticker.get("bid", 0),
            "ask": ticker.get("ask", 0),
            "volume": ticker.get("baseVolume", 0),
            "timestamp": datetime.now(timezone.utc),
        }

    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float | None = None,
    ) -> Order:
        ccxt_side = "buy" if side == OrderSide.BUY else "sell"
        ccxt_type = "market" if order_type == OrderType.MARKET else "limit"

        result = await self._retry(
            self._exchange.create_order,
            symbol, ccxt_type, ccxt_side, quantity, price,
        )

        return Order(
            id=result["id"],
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.PENDING,
            created_at=datetime.now(timezone.utc),
        )

    async def fetch_order(self, order_id: str, symbol: str) -> Order:
        raw = await self._retry(self._exchange.fetch_order, order_id, symbol)
        status_map = {
            "open": OrderStatus.PENDING,
            "closed": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "expired": OrderStatus.CANCELLED,
        }
        filled_price = raw.get("average") or raw.get("price")
        filled_at = None
        if raw.get("lastTradeTimestamp"):
            filled_at = datetime.fromtimestamp(
                raw["lastTradeTimestamp"] / 1000, tz=timezone.utc
            )

        # Safe parsing for side (can be None in some CCXT responses)
        raw_side = raw.get("side", "buy")
        side = OrderSide.BUY if raw_side == "buy" else OrderSide.SELL

        # Safe fee parsing (fee.cost can be None)
        raw_fee = raw.get("fee")
        fee_cost = float(raw_fee.get("cost") or 0) if raw_fee else 0.0

        return Order(
            id=raw["id"],
            symbol=raw.get("symbol", symbol),
            side=side,
            order_type=OrderType.MARKET if raw.get("type") == "market" else OrderType.LIMIT,
            quantity=float(raw.get("filled") or raw.get("amount") or 0),
            price=raw.get("price"),
            status=status_map.get(raw.get("status", ""), OrderStatus.PENDING),
            created_at=datetime.now(timezone.utc),
            filled_at=filled_at,
            filled_price=float(filled_price) if filled_price else None,
            fee=fee_cost,
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        try:
            await self._retry(self._exchange.cancel_order, order_id, symbol)
            return True
        except ccxt_async.OrderNotFound:
            logger.warning("order_not_found", order_id=order_id)
            return False

    async def get_balance(self) -> dict[str, float]:
        balance = await self._retry(self._exchange.fetch_balance)
        result = {}
        for currency, info in balance.get("free", {}).items():
            if info and float(info) > 0:
                result[currency] = float(info)
        return result

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        raw_orders = await self._retry(self._exchange.fetch_open_orders, symbol)
        orders = []
        for raw in raw_orders:
            orders.append(Order(
                id=raw["id"],
                symbol=raw["symbol"],
                side=OrderSide.BUY if raw["side"] == "buy" else OrderSide.SELL,
                order_type=OrderType.MARKET if raw["type"] == "market" else OrderType.LIMIT,
                quantity=raw["amount"],
                price=raw.get("price"),
                status=OrderStatus.PENDING,
                created_at=datetime.now(timezone.utc),
            ))
        return orders

    async def close(self) -> None:
        await self._exchange.close()
