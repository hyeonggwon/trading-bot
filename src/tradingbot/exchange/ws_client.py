"""Upbit WebSocket client for real-time market data.

Subscribes to ticker and trade streams. Provides:
- Real-time price updates for multiple symbols
- Automatic reconnection with exponential backoff
- Async callback interface for price updates

Upbit WebSocket endpoint: wss://api.upbit.com/websocket/v1
Subscription format:
  [{"ticket":"bot"}, {"type":"ticker","codes":["KRW-BTC","KRW-ETH"]}]
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Awaitable, Callable, Union

import structlog

try:
    import websockets
    from websockets.asyncio.client import connect as ws_connect
except ImportError:
    websockets = None  # type: ignore

logger = structlog.get_logger()

UPBIT_WS_URL = "wss://api.upbit.com/websocket/v1"
RECONNECT_BASE_DELAY = 2.0
RECONNECT_MAX_DELAY = 60.0
MAX_RECONNECT_ATTEMPTS = 50  # Consecutive failures before cooldown
COOLDOWN_SECONDS = 300  # 5 min cooldown after max retries, then retry again
PING_INTERVAL = 30


def _symbol_to_upbit_code(symbol: str) -> str:
    """Convert 'BTC/KRW' to 'KRW-BTC' (Upbit format)."""
    parts = symbol.split("/")
    if len(parts) == 2:
        return f"{parts[1]}-{parts[0]}"
    return symbol


def _upbit_code_to_symbol(code: str) -> str:
    """Convert 'KRW-BTC' to 'BTC/KRW'."""
    parts = code.split("-")
    if len(parts) == 2:
        return f"{parts[1]}/{parts[0]}"
    return code


class TickerData:
    """Parsed ticker update from WebSocket."""

    __slots__ = ("symbol", "price", "volume", "change", "timestamp")

    def __init__(
        self,
        symbol: str,
        price: float,
        volume: float,
        change: str,
        timestamp: datetime,
    ):
        self.symbol = symbol
        self.price = price
        self.volume = volume
        self.change = change
        self.timestamp = timestamp


# Callback: can be sync or async
TickerCallback = Union[Callable[[TickerData], None], Callable[[TickerData], Awaitable[None]]]


class UpbitWebSocketClient:
    """Async WebSocket client for Upbit real-time data.

    Usage:
        client = UpbitWebSocketClient(["BTC/KRW", "ETH/KRW"])
        client.on_ticker(my_callback)
        await client.run()  # Runs forever with auto-reconnect
    """

    def __init__(self, symbols: list[str]):
        if websockets is None:
            raise ImportError("websockets package required: pip install websockets")

        self._symbols = symbols
        self._codes = [_symbol_to_upbit_code(s) for s in symbols]
        self._callbacks: list[TickerCallback] = []
        self._running = False
        self._stop_event: asyncio.Event | None = None
        self._last_prices: dict[str, float] = {}
        self._reconnect_delay = RECONNECT_BASE_DELAY
        self._reconnect_attempts = 0

    @property
    def last_prices(self) -> dict[str, float]:
        """Last known prices per symbol (e.g., {'BTC/KRW': 50000000})."""
        return dict(self._last_prices)

    def on_ticker(self, callback: TickerCallback) -> None:
        """Register a callback for ticker updates."""
        self._callbacks.append(callback)

    async def run(self) -> None:
        """Connect and stream data. Reconnects automatically on failure."""
        self._running = True
        self._stop_event = asyncio.Event()

        while self._running:
            try:
                await self._connect_and_stream()
            except Exception as e:
                if not self._running:
                    break

                self._reconnect_attempts += 1
                if self._reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                    logger.warning(
                        "ws_cooldown",
                        attempts=self._reconnect_attempts,
                        cooldown=f"{COOLDOWN_SECONDS}s",
                    )
                    # Cooldown then reset — allows recovery after temporary outages
                    try:
                        await asyncio.wait_for(
                            self._stop_event.wait(), timeout=COOLDOWN_SECONDS
                        )
                        break  # stop() was called during cooldown
                    except asyncio.TimeoutError:
                        pass
                    self._reconnect_attempts = 0
                    self._reconnect_delay = RECONNECT_BASE_DELAY
                    continue

                logger.warning(
                    "ws_disconnected",
                    error=str(e),
                    attempt=self._reconnect_attempts,
                    reconnect_in=f"{self._reconnect_delay:.0f}s",
                )

                # Interruptible sleep — stop() wakes us immediately
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self._reconnect_delay
                    )
                    break  # stop() was called
                except asyncio.TimeoutError:
                    pass  # Normal timeout, retry

                self._reconnect_delay = min(
                    self._reconnect_delay * 2, RECONNECT_MAX_DELAY
                )

        logger.info("ws_client_stopped")

    async def _connect_and_stream(self) -> None:
        """Single connection lifecycle: connect, subscribe, receive messages."""
        logger.info("ws_connecting", url=UPBIT_WS_URL, symbols=len(self._codes))

        async with ws_connect(
            UPBIT_WS_URL,
            ping_interval=PING_INTERVAL,
            ping_timeout=10,
        ) as ws:
            # Subscribe to ticker stream
            subscribe_msg = json.dumps([
                {"ticket": str(uuid.uuid4())[:8]},
                {
                    "type": "ticker",
                    "codes": self._codes,
                    "isOnlyRealtime": True,
                },
            ])
            await ws.send(subscribe_msg)
            logger.info("ws_subscribed", codes=self._codes)

            # Reset reconnect state on successful connection
            self._reconnect_delay = RECONNECT_BASE_DELAY
            self._reconnect_attempts = 0

            # Receive loop
            async for raw_msg in ws:
                if not self._running:
                    break

                try:
                    if isinstance(raw_msg, bytes):
                        raw_msg = raw_msg.decode("utf-8")
                    data = json.loads(raw_msg)
                    await self._handle_message(data)
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.debug("ws_parse_error", error=str(e))

    async def _handle_message(self, data: dict) -> None:
        """Parse and dispatch a ticker message."""
        msg_type = data.get("type")
        if msg_type != "ticker":
            return

        code = data.get("code", "")
        symbol = _upbit_code_to_symbol(code)
        price = float(data.get("trade_price", 0))

        if price <= 0:
            return

        self._last_prices[symbol] = price

        ticker = TickerData(
            symbol=symbol,
            price=price,
            volume=float(data.get("acc_trade_volume_24h", 0)),
            change=data.get("change", ""),
            timestamp=datetime.fromtimestamp(data.get("timestamp", 0) / 1000, tz=timezone.utc),
        )

        for callback in self._callbacks:
            try:
                result = callback(ticker)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("ws_callback_error")

    def stop(self) -> None:
        """Request graceful shutdown. Interrupts reconnect sleep immediately."""
        self._running = False
        if self._stop_event is not None:
            self._stop_event.set()
