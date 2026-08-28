"""Tests for WebSocket client."""

from __future__ import annotations

from datetime import UTC

import pytest

from tradingbot.exchange.ws_client import (
    _symbol_to_upbit_code,
    _upbit_code_to_symbol,
)


class TestSymbolConversion:
    def test_to_upbit_code(self):
        assert _symbol_to_upbit_code("BTC/KRW") == "KRW-BTC"
        assert _symbol_to_upbit_code("ETH/KRW") == "KRW-ETH"
        assert _symbol_to_upbit_code("XRP/KRW") == "KRW-XRP"

    def test_from_upbit_code(self):
        assert _upbit_code_to_symbol("KRW-BTC") == "BTC/KRW"
        assert _upbit_code_to_symbol("KRW-ETH") == "ETH/KRW"

    def test_roundtrip(self):
        symbols = ["BTC/KRW", "ETH/KRW", "XRP/KRW", "SOL/KRW"]
        for sym in symbols:
            code = _symbol_to_upbit_code(sym)
            assert _upbit_code_to_symbol(code) == sym


class TestUpbitWebSocketClient:
    def test_init(self):
        from tradingbot.exchange.ws_client import UpbitWebSocketClient

        client = UpbitWebSocketClient(["BTC/KRW", "ETH/KRW"])
        assert client.last_prices == {}
        assert len(client._codes) == 2
        assert "KRW-BTC" in client._codes
        assert "KRW-ETH" in client._codes

    def test_message_records_last_price_timestamp(self):
        """A parsed ticker message stamps _last_price_ts for that symbol."""
        import asyncio

        from tradingbot.exchange.ws_client import UpbitWebSocketClient

        client = UpbitWebSocketClient(["BTC/KRW"])
        assert "BTC/KRW" not in client._last_price_ts

        asyncio.run(
            client._handle_message(
                {
                    "type": "ticker",
                    "code": "KRW-BTC",
                    "trade_price": 50000000,
                }
            )
        )

        assert "BTC/KRW" in client._last_price_ts

    def test_stop(self):
        from tradingbot.exchange.ws_client import UpbitWebSocketClient

        client = UpbitWebSocketClient(["BTC/KRW"])
        client._running = True
        client.stop()
        assert client._running is False

    def test_handle_message(self):
        """Test message parsing without actual WebSocket connection."""
        import asyncio

        from tradingbot.exchange.ws_client import UpbitWebSocketClient

        client = UpbitWebSocketClient(["BTC/KRW"])

        msg = {
            "type": "ticker",
            "code": "KRW-BTC",
            "trade_price": 50000000,
            "acc_trade_volume_24h": 1234.5,
            "change": "RISE",
        }
        asyncio.run(client._handle_message(msg))

        assert client.last_prices["BTC/KRW"] == 50000000

    def test_ignore_non_ticker(self):
        """Non-ticker messages should be ignored."""
        import asyncio

        from tradingbot.exchange.ws_client import UpbitWebSocketClient

        client = UpbitWebSocketClient(["BTC/KRW"])

        asyncio.run(client._handle_message({"type": "trade", "code": "KRW-BTC"}))
        assert client.last_prices == {}

    def test_ignore_zero_price(self):
        """Zero/negative prices should be ignored."""
        import asyncio

        from tradingbot.exchange.ws_client import UpbitWebSocketClient

        client = UpbitWebSocketClient(["BTC/KRW"])
        asyncio.run(
            client._handle_message(
                {
                    "type": "ticker",
                    "code": "KRW-BTC",
                    "trade_price": 0,
                }
            )
        )
        assert "BTC/KRW" not in client.last_prices


class TestFreshPrices:
    """Regression: stale cached WS prices must not be served as current."""

    def test_drops_stale_keeps_fresh(self):
        from datetime import datetime, timedelta

        from tradingbot.exchange.ws_client import UpbitWebSocketClient

        client = UpbitWebSocketClient(["BTC/KRW", "ETH/KRW"])
        now = datetime.now(UTC)
        client._last_prices = {"BTC/KRW": 50_000_000, "ETH/KRW": 3_000_000}
        client._last_price_ts = {
            "BTC/KRW": now,  # fresh
            "ETH/KRW": now - timedelta(seconds=120),  # stale
        }
        assert client.fresh_prices(max_age_seconds=60) == {"BTC/KRW": 50_000_000}

    def test_price_without_timestamp_is_dropped(self):
        from tradingbot.exchange.ws_client import UpbitWebSocketClient

        client = UpbitWebSocketClient(["BTC/KRW"])
        client._last_prices = {"BTC/KRW": 50_000_000}  # no receive timestamp
        assert client.fresh_prices(max_age_seconds=60) == {}

    def test_handle_message_records_timestamp(self):
        import asyncio

        from tradingbot.exchange.ws_client import UpbitWebSocketClient

        client = UpbitWebSocketClient(["BTC/KRW"])
        asyncio.run(
            client._handle_message(
                {
                    "type": "ticker",
                    "code": "KRW-BTC",
                    "trade_price": 50_000_000,
                }
            )
        )
        # A just-received price is fresh under any sane age bound.
        assert client.fresh_prices(max_age_seconds=60) == {"BTC/KRW": 50_000_000}


class _FakeWs:
    """Minimal websockets-style connection yielding a fixed message list."""

    def __init__(self, messages: list[str]):
        self._messages = messages
        self.sent: list[str] = []

    async def send(self, msg: str) -> None:
        self.sent.append(msg)

    async def __aiter__(self):
        for msg in self._messages:
            yield msg


class _FakeConnect:
    def __init__(self, ws: _FakeWs):
        self._ws = ws

    async def __aenter__(self) -> _FakeWs:
        return self._ws

    async def __aexit__(self, *exc: object) -> bool:
        return False


class TestReconnectBackoffReset:
    """The backoff must reset only once data actually flows. Resetting at
    subscribe time lets a connection that drops right after subscribing
    reconnect at the base delay forever."""

    def _client(self, monkeypatch, messages: list[str]):
        from tradingbot.exchange import ws_client as module

        client = module.UpbitWebSocketClient(["BTC/KRW"])
        client._running = True
        client._reconnect_attempts = 3
        client._reconnect_delay = 16.0
        monkeypatch.setattr(module, "ws_connect", lambda *a, **k: _FakeConnect(_FakeWs(messages)))
        return client

    @pytest.mark.asyncio
    async def test_drop_before_any_message_keeps_backoff(self, monkeypatch):
        client = self._client(monkeypatch, [])

        await client._connect_and_stream()

        assert client._reconnect_attempts == 3
        assert client._reconnect_delay == 16.0

    @pytest.mark.asyncio
    async def test_first_message_resets_backoff(self, monkeypatch):
        import json

        from tradingbot.exchange.ws_client import RECONNECT_BASE_DELAY

        msg = json.dumps({"type": "ticker", "code": "KRW-BTC", "trade_price": 50_000_000})
        client = self._client(monkeypatch, [msg])

        await client._connect_and_stream()

        assert client._reconnect_attempts == 0
        assert client._reconnect_delay == RECONNECT_BASE_DELAY
        assert client.last_prices["BTC/KRW"] == 50_000_000
