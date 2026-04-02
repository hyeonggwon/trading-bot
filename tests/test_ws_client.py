"""Tests for WebSocket client."""

from __future__ import annotations

from tradingbot.exchange.ws_client import (
    TickerData,
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


class TestTickerData:
    def test_creation(self):
        from datetime import datetime, timezone

        td = TickerData(
            symbol="BTC/KRW",
            price=50_000_000,
            volume=1234.5,
            change="RISE",
            timestamp=datetime.now(timezone.utc),
        )
        assert td.symbol == "BTC/KRW"
        assert td.price == 50_000_000
        assert td.change == "RISE"


class TestUpbitWebSocketClient:
    def test_init(self):
        from tradingbot.exchange.ws_client import UpbitWebSocketClient

        client = UpbitWebSocketClient(["BTC/KRW", "ETH/KRW"])
        assert client.last_prices == {}
        assert len(client._codes) == 2
        assert "KRW-BTC" in client._codes
        assert "KRW-ETH" in client._codes

    def test_callback_registration(self):
        from tradingbot.exchange.ws_client import UpbitWebSocketClient

        client = UpbitWebSocketClient(["BTC/KRW"])
        calls = []
        client.on_ticker(lambda td: calls.append(td))
        assert len(client._callbacks) == 1

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
        received = []
        client.on_ticker(lambda td: received.append(td))

        msg = {
            "type": "ticker",
            "code": "KRW-BTC",
            "trade_price": 50000000,
            "acc_trade_volume_24h": 1234.5,
            "change": "RISE",
        }
        asyncio.run(client._handle_message(msg))

        assert len(received) == 1
        assert received[0].symbol == "BTC/KRW"
        assert received[0].price == 50000000
        assert client.last_prices["BTC/KRW"] == 50000000

    def test_ignore_non_ticker(self):
        """Non-ticker messages should be ignored."""
        import asyncio
        from tradingbot.exchange.ws_client import UpbitWebSocketClient

        client = UpbitWebSocketClient(["BTC/KRW"])
        received = []
        client.on_ticker(lambda td: received.append(td))

        asyncio.run(client._handle_message({"type": "trade", "code": "KRW-BTC"}))
        assert len(received) == 0

    def test_ignore_zero_price(self):
        """Zero/negative prices should be ignored."""
        import asyncio
        from tradingbot.exchange.ws_client import UpbitWebSocketClient

        client = UpbitWebSocketClient(["BTC/KRW"])
        asyncio.run(client._handle_message({
            "type": "ticker", "code": "KRW-BTC", "trade_price": 0,
        }))
        assert "BTC/KRW" not in client.last_prices
