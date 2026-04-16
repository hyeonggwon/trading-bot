"""Tests for live engine bug fixes (price handling, stop loss, equity)."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from tradingbot.config import AppConfig, RiskConfig
from tradingbot.core.enums import OrderSide, OrderStatus, OrderType, PositionSide, SignalType
from tradingbot.core.models import Position, Signal
from tradingbot.exchange.base import BaseExchange
from tradingbot.exchange.paper import PaperExchange
from tradingbot.live.engine import LiveEngine
from tradingbot.live.state import StateManager
from tradingbot.strategy.base import Strategy

# --- Helpers ---

class MockDataFeed(BaseExchange):
    """Minimal mock exchange for testing."""

    def __init__(self, price: float = 50_000_000):
        self._price = price

    async def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=100):
        dates = pd.date_range("2024-01-01", periods=limit, freq="h", tz="UTC")
        return pd.DataFrame({
            "open": [self._price] * limit,
            "high": [self._price * 1.01] * limit,
            "low": [self._price * 0.99] * limit,
            "close": [self._price] * limit,
            "volume": [100] * limit,
        }, index=dates)

    async def fetch_ticker(self, symbol):
        return {"last": self._price, "bid": self._price * 0.999,
                "ask": self._price * 1.001, "volume": 100,
                "timestamp": datetime.now(UTC)}

    async def create_order(self, symbol, side, order_type, quantity, price=None):
        raise NotImplementedError

    async def fetch_order(self, order_id, symbol):
        raise NotImplementedError

    async def cancel_order(self, order_id, symbol):
        return False

    async def get_balance(self):
        return {"KRW": 10_000_000}

    async def get_open_orders(self, symbol=None):
        return []

    async def close(self):
        pass


class StubStrategy(Strategy):
    """Strategy that never signals."""

    def __init__(self):
        self._symbols = ["BTC/KRW"]
        self._timeframe = "1h"

    @property
    def symbols(self):
        return self._symbols

    @property
    def timeframe(self):
        return self._timeframe

    def indicators(self, df):
        return df

    def should_entry(self, df, symbol):
        return None

    def should_exit(self, df, symbol, position=None):
        return None


# --- Bug 1: update_prices syncs cache ---

class TestUpdatePrices:
    def test_update_prices_syncs_cache(self):
        """update_prices() should update _last_prices."""
        feed = MockDataFeed(price=50_000_000)
        paper = PaperExchange(data_feed=feed, initial_balance=10_000_000)

        paper.update_prices({"BTC/KRW": 51_000_000, "ETH/KRW": 3_500_000})

        assert paper._last_prices["BTC/KRW"] == 51_000_000
        assert paper._last_prices["ETH/KRW"] == 3_500_000

    @pytest.mark.asyncio
    async def test_paper_fill_uses_updated_price(self):
        """After update_prices, market order should fill at updated price."""
        feed = MockDataFeed(price=50_000_000)
        paper = PaperExchange(
            data_feed=feed, initial_balance=10_000_000,
            fee_rate=0.0005, slippage_pct=0.001,
        )

        # Set price via update_prices (simulating WebSocket)
        paper.update_prices({"BTC/KRW": 60_000_000})

        order = await paper.create_order(
            "BTC/KRW", OrderSide.BUY, OrderType.MARKET, 0.001
        )

        assert order.status == OrderStatus.FILLED
        # Fill price should be based on 60M (with slippage), not 50M
        expected_fill = 60_000_000 * 1.001
        assert order.filled_price == pytest.approx(expected_fill, rel=1e-6)


# --- Bug 2: stop_loss uses filled_price ---

class TestStopLossCalculation:
    @pytest.mark.asyncio
    async def test_stop_loss_uses_filled_price(self, tmp_path):
        """Position stop_loss should be based on filled_price, not current_price."""
        feed = MockDataFeed(price=50_000_000)
        paper = PaperExchange(
            data_feed=feed, initial_balance=10_000_000,
            fee_rate=0.0005, slippage_pct=0.001,
        )
        # Set price so fills happen at known price
        paper.update_prices({"BTC/KRW": 50_000_000})

        config = AppConfig(risk=RiskConfig(
            default_stop_loss_pct=0.02,
            risk_per_trade_pct=0.01,
            max_position_size_pct=0.1,
        ))
        state = StateManager(tmp_path / "state.json")
        strategy = StubStrategy()

        engine = LiveEngine(
            strategy=strategy, exchange=paper, config=config,
            state_manager=state,
        )

        # Simulate entry with current_price different from what paper will fill at
        current_price = 50_000_000
        signal = Signal(
            timestamp=datetime.now(UTC),
            symbol="BTC/KRW",
            signal_type=SignalType.LONG_ENTRY,
            price=current_price,
            strength=1.0,
        )
        await engine._handle_entry(signal, "BTC/KRW", current_price)

        pos = state.positions.get("BTC/KRW")
        assert pos is not None

        # filled_price = 50M * 1.001 = 50,050,000
        expected_fill = 50_000_000 * 1.001
        assert pos.entry_price == pytest.approx(expected_fill, rel=1e-4)

        # stop_loss should be based on filled_price, not current_price
        expected_stop = expected_fill * (1 - 0.02)
        assert pos.stop_loss == pytest.approx(expected_stop, rel=1e-4)


# --- Bug 4: stop loss triggers exit ---

class TestStopLossEnforcement:
    @pytest.mark.asyncio
    async def test_stop_loss_triggers_exit(self, tmp_path):
        """When current_price <= stop_loss, position should be closed."""
        feed = MockDataFeed(price=48_000_000)  # Below stop loss
        paper = PaperExchange(
            data_feed=feed, initial_balance=10_000_000,
            fee_rate=0.0005, slippage_pct=0.001,
        )
        paper.update_prices({"BTC/KRW": 48_000_000})
        # Give paper some holdings to sell
        paper._holdings["BTC"] = 0.001

        config = AppConfig(risk=RiskConfig(default_stop_loss_pct=0.02))
        state = StateManager(tmp_path / "state.json")
        strategy = StubStrategy()

        engine = LiveEngine(
            strategy=strategy, exchange=paper, config=config,
            state_manager=state,
        )

        # Create position with stop_loss at 49M
        state.positions["BTC/KRW"] = Position(
            symbol="BTC/KRW",
            side=PositionSide.LONG,
            size=0.001,
            entry_price=50_000_000,
            entry_time=datetime.now(UTC),
            stop_loss=49_000_000,
        )

        # Fetch candles for _tick_symbol
        df = await paper.fetch_ohlcv("BTC/KRW", "1h", limit=10)
        ticker = {"last": 48_000_000}  # Below stop_loss

        await engine._tick_symbol("BTC/KRW", df, ticker)

        # Position should be closed
        assert "BTC/KRW" not in state.positions

    @pytest.mark.asyncio
    async def test_stop_loss_no_trigger_above(self, tmp_path):
        """When current_price > stop_loss, position should remain."""
        feed = MockDataFeed(price=51_000_000)
        paper = PaperExchange(
            data_feed=feed, initial_balance=10_000_000,
            fee_rate=0.0005, slippage_pct=0.001,
        )
        paper.update_prices({"BTC/KRW": 51_000_000})

        config = AppConfig(risk=RiskConfig(default_stop_loss_pct=0.02))
        state = StateManager(tmp_path / "state.json")
        strategy = StubStrategy()

        engine = LiveEngine(
            strategy=strategy, exchange=paper, config=config,
            state_manager=state,
        )

        state.positions["BTC/KRW"] = Position(
            symbol="BTC/KRW",
            side=PositionSide.LONG,
            size=0.001,
            entry_price=50_000_000,
            entry_time=datetime.now(UTC),
            stop_loss=49_000_000,
        )

        df = await paper.fetch_ohlcv("BTC/KRW", "1h", limit=10)
        ticker = {"last": 51_000_000}  # Above stop_loss

        await engine._tick_symbol("BTC/KRW", df, ticker)

        # Position should still exist
        assert "BTC/KRW" in state.positions


# --- Bug 5, 6: slippage-adjusted sizing ---

class TestSlippageAdjustedSizing:
    @pytest.mark.asyncio
    async def test_position_size_uses_expected_price(self, tmp_path):
        """Position size should be calculated with slippage-adjusted price."""
        feed = MockDataFeed(price=50_000_000)
        paper = PaperExchange(
            data_feed=feed, initial_balance=10_000_000,
            fee_rate=0.0005, slippage_pct=0.001,
        )
        paper.update_prices({"BTC/KRW": 50_000_000})

        config = AppConfig(risk=RiskConfig(
            default_stop_loss_pct=0.02,
            risk_per_trade_pct=0.01,
            max_position_size_pct=0.1,
        ))
        state = StateManager(tmp_path / "state.json")
        strategy = StubStrategy()

        engine = LiveEngine(
            strategy=strategy, exchange=paper, config=config,
            state_manager=state,
        )

        current_price = 50_000_000
        signal = Signal(
            timestamp=datetime.now(UTC),
            symbol="BTC/KRW",
            signal_type=SignalType.LONG_ENTRY,
            price=current_price,
            strength=1.0,
        )
        await engine._handle_entry(signal, "BTC/KRW", current_price)

        pos = state.positions.get("BTC/KRW")
        assert pos is not None

        # Position value should be within max_position_size_pct of equity
        # Using expected_price (50M * 1.001 = 50.05M)
        position_value = pos.size * pos.entry_price
        assert position_value <= 10_000_000 * 0.1 * 1.01  # Allow small margin


# --- Bug 7: equity history recording ---

class TestEquityRecording:
    @pytest.mark.asyncio
    async def test_equity_recorded_each_tick(self, tmp_path):
        """Each tick should record equity in state history."""
        feed = MockDataFeed(price=50_000_000)
        paper = PaperExchange(
            data_feed=feed, initial_balance=10_000_000,
            fee_rate=0.0005, slippage_pct=0.001,
        )

        config = AppConfig(risk=RiskConfig())
        state = StateManager(tmp_path / "state.json")
        strategy = StubStrategy()

        engine = LiveEngine(
            strategy=strategy, exchange=paper, config=config,
            state_manager=state,
        )
        # Set last candle ts so _tick_symbol processes
        engine._last_candle_ts = {}

        assert len(state.equity_history) == 0

        await engine._tick_all(["BTC/KRW"], "1h")

        assert len(state.equity_history) == 1
        assert state.equity_history[0]["equity"] > 0

        await engine._tick_all(["BTC/KRW"], "1h")

        assert len(state.equity_history) == 2
