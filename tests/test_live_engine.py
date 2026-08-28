"""Tests for live engine bug fixes (price handling, stop loss, equity)."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pandas as pd
import pytest

from tradingbot.config import AppConfig, PyramidingConfig, RiskConfig
from tradingbot.core.enums import OrderSide, OrderStatus, OrderType, PositionSide, SignalType
from tradingbot.core.models import Order, Position, Signal
from tradingbot.exchange.base import BaseExchange
from tradingbot.exchange.paper import PaperExchange
from tradingbot.live.engine import LiveEngine, _order_has_fill
from tradingbot.live.state import StateManager
from tradingbot.strategy.base import Strategy

# --- Helpers ---


class MockDataFeed(BaseExchange):
    """Minimal mock exchange for testing."""

    def __init__(self, price: float = 50_000_000):
        self._price = price

    async def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=100):
        dates = pd.date_range("2024-01-01", periods=limit, freq="h", tz="UTC")
        return pd.DataFrame(
            {
                "open": [self._price] * limit,
                "high": [self._price * 1.01] * limit,
                "low": [self._price * 0.99] * limit,
                "close": [self._price] * limit,
                "volume": [100] * limit,
            },
            index=dates,
        )

    async def fetch_ticker(self, symbol):
        return {
            "last": self._price,
            "bid": self._price * 0.999,
            "ask": self._price * 1.001,
            "volume": 100,
            "timestamp": datetime.now(UTC),
        }

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
            data_feed=feed,
            initial_balance=10_000_000,
            fee_rate=0.0005,
            slippage_pct=0.001,
        )

        # Set price via update_prices (simulating WebSocket)
        paper.update_prices({"BTC/KRW": 60_000_000})

        order = await paper.create_order("BTC/KRW", OrderSide.BUY, OrderType.MARKET, 0.001)

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
            data_feed=feed,
            initial_balance=10_000_000,
            fee_rate=0.0005,
            slippage_pct=0.001,
        )
        # Set price so fills happen at known price
        paper.update_prices({"BTC/KRW": 50_000_000})

        config = AppConfig(
            risk=RiskConfig(
                default_stop_loss_pct=0.02,
                risk_per_trade_pct=0.01,
                max_position_size_pct=0.1,
            )
        )
        state = StateManager(tmp_path / "state.json")
        strategy = StubStrategy()

        engine = LiveEngine(
            strategy=strategy,
            exchange=paper,
            config=config,
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
            data_feed=feed,
            initial_balance=10_000_000,
            fee_rate=0.0005,
            slippage_pct=0.001,
        )
        paper.update_prices({"BTC/KRW": 48_000_000})
        # Give paper some holdings to sell
        paper._holdings["BTC"] = 0.001

        config = AppConfig(risk=RiskConfig(default_stop_loss_pct=0.02))
        state = StateManager(tmp_path / "state.json")
        strategy = StubStrategy()

        engine = LiveEngine(
            strategy=strategy,
            exchange=paper,
            config=config,
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
            data_feed=feed,
            initial_balance=10_000_000,
            fee_rate=0.0005,
            slippage_pct=0.001,
        )
        paper.update_prices({"BTC/KRW": 51_000_000})

        config = AppConfig(risk=RiskConfig(default_stop_loss_pct=0.02))
        state = StateManager(tmp_path / "state.json")
        strategy = StubStrategy()

        engine = LiveEngine(
            strategy=strategy,
            exchange=paper,
            config=config,
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
            data_feed=feed,
            initial_balance=10_000_000,
            fee_rate=0.0005,
            slippage_pct=0.001,
        )
        paper.update_prices({"BTC/KRW": 50_000_000})

        config = AppConfig(
            risk=RiskConfig(
                default_stop_loss_pct=0.02,
                risk_per_trade_pct=0.01,
                max_position_size_pct=0.1,
            )
        )
        state = StateManager(tmp_path / "state.json")
        strategy = StubStrategy()

        engine = LiveEngine(
            strategy=strategy,
            exchange=paper,
            config=config,
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
            data_feed=feed,
            initial_balance=10_000_000,
            fee_rate=0.0005,
            slippage_pct=0.001,
        )

        config = AppConfig(risk=RiskConfig())
        state = StateManager(tmp_path / "state.json")
        strategy = StubStrategy()

        engine = LiveEngine(
            strategy=strategy,
            exchange=paper,
            config=config,
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


# --- Bug: Signal.strength must scale position size (ML Half-Kelly) ---


class TestSignalStrengthSizing:
    @pytest.mark.asyncio
    async def test_strength_scales_position_size(self, tmp_path):
        """Entry quantity must scale linearly with Signal.strength.

        The backtest engine multiplies size by signal.strength; the live path
        must match or ML probability-based sizing silently over-trades.
        """

        async def _entry(strength: float) -> float:
            feed = MockDataFeed(price=50_000_000)
            paper = PaperExchange(
                data_feed=feed,
                initial_balance=10_000_000,
                fee_rate=0.0005,
                slippage_pct=0.001,
            )
            paper.update_prices({"BTC/KRW": 50_000_000})
            config = AppConfig(
                risk=RiskConfig(
                    default_stop_loss_pct=0.02,
                    risk_per_trade_pct=0.01,
                    max_position_size_pct=0.5,
                )
            )
            state = StateManager(tmp_path / f"state_{strength}.json")
            engine = LiveEngine(
                strategy=StubStrategy(),
                exchange=paper,
                config=config,
                state_manager=state,
            )
            signal = Signal(
                timestamp=datetime.now(UTC),
                symbol="BTC/KRW",
                signal_type=SignalType.LONG_ENTRY,
                price=50_000_000,
                strength=strength,
            )
            await engine._handle_entry(signal, "BTC/KRW", 50_000_000)
            pos = state.positions.get("BTC/KRW")
            return pos.size if pos else 0.0

        full = await _entry(1.0)
        half = await _entry(0.5)
        assert full > 0
        # strength is applied after sizing, so half is exactly half of full.
        assert half == pytest.approx(full * 0.5, rel=1e-9)


# --- Bug: stop loss must be enforced between candles, not only at close ---


class TestMonitorStopEnforcement:
    @pytest.mark.asyncio
    async def test_monitor_closes_position_on_stop_breach(self, tmp_path):
        """_monitor_prices must close a stopped-out position without a new candle.

        On a 4h timeframe, waiting for candle close leaves a breached stop
        unguarded for hours. The between-candle monitor must enforce it.
        """
        feed = MockDataFeed(price=48_000_000)  # below the stop
        paper = PaperExchange(
            data_feed=feed,
            initial_balance=10_000_000,
            fee_rate=0.0005,
            slippage_pct=0.001,
        )
        paper.update_prices({"BTC/KRW": 48_000_000})
        paper._holdings["BTC"] = 0.001  # holdings to sell on exit

        config = AppConfig(risk=RiskConfig(default_stop_loss_pct=0.02))
        state = StateManager(tmp_path / "state.json")
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=paper,
            config=config,
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

        # No candle is processed here — only the monitor runs.
        await engine._monitor_prices(["BTC/KRW"])

        assert "BTC/KRW" not in state.positions

    @pytest.mark.asyncio
    async def test_monitor_keeps_position_above_stop(self, tmp_path):
        """Monitor must not close a position whose price is above the stop."""
        feed = MockDataFeed(price=51_000_000)
        paper = PaperExchange(
            data_feed=feed,
            initial_balance=10_000_000,
            fee_rate=0.0005,
            slippage_pct=0.001,
        )
        paper.update_prices({"BTC/KRW": 51_000_000})

        config = AppConfig(risk=RiskConfig(default_stop_loss_pct=0.02))
        state = StateManager(tmp_path / "state.json")
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=paper,
            config=config,
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

        await engine._monitor_prices(["BTC/KRW"])

        assert "BTC/KRW" in state.positions


# --- Take profit enforcement (sibling of the stop loss rails) ---


class TestTakeProfitEnforcement:
    @pytest.mark.asyncio
    async def test_take_profit_triggers_exit(self, tmp_path):
        """When current_price >= take_profit, the position should be closed."""
        feed = MockDataFeed(price=52_000_000)  # above the target
        paper = PaperExchange(
            data_feed=feed,
            initial_balance=10_000_000,
            fee_rate=0.0005,
            slippage_pct=0.001,
        )
        paper.update_prices({"BTC/KRW": 52_000_000})
        paper._holdings["BTC"] = 0.001  # holdings to sell on exit

        config = AppConfig(risk=RiskConfig(default_stop_loss_pct=0.02))
        state = StateManager(tmp_path / "state.json")
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=paper,
            config=config,
            state_manager=state,
        )
        state.positions["BTC/KRW"] = Position(
            symbol="BTC/KRW",
            side=PositionSide.LONG,
            size=0.001,
            entry_price=50_000_000,
            entry_time=datetime.now(UTC),
            stop_loss=49_000_000,  # far below — must not trigger here
            take_profit=51_500_000,
        )

        df = await paper.fetch_ohlcv("BTC/KRW", "1h", limit=10)
        ticker = {"last": 52_000_000}  # at/above the target
        await engine._tick_symbol("BTC/KRW", df, ticker)

        assert "BTC/KRW" not in state.positions

    @pytest.mark.asyncio
    async def test_take_profit_no_trigger_below(self, tmp_path):
        """When current_price < take_profit, the position should remain open."""
        feed = MockDataFeed(price=50_500_000)
        paper = PaperExchange(
            data_feed=feed,
            initial_balance=10_000_000,
            fee_rate=0.0005,
            slippage_pct=0.001,
        )
        paper.update_prices({"BTC/KRW": 50_500_000})

        config = AppConfig(risk=RiskConfig(default_stop_loss_pct=0.02))
        state = StateManager(tmp_path / "state.json")
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=paper,
            config=config,
            state_manager=state,
        )
        state.positions["BTC/KRW"] = Position(
            symbol="BTC/KRW",
            side=PositionSide.LONG,
            size=0.001,
            entry_price=50_000_000,
            entry_time=datetime.now(UTC),
            stop_loss=49_000_000,
            take_profit=51_500_000,
        )

        df = await paper.fetch_ohlcv("BTC/KRW", "1h", limit=10)
        await engine._tick_symbol("BTC/KRW", df, {"last": 50_500_000})

        assert "BTC/KRW" in state.positions

    @pytest.mark.asyncio
    async def test_monitor_closes_position_on_take_profit(self, tmp_path):
        """_monitor_prices must realize a hit take profit between candle closes."""
        feed = MockDataFeed(price=52_000_000)
        paper = PaperExchange(
            data_feed=feed,
            initial_balance=10_000_000,
            fee_rate=0.0005,
            slippage_pct=0.001,
        )
        paper.update_prices({"BTC/KRW": 52_000_000})
        paper._holdings["BTC"] = 0.001

        config = AppConfig(risk=RiskConfig(default_stop_loss_pct=0.02))
        state = StateManager(tmp_path / "state.json")
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=paper,
            config=config,
            state_manager=state,
        )
        state.positions["BTC/KRW"] = Position(
            symbol="BTC/KRW",
            side=PositionSide.LONG,
            size=0.001,
            entry_price=50_000_000,
            entry_time=datetime.now(UTC),
            stop_loss=49_000_000,
            take_profit=51_500_000,
        )

        await engine._monitor_prices(["BTC/KRW"])

        assert "BTC/KRW" not in state.positions

    @pytest.mark.asyncio
    async def test_take_profit_set_on_entry(self, tmp_path):
        """A new position must carry a take profit derived from its filled price
        when default_take_profit_pct is configured."""
        feed = MockDataFeed(price=50_000_000)
        paper = PaperExchange(
            data_feed=feed,
            initial_balance=10_000_000,
            fee_rate=0.0005,
            slippage_pct=0.001,
        )
        paper.update_prices({"BTC/KRW": 50_000_000})

        config = AppConfig(
            risk=RiskConfig(
                default_stop_loss_pct=0.02,
                default_take_profit_pct=0.03,
                risk_per_trade_pct=0.01,
                max_position_size_pct=0.1,
            )
        )
        state = StateManager(tmp_path / "state.json")
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=paper,
            config=config,
            state_manager=state,
        )
        signal = Signal(
            timestamp=datetime.now(UTC),
            symbol="BTC/KRW",
            signal_type=SignalType.LONG_ENTRY,
            price=50_000_000,
            strength=1.0,
        )
        await engine._handle_entry(signal, "BTC/KRW", 50_000_000)

        pos = state.positions.get("BTC/KRW")
        assert pos is not None
        expected_fill = 50_000_000 * 1.001
        assert pos.take_profit == pytest.approx(expected_fill * 1.03, rel=1e-4)


# --- Bug: partial WS staleness must not drop a symbol from price resolution ---


class StubWsClient:
    """WS client stub whose fresh_prices returns only a chosen subset."""

    def __init__(self, fresh: dict):
        self._fresh = fresh

    def fresh_prices(self, max_age_seconds):
        return dict(self._fresh)


class TestResolveTickersPartialStaleness:
    @pytest.mark.asyncio
    async def test_stale_symbol_falls_back_to_rest_per_symbol(self, tmp_path):
        """A symbol absent from fresh WS prices must still get a REST ticker.

        Previously _resolve_tickers was all-or-nothing: any single fresh WS
        price made it return the WS-only dict and skip REST for everyone, so a
        quiet symbol that hadn't ticked within the staleness window got no
        price at all — and its stop loss was silently dropped from monitoring.
        Each missing symbol must fall back to REST individually.
        """
        feed = MockDataFeed(price=3_000_000)  # REST price (distinct from WS)
        config = AppConfig(risk=RiskConfig())
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=feed,
            config=config,
            state_manager=StateManager(tmp_path / "state.json"),
            ws_client=StubWsClient({"BTC/KRW": 50_000_000}),  # only BTC is fresh
        )

        tickers = await engine._resolve_tickers(["BTC/KRW", "ETH/KRW"])

        # Fresh WS price used where available...
        assert tickers["BTC/KRW"]["last"] == 50_000_000
        # ...and the stale symbol is REST-filled rather than dropped.
        assert "ETH/KRW" in tickers
        assert tickers["ETH/KRW"]["last"] == 3_000_000


# --- Bug: real-money safety rails must survive a restart ---


class TestStatePersistenceAcrossRestart:
    @pytest.mark.asyncio
    async def test_peak_equity_and_daily_pnl_survive_restart(self, tmp_path):
        """peak_equity (drawdown breaker) and daily PnL (loss limit) must persist.

        A restart that zeroes either one silently disarms a real-money safety
        rail: the drawdown circuit breaker re-baselines and the daily-loss
        counter forgets losses already booked.
        """
        from tradingbot.risk.validators import TradeValidator

        state_path = tmp_path / "state.json"
        feed = MockDataFeed(price=50_000_000)
        paper = PaperExchange(data_feed=feed, initial_balance=10_000_000)
        config = AppConfig(risk=RiskConfig())

        # First session: accumulate safety-rail state, then persist.
        v1 = TradeValidator(daily_loss_limit_krw=200_000)
        v1.record_trade_pnl(-150_000)
        engine1 = LiveEngine(
            strategy=StubStrategy(),
            exchange=paper,
            config=config,
            state_manager=StateManager(state_path),
            trade_validator=v1,
        )
        engine1.risk_manager.peak_equity = 12_345_678.0
        engine1._persist_state()

        # Second session (restart): fresh objects load from the same file.
        v2 = TradeValidator(daily_loss_limit_krw=200_000)
        engine2 = LiveEngine(
            strategy=StubStrategy(),
            exchange=paper,
            config=config,
            state_manager=StateManager(state_path),
            trade_validator=v2,
        )
        engine2._restore_state()

        assert engine2.risk_manager.peak_equity == 12_345_678.0
        assert v2.daily_state()[0] == -150_000

    @pytest.mark.asyncio
    async def test_corrupt_daily_reset_date_does_not_crash_restart(self, tmp_path):
        """A corrupt daily_reset_date in state.json must not raise out of
        _restore_state and abort startup — that would defeat the crash-recovery
        the restore exists for. The bad date is dropped; daily PnL still loads."""
        from tradingbot.risk.validators import TradeValidator

        state_path = tmp_path / "state.json"
        feed = MockDataFeed(price=50_000_000)
        paper = PaperExchange(data_feed=feed, initial_balance=10_000_000)
        config = AppConfig(risk=RiskConfig())

        # Persist a state whose daily_reset_date is not a valid ISO date.
        sm = StateManager(state_path)
        sm.daily_pnl = -75_000
        sm.daily_reset_date = "not-a-date"
        sm.save()

        v = TradeValidator(daily_loss_limit_krw=200_000)
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=paper,
            config=config,
            state_manager=StateManager(state_path),
            trade_validator=v,
        )
        # Old code: date.fromisoformat("not-a-date") raises ValueError.
        engine._restore_state()

        # Daily PnL still restored (loss counter not silently zeroed).
        assert v.daily_state()[0] == -75_000


# --- CRITICAL: exchange <-> local state reconciliation ---


class TestExchangeReconciliation:
    @pytest.mark.asyncio
    async def test_adopts_orphan_holding_with_stop(self, tmp_path):
        """A tradable holding the exchange reports but local state has no
        position for (e.g. a fill whose response was lost) must be adopted with
        a synthesized stop — otherwise it runs unmanaged and the engine, still
        believing it is flat, could buy more and double exposure."""

        class OrphanFeed(MockDataFeed):
            async def get_balance(self):
                return {"KRW": 5_000_000, "BTC": 0.1}

        feed = OrphanFeed(price=50_000_000)
        config = AppConfig(risk=RiskConfig(default_stop_loss_pct=0.02))
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=feed,
            config=config,
            state_manager=StateManager(tmp_path / "state.json"),
        )

        assert "BTC/KRW" not in engine.state.positions
        await engine._reconcile_with_exchange()

        pos = engine.state.positions.get("BTC/KRW")
        assert pos is not None
        assert pos.size == pytest.approx(0.1)
        assert pos.entry_price == pytest.approx(50_000_000)
        assert pos.stop_loss is not None and pos.stop_loss < pos.entry_price

    @pytest.mark.asyncio
    async def test_ignores_untracked_currency(self, tmp_path):
        """A holding in a currency the strategy does not trade is not ours to
        manage and must be left alone."""

        class EthFeed(MockDataFeed):
            async def get_balance(self):
                return {"KRW": 5_000_000, "ETH": 1.0}

        feed = EthFeed(price=50_000_000)
        config = AppConfig(risk=RiskConfig())
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=feed,
            config=config,
            state_manager=StateManager(tmp_path / "state.json"),
        )

        await engine._reconcile_with_exchange()
        assert engine.state.positions == {}

    @pytest.mark.asyncio
    async def test_drops_phantom_position(self, tmp_path):
        """A locally-tracked position the exchange no longer holds must be
        dropped — selling it would fail and PnL accounting would be wrong."""

        class FlatFeed(MockDataFeed):
            async def get_balance(self):
                return {"KRW": 5_000_000}  # no BTC held

        feed = FlatFeed(price=50_000_000)
        config = AppConfig(risk=RiskConfig())
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=feed,
            config=config,
            state_manager=StateManager(tmp_path / "state.json"),
        )
        engine.state.positions["BTC/KRW"] = Position(
            symbol="BTC/KRW",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=50_000_000,
            entry_time=datetime.now(UTC),
            stop_loss=49_000_000,
        )

        await engine._reconcile_with_exchange()
        assert "BTC/KRW" not in engine.state.positions

    @pytest.mark.asyncio
    async def test_shrinks_partially_backed_position(self, tmp_path):
        """If the exchange holds less than the tracked size, shrink to the real
        amount so an exit never tries to over-sell."""

        class PartialFeed(MockDataFeed):
            async def get_balance(self):
                return {"KRW": 5_000_000, "BTC": 0.04}

        feed = PartialFeed(price=50_000_000)
        config = AppConfig(risk=RiskConfig())
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=feed,
            config=config,
            state_manager=StateManager(tmp_path / "state.json"),
        )
        engine.state.positions["BTC/KRW"] = Position(
            symbol="BTC/KRW",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=50_000_000,
            entry_time=datetime.now(UTC),
            stop_loss=49_000_000,
        )

        await engine._reconcile_with_exchange()
        pos = engine.state.positions.get("BTC/KRW")
        assert pos is not None
        assert pos.size == pytest.approx(0.04)

    @pytest.mark.asyncio
    async def test_entry_exception_reconciles_orphan_fill(self, tmp_path):
        """If the buy raises after executing on the exchange (lost response),
        _handle_entry must reconcile so the resulting holding is adopted with a
        stop rather than propagating the error and leaving an unmanaged orphan."""

        class LostResponseFeed(MockDataFeed):
            def __init__(self, price):
                super().__init__(price)
                self._executed = False

            async def get_balance(self):
                # After the (lost) fill the exchange reports the BTC holding.
                if self._executed:
                    return {"KRW": 0, "BTC": 0.1}
                return {"KRW": 5_000_000}

            async def create_order(self, symbol, side, order_type, quantity, price=None):
                # The order executes on the exchange, then the response is lost.
                self._executed = True
                raise ConnectionError("response lost")

        feed = LostResponseFeed(price=50_000_000)
        config = AppConfig(
            risk=RiskConfig(
                default_stop_loss_pct=0.02,
                risk_per_trade_pct=0.01,
                max_position_size_pct=0.5,
            )
        )
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=feed,
            config=config,
            state_manager=StateManager(tmp_path / "state.json"),
        )
        signal = Signal(
            timestamp=datetime.now(UTC),
            symbol="BTC/KRW",
            signal_type=SignalType.LONG_ENTRY,
            price=50_000_000,
            strength=1.0,
        )

        # Must not raise — the lost-response order is reconciled internally.
        await engine._handle_entry(signal, "BTC/KRW", 50_000_000)

        pos = engine.state.positions.get("BTC/KRW")
        assert pos is not None  # orphan fill adopted
        assert pos.size == pytest.approx(0.1)
        assert pos.stop_loss is not None and pos.stop_loss < pos.entry_price

    @pytest.mark.asyncio
    async def test_entry_unconfirmed_fill_reconciles_orphan(self, tmp_path):
        """A buy that returns without a confirmed fill (e.g. a market order that
        timed out as PENDING because every status poll errored) must reconcile.
        The fill may have executed; left unreconciled it becomes an unmanaged
        orphan the engine believes it doesn't hold and could double up on."""

        class UnconfirmedFeed(MockDataFeed):
            def __init__(self, price):
                super().__init__(price)
                self._submitted = False

            async def get_balance(self):
                # After the (unconfirmed) fill the exchange reports the holding.
                if self._submitted:
                    return {"KRW": 0, "BTC": 0.1}
                return {"KRW": 5_000_000}

            async def create_order(self, symbol, side, order_type, quantity, price=None):
                # Accepted by the exchange but never confirmed FILLED to us.
                self._submitted = True
                return Order(
                    id="u1",
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    status=OrderStatus.PENDING,
                    created_at=datetime.now(UTC),
                )

        feed = UnconfirmedFeed(price=50_000_000)
        config = AppConfig(
            risk=RiskConfig(
                default_stop_loss_pct=0.02,
                risk_per_trade_pct=0.01,
                max_position_size_pct=0.5,
            )
        )
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=feed,
            config=config,
            state_manager=StateManager(tmp_path / "state.json"),
        )
        signal = Signal(
            timestamp=datetime.now(UTC),
            symbol="BTC/KRW",
            signal_type=SignalType.LONG_ENTRY,
            price=50_000_000,
            strength=1.0,
        )
        await engine._handle_entry(signal, "BTC/KRW", 50_000_000)

        pos = engine.state.positions.get("BTC/KRW")
        assert pos is not None  # unconfirmed fill adopted via reconcile
        assert pos.size == pytest.approx(0.1)
        assert pos.stop_loss is not None and pos.stop_loss < pos.entry_price

    @pytest.mark.asyncio
    async def test_exit_unconfirmed_reconciles_phantom(self, tmp_path):
        """A sell that returns without a confirmed fill must reconcile: if it
        actually executed, the now-phantom position is dropped so the engine
        doesn't keep trying to sell shares it no longer holds."""

        class UnconfirmedSellFeed(MockDataFeed):
            async def get_balance(self):
                return {"KRW": 5_000_000}  # BTC already gone (sold, unconfirmed)

            async def create_order(self, symbol, side, order_type, quantity, price=None):
                return Order(
                    id="s1",
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    status=OrderStatus.PENDING,
                    created_at=datetime.now(UTC),
                )

        feed = UnconfirmedSellFeed(price=50_000_000)
        config = AppConfig(risk=RiskConfig())
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=feed,
            config=config,
            state_manager=StateManager(tmp_path / "state.json"),
        )
        engine.state.positions["BTC/KRW"] = Position(
            symbol="BTC/KRW",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=50_000_000,
            entry_time=datetime.now(UTC),
            stop_loss=49_000_000,
        )
        sig = Signal(
            timestamp=datetime.now(UTC),
            symbol="BTC/KRW",
            signal_type=SignalType.LONG_EXIT,
            price=50_000_000,
            strength=1.0,
        )
        await engine._handle_exit(sig, "BTC/KRW", engine.state.positions["BTC/KRW"])

        assert "BTC/KRW" not in engine.state.positions  # phantom dropped


# --- HIGH: per-tick safety-rail enforcement (drawdown breaker + daily loss) ---


class TestSafetyRailEnforcement:
    @pytest.mark.asyncio
    async def test_circuit_breaker_flattens_open_position(self, tmp_path):
        """A drawdown breach must flatten open positions every tick — even with
        no candle and a price above the position's own stop. Previously the
        breaker was consulted only at entry-signal time and never closed an
        existing position, so a held position could bleed past the limit."""
        feed = MockDataFeed(price=49_500_000)  # above the 49M stop
        paper = PaperExchange(
            data_feed=feed,
            initial_balance=8_000_000,
            fee_rate=0.0005,
            slippage_pct=0.001,
        )
        paper.update_prices({"BTC/KRW": 49_500_000})
        paper._holdings["BTC"] = 0.001  # holdings to sell on flatten

        config = AppConfig(
            risk=RiskConfig(
                max_drawdown_pct=0.10,
                default_stop_loss_pct=0.02,
            )
        )
        state = StateManager(tmp_path / "state.json")
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=paper,
            config=config,
            state_manager=state,
        )
        # Peak well above current equity -> drawdown breaches the 10% limit.
        engine.risk_manager.peak_equity = 10_000_000.0
        state.positions["BTC/KRW"] = Position(
            symbol="BTC/KRW",
            side=PositionSide.LONG,
            size=0.001,
            entry_price=50_000_000,
            entry_time=datetime.now(UTC),
            stop_loss=49_000_000,  # below current price: no stop-out
        )

        await engine._monitor_prices(["BTC/KRW"])

        assert "BTC/KRW" not in state.positions

    def test_daily_loss_breached_folds_unrealized(self):
        """The daily-loss limit must account for open-position unrealized PnL,
        not just realized PnL booked on exit."""
        from tradingbot.risk.validators import TradeValidator

        v = TradeValidator(daily_loss_limit_krw=200_000)
        v.record_trade_pnl(-50_000)  # realized so far

        # Realized -50k alone is within the -200k limit.
        assert v.daily_loss_breached(0.0) is False
        # Realized -50k + unrealized -100k = -150k: still within limit.
        assert v.daily_loss_breached(-100_000) is False
        # Realized -50k + unrealized -160k = -210k: breaches the -200k limit.
        assert v.daily_loss_breached(-160_000) is True

    @pytest.mark.asyncio
    async def test_daily_loss_unrealized_flattens_position(self, tmp_path):
        """An open position whose unrealized loss breaches the daily-loss limit
        must be flattened per tick, even when the drawdown breaker is calm and
        the price is above the position's own stop."""
        from tradingbot.risk.validators import TradeValidator

        feed = MockDataFeed(price=45_000_000)  # above the 40M stop
        paper = PaperExchange(
            data_feed=feed,
            initial_balance=1_000_000,
            fee_rate=0.0005,
            slippage_pct=0.001,
        )
        paper.update_prices({"BTC/KRW": 45_000_000})
        paper._holdings["BTC"] = 0.01

        config = AppConfig(
            risk=RiskConfig(
                max_drawdown_pct=0.99,
                default_stop_loss_pct=0.02,
            )
        )
        state = StateManager(tmp_path / "state.json")
        validator = TradeValidator(daily_loss_limit_krw=40_000)
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=paper,
            config=config,
            state_manager=state,
            trade_validator=validator,
        )
        state.positions["BTC/KRW"] = Position(
            symbol="BTC/KRW",
            side=PositionSide.LONG,
            size=0.01,
            entry_price=50_000_000,
            entry_time=datetime.now(UTC),
            stop_loss=40_000_000,  # far below price: no stop-out
        )
        # Unrealized = (45M - 50M) * 0.01 = -50,000 < -40,000 limit.
        # The day started flat, so all of it is today's loss (the first reading
        # of the day is what the validator baselines against).
        validator.daily_loss_breached(0.0)

        await engine._monitor_prices(["BTC/KRW"])

        assert "BTC/KRW" not in state.positions

    @pytest.mark.asyncio
    async def test_missing_price_skips_rail_no_spurious_flatten(self, tmp_path):
        """A held position with no resolvable price this tick must NOT be
        flattened. _calculate_equity contributes 0 for a priceless holding, so
        equity collapses to ~cash; the now-continuous breaker would otherwise
        liquidate the whole book on a single missing tick. The rail must skip
        (unreliable equity) instead of firing."""

        class NoPriceFeed(MockDataFeed):
            async def fetch_ticker(self, symbol):
                raise RuntimeError("ticker unavailable")

            async def get_balance(self):
                return {"KRW": 1_000_000, "BTC": 0.001}

            async def create_order(self, symbol, side, order_type, quantity, price=None):
                # Records the erroneous sell if the buggy rail flattens.
                return Order(
                    id="spurious-flatten",
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    status=OrderStatus.FILLED,
                    filled_price=50_000_000,
                    fee=0.0,
                    filled_at=datetime.now(UTC),
                )

        feed = NoPriceFeed(price=50_000_000)
        config = AppConfig(risk=RiskConfig(max_drawdown_pct=0.10))
        state = StateManager(tmp_path / "state.json")
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=feed,
            config=config,
            state_manager=state,
        )
        # Peak far above the cash-only equity: without a BTC price, equity
        # collapses to ~cash and the drawdown breaker would breach.
        engine.risk_manager.peak_equity = 100_000_000.0
        state.positions["BTC/KRW"] = Position(
            symbol="BTC/KRW",
            side=PositionSide.LONG,
            size=0.001,
            entry_price=50_000_000,
            entry_time=datetime.now(UTC),
            stop_loss=49_000_000,
        )
        state.entry_fees["BTC/KRW"] = 0.0

        await engine._monitor_prices(["BTC/KRW"])

        # Rail skipped on unreliable equity -> position retained, not liquidated.
        assert "BTC/KRW" in state.positions


# --- Bug: external transfers must not move the drawdown breaker ---


class TestTransferImmuneBreaker:
    """The breaker runs on the bot's ledger (baseline + realized + unrealized),
    not raw account equity: a withdrawal would otherwise read as a phantom
    drawdown and force-flatten every position; a deposit would mask a real
    drawdown and keep trading through a breached limit."""

    @pytest.mark.asyncio
    async def test_withdrawal_does_not_trip_breaker(self, tmp_path):
        """A KRW withdrawal drops raw equity ~30% with zero trading loss —
        the breaker must not fire and the position must stay open."""
        feed = MockDataFeed(price=50_000_000)
        paper = PaperExchange(
            data_feed=feed,
            initial_balance=10_000_000,
            fee_rate=0.0005,
            slippage_pct=0.001,
        )
        paper.update_prices({"BTC/KRW": 50_000_000})
        paper._holdings["BTC"] = 0.001

        config = AppConfig(
            risk=RiskConfig(
                max_drawdown_pct=0.10,
                default_stop_loss_pct=0.02,
            )
        )
        state = StateManager(tmp_path / "state.json")
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=paper,
            config=config,
            state_manager=state,
        )
        state.positions["BTC/KRW"] = Position(
            symbol="BTC/KRW",
            side=PositionSide.LONG,
            size=0.001,
            entry_price=50_000_000,
            entry_time=datetime.now(UTC),
            stop_loss=45_000_000,  # below price: no stop-out
        )

        # First tick latches the ledger baseline and peak (~10.05M).
        await engine._monitor_prices(["BTC/KRW"])
        assert "BTC/KRW" in state.positions

        # External withdrawal: raw equity -3M (~30% > 10% limit), price flat.
        paper._cash -= 3_000_000
        await engine._monitor_prices(["BTC/KRW"])

        assert "BTC/KRW" in state.positions  # no phantom-drawdown flatten

    @pytest.mark.asyncio
    async def test_deposit_does_not_mask_real_drawdown(self, tmp_path):
        """A big deposit lands on the same tick as a 15% trading loss. Raw
        equity ends ABOVE its old peak, but the ledger is down 15% — the
        breaker must still fire and flatten."""
        feed = MockDataFeed(price=50_000_000)
        paper = PaperExchange(
            data_feed=feed,
            initial_balance=5_000_000,
            fee_rate=0.0005,
            slippage_pct=0.001,
        )
        paper.update_prices({"BTC/KRW": 50_000_000})
        paper._holdings["BTC"] = 0.1

        config = AppConfig(
            risk=RiskConfig(
                max_drawdown_pct=0.10,
                default_stop_loss_pct=0.02,
            )
        )
        state = StateManager(tmp_path / "state.json")
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=paper,
            config=config,
            state_manager=state,
        )
        state.positions["BTC/KRW"] = Position(
            symbol="BTC/KRW",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=50_000_000,
            entry_time=datetime.now(UTC),
            stop_loss=30_000_000,  # far below: the rail, not the stop, must act
        )

        # Baseline/peak latch at 5M cash + 5M position = 10M.
        await engine._monitor_prices(["BTC/KRW"])

        # Price crashes 30% (unrealized -1.5M = 15% of the 10M ledger) while
        # a 5M deposit lands: raw equity 5+5+3.5 = 13.5M — above the old peak,
        # which previously reset the peak and masked the drawdown.
        feed._price = 35_000_000
        paper.update_prices({"BTC/KRW": 35_000_000})
        paper._cash += 5_000_000
        await engine._monitor_prices(["BTC/KRW"])

        assert "BTC/KRW" not in state.positions  # ledger dd 15% >= 10% limit

    @pytest.mark.asyncio
    async def test_realized_pnl_books_into_ledger_and_persists(self, tmp_path):
        """Closed-trade PnL must accumulate into the ledger (same figure the
        daily-loss validator receives) and survive a state save/load."""
        from tradingbot.risk.validators import TradeValidator

        feed = MockDataFeed(price=55_000_000)
        paper = PaperExchange(
            data_feed=feed,
            initial_balance=1_000_000,
            fee_rate=0.0005,
            slippage_pct=0.001,
        )
        paper.update_prices({"BTC/KRW": 55_000_000})
        paper._holdings["BTC"] = 0.001

        config = AppConfig(
            risk=RiskConfig(
                max_drawdown_pct=0.99,
                default_stop_loss_pct=0.02,
            )
        )
        state = StateManager(tmp_path / "state.json")
        validator = TradeValidator(daily_loss_limit_krw=10_000_000)
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=paper,
            config=config,
            state_manager=state,
            trade_validator=validator,
        )
        state.positions["BTC/KRW"] = Position(
            symbol="BTC/KRW",
            side=PositionSide.LONG,
            size=0.001,
            entry_price=50_000_000,
            entry_time=datetime.now(UTC),
            stop_loss=45_000_000,
        )
        sig = Signal(
            timestamp=datetime.now(UTC),
            symbol="BTC/KRW",
            signal_type=SignalType.LONG_EXIT,
            price=55_000_000,
            strength=1.0,
        )
        await engine._handle_exit(sig, "BTC/KRW", state.positions["BTC/KRW"])

        assert "BTC/KRW" not in state.positions
        daily_pnl = validator.daily_state()[0]
        assert state.cum_realized_pnl == pytest.approx(daily_pnl)
        assert state.cum_realized_pnl > 0  # profitable close actually booked

        state.ledger_baseline = 10_000_000.0
        state.save()
        reloaded = StateManager(tmp_path / "state.json")
        reloaded.load()
        assert reloaded.ledger_baseline == 10_000_000.0
        assert reloaded.cum_realized_pnl == pytest.approx(state.cum_realized_pnl)


# --- Bug: a partial exit fill must not orphan the unsold remainder ---


class TestExitPartialFill:
    @pytest.mark.asyncio
    async def test_partial_exit_keeps_residual_position(self, tmp_path):
        """A market sell that fills less than the position must NOT delete the
        whole position. The unsold remainder would otherwise become an
        unmanaged orphan with no stop loss while the engine believes it is flat.
        """

        class PartialSellFeed(MockDataFeed):
            SOLD = 0.6  # of the requested 1.0

            async def create_order(self, symbol, side, order_type, quantity, price=None):
                return Order(
                    id="sell-partial",
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=self.SOLD,  # < requested
                    status=OrderStatus.FILLED,
                    filled_price=self._price,
                    fee=1_000.0,
                    filled_at=datetime.now(UTC),
                )

        feed = PartialSellFeed(price=50_000_000)
        config = AppConfig(risk=RiskConfig(default_stop_loss_pct=0.02))
        state = StateManager(tmp_path / "state.json")
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=feed,
            config=config,
            state_manager=state,
        )
        state.positions["BTC/KRW"] = Position(
            symbol="BTC/KRW",
            side=PositionSide.LONG,
            size=1.0,
            entry_price=50_000_000,
            entry_time=datetime.now(UTC),
            stop_loss=49_000_000,
        )
        state.entry_fees["BTC/KRW"] = 1_000.0

        sig = Signal(
            timestamp=datetime.now(UTC),
            symbol="BTC/KRW",
            signal_type=SignalType.LONG_EXIT,
            price=50_000_000,
        )
        await engine._handle_exit(sig, "BTC/KRW", state.positions["BTC/KRW"])

        # Position retained (still managed, keeps its stop), size reduced to the
        # unsold remainder rather than deleted.
        assert "BTC/KRW" in state.positions
        assert state.positions["BTC/KRW"].size == pytest.approx(1.0 - PartialSellFeed.SOLD)
        assert state.positions["BTC/KRW"].stop_loss == 49_000_000


# --- Bug: one symbol's tick error must not abort the rest or skip persist ---


class TestTickAllPerSymbolIsolation:
    @pytest.mark.asyncio
    async def test_one_symbol_error_does_not_abort_others_or_skip_persist(self, tmp_path):
        """A single symbol raising in _tick_symbol must not abort the rest of
        the batch nor skip _persist_state. Otherwise one bad symbol silently
        halts processing for every other symbol and drops the tick's state
        snapshot (equity history + any state mutated this tick).
        """

        class FlakyStrategy(Strategy):
            def __init__(self):
                self.entry_calls: list[str] = []

            @property
            def symbols(self):
                return ["AAA/KRW", "BBB/KRW"]

            @property
            def timeframe(self):
                return "1h"

            def indicators(self, df):
                return df

            def should_entry(self, df, symbol):
                self.entry_calls.append(symbol)
                if symbol == "AAA/KRW":
                    raise RuntimeError("boom")
                return None

            def should_exit(self, df, symbol, position=None):
                return None

        feed = MockDataFeed(price=50_000_000)
        config = AppConfig(risk=RiskConfig())
        state = StateManager(tmp_path / "state.json")
        strat = FlakyStrategy()
        engine = LiveEngine(
            strategy=strat,
            exchange=feed,
            config=config,
            state_manager=state,
        )

        # AAA is first and raises; on old code this propagated out of _tick_all.
        await engine._tick_all(["AAA/KRW", "BBB/KRW"], "1h")

        # BBB was still processed despite AAA's error...
        assert "BBB/KRW" in strat.entry_calls
        # ...and the post-loop state snapshot was still written.
        assert (tmp_path / "state.json").exists()


# --- Bug: a position held across restart must get its exit re-evaluated ---


class TestRestartExitReeval:
    @pytest.mark.asyncio
    async def test_held_position_exit_reevaluated_on_restart(self, tmp_path):
        """Warmup marks the last closed candle as already seen. A position held
        across restart whose strategy wants OUT on that candle would otherwise
        be stranded until the next candle closes (up to a full timeframe on a 4h
        chart) — the first tick must re-evaluate the exit once.
        """

        class ExitingStrategy(Strategy):
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
                return Signal(
                    timestamp=datetime.now(UTC),
                    symbol=symbol,
                    signal_type=SignalType.LONG_EXIT,
                    price=50_000_000,
                )

        class FullSellFeed(MockDataFeed):
            async def create_order(self, symbol, side, order_type, quantity, price=None):
                return Order(
                    id="sell-full",
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    status=OrderStatus.FILLED,
                    filled_price=self._price,
                    fee=1_000.0,
                    filled_at=datetime.now(UTC),
                )

        feed = FullSellFeed(price=50_000_000)
        config = AppConfig(risk=RiskConfig(default_stop_loss_pct=0.02))
        state = StateManager(tmp_path / "state.json")
        engine = LiveEngine(
            strategy=ExitingStrategy(),
            exchange=feed,
            config=config,
            state_manager=state,
        )
        state.positions["BTC/KRW"] = Position(
            symbol="BTC/KRW",
            side=PositionSide.LONG,
            size=0.001,
            entry_price=50_000_000,
            entry_time=datetime.now(UTC),
            stop_loss=40_000_000,  # far below price → no stop-out
        )

        # Mark the last CLOSED candle as already processed, exactly as warmup
        # does — so this is NOT a new candle.
        df = await feed.fetch_ohlcv("BTC/KRW", "1h", limit=10)
        engine._last_candle_ts["BTC/KRW"] = df.index[-2].to_pydatetime()
        # The restart flag is the fix: warmup sets it for symbols held at start.
        if hasattr(engine, "_pending_restart_exit"):
            engine._pending_restart_exit.add("BTC/KRW")

        await engine._tick_symbol("BTC/KRW", df, {"last": 50_000_000})

        # Exit re-evaluated → position closed despite the candle being "seen".
        assert "BTC/KRW" not in state.positions


# --- Sizing coherence: live gate must mark all positions to market ---


class MultiPriceFeed(BaseExchange):
    """Mock feed returning a distinct ticker price per symbol."""

    def __init__(self, prices: dict[str, float]):
        self._prices = dict(prices)

    async def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=100):
        p = self._prices.get(symbol, 1_000_000)
        dates = pd.date_range("2024-01-01", periods=limit, freq="h", tz="UTC")
        return pd.DataFrame(
            {
                "open": [p] * limit,
                "high": [p * 1.01] * limit,
                "low": [p * 0.99] * limit,
                "close": [p] * limit,
                "volume": [100] * limit,
            },
            index=dates,
        )

    async def fetch_ticker(self, symbol):
        p = self._prices.get(symbol, 1_000_000)
        return {
            "last": p,
            "bid": p * 0.999,
            "ask": p * 1.001,
            "volume": 100,
            "timestamp": datetime.now(UTC),
        }

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


class TestGateMarksAllPositions:
    @pytest.mark.asyncio
    async def test_circuit_breaker_uses_mark_to_market_equity(self, tmp_path):
        """The entry risk gate must value OTHER open positions at their current
        price, not entry cost. A held position deep underwater must trip the
        drawdown circuit breaker; cost-basis equity would hide the loss and let
        a fresh entry through.
        """
        # ETH bought 2.0 @ 4M (=8M) + 1M cash → peak 9M. ETH crashes to 2M →
        # mark-to-market equity 5M = 44% drawdown (> 20% limit). Cost basis
        # would still read 9M and miss it entirely.
        feed = MultiPriceFeed({"BTC/KRW": 50_000_000, "ETH/KRW": 2_000_000})
        paper = PaperExchange(
            data_feed=feed,
            initial_balance=1_000_000,
            fee_rate=0.0005,
            slippage_pct=0.001,
        )
        paper._holdings["ETH"] = 2.0  # existing holding, now underwater

        config = AppConfig(
            risk=RiskConfig(
                max_position_size_pct=0.5,
                max_open_positions=5,
                max_drawdown_pct=0.20,
                default_stop_loss_pct=0.02,
                risk_per_trade_pct=0.01,
            )
        )
        state = StateManager(tmp_path / "state.json")
        state.positions["ETH/KRW"] = Position(
            symbol="ETH/KRW",
            side=PositionSide.LONG,
            size=2.0,
            entry_price=4_000_000,
            entry_time=datetime.now(UTC),
            stop_loss=3_920_000,
        )
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=paper,
            config=config,
            state_manager=state,
        )
        engine.risk_manager.peak_equity = 9_000_000

        signal = Signal(
            timestamp=datetime.now(UTC),
            symbol="BTC/KRW",
            signal_type=SignalType.LONG_ENTRY,
            price=50_000_000,
            strength=1.0,
        )
        await engine._handle_entry(signal, "BTC/KRW", 50_000_000)

        # Breaker fires on the mark-to-market drawdown → no BTC entry.
        assert "BTC/KRW" not in state.positions


class TestLiveCashClamp:
    @pytest.mark.asyncio
    async def test_entry_quantity_clamped_to_free_cash(self, tmp_path):
        """Sizing runs on total mark-to-market equity, but only KRW cash is
        spendable. The live path must clamp the requested quantity to free cash
        before submitting — not rely on the exchange to truncate/reject.
        """
        # 8M in ETH + 500k cash → equity 8.5M. Sizing wants ~max_position_size
        # notional (huge vs 500k cash); the order must be clamped to what 500k
        # can buy (fee-inclusive).
        feed = MultiPriceFeed({"BTC/KRW": 50_000_000, "ETH/KRW": 4_000_000})
        paper = PaperExchange(
            data_feed=feed,
            initial_balance=500_000,
            fee_rate=0.0005,
            slippage_pct=0.001,
        )
        paper._holdings["ETH"] = 2.0  # 8M at 4M

        config = AppConfig(
            risk=RiskConfig(
                max_position_size_pct=1.0,
                max_open_positions=5,
                max_drawdown_pct=0.99,
                default_stop_loss_pct=0.02,
                risk_per_trade_pct=0.02,
            )
        )
        state = StateManager(tmp_path / "state.json")
        state.positions["ETH/KRW"] = Position(
            symbol="ETH/KRW",
            side=PositionSide.LONG,
            size=2.0,
            entry_price=4_000_000,
            entry_time=datetime.now(UTC),
            stop_loss=3_920_000,
        )
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=paper,
            config=config,
            state_manager=state,
        )
        engine.risk_manager.peak_equity = 8_500_000  # no breaker interference

        captured: dict[str, float] = {}
        orig_create = paper.create_order

        async def spy_create(symbol, side, order_type, quantity, price=None):
            captured["quantity"] = quantity
            return await orig_create(symbol, side, order_type, quantity, price)

        paper.create_order = spy_create

        signal = Signal(
            timestamp=datetime.now(UTC),
            symbol="BTC/KRW",
            signal_type=SignalType.LONG_ENTRY,
            price=50_000_000,
            strength=1.0,
        )
        await engine._handle_entry(signal, "BTC/KRW", 50_000_000)

        expected_price = 50_000_000 * 1.001
        affordable = 500_000 / (expected_price * (1 + 0.0005))
        assert "quantity" in captured
        # Requested quantity clamped to free cash, not the ~8.5M-equity size.
        assert captured["quantity"] == pytest.approx(affordable, rel=1e-6)


# --- Sizing coherence: equity scoped to the managed universe ---


class TestEquityScope:
    @pytest.mark.asyncio
    async def test_untraded_holding_excluded_and_not_fetched(self, tmp_path):
        """An untraded balance (outside strategy.symbols and open positions) must
        not be priced into equity nor cost a per-tick fetch_ticker. Otherwise a
        user's unrelated coin bag both inflates the risk/sizing budget and
        hammers the ticker endpoint every tick (rate-limit). This mirrors the
        backtest engine, which only values managed positions + cash.
        """
        fetched: list[str] = []

        class ExtraBagFeed(MockDataFeed):
            async def get_balance(self):
                return {"KRW": 1_000_000, "DOGE": 5_000.0}

            async def fetch_ticker(self, symbol):
                fetched.append(symbol)
                return await super().fetch_ticker(symbol)

        feed = ExtraBagFeed(price=50_000_000)
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=feed,  # trades BTC/KRW only
            config=AppConfig(risk=RiskConfig()),
            state_manager=StateManager(tmp_path / "state.json"),
        )

        equity = await engine._calculate_equity()

        # DOGE is neither traded nor an open position → excluded, never fetched.
        assert equity == pytest.approx(1_000_000)
        assert "DOGE/KRW" not in fetched

    @pytest.mark.asyncio
    async def test_open_position_priced_even_if_not_in_symbols(self, tmp_path):
        """A held position outside the configured symbol list is still ours to
        manage, so it must be valued — the scope is the union of strategy.symbols
        and open positions, not strategy.symbols alone.
        """

        class EthBagFeed(MultiPriceFeed):
            async def get_balance(self):
                return {"KRW": 1_000_000, "ETH": 2.0}

        feed = EthBagFeed({"ETH/KRW": 3_000_000})
        state = StateManager(tmp_path / "state.json")
        state.positions["ETH/KRW"] = Position(
            symbol="ETH/KRW",
            side=PositionSide.LONG,
            size=2.0,
            entry_price=3_000_000,
            entry_time=datetime.now(UTC),
            stop_loss=2_940_000,
        )
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=feed,  # trades BTC/KRW only
            config=AppConfig(risk=RiskConfig()),
            state_manager=state,
        )

        equity = await engine._calculate_equity()

        # 1M cash + 2.0 ETH * 3M = 7M — the open ETH position is priced despite
        # not appearing in strategy.symbols.
        assert equity == pytest.approx(1_000_000 + 2.0 * 3_000_000)


# --- Feature: operator kill-switch (dashboard control file) ---


class EnterAlwaysStrategy(StubStrategy):
    """Strategy that always wants in — exercises the entry pause gate."""

    def should_entry(self, df, symbol):
        return Signal(
            timestamp=datetime.now(UTC),
            symbol=symbol,
            signal_type=SignalType.LONG_ENTRY,
            price=float(df["close"].iloc[-1]),
            strength=1.0,
        )


class TestEntryPauseControl:
    """control.json pauses NEW entries only; the engine keeps managing
    existing positions (stops/TP/exits/rails) while paused."""

    def test_pause_flag_roundtrip(self, tmp_path):
        from tradingbot.live.control import control_path_for, read_pause, set_pause

        control = control_path_for(tmp_path / "state.json")
        assert read_pause(control) is False  # missing file == not paused
        set_pause(control, True)
        assert read_pause(control) is True
        set_pause(control, False)
        assert read_pause(control) is False
        control.write_text("{not json")
        assert read_pause(control) is False  # corrupt == fail-open (logged)

    @pytest.mark.asyncio
    async def test_paused_engine_skips_entry_resumed_enters(self, tmp_path):
        from tradingbot.live.control import control_path_for, set_pause

        def make_engine():
            feed = MockDataFeed(price=50_000_000)
            paper = PaperExchange(
                data_feed=feed,
                initial_balance=10_000_000,
                fee_rate=0.0005,
                slippage_pct=0.001,
            )
            paper.update_prices({"BTC/KRW": 50_000_000})
            config = AppConfig(
                risk=RiskConfig(
                    max_drawdown_pct=0.99,
                    default_stop_loss_pct=0.02,
                )
            )
            state = StateManager(tmp_path / "state.json")
            return LiveEngine(
                strategy=EnterAlwaysStrategy(),
                exchange=paper,
                config=config,
                state_manager=state,
            )

        control = control_path_for(tmp_path / "state.json")

        set_pause(control, True)
        engine = make_engine()
        await engine._tick_all(["BTC/KRW"], "1h")
        assert engine._entries_paused is True
        assert "BTC/KRW" not in engine.state.positions  # entry gated

        set_pause(control, False)
        engine2 = make_engine()  # fresh candle tracking → entry re-evaluates
        await engine2._tick_all(["BTC/KRW"], "1h")
        assert engine2._entries_paused is False
        assert "BTC/KRW" in engine2.state.positions  # same setup enters when live


# --- Signal-triggered pyramiding (live path) ---


class TestLivePyramiding:
    def _engine(self, tmp_path, enabled: bool):
        feed = MockDataFeed(price=50_000_000)
        paper = PaperExchange(
            data_feed=feed,
            initial_balance=10_000_000,
            fee_rate=0.0005,
            slippage_pct=0.001,
        )
        paper.update_prices({"BTC/KRW": 50_000_000})
        config = AppConfig(
            risk=RiskConfig(
                max_drawdown_pct=0.99,
                default_stop_loss_pct=0.02,
                # The cap bounds the whole position, not one tranche: risk
                # sizing yields ≈5% of equity, leaving room under the 10% cap.
                risk_per_trade_pct=0.001,
                max_position_size_pct=0.1,
            ),
            pyramiding=PyramidingConfig(enabled=enabled),
        )
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=paper,
            config=config,
            state_manager=StateManager(tmp_path / "state.json"),
        )
        return engine, feed

    @staticmethod
    def _entry_signal(price: float) -> Signal:
        return Signal(
            timestamp=datetime.now(UTC),
            symbol="BTC/KRW",
            signal_type=SignalType.LONG_ENTRY,
            price=price,
            strength=1.0,
        )

    @pytest.mark.asyncio
    async def test_second_entry_merges_into_open_position(self, tmp_path):
        engine, feed = self._engine(tmp_path, enabled=True)

        await engine._handle_entry(self._entry_signal(50_000_000), "BTC/KRW", 50_000_000)
        first = replace(engine.state.positions["BTC/KRW"])
        first_fee = engine.state.entry_fees["BTC/KRW"]

        feed._price = 60_000_000  # the add fills higher, so the average moves
        await engine._handle_entry(self._entry_signal(60_000_000), "BTC/KRW", 60_000_000)

        position = engine.state.positions["BTC/KRW"]
        assert position.adds == 1
        assert position.size > first.size
        # Blended average sits between the two fills, and the stop follows it
        assert first.entry_price < position.entry_price < 60_000_000 * 1.001
        assert position.stop_loss == pytest.approx(position.entry_price * 0.98)
        # entry_time pins the first entry — trailing exits anchor on it
        assert position.entry_time == first.entry_time
        # Entry fees accumulate, else exit PnL would drop the first tranche's
        assert engine.state.entry_fees["BTC/KRW"] > first_fee

    @pytest.mark.asyncio
    async def test_entry_gate_respects_config(self, tmp_path):
        engine, _feed = self._engine(tmp_path, enabled=False)
        assert await engine._entry_allowed("BTC/KRW") is True  # flat

        await engine._handle_entry(self._entry_signal(50_000_000), "BTC/KRW", 50_000_000)
        assert await engine._entry_allowed("BTC/KRW") is False  # held, pyramiding off

        engine.config = engine.config.model_copy(
            update={"pyramiding": PyramidingConfig(enabled=True)}
        )
        assert await engine._entry_allowed("BTC/KRW") is True  # held, cash available

        engine.config.pyramiding.min_add_cash_pct = 0.99  # more idle cash than we hold
        assert await engine._entry_allowed("BTC/KRW") is False


# --- Bug: fills reported on a CANCELLED order were discarded ---


class TestOrderHasFill:
    """Upbit market buys (ord_type='price') settle as CANCELLED once the unused
    KRW is returned, and a partial fill followed by a cancel looks identical.
    Gating on FILLED alone loses those real fills, leaving the engine believing
    it is flat while holding coin."""

    def _order(self, status, quantity, filled_price):
        return Order(
            id="o1",
            symbol="BTC/KRW",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            status=status,
            filled_price=filled_price,
        )

    def test_filled_status_counts(self):
        assert _order_has_fill(self._order(OrderStatus.FILLED, 1.0, 50_000_000))

    def test_cancelled_with_execution_counts(self):
        assert _order_has_fill(self._order(OrderStatus.CANCELLED, 0.4, 50_000_000))

    def test_cancelled_without_fill_price_does_not_count(self):
        assert not _order_has_fill(self._order(OrderStatus.CANCELLED, 0.4, None))

    def test_freshly_submitted_order_does_not_count(self):
        # Carries the requested quantity but no fill price yet.
        assert not _order_has_fill(self._order(OrderStatus.PENDING, 1.0, None))


class TestCancelledPartialExit:
    @pytest.mark.asyncio
    async def test_cancelled_partial_sell_books_pnl_and_shrinks(self, tmp_path):
        """A sell cancelled after a partial fill must settle the sold portion
        and keep the remainder as a managed position — not fall through to the
        unconfirmed branch, which books nothing."""

        class CancelledPartialSellFeed(MockDataFeed):
            SOLD = 0.6  # of the requested 1.0

            async def create_order(self, symbol, side, order_type, quantity, price=None):
                return Order(
                    id="sell-cancelled",
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=self.SOLD,
                    status=OrderStatus.CANCELLED,  # real fill, cancelled remainder
                    filled_price=55_000_000,
                    fee=1_000.0,
                    filled_at=datetime.now(UTC),
                )

            async def get_balance(self):
                return {"KRW": 5_000_000, "BTC": 0.4}

        feed = CancelledPartialSellFeed(price=50_000_000)
        state = StateManager(tmp_path / "state.json")
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=feed,
            config=AppConfig(risk=RiskConfig(default_stop_loss_pct=0.02)),
            state_manager=state,
        )
        state.positions["BTC/KRW"] = Position(
            symbol="BTC/KRW",
            side=PositionSide.LONG,
            size=1.0,
            entry_price=50_000_000,
            entry_time=datetime.now(UTC),
            stop_loss=49_000_000,
        )
        state.entry_fees["BTC/KRW"] = 1_000.0

        sig = Signal(
            timestamp=datetime.now(UTC),
            symbol="BTC/KRW",
            signal_type=SignalType.LONG_EXIT,
            price=55_000_000,
        )
        await engine._handle_exit(sig, "BTC/KRW", state.positions["BTC/KRW"])

        assert state.positions["BTC/KRW"].size == pytest.approx(0.4)
        # 5,000,000 x 0.6 gain, less the pro-rata entry fee and the exit fee
        assert state.cum_realized_pnl == pytest.approx(3_000_000 - 600 - 1_000)


class TestExitFillPriceFallback:
    @pytest.mark.asyncio
    async def test_missing_fill_price_falls_back_to_signal_price(self, tmp_path):
        """An exchange response without a fill price must not book the exit at
        0 — that reads as a total loss and can trip the breaker and the daily
        loss limit on a trade that was actually flat."""

        class NoFillPriceFeed(MockDataFeed):
            async def create_order(self, symbol, side, order_type, quantity, price=None):
                return Order(
                    id="sell-nofillprice",
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    status=OrderStatus.FILLED,
                    filled_price=None,
                    filled_at=datetime.now(UTC),
                )

        state = StateManager(tmp_path / "state.json")
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=NoFillPriceFeed(price=50_000_000),
            config=AppConfig(risk=RiskConfig()),
            state_manager=state,
        )
        state.positions["BTC/KRW"] = Position(
            symbol="BTC/KRW",
            side=PositionSide.LONG,
            size=1.0,
            entry_price=50_000_000,
            entry_time=datetime.now(UTC),
        )

        sig = Signal(
            timestamp=datetime.now(UTC),
            symbol="BTC/KRW",
            signal_type=SignalType.LONG_EXIT,
            price=50_000_000,
        )
        await engine._handle_exit(sig, "BTC/KRW", state.positions["BTC/KRW"])

        assert state.cum_realized_pnl == pytest.approx(0.0)


# --- Bug: reconciliation adjusted sizes without booking the realized PnL ---


class TestReconcileBooksRealizedPnl:
    def _engine(self, feed, tmp_path):
        from tradingbot.risk.validators import TradeValidator

        state = StateManager(tmp_path / "state.json")
        validator = TradeValidator(daily_loss_limit_krw=10_000_000)
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=feed,
            config=AppConfig(risk=RiskConfig()),
            state_manager=state,
            trade_validator=validator,
        )
        state.positions["BTC/KRW"] = Position(
            symbol="BTC/KRW",
            side=PositionSide.LONG,
            size=1.0,
            entry_price=50_000_000,
            entry_time=datetime.now(UTC),
        )
        state.entry_fees["BTC/KRW"] = 1_000.0
        return engine, state, validator

    @pytest.mark.asyncio
    async def test_phantom_drop_books_estimated_pnl(self, tmp_path):
        """A position sold out-of-band still moved real money. Dropping it
        without booking the PnL leaves the ledger the drawdown breaker reads
        permanently wrong."""

        class FlatFeed(MockDataFeed):
            async def get_balance(self):
                return {"KRW": 5_000_000}

        engine, state, validator = self._engine(FlatFeed(price=55_000_000), tmp_path)
        await engine._reconcile_with_exchange()

        assert "BTC/KRW" not in state.positions
        expected = 5_000_000 - 1_000  # mark-to-market gain less the entry fee
        assert state.cum_realized_pnl == pytest.approx(expected)
        assert validator.daily_state()[0] == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_shrink_books_pro_rata_pnl_and_fee(self, tmp_path):
        class PartialFeed(MockDataFeed):
            async def get_balance(self):
                return {"KRW": 5_000_000, "BTC": 0.4}

        engine, state, validator = self._engine(PartialFeed(price=55_000_000), tmp_path)
        await engine._reconcile_with_exchange()

        assert state.positions["BTC/KRW"].size == pytest.approx(0.4)
        expected = 5_000_000 * 0.6 - 600  # 0.6 sold, entry fee charged pro rata
        assert state.cum_realized_pnl == pytest.approx(expected)
        assert validator.daily_state()[0] == pytest.approx(expected)
        # The unsold remainder keeps its share of the entry fee for its own exit
        assert state.entry_fees["BTC/KRW"] == pytest.approx(400)

    @pytest.mark.asyncio
    async def test_unreachable_price_books_fee_only(self, tmp_path):
        """Without a mark price we can't estimate the move, but the entry fee
        was certainly paid — book that rather than nothing."""

        class NoTickerFeed(MockDataFeed):
            async def get_balance(self):
                return {"KRW": 5_000_000}

            async def fetch_ticker(self, symbol):
                raise RuntimeError("ticker unavailable")

        engine, state, _validator = self._engine(NoTickerFeed(price=55_000_000), tmp_path)
        await engine._reconcile_with_exchange()

        assert "BTC/KRW" not in state.positions
        assert state.cum_realized_pnl == pytest.approx(-1_000)


# --- Bug: a stopped-out candle must not re-enter (backtest already blocks it) ---


class TestStopCandleBlocksEntry:
    @pytest.mark.asyncio
    async def test_stop_out_candle_skips_entry(self, tmp_path):
        """The candle that fired a stop must not open a new position on the
        same tick. The backtest blocks this (stop_loss_fired_symbols), so live
        allowing it would trade a setup the walk-forward never vetted."""
        feed = MockDataFeed(price=50_000_000)
        paper = PaperExchange(
            data_feed=feed,
            initial_balance=10_000_000,
            fee_rate=0.0005,
            slippage_pct=0.001,
        )
        paper.update_prices({"BTC/KRW": 50_000_000})
        paper._holdings["BTC"] = 0.01

        state = StateManager(tmp_path / "state.json")
        engine = LiveEngine(
            strategy=EnterAlwaysStrategy(),
            exchange=paper,
            config=AppConfig(risk=RiskConfig(max_drawdown_pct=0.99)),
            state_manager=state,
        )
        state.positions["BTC/KRW"] = Position(
            symbol="BTC/KRW",
            side=PositionSide.LONG,
            size=0.01,
            entry_price=52_000_000,
            entry_time=datetime.now(UTC),
            stop_loss=51_000_000,  # price 50M is below the stop → it fires
        )

        await engine._tick_all(["BTC/KRW"], "1h")

        # Stopped out, and the always-entering strategy did NOT get back in.
        assert "BTC/KRW" not in state.positions


# --- Bug: balance locked in an open order read as "not held" ---


class TestReconcileLockedBalance:
    @pytest.mark.asyncio
    async def test_locked_holding_is_not_a_phantom(self, tmp_path):
        """Quantity reserved by a resting order is absent from the free
        balance. Reconciling on free alone drops the position as a phantom,
        after which the engine believes it is flat and can buy the same coin
        again — so reconciliation must read the total balance."""

        class LockedFeed(MockDataFeed):
            async def get_balance(self):
                return {"KRW": 5_000_000}  # all BTC locked in an open order

            async def get_total_balance(self):
                return {"KRW": 5_000_000, "BTC": 0.1}

        state = StateManager(tmp_path / "state.json")
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=LockedFeed(price=50_000_000),
            config=AppConfig(risk=RiskConfig()),
            state_manager=state,
        )
        state.positions["BTC/KRW"] = Position(
            symbol="BTC/KRW",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=50_000_000,
            entry_time=datetime.now(UTC),
            stop_loss=49_000_000,
        )

        await engine._reconcile_with_exchange()

        pos = state.positions.get("BTC/KRW")
        assert pos is not None and pos.size == pytest.approx(0.1)
        assert state.cum_realized_pnl == 0.0  # nothing booked as a phantom exit


# --- Bug: orders below the exchange minimum loop through reject + reconcile ---


class TestMinOrderValue:
    def _engine(self, tmp_path, max_position_size_pct: float):
        feed = MockDataFeed(price=50_000_000)
        paper = PaperExchange(
            data_feed=feed,
            initial_balance=100_000,
            fee_rate=0.0005,
            slippage_pct=0.001,
        )
        paper.update_prices({"BTC/KRW": 50_000_000})
        config = AppConfig(
            risk=RiskConfig(
                max_drawdown_pct=0.99,
                default_stop_loss_pct=0.02,
                risk_per_trade_pct=0.5,  # so the position cap is what binds
                max_position_size_pct=max_position_size_pct,
            )
        )
        return LiveEngine(
            strategy=StubStrategy(),
            exchange=paper,
            config=config,
            state_manager=StateManager(tmp_path / "state.json"),
        )

    @staticmethod
    def _signal() -> Signal:
        return Signal(
            timestamp=datetime.now(UTC),
            symbol="BTC/KRW",
            signal_type=SignalType.LONG_ENTRY,
            price=50_000_000,
            strength=1.0,
        )

    @pytest.mark.asyncio
    async def test_order_below_minimum_is_skipped(self, tmp_path):
        # 5% of 100,000 = 5,000 KRW, under Upbit's minimum once fees are on top.
        engine = self._engine(tmp_path, max_position_size_pct=0.05)
        await engine._handle_entry(self._signal(), "BTC/KRW", 50_000_000)
        assert engine.state.positions == {}

    @pytest.mark.asyncio
    async def test_order_above_minimum_still_placed(self, tmp_path):
        engine = self._engine(tmp_path, max_position_size_pct=0.2)  # 20,000 KRW
        await engine._handle_entry(self._signal(), "BTC/KRW", 50_000_000)
        assert "BTC/KRW" in engine.state.positions


# --- Bug: a failing exit retried on every 10s monitor tick ---


class FailingExitFeed(MockDataFeed):
    """Sells always raise; the holding stays on the exchange."""

    def __init__(self, price: float):
        super().__init__(price)
        self.attempts = 0

    async def get_balance(self):
        return {"KRW": 1_000_000, "BTC": 0.01}

    async def create_order(self, symbol, side, order_type, quantity, price=None):
        self.attempts += 1
        raise RuntimeError("exchange rejected the sell")


class TestExitRetryBackoff:
    def _engine(self, tmp_path, exchange):
        state = StateManager(tmp_path / "state.json")
        engine = LiveEngine(
            strategy=StubStrategy(),
            exchange=exchange,
            config=AppConfig(risk=RiskConfig(max_drawdown_pct=0.99)),
            state_manager=state,
        )
        state.positions["BTC/KRW"] = Position(
            symbol="BTC/KRW",
            side=PositionSide.LONG,
            size=0.01,
            entry_price=50_000_000,
            entry_time=datetime.now(UTC),
            stop_loss=49_000_000,
        )
        return engine, state

    @pytest.mark.asyncio
    async def test_failed_exit_is_not_retried_immediately(self, tmp_path):
        import time

        feed = FailingExitFeed(price=48_000_000)  # below the stop
        engine, state = self._engine(tmp_path, feed)
        position = state.positions["BTC/KRW"]

        await engine._maybe_stop_out("BTC/KRW", position, 48_000_000)
        assert feed.attempts == 1

        # The monitor runs again 10s later: still backing off, no second order.
        await engine._maybe_stop_out("BTC/KRW", position, 48_000_000)
        assert feed.attempts == 1

        # Once the window expires the stop is retried.
        engine._exit_retry_after["BTC/KRW"] = time.monotonic() - 1
        await engine._maybe_stop_out("BTC/KRW", position, 48_000_000)
        assert feed.attempts == 2

    @pytest.mark.asyncio
    async def test_successful_exit_clears_backoff(self, tmp_path):
        import time

        feed = MockDataFeed(price=48_000_000)
        paper = PaperExchange(
            data_feed=feed,
            initial_balance=1_000_000,
            fee_rate=0.0005,
            slippage_pct=0.001,
        )
        paper.update_prices({"BTC/KRW": 48_000_000})
        paper._holdings["BTC"] = 0.01
        engine, state = self._engine(tmp_path, paper)
        engine._exit_retry_after["BTC/KRW"] = time.monotonic() + 60

        sig = Signal(
            timestamp=datetime.now(UTC),
            symbol="BTC/KRW",
            signal_type=SignalType.LONG_EXIT,
            price=48_000_000,
            strength=1.0,
        )
        await engine._handle_exit(sig, "BTC/KRW", state.positions["BTC/KRW"])

        assert "BTC/KRW" not in state.positions
        assert "BTC/KRW" not in engine._exit_retry_after
