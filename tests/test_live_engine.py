"""Tests for live engine bug fixes (price handling, stop loss, equity)."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from tradingbot.config import AppConfig, RiskConfig
from tradingbot.core.enums import OrderSide, OrderStatus, OrderType, PositionSide, SignalType
from tradingbot.core.models import Order, Position, Signal
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
                data_feed=feed, initial_balance=10_000_000,
                fee_rate=0.0005, slippage_pct=0.001,
            )
            paper.update_prices({"BTC/KRW": 50_000_000})
            config = AppConfig(risk=RiskConfig(
                default_stop_loss_pct=0.02,
                risk_per_trade_pct=0.01,
                max_position_size_pct=0.5,
            ))
            state = StateManager(tmp_path / f"state_{strength}.json")
            engine = LiveEngine(
                strategy=StubStrategy(), exchange=paper, config=config,
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
            data_feed=feed, initial_balance=10_000_000,
            fee_rate=0.0005, slippage_pct=0.001,
        )
        paper.update_prices({"BTC/KRW": 48_000_000})
        paper._holdings["BTC"] = 0.001  # holdings to sell on exit

        config = AppConfig(risk=RiskConfig(default_stop_loss_pct=0.02))
        state = StateManager(tmp_path / "state.json")
        engine = LiveEngine(
            strategy=StubStrategy(), exchange=paper, config=config,
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
            data_feed=feed, initial_balance=10_000_000,
            fee_rate=0.0005, slippage_pct=0.001,
        )
        paper.update_prices({"BTC/KRW": 51_000_000})

        config = AppConfig(risk=RiskConfig(default_stop_loss_pct=0.02))
        state = StateManager(tmp_path / "state.json")
        engine = LiveEngine(
            strategy=StubStrategy(), exchange=paper, config=config,
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
            strategy=StubStrategy(), exchange=feed, config=config,
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
            strategy=StubStrategy(), exchange=paper, config=config,
            state_manager=StateManager(state_path), trade_validator=v1,
        )
        engine1.risk_manager.peak_equity = 12_345_678.0
        engine1._persist_state()

        # Second session (restart): fresh objects load from the same file.
        v2 = TradeValidator(daily_loss_limit_krw=200_000)
        engine2 = LiveEngine(
            strategy=StubStrategy(), exchange=paper, config=config,
            state_manager=StateManager(state_path), trade_validator=v2,
        )
        engine2._restore_state()

        assert engine2.risk_manager.peak_equity == 12_345_678.0
        assert v2.daily_state()[0] == -150_000


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
            strategy=StubStrategy(), exchange=feed, config=config,
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
            strategy=StubStrategy(), exchange=feed, config=config,
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
            strategy=StubStrategy(), exchange=feed, config=config,
            state_manager=StateManager(tmp_path / "state.json"),
        )
        engine.state.positions["BTC/KRW"] = Position(
            symbol="BTC/KRW", side=PositionSide.LONG, size=0.1,
            entry_price=50_000_000, entry_time=datetime.now(UTC),
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
            strategy=StubStrategy(), exchange=feed, config=config,
            state_manager=StateManager(tmp_path / "state.json"),
        )
        engine.state.positions["BTC/KRW"] = Position(
            symbol="BTC/KRW", side=PositionSide.LONG, size=0.1,
            entry_price=50_000_000, entry_time=datetime.now(UTC),
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
        config = AppConfig(risk=RiskConfig(
            default_stop_loss_pct=0.02,
            risk_per_trade_pct=0.01,
            max_position_size_pct=0.5,
        ))
        engine = LiveEngine(
            strategy=StubStrategy(), exchange=feed, config=config,
            state_manager=StateManager(tmp_path / "state.json"),
        )
        signal = Signal(
            timestamp=datetime.now(UTC), symbol="BTC/KRW",
            signal_type=SignalType.LONG_ENTRY, price=50_000_000, strength=1.0,
        )

        # Must not raise — the lost-response order is reconciled internally.
        await engine._handle_entry(signal, "BTC/KRW", 50_000_000)

        pos = engine.state.positions.get("BTC/KRW")
        assert pos is not None  # orphan fill adopted
        assert pos.size == pytest.approx(0.1)
        assert pos.stop_loss is not None and pos.stop_loss < pos.entry_price


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
            data_feed=feed, initial_balance=8_000_000,
            fee_rate=0.0005, slippage_pct=0.001,
        )
        paper.update_prices({"BTC/KRW": 49_500_000})
        paper._holdings["BTC"] = 0.001  # holdings to sell on flatten

        config = AppConfig(risk=RiskConfig(
            max_drawdown_pct=0.10, default_stop_loss_pct=0.02,
        ))
        state = StateManager(tmp_path / "state.json")
        engine = LiveEngine(
            strategy=StubStrategy(), exchange=paper, config=config,
            state_manager=state,
        )
        # Peak well above current equity -> drawdown breaches the 10% limit.
        engine.risk_manager.peak_equity = 10_000_000.0
        state.positions["BTC/KRW"] = Position(
            symbol="BTC/KRW", side=PositionSide.LONG, size=0.001,
            entry_price=50_000_000, entry_time=datetime.now(UTC),
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
            data_feed=feed, initial_balance=1_000_000,
            fee_rate=0.0005, slippage_pct=0.001,
        )
        paper.update_prices({"BTC/KRW": 45_000_000})
        paper._holdings["BTC"] = 0.01

        config = AppConfig(risk=RiskConfig(
            max_drawdown_pct=0.99, default_stop_loss_pct=0.02,
        ))
        state = StateManager(tmp_path / "state.json")
        validator = TradeValidator(daily_loss_limit_krw=40_000)
        engine = LiveEngine(
            strategy=StubStrategy(), exchange=paper, config=config,
            state_manager=state, trade_validator=validator,
        )
        state.positions["BTC/KRW"] = Position(
            symbol="BTC/KRW", side=PositionSide.LONG, size=0.01,
            entry_price=50_000_000, entry_time=datetime.now(UTC),
            stop_loss=40_000_000,  # far below price: no stop-out
        )
        # Unrealized = (45M - 50M) * 0.01 = -50,000 < -40,000 limit.

        await engine._monitor_prices(["BTC/KRW"])

        assert "BTC/KRW" not in state.positions


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
                    id="sell-partial", symbol=symbol, side=side,
                    order_type=order_type, quantity=self.SOLD,  # < requested
                    status=OrderStatus.FILLED,
                    filled_price=self._price, fee=1_000.0,
                    filled_at=datetime.now(UTC),
                )

        feed = PartialSellFeed(price=50_000_000)
        config = AppConfig(risk=RiskConfig(default_stop_loss_pct=0.02))
        state = StateManager(tmp_path / "state.json")
        engine = LiveEngine(
            strategy=StubStrategy(), exchange=feed, config=config,
            state_manager=state,
        )
        state.positions["BTC/KRW"] = Position(
            symbol="BTC/KRW", side=PositionSide.LONG, size=1.0,
            entry_price=50_000_000, entry_time=datetime.now(UTC),
            stop_loss=49_000_000,
        )
        state.entry_fees["BTC/KRW"] = 1_000.0

        sig = Signal(
            timestamp=datetime.now(UTC), symbol="BTC/KRW",
            signal_type=SignalType.LONG_EXIT, price=50_000_000,
        )
        await engine._handle_exit(sig, "BTC/KRW", state.positions["BTC/KRW"])

        # Position retained (still managed, keeps its stop), size reduced to the
        # unsold remainder rather than deleted.
        assert "BTC/KRW" in state.positions
        assert state.positions["BTC/KRW"].size == pytest.approx(1.0 - PartialSellFeed.SOLD)
        assert state.positions["BTC/KRW"].stop_loss == 49_000_000
