from __future__ import annotations

from datetime import datetime, timezone

from tradingbot.core.enums import (
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    SignalType,
)
from tradingbot.core.models import (
    Candle,
    Order,
    PortfolioState,
    Position,
    Signal,
    Trade,
    candles_to_dataframe,
    dataframe_to_candles,
)


class TestCandle:
    def test_frozen(self):
        c = Candle(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=100, high=110, low=95, close=105, volume=1000,
        )
        assert c.close == 105

    def test_to_dict(self):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        c = Candle(timestamp=ts, open=100, high=110, low=95, close=105, volume=1000)
        d = c.to_dict()
        assert d["open"] == 100
        assert d["timestamp"] == ts


class TestCandleConversion:
    def test_roundtrip(self, sample_candles):
        df = candles_to_dataframe(sample_candles)
        assert len(df) == 10
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

        recovered = dataframe_to_candles(df)
        assert len(recovered) == 10
        assert recovered[0].open == sample_candles[0].open
        assert recovered[0].close == sample_candles[0].close

    def test_empty_list(self):
        df = candles_to_dataframe([])
        assert df.empty


class TestSignal:
    def test_creation(self):
        s = Signal(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            symbol="BTC/KRW",
            signal_type=SignalType.LONG_ENTRY,
            price=50_000_000,
            strength=0.8,
        )
        assert s.signal_type == SignalType.LONG_ENTRY
        assert s.strength == 0.8


class TestTrade:
    def test_pnl_calculation(self):
        entry = Order(
            id="1", symbol="BTC/KRW", side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=0.01,
            filled_price=50_000_000, fee=2500,
            status=OrderStatus.FILLED,
            filled_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        exit_ = Order(
            id="2", symbol="BTC/KRW", side=OrderSide.SELL,
            order_type=OrderType.MARKET, quantity=0.01,
            filled_price=51_000_000, fee=2550,
            status=OrderStatus.FILLED,
            filled_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        trade = Trade(symbol="BTC/KRW", entry_order=entry, exit_order=exit_)

        # gross = (51M - 50M) * 0.01 = 10000
        # net = 10000 - 2500 - 2550 = 4950
        assert trade.pnl == 4950.0
        assert trade.is_win is True
        assert trade.duration is not None
        assert trade.duration == 24.0  # 24 hours

    def test_losing_trade(self):
        entry = Order(
            id="1", symbol="BTC/KRW", side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=0.01,
            filled_price=50_000_000, fee=2500,
            status=OrderStatus.FILLED,
        )
        exit_ = Order(
            id="2", symbol="BTC/KRW", side=OrderSide.SELL,
            order_type=OrderType.MARKET, quantity=0.01,
            filled_price=49_000_000, fee=2450,
            status=OrderStatus.FILLED,
        )
        trade = Trade(symbol="BTC/KRW", entry_order=entry, exit_order=exit_)
        # gross = (49M - 50M) * 0.01 = -10000
        # net = -10000 - 2500 - 2450 = -14950
        assert trade.pnl == -14950.0
        assert trade.is_win is False


class TestPosition:
    def test_unrealized_pnl(self):
        pos = Position(
            symbol="BTC/KRW",
            side=PositionSide.LONG,
            size=0.01,
            entry_price=50_000_000,
            entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        # Price went up to 52M
        assert pos.unrealized_pnl(52_000_000) == 20000.0
        assert pos.unrealized_pnl_pct(52_000_000) == 20000.0 / 500000.0


class TestPortfolioState:
    def test_equity(self):
        pos = Position(
            symbol="BTC/KRW",
            side=PositionSide.LONG,
            size=0.01,
            entry_price=50_000_000,
            entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        state = PortfolioState(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            cash=500_000,
            positions=[pos],
        )
        # equity = cash + position value at current price
        equity = state.equity({"BTC/KRW": 52_000_000})
        assert equity == 500_000 + 0.01 * 52_000_000  # 500000 + 520000 = 1020000
