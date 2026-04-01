from __future__ import annotations

from datetime import datetime, timezone

from tradingbot.config import RiskConfig
from tradingbot.core.enums import PositionSide, SignalType
from tradingbot.core.models import PortfolioState, Position, Signal
from tradingbot.risk.manager import RiskManager


class TestRiskManager:
    def setup_method(self):
        self.config = RiskConfig(
            max_position_size_pct=0.1,
            max_open_positions=2,
            max_drawdown_pct=0.20,
            default_stop_loss_pct=0.02,
            risk_per_trade_pct=0.01,
        )
        self.rm = RiskManager(self.config)
        self.rm.peak_equity = 1_000_000

    def _make_signal(self, signal_type: SignalType) -> Signal:
        return Signal(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            symbol="BTC/KRW",
            signal_type=signal_type,
            price=50_000_000,
        )

    def _make_portfolio(self, cash: float, positions: list[Position] | None = None) -> PortfolioState:
        return PortfolioState(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            cash=cash,
            positions=positions or [],
        )

    def test_exit_always_allowed(self):
        signal = self._make_signal(SignalType.LONG_EXIT)
        portfolio = self._make_portfolio(100_000)
        assert self.rm.validate_signal(signal, portfolio, {"BTC/KRW": 50_000_000}) is True

    def test_circuit_breaker(self):
        # 25% drawdown from peak of 1M
        signal = self._make_signal(SignalType.LONG_ENTRY)
        portfolio = self._make_portfolio(750_000)
        assert self.rm.validate_signal(signal, portfolio, {"BTC/KRW": 50_000_000}) is False

    def test_max_positions(self):
        positions = [
            Position("BTC/KRW", PositionSide.LONG, 0.01, 50_000_000,
                     datetime(2024, 1, 1, tzinfo=timezone.utc)),
            Position("ETH/KRW", PositionSide.LONG, 0.1, 3_000_000,
                     datetime(2024, 1, 1, tzinfo=timezone.utc)),
        ]
        signal = self._make_signal(SignalType.LONG_ENTRY)
        portfolio = self._make_portfolio(500_000, positions)
        assert self.rm.validate_signal(signal, portfolio, {"BTC/KRW": 50_000_000, "ETH/KRW": 3_000_000}) is False

    def test_position_sizing_with_stop_loss(self):
        # Risk 1% of 1M = 10,000 KRW per trade
        # Price 50M, stop at 49M → risk per unit = 1M
        # Risk-based quantity = 10,000 / 1,000,000 = 0.01
        # But capped by max_position_size_pct (10%) → 100,000 / 50M = 0.002
        qty = self.rm.calculate_position_size(50_000_000, 49_000_000, 1_000_000)
        max_qty = (1_000_000 * 0.1) / 50_000_000  # 0.002
        assert abs(qty - max_qty) < 0.0001  # capped at max position size

    def test_position_sizing_capped(self):
        # Without stop loss, fallback to max_position_size_pct
        qty = self.rm.calculate_position_size(50_000_000, None, 1_000_000)
        max_qty = (1_000_000 * 0.1) / 50_000_000  # 0.002
        assert abs(qty - max_qty) < 0.0001

    def test_stop_loss_calculation(self):
        stop = self.rm.calculate_stop_loss(50_000_000)
        assert stop == 50_000_000 * 0.98

    def test_drawdown_tracking(self):
        self.rm.update_peak_equity(1_200_000)
        assert self.rm.peak_equity == 1_200_000
        assert abs(self.rm.current_drawdown(1_000_000) - 1 / 6) < 0.001
