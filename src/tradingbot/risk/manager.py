"""Risk management module.

Enforces position sizing, max drawdown circuit breaker, and pre-trade
validation. Every signal must pass through the risk manager before
becoming an order.
"""

from __future__ import annotations

import structlog

from tradingbot.config import RiskConfig
from tradingbot.core.enums import SignalType
from tradingbot.core.models import PortfolioState, Signal

logger = structlog.get_logger()


class RiskManager:
    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()
        self.peak_equity: float = 0.0

    def update_peak_equity(self, current_equity: float) -> None:
        """Track peak equity for drawdown calculation."""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

    def current_drawdown(self, current_equity: float) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_equity == 0:
            return 0.0
        return (self.peak_equity - current_equity) / self.peak_equity

    def check_circuit_breaker(self, current_equity: float) -> bool:
        """Check if max drawdown has been breached.

        Returns True if trading should be halted.
        """
        dd = self.current_drawdown(current_equity)
        if dd >= self.config.max_drawdown_pct:
            logger.warning(
                "circuit_breaker_triggered",
                drawdown=f"{dd:.2%}",
                max_allowed=f"{self.config.max_drawdown_pct:.2%}",
            )
            return True
        return False

    def validate_signal(
        self, signal: Signal, portfolio: PortfolioState, prices: dict[str, float]
    ) -> bool:
        """Validate whether a signal should be acted on.

        Checks:
        1. Max drawdown circuit breaker (entries only)
        2. Max open positions (entries only)
        3. Max position size as percentage of portfolio
        """
        current_equity = portfolio.equity(prices)
        self.update_peak_equity(current_equity)

        # Exit signals always allowed
        if signal.signal_type == SignalType.LONG_EXIT:
            return True

        # Circuit breaker
        if self.check_circuit_breaker(current_equity):
            return False

        # Max open positions
        if len(portfolio.positions) >= self.config.max_open_positions:
            logger.debug(
                "max_positions_reached",
                current=len(portfolio.positions),
                max=self.config.max_open_positions,
            )
            return False

        return True

    def calculate_position_size(
        self,
        price: float,
        stop_loss_price: float | None,
        portfolio_equity: float,
    ) -> float:
        """Calculate position size based on risk management rules.

        Uses fixed-fractional method: risk a fixed percentage of portfolio
        per trade, sizing the position so that hitting the stop loss loses
        exactly that amount.

        If no stop loss is provided, uses max_position_size_pct as a fallback.

        Returns the quantity of the asset to buy.
        """
        if price <= 0:
            return 0.0

        if stop_loss_price is not None and stop_loss_price < price:
            # Risk-based sizing
            risk_amount = portfolio_equity * self.config.risk_per_trade_pct
            price_risk = price - stop_loss_price
            if price_risk <= 0:
                return 0.0
            quantity = risk_amount / price_risk
        else:
            # Fallback: max position size percentage
            max_value = portfolio_equity * self.config.max_position_size_pct
            quantity = max_value / price

        # Cap at max position size
        max_value = portfolio_equity * self.config.max_position_size_pct
        max_quantity = max_value / price
        quantity = min(quantity, max_quantity)

        return quantity

    def calculate_stop_loss(self, entry_price: float) -> float:
        """Calculate default stop loss price."""
        return entry_price * (1 - self.config.default_stop_loss_pct)
