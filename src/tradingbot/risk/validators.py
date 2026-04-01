"""Pre-trade validators for live trading safety.

Provides guards against:
- Orders exceeding hard size limits
- Duplicate orders within a cooldown period
- Daily loss limits
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import structlog

logger = structlog.get_logger()


class TradeValidator:
    """Validates trades before execution for live trading safety."""

    def __init__(
        self,
        max_order_value_krw: float = 500_000,
        daily_loss_limit_krw: float = 200_000,
        order_cooldown_seconds: int = 10,
    ):
        self.max_order_value_krw = max_order_value_krw
        self.daily_loss_limit_krw = daily_loss_limit_krw
        self.order_cooldown_seconds = order_cooldown_seconds

        self._last_order_time: datetime | None = None
        self._daily_pnl: float = 0.0
        self._daily_reset_date: datetime | None = None

    def validate_order_size(self, quantity: float, price: float) -> bool:
        """Check that order value doesn't exceed hard limit."""
        value = quantity * price
        if value > self.max_order_value_krw:
            logger.warning(
                "order_rejected_size_limit",
                value=f"{value:,.0f}",
                limit=f"{self.max_order_value_krw:,.0f}",
            )
            return False
        return True

    def validate_cooldown(self) -> bool:
        """Check that enough time has passed since last order (anti-duplicate)."""
        if self._last_order_time is None:
            return True

        elapsed = (datetime.now(timezone.utc) - self._last_order_time).total_seconds()
        if elapsed < self.order_cooldown_seconds:
            logger.warning(
                "order_rejected_cooldown",
                elapsed=f"{elapsed:.1f}s",
                cooldown=f"{self.order_cooldown_seconds}s",
            )
            return False
        return True

    def validate_daily_loss(self) -> bool:
        """Check that daily loss limit hasn't been breached."""
        self._reset_daily_if_needed()

        if self._daily_pnl < -self.daily_loss_limit_krw:
            logger.warning(
                "order_rejected_daily_loss",
                daily_pnl=f"{self._daily_pnl:,.0f}",
                limit=f"{-self.daily_loss_limit_krw:,.0f}",
            )
            return False
        return True

    def validate_all(self, quantity: float, price: float) -> bool:
        """Run all validations. Returns True if order is safe to execute."""
        if not self.validate_order_size(quantity, price):
            return False
        if not self.validate_cooldown():
            return False
        if not self.validate_daily_loss():
            return False
        return True

    def record_order(self) -> None:
        """Record that an order was placed (for cooldown tracking)."""
        self._last_order_time = datetime.now(timezone.utc)

    def record_trade_pnl(self, pnl: float) -> None:
        """Record a completed trade's PnL for daily tracking."""
        self._reset_daily_if_needed()
        self._daily_pnl += pnl
        logger.debug("daily_pnl_updated", daily_pnl=f"{self._daily_pnl:,.0f}")

    def _reset_daily_if_needed(self) -> None:
        """Reset daily PnL at midnight UTC."""
        today = datetime.now(timezone.utc).date()
        if self._daily_reset_date != today:
            if self._daily_reset_date is not None:
                logger.info("daily_pnl_reset", previous=f"{self._daily_pnl:,.0f}")
            self._daily_pnl = 0.0
            self._daily_reset_date = today
