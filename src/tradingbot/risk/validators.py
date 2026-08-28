"""Pre-trade validators for live trading safety.

Provides guards against:
- Orders exceeding hard size limits
- Duplicate orders within a cooldown period
- Daily loss limits
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import structlog

logger = structlog.get_logger()


class TradeValidator:
    """Validates trades before execution for live trading safety."""

    def __init__(
        self,
        max_order_value_krw: float = 500_000,
        daily_loss_limit_krw: float = 200_000,
        order_cooldown_seconds: int = 10,
        max_order_pct: float | None = None,
        daily_loss_limit_pct: float | None = None,
    ):
        self.max_order_value_krw = max_order_value_krw
        self.daily_loss_limit_krw = daily_loss_limit_krw
        self.order_cooldown_seconds = order_cooldown_seconds
        self.max_order_pct = max_order_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct

        self._last_order_time: datetime | None = None
        self._daily_pnl: float = 0.0
        self._daily_reset_date: date | None = None

    def validate_order_size(
        self, quantity: float, price: float, equity: float | None = None
    ) -> bool:
        """Check that order value doesn't exceed the effective size limit.

        The effective limit is the min of the absolute ``max_order_value_krw``
        and, when both ``equity`` and ``max_order_pct`` are given, ``equity *
        max_order_pct``. This lets the limit scale automatically as equity
        grows (e.g. after a deposit) without redeploying with a new absolute
        value. Without equity or max_order_pct, behavior is unchanged.
        """
        value = quantity * price
        limit = self.max_order_value_krw
        if equity is not None and self.max_order_pct is not None:
            limit = min(limit, equity * self.max_order_pct)
        if value > limit:
            logger.warning(
                "order_rejected_size_limit",
                value=f"{value:,.0f}",
                limit=f"{limit:,.0f}",
            )
            return False
        return True

    def validate_cooldown(self) -> bool:
        """Check that enough time has passed since last order (anti-duplicate)."""
        if self._last_order_time is None:
            return True

        elapsed = (datetime.now(UTC) - self._last_order_time).total_seconds()
        if elapsed < self.order_cooldown_seconds:
            logger.warning(
                "order_rejected_cooldown",
                elapsed=f"{elapsed:.1f}s",
                cooldown=f"{self.order_cooldown_seconds}s",
            )
            return False
        return True

    def _effective_daily_limit(self, equity: float | None) -> float:
        """Daily-loss limit in KRW: min(absolute, equity * pct) of those set.

        Same pattern as ``validate_order_size`` — see that docstring for
        rationale. Shared by the entry gate and the between-candle rail so
        both halt at the same threshold.
        """
        limit = self.daily_loss_limit_krw
        if equity is not None and self.daily_loss_limit_pct is not None:
            limit = min(limit, equity * self.daily_loss_limit_pct)
        return limit

    def validate_daily_loss(self, equity: float | None = None) -> bool:
        """Check that daily loss limit hasn't been breached."""
        self._reset_daily_if_needed()

        limit = self._effective_daily_limit(equity)
        if self._daily_pnl < -limit:
            logger.warning(
                "order_rejected_daily_loss",
                daily_pnl=f"{self._daily_pnl:,.0f}",
                limit=f"{-limit:,.0f}",
            )
            return False
        return True

    def daily_loss_breached(self, unrealized_pnl: float = 0.0, equity: float | None = None) -> bool:
        """Return True if realized + unrealized daily PnL breaches the limit.

        The realized-only ``validate_daily_loss`` gate fires only after a loss
        is booked. Folding in open-position unrealized PnL lets the limit halt
        trading while a position is still bleeding, before the loss is locked
        in by an exit. Uses the same effective (dynamic) limit as the entry
        gate so both halt at the same threshold.
        """
        self._reset_daily_if_needed()
        total = self._daily_pnl + unrealized_pnl
        limit = self._effective_daily_limit(equity)
        if total < -limit:
            logger.warning(
                "daily_loss_breached",
                realized=f"{self._daily_pnl:,.0f}",
                unrealized=f"{unrealized_pnl:,.0f}",
                limit=f"{-limit:,.0f}",
            )
            return True
        return False

    def validate_all(self, quantity: float, price: float, equity: float | None = None) -> bool:
        """Run all validations. Returns True if order is safe to execute."""
        if not self.validate_order_size(quantity, price, equity):
            return False
        if not self.validate_cooldown():
            return False
        if not self.validate_daily_loss(equity):
            return False
        return True

    def record_order(self) -> None:
        """Record that an order was placed (for cooldown tracking)."""
        self._last_order_time = datetime.now(UTC)

    def record_trade_pnl(self, pnl: float) -> None:
        """Record a completed trade's PnL for daily tracking."""
        self._reset_daily_if_needed()
        self._daily_pnl += pnl
        logger.debug("daily_pnl_updated", daily_pnl=f"{self._daily_pnl:,.0f}")

    def daily_state(self) -> tuple[float, date | None]:
        """Return (daily_pnl, daily_reset_date) for persistence across restarts."""
        return self._daily_pnl, self._daily_reset_date

    def restore_daily_state(self, daily_pnl: float, reset_date: date | None) -> None:
        """Restore persisted daily PnL tracking after a restart.

        Without this a restart would zero the daily-loss counter, letting the
        bot keep trading past a daily loss limit it had already breached.
        """
        self._daily_pnl = daily_pnl
        self._daily_reset_date = reset_date

    def _reset_daily_if_needed(self) -> None:
        """Reset daily PnL at midnight UTC."""
        today = datetime.now(UTC).date()
        if self._daily_reset_date != today:
            if self._daily_reset_date is not None:
                logger.info("daily_pnl_reset", previous=f"{self._daily_pnl:,.0f}")
            self._daily_pnl = 0.0
            self._daily_reset_date = today
