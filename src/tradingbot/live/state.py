"""State persistence for live/paper trading.

Saves and restores positions, equity history, and engine state
to a JSON file so the bot can survive restarts.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from tradingbot.core.enums import PositionSide
from tradingbot.core.models import Position
from tradingbot.utils.io import atomic_write_json

logger = structlog.get_logger()

DEFAULT_STATE_PATH = Path("state.json")


class StateManager:
    """Manages persistent state for the live trading engine."""

    def __init__(self, state_path: Path = DEFAULT_STATE_PATH):
        self.state_path = state_path
        self.positions: dict[str, Position] = {}
        self.entry_fees: dict[str, float] = {}
        self.equity_history: list[dict[str, Any]] = []
        self.last_save: datetime | None = None
        # Real-money safety state that must survive restarts:
        # peak_equity drives the drawdown circuit breaker; daily_pnl /
        # daily_reset_date (ISO date string) drive the daily-loss limit.
        self.peak_equity: float = 0.0
        self.daily_pnl: float = 0.0
        self.daily_reset_date: str | None = None
        # Drawdown-breaker ledger (live/engine._ledger_equity): baseline
        # latches once to cost-basis equity; cum_realized_pnl books every
        # closed trade. Keeps the breaker blind to external deposits and
        # withdrawals, which move raw balance but not trading performance.
        self.ledger_baseline: float | None = None
        self.cum_realized_pnl: float = 0.0

    def save(self) -> None:
        """Save current state to JSON file."""
        data = {
            "positions": {symbol: _position_to_dict(pos) for symbol, pos in self.positions.items()},
            "entry_fees": self.entry_fees,
            "equity_history": self.equity_history[-1000:],  # Keep last 1000
            "peak_equity": self.peak_equity,
            "daily_pnl": self.daily_pnl,
            "daily_reset_date": self.daily_reset_date,
            "ledger_baseline": self.ledger_baseline,
            "cum_realized_pnl": self.cum_realized_pnl,
            "saved_at": datetime.now(UTC).isoformat(),
        }

        atomic_write_json(self.state_path, data, fsync=True)
        self.last_save = datetime.now(UTC)
        logger.debug("state_saved", positions=len(self.positions))

    def load(self) -> None:
        """Load state from JSON file if it exists."""
        if not self.state_path.exists():
            logger.debug("no_state_file", path=str(self.state_path))
            return

        try:
            data = json.loads(self.state_path.read_text())

            self.positions = {}
            for symbol, pos_data in data.get("positions", {}).items():
                self.positions[symbol] = _dict_to_position(pos_data)

            self.entry_fees = data.get("entry_fees", {})
            self.equity_history = data.get("equity_history", [])
            self.peak_equity = data.get("peak_equity", 0.0)
            self.daily_pnl = data.get("daily_pnl", 0.0)
            self.daily_reset_date = data.get("daily_reset_date")
            self.ledger_baseline = data.get("ledger_baseline")
            self.cum_realized_pnl = data.get("cum_realized_pnl", 0.0)

            logger.info(
                "state_loaded",
                positions=len(self.positions),
                equity_entries=len(self.equity_history),
                saved_at=data.get("saved_at"),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error("state_load_error", error=str(e))
            # Start fresh on corrupt state
            self.positions = {}
            self.entry_fees = {}
            self.equity_history = []
            self.peak_equity = 0.0
            self.daily_pnl = 0.0
            self.daily_reset_date = None
            self.ledger_baseline = None
            self.cum_realized_pnl = 0.0

    def record_equity(self, equity: float) -> None:
        """Record an equity snapshot.

        Capped in memory to the last 1000 samples (matching the persisted
        slice in ``save()``); the between-candle monitor records every few
        seconds, so an unbounded list would grow without limit on a long run.
        """
        self.equity_history.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "equity": equity,
            }
        )
        if len(self.equity_history) > 1000:
            self.equity_history = self.equity_history[-1000:]

    def annotate_last_equity(self, ledger: float) -> None:
        """Stamp the transfer-immune ledger equity onto the newest snapshot.

        Called by the safety rails immediately after ``record_equity`` (both
        tick paths call them back-to-back), so the dashboard can report a
        deposit-immune return alongside the raw balance. Skipped ticks (e.g.
        stale prices) simply leave no ledger key on that snapshot.
        """
        if self.equity_history:
            self.equity_history[-1]["ledger"] = ledger

    def clear(self) -> None:
        """Reset all state."""
        self.positions = {}
        self.entry_fees = {}
        self.equity_history = []
        self.peak_equity = 0.0
        self.daily_pnl = 0.0
        self.daily_reset_date = None
        if self.state_path.exists():
            self.state_path.unlink()


def _position_to_dict(pos: Position) -> dict[str, Any]:
    return {
        "symbol": pos.symbol,
        "side": pos.side.value,
        "size": pos.size,
        "entry_price": pos.entry_price,
        "entry_time": pos.entry_time.isoformat(),
        "stop_loss": pos.stop_loss,
        "take_profit": pos.take_profit,
        "adds": pos.adds,
    }


def _dict_to_position(data: dict[str, Any]) -> Position:
    return Position(
        symbol=data["symbol"],
        side=PositionSide(data["side"]),
        size=data["size"],
        entry_price=data["entry_price"],
        entry_time=datetime.fromisoformat(data["entry_time"]),
        stop_loss=data.get("stop_loss"),
        take_profit=data.get("take_profit"),
        adds=data.get("adds", 0),  # absent in states written before pyramiding
    )
