"""State persistence for live/paper trading.

Saves and restores positions, equity history, and engine state
to a JSON file so the bot can survive restarts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import structlog

from tradingbot.core.enums import PositionSide
from tradingbot.core.models import Position

logger = structlog.get_logger()

DEFAULT_STATE_PATH = Path("state.json")


class StateManager:
    """Manages persistent state for the live trading engine."""

    def __init__(self, state_path: Path = DEFAULT_STATE_PATH):
        self.state_path = state_path
        self.positions: dict[str, Position] = {}
        self.equity_history: list[dict] = []
        self.last_save: datetime | None = None

    def save(self) -> None:
        """Save current state to JSON file."""
        data = {
            "positions": {
                symbol: _position_to_dict(pos)
                for symbol, pos in self.positions.items()
            },
            "equity_history": self.equity_history[-1000:],  # Keep last 1000
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        self.state_path.write_text(json.dumps(data, indent=2, default=str))
        self.last_save = datetime.now(timezone.utc)
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

            self.equity_history = data.get("equity_history", [])

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
            self.equity_history = []

    def record_equity(self, equity: float) -> None:
        """Record an equity snapshot."""
        self.equity_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "equity": equity,
        })

    def clear(self) -> None:
        """Reset all state."""
        self.positions = {}
        self.equity_history = []
        if self.state_path.exists():
            self.state_path.unlink()


def _position_to_dict(pos: Position) -> dict:
    return {
        "symbol": pos.symbol,
        "side": pos.side.value,
        "size": pos.size,
        "entry_price": pos.entry_price,
        "entry_time": pos.entry_time.isoformat(),
        "stop_loss": pos.stop_loss,
        "take_profit": pos.take_profit,
    }


def _dict_to_position(data: dict) -> Position:
    return Position(
        symbol=data["symbol"],
        side=PositionSide(data["side"]),
        size=data["size"],
        entry_price=data["entry_price"],
        entry_time=datetime.fromisoformat(data["entry_time"]),
        stop_loss=data.get("stop_loss"),
        take_profit=data.get("take_profit"),
    )
