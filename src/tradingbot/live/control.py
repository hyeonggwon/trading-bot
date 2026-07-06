"""Dashboard→engine control channel (entry pause/resume kill-switch).

``state.json`` is written by the engine every tick, so an operator flag
stored there would be overwritten immediately. The control file is the
reverse channel: the dashboard (or any operator tool) writes it, the engine
only reads it. A missing file means "not paused" (normal operation).

Pause blocks NEW entries only — the engine keeps managing existing
positions (stop losses, take profits, strategy exits, safety rails).
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import structlog

logger = structlog.get_logger()


def control_path_for(state_path: Path) -> Path:
    """Control file lives next to the state file (state.json → state.control.json)."""
    return state_path.with_suffix(".control.json")


def read_pause(control_path: Path) -> bool:
    """True if entries are paused. Missing or unreadable file == not paused.

    Fail-open on corruption: the kill-switch is an operator convenience, not
    a safety rail — the drawdown breaker and daily-loss limit stay in charge
    of risk regardless of this flag.
    """
    try:
        data = json.loads(control_path.read_text())
    except FileNotFoundError:
        return False
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("control_file_unreadable", path=str(control_path), error=str(e))
        return False
    return bool(data.get("pause_entries", False))


def set_pause(control_path: Path, paused: bool) -> None:
    """Atomically write the pause flag (same tmp+rename pattern as state.json)."""
    data = {
        "pause_entries": paused,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    control_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=control_path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, control_path)
    except Exception:
        os.unlink(tmp_path)
        raise
