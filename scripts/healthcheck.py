#!/usr/bin/env python3
"""Docker health check script.

Checks that the trading bot is alive by verifying:
1. state.json exists and was recently updated
2. The update time is within a reasonable window (3x timeframe)

Exit codes:
  0 = healthy
  1 = unhealthy
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

STATE_FILE = Path(os.environ.get("STATE_FILE", "state.json"))
# Maximum allowed staleness in seconds (default: 3 hours = 3x 1h timeframe)
MAX_STALE_SECONDS = int(os.environ.get("HEALTHCHECK_MAX_STALE_SECONDS", 10800))


def main() -> int:
    if not STATE_FILE.exists():
        # No state file yet — bot may still be warming up
        print("WARN: state file not found, bot may be starting up")
        return 0

    try:
        data = json.loads(STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"ERROR: cannot read state file: {e}")
        return 1

    saved_at = data.get("saved_at")
    if not saved_at:
        print("WARN: state file has no saved_at timestamp")
        return 0

    try:
        last_save = datetime.fromisoformat(saved_at)
        if last_save.tzinfo is None:
            last_save = last_save.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        print(f"ERROR: invalid timestamp: {saved_at}")
        return 1

    now = datetime.now(timezone.utc)
    age_seconds = (now - last_save).total_seconds()

    if age_seconds > MAX_STALE_SECONDS:
        print(f"UNHEALTHY: state file is {age_seconds:.0f}s old (max: {MAX_STALE_SECONDS}s)")
        return 1

    print(f"OK: state updated {age_seconds:.0f}s ago")
    return 0


if __name__ == "__main__":
    sys.exit(main())
