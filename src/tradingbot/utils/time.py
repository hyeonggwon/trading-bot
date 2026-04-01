from __future__ import annotations

from datetime import datetime, timezone


def parse_date(date_str: str) -> datetime:
    """Parse a date string (YYYY-MM-DD) into a timezone-aware UTC datetime."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.replace(tzinfo=timezone.utc)


def now_utc() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)
