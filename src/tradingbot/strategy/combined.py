"""Combined strategy — composes multiple filters into a single strategy.

Entry: ALL entry filters must be satisfied (AND logic)
Exit: ANY exit filter triggers (OR logic)

Usage:
    from tradingbot.strategy.filters.registry import parse_filter_string
    from tradingbot.strategy.combined import CombinedStrategy

    entry_filters = parse_filter_string("trend_up:4 + rsi_oversold:30")
    exit_filters = parse_filter_string("rsi_overbought:70")
    strategy = CombinedStrategy(entry_filters=entry_filters, exit_filters=exit_filters)
"""

from __future__ import annotations

import logging

import pandas as pd

from tradingbot.core.enums import SignalType
from tradingbot.core.models import Position, Signal
from tradingbot.strategy.base import Strategy
from tradingbot.strategy.filters.base import BaseFilter

log = logging.getLogger(__name__)


class CombinedStrategy(Strategy):
    """Strategy that combines multiple filters via AND (entry) / OR (exit)."""

    name = "combined"
    timeframe = "1h"
    symbols = ["BTC/KRW"]

    def __init__(
        self,
        entry_filters: list[BaseFilter] | None = None,
        exit_filters: list[BaseFilter] | None = None,
    ):
        super().__init__()
        self.entry_filters = entry_filters or []
        self.exit_filters = exit_filters or []
        self._entry_indices: dict[str, int] = {}
        self._entry_times: dict[str, pd.Timestamp] = {}
        self._unique_filters = self._deduplicate_filters()

    @property
    def min_history(self) -> int:
        """Candles needed so every filter's last-candle value matches full history."""
        return max((f.min_history for f in self._unique_filters), default=0)

    def _deduplicate_filters(self) -> list[BaseFilter]:
        """Pre-compute unique filter list for indicators() (avoid per-call key sorting)."""
        seen: set[tuple] = set()
        unique: list[BaseFilter] = []
        for f in self.entry_filters + self.exit_filters:
            key = (f.__class__.__name__, tuple(sorted(f.params.items())))
            if key not in seen:
                unique.append(f)
                seen.add(key)
        return unique

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators needed by all filters (deduplicated)."""
        for f in self._unique_filters:
            df = f.compute(df)
        return df

    def should_entry(self, df: pd.DataFrame, symbol: str) -> Signal | None:
        if len(df) < 2 or not self.entry_filters:
            return None

        # AND logic: all entry filters must pass (skip exit-role filters)
        checked = 0
        strength = 1.0
        for f in self.entry_filters:
            if f.role == "exit":
                continue
            checked += 1
            if not f.check_entry(df):
                return None
            # Collect ML-based strength if available
            if hasattr(f, "last_strength") and f.last_strength is not None:
                strength = f.last_strength

        if checked == 0:
            return None  # No non-exit filters to evaluate

        # Cache entry index (fast path) AND the signal-candle timestamp so
        # trailing-style exits can re-anchor when the live fetch window slides.
        self._entry_indices[symbol] = len(df) - 1
        self._entry_times[symbol] = df.index[-1]

        return Signal(
            timestamp=df.index[-1].to_pydatetime(),
            symbol=symbol,
            signal_type=SignalType.LONG_ENTRY,
            price=df["close"].iloc[-1],
            strength=strength,
        )

    def should_exit(self, df: pd.DataFrame, symbol: str, position: Position) -> Signal | None:
        if len(df) < 2 or not self.exit_filters:
            return None

        # Re-anchor the entry candle by timestamp against the *current* window.
        entry_index = self._resolve_entry_index(df, symbol, position)

        # OR logic: any exit filter triggers exit
        for f in self.exit_filters:
            if f.check_exit(df, entry_index=entry_index):
                self._entry_indices.pop(symbol, None)
                self._entry_times.pop(symbol, None)
                return Signal(
                    timestamp=df.index[-1].to_pydatetime(),
                    symbol=symbol,
                    signal_type=SignalType.LONG_EXIT,
                    price=df["close"].iloc[-1],
                )

        return None

    def _resolve_entry_index(self, df: pd.DataFrame, symbol: str, position: Position) -> int | None:
        """Positional index of the entry candle in the *current* df window.

        Anchored on the entry *timestamp* (the signal candle, cached in
        ``_entry_times``; the persisted ``position.entry_time`` is the
        restart fallback) rather than a positional index frozen at signal
        time. A frozen positional index is stable in backtest — slices are
        anchored at index 0, so a row keeps its position — but in live trading
        the rolling ``fetch_ohlcv`` window slides forward each tick, so the
        cached index drifts off the entry candle, and a restart loses the cache
        entirely. Re-locating the timestamp keeps "highest high since entry"
        (e.g. ``AtrTrailingExitFilter``) anchored across both. Returns None when
        no anchor is known so exit filters fall back to their own heuristic.
        """
        anchor = self._entry_times.get(symbol)
        if anchor is None:
            anchor = getattr(position, "entry_time", None)
        if anchor is None:
            return None

        ts = pd.Timestamp(anchor)
        index = df.index
        index_tz = getattr(index, "tz", None)
        if index_tz is not None and ts.tz is None:
            ts = ts.tz_localize(index_tz)
        elif index_tz is None and ts.tz is not None:
            ts = ts.tz_localize(None)

        pos = int(index.searchsorted(ts, side="left"))
        if pos >= len(index):
            pos = len(index) - 1
        # When the entry candle has scrolled out of the current window (held
        # longer than the rolling fetch window), its timestamp predates every
        # bar and searchsorted returns 0 for a bar that is NOT the entry candle.
        # Returning 0 would silently anchor "highest high since entry" on the
        # oldest available bar; fall back to the filter's own heuristic instead.
        if pos == 0 and len(index) > 0 and index[0] > ts:
            return None
        return max(pos, 0)

    def describe(self) -> str:
        """Human-readable description of the strategy."""
        entry_desc = " + ".join(f.name for f in self.entry_filters) or "none"
        exit_desc = " + ".join(f.name for f in self.exit_filters) or "none"
        return f"Entry[{entry_desc}] → Exit[{exit_desc}]"
