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

import pandas as pd

from tradingbot.core.enums import SignalType
from tradingbot.core.models import Position, Signal
from tradingbot.strategy.base import Strategy
from tradingbot.strategy.filters.base import BaseFilter


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

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators needed by all filters (deduplicated)."""
        seen: set[tuple] = set()
        for f in self.entry_filters + self.exit_filters:
            key = (f.__class__.__name__, tuple(sorted(f.params.items())))
            if key not in seen:
                df = f.compute(df)
                seen.add(key)
        return df

    def should_entry(self, df: pd.DataFrame, symbol: str) -> Signal | None:
        if len(df) < 2 or not self.entry_filters:
            return None

        # AND logic: all entry filters must pass
        for f in self.entry_filters:
            if not f.check_entry(df):
                return None

        return Signal(
            timestamp=df.index[-1].to_pydatetime(),
            symbol=symbol,
            signal_type=SignalType.LONG_ENTRY,
            price=df["close"].iloc[-1],
        )

    def should_exit(
        self, df: pd.DataFrame, symbol: str, position: Position
    ) -> Signal | None:
        if len(df) < 2 or not self.exit_filters:
            return None

        # OR logic: any exit filter triggers exit
        for f in self.exit_filters:
            if f.check_exit(df):
                return Signal(
                    timestamp=df.index[-1].to_pydatetime(),
                    symbol=symbol,
                    signal_type=SignalType.LONG_EXIT,
                    price=df["close"].iloc[-1],
                )

        return None

    def describe(self) -> str:
        """Human-readable description of the strategy."""
        entry_desc = " + ".join(f.name for f in self.entry_filters) or "none"
        exit_desc = " + ".join(f.name for f in self.exit_filters) or "none"
        return f"Entry[{entry_desc}] → Exit[{exit_desc}]"
