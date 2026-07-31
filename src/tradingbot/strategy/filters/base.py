"""Base filter interface for the combine engine.

Each filter is a single, reusable condition (entry or exit).
Filters are combined via AND (entry) / OR (exit) in CombinedStrategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseFilter(ABC):
    """Abstract base for all signal filters."""

    name: str = "base"
    role: str = "entry"  # "entry" | "trend" | "volatility" | "volume" | "exit"
    # Candles required for the last-candle value to match a full-history
    # computation. Filters whose indicators look further back than the live
    # engine's fetch window declare it so the engine can warn at startup.
    min_history: int = 0

    def __init__(self, **kwargs: Any) -> None:
        self.params = kwargs

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicator columns needed by this filter. Returns df with new columns."""
        ...

    def check_entry(self, df: pd.DataFrame) -> bool:
        """Check if entry condition is met on the last confirmed candle.

        Default no-op: exit-only filters never get this called.
        """
        return False

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        """Check if exit condition is met on the last confirmed candle.

        Args:
            entry_index: Index position of entry candle in df (for trailing-style exits).
                         Most filters ignore this parameter.

        Default no-op: entry-only filters never get this called.
        """
        return False

    # -- Vectorized interface (for scan/combine-scan screening) --

    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        """Return boolean Series for entry condition across all rows.

        Default no-op: exit-only filters never get this called.
        """
        return pd.Series(False, index=df.index)

    def vectorized_exit(self, df: pd.DataFrame) -> pd.Series:
        """Return boolean Series for exit condition across all rows.

        Default no-op: entry-only filters never get this called.
        """
        return pd.Series(False, index=df.index)

    @property
    def supports_vectorized(self) -> bool:
        """Whether this filter supports vectorized evaluation.

        Vectorized evaluation is assumed by default; filters without a
        vectorized implementation must override this to return False.
        """
        return True
