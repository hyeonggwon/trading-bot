"""Base filter interface for the combine engine.

Each filter is a single, reusable condition (entry or exit).
Filters are combined via AND (entry) / OR (exit) in CombinedStrategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseFilter(ABC):
    """Abstract base for all signal filters."""

    name: str = "base"
    role: str = "entry"  # "entry" | "trend" | "volatility" | "volume" | "exit"

    def __init__(self, **kwargs):
        self.params = kwargs

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicator columns needed by this filter. Returns df with new columns."""
        ...

    @abstractmethod
    def check_entry(self, df: pd.DataFrame) -> bool:
        """Check if entry condition is met on the last confirmed candle."""
        ...

    @abstractmethod
    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        """Check if exit condition is met on the last confirmed candle.

        Args:
            entry_index: Index position of entry candle in df (for trailing-style exits).
                         Most filters ignore this parameter.
        """
        ...
