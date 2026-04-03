"""Abstract strategy interface.

Inspired by Freqtrade's IStrategy and Jesse's anti-lookahead enforcement.
The engine guarantees that only confirmed (closed) candles are passed to
strategy methods — the current incomplete candle is never visible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from tradingbot.core.enums import SignalType
from tradingbot.core.models import Position, Signal


@dataclass
class StrategyParams:
    """Container for strategy parameters, enabling optimization."""

    values: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


class Strategy(ABC):
    """Base class for all trading strategies.

    Subclasses must implement:
        - indicators(): compute and append indicator columns
        - should_entry(): decide whether to enter a position
        - should_exit(): decide whether to exit a position

    The backtest engine enforces anti-lookahead by only passing confirmed
    candles (up to and including the most recently closed candle). Attempting
    to access "future" data is structurally impossible.
    """

    # Default configuration — override in subclass
    name: str = "base"
    timeframe: str = "1h"
    symbols: list[str] = ["BTC/KRW"]
    # Set to False for strategies that depend on DataFrame length in indicators()
    # (e.g., higher-timeframe resampling). These will use per-iteration computation.
    supports_precompute: bool = True

    def __init__(self, params: StrategyParams | None = None):
        self.params = params or StrategyParams()
        # Copy class-level mutable defaults to instance to prevent shared state
        self.symbols = list(self.symbols)

    @abstractmethod
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute indicators and append columns to the DataFrame.

        Args:
            df: OHLCV DataFrame with DatetimeIndex. Contains only confirmed candles.

        Returns:
            The same DataFrame with indicator columns added.
        """
        ...

    @abstractmethod
    def should_entry(self, df: pd.DataFrame, symbol: str) -> Signal | None:
        """Determine if a new position should be opened.

        Args:
            df: OHLCV DataFrame with indicators. Only confirmed candles.
            symbol: The trading pair being evaluated.

        Returns:
            A Signal if entry conditions are met, None otherwise.
        """
        ...

    @abstractmethod
    def should_exit(
        self, df: pd.DataFrame, symbol: str, position: Position
    ) -> Signal | None:
        """Determine if an existing position should be closed.

        Args:
            df: OHLCV DataFrame with indicators. Only confirmed candles.
            symbol: The trading pair being evaluated.
            position: The current open position.

        Returns:
            A Signal if exit conditions are met, None otherwise.
        """
        ...

    @classmethod
    def param_space(cls) -> dict[str, list[Any]]:
        """Define parameter search space for optimization.

        Override in subclass to enable parameter optimization.
        Returns a dict mapping parameter names to lists of values to try.
        """
        return {}
