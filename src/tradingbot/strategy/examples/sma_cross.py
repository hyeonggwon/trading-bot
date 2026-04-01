"""SMA Crossover Strategy.

Golden Cross (fast SMA crosses above slow SMA) → Long entry
Dead Cross (fast SMA crosses below slow SMA) → Long exit
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tradingbot.core.enums import SignalType
from tradingbot.core.models import Position, Signal
from tradingbot.data.indicators import add_sma
from tradingbot.strategy.base import Strategy, StrategyParams


class SmaCrossStrategy(Strategy):
    name = "sma_cross"
    timeframe = "1h"
    symbols = ["BTC/KRW"]

    def __init__(self, params: StrategyParams | None = None):
        super().__init__(params)
        self.fast_period: int = self.params.get("fast_period", 20)
        self.slow_period: int = self.params.get("slow_period", 50)

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_sma(df, period=self.fast_period)
        df = add_sma(df, period=self.slow_period)
        return df

    def should_entry(self, df: pd.DataFrame, symbol: str) -> Signal | None:
        if len(df) < 2:
            return None

        fast_col = f"sma_{self.fast_period}"
        slow_col = f"sma_{self.slow_period}"

        if fast_col not in df.columns or slow_col not in df.columns:
            return None

        curr_fast = df[fast_col].iloc[-1]
        curr_slow = df[slow_col].iloc[-1]
        prev_fast = df[fast_col].iloc[-2]
        prev_slow = df[slow_col].iloc[-2]

        if pd.isna(curr_fast) or pd.isna(curr_slow) or pd.isna(prev_fast) or pd.isna(prev_slow):
            return None

        # Golden cross: fast crosses above slow
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            return Signal(
                timestamp=df.index[-1].to_pydatetime(),
                symbol=symbol,
                signal_type=SignalType.LONG_ENTRY,
                price=df["close"].iloc[-1],
            )

        return None

    def should_exit(
        self, df: pd.DataFrame, symbol: str, position: Position
    ) -> Signal | None:
        if len(df) < 2:
            return None

        fast_col = f"sma_{self.fast_period}"
        slow_col = f"sma_{self.slow_period}"

        if fast_col not in df.columns or slow_col not in df.columns:
            return None

        curr_fast = df[fast_col].iloc[-1]
        curr_slow = df[slow_col].iloc[-1]
        prev_fast = df[fast_col].iloc[-2]
        prev_slow = df[slow_col].iloc[-2]

        if pd.isna(curr_fast) or pd.isna(curr_slow) or pd.isna(prev_fast) or pd.isna(prev_slow):
            return None

        # Dead cross: fast crosses below slow
        if prev_fast >= prev_slow and curr_fast < curr_slow:
            return Signal(
                timestamp=df.index[-1].to_pydatetime(),
                symbol=symbol,
                signal_type=SignalType.LONG_EXIT,
                price=df["close"].iloc[-1],
            )

        return None

    @classmethod
    def param_space(cls) -> dict[str, list[Any]]:
        return {
            "fast_period": [10, 15, 20, 25, 30],
            "slow_period": [40, 50, 60, 70, 80],
        }
