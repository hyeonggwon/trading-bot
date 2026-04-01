"""Bollinger Band Breakout Strategy.

Entry: Price closes above upper Bollinger Band → momentum breakout → Long entry
Exit: Price closes below middle band (SMA) → momentum fading → Long exit
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tradingbot.core.enums import SignalType
from tradingbot.core.models import Position, Signal
from tradingbot.data.indicators import add_bollinger_bands
from tradingbot.strategy.base import Strategy, StrategyParams


class BollingerBreakoutStrategy(Strategy):
    name = "bollinger_breakout"
    timeframe = "1h"
    symbols = ["BTC/KRW"]

    def __init__(self, params: StrategyParams | None = None):
        super().__init__(params)
        self.period: int = self.params.get("period", 20)
        self.std: float = self.params.get("std", 2.0)

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_bollinger_bands(df, period=self.period, std=self.std)
        return df

    def should_entry(self, df: pd.DataFrame, symbol: str) -> Signal | None:
        if len(df) < 2:
            return None

        upper_col = f"bb_upper_{self.period}"
        if upper_col not in df.columns:
            return None

        curr_close = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2]
        curr_upper = df[upper_col].iloc[-1]
        prev_upper = df[upper_col].iloc[-2]

        if pd.isna(curr_upper) or pd.isna(prev_upper):
            return None

        # Price breaks above upper band
        if prev_close <= prev_upper and curr_close > curr_upper:
            return Signal(
                timestamp=df.index[-1].to_pydatetime(),
                symbol=symbol,
                signal_type=SignalType.LONG_ENTRY,
                price=curr_close,
            )
        return None

    def should_exit(
        self, df: pd.DataFrame, symbol: str, position: Position
    ) -> Signal | None:
        if len(df) < 2:
            return None

        mid_col = f"bb_middle_{self.period}"
        if mid_col not in df.columns:
            return None

        curr_close = df["close"].iloc[-1]
        curr_mid = df[mid_col].iloc[-1]

        if pd.isna(curr_mid):
            return None

        # Price drops below middle band
        if curr_close < curr_mid:
            return Signal(
                timestamp=df.index[-1].to_pydatetime(),
                symbol=symbol,
                signal_type=SignalType.LONG_EXIT,
                price=curr_close,
            )
        return None

    @classmethod
    def param_space(cls) -> dict[str, list[Any]]:
        return {
            "period": [15, 20, 25, 30],
            "std": [1.5, 2.0, 2.5, 3.0],
        }
