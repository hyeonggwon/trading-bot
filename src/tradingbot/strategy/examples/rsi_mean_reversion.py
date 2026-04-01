"""RSI Mean Reversion Strategy.

Buy when RSI drops below oversold level (e.g., 30) → Long entry
Sell when RSI rises above overbought level (e.g., 70) → Long exit
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tradingbot.core.enums import SignalType
from tradingbot.core.models import Position, Signal
from tradingbot.data.indicators import add_rsi
from tradingbot.strategy.base import Strategy, StrategyParams


class RsiMeanReversionStrategy(Strategy):
    name = "rsi_mean_reversion"
    timeframe = "1h"
    symbols = ["BTC/KRW"]

    def __init__(self, params: StrategyParams | None = None):
        super().__init__(params)
        self.rsi_period: int = self.params.get("rsi_period", 14)
        self.oversold: float = self.params.get("oversold", 30.0)
        self.overbought: float = self.params.get("overbought", 70.0)

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_rsi(df, period=self.rsi_period)
        return df

    def should_entry(self, df: pd.DataFrame, symbol: str) -> Signal | None:
        if len(df) < 2:
            return None

        rsi_col = f"rsi_{self.rsi_period}"
        if rsi_col not in df.columns:
            return None

        curr_rsi = df[rsi_col].iloc[-1]
        prev_rsi = df[rsi_col].iloc[-2]

        if pd.isna(curr_rsi) or pd.isna(prev_rsi):
            return None

        # RSI crosses above oversold from below
        if prev_rsi <= self.oversold and curr_rsi > self.oversold:
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

        rsi_col = f"rsi_{self.rsi_period}"
        if rsi_col not in df.columns:
            return None

        curr_rsi = df[rsi_col].iloc[-1]

        if pd.isna(curr_rsi):
            return None

        # RSI hits overbought → take profit
        if curr_rsi >= self.overbought:
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
            "rsi_period": [7, 10, 14, 21],
            "oversold": [20, 25, 30, 35],
            "overbought": [65, 70, 75, 80],
        }
