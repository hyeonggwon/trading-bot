"""MACD Momentum Strategy.

Entry: MACD histogram turns positive (crosses above zero) → Long entry
Exit: MACD histogram turns negative (crosses below zero) → Long exit
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tradingbot.core.enums import SignalType
from tradingbot.core.models import Position, Signal
from tradingbot.data.indicators import add_macd
from tradingbot.strategy.base import Strategy, StrategyParams


class MacdMomentumStrategy(Strategy):
    name = "macd_momentum"
    timeframe = "1h"
    symbols = ["BTC/KRW"]

    def __init__(self, params: StrategyParams | None = None):
        super().__init__(params)
        self.fast: int = self.params.get("fast", 12)
        self.slow: int = self.params.get("slow", 26)
        self.signal: int = self.params.get("signal", 9)
        self._valid = self.fast < self.slow

    def _hist_col(self) -> str:
        return f"macd_hist_{self.fast}_{self.slow}_{self.signal}"

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_macd(df, fast=self.fast, slow=self.slow, signal=self.signal)
        return df

    def should_entry(self, df: pd.DataFrame, symbol: str) -> Signal | None:
        if not self._valid or len(df) < 2:
            return None

        hist_col = self._hist_col()
        if hist_col not in df.columns:
            return None

        curr = df[hist_col].iloc[-1]
        prev = df[hist_col].iloc[-2]

        if pd.isna(curr) or pd.isna(prev):
            return None

        # Histogram crosses above zero
        if prev <= 0 and curr > 0:
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
        if not self._valid or len(df) < 2:
            return None

        hist_col = self._hist_col()
        if hist_col not in df.columns:
            return None

        curr = df[hist_col].iloc[-1]
        prev = df[hist_col].iloc[-2]

        if pd.isna(curr) or pd.isna(prev):
            return None

        # Histogram crosses below zero
        if prev >= 0 and curr < 0:
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
            "fast": [8, 10, 12, 15],
            "slow": [20, 24, 26, 30],
            "signal": [7, 9, 11],
        }
