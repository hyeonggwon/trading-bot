"""Momentum filters — RSI, MACD."""

from __future__ import annotations

import pandas as pd

from tradingbot.data.indicators import add_macd, add_rsi
from tradingbot.strategy.filters.base import BaseFilter


class RsiOversoldFilter(BaseFilter):
    """RSI crosses above oversold level → entry signal."""

    name = "rsi_oversold"

    def __init__(self, period: int = 14, threshold: float = 30.0):
        super().__init__(period=period, threshold=threshold)
        self.period = period
        self.threshold = threshold

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        col = f"rsi_{self.period}"
        if col not in df.columns:
            df = add_rsi(df, period=self.period)
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        if len(df) < 2:
            return False
        col = f"rsi_{self.period}"
        curr = df[col].iloc[-1]
        prev = df[col].iloc[-2]
        if pd.isna(curr) or pd.isna(prev):
            return False
        return prev <= self.threshold and curr > self.threshold

    def check_exit(self, df: pd.DataFrame) -> bool:
        return False  # Not an exit filter


class RsiOverboughtFilter(BaseFilter):
    """RSI reaches overbought level → exit signal."""

    name = "rsi_overbought"

    def __init__(self, period: int = 14, threshold: float = 70.0):
        super().__init__(period=period, threshold=threshold)
        self.period = period
        self.threshold = threshold

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        col = f"rsi_{self.period}"
        if col not in df.columns:
            df = add_rsi(df, period=self.period)
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        return False  # Not an entry filter

    def check_exit(self, df: pd.DataFrame) -> bool:
        if len(df) < 1:
            return False
        col = f"rsi_{self.period}"
        curr = df[col].iloc[-1]
        if pd.isna(curr):
            return False
        return curr >= self.threshold


class MacdCrossUpFilter(BaseFilter):
    """MACD histogram crosses above zero → entry signal."""

    name = "macd_cross_up"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(fast=fast, slow=slow, signal=signal)
        self.fast = fast
        self.slow = slow
        self.signal = signal
        if self.fast >= self.slow:
            self.fast, self.slow = min(self.fast, self.slow), max(self.fast, self.slow)

    def _hist_col(self) -> str:
        return f"macd_hist_{self.fast}_{self.slow}_{self.signal}"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._hist_col() not in df.columns:
            df = add_macd(df, fast=self.fast, slow=self.slow, signal=self.signal)
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        if len(df) < 2:
            return False
        col = self._hist_col()
        if col not in df.columns:
            return False
        curr = df[col].iloc[-1]
        prev = df[col].iloc[-2]
        if pd.isna(curr) or pd.isna(prev):
            return False
        return prev <= 0 and curr > 0

    def check_exit(self, df: pd.DataFrame) -> bool:
        if len(df) < 2:
            return False
        col = self._hist_col()
        if col not in df.columns:
            return False
        curr = df[col].iloc[-1]
        prev = df[col].iloc[-2]
        if pd.isna(curr) or pd.isna(prev):
            return False
        return prev >= 0 and curr < 0
