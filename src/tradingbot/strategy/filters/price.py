"""Price filters — breakout, EMA, Bollinger."""

from __future__ import annotations

import pandas as pd

from tradingbot.data.indicators import add_bollinger_bands, add_ema
from tradingbot.strategy.filters.base import BaseFilter


class PriceBreakoutFilter(BaseFilter):
    """Price closes above recent N-candle high → breakout."""

    name = "price_breakout"

    def __init__(self, lookback: int = 10):
        super().__init__(lookback=lookback)
        self.lookback = lookback

    def _col_high(self) -> str:
        return f"_recent_high_{self.lookback}"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._col_high() in df.columns:
            return df
        df[self._col_high()] = df["high"].rolling(window=self.lookback).max()
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        if len(df) < 2:
            return False
        col = self._col_high()
        if col not in df.columns:
            return False
        curr_close = df["close"].iloc[-1]
        prev_high = df[col].iloc[-2]
        if pd.isna(prev_high):
            return False
        return curr_close > prev_high

    def check_exit(self, df: pd.DataFrame) -> bool:
        return False


class EmaAboveFilter(BaseFilter):
    """Price above EMA → confirms uptrend."""

    name = "ema_above"

    def __init__(self, period: int = 20):
        super().__init__(period=period)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        col = f"ema_{self.period}"
        if col not in df.columns:
            df = add_ema(df, period=self.period)
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        col = f"ema_{self.period}"
        if col not in df.columns:
            return False
        curr_close = df["close"].iloc[-1]
        curr_ema = df[col].iloc[-1]
        if pd.isna(curr_ema):
            return False
        return curr_close > curr_ema

    def check_exit(self, df: pd.DataFrame) -> bool:
        col = f"ema_{self.period}"
        if col not in df.columns:
            return False
        curr_close = df["close"].iloc[-1]
        curr_ema = df[col].iloc[-1]
        if pd.isna(curr_ema):
            return False
        return curr_close < curr_ema


class BbUpperBreakFilter(BaseFilter):
    """Price closes above upper Bollinger Band → momentum breakout."""

    name = "bb_upper_break"

    def __init__(self, period: int = 20, std: float = 2.0):
        super().__init__(period=period, std=std)
        self.period = period
        self.std = std

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        col = f"bb_upper_{self.period}"
        if col not in df.columns:
            df = add_bollinger_bands(df, period=self.period, std=self.std)
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        if len(df) < 2:
            return False
        col = f"bb_upper_{self.period}"
        if col not in df.columns:
            return False
        curr_close = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2]
        curr_bb = df[col].iloc[-1]
        prev_bb = df[col].iloc[-2]
        if pd.isna(curr_bb) or pd.isna(prev_bb):
            return False
        return prev_close <= prev_bb and curr_close > curr_bb

    def check_exit(self, df: pd.DataFrame) -> bool:
        mid_col = f"bb_middle_{self.period}"
        if mid_col not in df.columns:
            return False
        curr_close = df["close"].iloc[-1]
        curr_mid = df[mid_col].iloc[-1]
        if pd.isna(curr_mid):
            return False
        return curr_close < curr_mid
