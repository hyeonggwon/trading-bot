"""Price filters — breakout, EMA, Bollinger, Donchian."""

from __future__ import annotations

import pandas as pd

from tradingbot.data.indicators import add_bollinger_bands, add_donchian_channel, add_ema
from tradingbot.strategy.filters.base import BaseFilter


class PriceBreakoutFilter(BaseFilter):
    """Price closes above recent N-candle high → breakout."""

    name = "price_breakout"
    role = "entry"

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

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        return False

    @property
    def supports_vectorized(self) -> bool:
        return True

    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        return df["close"] > df[self._col_high()].shift(1)

    def vectorized_exit(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(False, index=df.index)


class EmaAboveFilter(BaseFilter):
    """Price above EMA → confirms uptrend."""

    name = "ema_above"
    role = "trend"

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

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        col = f"ema_{self.period}"
        if col not in df.columns:
            return False
        curr_close = df["close"].iloc[-1]
        curr_ema = df[col].iloc[-1]
        if pd.isna(curr_ema):
            return False
        return curr_close < curr_ema

    @property
    def supports_vectorized(self) -> bool:
        return True

    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        return df["close"] > df[f"ema_{self.period}"]

    def vectorized_exit(self, df: pd.DataFrame) -> pd.Series:
        return df["close"] < df[f"ema_{self.period}"]


class BbUpperBreakFilter(BaseFilter):
    """Price closes above upper Bollinger Band → momentum breakout."""

    name = "bb_upper_break"
    role = "entry"

    def __init__(self, period: int = 20, std: float = 2.0):
        super().__init__(period=period, std=std)
        self.period = period
        self.std = std

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        col = f"bb_upper_{self.period}_{self.std}"
        if col not in df.columns:
            df = add_bollinger_bands(df, period=self.period, std=self.std)
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        if len(df) < 2:
            return False
        col = f"bb_upper_{self.period}_{self.std}"
        if col not in df.columns:
            return False
        curr_close = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2]
        curr_bb = df[col].iloc[-1]
        prev_bb = df[col].iloc[-2]
        if pd.isna(curr_bb) or pd.isna(prev_bb):
            return False
        return prev_close <= prev_bb and curr_close > curr_bb

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        mid_col = f"bb_middle_{self.period}_{self.std}"
        if mid_col not in df.columns:
            return False
        curr_close = df["close"].iloc[-1]
        curr_mid = df[mid_col].iloc[-1]
        if pd.isna(curr_mid):
            return False
        return curr_close < curr_mid

    @property
    def supports_vectorized(self) -> bool:
        return True

    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        col = f"bb_upper_{self.period}_{self.std}"
        return (df["close"].shift(1) <= df[col].shift(1)) & (df["close"] > df[col])

    def vectorized_exit(self, df: pd.DataFrame) -> pd.Series:
        mid_col = f"bb_middle_{self.period}_{self.std}"
        return df["close"] < df[mid_col]


class EmaCrossUpFilter(BaseFilter):
    """Fast EMA crosses above slow EMA → entry signal."""

    name = "ema_cross_up"
    role = "entry"

    def __init__(self, fast: int = 12, slow: int = 26):
        super().__init__(fast=fast, slow=slow)
        self.fast = fast
        self.slow = slow

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        for p in (self.fast, self.slow):
            col = f"ema_{p}"
            if col not in df.columns:
                df = add_ema(df, period=p)
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        if len(df) < 2:
            return False
        fast_col = f"ema_{self.fast}"
        slow_col = f"ema_{self.slow}"
        curr_fast, prev_fast = df[fast_col].iloc[-1], df[fast_col].iloc[-2]
        curr_slow, prev_slow = df[slow_col].iloc[-1], df[slow_col].iloc[-2]
        if pd.isna(curr_fast) or pd.isna(prev_fast) or pd.isna(curr_slow) or pd.isna(prev_slow):
            return False
        return prev_fast <= prev_slow and curr_fast > curr_slow

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        return False

    @property
    def supports_vectorized(self) -> bool:
        return True

    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        fast = df[f"ema_{self.fast}"]
        slow = df[f"ema_{self.slow}"]
        return (fast.shift(1) <= slow.shift(1)) & (fast > slow)

    def vectorized_exit(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(False, index=df.index)


class DonchianBreakFilter(BaseFilter):
    """Close above previous Donchian upper band → breakout entry."""

    name = "donchian_break"
    role = "entry"

    def __init__(self, period: int = 20):
        super().__init__(period=period)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        col = f"dc_upper_{self.period}"
        if col not in df.columns:
            df = add_donchian_channel(df, period=self.period)
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        if len(df) < 2:
            return False
        col = f"dc_upper_{self.period}"
        if col not in df.columns:
            return False
        curr_close = df["close"].iloc[-1]
        prev_upper = df[col].iloc[-2]
        if pd.isna(prev_upper):
            return False
        return curr_close > prev_upper

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        mid_col = f"dc_middle_{self.period}"
        if mid_col not in df.columns:
            return False
        curr_close = df["close"].iloc[-1]
        curr_mid = df[mid_col].iloc[-1]
        if pd.isna(curr_mid):
            return False
        return curr_close < curr_mid

    @property
    def supports_vectorized(self) -> bool:
        return True

    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        return df["close"] > df[f"dc_upper_{self.period}"].shift(1)

    def vectorized_exit(self, df: pd.DataFrame) -> pd.Series:
        return df["close"] < df[f"dc_middle_{self.period}"]
