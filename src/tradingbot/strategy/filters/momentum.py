"""Momentum filters — RSI, MACD, Stochastic, CCI, ROC, MFI."""

from __future__ import annotations

import logging

import pandas as pd

from tradingbot.data.indicators import add_cci, add_macd, add_mfi, add_roc, add_rsi, add_stochastic

log = logging.getLogger(__name__)
from tradingbot.strategy.filters.base import BaseFilter


class RsiOversoldFilter(BaseFilter):
    """RSI crosses above oversold level → entry signal."""

    name = "rsi_oversold"
    role = "entry"

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

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        return False  # Not an exit filter

    @property
    def supports_vectorized(self) -> bool:
        return True

    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        col = f"rsi_{self.period}"
        return (df[col].shift(1) <= self.threshold) & (df[col] > self.threshold)

    def vectorized_exit(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(False, index=df.index)


class RsiOverboughtFilter(BaseFilter):
    """RSI reaches overbought level → exit signal."""

    name = "rsi_overbought"
    role = "exit"

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

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        if len(df) < 1:
            return False
        col = f"rsi_{self.period}"
        curr = df[col].iloc[-1]
        if pd.isna(curr):
            return False
        return curr >= self.threshold

    @property
    def supports_vectorized(self) -> bool:
        return True

    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(False, index=df.index)

    def vectorized_exit(self, df: pd.DataFrame) -> pd.Series:
        return df[f"rsi_{self.period}"] >= self.threshold


class MacdCrossUpFilter(BaseFilter):
    """MACD histogram crosses above zero → entry signal."""

    name = "macd_cross_up"
    role = "entry"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        if fast > slow:
            log.warning(f"MACD fast({fast}) > slow({slow}). Swapping parameters.")
            fast, slow = slow, fast
        self.fast = fast
        self.slow = slow
        self.signal = signal
        super().__init__(fast=self.fast, slow=self.slow, signal=self.signal)

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

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
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

    @property
    def supports_vectorized(self) -> bool:
        return True

    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        col = self._hist_col()
        return (df[col].shift(1) <= 0) & (df[col] > 0)

    def vectorized_exit(self, df: pd.DataFrame) -> pd.Series:
        col = self._hist_col()
        return (df[col].shift(1) >= 0) & (df[col] < 0)


class StochOversoldFilter(BaseFilter):
    """Stochastic %K crosses above oversold level → entry signal."""

    name = "stoch_oversold"
    role = "entry"

    def __init__(self, threshold: float = 20.0, k_period: int = 14, d_period: int = 3):
        super().__init__(threshold=threshold, k_period=k_period, d_period=d_period)
        self.threshold = threshold
        self.k_period = k_period
        self.d_period = d_period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        col = f"stoch_k_{self.k_period}"
        if col not in df.columns:
            df = add_stochastic(df, k_period=self.k_period, d_period=self.d_period)
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        if len(df) < 2:
            return False
        col = f"stoch_k_{self.k_period}"
        curr = df[col].iloc[-1]
        prev = df[col].iloc[-2]
        if pd.isna(curr) or pd.isna(prev):
            return False
        return prev <= self.threshold and curr > self.threshold

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        return False

    @property
    def supports_vectorized(self) -> bool:
        return True

    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        col = f"stoch_k_{self.k_period}"
        return (df[col].shift(1) <= self.threshold) & (df[col] > self.threshold)

    def vectorized_exit(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(False, index=df.index)


class CciOversoldFilter(BaseFilter):
    """CCI crosses above -threshold → entry signal."""

    name = "cci_oversold"
    role = "entry"

    def __init__(self, threshold: float = 100.0, period: int = 20):
        super().__init__(threshold=threshold, period=period)
        self.threshold = threshold
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        col = f"cci_{self.period}"
        if col not in df.columns:
            df = add_cci(df, period=self.period)
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        if len(df) < 2:
            return False
        col = f"cci_{self.period}"
        curr = df[col].iloc[-1]
        prev = df[col].iloc[-2]
        if pd.isna(curr) or pd.isna(prev):
            return False
        return prev <= -self.threshold and curr > -self.threshold

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        return False

    @property
    def supports_vectorized(self) -> bool:
        return True

    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        col = f"cci_{self.period}"
        return (df[col].shift(1) <= -self.threshold) & (df[col] > -self.threshold)

    def vectorized_exit(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(False, index=df.index)


class RocPositiveFilter(BaseFilter):
    """ROC crosses above zero → entry signal."""

    name = "roc_positive"
    role = "entry"

    def __init__(self, period: int = 12):
        super().__init__(period=period)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        col = f"roc_{self.period}"
        if col not in df.columns:
            df = add_roc(df, period=self.period)
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        if len(df) < 2:
            return False
        col = f"roc_{self.period}"
        curr = df[col].iloc[-1]
        prev = df[col].iloc[-2]
        if pd.isna(curr) or pd.isna(prev):
            return False
        return prev <= 0 and curr > 0

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        return False

    @property
    def supports_vectorized(self) -> bool:
        return True

    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        col = f"roc_{self.period}"
        return (df[col].shift(1) <= 0) & (df[col] > 0)

    def vectorized_exit(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(False, index=df.index)


class MfiOversoldFilter(BaseFilter):
    """MFI crosses above oversold level → entry signal."""

    name = "mfi_oversold"
    role = "entry"

    def __init__(self, threshold: float = 20.0, period: int = 14):
        super().__init__(threshold=threshold, period=period)
        self.threshold = threshold
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        col = f"mfi_{self.period}"
        if col not in df.columns:
            df = add_mfi(df, period=self.period)
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        if len(df) < 2:
            return False
        col = f"mfi_{self.period}"
        curr = df[col].iloc[-1]
        prev = df[col].iloc[-2]
        if pd.isna(curr) or pd.isna(prev):
            return False
        return prev <= self.threshold and curr > self.threshold

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        return False

    @property
    def supports_vectorized(self) -> bool:
        return True

    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        col = f"mfi_{self.period}"
        return (df[col].shift(1) <= self.threshold) & (df[col] > self.threshold)

    def vectorized_exit(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(False, index=df.index)
