"""Exit signal filters — Stochastic, CCI, MFI overbought, Z-score, PctFromMA, ATR trailing."""

from __future__ import annotations

import pandas as pd

from tradingbot.data.indicators import (
    add_atr,
    add_cci,
    add_mfi,
    add_pct_from_ma,
    add_stochastic,
    add_zscore,
)
from tradingbot.strategy.filters.base import BaseFilter


class StochOverboughtFilter(BaseFilter):
    """Stochastic %K reaches overbought level → exit signal."""

    name = "stoch_overbought"
    role = "exit"

    def __init__(self, threshold: float = 80.0, k_period: int = 14, d_period: int = 3):
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
        return False

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        col = f"stoch_k_{self.k_period}"
        if col not in df.columns:
            return False
        val = df[col].iloc[-1]
        if pd.isna(val):
            return False
        return val >= self.threshold


class CciOverboughtFilter(BaseFilter):
    """CCI above +threshold → exit signal."""

    name = "cci_overbought"
    role = "exit"

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
        return False

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        col = f"cci_{self.period}"
        if col not in df.columns:
            return False
        val = df[col].iloc[-1]
        if pd.isna(val):
            return False
        return val > self.threshold


class MfiOverboughtFilter(BaseFilter):
    """MFI reaches overbought level → exit signal."""

    name = "mfi_overbought"
    role = "exit"

    def __init__(self, threshold: float = 80.0, period: int = 14):
        super().__init__(threshold=threshold, period=period)
        self.threshold = threshold
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        col = f"mfi_{self.period}"
        if col not in df.columns:
            df = add_mfi(df, period=self.period)
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        return False

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        col = f"mfi_{self.period}"
        if col not in df.columns:
            return False
        val = df[col].iloc[-1]
        if pd.isna(val):
            return False
        return val >= self.threshold


class ZscoreExtremeFilter(BaseFilter):
    """Z-score above +threshold → overbought exit."""

    name = "zscore_extreme"
    role = "exit"

    def __init__(self, threshold: float = 2.0, period: int = 20):
        super().__init__(threshold=threshold, period=period)
        self.threshold = threshold
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        col = f"zscore_{self.period}"
        if col not in df.columns:
            df = add_zscore(df, period=self.period)
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        return False

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        col = f"zscore_{self.period}"
        if col not in df.columns:
            return False
        val = df[col].iloc[-1]
        if pd.isna(val):
            return False
        return val > self.threshold


class PctFromMaExitFilter(BaseFilter):
    """Percent from MA exceeds threshold → overbought exit."""

    name = "pct_from_ma_exit"
    role = "exit"

    def __init__(self, period: int = 20, threshold: float = 5.0):
        super().__init__(period=period, threshold=threshold)
        self.period = period
        self.threshold = threshold

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        col = f"pct_from_ma_{self.period}"
        if col not in df.columns:
            df = add_pct_from_ma(df, period=self.period)
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        return False

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        col = f"pct_from_ma_{self.period}"
        if col not in df.columns:
            return False
        val = df[col].iloc[-1]
        if pd.isna(val):
            return False
        return val > self.threshold


class AtrTrailingExitFilter(BaseFilter):
    """ATR trailing stop — exit when price drops below highest since entry minus ATR × multiplier."""

    name = "atr_trailing_exit"
    role = "exit"

    _FALLBACK_LOOKBACK = 20

    def __init__(self, period: int = 14, multiplier: float = 2.5):
        super().__init__(period=period, multiplier=multiplier)
        self.period = period
        self.multiplier = multiplier

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        col = f"atr_{self.period}"
        if col not in df.columns:
            df = add_atr(df, period=self.period)
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        return False

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        atr_col = f"atr_{self.period}"
        if atr_col not in df.columns:
            return False
        atr = df[atr_col].iloc[-1]
        close = df["close"].iloc[-1]
        if pd.isna(atr):
            return False

        # Determine highest high since entry
        if entry_index is not None and 0 <= entry_index < len(df):
            highest = df["high"].iloc[entry_index:].max()
        else:
            # Fallback: use recent N candles
            lookback = min(self._FALLBACK_LOOKBACK, len(df))
            highest = df["high"].iloc[-lookback:].max()

        if pd.isna(highest):
            return False
        return close < highest - atr * self.multiplier
