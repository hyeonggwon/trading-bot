"""Volatility filters — ATR breakout, Keltner, BB squeeze, bandwidth."""

from __future__ import annotations

import pandas as pd

from tradingbot.data.indicators import (
    add_atr,
    add_bollinger_bands,
    add_ema,
    add_keltner_channel,
)
from tradingbot.strategy.filters.base import BaseFilter


class AtrBreakoutFilter(BaseFilter):
    """Price breaks above EMA + ATR × multiplier → volatility breakout."""

    name = "atr_breakout"
    role = "volatility"

    def __init__(self, period: int = 14, multiplier: float = 2.0, ema_period: int = 20):
        super().__init__(period=period, multiplier=multiplier, ema_period=ema_period)
        self.period = period
        self.multiplier = multiplier
        self.ema_period = ema_period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if f"atr_{self.period}" not in df.columns:
            df = add_atr(df, period=self.period)
        if f"ema_{self.ema_period}" not in df.columns:
            df = add_ema(df, period=self.ema_period)
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        atr_col = f"atr_{self.period}"
        ema_col = f"ema_{self.ema_period}"
        close = df["close"].iloc[-1]
        atr = df[atr_col].iloc[-1]
        ema = df[ema_col].iloc[-1]
        if pd.isna(atr) or pd.isna(ema):
            return False
        return close > ema + atr * self.multiplier

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        atr_col = f"atr_{self.period}"
        ema_col = f"ema_{self.ema_period}"
        close = df["close"].iloc[-1]
        atr = df[atr_col].iloc[-1]
        ema = df[ema_col].iloc[-1]
        if pd.isna(atr) or pd.isna(ema):
            return False
        return close < ema - atr * self.multiplier


class KeltnerBreakFilter(BaseFilter):
    """Price breaks above Keltner upper band."""

    name = "keltner_break"
    role = "volatility"

    def __init__(self, period: int = 20, atr_period: int = 10):
        super().__init__(period=period, atr_period=atr_period)
        self.period = period
        self.atr_period = atr_period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        col = f"kc_upper_{self.period}"
        if col not in df.columns:
            df = add_keltner_channel(df, period=self.period, atr_period=self.atr_period)
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        col = f"kc_upper_{self.period}"
        if col not in df.columns:
            return False
        close = df["close"].iloc[-1]
        upper = df[col].iloc[-1]
        if pd.isna(upper):
            return False
        return close > upper

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        mid_col = f"kc_middle_{self.period}"
        if mid_col not in df.columns:
            return False
        close = df["close"].iloc[-1]
        mid = df[mid_col].iloc[-1]
        if pd.isna(mid):
            return False
        return close < mid


class BbSqueezeFilter(BaseFilter):
    """Bollinger Bands squeeze release — BB exits Keltner Channel."""

    name = "bb_squeeze"
    role = "volatility"

    def __init__(self, bb_period: int = 20, kc_period: int = 20, bb_std: float = 2.0):
        super().__init__(bb_period=bb_period, kc_period=kc_period, bb_std=bb_std)
        self.bb_period = bb_period
        self.kc_period = kc_period
        self.bb_std = bb_std

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if f"bb_upper_{self.bb_period}_{self.bb_std}" not in df.columns:
            df = add_bollinger_bands(df, period=self.bb_period, std=self.bb_std)
        if f"kc_upper_{self.kc_period}" not in df.columns:
            df = add_keltner_channel(df, period=self.kc_period)
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        if len(df) < 2:
            return False
        bb_col = f"bb_upper_{self.bb_period}_{self.bb_std}"
        kc_col = f"kc_upper_{self.kc_period}"
        if bb_col not in df.columns or kc_col not in df.columns:
            return False
        prev_bb = df[bb_col].iloc[-2]
        prev_kc = df[kc_col].iloc[-2]
        curr_bb = df[bb_col].iloc[-1]
        curr_kc = df[kc_col].iloc[-1]
        if pd.isna(prev_bb) or pd.isna(prev_kc) or pd.isna(curr_bb) or pd.isna(curr_kc):
            return False
        # Transition: BB was inside KC → BB now outside KC
        return prev_bb < prev_kc and curr_bb >= curr_kc

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        return False  # Confirmation filter only


class BbBandwidthLowFilter(BaseFilter):
    """Bollinger Bandwidth below threshold → low volatility (squeeze precursor)."""

    name = "bb_bandwidth_low"
    role = "volatility"

    def __init__(self, threshold: float = 0.05, period: int = 20, std: float = 2.0):
        super().__init__(threshold=threshold, period=period, std=std)
        self.threshold = threshold
        self.period = period
        self.std = std

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if f"bb_upper_{self.period}_{self.std}" not in df.columns:
            df = add_bollinger_bands(df, period=self.period, std=self.std)
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        upper_col = f"bb_upper_{self.period}_{self.std}"
        lower_col = f"bb_lower_{self.period}_{self.std}"
        mid_col = f"bb_middle_{self.period}_{self.std}"
        upper = df[upper_col].iloc[-1]
        lower = df[lower_col].iloc[-1]
        mid = df[mid_col].iloc[-1]
        if pd.isna(upper) or pd.isna(lower) or pd.isna(mid) or mid == 0:
            return False
        bandwidth = (upper - lower) / mid
        return bandwidth < self.threshold

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        return False  # Confirmation filter only
