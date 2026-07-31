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
        return bool(close > ema + atr * self.multiplier)

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        atr_col = f"atr_{self.period}"
        ema_col = f"ema_{self.ema_period}"
        close = df["close"].iloc[-1]
        atr = df[atr_col].iloc[-1]
        ema = df[ema_col].iloc[-1]
        if pd.isna(atr) or pd.isna(ema):
            return False
        return bool(close < ema - atr * self.multiplier)

    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        return (
            df["close"] > df[f"ema_{self.ema_period}"] + df[f"atr_{self.period}"] * self.multiplier
        )

    def vectorized_exit(self, df: pd.DataFrame) -> pd.Series:
        return (
            df["close"] < df[f"ema_{self.ema_period}"] - df[f"atr_{self.period}"] * self.multiplier
        )


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
        return bool(close > upper)

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        mid_col = f"kc_middle_{self.period}"
        if mid_col not in df.columns:
            return False
        close = df["close"].iloc[-1]
        mid = df[mid_col].iloc[-1]
        if pd.isna(mid):
            return False
        return bool(close < mid)

    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        return df["close"] > df[f"kc_upper_{self.period}"]

    def vectorized_exit(self, df: pd.DataFrame) -> pd.Series:
        return df["close"] < df[f"kc_middle_{self.period}"]


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
        return bool(prev_bb < prev_kc and curr_bb >= curr_kc)

    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        bb_col = f"bb_upper_{self.bb_period}_{self.bb_std}"
        kc_col = f"kc_upper_{self.kc_period}"
        return (df[bb_col].shift(1) < df[kc_col].shift(1)) & (df[bb_col] >= df[kc_col])


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
        return bool(bandwidth < self.threshold)

    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        upper = df[f"bb_upper_{self.period}_{self.std}"]
        lower = df[f"bb_lower_{self.period}_{self.std}"]
        mid = df[f"bb_middle_{self.period}_{self.std}"]
        bandwidth = (upper - lower) / mid.replace(0, float("nan"))
        return bandwidth < self.threshold


def _add_realized_vol(df: pd.DataFrame, vol_period: int, rank_period: int) -> pd.DataFrame:
    """Add the realized-vol percentile-rank column (mirrors ml/features.py extras)."""
    vol_col = f"realized_vol_{vol_period}"
    pct_col = f"realized_vol_pct_{vol_period}_{rank_period}"
    if pct_col not in df.columns:
        ret = df["close"].pct_change()
        df[vol_col] = ret.rolling(vol_period, min_periods=10).std()
        df[pct_col] = df[vol_col].rolling(rank_period, min_periods=10).rank(pct=True)
    return df


class RealizedVolLowFilter(BaseFilter):
    """Realized-vol percentile below threshold → calm regime confirmation.

    The regime idea from the ML extras (``features.py`` ``realized_vol_pct_50``)
    as a tradable gate: mean-reversion entries behave in quiet tape and get run
    over in expanding volatility. Percentile rank (not absolute vol) keeps one
    threshold meaningful across symbols and timeframes.
    """

    name = "realized_vol_low"
    role = "volatility"

    def __init__(self, threshold: float = 0.3, vol_period: int = 20, rank_period: int = 50):
        super().__init__(threshold=threshold, vol_period=vol_period, rank_period=rank_period)
        self.threshold = threshold
        self.vol_period = vol_period
        self.rank_period = rank_period
        # Full-window parity bound: pct_change eats 1 candle and the rank
        # window must hold only full-window vols (bit-exact at the bound;
        # beyond it live values silently drift from backtest).
        self.min_history = vol_period + rank_period + 1

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        return _add_realized_vol(df, self.vol_period, self.rank_period)

    def check_entry(self, df: pd.DataFrame) -> bool:
        col = f"realized_vol_pct_{self.vol_period}_{self.rank_period}"
        if col not in df.columns:
            return False
        val = df[col].iloc[-1]
        if pd.isna(val):
            return False
        return bool(val < self.threshold)

    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        return df[f"realized_vol_pct_{self.vol_period}_{self.rank_period}"] < self.threshold


class RealizedVolHighFilter(BaseFilter):
    """Realized-vol percentile above threshold → expanding-vol regime.

    Sibling of ``RealizedVolLowFilter`` for breakout-style entries that want
    the tape moving (momentum/breakout signals fire best when volatility is
    ranking high against its own recent history).
    """

    name = "realized_vol_high"
    role = "volatility"

    def __init__(self, threshold: float = 0.7, vol_period: int = 20, rank_period: int = 50):
        super().__init__(threshold=threshold, vol_period=vol_period, rank_period=rank_period)
        self.threshold = threshold
        self.vol_period = vol_period
        self.rank_period = rank_period
        # Same parity bound as RealizedVolLowFilter.
        self.min_history = vol_period + rank_period + 1

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        return _add_realized_vol(df, self.vol_period, self.rank_period)

    def check_entry(self, df: pd.DataFrame) -> bool:
        col = f"realized_vol_pct_{self.vol_period}_{self.rank_period}"
        if col not in df.columns:
            return False
        val = df[col].iloc[-1]
        if pd.isna(val):
            return False
        return bool(val > self.threshold)

    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        return df[f"realized_vol_pct_{self.vol_period}_{self.rank_period}"] > self.threshold
