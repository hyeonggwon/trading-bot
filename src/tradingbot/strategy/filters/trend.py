"""Trend filters — higher timeframe trend, ADX, Ichimoku, Aroon."""

from __future__ import annotations

import pandas as pd

from tradingbot.data.indicators import add_adx, add_aroon, add_ichimoku
from tradingbot.strategy.filters.base import BaseFilter


class TrendUpFilter(BaseFilter):
    """Higher TF uptrend: resampled close above SMA."""

    name = "trend_up"
    role = "trend"

    def __init__(self, tf_factor: int = 4, sma_period: int = 50, base_timeframe: str = "1h"):
        super().__init__(tf_factor=tf_factor, sma_period=sma_period, base_timeframe=base_timeframe)
        self.tf_factor = tf_factor
        self.sma_period = sma_period
        self.base_timeframe = base_timeframe

    def _col_sma(self) -> str:
        return f"_trend_up_sma_{self.tf_factor}_{self.sma_period}_{self.base_timeframe}"

    def _col_close(self) -> str:
        return f"_trend_up_close_{self.tf_factor}_{self.sma_period}_{self.base_timeframe}"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._col_sma() in df.columns:
            return df  # Already computed

        from tradingbot.strategy.examples.multi_timeframe import _resample_to_higher_tf

        higher = _resample_to_higher_tf(df, self.tf_factor, self.base_timeframe)
        if len(higher) >= self.sma_period:
            htf_sma = higher["close"].rolling(self.sma_period).mean()
            df[self._col_sma()] = htf_sma.reindex(df.index, method="ffill")
            df[self._col_close()] = higher["close"].reindex(df.index, method="ffill")
        else:
            df[self._col_sma()] = float("nan")
            df[self._col_close()] = float("nan")
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        close = df[self._col_close()].iloc[-1]
        sma = df[self._col_sma()].iloc[-1]
        if pd.isna(close) or pd.isna(sma):
            return False
        return close > sma

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        close = df[self._col_close()].iloc[-1]
        sma = df[self._col_sma()].iloc[-1]
        if pd.isna(close) or pd.isna(sma):
            return False
        return close < sma

    @property
    def supports_vectorized(self) -> bool:
        return True

    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        return df[self._col_close()] > df[self._col_sma()]

    def vectorized_exit(self, df: pd.DataFrame) -> pd.Series:
        return df[self._col_close()] < df[self._col_sma()]


class TrendDownFilter(BaseFilter):
    """Higher TF downtrend: resampled close below SMA."""

    name = "trend_down"
    role = "trend"

    def __init__(self, tf_factor: int = 4, sma_period: int = 50, base_timeframe: str = "1h"):
        super().__init__(tf_factor=tf_factor, sma_period=sma_period, base_timeframe=base_timeframe)
        self.tf_factor = tf_factor
        self.sma_period = sma_period
        self.base_timeframe = base_timeframe

    def _col_sma(self) -> str:
        return f"_trend_dn_sma_{self.tf_factor}_{self.sma_period}_{self.base_timeframe}"

    def _col_close(self) -> str:
        return f"_trend_dn_close_{self.tf_factor}_{self.sma_period}_{self.base_timeframe}"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._col_sma() in df.columns:
            return df

        from tradingbot.strategy.examples.multi_timeframe import _resample_to_higher_tf

        higher = _resample_to_higher_tf(df, self.tf_factor, self.base_timeframe)
        if len(higher) >= self.sma_period:
            htf_sma = higher["close"].rolling(self.sma_period).mean()
            df[self._col_sma()] = htf_sma.reindex(df.index, method="ffill")
            df[self._col_close()] = higher["close"].reindex(df.index, method="ffill")
        else:
            df[self._col_sma()] = float("nan")
            df[self._col_close()] = float("nan")
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        close = df[self._col_close()].iloc[-1]
        sma = df[self._col_sma()].iloc[-1]
        if pd.isna(close) or pd.isna(sma):
            return False
        return close < sma

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        close = df[self._col_close()].iloc[-1]
        sma = df[self._col_sma()].iloc[-1]
        if pd.isna(close) or pd.isna(sma):
            return False
        return close > sma

    @property
    def supports_vectorized(self) -> bool:
        return True

    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        return df[self._col_close()] < df[self._col_sma()]

    def vectorized_exit(self, df: pd.DataFrame) -> pd.Series:
        return df[self._col_close()] > df[self._col_sma()]


class AdxStrongFilter(BaseFilter):
    """ADX above threshold → strong trend exists (direction agnostic)."""

    name = "adx_strong"
    role = "trend"

    def __init__(self, threshold: float = 25.0, period: int = 14):
        super().__init__(threshold=threshold, period=period)
        self.threshold = threshold
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        col = f"adx_{self.period}"
        if col not in df.columns:
            df = add_adx(df, period=self.period)
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        col = f"adx_{self.period}"
        if col not in df.columns:
            return False
        val = df[col].iloc[-1]
        if pd.isna(val):
            return False
        return val > self.threshold

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        col = f"adx_{self.period}"
        if col not in df.columns:
            return False
        val = df[col].iloc[-1]
        if pd.isna(val):
            return False
        return val < self.threshold

    @property
    def supports_vectorized(self) -> bool:
        return True

    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        return df[f"adx_{self.period}"] > self.threshold

    def vectorized_exit(self, df: pd.DataFrame) -> pd.Series:
        return df[f"adx_{self.period}"] < self.threshold


class IchimokuAboveFilter(BaseFilter):
    """Price above Ichimoku cloud (both span A and span B)."""

    name = "ichimoku_above"
    role = "trend"

    def __init__(self, window1: int = 9, window2: int = 26, window3: int = 52):
        super().__init__(window1=window1, window2=window2, window3=window3)
        self.window1 = window1
        self.window2 = window2
        self.window3 = window3

    def _suffix(self) -> str:
        return f"_{self.window1}_{self.window2}_{self.window3}"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        col = f"ichi_a{self._suffix()}"
        if col not in df.columns:
            df = add_ichimoku(df, window1=self.window1, window2=self.window2, window3=self.window3)
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        s = self._suffix()
        close = df["close"].iloc[-1]
        span_a = df[f"ichi_a{s}"].iloc[-1]
        span_b = df[f"ichi_b{s}"].iloc[-1]
        if pd.isna(span_a) or pd.isna(span_b):
            return False
        return close > span_a and close > span_b

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        s = self._suffix()
        close = df["close"].iloc[-1]
        span_a = df[f"ichi_a{s}"].iloc[-1]
        span_b = df[f"ichi_b{s}"].iloc[-1]
        if pd.isna(span_a) or pd.isna(span_b):
            return False
        return close < min(span_a, span_b)

    @property
    def supports_vectorized(self) -> bool:
        return True

    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        s = self._suffix()
        cloud_top = df[[f"ichi_a{s}", f"ichi_b{s}"]].max(axis=1)
        return df["close"] > cloud_top

    def vectorized_exit(self, df: pd.DataFrame) -> pd.Series:
        s = self._suffix()
        cloud_bottom = df[[f"ichi_a{s}", f"ichi_b{s}"]].min(axis=1)
        return df["close"] < cloud_bottom


class AroonUpFilter(BaseFilter):
    """Aroon Up dominant → uptrend confirmation."""

    name = "aroon_up"
    role = "trend"

    def __init__(self, threshold: float = 70.0, period: int = 25):
        super().__init__(threshold=threshold, period=period)
        self.threshold = threshold
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        col = f"aroon_up_{self.period}"
        if col not in df.columns:
            df = add_aroon(df, period=self.period)
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        up_col = f"aroon_up_{self.period}"
        dn_col = f"aroon_down_{self.period}"
        if up_col not in df.columns or dn_col not in df.columns:
            return False
        up = df[up_col].iloc[-1]
        dn = df[dn_col].iloc[-1]
        if pd.isna(up) or pd.isna(dn):
            return False
        return up > self.threshold and up > dn

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        up_col = f"aroon_up_{self.period}"
        dn_col = f"aroon_down_{self.period}"
        if up_col not in df.columns or dn_col not in df.columns:
            return False
        up = df[up_col].iloc[-1]
        dn = df[dn_col].iloc[-1]
        if pd.isna(up) or pd.isna(dn):
            return False
        return dn > up

    @property
    def supports_vectorized(self) -> bool:
        return True

    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        up = df[f"aroon_up_{self.period}"]
        dn = df[f"aroon_down_{self.period}"]
        return (up > self.threshold) & (up > dn)

    def vectorized_exit(self, df: pd.DataFrame) -> pd.Series:
        up = df[f"aroon_up_{self.period}"]
        dn = df[f"aroon_down_{self.period}"]
        return dn > up
