"""Trend filters — higher timeframe trend direction."""

from __future__ import annotations

import pandas as pd

from tradingbot.strategy.filters.base import BaseFilter


class TrendUpFilter(BaseFilter):
    """Higher TF uptrend: resampled close above SMA."""

    name = "trend_up"

    def __init__(self, tf_factor: int = 4, sma_period: int = 50, base_timeframe: str = "1h"):
        super().__init__(tf_factor=tf_factor, sma_period=sma_period)
        self.tf_factor = tf_factor
        self.sma_period = sma_period
        self.base_timeframe = base_timeframe

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        from tradingbot.strategy.examples.multi_timeframe import _resample_to_higher_tf

        higher = _resample_to_higher_tf(df, self.tf_factor, self.base_timeframe)
        if len(higher) >= self.sma_period:
            htf_sma = higher["close"].rolling(self.sma_period).mean()
            df["_trend_htf_sma"] = htf_sma.reindex(df.index, method="ffill")
            df["_trend_htf_close"] = higher["close"].reindex(df.index, method="ffill")
        else:
            df["_trend_htf_sma"] = float("nan")
            df["_trend_htf_close"] = float("nan")
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        close = df["_trend_htf_close"].iloc[-1]
        sma = df["_trend_htf_sma"].iloc[-1]
        if pd.isna(close) or pd.isna(sma):
            return False
        return close > sma

    def check_exit(self, df: pd.DataFrame) -> bool:
        # Trend reversal = exit
        close = df["_trend_htf_close"].iloc[-1]
        sma = df["_trend_htf_sma"].iloc[-1]
        if pd.isna(close) or pd.isna(sma):
            return False
        return close < sma


class TrendDownFilter(BaseFilter):
    """Higher TF downtrend: resampled close below SMA."""

    name = "trend_down"

    def __init__(self, tf_factor: int = 4, sma_period: int = 50, base_timeframe: str = "1h"):
        super().__init__(tf_factor=tf_factor, sma_period=sma_period)
        self.tf_factor = tf_factor
        self.sma_period = sma_period
        self.base_timeframe = base_timeframe

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        from tradingbot.strategy.examples.multi_timeframe import _resample_to_higher_tf

        higher = _resample_to_higher_tf(df, self.tf_factor, self.base_timeframe)
        if len(higher) >= self.sma_period:
            htf_sma = higher["close"].rolling(self.sma_period).mean()
            df["_trend_dn_htf_sma"] = htf_sma.reindex(df.index, method="ffill")
            df["_trend_dn_htf_close"] = higher["close"].reindex(df.index, method="ffill")
        else:
            df["_trend_dn_htf_sma"] = float("nan")
            df["_trend_dn_htf_close"] = float("nan")
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        close = df["_trend_dn_htf_close"].iloc[-1]
        sma = df["_trend_dn_htf_sma"].iloc[-1]
        if pd.isna(close) or pd.isna(sma):
            return False
        return close < sma

    def check_exit(self, df: pd.DataFrame) -> bool:
        close = df["_trend_dn_htf_close"].iloc[-1]
        sma = df["_trend_dn_htf_sma"].iloc[-1]
        if pd.isna(close) or pd.isna(sma):
            return False
        return close > sma
