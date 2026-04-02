"""Volume filters — spike detection."""

from __future__ import annotations

import pandas as pd

from tradingbot.data.indicators import add_volume_sma
from tradingbot.strategy.filters.base import BaseFilter


class VolumeSpikeFilter(BaseFilter):
    """Volume exceeds average by N times → confirms signal strength."""

    name = "volume_spike"

    def __init__(self, sma_period: int = 20, threshold: float = 2.5):
        super().__init__(sma_period=sma_period, threshold=threshold)
        self.sma_period = sma_period
        self.threshold = threshold

    def _col_ratio(self) -> str:
        return f"_vol_ratio_{self.sma_period}"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._col_ratio() in df.columns:
            return df

        col = f"volume_sma_{self.sma_period}"
        if col not in df.columns:
            df = add_volume_sma(df, period=self.sma_period)
        df[self._col_ratio()] = df["volume"] / df[col]
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        col = self._col_ratio()
        if col not in df.columns:
            return False
        ratio = df[col].iloc[-1]
        if pd.isna(ratio):
            return False
        return ratio >= self.threshold

    def check_exit(self, df: pd.DataFrame) -> bool:
        return False
