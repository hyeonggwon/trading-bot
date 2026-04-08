"""Volume filters — spike, OBV, MFI confirmation."""

from __future__ import annotations

import pandas as pd

from tradingbot.data.indicators import add_mfi, add_obv, add_volume_sma
from tradingbot.strategy.filters.base import BaseFilter


class VolumeSpikeFilter(BaseFilter):
    """Volume exceeds average by N times → confirms signal strength."""

    name = "volume_spike"
    role = "volume"

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

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        return False

    @property
    def supports_vectorized(self) -> bool:
        return True

    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        return df[self._col_ratio()] >= self.threshold

    def vectorized_exit(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(False, index=df.index)


class ObvRisingFilter(BaseFilter):
    """OBV above its SMA → accumulation in progress."""

    name = "obv_rising"
    role = "volume"

    def __init__(self, obv_sma_period: int = 20):
        super().__init__(obv_sma_period=obv_sma_period)
        self.obv_sma_period = obv_sma_period

    def _col_sma(self) -> str:
        return f"_obv_sma_{self.obv_sma_period}"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if "obv" not in df.columns:
            df = add_obv(df)
        col = self._col_sma()
        if col not in df.columns:
            df[col] = df["obv"].rolling(window=self.obv_sma_period).mean()
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        col = self._col_sma()
        if "obv" not in df.columns or col not in df.columns:
            return False
        obv = df["obv"].iloc[-1]
        sma = df[col].iloc[-1]
        if pd.isna(obv) or pd.isna(sma):
            return False
        return obv > sma

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        col = self._col_sma()
        if "obv" not in df.columns or col not in df.columns:
            return False
        obv = df["obv"].iloc[-1]
        sma = df[col].iloc[-1]
        if pd.isna(obv) or pd.isna(sma):
            return False
        return obv < sma

    @property
    def supports_vectorized(self) -> bool:
        return True

    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        return df["obv"] > df[self._col_sma()]

    def vectorized_exit(self, df: pd.DataFrame) -> pd.Series:
        return df["obv"] < df[self._col_sma()]


class MfiConfirmFilter(BaseFilter):
    """MFI above threshold → money flow confirms entry."""

    name = "mfi_confirm"
    role = "volume"

    def __init__(self, threshold: float = 50.0, period: int = 14):
        super().__init__(threshold=threshold, period=period)
        self.threshold = threshold
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        col = f"mfi_{self.period}"
        if col not in df.columns:
            df = add_mfi(df, period=self.period)
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        col = f"mfi_{self.period}"
        if col not in df.columns:
            return False
        val = df[col].iloc[-1]
        if pd.isna(val):
            return False
        return val > self.threshold

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        col = f"mfi_{self.period}"
        if col not in df.columns:
            return False
        val = df[col].iloc[-1]
        if pd.isna(val):
            return False
        return val < (100 - self.threshold)

    @property
    def supports_vectorized(self) -> bool:
        return True

    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        return df[f"mfi_{self.period}"] > self.threshold

    def vectorized_exit(self, df: pd.DataFrame) -> pd.Series:
        return df[f"mfi_{self.period}"] < (100 - self.threshold)
