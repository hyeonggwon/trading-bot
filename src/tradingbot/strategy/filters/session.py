"""Session filter — KST time-of-day entry gate."""

from __future__ import annotations

import pandas as pd

from tradingbot.strategy.filters.base import BaseFilter


class SessionKstFilter(BaseFilter):
    """Allow entries only inside a KST hour window (end-exclusive).

    Upbit trades 24/7, but KRW-market liquidity and volatility concentrate in
    Korean waking hours — this gates NEW entries to a session window without
    ever touching exits (an open position must always be closable). Supports
    an overnight wrap: ``start_hour=22, end_hour=6`` means 22:00–06:00 KST.
    Hours use the same UTC+9 shift convention as the ML session features
    (``ml/features.py`` ``hour_kst``), so backtest and model views agree.
    """

    name = "session_kst"
    role = "entry"

    def __init__(self, start_hour: int = 9, end_hour: int = 23):
        super().__init__(start_hour=start_hour, end_hour=end_hour)
        if not (0 <= start_hour <= 23 and 0 <= end_hour <= 23):
            raise ValueError("session hours must be in 0..23")
        if start_hour == end_hour:
            raise ValueError("empty session window (start_hour == end_hour)")
        self.start_hour = start_hour
        self.end_hour = end_hour

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        return df  # time comes from the index; no indicator columns needed

    def _in_session(self, hour: int) -> bool:
        if self.start_hour < self.end_hour:
            return self.start_hour <= hour < self.end_hour
        # Overnight wrap (e.g. 22 → 6)
        return hour >= self.start_hour or hour < self.end_hour

    def check_entry(self, df: pd.DataFrame) -> bool:
        if len(df) == 0 or not isinstance(df.index, pd.DatetimeIndex):
            return False
        kst_hour = int((df.index[-1] + pd.Timedelta(hours=9)).hour)
        return self._in_session(kst_hour)

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        return False  # Entry gate only — never blocks or forces an exit

    @property
    def supports_vectorized(self) -> bool:
        return True

    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        if not isinstance(df.index, pd.DatetimeIndex):
            return pd.Series(False, index=df.index)
        hours = (df.index + pd.Timedelta(hours=9)).hour
        if self.start_hour < self.end_hour:
            mask = (hours >= self.start_hour) & (hours < self.end_hour)
        else:
            mask = (hours >= self.start_hour) | (hours < self.end_hour)
        return pd.Series(mask, index=df.index)

    def vectorized_exit(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(False, index=df.index)
