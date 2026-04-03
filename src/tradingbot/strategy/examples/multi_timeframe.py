"""Multi-Timeframe Trend Following Strategy.

Uses higher timeframe (resampled from base data) for trend direction,
and base timeframe for entry/exit timing.

Logic:
- Resample base candles to higher TF (e.g., 1h → 4h)
- Higher TF: SMA trend filter (price above SMA = uptrend)
- Base TF entry: RSI oversold + in uptrend → Long entry
- Base TF exit: RSI overbought OR trend reversal → Long exit

This combines the best of trend-following (fewer false signals)
with mean-reversion timing (better entries).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tradingbot.core.enums import SignalType
from tradingbot.core.models import Position, Signal
from tradingbot.data.indicators import add_rsi
from tradingbot.strategy.base import Strategy, StrategyParams


TIMEFRAME_TO_MINUTES: dict[str, int] = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "4h": 240, "1d": 1440,
}


def _resample_to_higher_tf(
    df: pd.DataFrame, factor: int, base_timeframe: str = "1h"
) -> pd.DataFrame:
    """Resample OHLCV data to a higher timeframe by grouping N candles.

    Uses label='right', closed='right' to prevent lookahead bias:
    the resampled candle is timestamped at its END, so forward-filling
    only exposes it to base TF candles that come AFTER the higher TF period.
    """
    if factor <= 1:
        return df

    base_minutes = TIMEFRAME_TO_MINUTES.get(base_timeframe, 60)
    higher_minutes = base_minutes * factor

    resampled = df.resample(
        f"{higher_minutes}min", label="right", closed="right"
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    return resampled


class MultiTimeframeStrategy(Strategy):
    """Multi-timeframe trend following with RSI timing."""

    name = "multi_tf"
    timeframe = "1h"  # Base timeframe
    symbols = ["BTC/KRW"]
    supports_precompute = False  # Resampling depends on visible data length

    def __init__(self, params: StrategyParams | None = None):
        super().__init__(params)
        self.higher_tf_factor: int = self.params.get("higher_tf_factor", 4)  # 1h → 4h
        self.trend_sma_period: int = self.params.get("trend_sma_period", 50)
        self.rsi_period: int = self.params.get("rsi_period", 14)
        self.rsi_oversold: float = self.params.get("rsi_oversold", 35.0)
        self.rsi_overbought: float = self.params.get("rsi_overbought", 70.0)

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # NOTE: resample runs on each call (O(N²) in backtest loop).
        # For large datasets, consider caching. See Phase 6-8 performance optimization.
        # Base timeframe: RSI for entry timing
        df = add_rsi(df, period=self.rsi_period)

        # Higher timeframe: SMA for trend direction
        # Resample, compute SMA, then map back to base timeframe
        higher = _resample_to_higher_tf(df, self.higher_tf_factor, self.timeframe)
        if len(higher) >= self.trend_sma_period:
            higher_sma = higher["close"].rolling(self.trend_sma_period).mean()
            # Map higher TF SMA back to base TF index (forward-fill, no lookahead)
            df["htf_sma"] = higher_sma.reindex(df.index, method="ffill")
            df["htf_close"] = higher["close"].reindex(df.index, method="ffill")
        else:
            df["htf_sma"] = float("nan")
            df["htf_close"] = float("nan")

        return df

    def should_entry(self, df: pd.DataFrame, symbol: str) -> Signal | None:
        if len(df) < 2:
            return None

        rsi_col = f"rsi_{self.rsi_period}"
        if rsi_col not in df.columns:
            return None

        curr_rsi = df[rsi_col].iloc[-1]
        prev_rsi = df[rsi_col].iloc[-2]
        htf_close = df["htf_close"].iloc[-1]
        htf_sma = df["htf_sma"].iloc[-1]

        if pd.isna(curr_rsi) or pd.isna(prev_rsi) or pd.isna(htf_close) or pd.isna(htf_sma):
            return None

        # Trend filter: higher TF close above SMA = uptrend
        in_uptrend = htf_close > htf_sma

        # Entry: RSI crosses above oversold level AND in uptrend
        if in_uptrend and prev_rsi <= self.rsi_oversold and curr_rsi > self.rsi_oversold:
            return Signal(
                timestamp=df.index[-1].to_pydatetime(),
                symbol=symbol,
                signal_type=SignalType.LONG_ENTRY,
                price=df["close"].iloc[-1],
            )

        return None

    def should_exit(
        self, df: pd.DataFrame, symbol: str, position: Position
    ) -> Signal | None:
        if len(df) < 2:
            return None

        rsi_col = f"rsi_{self.rsi_period}"
        if rsi_col not in df.columns:
            return None

        curr_rsi = df[rsi_col].iloc[-1]
        htf_close = df["htf_close"].iloc[-1]
        htf_sma = df["htf_sma"].iloc[-1]

        if pd.isna(curr_rsi) or pd.isna(htf_close) or pd.isna(htf_sma):
            return None

        # Exit: RSI overbought OR trend reversal (higher TF close below SMA)
        trend_reversed = htf_close < htf_sma

        if curr_rsi >= self.rsi_overbought or trend_reversed:
            return Signal(
                timestamp=df.index[-1].to_pydatetime(),
                symbol=symbol,
                signal_type=SignalType.LONG_EXIT,
                price=df["close"].iloc[-1],
            )

        return None

    @classmethod
    def param_space(cls) -> dict[str, list[Any]]:
        return {
            "higher_tf_factor": [4, 6],
            "trend_sma_period": [20, 50],
            "rsi_period": [10, 14, 21],
            "rsi_oversold": [25, 30, 35],
            "rsi_overbought": [65, 70, 75],
        }
