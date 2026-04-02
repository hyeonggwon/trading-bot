"""Volume Spike Breakout Strategy.

Detects abnormal volume spikes as a leading indicator of large price moves.
Combines volume anomaly detection with price breakout confirmation.

Logic:
- Track rolling volume average (20-period)
- Volume spike: current volume > avg * threshold (e.g., 2.5x)
- Entry: volume spike + price closes above recent high → breakout confirmed
- Exit: EMA crossdown or trailing stop via ATR

This captures the "smart money" accumulation pattern where
large volume precedes significant price movements.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tradingbot.core.enums import SignalType
from tradingbot.core.models import Position, Signal
from tradingbot.data.indicators import add_ema, add_volume_sma
from tradingbot.strategy.base import Strategy, StrategyParams


class VolumeBreakoutStrategy(Strategy):
    """Volume spike + price breakout strategy."""

    name = "volume_breakout"
    timeframe = "1h"
    symbols = ["BTC/KRW"]

    def __init__(self, params: StrategyParams | None = None):
        super().__init__(params)
        self.volume_sma_period: int = self.params.get("volume_sma_period", 20)
        self.volume_spike_threshold: float = self.params.get("volume_spike_threshold", 2.5)
        self.price_lookback: int = self.params.get("price_lookback", 10)
        self.exit_ema_period: int = self.params.get("exit_ema_period", 20)

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_volume_sma(df, period=self.volume_sma_period)
        df = add_ema(df, period=self.exit_ema_period)

        # Volume spike detection
        vol_sma_col = f"volume_sma_{self.volume_sma_period}"
        if vol_sma_col in df.columns:
            df["volume_ratio"] = df["volume"] / df[vol_sma_col]
        else:
            df["volume_ratio"] = 0.0

        # Recent high for breakout confirmation
        df["recent_high"] = df["high"].rolling(window=self.price_lookback).max()

        return df

    def should_entry(self, df: pd.DataFrame, symbol: str) -> Signal | None:
        if len(df) < self.price_lookback + 2:
            return None

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        volume_ratio = curr.get("volume_ratio", 0)
        recent_high = prev.get("recent_high", float("inf"))

        if pd.isna(volume_ratio) or pd.isna(recent_high):
            return None

        # Volume spike + price breakout above recent high
        has_volume_spike = volume_ratio >= self.volume_spike_threshold
        price_breakout = curr["close"] > recent_high

        if has_volume_spike and price_breakout:
            return Signal(
                timestamp=df.index[-1].to_pydatetime(),
                symbol=symbol,
                signal_type=SignalType.LONG_ENTRY,
                price=curr["close"],
            )

        return None

    def should_exit(
        self, df: pd.DataFrame, symbol: str, position: Position
    ) -> Signal | None:
        if len(df) < 2:
            return None

        curr = df.iloc[-1]
        ema_col = f"ema_{self.exit_ema_period}"

        if ema_col not in df.columns:
            return None

        curr_close = curr["close"]
        curr_ema = curr.get(ema_col)

        if pd.isna(curr_ema):
            return None

        # Exit: price closes below EMA (momentum fading)
        if curr_close < curr_ema:
            return Signal(
                timestamp=df.index[-1].to_pydatetime(),
                symbol=symbol,
                signal_type=SignalType.LONG_EXIT,
                price=curr_close,
            )

        return None

    @classmethod
    def param_space(cls) -> dict[str, list[Any]]:
        return {
            "volume_sma_period": [10, 20, 30],
            "volume_spike_threshold": [2.0, 2.5, 3.0],
            "price_lookback": [5, 10, 20],
            "exit_ema_period": [10, 20, 30],
        }
