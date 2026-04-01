from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from tradingbot.core.models import Candle, Order, Signal, Trade


@dataclass(frozen=True)
class CandleEvent:
    """New confirmed candle received."""

    candle: Candle
    symbol: str


@dataclass(frozen=True)
class SignalEvent:
    """Strategy generated a signal."""

    signal: Signal


@dataclass(frozen=True)
class OrderEvent:
    """Order state changed."""

    order: Order


@dataclass(frozen=True)
class TradeEvent:
    """Round-trip trade completed."""

    trade: Trade
