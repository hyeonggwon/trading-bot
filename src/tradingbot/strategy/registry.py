"""Strategy registry — single source of truth for all built-in strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tradingbot.strategy.base import Strategy


def get_strategy_map() -> dict[str, type[Strategy]]:
    """Return map of strategy name → class. Lazy imports to avoid circular deps."""
    from tradingbot.strategy.examples.bollinger_breakout import BollingerBreakoutStrategy
    from tradingbot.strategy.examples.macd_momentum import MacdMomentumStrategy
    from tradingbot.strategy.examples.multi_timeframe import MultiTimeframeStrategy
    from tradingbot.strategy.examples.rsi_mean_reversion import RsiMeanReversionStrategy
    from tradingbot.strategy.examples.sma_cross import SmaCrossStrategy
    from tradingbot.strategy.examples.volume_breakout import VolumeBreakoutStrategy

    from tradingbot.strategy.lgbm_strategy import LGBMStrategy

    return {
        "sma_cross": SmaCrossStrategy,
        "rsi_mean_reversion": RsiMeanReversionStrategy,
        "macd_momentum": MacdMomentumStrategy,
        "bollinger_breakout": BollingerBreakoutStrategy,
        "multi_tf": MultiTimeframeStrategy,
        "volume_breakout": VolumeBreakoutStrategy,
        "lgbm": LGBMStrategy,
    }
