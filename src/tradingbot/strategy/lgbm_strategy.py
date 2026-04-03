"""LightGBM-based trading strategy.

Uses a pre-trained LightGBM model for entry/exit decisions.
Model outputs probability of profitable trade → mapped to Signal.strength via Half-Kelly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from tradingbot.core.enums import SignalType
from tradingbot.core.models import Position, Signal
from tradingbot.ml.features import FEATURE_COLS, WARMUP_CANDLES, build_feature_matrix
from tradingbot.ml.trainer import LGBMTrainer
from tradingbot.strategy.base import Strategy, StrategyParams

log = logging.getLogger(__name__)


def _half_kelly(p: float, avg_win_loss_ratio: float = 1.5) -> float:
    """Half-Kelly criterion for position sizing.

    Args:
        p: Predicted win probability from model.
        avg_win_loss_ratio: Historical avg_win / avg_loss.

    Returns:
        Fraction of capital to risk (0.0–1.0).
    """
    q = 1.0 - p
    b = avg_win_loss_ratio
    full_kelly = (p * b - q) / b
    return max(0.0, full_kelly * 0.5)


class LGBMStrategy(Strategy):
    """Strategy that uses a LightGBM model for entry/exit decisions."""

    name = "lgbm"
    timeframe = "1h"
    symbols = ["BTC/KRW"]

    def __init__(self, params: StrategyParams | None = None):
        super().__init__(params)
        self.entry_threshold: float = self.params.get("entry_threshold", 0.60)
        self.exit_threshold: float = self.params.get("exit_threshold", 0.45)
        self.model_dir = Path(self.params.get("model_dir", "models"))

        # Models loaded lazily per symbol
        self._models: dict = {}
        self._feature_cols: list[str] = FEATURE_COLS

    def _load_model(self, symbol: str):
        """Lazy-load model for a specific symbol."""
        if symbol not in self._models:
            model = LGBMTrainer.load(symbol, self.timeframe, self.model_dir)
            if model is not None:
                self._models[symbol] = model
                log.info(f"LightGBM model loaded: {symbol} {self.timeframe}")
            else:
                self._models[symbol] = None
        return self._models.get(symbol)

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicator features needed by the model."""
        df, self._feature_cols = build_feature_matrix(df)
        return df

    def _predict(self, df: pd.DataFrame, symbol: str) -> float | None:
        """Run model inference on last candle. Returns probability or None."""
        model = self._load_model(symbol)
        if model is None:
            return None

        if len(df) < WARMUP_CANDLES + 2:
            return None

        X = df[self._feature_cols].iloc[[-1]]
        if X.isna().any(axis=1).iloc[0]:
            return None

        return float(model.predict(X)[0])

    def should_entry(self, df: pd.DataFrame, symbol: str) -> Signal | None:
        prob = self._predict(df, symbol)
        if prob is None or prob < self.entry_threshold:
            return None

        strength = min(_half_kelly(prob), 1.0)

        return Signal(
            timestamp=df.index[-1].to_pydatetime(),
            symbol=symbol,
            signal_type=SignalType.LONG_ENTRY,
            price=df["close"].iloc[-1],
            strength=strength,
        )

    def should_exit(
        self, df: pd.DataFrame, symbol: str, position: Position
    ) -> Signal | None:
        prob = self._predict(df, symbol)
        if prob is None:
            return None

        # Exit when model confidence drops below threshold
        if prob < self.exit_threshold:
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
            "entry_threshold": [0.55, 0.60, 0.65],
            "exit_threshold": [0.40, 0.45, 0.50],
        }
