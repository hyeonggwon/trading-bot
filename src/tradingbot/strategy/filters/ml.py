"""ML probability filter — wraps LightGBM model output as a BaseFilter.

Allows ML predictions to be combined with rule-based filters via CombinedStrategy.
Example: tradingbot combine --entry "trend_up:4 + rsi_oversold:30 + lgbm_prob:0.55"
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from tradingbot.strategy.filters.base import BaseFilter

log = logging.getLogger(__name__)


def _half_kelly(p: float, avg_win_loss_ratio: float = 1.5) -> float:
    """Half-Kelly criterion for position sizing."""
    q = 1.0 - p
    b = avg_win_loss_ratio
    full_kelly = (p * b - q) / b if b > 0 else 0.0
    return max(0.0, full_kelly * 0.5)


class LgbmProbFilter(BaseFilter):
    """LightGBM probability filter — entry veto based on model confidence.

    Uses a pre-trained LightGBM model to predict win probability.
    Entry passes only when prob >= threshold. Also computes Half-Kelly
    strength for position sizing via CombinedStrategy.
    """

    name = "lgbm_prob"
    role = "entry"

    def __init__(
        self,
        threshold: float = 0.55,
        symbol: str = "BTC/KRW",
        timeframe: str = "1h",
        model_dir: str = "models",
    ):
        super().__init__(threshold=threshold)
        self.threshold = threshold
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_dir = Path(model_dir)

        self._model = None
        self._loaded = False
        self.last_prob: float | None = None
        self.last_strength: float | None = None

    def _load_model(self):
        """Lazy-load LightGBM model. Only attempts once."""
        if self._loaded:
            return self._model

        self._loaded = True
        try:
            from tradingbot.ml.trainer import LGBMTrainer

            model = LGBMTrainer.load(self.symbol, self.timeframe, self.model_dir)
            if model is not None:
                self._model = model
                log.info(f"LgbmProbFilter: model loaded for {self.symbol} {self.timeframe}")
            else:
                log.warning(f"LgbmProbFilter: no model found for {self.symbol} {self.timeframe}")
        except ImportError:
            log.warning("LgbmProbFilter: lightgbm not installed — filter disabled")
        return self._model

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ML feature columns to DataFrame."""
        try:
            from tradingbot.ml.features import FEATURE_COLS, build_feature_matrix

            # Skip if features already computed
            if all(col in df.columns for col in FEATURE_COLS):
                return df

            df, _ = build_feature_matrix(df)
        except ImportError:
            log.warning("LgbmProbFilter: lightgbm/ml not installed — skipping compute")
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        """Check if model probability exceeds threshold."""
        self.last_prob = None
        self.last_strength = None

        model = self._load_model()
        if model is None:
            return False

        try:
            from tradingbot.ml.features import FEATURE_COLS, WARMUP_CANDLES
        except ImportError:
            return False

        if len(df) < WARMUP_CANDLES + 2:
            return False

        X = df[FEATURE_COLS].iloc[[-1]]
        if X.isna().any(axis=1).iloc[0]:
            return False

        prob = float(model.predict(X)[0])
        self.last_prob = prob

        if prob >= self.threshold:
            self.last_strength = min(_half_kelly(prob), 1.0)
            return True

        return False

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        return False  # Entry-only filter
