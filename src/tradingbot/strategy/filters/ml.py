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
        external_data_dir: str | bool | None = None,
    ):
        super().__init__(threshold=threshold)
        self.threshold = threshold
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_dir = Path(model_dir)
        from tradingbot.data.external_fetcher import resolve_external_data_dir

        self.external_data_dir = resolve_external_data_dir(external_data_dir)

        self._model = None
        self._calibrator = None
        self._feature_names: list[str] | None = None
        self._loaded = False
        self._external_components: dict | None = None
        self._external_load_tried: bool = False
        self._warned_missing = False
        self.last_prob: float | None = None
        self.last_strength: float | None = None

    def _load_model(self):
        """Lazy-load LightGBM model and calibrator. Only attempts once."""
        if self._loaded:
            return self._model

        self._loaded = True
        try:
            from tradingbot.ml.trainer import LGBMTrainer

            model = LGBMTrainer.load(self.symbol, self.timeframe, self.model_dir)
            if model is not None:
                self._model = model
                self._calibrator = LGBMTrainer.load_calibrator(
                    self.symbol, self.timeframe, self.model_dir
                )
                # Load feature names from metadata (may include external features)
                meta = LGBMTrainer.load_meta(self.symbol, self.timeframe, self.model_dir)
                if meta and "feature_names" in meta:
                    self._feature_names = meta["feature_names"]
                log.info(f"LgbmProbFilter: model loaded for {self.symbol} {self.timeframe}")
            else:
                log.warning(f"LgbmProbFilter: no model found for {self.symbol} {self.timeframe}")
        except ImportError:
            log.warning("LgbmProbFilter: lightgbm not installed — filter disabled")
        return self._model

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ML feature columns to DataFrame.

        Merges external features when ``external_data_dir`` is configured
        so models trained with kimchi/funding/etc. see those columns here.
        """
        try:
            from tradingbot.ml.features import FEATURE_COLS, build_feature_matrix

            # Load model first so we can check the full set of required feature
            # names (10 for pure-technical models, 16 for external-feature models)
            self._load_model()
            required_cols = self._feature_names or FEATURE_COLS

            # Skip if all required features already computed
            if all(col in df.columns for col in required_cols):
                return df

            if not self._external_load_tried and self.external_data_dir is not None:
                self._external_load_tried = True
                try:
                    from tradingbot.data.external_fetcher import load_external_components

                    self._external_components = load_external_components(self.external_data_dir)
                except Exception as e:
                    log.warning(f"LgbmProbFilter: failed to load external data: {e}")

            external_df = None
            if self._external_components is not None:
                from tradingbot.data.external_fetcher import align_external_to

                external_df = align_external_to(df, self._external_components)

            df, _ = build_feature_matrix(df, external_df=external_df)
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

        # Use model's trained feature names if available (may include external features)
        cols = self._feature_names if self._feature_names else FEATURE_COLS
        # Skip if required columns are missing from DataFrame
        missing = [c for c in cols if c not in df.columns]
        if missing:
            if not self._warned_missing:
                log.warning(
                    f"LgbmProbFilter: model expects {missing} but not in features — "
                    f"filter will always return False. Set external_data_dir or retrain."
                )
                self._warned_missing = True
            return False

        X = df[cols].iloc[[-1]]
        if X.isna().any(axis=1).iloc[0]:
            return False

        raw_prob = float(model.predict(X)[0])
        prob = raw_prob
        if self._calibrator is not None:
            prob = float(self._calibrator.transform([raw_prob])[0])
        self.last_prob = prob

        if prob >= self.threshold:
            from tradingbot.ml.utils import half_kelly

            self.last_strength = min(half_kelly(prob), 1.0)
            return True

        return False

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        return False  # Entry-only filter
