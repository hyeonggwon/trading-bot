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
from tradingbot.ml.utils import half_kelly
from tradingbot.strategy.base import Strategy, StrategyParams

log = logging.getLogger(__name__)


class LGBMStrategy(Strategy):
    """Strategy that uses a LightGBM model for entry/exit decisions."""

    name = "lgbm"
    timeframe = "1h"
    symbols = ["BTC/KRW"]

    def __init__(self, params: StrategyParams | None = None):
        super().__init__(params)
        # Defaults sit inside the calibrated probability range produced by
        # MLWalkForwardTrainer. Isotonic calibration maps raw scores to
        # empirical positive frequencies, so the calibrator's max output is
        # bounded by the highest base-rate bin in its calibration data.
        # Across our 24 symbol×timeframe models the calibrated max ranges
        # from 0.33 (LINK/KRW 1h) up to 1.00 (several models with a perfect
        # high-confidence bin). 0.45 fires meaningfully on most models we
        # tested (8–250 entries on a ~1.3y holdout); a few thin models
        # produce few signals and need to be tuned via param_space() / the
        # optimizer. The previous 0.60 default was unreachable on tighter
        # calibrators (e.g. BTC/KRW 4h max 0.48) — that's the bug this fixes.
        self.entry_threshold: float = self.params.get("entry_threshold", 0.45)
        self.exit_threshold: float = self.params.get("exit_threshold", 0.30)
        self.model_dir = Path(self.params.get("model_dir", "models"))
        from tradingbot.data.external_fetcher import resolve_external_data_dir

        self.external_data_dir = resolve_external_data_dir(
            self.params.get("external_data_dir", None)
        )

        # Models, calibrators, and feature lists loaded lazily per symbol.
        # Feature lists must be per-symbol because different models can have
        # different feature counts (10 technical vs 16 technical+external).
        self._models: dict = {}
        self._calibrators: dict = {}
        self._feature_cols: dict[str, list[str]] = {}
        self._win_loss_ratios: dict[str, float] = {}
        # Raw external components loaded once, aligned per symbol/df
        self._external_components: dict | None = None
        self._external_load_tried: bool = False
        self._warned_missing: set[str] = set()
        # Cache for include_extra auto-detection so multi-symbol backtests
        # don't reload meta from disk on every indicators() call.
        self._include_extra_detected: bool | None = None

    def set_model(
        self,
        symbol: str,
        model: Any,
        calibrator: Any,
        feature_cols: list[str],
        win_loss_ratio: float,
    ) -> None:
        """Inject a pre-trained model, bypassing file I/O.

        Used by ml-walk-forward to swap a fresh per-window model into the
        strategy without writing to disk. ``_load_model`` checks membership in
        ``self._models``, so populating these dicts short-circuits any later
        lazy load for the same symbol.
        """
        self._models[symbol] = model
        self._calibrators[symbol] = calibrator
        self._feature_cols[symbol] = feature_cols
        self._win_loss_ratios[symbol] = win_loss_ratio

    def _load_model(self, symbol: str):
        """Lazy-load model, calibrator, and feature names for a specific symbol."""
        if symbol not in self._models:
            model = LGBMTrainer.load(symbol, self.timeframe, self.model_dir)
            if model is not None:
                self._models[symbol] = model
                self._calibrators[symbol] = LGBMTrainer.load_calibrator(
                    symbol, self.timeframe, self.model_dir
                )
                # Use feature names from model metadata (handles external features).
                # Stored per-symbol so 10- and 16-feature models can coexist.
                meta = LGBMTrainer.load_meta(symbol, self.timeframe, self.model_dir)
                if meta and "feature_names" in meta:
                    self._feature_cols[symbol] = meta["feature_names"]
                else:
                    self._feature_cols[symbol] = FEATURE_COLS
                # Empirical avg_win/avg_loss ratio from training (Kelly sizing)
                self._win_loss_ratios[symbol] = (meta or {}).get("avg_win_loss_ratio", 1.5)
                log.info(
                    f"LightGBM model loaded: {symbol} {self.timeframe} "
                    f"(win_loss_ratio={self._win_loss_ratios[symbol]})"
                )
            else:
                self._models[symbol] = None
                self._calibrators[symbol] = None
        return self._models.get(symbol)

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicator features needed by the model.

        Does not set ``self._feature_cols`` — those come from the model's
        metadata in ``_load_model`` so 10- and 16-feature models behave
        correctly. External components are loaded once (cached in
        ``self._external_components``) then aligned per-df so multi-symbol
        backtests get correctly-aligned external features per symbol.

        ``include_extra`` is auto-detected by peeking at each symbol's saved
        meta — if any model trained with the Phase 4 extras then this
        ``indicators()`` builds the wider feature frame so
        ``_predict``'s column lookup doesn't fail. Models that don't use
        extras simply ignore the additional columns.
        """
        if not self._external_load_tried and self.external_data_dir is not None:
            self._external_load_tried = True
            try:
                from tradingbot.data.external_fetcher import load_external_components

                self._external_components = load_external_components(self.external_data_dir)
            except Exception as e:
                log.warning(f"LGBMStrategy: failed to load external data: {e}")
                self._external_components = None

        external_df = None
        if self._external_components is not None:
            from tradingbot.data.external_fetcher import align_external_to

            external_df = align_external_to(df, self._external_components)

        # Decide whether to build the Phase 4 extras: any saved meta with
        # include_extra=True forces the wider build. Param override wins if set.
        # Cache the meta scan — indicators() is called per-symbol per-iteration
        # in multi-symbol backtests, and reading 8 meta files every call is
        # O(N²) I/O.
        include_extra = bool(self.params.get("include_extra", False))
        if not include_extra:
            if self._include_extra_detected is None:
                detected = False
                for sym in self.symbols:
                    meta = LGBMTrainer.load_meta(sym, self.timeframe, self.model_dir)
                    if meta and meta.get("include_extra"):
                        detected = True
                        break
                self._include_extra_detected = detected
            include_extra = self._include_extra_detected

        df, _ = build_feature_matrix(
            df,
            external_df=external_df,
            include_extra=include_extra,
        )
        return df

    def _predict(self, df: pd.DataFrame, symbol: str) -> float | None:
        """Run model inference on last candle. Returns probability or None."""
        model = self._load_model(symbol)
        if model is None:
            return None

        if len(df) < WARMUP_CANDLES + 2:
            return None

        # Guard against missing feature columns (e.g., external fetch failed
        # but model was trained with external features). Raising KeyError
        # here would abort the whole backtest — warn once per symbol instead.
        cols = self._feature_cols.get(symbol, FEATURE_COLS)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            if symbol not in self._warned_missing:
                log.warning(
                    f"LGBMStrategy[{symbol}]: missing feature columns {missing} — "
                    f"model expected {len(cols)} features. "
                    f"Check external_data_dir setting. Predictions disabled."
                )
                self._warned_missing.add(symbol)
            return None

        X = df[cols].iloc[[-1]]
        if X.isna().any(axis=1).iloc[0]:
            return None

        raw_prob = float(model.predict(X)[0])

        # Apply probability calibration if available
        calibrator = self._calibrators.get(symbol)
        if calibrator is not None:
            return float(calibrator.transform([raw_prob])[0])
        return raw_prob

    def should_entry(self, df: pd.DataFrame, symbol: str) -> Signal | None:
        prob = self._predict(df, symbol)
        if prob is None or prob < self.entry_threshold:
            return None

        ratio = self._win_loss_ratios.get(symbol, 1.5)
        strength = min(half_kelly(prob, avg_win_loss_ratio=ratio), 1.0)

        return Signal(
            timestamp=df.index[-1].to_pydatetime(),
            symbol=symbol,
            signal_type=SignalType.LONG_ENTRY,
            price=df["close"].iloc[-1],
            strength=strength,
        )

    def should_exit(self, df: pd.DataFrame, symbol: str, position: Position) -> Signal | None:
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
            "entry_threshold": [0.40, 0.45, 0.50, 0.55],
            "exit_threshold": [0.20, 0.25, 0.30, 0.35],
        }
