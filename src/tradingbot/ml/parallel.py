"""Parallel ML training worker for ProcessPoolExecutor (spawn-safe)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PairTrainResult:
    """Result from training a single symbol x timeframe pair."""

    symbol: str
    timeframe: str
    avg_auc: float
    avg_precision: float
    holdout_auc: float
    holdout_precision: float
    n_windows: int
    model_path: str
    error: str | None = None


def train_pair(
    symbol: str,
    timeframe: str,
    data_dir: str,
    model_dir: str,
    train_months: int,
    test_months: int,
    num_threads: int,
    external_data_dir: str | None = None,
    target_kind: str = "binary",
    atr_mult: float = 1.0,
) -> PairTrainResult:
    """Train a single symbol x timeframe pair.

    Top-level function for spawn-safe pickling. Imports are deferred
    to avoid loading heavy libraries in the main process.
    """
    import logging

    import structlog

    from tradingbot.data.storage import load_candles
    from tradingbot.ml.walk_forward import MLWalkForwardTrainer

    # Suppress logs in child process to avoid polluting the parent's progress bar
    logging.getLogger().setLevel(logging.CRITICAL)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
    )

    try:
        df = load_candles(symbol, timeframe, Path(data_dir))
    except FileNotFoundError:
        return PairTrainResult(
            symbol=symbol, timeframe=timeframe,
            avg_auc=0.0, avg_precision=0.0,
            holdout_auc=0.0, holdout_precision=0.0,
            n_windows=0,
            model_path="", error="no data",
        )

    try:
        # Load external features if available
        external_df = None
        if external_data_dir:
            from tradingbot.data.external_fetcher import build_external_df

            external_df = build_external_df(df, Path(external_data_dir))

        trainer = MLWalkForwardTrainer(
            symbol=symbol,
            timeframe=timeframe,
            train_months=train_months,
            test_months=test_months,
            target_kind=target_kind,
            atr_mult=atr_mult,
            model_dir=Path(model_dir),
            lgbm_params={"num_threads": num_threads},
        )
        report = trainer.run(df, external_df=external_df)
        return PairTrainResult(
            symbol=symbol,
            timeframe=timeframe,
            avg_auc=report.avg_auc,
            avg_precision=report.avg_precision,
            holdout_auc=report.holdout_auc,
            holdout_precision=report.holdout_precision,
            n_windows=len(report.windows),
            model_path=str(report.model_path) if report.model_path else "",
        )
    except Exception as exc:
        return PairTrainResult(
            symbol=symbol, timeframe=timeframe,
            avg_auc=0.0, avg_precision=0.0,
            holdout_auc=0.0, holdout_precision=0.0,
            n_windows=0,
            model_path="", error=str(exc),
        )
