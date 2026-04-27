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


@dataclass
class ThresholdTunePairResult:
    """Result from tuning thresholds for one (symbol, timeframe) pair."""

    symbol: str
    timeframe: str
    best_entry: float = 0.45
    best_exit: float = 0.30
    best_sharpe: float = float("-inf")
    best_return_pct: float = 0.0
    best_trades: int = 0
    best_win_rate: float = 0.0
    best_max_dd_pct: float = 0.0
    baseline_sharpe: float = float("-inf")
    baseline_return_pct: float = 0.0
    baseline_trades: int = 0
    n_combos_evaluated: int = 0
    holdout_start: str = ""
    holdout_end: str = ""
    meta_patched: bool = False
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
    include_extra: bool = False,
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
            symbol=symbol,
            timeframe=timeframe,
            avg_auc=0.0,
            avg_precision=0.0,
            holdout_auc=0.0,
            holdout_precision=0.0,
            n_windows=0,
            model_path="",
            error="no data",
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
            include_extra=include_extra,
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
            symbol=symbol,
            timeframe=timeframe,
            avg_auc=0.0,
            avg_precision=0.0,
            holdout_auc=0.0,
            holdout_precision=0.0,
            n_windows=0,
            model_path="",
            error=str(exc),
        )


def tune_thresholds_pair(
    symbol: str,
    timeframe: str,
    data_dir: str,
    model_dir: str,
    external_data_dir: str | None,
    entry_grid: tuple[float, ...],
    exit_grid: tuple[float, ...],
    baseline_entry: float,
    baseline_exit: float,
    balance: float,
    write_meta: bool,
    output_dir: str,
    label: str,
    config_dump: dict | None = None,
) -> ThresholdTunePairResult:
    """Tune (entry, exit) thresholds for a single (symbol, timeframe) pair.

    Top-level function for spawn-safe pickling. Mirrors the pattern in
    ``train_pair``: defer heavy imports, silence logging in the child, run
    the work inside a broad try/except so one failed pair never kills the
    parent batch. Each worker writes its own per-model JSON to ``output_dir``
    so the parent only needs the lightweight summary fields.

    ``config_dump`` is the parent's resolved ``AppConfig.model_dump()`` so
    YAML overrides (sizing, fees, slippage) flow into the worker; without it
    the worker would silently use bare defaults and produce different
    backtests than the sequential path.
    """
    import json
    import logging

    import structlog

    from tradingbot.config import AppConfig
    from tradingbot.data.storage import load_candles
    from tradingbot.ml.threshold_tuner import ThresholdTuner, patch_meta_thresholds

    logging.getLogger().setLevel(logging.CRITICAL)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
    )

    try:
        df = load_candles(symbol, timeframe, Path(data_dir))
    except FileNotFoundError:
        return ThresholdTunePairResult(
            symbol=symbol,
            timeframe=timeframe,
            error="no_data",
        )
    except Exception as exc:
        return ThresholdTunePairResult(
            symbol=symbol,
            timeframe=timeframe,
            error=f"load_candles: {exc}",
        )

    try:
        if config_dump is not None:
            config = AppConfig.model_validate(config_dump)
        else:
            config = AppConfig()
        config.trading.initial_balance = balance

        ext_dir = Path(external_data_dir) if external_data_dir else None

        tuner = ThresholdTuner(
            symbol=symbol,
            timeframe=timeframe,
            model_dir=Path(model_dir),
            external_data_dir=ext_dir,
            config=config,
            balance=balance,
            baseline_entry=baseline_entry,
            baseline_exit=baseline_exit,
        )
        result = tuner.search(df, entry_grid=entry_grid, exit_grid=exit_grid)

        if result.error and not result.grid:
            return ThresholdTunePairResult(
                symbol=symbol,
                timeframe=timeframe,
                holdout_start=result.holdout_start,
                holdout_end=result.holdout_end,
                error=result.error,
            )

        meta_path: Path | None = None
        if write_meta:
            try:
                meta_path = patch_meta_thresholds(
                    symbol, timeframe, Path(model_dir), result
                )
            except Exception as exc:
                return ThresholdTunePairResult(
                    symbol=symbol,
                    timeframe=timeframe,
                    holdout_start=result.holdout_start,
                    holdout_end=result.holdout_end,
                    error=f"patch_meta: {exc}",
                )

        # Per-model JSON written here so the parent doesn't need the full
        # grid back across the pickle boundary. Filename matches the
        # ml-tune-thresholds layout so users can drill into any one result.
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = f"{label}_{symbol.replace('/', '_')}_{timeframe}"
        try:
            (out_dir / f"{base}.json").write_text(
                json.dumps(
                    {
                        "symbol": result.symbol,
                        "timeframe": result.timeframe,
                        "holdout_start": result.holdout_start,
                        "holdout_end": result.holdout_end,
                        "best_entry": result.best_entry,
                        "best_exit": result.best_exit,
                        "best_sharpe": result.best_sharpe,
                        "best_return_pct": result.best_return_pct,
                        "best_trades": result.best_trades,
                        "best_win_rate": result.best_win_rate,
                        "best_max_dd_pct": result.best_max_dd_pct,
                        "baseline_entry": result.baseline_entry,
                        "baseline_exit": result.baseline_exit,
                        "baseline_sharpe": result.baseline_sharpe,
                        "baseline_return_pct": result.baseline_return_pct,
                        "baseline_trades": result.baseline_trades,
                        "n_combos_evaluated": result.n_combos_evaluated,
                        "n_combos_skipped": result.n_combos_skipped,
                        "entry_grid": list(entry_grid),
                        "exit_grid": list(exit_grid),
                        "grid": result.grid,
                        "meta_path": str(meta_path) if meta_path else None,
                        "error": result.error,
                    },
                    indent=2,
                    default=str,
                )
            )
        except OSError as exc:
            return ThresholdTunePairResult(
                symbol=symbol,
                timeframe=timeframe,
                holdout_start=result.holdout_start,
                holdout_end=result.holdout_end,
                error=f"write_report: {exc}",
            )

        return ThresholdTunePairResult(
            symbol=symbol,
            timeframe=timeframe,
            best_entry=float(result.best_entry),
            best_exit=float(result.best_exit),
            best_sharpe=float(result.best_sharpe),
            best_return_pct=float(result.best_return_pct),
            best_trades=int(result.best_trades),
            best_win_rate=float(result.best_win_rate),
            best_max_dd_pct=float(result.best_max_dd_pct),
            baseline_sharpe=float(result.baseline_sharpe),
            baseline_return_pct=float(result.baseline_return_pct),
            baseline_trades=int(result.baseline_trades),
            n_combos_evaluated=int(result.n_combos_evaluated),
            holdout_start=result.holdout_start,
            holdout_end=result.holdout_end,
            meta_patched=meta_path is not None,
            error=result.error,
        )
    except Exception as exc:
        return ThresholdTunePairResult(
            symbol=symbol,
            timeframe=timeframe,
            error=str(exc),
        )
