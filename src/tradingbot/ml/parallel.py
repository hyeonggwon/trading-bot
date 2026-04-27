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
class TunePairResult:
    """Result from Optuna hyperparameter tuning for one (symbol, timeframe) pair."""

    symbol: str
    timeframe: str
    best_value: float = float("-inf")
    n_trials_completed: int = 0
    n_trials_pruned: int = 0
    elapsed_sec: float = 0.0
    objective: str = "holdout_sharpe"
    best_params: dict = None  # type: ignore[assignment]
    final_holdout_auc: float | None = None
    final_holdout_precision: float | None = None
    final_model_path: str = ""
    error: str | None = None

    def __post_init__(self):
        if self.best_params is None:
            self.best_params = {}


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


def tune_pair(
    symbol: str,
    timeframe: str,
    data_dir: str,
    model_dir: str,
    external_data_dir: str | None,
    train_months: int,
    test_months: int,
    forward_candles: int,
    threshold: float,
    target_kind: str,
    atr_mult: float,
    include_extra: bool,
    entry_threshold: float,
    exit_threshold: float,
    balance: float,
    trials: int,
    time_budget_sec: float,
    objective: str,
    seed: int,
    output_dir: str,
    label: str,
    num_threads: int,
    config_dump: dict | None = None,
) -> TunePairResult:
    """Run an Optuna LGBM search + final-model train for one (symbol, timeframe).

    Top-level for spawn-safe pickling. Mirrors the body of the ``ml-tune``
    CLI command but writes its own per-model JSON+MD report inside the
    worker so the parent only ferries summary fields back across the
    pickle boundary. ``num_threads`` clamps OpenMP/LightGBM thread usage
    so multiple workers don't oversubscribe the host CPU.
    """
    import json
    import logging
    import os
    import time

    import structlog

    from tradingbot.config import AppConfig
    from tradingbot.data.external_fetcher import build_external_df
    from tradingbot.data.storage import load_candles
    from tradingbot.ml.tuner import LGBMTuner
    from tradingbot.ml.walk_forward import MLWalkForwardTrainer

    # LightGBM honours OMP_NUM_THREADS for its OpenMP parallelism. Set it
    # before any LightGBM import paths take effect — otherwise N workers
    # × cpu_count threads each thrash the box.
    os.environ["OMP_NUM_THREADS"] = str(max(1, num_threads))
    os.environ["MKL_NUM_THREADS"] = str(max(1, num_threads))

    logging.getLogger().setLevel(logging.CRITICAL)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
    )

    try:
        df = load_candles(symbol, timeframe, Path(data_dir))
    except FileNotFoundError:
        return TunePairResult(
            symbol=symbol, timeframe=timeframe, objective=objective, error="no_data"
        )
    except Exception as exc:
        return TunePairResult(
            symbol=symbol,
            timeframe=timeframe,
            objective=objective,
            error=f"load_candles: {exc}",
        )

    try:
        if config_dump is not None:
            config = AppConfig.model_validate(config_dump)
        else:
            config = AppConfig()
        config.trading.initial_balance = balance

        ext_dir = Path(external_data_dir) if external_data_dir else None
        external_df = build_external_df(df, ext_dir) if ext_dir else None
        has_external = external_df is not None and len(external_df.columns) > 0
        ext_dir_for_runner = ext_dir if has_external else None

        start = time.time()
        tuner = LGBMTuner(
            symbol=symbol,
            timeframe=timeframe,
            train_months=train_months,
            test_months=test_months,
            forward_candles=forward_candles,
            threshold=threshold,
            target_kind=target_kind,
            atr_mult=atr_mult,
            include_extra=include_extra,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            balance=balance,
            external_data_dir=ext_dir_for_runner,
            config=config,
            objective=objective,
            seed=seed,
        )
        result = tuner.search(df, n_trials=trials, time_budget_sec=time_budget_sec)

        if not result.best_params:
            return TunePairResult(
                symbol=symbol,
                timeframe=timeframe,
                objective=objective,
                n_trials_completed=result.n_trials_completed,
                elapsed_sec=time.time() - start,
                error="no_successful_trial",
            )

        # Train final model with best params so downstream commands (scan,
        # combine-scan, ml-backtest) pick up the tuned booster.
        final_lgbm_params = dict(result.best_params)
        # Force num_threads on the final fit too — otherwise the saved model
        # logs warn-level "Number of positive: ..." spam from background
        # threads when the parent later loads it.
        final_lgbm_params.setdefault("num_threads", max(1, num_threads))

        final_trainer = MLWalkForwardTrainer(
            symbol=symbol,
            timeframe=timeframe,
            train_months=train_months,
            test_months=test_months,
            forward_candles=forward_candles,
            threshold=threshold,
            target_kind=target_kind,
            atr_mult=atr_mult,
            include_extra=include_extra,
            model_dir=Path(model_dir),
            lgbm_params=final_lgbm_params,
        )
        try:
            final_report = final_trainer.run(df, external_df=external_df)
        except Exception as exc:
            return TunePairResult(
                symbol=symbol,
                timeframe=timeframe,
                objective=objective,
                best_value=float(result.best_value),
                best_params=dict(result.best_params),
                n_trials_completed=result.n_trials_completed,
                n_trials_pruned=result.n_trials_pruned,
                elapsed_sec=time.time() - start,
                error=f"final_train: {exc}",
            )

        final_holdout_auc = None
        final_holdout_precision = None
        final_model_path: Path | None = None
        if final_report.windows:
            final_holdout_auc = float(final_report.holdout_auc)
            final_holdout_precision = float(final_report.holdout_precision)
            final_model_path = final_report.model_path

            # Patch tuning info into the saved meta so users can audit which
            # params produced this booster (mirrors ml-tune's behaviour).
            if final_model_path is not None:
                symbol_key = symbol.replace("/", "_")
                meta_path = Path(model_dir) / f"lgbm_{symbol_key}_{timeframe}_meta.json"
                if meta_path.exists():
                    try:
                        meta_dict = json.loads(meta_path.read_text())
                        meta_dict["tuning"] = {
                            "best_params": dict(result.best_params),
                            "best_value": result.best_value,
                            "objective": objective,
                            "n_trials_completed": result.n_trials_completed,
                            "elapsed_sec": result.elapsed_sec,
                        }
                        tmp_path = meta_path.with_suffix(".json.tmp")
                        tmp_path.write_text(
                            json.dumps(meta_dict, indent=2, default=str)
                        )
                        os.replace(tmp_path, meta_path)
                    except (json.JSONDecodeError, OSError):
                        # Meta corruption shouldn't kill the run; users can
                        # inspect the per-model JSON instead.
                        pass

        # Per-model JSON+MD reports (same layout as ``ml-tune``).
        out_dir = Path(output_dir)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            base = f"{label}_{symbol.replace('/', '_')}_{timeframe}"
            top_trials = sorted(
                [t for t in result.trials if t.get("score", float("-inf")) != float("-inf")],
                key=lambda t: -t["score"],
            )[:5]
            payload = {
                "symbol": symbol,
                "timeframe": timeframe,
                "train_months": train_months,
                "test_months": test_months,
                "target_kind": target_kind,
                "atr_mult": atr_mult,
                "threshold": threshold,
                "entry_threshold": entry_threshold,
                "exit_threshold": exit_threshold,
                "initial_balance": balance,
                "objective": objective,
                "trials_requested": trials,
                "time_budget_sec": time_budget_sec,
                "n_trials_completed": result.n_trials_completed,
                "elapsed_sec": result.elapsed_sec,
                "best_params": result.best_params,
                "best_value": result.best_value,
                "trials": result.trials,
                "final_holdout_auc": final_holdout_auc,
                "final_holdout_precision": final_holdout_precision,
                "final_model_path": str(final_model_path) if final_model_path else None,
            }
            (out_dir / f"{base}.json").write_text(
                json.dumps(payload, indent=2, default=str)
            )

            md_lines = [
                f"# ML Tuning — {symbol} {timeframe} ({label})",
                "",
                f"- Walk-Forward: {train_months}m train / {test_months}m test",
                (
                    f"- Target: binary (forward_candles={forward_candles}, threshold={threshold})"
                    if target_kind == "binary"
                    else f"- Target: {target_kind} (forward_candles={forward_candles}, atr_mult={atr_mult})"
                ),
                f"- Objective: **{objective}**",
                f"- Trials: {result.n_trials_completed} / {trials} requested",
                f"- Elapsed: {result.elapsed_sec:.1f}s",
                f"- Best **{objective}**: **{result.best_value:.4f}**",
                "",
                "## Best params",
                "",
                "| Param | Value |",
                "|-------|------:|",
            ]
            for k, v in result.best_params.items():
                if isinstance(v, float):
                    md_lines.append(f"| {k} | {v:.4f} |")
                else:
                    md_lines.append(f"| {k} | {v} |")
            if top_trials:
                md_lines += [
                    "",
                    "## Top trials",
                    "",
                    "| # | Score | Sharpe | Cum % | Trades |",
                    "|---|------:|-------:|------:|-------:|",
                ]
                for t in top_trials:
                    md_lines.append(
                        f"| {t['trial']} | {t['score']:.3f} | "
                        f"{t.get('avg_sharpe', 0):.2f} | "
                        f"{t.get('cumulative_return_pct', 0):+.2f} | "
                        f"{t.get('total_trades', 0)} |"
                    )
            if final_report.windows:
                md_lines += [
                    "",
                    "## Final model (trained with best params)",
                    "",
                    f"- Holdout AUC: {final_holdout_auc:.4f}",
                    f"- Holdout Precision: {final_holdout_precision:.4f}",
                    f"- Saved: `{final_model_path}`",
                ]
            (out_dir / f"{base}.md").write_text("\n".join(md_lines) + "\n")
        except OSError as exc:
            return TunePairResult(
                symbol=symbol,
                timeframe=timeframe,
                objective=objective,
                best_value=float(result.best_value),
                best_params=dict(result.best_params),
                n_trials_completed=result.n_trials_completed,
                n_trials_pruned=result.n_trials_pruned,
                elapsed_sec=time.time() - start,
                final_holdout_auc=final_holdout_auc,
                final_holdout_precision=final_holdout_precision,
                final_model_path=str(final_model_path) if final_model_path else "",
                error=f"write_report: {exc}",
            )

        return TunePairResult(
            symbol=symbol,
            timeframe=timeframe,
            best_value=float(result.best_value),
            n_trials_completed=result.n_trials_completed,
            n_trials_pruned=result.n_trials_pruned,
            elapsed_sec=time.time() - start,
            objective=objective,
            best_params=dict(result.best_params),
            final_holdout_auc=final_holdout_auc,
            final_holdout_precision=final_holdout_precision,
            final_model_path=str(final_model_path) if final_model_path else "",
        )
    except Exception as exc:
        return TunePairResult(
            symbol=symbol,
            timeframe=timeframe,
            objective=objective,
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
