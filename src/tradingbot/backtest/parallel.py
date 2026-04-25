"""Parallel backtest workers for scan/combine-scan (spawn-safe)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScanResult:
    """Result from a single backtest scan."""

    strategy: str
    symbol: str
    timeframe: str
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    # combine-scan extras
    entry: str = ""
    exit: str = ""
    error: str | None = None


def _run_batch(
    symbol: str,
    timeframe: str,
    jobs: list[tuple[str, str, str]],
    data_dir: str,
    balance: float,
    config_dir: str = "config",
    force_engine: bool = False,
    start: str | None = None,
    end: str | None = None,
    include_train: bool = False,
) -> list[ScanResult]:
    """Run a batch of backtests sharing the same (symbol, timeframe) data.

    Each job is (strategy_name, entry, exit_).
    Data is loaded once and reused for all jobs in the batch.
    Combined strategies use vectorized engine when possible (no ML filters).

    Evaluation window precedence:
    ``start``/``end`` > ``include_train`` (full range) > auto holdout
    (last 20% of this batch's data — same policy as the ``backtest`` /
    ``combine`` CLI commands so scan results match a follow-up single
    run on the same symbol/timeframe).

    The full engine path picks the resolved window up via
    ``config.backtest.start_date/end_date``; the vectorized path needs
    explicit slicing because ``vectorized_backtest`` does not consult
    config.
    """
    import logging

    import pandas as pd
    import structlog

    logging.getLogger().setLevel(logging.CRITICAL)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
    )

    from tradingbot.backtest.holdout import resolve_holdout_window
    from tradingbot.config import load_config
    from tradingbot.data.storage import load_candles

    try:
        df = load_candles(symbol, timeframe, Path(data_dir))
    except FileNotFoundError:
        return [
            ScanResult(
                strategy=name, symbol=symbol, timeframe=timeframe,
                sharpe_ratio=0, total_return=0, max_drawdown=0,
                win_rate=0, profit_factor=0, total_trades=0,
                entry=entry, exit=exit_, error="no data",
            )
            for name, entry, exit_ in jobs
        ]

    # Resolve evaluation window for this batch. None → keep edge.
    effective_start, effective_end, _note = resolve_holdout_window(
        df, start, end, include_train,
    )

    config = load_config(Path(config_dir), overrides={
        "trading": {"symbols": [symbol], "timeframe": timeframe, "initial_balance": balance},
        "backtest": {"start_date": effective_start, "end_date": effective_end},
    })

    # Empty-range short-circuit. Indicators are still computed on the full df
    # below so the eval window keeps proper warmup; this check just avoids
    # wasted work + surfaces a clean error when the user picks a range that
    # doesn't overlap the data.
    start_ts = pd.Timestamp(effective_start, tz="UTC") if effective_start is not None else None
    end_ts = pd.Timestamp(effective_end, tz="UTC") if effective_end is not None else None
    df_in_range = df
    if start_ts is not None:
        df_in_range = df_in_range[df_in_range.index >= start_ts]
    if end_ts is not None:
        df_in_range = df_in_range[df_in_range.index <= end_ts]
    if df_in_range.empty:
        return [
            ScanResult(
                strategy=name, symbol=symbol, timeframe=timeframe,
                sharpe_ratio=0, total_return=0, max_drawdown=0,
                win_rate=0, profit_factor=0, total_trades=0,
                entry=entry, exit=exit_, error="no data in range",
            )
            for name, entry, exit_ in jobs
        ]

    results: list[ScanResult] = []

    if force_engine:
        # Re-verification: all jobs go through full engine (honors config dates).
        results.extend(_run_engine_batch(
            df, symbol, timeframe, jobs, config, balance,
        ))
    else:
        # Split jobs: vectorizable combined vs fallback (registered strategies + ML)
        vectorizable_jobs = []
        fallback_jobs = []
        for name, entry, exit_ in jobs:
            if not entry:
                fallback_jobs.append((name, entry, exit_))  # registered strategy
            elif "lgbm_prob" in entry:
                fallback_jobs.append((name, entry, exit_))  # ML filter → fallback
            else:
                vectorizable_jobs.append((name, entry, exit_))

        # --- Vectorized path: combined templates without ML ---
        if vectorizable_jobs:
            results.extend(_run_vectorized_batch(
                df, symbol, timeframe, vectorizable_jobs, config, balance,
                start_ts=start_ts, end_ts=end_ts,
            ))

        # --- Fallback path: registered strategies + ML templates ---
        if fallback_jobs:
            results.extend(_run_engine_batch(
                df, symbol, timeframe, fallback_jobs, config, balance,
            ))

    return results


def _run_vectorized_batch(
    df, symbol, timeframe, jobs, config, balance,
    start_ts=None, end_ts=None,
) -> list[ScanResult]:
    """Run combined templates via vectorized engine.

    ``start_ts``/``end_ts`` (UTC pandas Timestamps) optionally restrict the
    evaluation window. Indicators are computed on the full ``df`` first so
    the first row of the eval window has a real value (warmup-correct);
    the indicator dataframe is then sliced by timestamp before backtesting.
    """
    from tradingbot.backtest.vectorized import vectorized_backtest
    from tradingbot.strategy.combined import CombinedStrategy
    from tradingbot.strategy.filters.registry import parse_filter_string

    # Parse filters once per job, reuse for both union and per-job backtest
    def _set_symbol_tf(filters):
        for f in filters:
            if hasattr(f, "symbol"):
                f.symbol = symbol
            if hasattr(f, "timeframe"):
                f.timeframe = timeframe

    parsed_jobs: list[tuple[str, list, list, str, str]] = []
    all_filters = []
    for name, entry, exit_ in jobs:
        entry_filters = parse_filter_string(entry, base_timeframe=timeframe)
        exit_filters = parse_filter_string(exit_, base_timeframe=timeframe)
        _set_symbol_tf(entry_filters + exit_filters)
        all_filters += entry_filters + exit_filters
        parsed_jobs.append((name, entry_filters, exit_filters, entry, exit_))

    # Compute union indicators on the full df (warmup-correct), then slice
    # by timestamp to the eval window before backtesting.
    union_strategy = CombinedStrategy(entry_filters=all_filters, exit_filters=[])
    union_strategy.symbols = [symbol]
    union_strategy.timeframe = timeframe
    indicator_df = union_strategy.indicators(df.copy())
    if start_ts is not None:
        indicator_df = indicator_df[indicator_df.index >= start_ts]
    if end_ts is not None:
        indicator_df = indicator_df[indicator_df.index <= end_ts]

    results: list[ScanResult] = []
    for name, entry_filters, exit_filters, entry, exit_ in parsed_jobs:
        try:
            result = vectorized_backtest(
                df=indicator_df,
                entry_filters=entry_filters,
                exit_filters=exit_filters,
                initial_balance=balance,
                fee_rate=config.backtest.fee_rate,
                slippage_pct=config.backtest.slippage_pct,
                stop_loss_pct=config.risk.default_stop_loss_pct,
                max_position_pct=config.risk.max_position_size_pct,
                timeframe=timeframe,
            )
            results.append(ScanResult(
                strategy=name, symbol=symbol, timeframe=timeframe,
                sharpe_ratio=result.sharpe_ratio,
                total_return=result.total_return,
                max_drawdown=result.max_drawdown,
                win_rate=result.win_rate,
                profit_factor=result.profit_factor,
                total_trades=result.total_trades,
                entry=entry, exit=exit_,
            ))
        except Exception as e:
            results.append(ScanResult(
                strategy=name, symbol=symbol, timeframe=timeframe,
                sharpe_ratio=0, total_return=0, max_drawdown=0,
                win_rate=0, profit_factor=0, total_trades=0,
                entry=entry, exit=exit_, error=str(e),
            ))

    return results


def _run_engine_batch(
    df, symbol, timeframe, jobs, config, balance,
) -> list[ScanResult]:
    """Run jobs via the full BacktestEngine (registered strategies + ML)."""
    from tradingbot.backtest.engine import BacktestEngine
    from tradingbot.strategy.combined import CombinedStrategy
    from tradingbot.strategy.filters.registry import parse_filter_string

    # Pre-compute shared indicators for combined strategies
    combined_jobs = [(n, e, x) for n, e, x in jobs if e]
    precomputed = None
    if combined_jobs:
        all_filters = []
        for _, entry, exit_ in combined_jobs:
            all_filters += parse_filter_string(entry, base_timeframe=timeframe)
            all_filters += parse_filter_string(exit_, base_timeframe=timeframe)
        for f in all_filters:
            if hasattr(f, "symbol"):
                f.symbol = symbol
            if hasattr(f, "timeframe"):
                f.timeframe = timeframe

        union_strategy = CombinedStrategy(entry_filters=all_filters, exit_filters=[])
        union_strategy.symbols = [symbol]
        union_strategy.timeframe = timeframe
        precomputed_df = union_strategy.indicators(df.copy())
        precomputed_df.values.flags.writeable = False
        precomputed = {symbol: precomputed_df}

    results: list[ScanResult] = []
    for strategy_name, entry, exit_ in jobs:
        try:
            if entry:
                entry_filters = parse_filter_string(entry, base_timeframe=timeframe)
                exit_filters = parse_filter_string(exit_, base_timeframe=timeframe)
                for f in entry_filters + exit_filters:
                    if hasattr(f, "symbol"):
                        f.symbol = symbol
                    if hasattr(f, "timeframe"):
                        f.timeframe = timeframe
                strategy = CombinedStrategy(entry_filters=entry_filters, exit_filters=exit_filters)
            else:
                from tradingbot.strategy.registry import get_strategy_map
                strategy_cls = get_strategy_map()[strategy_name]
                strategy = strategy_cls()

            strategy.symbols = [symbol]
            strategy.timeframe = timeframe

            engine = BacktestEngine(strategy=strategy, config=config)
            report = engine.run(
                {symbol: df.copy()},
                precomputed_indicators=precomputed if entry else None,
            )

            results.append(ScanResult(
                strategy=strategy_name, symbol=symbol, timeframe=timeframe,
                sharpe_ratio=report.sharpe_ratio,
                total_return=report.total_return,
                max_drawdown=report.max_drawdown,
                win_rate=report.win_rate,
                profit_factor=report.profit_factor,
                total_trades=report.total_trades,
                entry=entry, exit=exit_,
            ))
        except Exception as e:
            results.append(ScanResult(
                strategy=strategy_name, symbol=symbol, timeframe=timeframe,
                sharpe_ratio=0, total_return=0, max_drawdown=0,
                win_rate=0, profit_factor=0, total_trades=0,
                entry=entry, exit=exit_, error=str(e),
            ))

    return results
