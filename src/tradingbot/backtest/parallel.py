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
) -> list[ScanResult]:
    """Run a batch of backtests sharing the same (symbol, timeframe) data.

    Each job is (strategy_name, entry, exit_).
    Data is loaded once and reused for all jobs in the batch.
    Combined strategies use vectorized engine when possible (no ML filters).
    """
    import logging

    import structlog

    logging.getLogger().setLevel(logging.CRITICAL)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
    )

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

    config = load_config(Path(config_dir), overrides={
        "trading": {"symbols": [symbol], "timeframe": timeframe, "initial_balance": balance},
    })

    results: list[ScanResult] = []

    if force_engine:
        # Re-verification: all jobs go through full engine
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
            ))

        # --- Fallback path: registered strategies + ML templates ---
        if fallback_jobs:
            results.extend(_run_engine_batch(
                df, symbol, timeframe, fallback_jobs, config, balance,
            ))

    return results


def _run_vectorized_batch(
    df, symbol, timeframe, jobs, config, balance,
) -> list[ScanResult]:
    """Run combined templates via vectorized engine."""
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

    # Compute union indicators once for all vectorizable jobs
    union_strategy = CombinedStrategy(entry_filters=all_filters, exit_filters=[])
    union_strategy.symbols = [symbol]
    union_strategy.timeframe = timeframe
    indicator_df = union_strategy.indicators(df.copy())

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
