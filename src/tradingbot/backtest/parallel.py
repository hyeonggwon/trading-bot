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


def _run_backtest(
    strategy_name: str,
    symbol: str,
    timeframe: str,
    data_dir: str,
    balance: float,
    entry: str = "",
    exit_: str = "",
    config_dir: str = "config",
) -> ScanResult:
    """Run a single backtest. Top-level function for spawn-safe pickling."""
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
        return ScanResult(
            strategy=strategy_name, symbol=symbol, timeframe=timeframe,
            sharpe_ratio=0, total_return=0, max_drawdown=0,
            win_rate=0, profit_factor=0, total_trades=0,
            entry=entry, exit=exit_, error="no data",
        )

    try:
        from tradingbot.backtest.engine import BacktestEngine

        config = load_config(Path(config_dir), overrides={
            "trading": {"symbols": [symbol], "timeframe": timeframe, "initial_balance": balance},
        })

        if entry:
            # Combined strategy mode
            from tradingbot.strategy.combined import CombinedStrategy
            from tradingbot.strategy.filters.registry import parse_filter_string

            entry_filters = parse_filter_string(entry, base_timeframe=timeframe)
            exit_filters = parse_filter_string(exit_, base_timeframe=timeframe)
            for f in entry_filters + exit_filters:
                if hasattr(f, "symbol"):
                    f.symbol = symbol
                if hasattr(f, "timeframe"):
                    f.timeframe = timeframe
            strategy = CombinedStrategy(entry_filters=entry_filters, exit_filters=exit_filters)
        else:
            # Registered strategy mode
            from tradingbot.strategy.registry import get_strategy_map
            strategy_cls = get_strategy_map()[strategy_name]
            strategy = strategy_cls()

        strategy.symbols = [symbol]
        strategy.timeframe = timeframe

        engine = BacktestEngine(strategy=strategy, config=config)
        report = engine.run({symbol: df})

        return ScanResult(
            strategy=strategy_name, symbol=symbol, timeframe=timeframe,
            sharpe_ratio=report.sharpe_ratio,
            total_return=report.total_return,
            max_drawdown=report.max_drawdown,
            win_rate=report.win_rate,
            profit_factor=report.profit_factor,
            total_trades=report.total_trades,
            entry=entry, exit=exit_,
        )
    except Exception as e:
        return ScanResult(
            strategy=strategy_name, symbol=symbol, timeframe=timeframe,
            sharpe_ratio=0, total_return=0, max_drawdown=0,
            win_rate=0, profit_factor=0, total_trades=0,
            entry=entry, exit=exit_, error=str(e),
        )
