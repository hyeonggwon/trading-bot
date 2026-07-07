"""Backtest commands: backtest, optimize, walk-forward, scan."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from rich.table import Table

from tradingbot.cli._shared import (
    STRATEGY_MAP,
    _load_strategies,
    _progress_context,
    _resolve_holdout_window,
    _validate_date_range,
    _write_scan_markdown_report,
    app,
    console,
)
from tradingbot.cli.combine import _resolve_strategy
from tradingbot.config import load_config
from tradingbot.utils.logging import setup_logging

if TYPE_CHECKING:
    import pandas as pd

    from tradingbot.config import AppConfig
    from tradingbot.strategy.base import Strategy


@app.command()
def backtest(
    strategy_name: str = typer.Option("sma_cross", "--strategy", "-S", help="Strategy name"),
    symbol: str = typer.Option(
        None, "--symbol", "-s", help="Trading pair (omit for all config symbols)"
    ),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Candle timeframe"),
    start: str = typer.Option(None, "--start", help="Override evaluation start (YYYY-MM-DD)"),
    end: str = typer.Option(None, "--end", help="Override evaluation end (YYYY-MM-DD)"),
    balance: float = typer.Option(1_000_000, "--balance", "-b", help="Initial balance (KRW)"),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
    include_train: bool = typer.Option(
        False,
        "--include-train",
        help="Disable holdout-only filtering and evaluate on the full data range.",
    ),
) -> None:
    """Run a backtest on historical data. Supports single or multiple symbols.

    By default the strategy is evaluated only on the data's last 20% so the
    result is comparable to ``ml-backtest`` (which uses the model's recorded
    holdout window). Pass ``--include-train`` to evaluate the full data range
    or use ``--start``/``--end`` to set an explicit window.
    """
    setup_logging()
    _validate_date_range(start, end)

    from tradingbot.backtest.engine import BacktestEngine
    from tradingbot.data.storage import load_candles

    # Determine symbols: CLI override or config default
    if symbol:
        symbols = [symbol]
    else:
        base_config = load_config(Path("config"))
        symbols = base_config.trading.symbols

    if not symbols:
        console.print("[red]No symbols found in config or --symbol flag.[/red]")
        raise typer.Exit(1)

    strategy, strategy_name, _ = _resolve_strategy(
        strategy_name,
        symbols[0],
        timeframe,
        symbols=symbols,
    )

    # Load data for all symbols
    data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            data[sym] = load_candles(sym, timeframe, Path(data_dir))
            console.print(f"  {sym}: {len(data[sym])} candles")
        except FileNotFoundError:
            console.print(f"  [yellow]{sym}: no data (skipped)[/yellow]")

    if not data:
        console.print("[red]No data available for any symbol.[/red]")
        raise typer.Exit(1)

    # Resolve holdout window AFTER data is loaded (need timestamps for auto cutoff).
    effective_start, effective_end, period_note = _resolve_holdout_window(
        data,
        start,
        end,
        include_train,
    )
    config = load_config(
        Path("config"),
        overrides={
            "trading": {"symbols": symbols, "timeframe": timeframe, "initial_balance": balance},
            "backtest": {"start_date": effective_start, "end_date": effective_end},
        },
    )

    console.print(
        f"[bold]Running backtest: {strategy_name} on {len(data)} symbol(s) {timeframe}[/bold]"
    )
    eval_start_str = effective_start or "data start"
    eval_end_str = effective_end or "data end"
    console.print(f"  Evaluation period: {eval_start_str} → {eval_end_str} ({period_note})")

    engine = BacktestEngine(strategy=strategy, config=config)
    report = engine.run(data)
    report.print_summary()


@app.command()
def optimize(
    strategy_name: str = typer.Option("sma_cross", "--strategy", "-S", help="Strategy name"),
    symbol: str = typer.Option("BTC/KRW", "--symbol", "-s", help="Trading pair"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Candle timeframe"),
    balance: float = typer.Option(1_000_000, "--balance", "-b", help="Initial balance (KRW)"),
    sort_by: str = typer.Option("sharpe_ratio", "--sort-by", help="Metric to sort by"),
    top_n: int = typer.Option(10, "--top", help="Show top N results"),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
    param_grid: str = typer.Option(None, "--param-grid", help="JSON param grid override"),
) -> None:
    """Optimize strategy parameters via grid search."""
    import json

    setup_logging()

    _, strategy_name, strategy_cls = _resolve_strategy(
        strategy_name,
        symbol,
        timeframe,
    )
    if strategy_cls is None:
        console.print(
            "[red]Combined templates cannot be optimized (no param_space). "
            "Use backtest instead.[/red]"
        )
        raise typer.Exit(1)

    from tradingbot.backtest.optimizer import GridSearchOptimizer
    from tradingbot.data.storage import load_candles

    config = load_config(
        Path("config"),
        overrides={
            "trading": {"symbols": [symbol], "timeframe": timeframe, "initial_balance": balance},
        },
    )
    space = None
    if param_grid:
        try:
            space = json.loads(param_grid)
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON in --param-grid: {e}[/red]")
            raise typer.Exit(1)

    df = load_candles(symbol, timeframe, Path(data_dir))
    console.print(f"[bold]Optimizing {strategy_name} on {symbol} ({len(df)} candles)[/bold]")

    optimizer = GridSearchOptimizer(strategy_cls=strategy_cls, config=config, max_workers=1)
    with _progress_context() as progress:
        results = optimizer.optimize(
            {symbol: df},
            param_space=space,
            sort_by=sort_by,
            progress=progress,
        )
    optimizer.print_results(results, top_n=top_n)


@app.command()
def walk_forward(
    strategy_name: str = typer.Option("sma_cross", "--strategy", "-S", help="Strategy name"),
    symbol: str = typer.Option("BTC/KRW", "--symbol", "-s", help="Trading pair"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Candle timeframe"),
    balance: float = typer.Option(1_000_000, "--balance", "-b", help="Initial balance (KRW)"),
    train_months: int = typer.Option(3, "--train-months", help="Training window (months)"),
    test_months: int = typer.Option(1, "--test-months", help="Test window (months)"),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
) -> None:
    """Run walk-forward validation."""
    setup_logging()

    strategy, strategy_name, strategy_cls = _resolve_strategy(
        strategy_name,
        symbol,
        timeframe,
    )

    from tradingbot.data.storage import load_candles

    config = load_config(
        Path("config"),
        overrides={
            "trading": {"symbols": [symbol], "timeframe": timeframe, "initial_balance": balance},
        },
    )

    df = load_candles(symbol, timeframe, Path(data_dir))
    console.print(
        f"[bold]Walk-forward: {strategy_name} on {symbol} "
        f"(train={train_months}m, test={test_months}m)[/bold]"
    )

    if strategy_cls is not None:
        # Registered strategy: optimize params per window
        from tradingbot.backtest.walk_forward import WalkForwardValidator

        validator = WalkForwardValidator(
            strategy_cls=strategy_cls,
            config=config,
            train_months=train_months,
            test_months=test_months,
        )
        with _progress_context() as progress:
            report = validator.validate({symbol: df}, progress=progress)
        report.print_summary()
    else:
        # Combined template: fixed filters, no optimization — test each window
        _walk_forward_combined(
            strategy,
            strategy_name,
            symbol,
            df,
            config,
            train_months,
            test_months,
        )


def _walk_forward_combined(
    strategy: Strategy,
    strategy_name: str,
    symbol: str,
    df: pd.DataFrame,
    config: AppConfig,
    train_months: int,
    test_months: int,
) -> None:
    """Walk-forward for combined strategies (no param optimization)."""
    import copy

    from tradingbot.backtest.engine import BacktestEngine
    from tradingbot.backtest.report import BacktestReport
    from tradingbot.backtest.walk_forward import (
        WalkForwardReport,
        WalkForwardWindow,
        create_walk_forward_windows,
    )

    # Warmup buffer: enough for the most demanding indicators
    # (e.g., trend_up:4 with SMA_50 at 4x = 200 bars, plus margin)
    warmup_bars = 300

    wf_config = config.model_copy(deep=True)
    wf_config.backtest.start_date = None
    wf_config.backtest.end_date = None

    windows = create_walk_forward_windows(df, train_months, test_months)
    if not windows:
        console.print("[red]Insufficient data for walk-forward windows.[/red]")
        return

    results: list[WalkForwardWindow] = []

    with _progress_context() as progress:
        task = progress.add_task("Walk-Forward (combined)", total=len(windows))

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            progress.update(
                task,
                description=f"WF {i + 1}/{len(windows)}: {train_start.date()}~{test_end.date()}",
            )

            # Train window — include warmup buffer for indicator computation
            train_start_idx = df.index.searchsorted(train_start)
            train_warmup_idx = max(0, train_start_idx - warmup_bars)
            train_with_warmup = df.iloc[train_warmup_idx:].copy()
            train_with_warmup = train_with_warmup[train_with_warmup.index < train_end]

            train_strategy = copy.deepcopy(strategy)
            engine = BacktestEngine(strategy=train_strategy, config=wf_config)
            full_train_report = engine.run({symbol: train_with_warmup})

            # Filter to train period only
            train_start_dt = (
                train_start.to_pydatetime()
                if hasattr(train_start, "to_pydatetime")
                else train_start
            )
            train_trades = [
                t
                for t in full_train_report.trades
                if t.entry_order.created_at is not None
                and t.entry_order.created_at >= train_start_dt
            ]
            train_equity = full_train_report.equity_curve[
                full_train_report.equity_curve.index >= train_start
            ]

            if len(train_equity) < 2:
                train_report = full_train_report
            else:
                train_report = BacktestReport(
                    trades=train_trades,
                    equity_curve=train_equity,
                    initial_balance=float(train_equity.iloc[0]),
                    final_balance=float(train_equity.iloc[-1]),
                    timeframe=wf_config.trading.timeframe,
                )

            # Test window — include warmup buffer for indicator computation
            test_start_idx = df.index.searchsorted(test_start)
            warmup_idx = max(0, test_start_idx - warmup_bars)
            test_with_warmup = df.iloc[warmup_idx:].copy()
            test_with_warmup = test_with_warmup[test_with_warmup.index < test_end]

            test_strategy = copy.deepcopy(strategy)
            engine = BacktestEngine(strategy=test_strategy, config=wf_config)
            full_report = engine.run({symbol: test_with_warmup})

            # Filter to test period only (exclude warmup trades)
            test_start_dt = (
                test_start.to_pydatetime() if hasattr(test_start, "to_pydatetime") else test_start
            )
            test_trades = [
                t
                for t in full_report.trades
                if t.entry_order.created_at is not None
                and t.entry_order.created_at >= test_start_dt
            ]
            test_equity = full_report.equity_curve[full_report.equity_curve.index >= test_start]

            if len(test_equity) < 2:
                test_sharpe = 0.0
                test_return_val = 0.0
                test_dd = 0.0
                test_trade_count = 0
            else:
                filtered_report = BacktestReport(
                    trades=test_trades,
                    equity_curve=test_equity,
                    initial_balance=float(test_equity.iloc[0]),
                    final_balance=float(test_equity.iloc[-1]),
                    timeframe=wf_config.trading.timeframe,
                )
                test_sharpe = filtered_report.sharpe_ratio
                test_return_val = filtered_report.total_return
                test_dd = filtered_report.max_drawdown
                test_trade_count = filtered_report.total_trades

            results.append(
                WalkForwardWindow(
                    window_index=i,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    best_params={"filters": "fixed"},
                    train_sharpe=train_report.sharpe_ratio,
                    train_return=train_report.total_return,
                    test_sharpe=test_sharpe,
                    test_return=test_return_val,
                    test_trades=test_trade_count,
                    test_max_drawdown=test_dd,
                )
            )

            progress.advance(task)

    report = WalkForwardReport(windows=results, strategy_name=strategy_name)
    report.print_summary()


@app.command()
def scan(
    top_n: int = typer.Option(10, "--top", help="Show top N results"),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
    balance: float = typer.Option(1_000_000, "--balance", "-b", help="Initial balance (KRW)"),
    sort_by: str = typer.Option("sharpe_ratio", "--sort-by", help="Sort metric"),
    workers: int = typer.Option(0, "--workers", "-w", help="Parallel workers (0=auto)"),
    start: str = typer.Option(None, "--start", help="Override evaluation start date (YYYY-MM-DD)"),
    end: str = typer.Option(None, "--end", help="Override evaluation end date (YYYY-MM-DD)"),
    include_train: bool = typer.Option(
        False,
        "--include-train",
        help="Disable per-batch holdout filtering and scan over the full data range.",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        help=(
            "Write the Top-N table as markdown to this path (e.g. personal/scan_holdout_result.md)."
        ),
    ),
) -> None:
    """Scan all strategy × timeframe × symbol combinations to find the best.

    By default each (symbol, timeframe) batch is evaluated only on its
    last 20% — the same policy as ``backtest`` / ``combine`` so a scan
    result and a follow-up single run report on the same window.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    setup_logging()
    _load_strategies()
    _validate_date_range(start, end)

    valid_metrics = {
        "sharpe_ratio",
        "total_return",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "total_trades",
    }
    if sort_by not in valid_metrics:
        console.print(f"[red]Invalid sort metric: {sort_by}[/red]")
        console.print(f"Available: {', '.join(sorted(valid_metrics))}")
        raise typer.Exit(1)

    from tradingbot.data.storage import list_available_data

    available = list_available_data(Path(data_dir))
    if not available:
        console.print("[red]No data found. Run tradingbot download first.[/red]")
        raise typer.Exit(1)

    symbol_timeframes: dict[str, list[str]] = {}
    for item in available:
        symbol_timeframes.setdefault(item["symbol"], []).append(item["timeframe"])

    strategies = list(STRATEGY_MAP.keys())
    results: list[dict[str, Any]] = []
    failures: list[str] = []

    # Build batched jobs: group by (symbol, timeframe) to load data once
    batches: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    total = 0
    for sym, timeframes in symbol_timeframes.items():
        for tf in timeframes:
            batch_jobs = [(strat_name, "", "") for strat_name in strategies]
            batches[(sym, tf)] = batch_jobs
            total += len(batch_jobs)

    cpu = multiprocessing.cpu_count() or 4
    n_workers = workers if workers > 0 else min(cpu, 8)
    abs_data_dir = str(Path(data_dir).resolve())
    abs_config_dir = str(Path("config").resolve())
    if start or end:
        range_note = f" [{start or 'start'} → {end or 'end'}]"
    elif include_train:
        range_note = " [full data range (--include-train)]"
    else:
        range_note = " [auto holdout: last 20% per batch]"
    console.print(
        f"[bold]Scanning {len(strategies)} strategies × {len(symbol_timeframes)} symbols "
        f"× timeframes ({total} combinations, {n_workers} workers, "
        f"{len(batches)} batches){range_note}...[/bold]"
    )

    from tradingbot.backtest.parallel import _run_batch

    with _progress_context() as progress:
        task = progress.add_task("Scanning strategies", total=total)

        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = {
                pool.submit(
                    _run_batch,
                    sym,
                    tf,
                    batch_jobs,
                    abs_data_dir,
                    balance,
                    abs_config_dir,
                    False,
                    start,
                    end,
                    include_train,
                ): (sym, tf)
                for (sym, tf), batch_jobs in batches.items()
            }
            for future in as_completed(futures):
                sym, tf = futures[future]
                progress.update(task, description=f"{sym} {tf}")
                try:
                    batch_results = future.result(timeout=600)
                except Exception as exc:
                    failures.append(f"{sym}/{tf}: worker crashed: {exc}")
                    progress.advance(task, advance=len(batches[(sym, tf)]))
                    continue
                for res in batch_results:
                    if res.error:
                        failures.append(f"{res.strategy}/{res.symbol}/{res.timeframe}: {res.error}")
                    else:
                        results.append(
                            {
                                "strategy": res.strategy,
                                "symbol": res.symbol,
                                "timeframe": res.timeframe,
                                "sharpe_ratio": res.sharpe_ratio,
                                "total_return": res.total_return,
                                "max_drawdown": res.max_drawdown,
                                "win_rate": res.win_rate,
                                "profit_factor": res.profit_factor,
                                "total_trades": res.total_trades,
                            }
                        )
                    progress.advance(task)
    if failures:
        console.print(f"[yellow]{len(failures)} combinations failed:[/yellow]")
        for f in failures[:5]:
            console.print(f"  {f}")
        if len(failures) > 5:
            console.print(f"  ... and {len(failures) - 5} more")

    if not results:
        console.print("[red]No results.[/red]")
        raise typer.Exit(1)

    # Sort results
    reverse = sort_by != "max_drawdown"
    results.sort(key=lambda r: r.get(sort_by, 0), reverse=reverse)

    # Display top N
    table = Table(title=f"Best Combinations (Top {min(top_n, len(results))})")
    table.add_column("#", justify="right")
    table.add_column("Strategy")
    table.add_column("Symbol")
    table.add_column("TF")
    table.add_column("Sharpe", justify="right")
    table.add_column("Return", justify="right")
    table.add_column("MaxDD", justify="right")
    table.add_column("Win%", justify="right")
    table.add_column("PF", justify="right")
    table.add_column("Trades", justify="right")

    for i, r in enumerate(results[:top_n], 1):
        sharpe_style = (
            "green" if r["sharpe_ratio"] > 1.0 else ("yellow" if r["sharpe_ratio"] > 0 else "red")
        )
        table.add_row(
            str(i),
            r["strategy"],
            r["symbol"],
            r["timeframe"],
            f"[{sharpe_style}]{r['sharpe_ratio']:.2f}[/{sharpe_style}]",
            f"{r['total_return']:.2%}",
            f"{r['max_drawdown']:.2%}",
            f"{r['win_rate']:.1%}",
            f"{r['profit_factor']:.2f}",
            str(r["total_trades"]),
        )

    console.print(table)

    if output:
        from datetime import UTC, datetime

        if start or end:
            range_md = f"{start or 'start'} → {end or 'end'}"
        elif include_train:
            range_md = "full data range (--include-train)"
        else:
            range_md = "**각 (symbol, timeframe) 배치의 마지막 20%** (auto holdout)"
        md_rows = [
            [
                str(i),
                r["strategy"],
                r["symbol"],
                r["timeframe"],
                f"{r['sharpe_ratio']:.2f}",
                f"{r['total_return']:.2%}",
                f"{r['max_drawdown']:.2%}",
                f"{r['win_rate']:.1%}",
                f"{r['profit_factor']:.2f}",
                str(r["total_trades"]),
            ]
            for i, r in enumerate(results[:top_n], 1)
        ]
        n_top = min(top_n, len(results))
        out_path = Path(output)
        _write_scan_markdown_report(
            output=out_path,
            title=f"Scan Result — Top {n_top}",
            metadata=[
                f"일시: {datetime.now(UTC).strftime('%Y-%m-%d')}",
                f"대상: {len(strategies)} strategies × {len(symbol_timeframes)} symbols × "
                f"timeframes ({total} combinations)",
                f"Workers: {n_workers}",
                f"평가 기간: {range_md}",
                f"정렬: {sort_by}",
            ],
            section=f"Best Combinations (Top {n_top})",
            columns=[
                "#",
                "Strategy",
                "Symbol",
                "TF",
                "Sharpe",
                "Return",
                "MaxDD",
                "Win%",
                "PF",
                "Trades",
            ],
            rows=md_rows,
        )
        console.print(f"[green]Wrote {n_top} rows to {out_path}[/green]")
