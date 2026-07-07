"""Filter-combination commands: combine, combine-scan, plus templates and strategy resolution."""

from __future__ import annotations

from pathlib import Path

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
from tradingbot.config import load_config
from tradingbot.utils.logging import setup_logging

# Predefined meaningful filter combination templates
COMBINE_TEMPLATES = [
    # Trend + timing
    {"entry": "trend_up:4 + rsi_oversold:30", "exit": "rsi_overbought:70", "label": "Trend+RSI"},
    {
        "entry": "trend_up:4 + rsi_oversold:35",
        "exit": "rsi_overbought:65",
        "label": "Trend+RSI(tight)",
    },
    {"entry": "trend_up:4 + rsi_oversold:30", "exit": "trend_up:4", "label": "Trend+RSI→TrendExit"},
    # Trend + volume
    {"entry": "trend_up:4 + volume_spike:2.5", "exit": "rsi_overbought:70", "label": "Trend+Vol"},
    {"entry": "trend_up:4 + volume_spike:2.0", "exit": "ema_above:20", "label": "Trend+Vol→EMA"},
    # Triple filter
    {
        "entry": "trend_up:4 + rsi_oversold:30 + volume_spike:2.0",
        "exit": "rsi_overbought:70",
        "label": "Triple",
    },
    {
        "entry": "trend_up:4 + rsi_oversold:35 + volume_spike:2.5",
        "exit": "trend_up:4",
        "label": "Triple(strict)",
    },
    # Momentum combos
    {"entry": "ema_above:50 + macd_cross_up", "exit": "rsi_overbought:70", "label": "EMA+MACD"},
    {"entry": "ema_above:20 + rsi_oversold:30", "exit": "rsi_overbought:70", "label": "EMA+RSI"},
    {
        "entry": "ema_above:50 + macd_cross_up + volume_spike:2.0",
        "exit": "rsi_overbought:70",
        "label": "EMA+MACD+Vol",
    },
    # Breakout combos
    {
        "entry": "volume_spike:2.5 + price_breakout:10",
        "exit": "ema_above:20",
        "label": "Vol+Breakout",
    },
    {"entry": "bb_upper_break:20 + volume_spike:2.0", "exit": "ema_above:20", "label": "BB+Vol"},
    {"entry": "price_breakout:10 + trend_up:4", "exit": "trend_up:4", "label": "Breakout+Trend"},
    # Simple combos
    {
        "entry": "rsi_oversold:30 + volume_spike:2.0",
        "exit": "rsi_overbought:70",
        "label": "RSI+Vol",
    },
    {"entry": "macd_cross_up + volume_spike:2.5", "exit": "macd_cross_up", "label": "MACD+Vol"},
    # ── Trend Following (new filters) ──
    {
        "entry": "ema_cross_up:12:26 + adx_strong:25 + volume_spike:2.0",
        "exit": "atr_trailing_exit:14:2.5",
        "label": "EMACross+ADX+Vol→ATR",
    },
    {
        "entry": "ema_cross_up:12:26 + adx_strong:25",
        "exit": "rsi_overbought:70",
        "label": "EMACross+ADX",
    },
    {
        "entry": "stoch_oversold:20 + aroon_up:70",
        "exit": "stoch_overbought:80",
        "label": "Stoch+Aroon",
    },
    {
        "entry": "roc_positive:12 + ichimoku_above + obv_rising",
        "exit": "rsi_overbought:70",
        "label": "ROC+Ichi+OBV",
    },
    {
        "entry": "donchian_break:20 + adx_strong:25 + volume_spike:2.0",
        "exit": "donchian_break:20",
        "label": "Donchian+ADX+Vol",
    },
    {
        "entry": "macd_cross_up + aroon_up:70 + mfi_confirm:50",
        "exit": "mfi_overbought:80",
        "label": "MACD+Aroon+MFI",
    },
    # ── Mean Reversion (new filters) ──
    {
        "entry": "rsi_oversold:30 + adx_strong:20 + obv_rising",
        "exit": "zscore_extreme:2.0",
        "label": "RSI+ADX+OBV→Zscore",
    },
    {
        "entry": "stoch_oversold:20 + ema_above:50 + mfi_confirm:40",
        "exit": "stoch_overbought:80",
        "label": "Stoch+EMA+MFI",
    },
    {
        "entry": "cci_oversold:100 + trend_up:4 + volume_spike:2.0",
        "exit": "cci_overbought:100",
        "label": "CCI+Trend+Vol",
    },
    {"entry": "mfi_oversold:20 + trend_up:4", "exit": "mfi_overbought:80", "label": "MFI+Trend"},
    {
        "entry": "rsi_oversold:30 + ichimoku_above",
        "exit": "pct_from_ma_exit:20:5.0",
        "label": "RSI+Ichi→PctMA",
    },
    # ── Volatility Breakout (new filters) ──
    {
        "entry": "bb_upper_break:20 + bb_squeeze + volume_spike:2.0",
        "exit": "ema_above:20",
        "label": "BB+Squeeze+Vol",
    },
    {
        "entry": "atr_breakout:14:2.0 + adx_strong:25 + obv_rising",
        "exit": "atr_trailing_exit:14:2.5",
        "label": "ATR+ADX+OBV→ATR",
    },
    {
        "entry": "keltner_break + trend_up:4 + volume_spike:2.0",
        "exit": "keltner_break",
        "label": "KC+Trend+Vol",
    },
    {
        "entry": "price_breakout:20 + bb_bandwidth_low:0.05 + volume_spike:2.5",
        "exit": "pct_from_ma_exit:20:5.0",
        "label": "Breakout+BBW+Vol",
    },
    # ── Multi-Confirm (new filters) ──
    {
        "entry": "rsi_oversold:30 + stoch_oversold:20 + adx_strong:25",
        "exit": "rsi_overbought:70",
        "label": "RSI+Stoch+ADX",
    },
    {
        "entry": "macd_cross_up + obv_rising + adx_strong:25",
        "exit": "atr_trailing_exit:14:2.0",
        "label": "MACD+OBV+ADX→ATR",
    },
    {
        "entry": "ema_cross_up:12:26 + mfi_confirm:50 + bb_bandwidth_low:0.04",
        "exit": "zscore_extreme:2.0",
        "label": "EMA+MFI+BBW→Zscore",
    },
    # ── ML + Rule combos (threshold 0.35 = veto filter mode) ──
    {
        "entry": "trend_up:4 + rsi_oversold:30 + lgbm_prob:0.35",
        "exit": "rsi_overbought:70",
        "label": "Trend+RSI+ML",
    },
    {
        "entry": "ema_cross_up:12:26 + lgbm_prob:0.35",
        "exit": "atr_trailing_exit:14:2.5",
        "label": "EMACross+ML→ATR",
    },
    {
        "entry": "volume_spike:2.0 + adx_strong:25 + lgbm_prob:0.35",
        "exit": "rsi_overbought:70",
        "label": "Vol+ADX+ML",
    },
    # ── ML Veto: Trend Following ──
    {
        "entry": "ema_cross_up:12:26 + trend_up:4 + lgbm_prob:0.35",
        "exit": "atr_trailing_exit:14:2.5",
        "label": "ML+TrendEMA",
    },
    {
        "entry": "adx_strong:25 + ema_above:50 + lgbm_prob:0.35",
        "exit": "rsi_overbought:70 + atr_trailing_exit:14:2.0",
        "label": "ML+ADXTrend",
    },
    {
        "entry": "ichimoku_above + aroon_up:70 + lgbm_prob:0.35",
        "exit": "pct_from_ma_exit:20:5.0",
        "label": "ML+IchimokuTrend",
    },
    # ── ML Veto: Mean Reversion ──
    {
        "entry": "rsi_oversold:30 + stoch_oversold:20 + lgbm_prob:0.35",
        "exit": "rsi_overbought:70 + stoch_overbought:80",
        "label": "ML+RSIStoch",
    },
    {
        "entry": "cci_oversold:100 + obv_rising + lgbm_prob:0.35",
        "exit": "cci_overbought:100 + zscore_extreme:2.0",
        "label": "ML+CCIMeanRev",
    },
    {
        "entry": "mfi_oversold:20 + ema_above:50 + lgbm_prob:0.35",
        "exit": "mfi_overbought:80 + pct_from_ma_exit:20:5.0",
        "label": "ML+MFIMeanRev",
    },
    # ── ML Veto: Breakout ──
    {
        "entry": "donchian_break:20 + volume_spike:2.0 + lgbm_prob:0.35",
        "exit": "atr_trailing_exit:14:2.5",
        "label": "ML+DonchianBreak",
    },
    {
        "entry": "bb_squeeze + price_breakout:10 + lgbm_prob:0.35",
        "exit": "atr_trailing_exit:14:2.0 + zscore_extreme:2.0",
        "label": "ML+BBSqueeze",
    },
    {
        "entry": "keltner_break + adx_strong:25 + lgbm_prob:0.35",
        "exit": "rsi_overbought:70 + atr_trailing_exit:14:2.5",
        "label": "ML+KeltnerBreak",
    },
    # ── ML Veto: Volume-Confirmed ──
    {
        "entry": "macd_cross_up + volume_spike:2.0 + mfi_confirm:50 + lgbm_prob:0.35",
        "exit": "rsi_overbought:70 + mfi_overbought:80",
        "label": "ML+VolMACDConfirm",
    },
    # ── ML Veto: Multi-Confluence ──
    {
        "entry": "roc_positive + obv_rising + trend_up:4 + lgbm_prob:0.35",
        "exit": "atr_trailing_exit:14:2.0",
        "label": "ML+ROCObvTrend",
    },
    {
        "entry": "stoch_oversold:25 + aroon_up:70 + lgbm_prob:0.35",
        "exit": "stoch_overbought:80 + pct_from_ma_exit:20:5.0",
        "label": "ML+StochAroonConfirm",
    },
]


def _build_combined_strategy(
    entry: str,
    exit_: str,
    symbol: str,
    timeframe: str,
):
    """Build a CombinedStrategy from filter strings."""
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
    strategy.symbols = [symbol]
    strategy.timeframe = timeframe
    return strategy


def _find_combine_template(label: str) -> dict | None:
    """Find a COMBINE_TEMPLATES entry by label (case-insensitive)."""
    label_lower = label.lower()
    for tmpl in COMBINE_TEMPLATES:
        if tmpl["label"].lower() == label_lower:
            return tmpl
    return None


def _resolve_strategy(
    strategy_name: str,
    symbol: str,
    timeframe: str,
    symbols: list[str] | None = None,
):
    """Resolve strategy by name — checks COMBINE_TEMPLATES, then STRATEGY_MAP.

    Returns (strategy_instance, strategy_name, strategy_cls_or_none).
    strategy_cls is only set for registered strategies (needed by optimize/walk-forward).
    """
    tmpl = _find_combine_template(strategy_name)
    if tmpl is not None:
        # ML filters bind to a single symbol; reject multi-symbol + ML combos
        has_ml = "lgbm_prob" in tmpl["entry"]
        if has_ml and symbols and len(symbols) > 1:
            console.print(
                "[red]Combined templates with lgbm_prob cannot be used with "
                "multiple symbols (ML model is per-symbol). "
                "Use --symbol to specify a single symbol.[/red]"
            )
            raise typer.Exit(1)
        strategy = _build_combined_strategy(
            tmpl["entry"],
            tmpl["exit"],
            symbol,
            timeframe,
        )
        if symbols:
            strategy.symbols = symbols
        return strategy, tmpl["label"], None

    _load_strategies()
    if strategy_name not in STRATEGY_MAP:
        console.print(f"[red]Unknown strategy: {strategy_name}[/red]")
        available = list(STRATEGY_MAP.keys()) + [t["label"] for t in COMBINE_TEMPLATES]
        console.print(f"Available: {', '.join(available)}")
        raise typer.Exit(1)

    strategy_cls = STRATEGY_MAP[strategy_name]
    strategy = strategy_cls()
    strategy.symbols = symbols or [symbol]
    strategy.timeframe = timeframe
    return strategy, strategy_name, strategy_cls


@app.command()
def combine(
    entry: str = typer.Option(
        ..., "--entry", help="Entry filters (e.g., 'trend_up:4 + rsi_oversold:30')"
    ),
    exit_: str = typer.Option(..., "--exit", help="Exit filters (e.g., 'rsi_overbought:70')"),
    symbol: str = typer.Option("BTC/KRW", "--symbol", "-s", help="Trading pair"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Candle timeframe"),
    balance: float = typer.Option(1_000_000, "--balance", "-b", help="Initial balance (KRW)"),
    start: str | None = typer.Option(
        None, "--start", help="Override evaluation start (YYYY-MM-DD)"
    ),
    end: str | None = typer.Option(None, "--end", help="Override evaluation end (YYYY-MM-DD)"),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
    include_train: bool = typer.Option(
        False,
        "--include-train",
        help="Disable holdout-only filtering and evaluate on the full data range.",
    ),
) -> None:
    """Backtest a combined filter strategy (no code needed).

    By default the strategy is evaluated only on the data's last 20% so the
    result is comparable to ``ml-backtest``. Pass ``--include-train`` to
    evaluate the full data range or use ``--start``/``--end`` for an
    explicit window.
    """
    setup_logging()
    _validate_date_range(start, end)

    from tradingbot.backtest.engine import BacktestEngine
    from tradingbot.data.storage import load_candles
    from tradingbot.strategy.combined import CombinedStrategy
    from tradingbot.strategy.filters.registry import parse_filter_string

    try:
        entry_filters = parse_filter_string(entry, base_timeframe=timeframe)
        exit_filters = parse_filter_string(exit_, base_timeframe=timeframe)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    # Pass symbol/timeframe to ML filters
    for f in entry_filters + exit_filters:
        if hasattr(f, "symbol"):
            f.symbol = symbol
        if hasattr(f, "timeframe"):
            f.timeframe = timeframe

    strategy = CombinedStrategy(entry_filters=entry_filters, exit_filters=exit_filters)
    strategy.symbols = [symbol]
    strategy.timeframe = timeframe

    console.print(f"[bold]Combined Strategy: {strategy.describe()}[/bold]")
    console.print(f"  Symbol: {symbol} ({timeframe})")

    try:
        df = load_candles(symbol, timeframe, Path(data_dir))
    except FileNotFoundError:
        console.print(f"[red]No data for {symbol} {timeframe}.[/red]")
        raise typer.Exit(1)

    # Resolve holdout window after data is loaded (auto cutoff needs timestamps).
    # Slice the df here (rather than letting engine.run slice via config dates) so
    # the "Data: N candles" line below reflects the actual evaluation length.
    effective_start, effective_end, period_note = _resolve_holdout_window(
        df,
        start,
        end,
        include_train,
    )
    if effective_start:
        df = df[df.index >= effective_start]
    if effective_end:
        df = df[df.index <= effective_end]

    console.print(f"  Data: {len(df)} candles")
    eval_start_str = effective_start or "data start"
    eval_end_str = effective_end or "data end"
    console.print(f"  Evaluation period: {eval_start_str} → {eval_end_str} ({period_note})")

    config = load_config(
        Path("config"),
        overrides={
            "trading": {"symbols": [symbol], "timeframe": timeframe, "initial_balance": balance},
        },
    )

    engine = BacktestEngine(strategy=strategy, config=config)
    report = engine.run({symbol: df})
    report.print_summary()


@app.command(name="combine-scan")
def combine_scan(
    top_n: int = typer.Option(10, "--top", help="Show top N results"),
    verify_top: int = typer.Option(0, "--verify-top", help="Re-verify top N with full engine"),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
    balance: float = typer.Option(1_000_000, "--balance", "-b", help="Initial balance (KRW)"),
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
            "Write the Top-N table as markdown to this path "
            "(e.g. personal/combine_scan_holdout_result.md)."
        ),
    ),
) -> None:
    """Scan predefined filter combinations across all symbols and timeframes.

    By default each (symbol, timeframe) batch is evaluated only on its
    last 20% — same policy as ``scan`` / ``backtest`` / ``combine``.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    setup_logging()
    _validate_date_range(start, end)

    from tradingbot.data.storage import list_available_data

    available = list_available_data(Path(data_dir))
    if not available:
        console.print("[red]No data found. Run tradingbot download first.[/red]")
        raise typer.Exit(1)

    symbol_timeframes: dict[str, list[str]] = {}
    for item in available:
        symbol_timeframes.setdefault(item["symbol"], []).append(item["timeframe"])

    # Build batched jobs: group by (symbol, timeframe) to load data once
    batches: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    total = 0
    for sym, timeframes in symbol_timeframes.items():
        for tf in timeframes:
            batch_jobs = [
                (tmpl["label"], tmpl["entry"], tmpl["exit"]) for tmpl in COMBINE_TEMPLATES
            ]
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
        f"[bold]Scanning {len(COMBINE_TEMPLATES)} templates × {len(symbol_timeframes)} symbols "
        f"× timeframes ({total} combinations, {n_workers} workers, "
        f"{len(batches)} batches){range_note}...[/bold]"
    )

    results: list[dict] = []
    failures: list[str] = []

    from tradingbot.backtest.parallel import _run_batch

    with _progress_context() as progress:
        task = progress.add_task("Scanning combinations", total=total)

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
                    batch_results = future.result(timeout=1800)
                except Exception as exc:
                    failures.append(f"{sym}/{tf}: worker crashed: {exc}")
                    progress.advance(task, advance=len(batches[(sym, tf)]))
                    continue
                for r in batch_results:
                    if r.error:
                        failures.append(f"{r.strategy}/{r.symbol}/{r.timeframe}: {r.error}")
                    else:
                        results.append(
                            {
                                "template": r.strategy,
                                "entry": r.entry,
                                "exit": r.exit,
                                "symbol": r.symbol,
                                "timeframe": r.timeframe,
                                "sharpe_ratio": r.sharpe_ratio,
                                "total_return": r.total_return,
                                "max_drawdown": r.max_drawdown,
                                "win_rate": r.win_rate,
                                "profit_factor": r.profit_factor,
                                "total_trades": r.total_trades,
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

    # Sort by Sharpe descending
    results.sort(key=lambda r: r["sharpe_ratio"], reverse=True)

    # Phase 2: Re-verify top N with full engine
    verified_set: set[tuple[str, str, str]] = set()
    if verify_top > 0 and results:
        n_verify = min(verify_top, len(results))
        to_verify = results[:n_verify]

        # ML templates already went through full engine — mark as verified, skip re-run
        verify_jobs: list[dict] = []
        for r in to_verify:
            if "lgbm_prob" in r["entry"]:
                verified_set.add((r["template"], r["symbol"], r["timeframe"]))
            else:
                verify_jobs.append(r)

        if verify_jobs:
            # Group by (symbol, timeframe)
            verify_batches: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
            for r in verify_jobs:
                key = (r["symbol"], r["timeframe"])
                verify_batches.setdefault(key, []).append((r["template"], r["entry"], r["exit"]))

            console.print(
                f"\n[bold]Re-verifying top {len(verify_jobs)} results "
                f"with full engine ({len(verify_batches)} batches)...[/bold]"
            )

            verified_results: dict[tuple[str, str, str], dict] = {}
            with _progress_context() as progress:
                task = progress.add_task("Verifying", total=len(verify_jobs))

                with ProcessPoolExecutor(
                    max_workers=n_workers,
                    mp_context=multiprocessing.get_context("spawn"),
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
                            True,
                            start,
                            end,
                            include_train,
                        ): (sym, tf)
                        for (sym, tf), batch_jobs in verify_batches.items()
                    }
                    for future in as_completed(futures):
                        sym, tf = futures[future]
                        try:
                            batch_results = future.result(timeout=1800)
                        except Exception as exc:
                            console.print(f"[yellow]Verify failed {sym}/{tf}: {exc}[/yellow]")
                            n_batch = len(verify_batches[(sym, tf)])
                            progress.advance(task, advance=n_batch)
                            continue
                        for r in batch_results:
                            if not r.error:
                                verified_results[(r.strategy, r.symbol, r.timeframe)] = {
                                    "sharpe_ratio": r.sharpe_ratio,
                                    "total_return": r.total_return,
                                    "max_drawdown": r.max_drawdown,
                                    "win_rate": r.win_rate,
                                    "profit_factor": r.profit_factor,
                                    "total_trades": r.total_trades,
                                }
                                verified_set.add((r.strategy, r.symbol, r.timeframe))
                            progress.advance(task)

            # Replace results with verified metrics
            for r in results:
                key = (r["template"], r["symbol"], r["timeframe"])
                if key in verified_results:
                    r.update(verified_results[key])

            # Re-sort after verification
            results.sort(key=lambda r: r["sharpe_ratio"], reverse=True)

            console.print(f"[green]Verified {len(verified_set)} results.[/green]")

    table = Table(title=f"Best Filter Combinations (Top {min(top_n, len(results))})")
    table.add_column("#", justify="right")
    table.add_column("Template")
    table.add_column("Symbol")
    table.add_column("TF")
    table.add_column("Sharpe", justify="right")
    table.add_column("Return", justify="right")
    table.add_column("MaxDD", justify="right")
    table.add_column("Win%", justify="right")
    table.add_column("PF", justify="right")
    table.add_column("Trades", justify="right")
    if verify_top > 0:
        table.add_column("V", justify="center")

    for i, r in enumerate(results[:top_n], 1):
        sharpe_style = (
            "green" if r["sharpe_ratio"] > 1.0 else ("yellow" if r["sharpe_ratio"] > 0 else "red")
        )
        row = [
            str(i),
            r["template"],
            r["symbol"],
            r["timeframe"],
            f"[{sharpe_style}]{r['sharpe_ratio']:.2f}[/{sharpe_style}]",
            f"{r['total_return']:.2%}",
            f"{r['max_drawdown']:.2%}",
            f"{r['win_rate']:.1%}",
            f"{r['profit_factor']:.2f}",
            str(r["total_trades"]),
        ]
        if verify_top > 0:
            key = (r["template"], r["symbol"], r["timeframe"])
            row.append("[green]✓[/green]" if key in verified_set else "")
        table.add_row(*row)

    console.print(table)

    # Show the entry/exit details of top results
    console.print("\n[bold]Top combination details:[/bold]")
    for i, r in enumerate(results[:3], 1):
        console.print(f"  #{i} {r['template']} ({r['symbol']} {r['timeframe']})")
        console.print(f"     Entry: {r['entry']}")
        console.print(f"     Exit:  {r['exit']}")

    if output:
        from datetime import UTC, datetime

        if start or end:
            range_md = f"{start or 'start'} → {end or 'end'}"
        elif include_train:
            range_md = "full data range (--include-train)"
        else:
            range_md = "**각 (symbol, timeframe) 배치의 마지막 20%** (auto holdout)"
        n_top = min(top_n, len(results))
        columns = [
            "#",
            "Template",
            "Symbol",
            "TF",
            "Sharpe",
            "Return",
            "MaxDD",
            "Win%",
            "PF",
            "Trades",
        ]
        if verify_top > 0:
            columns.append("V")
        md_rows: list[list[str]] = []
        for i, r in enumerate(results[:n_top], 1):
            row = [
                str(i),
                r["template"],
                r["symbol"],
                r["timeframe"],
                f"{r['sharpe_ratio']:.2f}",
                f"{r['total_return']:.2%}",
                f"{r['max_drawdown']:.2%}",
                f"{r['win_rate']:.1%}",
                f"{r['profit_factor']:.2f}",
                str(r["total_trades"]),
            ]
            if verify_top > 0:
                key = (r["template"], r["symbol"], r["timeframe"])
                row.append("✓" if key in verified_set else "")
            md_rows.append(row)
        out_path = Path(output)
        metadata = [
            f"일시: {datetime.now(UTC).strftime('%Y-%m-%d')}",
            f"대상: {len(COMBINE_TEMPLATES)} templates × {len(symbol_timeframes)} symbols × "
            f"timeframes ({total} combinations)",
            f"Workers: {n_workers}",
            f"평가 기간: {range_md}",
        ]
        if verify_top > 0:
            metadata.append(f"Verify-Top: {verify_top} (✓ = 풀 엔진 재검증 통과)")
        _write_scan_markdown_report(
            output=out_path,
            title=f"Combine-Scan Result — Top {n_top}",
            metadata=metadata,
            section=f"Best Filter Combinations (Top {n_top})",
            columns=columns,
            rows=md_rows,
        )
        console.print(f"[green]Wrote {n_top} rows to {out_path}[/green]")
