"""ML commands: train, walk-forward, backtest, diagnostics, and tuning (single and batch)."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from tradingbot.cli._shared import _progress_context, app, console
from tradingbot.config import load_config
from tradingbot.data.storage import EXTERNAL_SUBDIR, list_available_data
from tradingbot.utils.logging import setup_logging


@app.command(name="ml-train")
def ml_train(
    symbol: str = typer.Option("BTC/KRW", "--symbol", "-s", help="Symbol to train"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Timeframe"),
    train_months: int = typer.Option(3, "--train-months", help="Training window months"),
    test_months: int = typer.Option(1, "--test-months", help="Test window months"),
    target_kind: str = typer.Option(
        "binary",
        "--target-kind",
        help="Target labelling strategy: binary | atr | triple-barrier",
    ),
    atr_mult: float = typer.Option(
        1.0,
        "--atr-mult",
        help="ATR multiplier for atr / triple-barrier targets",
    ),
    include_extra: bool = typer.Option(
        False,
        "--include-extra",
        help="Add Phase 4 extra features (regime, lag/diff, session)",
    ),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
    model_dir: str = typer.Option("models", "--model-dir", help="Model output directory"),
) -> None:
    """Train a LightGBM model with walk-forward validation."""
    setup_logging()

    from tradingbot.data.external_fetcher import build_external_df
    from tradingbot.data.storage import load_candles
    from tradingbot.ml.walk_forward import MLWalkForwardTrainer

    try:
        df = load_candles(symbol, timeframe, Path(data_dir))
    except FileNotFoundError:
        console.print(
            f"[red]No data for {symbol} {timeframe}. Run tradingbot download first.[/red]"
        )
        raise typer.Exit(1)

    # Load external features if available
    ext_dir = Path(data_dir) / EXTERNAL_SUBDIR
    external_df = build_external_df(df, ext_dir)
    ext_count = (
        len([c for c in (external_df.columns if external_df is not None else [])])
        if external_df is not None
        else 0
    )

    console.print(f"[bold]Training LightGBM model for {symbol} {timeframe}...[/bold]")
    console.print(f"  Data: {len(df)} candles ({df.index[0]} → {df.index[-1]})")
    console.print(f"  Walk-Forward: {train_months}m train / {test_months}m test")
    console.print(f"  External features: {ext_count} sources loaded")

    trainer = MLWalkForwardTrainer(
        symbol=symbol,
        timeframe=timeframe,
        train_months=train_months,
        test_months=test_months,
        target_kind=target_kind,
        atr_mult=atr_mult,
        include_extra=include_extra,
        model_dir=Path(model_dir),
    )
    report = trainer.run(df, external_df=external_df)

    if not report.windows:
        console.print("[red]Training failed — insufficient data or no valid windows.[/red]")
        raise typer.Exit(1)

    # Display results
    console.print("\n[bold green]Training complete![/bold green]")
    console.print(f"  Inner-val AUC: {report.avg_auc:.4f}")
    console.print(f"  Inner-val Precision: {report.avg_precision:.4f}")
    console.print(f"  Holdout AUC: {report.holdout_auc:.4f}")
    console.print(f"  Holdout Precision: {report.holdout_precision:.4f}")
    console.print(f"  Model saved: {report.model_path}")

    # Per-split details (Path B: single training with inner train/val split)
    table = Table(title="Training Splits")
    table.add_column("Split", style="cyan")
    table.add_column("AUC", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("Train", justify="right")
    table.add_column("Val", justify="right")
    table.add_column("Best Iter", justify="right")

    for w in report.windows:
        table.add_row(
            str(w.get("split", "")),
            f"{w['auc']:.4f}" if "auc" in w else "—",
            f"{w['precision']:.4f}" if "precision" in w else "—",
            f"{w['recall']:.4f}" if "recall" in w else "—",
            str(w.get("n_train", "")),
            str(w.get("n_val", "")),
            str(w.get("best_iteration", "")),
        )
    console.print(table)

    # Top 10 feature importance
    if report.feature_importance:
        console.print("\n[bold]Top 10 Feature Importance:[/bold]")
        for i, (feat, imp) in enumerate(list(report.feature_importance.items())[:10], 1):
            console.print(f"  {i:2d}. {feat}: {imp:.1f}")


@app.command(name="ml-walk-forward")
def ml_walk_forward(
    symbol: str = typer.Option("BTC/KRW", "--symbol", "-s", help="Symbol"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Timeframe"),
    train_months: int = typer.Option(6, "--train-months", help="Training window months"),
    test_months: int = typer.Option(2, "--test-months", help="Test window months"),
    forward_candles: int = typer.Option(4, "--forward-candles", help="Target horizon"),
    threshold: float = typer.Option(0.006, "--threshold", help="Target return threshold"),
    target_kind: str = typer.Option(
        "binary",
        "--target-kind",
        help="Target labelling strategy: binary | atr | triple-barrier",
    ),
    atr_mult: float = typer.Option(
        1.0, "--atr-mult", help="ATR multiplier for atr / triple-barrier targets"
    ),
    include_extra: bool = typer.Option(
        False,
        "--include-extra",
        help="Add Phase 4 extra features (regime, lag/diff, session)",
    ),
    entry_threshold: float = typer.Option(
        0.45, "--entry-threshold", help="Entry probability threshold"
    ),
    exit_threshold: float = typer.Option(
        0.30, "--exit-threshold", help="Exit probability threshold"
    ),
    balance: float = typer.Option(1_000_000, "--balance", "-b", help="Initial balance (KRW)"),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
    output_dir: str = typer.Option(
        "results/ml_walkforward", "--output-dir", help="Where to write JSON + markdown reports"
    ),
) -> None:
    """Time-honest walk-forward for the LGBM strategy.

    Trains a fresh model per window using only past data (Path B: single
    training with inner train/val split for early stopping), then evaluates
    the strategy on the test window via the standard backtest engine.
    """
    setup_logging()

    import json

    from tradingbot.data.storage import load_candles
    from tradingbot.ml.strategy_walk_forward import MLStrategyWalkForward

    try:
        df = load_candles(symbol, timeframe, Path(data_dir))
    except FileNotFoundError:
        console.print(
            f"[red]No data for {symbol} {timeframe}. Run tradingbot download first.[/red]"
        )
        raise typer.Exit(1)

    ext_dir = Path(data_dir) / EXTERNAL_SUBDIR
    has_external = ext_dir.exists() and any(ext_dir.iterdir())

    console.print(f"[bold]ML walk-forward for {symbol} {timeframe}[/bold]")
    console.print(f"  Data: {len(df)} candles ({df.index[0]} → {df.index[-1]})")
    console.print(f"  Walk-Forward: {train_months}m train / {test_months}m test")
    console.print(f"  Entry threshold: {entry_threshold}, Exit threshold: {exit_threshold}")
    console.print(f"  External data dir: {ext_dir if has_external else '(none)'}")

    # Route through load_config so risk/backtest (fee rate, slippage, stop
    # loss, etc.) honor config/*.yaml instead of hardcoded defaults.
    config = load_config(
        overrides={
            "trading": {
                "symbols": [symbol],
                "timeframe": timeframe,
                "initial_balance": balance,
            }
        }
    )

    runner = MLStrategyWalkForward(
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
        external_data_dir=ext_dir if has_external else None,
        config=config,
    )
    report = runner.run(df)

    if not report.windows:
        console.print("[red]No windows produced — check data length vs train/test sizes.[/red]")
        raise typer.Exit(1)

    table = Table(title=f"Walk-Forward Windows ({len(report.windows)} total)")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Train End", style="dim")
    table.add_column("Test Start", style="dim")
    table.add_column("Test End", style="dim")
    table.add_column("Sharpe", justify="right")
    table.add_column("Return %", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Win %", justify="right")
    table.add_column("MaxDD %", justify="right")

    for w in report.windows:
        table.add_row(
            str(w["window"]),
            w["train_end"][:10],
            w["test_start"][:10],
            w["test_end"][:10],
            f"{w['sharpe']:.2f}",
            f"{w['return_pct']:+.2f}",
            str(w["trades"]),
            f"{w['win_rate'] * 100:.1f}",
            f"{w['max_dd_pct']:.2f}",
        )
    console.print(table)

    console.print(
        f"\n[bold green]Cumulative:[/bold green] "
        f"avg_sharpe={report.avg_sharpe:.2f}, "
        f"cumulative_return={report.cumulative_return_pct:+.2f}%, "
        f"trades={report.total_trades}, "
        f"avg_win_rate={report.avg_win_rate * 100:.1f}%, "
        f"skipped_windows={report.n_skipped}"
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{symbol.replace('/', '_')}_{timeframe}_walkforward"
    json_path = out_dir / f"{base}.json"
    md_path = out_dir / f"{base}.md"

    json_path.write_text(
        json.dumps(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "train_months": train_months,
                "test_months": test_months,
                "entry_threshold": entry_threshold,
                "exit_threshold": exit_threshold,
                "n_windows": report.n_windows,
                "n_skipped": report.n_skipped,
                "avg_sharpe": report.avg_sharpe,
                "cumulative_return_pct": report.cumulative_return_pct,
                "final_equity_multiple": report.final_equity_multiple,
                "total_trades": report.total_trades,
                "avg_win_rate": report.avg_win_rate,
                "windows": report.windows,
            },
            indent=2,
            default=str,
        )
    )

    md_lines = [
        f"# ML Walk-Forward — {symbol} {timeframe}",
        "",
        f"- Train/test: {train_months}m / {test_months}m",
        f"- Entry/Exit threshold: {entry_threshold} / {exit_threshold}",
        f"- Windows: {report.n_windows} (skipped: {report.n_skipped})",
        f"- Avg Sharpe: **{report.avg_sharpe:.2f}**",
        f"- Cumulative return: **{report.cumulative_return_pct:+.2f}%** "
        "_(compounded per-window % gains; balance resets to initial each window)_",
        f"- Total trades: {report.total_trades}",
        f"- Avg win rate (traded windows): {report.avg_win_rate * 100:.1f}%",
        "",
        "| # | Train End | Test Start | Test End | Sharpe | Return % | Trades | Win % | MaxDD % |",
        "|---|-----------|------------|----------|-------:|---------:|-------:|------:|--------:|",
    ]
    for w in report.windows:
        md_lines.append(
            f"| {w['window']} | {w['train_end'][:10]} | {w['test_start'][:10]} | "
            f"{w['test_end'][:10]} | {w['sharpe']:.2f} | {w['return_pct']:+.2f} | "
            f"{w['trades']} | {w['win_rate'] * 100:.1f} | {w['max_dd_pct']:.2f} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n")

    console.print(f"\n[dim]JSON: {json_path}[/dim]")
    console.print(f"[dim]Markdown: {md_path}[/dim]")


@app.command(name="ml-backtest")
def ml_backtest(
    symbol: str = typer.Option("BTC/KRW", "--symbol", "-s", help="Symbol"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Timeframe"),
    balance: float = typer.Option(1_000_000, "--balance", "-b", help="Initial balance (KRW)"),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
    model_dir: str = typer.Option("models", "--model-dir", help="Model directory"),
    entry_threshold: float = typer.Option(
        0.45, "--entry-threshold", help="Entry probability threshold"
    ),
    exit_threshold: float = typer.Option(
        0.30, "--exit-threshold", help="Exit probability threshold"
    ),
    start: str = typer.Option(
        None,
        "--start",
        help="Override evaluation start (YYYY-MM-DD). Default: meta.holdout_start.",
    ),
    end: str = typer.Option(None, "--end", help="Override evaluation end (YYYY-MM-DD)."),
    include_train: bool = typer.Option(
        False,
        "--include-train",
        help="Disable holdout-only filtering and evaluate on the full data range.",
    ),
) -> None:
    """Backtest using a pre-trained LightGBM model.

    By default the model is evaluated only on the holdout window recorded in
    its meta.json (everything after ``train_end`` plus an embargo). Pass
    ``--include-train`` to backtest the full data range, or use ``--start`` /
    ``--end`` to set an explicit window.
    """
    setup_logging()

    from tradingbot.backtest.engine import BacktestEngine
    from tradingbot.data.storage import load_candles
    from tradingbot.ml.trainer import LGBMTrainer
    from tradingbot.strategy.base import StrategyParams
    from tradingbot.strategy.lgbm_strategy import LGBMStrategy

    try:
        df = load_candles(symbol, timeframe, Path(data_dir))
    except FileNotFoundError:
        console.print(f"[red]No data for {symbol} {timeframe}.[/red]")
        raise typer.Exit(1)

    # Resolve evaluation window. Precedence: --start/--end > meta-derived holdout > full range.
    meta = LGBMTrainer.load_meta(symbol, timeframe, Path(model_dir))
    effective_start: str | None = start
    effective_end: str | None = end
    period_note = "user-specified range"

    if not include_train and effective_start is None:
        if meta is not None and meta.get("holdout_start"):
            effective_start = meta["holdout_start"]
            period_note = "holdout window (post training cut)"
            # Cap the eval end at the calibrator-leak-free boundary so we don't
            # score on candles the calibrator was fit on (the holdout cal half).
            if effective_end is None and meta.get("holdout_eval_end"):
                effective_end = meta["holdout_eval_end"]
                period_note = "holdout eval window (calibrator-leak-free)"
        elif meta is not None and meta.get("train_end"):
            effective_start = meta["train_end"]
            period_note = (
                "post train_end (legacy meta — pass --include-train if results look empty)"
            )
        else:
            console.print(
                "[yellow]Warning: meta.json missing train_end / holdout_start — "
                "evaluating on full data range. Pass --start to constrain manually.[/yellow]"
            )
            period_note = "full data range (no meta)"
    elif include_train and start is None and end is None:
        period_note = "full data range (--include-train)"

    strategy = LGBMStrategy(
        StrategyParams(
            values={
                "entry_threshold": entry_threshold,
                "exit_threshold": exit_threshold,
                "model_dir": model_dir,
            }
        )
    )
    strategy.symbols = [symbol]
    strategy.timeframe = timeframe

    config = load_config(
        Path("config"),
        overrides={
            "trading": {"symbols": [symbol], "timeframe": timeframe, "initial_balance": balance},
            "backtest": {"start_date": effective_start, "end_date": effective_end},
        },
    )

    console.print(f"[bold]Backtesting LightGBM strategy on {symbol} {timeframe}...[/bold]")
    eval_start_str = effective_start or str(df.index[0])
    eval_end_str = effective_end or str(df.index[-1])
    console.print(f"  Evaluation period: {eval_start_str} → {eval_end_str} ({period_note})")

    engine = BacktestEngine(strategy=strategy, config=config)
    report = engine.run({symbol: df})

    console.print("\n[bold]Results:[/bold]")
    console.print(f"  Final Balance: {report.final_balance:,.0f} KRW")
    console.print(f"  Total Return: {report.total_return:.2%}")
    console.print(f"  Sharpe Ratio: {report.sharpe_ratio:.2f}")
    console.print(f"  Max Drawdown: {report.max_drawdown:.2%}")
    console.print(f"  Win Rate: {report.win_rate:.2%}")
    console.print(f"  Profit Factor: {report.profit_factor:.2f}")
    console.print(f"  Total Trades: {report.total_trades}")


@app.command(name="ml-train-all")
def ml_train_all(
    timeframe: str | None = typer.Option(
        None,
        "--timeframe",
        "-t",
        help="Train only this timeframe",
    ),
    train_months: int = typer.Option(3, "--train-months", help="Training window months"),
    test_months: int = typer.Option(1, "--test-months", help="Test window months"),
    target_kind: str = typer.Option(
        "binary",
        "--target-kind",
        help="Target labelling strategy: binary | atr | triple-barrier",
    ),
    atr_mult: float = typer.Option(
        1.0, "--atr-mult", help="ATR multiplier for atr / triple-barrier targets"
    ),
    include_extra: bool = typer.Option(
        False,
        "--include-extra",
        help="Add Phase 4 extra features (regime, lag/diff, session)",
    ),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
    model_dir: str = typer.Option("models", "--model-dir", help="Model output directory"),
    workers: int = typer.Option(
        0,
        "--workers",
        "-w",
        help="Parallel workers (0=auto: cpu_count//2, 1=sequential)",
    ),
) -> None:
    """Train LightGBM models for all available symbol × timeframe combinations."""
    import multiprocessing as mp

    setup_logging()

    available = list_available_data(Path(data_dir))
    if not available:
        console.print("[red]No data found. Run tradingbot download first.[/red]")
        raise typer.Exit(1)

    # Build symbol × timeframe pairs
    pairs: list[tuple[str, str]] = []
    for item in available:
        if timeframe and item["timeframe"] != timeframe:
            continue
        pairs.append((item["symbol"], item["timeframe"]))

    if not pairs:
        tf_label = timeframe if timeframe else "all"
        console.print(f"[red]No data found for timeframe={tf_label}.[/red]")
        raise typer.Exit(1)

    # Resolve worker count
    cpu_count = mp.cpu_count()
    if workers <= 0:
        workers = max(1, min(cpu_count // 2, len(pairs)))
    workers = min(workers, len(pairs))
    threads_per_worker = max(1, cpu_count // workers)

    console.print(f"[bold]Training ML models for {len(pairs)} symbol × timeframe pairs...[/bold]")
    console.print(f"  Walk-Forward: {train_months}m train / {test_months}m test")
    console.print(f"  Workers: {workers}  (threads/worker: {threads_per_worker})\n")

    results: list[dict] = []

    if workers == 1:
        # Sequential — zero subprocess overhead
        from tradingbot.data.external_fetcher import build_external_df
        from tradingbot.data.storage import load_candles
        from tradingbot.ml.walk_forward import MLWalkForwardTrainer

        ext_dir = Path(data_dir) / EXTERNAL_SUBDIR

        with _progress_context() as progress:
            task = progress.add_task("Training models", total=len(pairs))

            for sym, tf in pairs:
                progress.update(task, description=f"Training {sym} {tf}")

                try:
                    df = load_candles(sym, tf, Path(data_dir))
                except FileNotFoundError:
                    progress.log(f"[red]{sym} {tf}: no data[/red]")
                    progress.advance(task)
                    continue

                try:
                    external_df = build_external_df(df, ext_dir)
                    trainer = MLWalkForwardTrainer(
                        symbol=sym,
                        timeframe=tf,
                        train_months=train_months,
                        test_months=test_months,
                        target_kind=target_kind,
                        atr_mult=atr_mult,
                        include_extra=include_extra,
                        model_dir=Path(model_dir),
                        lgbm_params={"num_threads": threads_per_worker},
                    )
                    report = trainer.run(df, external_df=external_df)

                    if report.windows:
                        progress.log(
                            f"[green]{sym} {tf}: AUC={report.avg_auc:.4f} "
                            f"precision={report.avg_precision:.4f} "
                            f"holdout={report.holdout_auc:.4f} "
                            f"windows={len(report.windows)}[/green]"
                        )
                        results.append(
                            {
                                "symbol": sym,
                                "timeframe": tf,
                                "avg_auc": report.avg_auc,
                                "avg_precision": report.avg_precision,
                                "holdout_auc": report.holdout_auc,
                                "holdout_precision": report.holdout_precision,
                                "n_windows": len(report.windows),
                                "model_path": str(report.model_path),
                            }
                        )
                    else:
                        progress.log(f"[yellow]{sym} {tf}: insufficient data[/yellow]")
                except Exception as e:
                    progress.log(f"[red]{sym} {tf}: error: {e}[/red]")

                progress.advance(task)
    else:
        # Parallel — ProcessPoolExecutor with spawn context
        from concurrent.futures import ProcessPoolExecutor, as_completed

        from tradingbot.ml.parallel import train_pair

        ctx = mp.get_context("spawn")
        data_dir_abs = str(Path(data_dir).resolve())
        model_dir_abs = str(Path(model_dir).resolve())
        ext_dir = Path(data_dir) / EXTERNAL_SUBDIR
        ext_dir_abs = str(ext_dir.resolve()) if ext_dir.exists() else None

        with _progress_context() as progress:
            task = progress.add_task("Training models", total=len(pairs))

            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
                futures = {
                    executor.submit(
                        train_pair,
                        sym,
                        tf,
                        data_dir_abs,
                        model_dir_abs,
                        train_months,
                        test_months,
                        threads_per_worker,
                        ext_dir_abs,
                        target_kind,
                        atr_mult,
                        include_extra,
                    ): (sym, tf)
                    for sym, tf in pairs
                }

                try:
                    for future in as_completed(futures):
                        sym, tf = futures[future]
                        try:
                            r = future.result()
                        except Exception as exc:
                            progress.log(f"[red]{sym} {tf}: unexpected error: {exc}[/red]")
                            progress.advance(task)
                            continue

                        if r.error:
                            color = "yellow" if r.error == "no data" else "red"
                            progress.log(f"[{color}]{sym} {tf}: {r.error}[/{color}]")
                        elif r.n_windows == 0:
                            progress.log(f"[yellow]{sym} {tf}: insufficient data[/yellow]")
                        else:
                            progress.log(
                                f"[green]{sym} {tf}: AUC={r.avg_auc:.4f} "
                                f"precision={r.avg_precision:.4f} "
                                f"holdout={r.holdout_auc:.4f} "
                                f"windows={r.n_windows}[/green]"
                            )
                            results.append(
                                {
                                    "symbol": sym,
                                    "timeframe": tf,
                                    "avg_auc": r.avg_auc,
                                    "avg_precision": r.avg_precision,
                                    "holdout_auc": r.holdout_auc,
                                    "holdout_precision": r.holdout_precision,
                                    "n_windows": r.n_windows,
                                    "model_path": r.model_path,
                                }
                            )

                        progress.advance(task)
                except KeyboardInterrupt:
                    console.print("\n[yellow]Interrupted. Cancelling...[/yellow]")
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise typer.Exit(130)

    if not results:
        console.print("\n[red]No models were trained.[/red]")
        raise typer.Exit(1)

    # Summary table
    table = Table(title=f"\nML Training Summary ({len(results)} models)")
    table.add_column("Symbol")
    table.add_column("TF")
    table.add_column("AUC", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Holdout AUC", justify="right")
    table.add_column("Windows", justify="right")

    for r in sorted(results, key=lambda x: x["avg_auc"], reverse=True):
        auc_style = "green" if r["avg_auc"] > 0.55 else ("yellow" if r["avg_auc"] > 0.50 else "red")
        holdout = r.get("holdout_auc", 0.0)
        holdout_style = "green" if holdout > 0.55 else ("yellow" if holdout > 0.50 else "red")
        table.add_row(
            r["symbol"],
            r["timeframe"],
            f"[{auc_style}]{r['avg_auc']:.4f}[/{auc_style}]",
            f"{r['avg_precision']:.4f}",
            f"[{holdout_style}]{holdout:.4f}[/{holdout_style}]",
            str(r["n_windows"]),
        )

    console.print(table)


@app.command(name="ml-diagnostics")
def ml_diagnostics(
    symbol: str = typer.Option("BTC/KRW", "--symbol", "-s", help="Symbol"),
    timeframe: str = typer.Option("4h", "--timeframe", "-t", help="Timeframe"),
    train_months: int = typer.Option(6, "--train-months", help="Training window months"),
    test_months: int = typer.Option(2, "--test-months", help="Test window months"),
    forward_candles: int = typer.Option(4, "--forward-candles", help="Target horizon"),
    threshold: float = typer.Option(
        0.006, "--threshold", help="Binary target return threshold (used when target-kind=binary)"
    ),
    target_kind: str = typer.Option(
        "binary",
        "--target-kind",
        help="Target labelling strategy: binary | atr | triple-barrier",
    ),
    atr_mult: float = typer.Option(
        1.0,
        "--atr-mult",
        help="ATR multiplier for atr / triple-barrier targets (ignored when target-kind=binary)",
    ),
    include_extra: bool = typer.Option(
        False,
        "--include-extra",
        help="Add Phase 4 extra features (regime, lag/diff, session)",
    ),
    entry_threshold: float = typer.Option(
        0.45, "--entry-threshold", help="Entry probability threshold"
    ),
    exit_threshold: float = typer.Option(
        0.30, "--exit-threshold", help="Exit probability threshold"
    ),
    balance: float = typer.Option(
        1_000_000, "--balance", "-b", help="Initial balance for backtest (KRW)"
    ),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
    model_dir: str = typer.Option("models", "--model-dir", help="Model output directory"),
    output_dir: str = typer.Option(
        "personal/ml_iter", "--output-dir", help="Where to write the diagnostics report"
    ),
    label: str = typer.Option("00_baseline", "--label", help="Filename prefix for the report"),
    skip_backtest: bool = typer.Option(
        False, "--skip-backtest", help="Skip MLStrategyWalkForward (model metrics only)"
    ),
) -> None:
    """Combined ML model + strategy diagnostics for a single (symbol, timeframe).

    Trains a fresh model via ``MLWalkForwardTrainer`` (Path B), reads back its
    holdout-eval predictions to compute calibration error and prediction
    distribution, then runs ``MLStrategyWalkForward`` for per-window backtest
    metrics. Writes both a human-readable markdown table and a JSON dump for
    later iteration comparisons.
    """
    setup_logging()

    import json

    from tradingbot.data.external_fetcher import build_external_df
    from tradingbot.data.storage import load_candles
    from tradingbot.ml.diagnostics import (
        evaluate_calibration,
        summarize_distribution,
        top_features,
    )
    from tradingbot.ml.strategy_walk_forward import MLStrategyWalkForward
    from tradingbot.ml.walk_forward import MLWalkForwardTrainer

    try:
        df = load_candles(symbol, timeframe, Path(data_dir))
    except FileNotFoundError:
        console.print(
            f"[red]No data for {symbol} {timeframe}. Run tradingbot download first.[/red]"
        )
        raise typer.Exit(1)

    ext_dir = Path(data_dir) / EXTERNAL_SUBDIR
    external_df = build_external_df(df, ext_dir) if ext_dir.exists() else None
    has_external = external_df is not None and len(external_df.columns) > 0

    console.print(f"[bold]ML diagnostics — {symbol} {timeframe}[/bold]")
    console.print(f"  Data: {len(df)} candles ({df.index[0]} → {df.index[-1]})")
    console.print(f"  Walk-Forward: {train_months}m train / {test_months}m test")
    console.print(f"  External data: {'yes' if has_external else 'no'}")
    if target_kind == "binary":
        console.print(f"  Target: binary (forward={forward_candles}, threshold={threshold})")
    else:
        console.print(f"  Target: {target_kind} (forward={forward_candles}, atr_mult={atr_mult})")

    # ---- Step 1: model walk-forward (produces holdout AUC, calibrated probs) ----
    trainer = MLWalkForwardTrainer(
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
    )
    model_report = trainer.run(df, external_df=external_df)
    if not model_report.windows:
        console.print("[red]Training failed — insufficient data.[/red]")
        raise typer.Exit(1)

    # ---- Step 2: calibration + distribution metrics on the holdout eval half ----
    calibration = (
        evaluate_calibration(
            y_true=model_report.holdout_y_true,
            raw_proba=model_report.holdout_raw_proba,
            calibrated_proba=model_report.holdout_calibrated_proba,
        )
        if model_report.holdout_y_true is not None
        else None
    )

    distribution = (
        summarize_distribution(
            model_report.holdout_calibrated_proba
            if model_report.holdout_calibrated_proba is not None
            else (
                model_report.holdout_raw_proba
                if model_report.holdout_raw_proba is not None
                else None
            ),
            thresholds=(exit_threshold, entry_threshold, 0.50),
        )
        if (
            model_report.holdout_raw_proba is not None
            or model_report.holdout_calibrated_proba is not None
        )
        else None
    )

    feat_top10 = top_features(model_report.feature_importance, top_n=10)

    # ---- Step 3: strategy walk-forward (per-window backtest) ----
    strategy_report = None
    if not skip_backtest:
        # Route through load_config so risk/backtest (fee rate, slippage, stop
        # loss, etc.) honor config/*.yaml instead of hardcoded defaults.
        config = load_config(
            overrides={
                "trading": {
                    "symbols": [symbol],
                    "timeframe": timeframe,
                    "initial_balance": balance,
                }
            }
        )
        runner = MLStrategyWalkForward(
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
            external_data_dir=ext_dir if has_external else None,
            config=config,
        )
        strategy_report = runner.run(df)

    # ---- Step 4: print + persist ----
    console.print("\n[bold]Model Quality (holdout eval half)[/bold]")
    console.print(f"  Inner-val AUC: {model_report.avg_auc:.4f}")
    console.print(f"  Holdout AUC: {model_report.holdout_auc:.4f}")
    console.print(f"  Holdout Precision: {model_report.holdout_precision:.4f}")
    if calibration is not None:
        console.print(
            f"  Brier (raw / calibrated): {calibration.brier_raw:.4f} / "
            f"{calibration.brier_calibrated:.4f}"
        )
        console.print(
            f"  ECE (raw / calibrated): {calibration.ece_raw:.4f} / "
            f"{calibration.ece_calibrated:.4f}    MCE: {calibration.mce_calibrated:.4f}"
        )

    if distribution is not None:
        console.print("\n[bold]Calibrated Probability Distribution[/bold]")
        console.print(
            f"  min={distribution.min:.3f}  p25={distribution.p25:.3f}  "
            f"p50={distribution.p50:.3f}  mean={distribution.mean:.3f}  "
            f"p75={distribution.p75:.3f}  max={distribution.max:.3f}"
        )
        for thr_key, n_above in distribution.above.items():
            pct = distribution.above_pct[thr_key] * 100
            console.print(f"  ≥ {thr_key}: {n_above} ({pct:.1f}%)")

    if feat_top10:
        console.print("\n[bold]Top 10 Feature Importance (gain)[/bold]")
        for i, (feat, imp) in enumerate(feat_top10, 1):
            console.print(f"  {i:2d}. {feat}: {imp:.1f}")

    if strategy_report is not None and strategy_report.windows:
        console.print("\n[bold]Strategy Walk-Forward Backtest[/bold]")
        console.print(
            f"  Windows: {strategy_report.n_windows}  (skipped: {strategy_report.n_skipped})"
        )
        console.print(f"  Avg Sharpe: {strategy_report.avg_sharpe:.2f}")
        console.print(f"  Cumulative return: {strategy_report.cumulative_return_pct:+.2f}%")
        console.print(f"  Total trades: {strategy_report.total_trades}")
        console.print(f"  Avg win rate (traded windows): {strategy_report.avg_win_rate * 100:.1f}%")

    # ---- Persist ----
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{label}_{symbol.replace('/', '_')}_{timeframe}"
    json_path = out_dir / f"{base}.json"
    md_path = out_dir / f"{base}.md"

    payload: dict = {
        "symbol": symbol,
        "timeframe": timeframe,
        "train_months": train_months,
        "test_months": test_months,
        "forward_candles": forward_candles,
        "threshold": threshold,
        "target_kind": target_kind,
        "atr_mult": atr_mult,
        "entry_threshold": entry_threshold,
        "exit_threshold": exit_threshold,
        "initial_balance": balance,
        "has_external": has_external,
        "n_data": len(df),
        "data_start": str(df.index[0]),
        "data_end": str(df.index[-1]),
        "model": {
            "inner_val_auc": model_report.avg_auc,
            "inner_val_precision": model_report.avg_precision,
            "holdout_auc": model_report.holdout_auc,
            "holdout_precision": model_report.holdout_precision,
        },
        "calibration": (
            {
                "n_samples": calibration.n_samples,
                "positive_rate": calibration.positive_rate,
                "brier_raw": calibration.brier_raw,
                "brier_calibrated": calibration.brier_calibrated,
                "ece_raw": calibration.ece_raw,
                "ece_calibrated": calibration.ece_calibrated,
                "mce_calibrated": calibration.mce_calibrated,
            }
            if calibration is not None
            else None
        ),
        "distribution": (
            {
                "n": distribution.n,
                "min": distribution.min,
                "p25": distribution.p25,
                "p50": distribution.p50,
                "mean": distribution.mean,
                "p75": distribution.p75,
                "max": distribution.max,
                "above": distribution.above,
                "above_pct": distribution.above_pct,
            }
            if distribution is not None
            else None
        ),
        "feature_importance_top10": feat_top10,
        "backtest": (
            {
                "n_windows": strategy_report.n_windows,
                "n_skipped": strategy_report.n_skipped,
                "avg_sharpe": strategy_report.avg_sharpe,
                "cumulative_return_pct": strategy_report.cumulative_return_pct,
                "final_equity_multiple": strategy_report.final_equity_multiple,
                "total_trades": strategy_report.total_trades,
                "avg_win_rate": strategy_report.avg_win_rate,
                "windows": strategy_report.windows,
            }
            if strategy_report is not None
            else None
        ),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    md_lines = [
        f"# ML Diagnostics — {symbol} {timeframe} ({label})",
        "",
        f"- Data: {len(df)} candles ({df.index[0]} → {df.index[-1]})",
        f"- Walk-Forward: {train_months}m train / {test_months}m test",
        (
            f"- Target: binary (forward_candles={forward_candles}, threshold={threshold})"
            if target_kind == "binary"
            else f"- Target: {target_kind} (forward_candles={forward_candles}, atr_mult={atr_mult})"
        ),
        f"- Entry/Exit threshold: {entry_threshold} / {exit_threshold}",
        f"- External features: {'yes' if has_external else 'no'}",
        "",
        "## Model Quality (holdout eval half)",
        "",
        f"- Inner-val AUC: {model_report.avg_auc:.4f}",
        f"- Holdout AUC: **{model_report.holdout_auc:.4f}**",
        f"- Holdout Precision: {model_report.holdout_precision:.4f}",
    ]
    if calibration is not None:
        md_lines += [
            f"- Brier (raw / calibrated): {calibration.brier_raw:.4f} / "
            f"{calibration.brier_calibrated:.4f}",
            f"- ECE (raw / calibrated): {calibration.ece_raw:.4f} / "
            f"{calibration.ece_calibrated:.4f}",
            f"- MCE (calibrated): {calibration.mce_calibrated:.4f}",
            f"- Eval positive rate: {calibration.positive_rate:.4f}",
        ]
    if distribution is not None:
        md_lines += [
            "",
            "## Calibrated Probability Distribution (eval half)",
            "",
            "| Stat | Value |",
            "|------|------:|",
            f"| min | {distribution.min:.3f} |",
            f"| p25 | {distribution.p25:.3f} |",
            f"| p50 | {distribution.p50:.3f} |",
            f"| mean | {distribution.mean:.3f} |",
            f"| p75 | {distribution.p75:.3f} |",
            f"| max | {distribution.max:.3f} |",
        ]
        for thr_key, n_above in distribution.above.items():
            pct = distribution.above_pct[thr_key] * 100
            md_lines.append(f"| ≥ {thr_key} | {n_above} ({pct:.1f}%) |")

    if feat_top10:
        md_lines += [
            "",
            "## Top 10 Feature Importance",
            "",
            "| # | Feature | Gain |",
            "|---|---------|-----:|",
        ]
        for i, (feat, imp) in enumerate(feat_top10, 1):
            md_lines.append(f"| {i} | {feat} | {imp:.1f} |")

    if strategy_report is not None and strategy_report.windows:
        md_lines += [
            "",
            "## Strategy Walk-Forward Backtest",
            "",
            f"- Windows: {strategy_report.n_windows} (skipped: {strategy_report.n_skipped})",
            f"- Avg Sharpe: **{strategy_report.avg_sharpe:.2f}**",
            f"- Cumulative return: **{strategy_report.cumulative_return_pct:+.2f}%**",
            f"- Total trades: {strategy_report.total_trades}",
            f"- Avg win rate (traded windows): {strategy_report.avg_win_rate * 100:.1f}%",
            "",
            "| # | Test Start | Test End | Sharpe | Return % | Trades | Win % | MaxDD % |",
            "|---|-----------|----------|-------:|---------:|-------:|------:|--------:|",
        ]
        for w in strategy_report.windows:
            md_lines.append(
                f"| {w['window']} | {w['test_start'][:10]} | {w['test_end'][:10]} | "
                f"{w['sharpe']:.2f} | {w['return_pct']:+.2f} | {w['trades']} | "
                f"{w['win_rate'] * 100:.1f} | {w['max_dd_pct']:.2f} |"
            )

    md_path.write_text("\n".join(md_lines) + "\n")
    console.print(f"\n[dim]JSON: {json_path}[/dim]")
    console.print(f"[dim]Markdown: {md_path}[/dim]")


@app.command(name="ml-tune")
def ml_tune(
    symbol: str = typer.Option("BTC/KRW", "--symbol", "-s", help="Symbol"),
    timeframe: str = typer.Option("4h", "--timeframe", "-t", help="Timeframe"),
    train_months: int = typer.Option(6, "--train-months", help="Training window months"),
    test_months: int = typer.Option(2, "--test-months", help="Test window months"),
    forward_candles: int = typer.Option(4, "--forward-candles", help="Target horizon"),
    threshold: float = typer.Option(0.006, "--threshold", help="Binary target return threshold"),
    target_kind: str = typer.Option(
        "triple-barrier",
        "--target-kind",
        help="Target labelling strategy: binary | atr | triple-barrier",
    ),
    atr_mult: float = typer.Option(
        1.0, "--atr-mult", help="ATR multiplier for atr / triple-barrier targets"
    ),
    include_extra: bool = typer.Option(
        False,
        "--include-extra",
        help="Add Phase 4 extra features (regime, lag/diff, session)",
    ),
    entry_threshold: float = typer.Option(
        0.45, "--entry-threshold", help="Entry probability threshold"
    ),
    exit_threshold: float = typer.Option(
        0.30, "--exit-threshold", help="Exit probability threshold"
    ),
    balance: float = typer.Option(
        1_000_000, "--balance", "-b", help="Initial balance for backtest (KRW)"
    ),
    trials: int = typer.Option(
        50, "--trials", help="Optuna trial budget (study stops at this many)"
    ),
    time_budget: float = typer.Option(
        3600.0,
        "--time-budget",
        help="Wall-clock budget in seconds; study stops at trial count or this, whichever first",
    ),
    objective: str = typer.Option(
        "holdout_sharpe",
        "--objective",
        help="Objective metric: holdout_sharpe | holdout_cum_return | holdout_auc",
    ),
    seed: int = typer.Option(42, "--seed", help="Optuna sampler seed"),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
    model_dir: str = typer.Option("models", "--model-dir", help="Model output directory"),
    output_dir: str = typer.Option(
        "personal/ml_iter", "--output-dir", help="Where to write the tuner report"
    ),
    label: str = typer.Option("02_tuned", "--label", help="Filename prefix for the tuner report"),
) -> None:
    """Tune LGBM hyperparameters via Optuna, then save a model with the best params.

    Each trial runs a full ``MLStrategyWalkForward`` so the objective stays
    aligned with what we report (Phase 2 surfaced that AUC alone misleads
    when the label distribution shifts). After the study finishes the tuner
    invokes ``MLWalkForwardTrainer`` with the winning params to persist a
    final model + calibrator the rest of the pipeline can consume.
    """
    setup_logging()

    import json

    from tradingbot.data.external_fetcher import build_external_df
    from tradingbot.data.storage import load_candles
    from tradingbot.ml.tuner import LGBMTuner, reserve_tuning_window
    from tradingbot.ml.walk_forward import MLWalkForwardTrainer

    # Load the user's AppConfig so the tuner respects fee rate, slippage,
    # risk settings, etc. Falls back to defaults when no config dir exists.
    app_config = load_config(overrides={"trading": {"initial_balance": balance}})

    try:
        df = load_candles(symbol, timeframe, Path(data_dir))
    except FileNotFoundError:
        console.print(
            f"[red]No data for {symbol} {timeframe}. Run tradingbot download first.[/red]"
        )
        raise typer.Exit(1)

    ext_dir = Path(data_dir) / EXTERNAL_SUBDIR
    external_df = build_external_df(df, ext_dir) if ext_dir.exists() else None
    has_external = external_df is not None and len(external_df.columns) > 0
    ext_dir_for_runner = ext_dir if has_external else None

    console.print(f"[bold]ML hyperparameter tuning — {symbol} {timeframe}[/bold]")
    console.print(f"  Data: {len(df)} candles ({df.index[0]} → {df.index[-1]})")
    console.print(f"  Walk-Forward: {train_months}m train / {test_months}m test")
    console.print(
        f"  Target: {target_kind}"
        + (f" (atr_mult={atr_mult})" if target_kind != "binary" else f" (threshold={threshold})")
    )
    console.print(f"  Trials: {trials}  Time budget: {time_budget:.0f}s")
    console.print(f"  Objective: {objective}")

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
        config=app_config,
        objective=objective,
        seed=seed,
    )
    # Tune on the inner window only; the trailing outer-holdout is reserved so
    # the final model's holdout stays unseen by the search.
    result = tuner.search(reserve_tuning_window(df), n_trials=trials, time_budget_sec=time_budget)

    if not result.best_params:
        console.print("[red]No successful trial — no model saved.[/red]")
        raise typer.Exit(1)

    console.print(
        f"\n[bold green]Tuning complete[/bold green]  "
        f"trials={result.n_trials_completed}  "
        f"elapsed={result.elapsed_sec:.1f}s  "
        f"best {objective}={result.best_value:.4f}"
    )

    # Show top 5 trials by score
    top_trials = sorted(
        [t for t in result.trials if t["score"] != float("-inf")],
        key=lambda t: -t["score"],
    )[:5]
    if top_trials:
        table = Table(title="Top trials")
        table.add_column("#", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Sharpe", justify="right")
        table.add_column("Cum %", justify="right")
        table.add_column("Trades", justify="right")
        for t in top_trials:
            table.add_row(
                str(t["trial"]),
                f"{t['score']:.3f}",
                f"{t.get('avg_sharpe', 0):.2f}",
                f"{t.get('cumulative_return_pct', 0):+.2f}",
                str(t.get("total_trades", 0)),
            )
        console.print(table)

    console.print("\n[bold]Best params[/bold]")
    for k, v in result.best_params.items():
        if isinstance(v, float):
            console.print(f"  {k}: {v:.4f}")
        else:
            console.print(f"  {k}: {v}")

    # ---- Train final model with best params ----
    console.print("\n[bold]Training final model with best params...[/bold]")
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
        lgbm_params=dict(result.best_params),
    )
    final_report = final_trainer.run(df, external_df=external_df)
    if not final_report.windows:
        console.print(
            "[red]Final training failed (no windows).[/red] Best params still saved to report."
        )
        final_model_path: Path | None = None
    else:
        final_model_path = final_report.model_path
        console.print(f"[green]Final model saved: {final_model_path}[/green]")
        console.print(
            f"  Holdout AUC: {final_report.holdout_auc:.4f}  "
            f"precision: {final_report.holdout_precision:.4f}"
        )

        # Patch the model meta with tuning info so downstream tools can audit
        # which params produced the saved booster. We rewrite the file rather
        # than threading a callback because trainer.save() owns meta layout.
        # Write to a sibling temp file first then os.replace — keeps the meta
        # file uncorrupted if the process is interrupted mid-write.
        if final_model_path is not None:
            import os

            symbol_key = symbol.replace("/", "_")
            meta_path = Path(model_dir) / f"lgbm_{symbol_key}_{timeframe}_meta.json"
            if meta_path.exists():
                meta_dict = json.loads(meta_path.read_text())
                meta_dict["tuning"] = {
                    "best_params": dict(result.best_params),
                    "best_value": result.best_value,
                    "objective": objective,
                    "n_trials_completed": result.n_trials_completed,
                    "elapsed_sec": result.elapsed_sec,
                }
                tmp_path = meta_path.with_suffix(".json.tmp")
                tmp_path.write_text(json.dumps(meta_dict, indent=2, default=str))
                os.replace(tmp_path, meta_path)

    # ---- Persist tuner report ----
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{label}_{symbol.replace('/', '_')}_{timeframe}"
    json_path = out_dir / f"{base}.json"
    md_path = out_dir / f"{base}.md"

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
        "time_budget_sec": time_budget,
        "n_trials_completed": result.n_trials_completed,
        "elapsed_sec": result.elapsed_sec,
        "best_params": result.best_params,
        "best_value": result.best_value,
        "trials": result.trials,
        "final_holdout_auc": (final_report.holdout_auc if final_report.windows else None),
        "final_holdout_precision": (
            final_report.holdout_precision if final_report.windows else None
        ),
        "final_model_path": str(final_model_path) if final_model_path else None,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    md_lines = [
        f"# ML Tuning — {symbol} {timeframe} ({label})",
        "",
        f"- Data: {len(df)} candles ({df.index[0]} → {df.index[-1]})",
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
            f"- Holdout AUC: {final_report.holdout_auc:.4f}",
            f"- Holdout Precision: {final_report.holdout_precision:.4f}",
            f"- Saved: `{final_model_path}`",
        ]
    md_path.write_text("\n".join(md_lines) + "\n")
    console.print(f"\n[dim]JSON: {json_path}[/dim]")
    console.print(f"[dim]Markdown: {md_path}[/dim]")


@app.command(name="ml-tune-all")
def ml_tune_all(
    timeframe: str | None = typer.Option(
        None, "--timeframe", "-t", help="Restrict to this timeframe (default: all)"
    ),
    train_months: int = typer.Option(6, "--train-months"),
    test_months: int = typer.Option(2, "--test-months"),
    forward_candles: int = typer.Option(4, "--forward-candles"),
    threshold: float = typer.Option(0.006, "--threshold"),
    target_kind: str = typer.Option("triple-barrier", "--target-kind"),
    atr_mult: float = typer.Option(1.0, "--atr-mult"),
    include_extra: bool = typer.Option(False, "--include-extra"),
    entry_threshold: float = typer.Option(0.45, "--entry-threshold"),
    exit_threshold: float = typer.Option(0.30, "--exit-threshold"),
    balance: float = typer.Option(1_000_000, "--balance", "-b"),
    trials: int = typer.Option(50, "--trials"),
    time_budget: float = typer.Option(3600.0, "--time-budget"),
    objective: str = typer.Option("holdout_sharpe", "--objective"),
    seed: int = typer.Option(42, "--seed"),
    data_dir: str = typer.Option("data", "--data-dir"),
    model_dir: str = typer.Option("models", "--model-dir"),
    output_dir: str = typer.Option("personal/ml_iter", "--output-dir"),
    label: str = typer.Option("06_tune_all", "--label"),
    workers: int = typer.Option(
        0,
        "--workers",
        "-w",
        help="Parallel workers (0=auto: cpu_count//2, 1=sequential)",
    ),
) -> None:
    """Run Optuna tuning for every saved (symbol, timeframe) model.

    Iterates ``models/lgbm_*_meta.json``, runs ``LGBMTuner`` per model with
    the provided trials/time budget, then trains a final model with the
    best params (and patches its meta with the tuning record). Per-model
    JSON+MD reports land under ``output_dir/{label}_{sym}_{tf}.{json,md}``;
    a summary report sortable by ``best_value`` is written as
    ``{label}_summary.{json,md}``. Designed for Phase 6 — replaces the
    24-times manual ``ml-tune`` invocation needed before re-running
    ``scan`` / ``combine-scan``.
    """
    setup_logging()

    import json
    import multiprocessing as mp

    # Enumerate saved models the same way ml-tune-thresholds-all does.
    model_path = Path(model_dir)
    if not model_path.exists():
        console.print(f"[red]Model dir does not exist: {model_path}[/red]")
        raise typer.Exit(1)

    targets: list[tuple[str, str]] = []
    for meta_file in sorted(model_path.glob("lgbm_*_meta.json")):
        stem = meta_file.stem.removeprefix("lgbm_").removesuffix("_meta")
        if "_" not in stem:
            continue
        sym_key, tf = stem.rsplit("_", 1)
        sym = sym_key.replace("_", "/")
        if timeframe is not None and tf != timeframe:
            continue
        targets.append((sym, tf))

    if not targets:
        console.print(
            f"[yellow]No models found in {model_path}"
            + (f" matching timeframe={timeframe}" if timeframe else "")
            + "[/yellow]"
        )
        raise typer.Exit(1)

    cpu_count = mp.cpu_count()
    if workers <= 0:
        n_workers = max(1, min(cpu_count // 2, len(targets)))
    else:
        n_workers = min(workers, len(targets))
    threads_per_worker = max(1, cpu_count // n_workers)

    console.print(f"[bold]Optuna tuning — {len(targets)} (symbol, timeframe) models[/bold]")
    console.print(f"  Walk-Forward: {train_months}m train / {test_months}m test")
    console.print(
        f"  Target: {target_kind}"
        + (f" (atr_mult={atr_mult})" if target_kind != "binary" else f" (threshold={threshold})")
    )
    console.print(f"  Objective: {objective}")
    console.print(f"  Trials/model: {trials}  Time budget: {time_budget:.0f}s")
    console.print(f"  Workers: {n_workers}  (threads/worker: {threads_per_worker})")

    ext_dir = Path(data_dir) / EXTERNAL_SUBDIR
    ext_dir_abs = str(ext_dir.resolve()) if ext_dir.exists() else None
    data_dir_abs = str(Path(data_dir).resolve())
    model_dir_abs = str(model_path.resolve())
    output_dir_abs = str(Path(output_dir).resolve())

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    failed: list[tuple[str, str, str]] = []

    def _record_result(sym: str, tf: str, r) -> None:
        # ``no_successful_trial``, ``no_data``, etc. surface as failures
        # because the per-model meta wasn't updated and there's no Sharpe
        # to compare. Anything with a usable best_value goes into rows.
        if r.error and r.best_value == float("-inf"):
            failed.append((sym, tf, r.error))
            return
        rows.append(
            {
                "symbol": sym,
                "timeframe": tf,
                "objective": r.objective,
                "best_value": r.best_value,
                "n_trials_completed": r.n_trials_completed,
                "elapsed_sec": r.elapsed_sec,
                "final_holdout_auc": r.final_holdout_auc,
                "final_holdout_precision": r.final_holdout_precision,
                "final_model_path": r.final_model_path,
                "best_params": r.best_params,
                "error": r.error,
            }
        )

    if n_workers == 1:
        # Sequential — no worker, but still wrap each model in a broad
        # try/except so a single crash doesn't kill the run.
        from tradingbot.ml.parallel import tune_pair

        config_dump = load_config(overrides={"trading": {"initial_balance": balance}}).model_dump()

        with _progress_context() as progress:
            task = progress.add_task("Tuning models", total=len(targets))
            for sym, tf in targets:
                progress.update(task, description=f"Tuning {sym} {tf}")
                try:
                    r = tune_pair(
                        sym,
                        tf,
                        data_dir_abs,
                        model_dir_abs,
                        ext_dir_abs,
                        train_months,
                        test_months,
                        forward_candles,
                        threshold,
                        target_kind,
                        atr_mult,
                        include_extra,
                        entry_threshold,
                        exit_threshold,
                        balance,
                        trials,
                        time_budget,
                        objective,
                        seed,
                        output_dir_abs,
                        label,
                        threads_per_worker,
                        config_dump,
                    )
                except Exception as exc:
                    failed.append((sym, tf, f"unexpected: {exc}"))
                    progress.log(f"[red]{sym} {tf}: unexpected error: {exc}[/red]")
                    progress.advance(task)
                    continue

                _record_result(sym, tf, r)
                if r.error and r.best_value == float("-inf"):
                    color = "yellow" if r.error == "no_data" else "red"
                    progress.log(f"[{color}]{sym} {tf}: {r.error}[/{color}]")
                else:
                    progress.log(
                        f"[green]{sym} {tf}: {objective}={r.best_value:.4f} "
                        f"trials={r.n_trials_completed} elapsed={r.elapsed_sec:.0f}s[/green]"
                    )
                progress.advance(task)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        from tradingbot.ml.parallel import tune_pair

        ctx = mp.get_context("spawn")
        config_dump = load_config(overrides={"trading": {"initial_balance": balance}}).model_dump()

        with _progress_context() as progress:
            task = progress.add_task("Tuning models", total=len(targets))
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
                futures = {
                    executor.submit(
                        tune_pair,
                        sym,
                        tf,
                        data_dir_abs,
                        model_dir_abs,
                        ext_dir_abs,
                        train_months,
                        test_months,
                        forward_candles,
                        threshold,
                        target_kind,
                        atr_mult,
                        include_extra,
                        entry_threshold,
                        exit_threshold,
                        balance,
                        trials,
                        time_budget,
                        objective,
                        seed,
                        output_dir_abs,
                        label,
                        threads_per_worker,
                        config_dump,
                    ): (sym, tf)
                    for sym, tf in targets
                }
                try:
                    for future in as_completed(futures):
                        sym, tf = futures[future]
                        try:
                            r = future.result()
                        except Exception as exc:
                            failed.append((sym, tf, f"unexpected: {exc}"))
                            progress.log(f"[red]{sym} {tf}: unexpected error: {exc}[/red]")
                            progress.advance(task)
                            continue

                        _record_result(sym, tf, r)
                        if r.error and r.best_value == float("-inf"):
                            color = "yellow" if r.error == "no_data" else "red"
                            progress.log(f"[{color}]{sym} {tf}: {r.error}[/{color}]")
                        else:
                            progress.log(
                                f"[green]{sym} {tf}: {objective}={r.best_value:.4f} "
                                f"trials={r.n_trials_completed} "
                                f"elapsed={r.elapsed_sec:.0f}s[/green]"
                            )
                        progress.advance(task)
                except KeyboardInterrupt:
                    console.print("\n[yellow]Interrupted. Cancelling...[/yellow]")
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise

    # ---- Summary ----
    rows.sort(key=lambda r: r["best_value"], reverse=True)

    table = Table(title=f"Tune-all summary ({len(rows)} models)")
    table.add_column("Symbol")
    table.add_column("TF")
    table.add_column(objective, justify="right")
    table.add_column("Trials", justify="right")
    table.add_column("Elapsed s", justify="right")
    table.add_column("Holdout AUC", justify="right")
    for row in rows:
        auc_str = f"{row['final_holdout_auc']:.4f}" if row["final_holdout_auc"] is not None else "—"
        table.add_row(
            row["symbol"],
            row["timeframe"],
            f"{row['best_value']:.4f}",
            str(row["n_trials_completed"]),
            f"{row['elapsed_sec']:.0f}",
            auc_str,
        )
    console.print(table)

    if failed:
        console.print(f"\n[yellow]{len(failed)} models skipped:[/yellow]")
        for sym, tf, reason in failed:
            console.print(f"  {sym} {tf}: {reason}")

    summary_json = out_dir / f"{label}_summary.json"
    summary_md = out_dir / f"{label}_summary.md"
    summary_json.write_text(
        json.dumps(
            {
                "label": label,
                "timeframe_filter": timeframe,
                "objective": objective,
                "trials_per_model": trials,
                "time_budget_sec": time_budget,
                "target_kind": target_kind,
                "atr_mult": atr_mult,
                "include_extra": include_extra,
                "rows": rows,
                "failed": [{"symbol": s, "timeframe": t, "reason": r} for s, t, r in failed],
            },
            indent=2,
            default=str,
        )
    )

    md_lines = [
        f"# Optuna tuning — all models ({label})",
        "",
        f"- Models tuned: {len(rows)} (skipped: {len(failed)})",
        f"- Objective: **{objective}**",
        f"- Trials/model: {trials}  (time budget: {time_budget:.0f}s)",
        (
            f"- Target: binary (forward_candles={forward_candles}, threshold={threshold})"
            if target_kind == "binary"
            else f"- Target: {target_kind} (forward_candles={forward_candles}, atr_mult={atr_mult})"
        ),
        f"- Extras: {'on' if include_extra else 'off'}",
        "",
        f"## Summary (sorted by best {objective})",
        "",
        f"| Symbol | TF | {objective} | Trials | Elapsed s | Holdout AUC |",
        "|--------|----|------------:|-------:|----------:|------------:|",
    ]
    for row in rows:
        auc_str = f"{row['final_holdout_auc']:.4f}" if row["final_holdout_auc"] is not None else "—"
        md_lines.append(
            f"| {row['symbol']} | {row['timeframe']} | "
            f"{row['best_value']:.4f} | {row['n_trials_completed']} | "
            f"{row['elapsed_sec']:.0f} | {auc_str} |"
        )
    if failed:
        md_lines.extend(["", "## Skipped", ""])
        for sym, tf, reason in failed:
            md_lines.append(f"- {sym} {tf}: {reason}")

    summary_md.write_text("\n".join(md_lines) + "\n")
    console.print(f"\n[green]Summary JSON: {summary_json}[/green]")
    console.print(f"[green]Summary Markdown: {summary_md}[/green]")


@app.command(name="ml-tune-thresholds")
def ml_tune_thresholds(
    symbol: str = typer.Option("BTC/KRW", "--symbol", "-s", help="Symbol"),
    timeframe: str = typer.Option("4h", "--timeframe", "-t", help="Timeframe"),
    entry_grid: str = typer.Option(
        "0.40,0.45,0.50,0.55,0.60",
        "--entry-grid",
        help="Comma-separated entry threshold grid",
    ),
    exit_grid: str = typer.Option(
        "0.20,0.25,0.30,0.35",
        "--exit-grid",
        help="Comma-separated exit threshold grid",
    ),
    baseline_entry: float = typer.Option(
        0.45,
        "--baseline-entry",
        help="Entry threshold used as the baseline comparison row",
    ),
    baseline_exit: float = typer.Option(
        0.30,
        "--baseline-exit",
        help="Exit threshold used as the baseline comparison row",
    ),
    balance: float = typer.Option(
        1_000_000, "--balance", "-b", help="Initial balance for backtest (KRW)"
    ),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
    model_dir: str = typer.Option("models", "--model-dir", help="Model directory"),
    output_dir: str = typer.Option(
        "personal/ml_iter",
        "--output-dir",
        help="Where to write the threshold tuner report",
    ),
    label: str = typer.Option(
        "04_thresholds",
        "--label",
        help="Filename prefix for the tuner report",
    ),
    write_meta: bool = typer.Option(
        True,
        "--write-meta/--no-write-meta",
        help="Persist best thresholds back into the model meta file",
    ),
    min_trades: int = typer.Option(
        -1,
        "--min-trades",
        help=(
            "Floor on holdout trade count when picking the winning combo. "
            "-1 (default) auto-uses the baseline trade count so the tuner "
            "never recommends fewer trades. 0 disables the floor (any combo "
            "with at least 1 trade is eligible)."
        ),
    ),
) -> None:
    """Sweep entry/exit thresholds against the model's holdout window.

    Loads the saved booster + calibrator (no retraining), runs one cheap
    backtest per (entry, exit) combo on the holdout slice from meta.json,
    and writes the best pair back into the meta as ``entry_threshold`` /
    ``exit_threshold``. ``LGBMStrategy._load_model`` then reuses those
    overrides automatically on subsequent runs.
    """
    setup_logging()

    import json

    from tradingbot.data.storage import load_candles
    from tradingbot.ml.threshold_tuner import (
        ThresholdTuner,
        patch_meta_thresholds,
    )

    app_config = load_config(overrides={"trading": {"initial_balance": balance}})

    try:
        df = load_candles(symbol, timeframe, Path(data_dir))
    except FileNotFoundError:
        console.print(
            f"[red]No data for {symbol} {timeframe}. Run tradingbot download first.[/red]"
        )
        raise typer.Exit(1)

    try:
        entry_values = tuple(float(x) for x in entry_grid.split(",") if x.strip())
        exit_values = tuple(float(x) for x in exit_grid.split(",") if x.strip())
    except ValueError as exc:
        console.print(f"[red]Invalid grid value: {exc}[/red]")
        raise typer.Exit(1)
    if not entry_values or not exit_values:
        console.print("[red]entry_grid and exit_grid must each contain at least one value.[/red]")
        raise typer.Exit(1)

    ext_dir = Path(data_dir) / EXTERNAL_SUBDIR
    ext_dir_for_runner = ext_dir if ext_dir.exists() else None

    console.print(f"[bold]Threshold tuning — {symbol} {timeframe}[/bold]")
    console.print(f"  Data: {len(df)} candles ({df.index[0]} → {df.index[-1]})")
    console.print(f"  Entry grid: {list(entry_values)}")
    console.print(f"  Exit grid:  {list(exit_values)}")
    console.print(f"  Baseline:   entry={baseline_entry} exit={baseline_exit}")

    # ``-1`` is the CLI sentinel for "auto" — the tuner uses
    # ``baseline_trades`` as the floor so we never recommend a combo with
    # fewer trades than the current default. Any non-negative value is
    # forwarded verbatim (and the tuner floors it at 1 internally).
    min_trades_arg: int | None = None if min_trades < 0 else min_trades

    tuner = ThresholdTuner(
        symbol=symbol,
        timeframe=timeframe,
        model_dir=Path(model_dir),
        external_data_dir=ext_dir_for_runner,
        config=app_config,
        balance=balance,
        baseline_entry=baseline_entry,
        baseline_exit=baseline_exit,
        min_trades=min_trades_arg,
    )
    result = tuner.search(df, entry_grid=entry_values, exit_grid=exit_values)

    if result.error and not result.grid:
        console.print(f"[red]Threshold tuning failed: {result.error}[/red]")
        raise typer.Exit(1)

    console.print(
        f"\n[bold green]Best[/bold green]  "
        f"entry={result.best_entry:.2f} exit={result.best_exit:.2f}  "
        f"sharpe={result.best_sharpe:.3f} return={result.best_return_pct:+.2f}% "
        f"trades={result.best_trades}"
    )
    console.print(
        f"[dim]Baseline  entry={result.baseline_entry:.2f} exit={result.baseline_exit:.2f}  "
        f"sharpe={result.baseline_sharpe:.3f} return={result.baseline_return_pct:+.2f}% "
        f"trades={result.baseline_trades}[/dim]"
    )

    # Top combos by Sharpe (with at least one trade)
    sortable = [g for g in result.grid if g.get("trades", 0) > 0]
    sortable.sort(key=lambda g: (g["sharpe"], g["trades"]), reverse=True)
    if sortable:
        table = Table(title=f"Top combos — {symbol} {timeframe}")
        table.add_column("Entry", justify="right")
        table.add_column("Exit", justify="right")
        table.add_column("Sharpe", justify="right")
        table.add_column("Return %", justify="right")
        table.add_column("Trades", justify="right")
        table.add_column("Win %", justify="right")
        table.add_column("MaxDD %", justify="right")
        for g in sortable[:8]:
            table.add_row(
                f"{g['entry']:.2f}",
                f"{g['exit']:.2f}",
                f"{g['sharpe']:.3f}",
                f"{g['return_pct']:+.2f}",
                str(g["trades"]),
                f"{g['win_rate'] * 100:.1f}",
                f"{g['max_dd_pct']:+.2f}",
            )
        console.print(table)

    meta_path: Path | None = None
    if write_meta:
        meta_path = patch_meta_thresholds(symbol, timeframe, Path(model_dir), result)
        if meta_path is not None:
            console.print(f"[green]Meta updated: {meta_path}[/green]")
        else:
            console.print("[yellow]Meta not updated (missing file or no winning combo).[/yellow]")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{label}_{symbol.replace('/', '_')}_{timeframe}"
    json_path = out_dir / f"{base}.json"
    md_path = out_dir / f"{base}.md"

    payload = {
        "symbol": result.symbol,
        "timeframe": result.timeframe,
        "holdout_start": result.holdout_start,
        "holdout_end": result.holdout_end,
        "baseline_entry": result.baseline_entry,
        "baseline_exit": result.baseline_exit,
        "baseline_sharpe": result.baseline_sharpe,
        "baseline_return_pct": result.baseline_return_pct,
        "baseline_trades": result.baseline_trades,
        "best_entry": result.best_entry,
        "best_exit": result.best_exit,
        "best_sharpe": result.best_sharpe,
        "best_return_pct": result.best_return_pct,
        "best_trades": result.best_trades,
        "best_win_rate": result.best_win_rate,
        "best_max_dd_pct": result.best_max_dd_pct,
        "n_combos_evaluated": result.n_combos_evaluated,
        "n_combos_skipped": result.n_combos_skipped,
        "entry_grid": list(entry_values),
        "exit_grid": list(exit_values),
        "grid": result.grid,
        "meta_path": str(meta_path) if meta_path else None,
        "error": result.error,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    md_lines = [
        f"# Threshold tuning — {symbol} {timeframe} ({label})",
        "",
        f"- Holdout: {result.holdout_start} → {result.holdout_end}",
        f"- Combos evaluated: {result.n_combos_evaluated} (skipped: {result.n_combos_skipped})",
        f"- Best: **entry={result.best_entry:.2f} / exit={result.best_exit:.2f}**, "
        f"Sharpe **{result.best_sharpe:.3f}**, "
        f"Return **{result.best_return_pct:+.2f}%**, Trades {result.best_trades}",
        f"- Baseline (entry={result.baseline_entry:.2f}/exit={result.baseline_exit:.2f}): "
        f"Sharpe {result.baseline_sharpe:.3f}, Return {result.baseline_return_pct:+.2f}%, "
        f"Trades {result.baseline_trades}",
        "",
        "## Top combos",
        "",
        "| Entry | Exit | Sharpe | Return % | Trades | Win % | MaxDD % |",
        "|------:|-----:|-------:|---------:|-------:|------:|--------:|",
    ]
    for g in sortable[:12]:
        md_lines.append(
            f"| {g['entry']:.2f} | {g['exit']:.2f} | {g['sharpe']:.3f} | "
            f"{g['return_pct']:+.2f} | {g['trades']} | "
            f"{g['win_rate'] * 100:.1f} | {g['max_dd_pct']:+.2f} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n")
    console.print(f"[dim]JSON: {json_path}[/dim]")
    console.print(f"[dim]Markdown: {md_path}[/dim]")


@app.command(name="ml-tune-thresholds-all")
def ml_tune_thresholds_all(
    timeframe: str = typer.Option(
        None, "--timeframe", "-t", help="Restrict to this timeframe (default: all)"
    ),
    entry_grid: str = typer.Option(
        "0.40,0.45,0.50,0.55,0.60",
        "--entry-grid",
        help="Comma-separated entry threshold grid",
    ),
    exit_grid: str = typer.Option(
        "0.20,0.25,0.30,0.35",
        "--exit-grid",
        help="Comma-separated exit threshold grid",
    ),
    baseline_entry: float = typer.Option(0.45, "--baseline-entry"),
    baseline_exit: float = typer.Option(0.30, "--baseline-exit"),
    balance: float = typer.Option(1_000_000, "--balance", "-b"),
    data_dir: str = typer.Option("data", "--data-dir"),
    model_dir: str = typer.Option("models", "--model-dir"),
    output_dir: str = typer.Option(
        "personal/ml_iter",
        "--output-dir",
        help="Where to write the per-model + summary reports",
    ),
    label: str = typer.Option(
        "06_thresholds_all",
        "--label",
        help="Filename prefix for the summary report",
    ),
    write_meta: bool = typer.Option(
        True,
        "--write-meta/--no-write-meta",
        help="Persist best thresholds back into each model's meta file",
    ),
    workers: int = typer.Option(
        0,
        "--workers",
        "-w",
        help="Parallel workers (0=auto: cpu_count//2, 1=sequential)",
    ),
    min_trades: int = typer.Option(
        -1,
        "--min-trades",
        help=(
            "Floor on holdout trade count when picking each model's winning "
            "combo. -1 (default) auto-uses each model's baseline trade count "
            "so the tuner never recommends fewer trades. 0 disables the "
            "floor (any combo with at least 1 trade is eligible)."
        ),
    ),
) -> None:
    """Sweep entry/exit thresholds across every saved (symbol, timeframe) model.

    Enumerates ``models/lgbm_*_meta.json``, runs the per-model threshold tuner
    on each, and writes a combined summary (sortable by best Sharpe) to
    ``output_dir/{label}_summary.{json,md}``. Per-model reports are also
    emitted under the same ``label`` prefix so you can drill into any one
    result. Designed for Phase 6 — apply tuned thresholds to all 24 models
    before re-running ``scan`` / ``combine-scan``.
    """
    setup_logging()

    import json
    import multiprocessing as mp

    try:
        entry_values = tuple(float(x) for x in entry_grid.split(",") if x.strip())
        exit_values = tuple(float(x) for x in exit_grid.split(",") if x.strip())
    except ValueError as exc:
        console.print(f"[red]Invalid grid value: {exc}[/red]")
        raise typer.Exit(1)
    if not entry_values or not exit_values:
        console.print("[red]entry_grid and exit_grid must each contain at least one value.[/red]")
        raise typer.Exit(1)

    # Enumerate every saved model. ``meta.json`` files have the form
    # ``lgbm_{SYM_KEY}_{TF}_meta.json``; SYM_KEY uses ``_`` as the slash
    # replacement so we split on the *last* underscore to recover (sym, tf).
    model_path = Path(model_dir)
    if not model_path.exists():
        console.print(f"[red]Model dir does not exist: {model_path}[/red]")
        raise typer.Exit(1)

    targets: list[tuple[str, str]] = []
    for meta_file in sorted(model_path.glob("lgbm_*_meta.json")):
        stem = meta_file.stem.removeprefix("lgbm_").removesuffix("_meta")
        if "_" not in stem:
            continue
        sym_key, tf = stem.rsplit("_", 1)
        sym = sym_key.replace("_", "/")
        if timeframe is not None and tf != timeframe:
            continue
        targets.append((sym, tf))

    if not targets:
        console.print(
            f"[yellow]No models found in {model_path}"
            + (f" matching timeframe={timeframe}" if timeframe else "")
            + "[/yellow]"
        )
        raise typer.Exit(1)

    cpu_count = mp.cpu_count()
    if workers <= 0:
        n_workers = max(1, min(cpu_count // 2, len(targets)))
    else:
        n_workers = min(workers, len(targets))

    console.print(f"[bold]Threshold tuning — {len(targets)} (symbol, timeframe) models[/bold]")
    console.print(f"  Entry grid: {list(entry_values)}")
    console.print(f"  Exit grid:  {list(exit_values)}")
    console.print(f"  Baseline:   entry={baseline_entry} exit={baseline_exit}")
    console.print(
        "  Min trades: " + ("auto (= baseline)" if min_trades < 0 else str(max(min_trades, 1)))
    )
    console.print(f"  Workers:    {n_workers}")

    # ``-1`` is the CLI sentinel for "auto" — each tuner uses its own
    # ``baseline_trades`` as the floor. Any non-negative value is forwarded
    # verbatim (and the tuner floors it at 1 internally).
    min_trades_arg: int | None = None if min_trades < 0 else min_trades

    ext_dir = Path(data_dir) / EXTERNAL_SUBDIR
    ext_dir_for_runner = ext_dir if ext_dir.exists() else None
    ext_dir_abs = str(ext_dir.resolve()) if ext_dir_for_runner else None
    data_dir_abs = str(Path(data_dir).resolve())
    model_dir_abs = str(model_path.resolve())
    output_dir_abs = str(Path(output_dir).resolve())

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    failed: list[tuple[str, str, str]] = []

    def _record_result(sym: str, tf: str, r) -> None:
        """Translate a worker result into a summary row or failure entry."""
        # Non-fatal sentinel: search returned no usable grid (e.g. missing
        # meta.holdout_start, all combos zero-trade). The full reason is in
        # ``r.error``; keep the per-model JSON for drilling.
        if r.error and r.best_sharpe == float("-inf"):
            failed.append((sym, tf, r.error))
            return
        rows.append(
            {
                "symbol": sym,
                "timeframe": tf,
                "best_entry": r.best_entry,
                "best_exit": r.best_exit,
                "best_sharpe": r.best_sharpe,
                "best_return_pct": r.best_return_pct,
                "best_trades": r.best_trades,
                "best_win_rate": r.best_win_rate,
                "best_max_dd_pct": r.best_max_dd_pct,
                "baseline_sharpe": r.baseline_sharpe,
                "baseline_return_pct": r.baseline_return_pct,
                "baseline_trades": r.baseline_trades,
                "delta_sharpe": (
                    r.best_sharpe - r.baseline_sharpe
                    if r.baseline_sharpe != float("-inf")
                    else None
                ),
                "n_combos_evaluated": r.n_combos_evaluated,
                "holdout_start": r.holdout_start,
                "holdout_end": r.holdout_end,
                "meta_patched": r.meta_patched,
            }
        )

    if n_workers == 1:
        # Sequential — single process, broad per-model try/except so one
        # crash inside ThresholdTuner / patch_meta / file I/O doesn't kill
        # the batch (Gemini #33 review).
        from tradingbot.data.storage import load_candles
        from tradingbot.ml.parallel import ThresholdTunePairResult
        from tradingbot.ml.threshold_tuner import (
            ThresholdTuner,
            patch_meta_thresholds,
        )

        app_config = load_config(overrides={"trading": {"initial_balance": balance}})

        with _progress_context() as progress:
            task = progress.add_task("Tuning thresholds", total=len(targets))
            for sym, tf in targets:
                progress.update(task, description=f"Tuning {sym} {tf}")
                try:
                    try:
                        df = load_candles(sym, tf, Path(data_dir))
                    except FileNotFoundError:
                        failed.append((sym, tf, "no_data"))
                        progress.advance(task)
                        continue

                    tuner = ThresholdTuner(
                        symbol=sym,
                        timeframe=tf,
                        model_dir=model_path,
                        external_data_dir=ext_dir_for_runner,
                        config=app_config,
                        balance=balance,
                        baseline_entry=baseline_entry,
                        baseline_exit=baseline_exit,
                        min_trades=min_trades_arg,
                    )
                    result = tuner.search(df, entry_grid=entry_values, exit_grid=exit_values)

                    if result.error and not result.grid:
                        failed.append((sym, tf, result.error))
                        progress.advance(task)
                        continue

                    meta_path: Path | None = None
                    if write_meta:
                        meta_path = patch_meta_thresholds(sym, tf, model_path, result)

                    base = f"{label}_{sym.replace('/', '_')}_{tf}"
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
                                "entry_grid": list(entry_values),
                                "exit_grid": list(exit_values),
                                "grid": result.grid,
                                "meta_path": str(meta_path) if meta_path else None,
                                "error": result.error,
                            },
                            indent=2,
                            default=str,
                        )
                    )

                    pair_result = ThresholdTunePairResult(
                        symbol=sym,
                        timeframe=tf,
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
                    _record_result(sym, tf, pair_result)
                except Exception as exc:
                    failed.append((sym, tf, f"unexpected: {exc}"))
                    progress.log(f"[red]{sym} {tf}: unexpected error: {exc}[/red]")
                progress.advance(task)
    else:
        # Parallel — ProcessPoolExecutor with spawn context. Each worker
        # writes its own per-model JSON; parent only aggregates summary
        # rows. Broad except on the future result keeps one bad pair from
        # killing the batch (Gemini #33 review).
        from concurrent.futures import ProcessPoolExecutor, as_completed

        from tradingbot.ml.parallel import tune_thresholds_pair

        ctx = mp.get_context("spawn")
        # Pass the resolved config (YAML overrides + balance) into workers
        # via model_dump so spawn pickling is trivial and the worker's
        # backtests match the sequential path's sizing/fees/slippage.
        parent_config = load_config(overrides={"trading": {"initial_balance": balance}})
        config_dump = parent_config.model_dump()

        with _progress_context() as progress:
            task = progress.add_task("Tuning thresholds", total=len(targets))
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
                futures = {
                    executor.submit(
                        tune_thresholds_pair,
                        sym,
                        tf,
                        data_dir_abs,
                        model_dir_abs,
                        ext_dir_abs,
                        entry_values,
                        exit_values,
                        baseline_entry,
                        baseline_exit,
                        balance,
                        write_meta,
                        output_dir_abs,
                        label,
                        config_dump,
                        min_trades_arg,
                    ): (sym, tf)
                    for sym, tf in targets
                }
                try:
                    for future in as_completed(futures):
                        sym, tf = futures[future]
                        try:
                            r = future.result()
                        except Exception as exc:
                            failed.append((sym, tf, f"unexpected: {exc}"))
                            progress.log(f"[red]{sym} {tf}: unexpected error: {exc}[/red]")
                            progress.advance(task)
                            continue

                        _record_result(sym, tf, r)
                        if r.error and r.best_sharpe == float("-inf"):
                            color = "yellow" if r.error == "no_data" else "red"
                            progress.log(f"[{color}]{sym} {tf}: {r.error}[/{color}]")
                        else:
                            progress.log(
                                f"[green]{sym} {tf}: best entry={r.best_entry:.2f} "
                                f"exit={r.best_exit:.2f} Sharpe={r.best_sharpe:.3f}[/green]"
                            )
                        progress.advance(task)
                except KeyboardInterrupt:
                    console.print("\n[yellow]Interrupted. Cancelling...[/yellow]")
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise

    # Summary table — sort by best Sharpe descending.
    rows.sort(key=lambda r: r["best_sharpe"], reverse=True)

    table = Table(title=f"Threshold tuning summary ({len(rows)} models)")
    table.add_column("Symbol")
    table.add_column("TF")
    table.add_column("Entry", justify="right")
    table.add_column("Exit", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Return %", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("ΔSharpe", justify="right")
    for row in rows:
        delta_str = f"{row['delta_sharpe']:+.3f}" if row["delta_sharpe"] is not None else "—"
        table.add_row(
            row["symbol"],
            row["timeframe"],
            f"{row['best_entry']:.2f}",
            f"{row['best_exit']:.2f}",
            f"{row['best_sharpe']:.3f}",
            f"{row['best_return_pct']:+.2f}",
            str(row["best_trades"]),
            delta_str,
        )
    console.print(table)

    if failed:
        console.print(f"\n[yellow]{len(failed)} models skipped:[/yellow]")
        for sym, tf, reason in failed:
            console.print(f"  {sym} {tf}: {reason}")

    summary_json = out_dir / f"{label}_summary.json"
    summary_md = out_dir / f"{label}_summary.md"
    summary_json.write_text(
        json.dumps(
            {
                "label": label,
                "timeframe_filter": timeframe,
                "entry_grid": list(entry_values),
                "exit_grid": list(exit_values),
                "baseline_entry": baseline_entry,
                "baseline_exit": baseline_exit,
                "rows": rows,
                "failed": [{"symbol": s, "timeframe": t, "reason": r} for s, t, r in failed],
            },
            indent=2,
            default=str,
        )
    )

    md_lines = [
        f"# Threshold tuning — all models ({label})",
        "",
        f"- Models tuned: {len(rows)} (skipped: {len(failed)})",
        f"- Entry grid: {list(entry_values)}",
        f"- Exit grid:  {list(exit_values)}",
        f"- Baseline:   entry={baseline_entry} exit={baseline_exit}",
        "",
        "## Summary (sorted by best Sharpe)",
        "",
        "| Symbol | TF | Entry | Exit | Sharpe | Return % | Trades | Win % | "
        "Baseline Sharpe | ΔSharpe |",
        "|--------|----|------:|-----:|-------:|---------:|-------:|------:|"
        "----------------:|--------:|",
    ]
    for row in rows:
        delta_str = f"{row['delta_sharpe']:+.3f}" if row["delta_sharpe"] is not None else "—"
        baseline_str = (
            f"{row['baseline_sharpe']:.3f}" if row["baseline_sharpe"] != float("-inf") else "—"
        )
        md_lines.append(
            f"| {row['symbol']} | {row['timeframe']} | "
            f"{row['best_entry']:.2f} | {row['best_exit']:.2f} | "
            f"{row['best_sharpe']:.3f} | {row['best_return_pct']:+.2f} | "
            f"{row['best_trades']} | {row['best_win_rate'] * 100:.1f} | "
            f"{baseline_str} | {delta_str} |"
        )
    if failed:
        md_lines.extend(["", "## Skipped", ""])
        for sym, tf, reason in failed:
            md_lines.append(f"- {sym} {tf}: {reason}")

    summary_md.write_text("\n".join(md_lines) + "\n")
    console.print(f"\n[green]Summary JSON: {summary_json}[/green]")
    console.print(f"[green]Summary Markdown: {summary_md}[/green]")
