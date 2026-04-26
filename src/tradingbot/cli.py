from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

import structlog
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from tradingbot.backtest.holdout import resolve_holdout_window as _resolve_holdout_window
from tradingbot.config import ExchangeConfig, load_config
from tradingbot.data.fetcher import DataFetcher
from tradingbot.data.storage import EXTERNAL_SUBDIR, list_available_data, save_candles
from tradingbot.utils.logging import setup_logging
from tradingbot.utils.time import parse_date

app = typer.Typer(name="tradingbot", help="Algorithmic trading bot for Upbit")
console = Console()


@contextmanager
def _progress_context():
    """Create a Rich Progress bar and suppress structlog during display."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    # Suppress ALL structlog output to avoid breaking progress bar
    prev_config = structlog.get_config()
    suppressed_config = {
        **prev_config,
        "wrapper_class": structlog.make_filtering_bound_logger(logging.CRITICAL),
    }
    structlog.configure(**suppressed_config)
    try:
        with progress:
            yield progress
    finally:
        structlog.configure(**prev_config)


@app.command()
def download(
    symbol: str = typer.Option("BTC/KRW", "--symbol", "-s", help="Trading pair (e.g., BTC/KRW)"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Candle timeframe (e.g., 1m, 5m, 1h, 4h, 1d)"),
    since: str = typer.Option(..., "--since", help="Start date (YYYY-MM-DD)"),
    until: str = typer.Option(None, "--until", help="End date (YYYY-MM-DD). Defaults to now."),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
    exchange: str = typer.Option("upbit", "--exchange", "-e", help="Exchange name"),
) -> None:
    """Download historical OHLCV candle data."""
    setup_logging()

    since_dt = parse_date(since)
    until_dt = parse_date(until) if until else None

    config = ExchangeConfig(name=exchange)
    fetcher = DataFetcher(config)

    console.print(f"[bold]Downloading {symbol} {timeframe} candles from {exchange}...[/bold]")
    console.print(f"  Since: {since_dt.date()}")
    if until_dt:
        console.print(f"  Until: {until_dt.date()}")

    df = fetcher.fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        since=since_dt,
        until=until_dt,
    )

    if df.empty:
        console.print("[red]No data returned from exchange.[/red]")
        raise typer.Exit(1)

    path = save_candles(df, symbol, timeframe, Path(data_dir))
    console.print(f"[green]Saved {len(df)} candles to {path}[/green]")
    console.print(f"  Range: {df.index.min()} ~ {df.index.max()}")


@app.command()
def data_list(
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
) -> None:
    """List all downloaded data."""
    setup_logging()

    items = list_available_data(Path(data_dir))
    if not items:
        console.print("[yellow]No data found.[/yellow]")
        raise typer.Exit(0)

    table = Table(title="Available Data")
    table.add_column("Symbol", style="cyan")
    table.add_column("Timeframe", style="green")
    table.add_column("Rows", justify="right")
    table.add_column("Start")
    table.add_column("End")

    for item in items:
        table.add_row(item["symbol"], item["timeframe"], item["rows"], item["start"], item["end"])

    console.print(table)


@app.command()
def symbols(
    exchange: str = typer.Option("upbit", "--exchange", "-e", help="Exchange name"),
) -> None:
    """List available trading symbols on the exchange."""
    setup_logging()

    config = ExchangeConfig(name=exchange)
    fetcher = DataFetcher(config)

    available = fetcher.get_available_symbols()
    krw_symbols = [s for s in available if s.endswith("/KRW")]

    console.print(f"[bold]KRW markets on {exchange}: {len(krw_symbols)} pairs[/bold]")
    for s in sorted(krw_symbols):
        console.print(f"  {s}")


STRATEGY_MAP: dict[str, type] = {}


def _load_strategies() -> None:
    """Lazily load built-in strategies from the shared registry."""
    if STRATEGY_MAP:
        return
    from tradingbot.strategy.registry import get_strategy_map
    STRATEGY_MAP.update(get_strategy_map())


def _build_combined_strategy(
    entry: str, exit_: str, symbol: str, timeframe: str,
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
            tmpl["entry"], tmpl["exit"], symbol, timeframe,
        )
        if symbols:
            strategy.symbols = symbols
        return strategy, tmpl["label"], None

    _load_strategies()
    if strategy_name not in STRATEGY_MAP:
        console.print(f"[red]Unknown strategy: {strategy_name}[/red]")
        available = list(STRATEGY_MAP.keys()) + [
            t["label"] for t in COMBINE_TEMPLATES
        ]
        console.print(f"Available: {', '.join(available)}")
        raise typer.Exit(1)

    strategy_cls = STRATEGY_MAP[strategy_name]
    strategy = strategy_cls()
    strategy.symbols = symbols or [symbol]
    strategy.timeframe = timeframe
    return strategy, strategy_name, strategy_cls


@app.command()
def backtest(
    strategy_name: str = typer.Option("sma_cross", "--strategy", "-S", help="Strategy name"),
    symbol: str = typer.Option(None, "--symbol", "-s", help="Trading pair (omit for all config symbols)"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Candle timeframe"),
    start: str = typer.Option(None, "--start", help="Override evaluation start (YYYY-MM-DD)"),
    end: str = typer.Option(None, "--end", help="Override evaluation end (YYYY-MM-DD)"),
    balance: float = typer.Option(1_000_000, "--balance", "-b", help="Initial balance (KRW)"),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
    include_train: bool = typer.Option(
        False, "--include-train",
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
        strategy_name, symbols[0], timeframe, symbols=symbols,
    )

    # Load data for all symbols
    data: dict = {}
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
        data, start, end, include_train,
    )
    config = load_config(Path("config"), overrides={
        "trading": {"symbols": symbols, "timeframe": timeframe, "initial_balance": balance},
        "backtest": {"start_date": effective_start, "end_date": effective_end},
    })

    console.print(f"[bold]Running backtest: {strategy_name} on {len(data)} symbol(s) {timeframe}[/bold]")
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
        strategy_name, symbol, timeframe,
    )
    if strategy_cls is None:
        console.print("[red]Combined templates cannot be optimized (no param_space). Use backtest instead.[/red]")
        raise typer.Exit(1)

    from tradingbot.backtest.optimizer import GridSearchOptimizer
    from tradingbot.data.storage import load_candles

    config = load_config(Path("config"), overrides={
        "trading": {"symbols": [symbol], "timeframe": timeframe, "initial_balance": balance},
    })
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
            {symbol: df}, param_space=space, sort_by=sort_by, progress=progress,
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
        strategy_name, symbol, timeframe,
    )

    from tradingbot.data.storage import load_candles

    config = load_config(Path("config"), overrides={
        "trading": {"symbols": [symbol], "timeframe": timeframe, "initial_balance": balance},
    })

    df = load_candles(symbol, timeframe, Path(data_dir))
    console.print(
        f"[bold]Walk-forward: {strategy_name} on {symbol} "
        f"(train={train_months}m, test={test_months}m)[/bold]"
    )

    if strategy_cls is not None:
        # Registered strategy: optimize params per window
        from tradingbot.backtest.walk_forward import WalkForwardValidator

        validator = WalkForwardValidator(
            strategy_cls=strategy_cls, config=config,
            train_months=train_months, test_months=test_months,
        )
        with _progress_context() as progress:
            report = validator.validate({symbol: df}, progress=progress)
        report.print_summary()
    else:
        # Combined template: fixed filters, no optimization — test each window
        _walk_forward_combined(
            strategy, strategy_name, symbol, df, config,
            train_months, test_months,
        )


def _walk_forward_combined(
    strategy,
    strategy_name: str,
    symbol: str,
    df,
    config,
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
    WARMUP_BARS = 300

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
            progress.update(task, description=f"WF {i+1}/{len(windows)}: {train_start.date()}~{test_end.date()}")

            # Train window — include warmup buffer for indicator computation
            train_start_idx = df.index.searchsorted(train_start)
            train_warmup_idx = max(0, train_start_idx - WARMUP_BARS)
            train_with_warmup = df.iloc[train_warmup_idx:].copy()
            train_with_warmup = train_with_warmup[train_with_warmup.index < train_end]

            train_strategy = copy.deepcopy(strategy)
            engine = BacktestEngine(strategy=train_strategy, config=wf_config)
            full_train_report = engine.run({symbol: train_with_warmup})

            # Filter to train period only
            train_start_dt = train_start.to_pydatetime() if hasattr(train_start, "to_pydatetime") else train_start
            train_trades = [
                t for t in full_train_report.trades
                if t.entry_order.created_at is not None
                and t.entry_order.created_at >= train_start_dt
            ]
            train_equity = full_train_report.equity_curve[full_train_report.equity_curve.index >= train_start]

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
            warmup_idx = max(0, test_start_idx - WARMUP_BARS)
            test_with_warmup = df.iloc[warmup_idx:].copy()
            test_with_warmup = test_with_warmup[test_with_warmup.index < test_end]

            test_strategy = copy.deepcopy(strategy)
            engine = BacktestEngine(strategy=test_strategy, config=wf_config)
            full_report = engine.run({symbol: test_with_warmup})

            # Filter to test period only (exclude warmup trades)
            test_start_dt = test_start.to_pydatetime() if hasattr(test_start, "to_pydatetime") else test_start
            test_trades = [
                t for t in full_report.trades
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

            results.append(WalkForwardWindow(
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
            ))

            progress.advance(task)

    report = WalkForwardReport(windows=results, strategy_name=strategy_name)
    report.print_summary()


@app.command()
def paper(
    strategy_name: str = typer.Option("sma_cross", "--strategy", "-S", help="Strategy name"),
    symbol: str = typer.Option("BTC/KRW", "--symbol", "-s", help="Trading pair"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Candle timeframe"),
    balance: float = typer.Option(1_000_000, "--balance", "-b", help="Initial paper balance (KRW)"),
    exchange: str = typer.Option("upbit", "--exchange", "-e", help="Exchange for data feed"),
    state_file: str = typer.Option("state.json", "--state-file", help="State persistence file"),
    use_websocket: bool = typer.Option(False, "--websocket/--no-websocket", help="Use WebSocket for real-time prices"),
    entry: str | None = typer.Option(None, "--entry", help="Combined entry filters (e.g., 'trend_up:4 + rsi_oversold:30 + lgbm_prob:0.35')"),
    exit_: str | None = typer.Option(None, "--exit", help="Combined exit filters (e.g., 'rsi_overbought:70')"),
) -> None:
    """Start paper trading with simulated execution."""
    import asyncio

    setup_logging()

    if entry is not None:
        # --entry/--exit: custom combined strategy
        if exit_ is None:
            console.print("[red]--exit is required when using --entry[/red]")
            raise typer.Exit(1)
        strategy = _build_combined_strategy(entry, exit_, symbol, timeframe)
        strategy_name = strategy.describe()
    else:
        strategy, strategy_name, _ = _resolve_strategy(strategy_name, symbol, timeframe)

    from tradingbot.config import EnvSettings, ExchangeConfig
    from tradingbot.exchange.ccxt_exchange import CcxtExchange
    from tradingbot.exchange.paper import PaperExchange
    from tradingbot.live.engine import LiveEngine
    from tradingbot.live.state import StateManager
    from tradingbot.notifications.telegram import TelegramNotifier

    config = load_config(Path("config"), overrides={
        "trading": {"symbols": [symbol], "timeframe": timeframe, "initial_balance": balance},
    })

    env = EnvSettings()
    data_feed = CcxtExchange(ExchangeConfig(name=exchange), env)
    paper_exchange = PaperExchange(
        data_feed=data_feed,
        initial_balance=balance,
        fee_rate=config.backtest.fee_rate,
        slippage_pct=config.backtest.slippage_pct,
    )

    state = StateManager(Path(state_file))
    notifier = TelegramNotifier(env)

    # WebSocket client for real-time prices (optional)
    ws = None
    if use_websocket:
        from tradingbot.exchange.ws_client import UpbitWebSocketClient
        ws = UpbitWebSocketClient([symbol])

    console.print(f"[bold]Paper trading: {strategy_name} on {symbol} {timeframe}[/bold]")
    console.print(f"  Balance: {balance:,.0f} KRW")
    console.print(f"  WebSocket: {'enabled' if ws else 'disabled'}")
    console.print(f"  Telegram: {'enabled' if notifier.enabled else 'disabled'}")
    console.print("[yellow]Press Ctrl+C to stop[/yellow]")

    engine = LiveEngine(
        strategy=strategy,
        exchange=paper_exchange,
        config=config,
        state_manager=state,
        notifier=notifier if notifier.enabled else None,
        ws_client=ws,
    )
    asyncio.run(engine.run())


@app.command()
def status(
    state_file: str = typer.Option("state.json", "--state-file", help="State file path"),
) -> None:
    """Show current trading status (positions, equity)."""
    setup_logging()

    from tradingbot.live.state import StateManager

    state = StateManager(Path(state_file))
    state.load()

    if not state.positions and not state.equity_history:
        console.print("[yellow]No active trading state found.[/yellow]")
        raise typer.Exit(0)

    # Positions table
    if state.positions:
        pos_table = Table(title="Open Positions")
        pos_table.add_column("Symbol", style="cyan")
        pos_table.add_column("Side")
        pos_table.add_column("Size", justify="right")
        pos_table.add_column("Entry Price", justify="right")
        pos_table.add_column("Stop Loss", justify="right")
        pos_table.add_column("Entry Time")

        for symbol, pos in state.positions.items():
            pos_table.add_row(
                symbol,
                pos.side.value,
                f"{pos.size:.8f}",
                f"{pos.entry_price:,.0f}",
                f"{pos.stop_loss:,.0f}" if pos.stop_loss else "N/A",
                str(pos.entry_time),
            )
        console.print(pos_table)
    else:
        console.print("[green]No open positions.[/green]")

    # Recent equity
    if state.equity_history:
        recent = state.equity_history[-5:]
        eq_table = Table(title="Recent Equity")
        eq_table.add_column("Timestamp")
        eq_table.add_column("Equity", justify="right", style="green")

        for entry in recent:
            eq_table.add_row(entry["timestamp"], f"{entry['equity']:,.0f} KRW")
        console.print(eq_table)


@app.command()
def live(
    strategy_name: str = typer.Option("sma_cross", "--strategy", "-S", help="Strategy name"),
    symbol: str = typer.Option("BTC/KRW", "--symbol", "-s", help="Trading pair"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Candle timeframe"),
    exchange_name: str = typer.Option("upbit", "--exchange", "-e", help="Exchange"),
    state_file: str = typer.Option("state.json", "--state-file", help="State persistence file"),
    max_order_krw: float = typer.Option(500_000, "--max-order", help="Max order value (KRW)"),
    daily_loss_krw: float = typer.Option(200_000, "--daily-loss-limit", help="Daily loss limit (KRW)"),
    use_websocket: bool = typer.Option(False, "--websocket/--no-websocket", help="Use WebSocket for real-time prices"),
    entry: str | None = typer.Option(None, "--entry", help="Combined entry filters (e.g., 'trend_up:4 + rsi_oversold:30 + lgbm_prob:0.35')"),
    exit_: str | None = typer.Option(None, "--exit", help="Combined exit filters (e.g., 'rsi_overbought:70')"),
) -> None:
    """Start LIVE trading with real money. Use with caution."""
    import asyncio

    setup_logging()

    if entry is not None:
        # --entry/--exit: custom combined strategy
        if exit_ is None:
            console.print("[red]--exit is required when using --entry[/red]")
            raise typer.Exit(1)
        strategy = _build_combined_strategy(entry, exit_, symbol, timeframe)
        strategy_name = strategy.describe()
    else:
        strategy, strategy_name, _ = _resolve_strategy(strategy_name, symbol, timeframe)

    from tradingbot.config import EnvSettings, ExchangeConfig
    from tradingbot.exchange.ccxt_exchange import CcxtExchange
    from tradingbot.live.engine import LiveEngine
    from tradingbot.live.order_manager import OrderManager
    from tradingbot.live.state import StateManager
    from tradingbot.notifications.telegram import TelegramNotifier
    from tradingbot.risk.validators import TradeValidator

    env = EnvSettings()
    if not env.upbit_access_key or not env.upbit_secret_key:
        console.print("[red]Upbit API keys not configured. Set UPBIT_ACCESS_KEY and UPBIT_SECRET_KEY in .env[/red]")
        raise typer.Exit(1)

    config = load_config(Path("config"), overrides={
        "trading": {"symbols": [symbol], "timeframe": timeframe},
    })

    real_exchange = CcxtExchange(ExchangeConfig(name=exchange_name), env)
    order_mgr = OrderManager(exchange=real_exchange)
    validator = TradeValidator(
        max_order_value_krw=max_order_krw,
        daily_loss_limit_krw=daily_loss_krw,
    )
    state = StateManager(Path(state_file))
    notifier = TelegramNotifier(env)

    console.print("[bold red]⚠ LIVE TRADING MODE — REAL MONEY ⚠[/bold red]")
    console.print(f"  Strategy: {strategy_name}")
    console.print(f"  Symbol: {symbol} ({timeframe})")
    console.print(f"  Exchange: {exchange_name}")
    console.print(f"  Max order: {max_order_krw:,.0f} KRW")
    console.print(f"  Daily loss limit: {daily_loss_krw:,.0f} KRW")
    ws = None
    if use_websocket:
        from tradingbot.exchange.ws_client import UpbitWebSocketClient
        ws = UpbitWebSocketClient([symbol])

    console.print(f"  WebSocket: {'enabled' if ws else 'disabled'}")
    console.print(f"  Telegram: {'enabled' if notifier.enabled else 'disabled'}")
    console.print("[yellow]Press Ctrl+C to stop[/yellow]")

    engine = LiveEngine(
        strategy=strategy,
        exchange=real_exchange,
        config=config,
        state_manager=state,
        notifier=notifier if notifier.enabled else None,
        order_manager=order_mgr,
        trade_validator=validator,
        ws_client=ws,
    )
    asyncio.run(engine.run())


@app.command()
def balance(
    exchange_name: str = typer.Option("upbit", "--exchange", "-e", help="Exchange"),
) -> None:
    """Check exchange account balance."""
    import asyncio

    setup_logging()

    from tradingbot.config import EnvSettings, ExchangeConfig
    from tradingbot.exchange.ccxt_exchange import CcxtExchange

    env = EnvSettings()
    if not env.upbit_access_key or not env.upbit_secret_key:
        console.print("[red]API keys not configured.[/red]")
        raise typer.Exit(1)

    async def _fetch():
        ex = CcxtExchange(ExchangeConfig(name=exchange_name), env)
        try:
            bal = await ex.get_balance()
            table = Table(title=f"Balance ({exchange_name})")
            table.add_column("Currency", style="cyan")
            table.add_column("Amount", justify="right", style="green")
            for currency, amount in sorted(bal.items()):
                table.add_row(currency, f"{amount:,.8f}" if amount < 1 else f"{amount:,.0f}")
            console.print(table)
        finally:
            await ex.close()

    asyncio.run(_fetch())


@app.command()
def dashboard(
    state_file: str = typer.Option("state.json", "--state-file", help="State file for live monitor"),
) -> None:
    """Launch the web dashboard (Streamlit)."""
    import subprocess
    import sys

    try:
        import streamlit  # noqa: F401
    except ImportError:
        console.print("[red]Dashboard requires extra dependencies. Install with:[/red]")
        console.print('  pip install -e ".[dashboard]"')
        raise typer.Exit(1)

    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
    if not dashboard_path.exists():
        console.print("[red]Dashboard app not found.[/red]")
        raise typer.Exit(1)

    console.print("[bold]Launching dashboard...[/bold]")
    console.print(f"  State file: {state_file}")
    console.print("[yellow]Open http://localhost:8501 in your browser[/yellow]")

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path),
         "--", f"--state-file={state_file}"],
    )


def _validate_date_range(start: str | None, end: str | None) -> None:
    """Reject malformed --start/--end before workers spawn.

    Workers parse these inside spawned processes; an invalid string would
    crash the worker with an opaque ValueError. Validating up-front gives
    the user a clean Typer-style error message.
    """
    for label, value in (("--start", start), ("--end", end)):
        if value is None:
            continue
        try:
            parse_date(value)
        except ValueError as exc:
            console.print(f"[red]Invalid {label} ({value!r}): {exc}[/red]")
            raise typer.Exit(1) from exc


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
        False, "--include-train",
        help="Disable per-batch holdout filtering and scan over the full data range.",
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

    valid_metrics = {"sharpe_ratio", "total_return", "max_drawdown", "win_rate", "profit_factor", "total_trades"}
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
    results: list[dict] = []
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

        with ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {
                pool.submit(
                    _run_batch, sym, tf, batch_jobs, abs_data_dir, balance,
                    abs_config_dir, False, start, end, include_train,
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
                for r in batch_results:
                    if r.error:
                        failures.append(f"{r.strategy}/{r.symbol}/{r.timeframe}: {r.error}")
                    else:
                        results.append({
                            "strategy": r.strategy,
                            "symbol": r.symbol,
                            "timeframe": r.timeframe,
                            "sharpe_ratio": r.sharpe_ratio,
                            "total_return": r.total_return,
                            "max_drawdown": r.max_drawdown,
                            "win_rate": r.win_rate,
                            "profit_factor": r.profit_factor,
                            "total_trades": r.total_trades,
                        })
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
        sharpe_style = "green" if r["sharpe_ratio"] > 1.0 else ("yellow" if r["sharpe_ratio"] > 0 else "red")
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


@app.command()
def combine(
    entry: str = typer.Option(..., "--entry", help="Entry filters (e.g., 'trend_up:4 + rsi_oversold:30')"),
    exit_: str = typer.Option(..., "--exit", help="Exit filters (e.g., 'rsi_overbought:70')"),
    symbol: str = typer.Option("BTC/KRW", "--symbol", "-s", help="Trading pair"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Candle timeframe"),
    balance: float = typer.Option(1_000_000, "--balance", "-b", help="Initial balance (KRW)"),
    start: str | None = typer.Option(None, "--start", help="Override evaluation start (YYYY-MM-DD)"),
    end: str | None = typer.Option(None, "--end", help="Override evaluation end (YYYY-MM-DD)"),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
    include_train: bool = typer.Option(
        False, "--include-train",
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
        df, start, end, include_train,
    )
    if effective_start:
        df = df[df.index >= effective_start]
    if effective_end:
        df = df[df.index <= effective_end]

    console.print(f"  Data: {len(df)} candles")
    eval_start_str = effective_start or "data start"
    eval_end_str = effective_end or "data end"
    console.print(f"  Evaluation period: {eval_start_str} → {eval_end_str} ({period_note})")

    config = load_config(Path("config"), overrides={
        "trading": {"symbols": [symbol], "timeframe": timeframe, "initial_balance": balance},
    })

    engine = BacktestEngine(strategy=strategy, config=config)
    report = engine.run({symbol: df})
    report.print_summary()


# Predefined meaningful filter combination templates
COMBINE_TEMPLATES = [
    # Trend + timing
    {"entry": "trend_up:4 + rsi_oversold:30", "exit": "rsi_overbought:70", "label": "Trend+RSI"},
    {"entry": "trend_up:4 + rsi_oversold:35", "exit": "rsi_overbought:65", "label": "Trend+RSI(tight)"},
    {"entry": "trend_up:4 + rsi_oversold:30", "exit": "trend_up:4", "label": "Trend+RSI→TrendExit"},
    # Trend + volume
    {"entry": "trend_up:4 + volume_spike:2.5", "exit": "rsi_overbought:70", "label": "Trend+Vol"},
    {"entry": "trend_up:4 + volume_spike:2.0", "exit": "ema_above:20", "label": "Trend+Vol→EMA"},
    # Triple filter
    {"entry": "trend_up:4 + rsi_oversold:30 + volume_spike:2.0", "exit": "rsi_overbought:70", "label": "Triple"},
    {"entry": "trend_up:4 + rsi_oversold:35 + volume_spike:2.5", "exit": "trend_up:4", "label": "Triple(strict)"},
    # Momentum combos
    {"entry": "ema_above:50 + macd_cross_up", "exit": "rsi_overbought:70", "label": "EMA+MACD"},
    {"entry": "ema_above:20 + rsi_oversold:30", "exit": "rsi_overbought:70", "label": "EMA+RSI"},
    {"entry": "ema_above:50 + macd_cross_up + volume_spike:2.0", "exit": "rsi_overbought:70", "label": "EMA+MACD+Vol"},
    # Breakout combos
    {"entry": "volume_spike:2.5 + price_breakout:10", "exit": "ema_above:20", "label": "Vol+Breakout"},
    {"entry": "bb_upper_break:20 + volume_spike:2.0", "exit": "ema_above:20", "label": "BB+Vol"},
    {"entry": "price_breakout:10 + trend_up:4", "exit": "trend_up:4", "label": "Breakout+Trend"},
    # Simple combos
    {"entry": "rsi_oversold:30 + volume_spike:2.0", "exit": "rsi_overbought:70", "label": "RSI+Vol"},
    {"entry": "macd_cross_up + volume_spike:2.5", "exit": "macd_cross_up", "label": "MACD+Vol"},
    # ── Trend Following (new filters) ──
    {"entry": "ema_cross_up:12:26 + adx_strong:25 + volume_spike:2.0", "exit": "atr_trailing_exit:14:2.5", "label": "EMACross+ADX+Vol→ATR"},
    {"entry": "ema_cross_up:12:26 + adx_strong:25", "exit": "rsi_overbought:70", "label": "EMACross+ADX"},
    {"entry": "stoch_oversold:20 + aroon_up:70", "exit": "stoch_overbought:80", "label": "Stoch+Aroon"},
    {"entry": "roc_positive:12 + ichimoku_above + obv_rising", "exit": "rsi_overbought:70", "label": "ROC+Ichi+OBV"},
    {"entry": "donchian_break:20 + adx_strong:25 + volume_spike:2.0", "exit": "donchian_break:20", "label": "Donchian+ADX+Vol"},
    {"entry": "macd_cross_up + aroon_up:70 + mfi_confirm:50", "exit": "mfi_overbought:80", "label": "MACD+Aroon+MFI"},
    # ── Mean Reversion (new filters) ──
    {"entry": "rsi_oversold:30 + adx_strong:20 + obv_rising", "exit": "zscore_extreme:2.0", "label": "RSI+ADX+OBV→Zscore"},
    {"entry": "stoch_oversold:20 + ema_above:50 + mfi_confirm:40", "exit": "stoch_overbought:80", "label": "Stoch+EMA+MFI"},
    {"entry": "cci_oversold:100 + trend_up:4 + volume_spike:2.0", "exit": "cci_overbought:100", "label": "CCI+Trend+Vol"},
    {"entry": "mfi_oversold:20 + trend_up:4", "exit": "mfi_overbought:80", "label": "MFI+Trend"},
    {"entry": "rsi_oversold:30 + ichimoku_above", "exit": "pct_from_ma_exit:20:5.0", "label": "RSI+Ichi→PctMA"},
    # ── Volatility Breakout (new filters) ──
    {"entry": "bb_upper_break:20 + bb_squeeze + volume_spike:2.0", "exit": "ema_above:20", "label": "BB+Squeeze+Vol"},
    {"entry": "atr_breakout:14:2.0 + adx_strong:25 + obv_rising", "exit": "atr_trailing_exit:14:2.5", "label": "ATR+ADX+OBV→ATR"},
    {"entry": "keltner_break + trend_up:4 + volume_spike:2.0", "exit": "keltner_break", "label": "KC+Trend+Vol"},
    {"entry": "price_breakout:20 + bb_bandwidth_low:0.05 + volume_spike:2.5", "exit": "pct_from_ma_exit:20:5.0", "label": "Breakout+BBW+Vol"},
    # ── Multi-Confirm (new filters) ──
    {"entry": "rsi_oversold:30 + stoch_oversold:20 + adx_strong:25", "exit": "rsi_overbought:70", "label": "RSI+Stoch+ADX"},
    {"entry": "macd_cross_up + obv_rising + adx_strong:25", "exit": "atr_trailing_exit:14:2.0", "label": "MACD+OBV+ADX→ATR"},
    {"entry": "ema_cross_up:12:26 + mfi_confirm:50 + bb_bandwidth_low:0.04", "exit": "zscore_extreme:2.0", "label": "EMA+MFI+BBW→Zscore"},
    # ── ML + Rule combos (threshold 0.35 = veto filter mode) ──
    {"entry": "trend_up:4 + rsi_oversold:30 + lgbm_prob:0.35", "exit": "rsi_overbought:70", "label": "Trend+RSI+ML"},
    {"entry": "ema_cross_up:12:26 + lgbm_prob:0.35", "exit": "atr_trailing_exit:14:2.5", "label": "EMACross+ML→ATR"},
    {"entry": "volume_spike:2.0 + adx_strong:25 + lgbm_prob:0.35", "exit": "rsi_overbought:70", "label": "Vol+ADX+ML"},
    # ── ML Veto: Trend Following ──
    {"entry": "ema_cross_up:12:26 + trend_up:4 + lgbm_prob:0.35", "exit": "atr_trailing_exit:14:2.5", "label": "ML+TrendEMA"},
    {"entry": "adx_strong:25 + ema_above:50 + lgbm_prob:0.35", "exit": "rsi_overbought:70 + atr_trailing_exit:14:2.0", "label": "ML+ADXTrend"},
    {"entry": "ichimoku_above + aroon_up:70 + lgbm_prob:0.35", "exit": "pct_from_ma_exit:20:5.0", "label": "ML+IchimokuTrend"},
    # ── ML Veto: Mean Reversion ──
    {"entry": "rsi_oversold:30 + stoch_oversold:20 + lgbm_prob:0.35", "exit": "rsi_overbought:70 + stoch_overbought:80", "label": "ML+RSIStoch"},
    {"entry": "cci_oversold:100 + obv_rising + lgbm_prob:0.35", "exit": "cci_overbought:100 + zscore_extreme:2.0", "label": "ML+CCIMeanRev"},
    {"entry": "mfi_oversold:20 + ema_above:50 + lgbm_prob:0.35", "exit": "mfi_overbought:80 + pct_from_ma_exit:20:5.0", "label": "ML+MFIMeanRev"},
    # ── ML Veto: Breakout ──
    {"entry": "donchian_break:20 + volume_spike:2.0 + lgbm_prob:0.35", "exit": "atr_trailing_exit:14:2.5", "label": "ML+DonchianBreak"},
    {"entry": "bb_squeeze + price_breakout:10 + lgbm_prob:0.35", "exit": "atr_trailing_exit:14:2.0 + zscore_extreme:2.0", "label": "ML+BBSqueeze"},
    {"entry": "keltner_break + adx_strong:25 + lgbm_prob:0.35", "exit": "rsi_overbought:70 + atr_trailing_exit:14:2.5", "label": "ML+KeltnerBreak"},
    # ── ML Veto: Volume-Confirmed ──
    {"entry": "macd_cross_up + volume_spike:2.0 + mfi_confirm:50 + lgbm_prob:0.35", "exit": "rsi_overbought:70 + mfi_overbought:80", "label": "ML+VolMACDConfirm"},
    # ── ML Veto: Multi-Confluence ──
    {"entry": "roc_positive + obv_rising + trend_up:4 + lgbm_prob:0.35", "exit": "atr_trailing_exit:14:2.0", "label": "ML+ROCObvTrend"},
    {"entry": "stoch_oversold:25 + aroon_up:70 + lgbm_prob:0.35", "exit": "stoch_overbought:80 + pct_from_ma_exit:20:5.0", "label": "ML+StochAroonConfirm"},
]


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
        False, "--include-train",
        help="Disable per-batch holdout filtering and scan over the full data range.",
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
            batch_jobs = [(tmpl["label"], tmpl["entry"], tmpl["exit"]) for tmpl in COMBINE_TEMPLATES]
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

        with ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {
                pool.submit(
                    _run_batch, sym, tf, batch_jobs, abs_data_dir, balance,
                    abs_config_dir, False, start, end, include_train,
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
                        results.append({
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
                        })
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
                verify_batches.setdefault(key, []).append(
                    (r["template"], r["entry"], r["exit"])
                )

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
                            _run_batch, sym, tf, batch_jobs,
                            abs_data_dir, balance, abs_config_dir, True,
                            start, end, include_train,
                        ): (sym, tf)
                        for (sym, tf), batch_jobs in verify_batches.items()
                    }
                    for future in as_completed(futures):
                        sym, tf = futures[future]
                        try:
                            batch_results = future.result(timeout=1800)
                        except Exception as exc:
                            console.print(
                                f"[yellow]Verify failed {sym}/{tf}: {exc}[/yellow]"
                            )
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

            console.print(
                f"[green]Verified {len(verified_set)} results.[/green]"
            )

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
        sharpe_style = "green" if r["sharpe_ratio"] > 1.0 else ("yellow" if r["sharpe_ratio"] > 0 else "red")
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


@app.command(name="download-external")
def download_external(
    since: str = typer.Option(..., "--since", help="Start date (YYYY-MM-DD)"),
    until: str = typer.Option(None, "--until", help="End date (default: now)"),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
) -> None:
    """Download external data (Binance OHLCV, funding rate, FNG, USD/KRW)."""
    setup_logging()

    from tradingbot.data.external_fetcher import fetch_all_external

    since_dt = parse_date(since)
    until_dt = parse_date(until) if until else None

    console.print("[bold]Downloading external data...[/bold]")
    console.print(f"  Since: {since_dt.date()}")
    if until_dt:
        console.print(f"  Until: {until_dt.date()}")

    ext_dir = Path(data_dir) / EXTERNAL_SUBDIR
    results = fetch_all_external(since_dt, until_dt, ext_dir)

    if not results:
        console.print("[red]No external data fetched.[/red]")
        raise typer.Exit(1)

    for name, count in results.items():
        console.print(f"  [green]{name}: {count} rows[/green]")
    console.print(f"[green]External data saved to {ext_dir}[/green]")


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
        console.print(f"[red]No data for {symbol} {timeframe}. Run tradingbot download first.[/red]")
        raise typer.Exit(1)

    # Load external features if available
    ext_dir = Path(data_dir) / EXTERNAL_SUBDIR
    external_df = build_external_df(df, ext_dir)
    ext_count = len([c for c in (external_df.columns if external_df is not None else [])]) if external_df is not None else 0

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
        model_dir=Path(model_dir),
    )
    report = trainer.run(df, external_df=external_df)

    if not report.windows:
        console.print("[red]Training failed — insufficient data or no valid windows.[/red]")
        raise typer.Exit(1)

    # Display results
    console.print(f"\n[bold green]Training complete![/bold green]")
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
    entry_threshold: float = typer.Option(0.45, "--entry-threshold", help="Entry probability threshold"),
    exit_threshold: float = typer.Option(0.30, "--exit-threshold", help="Exit probability threshold"),
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

    from tradingbot.config import AppConfig, BacktestConfig, RiskConfig, TradingConfig
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

    config = AppConfig(
        trading=TradingConfig(
            symbols=[symbol], timeframe=timeframe, initial_balance=balance
        ),
        risk=RiskConfig(),
        backtest=BacktestConfig(),
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
    entry_threshold: float = typer.Option(0.45, "--entry-threshold", help="Entry probability threshold"),
    exit_threshold: float = typer.Option(0.30, "--exit-threshold", help="Exit probability threshold"),
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

    strategy = LGBMStrategy(StrategyParams(values={
        "entry_threshold": entry_threshold,
        "exit_threshold": exit_threshold,
        "model_dir": model_dir,
    }))
    strategy.symbols = [symbol]
    strategy.timeframe = timeframe

    config = load_config(Path("config"), overrides={
        "trading": {"symbols": [symbol], "timeframe": timeframe, "initial_balance": balance},
        "backtest": {"start_date": effective_start, "end_date": effective_end},
    })

    console.print(f"[bold]Backtesting LightGBM strategy on {symbol} {timeframe}...[/bold]")
    eval_start_str = effective_start or str(df.index[0])
    eval_end_str = effective_end or str(df.index[-1])
    console.print(f"  Evaluation period: {eval_start_str} → {eval_end_str} ({period_note})")

    engine = BacktestEngine(strategy=strategy, config=config)
    report = engine.run({symbol: df})

    console.print(f"\n[bold]Results:[/bold]")
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
        None, "--timeframe", "-t", help="Train only this timeframe",
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
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
    model_dir: str = typer.Option("models", "--model-dir", help="Model output directory"),
    workers: int = typer.Option(
        0, "--workers", "-w",
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
                        results.append({
                            "symbol": sym,
                            "timeframe": tf,
                            "avg_auc": report.avg_auc,
                            "avg_precision": report.avg_precision,
                            "holdout_auc": report.holdout_auc,
                            "holdout_precision": report.holdout_precision,
                            "n_windows": len(report.windows),
                            "model_path": str(report.model_path),
                        })
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
                        train_pair, sym, tf, data_dir_abs, model_dir_abs,
                        train_months, test_months, threads_per_worker,
                        ext_dir_abs, target_kind, atr_mult,
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
                            results.append({
                                "symbol": sym,
                                "timeframe": tf,
                                "avg_auc": r.avg_auc,
                                "avg_precision": r.avg_precision,
                                "holdout_auc": r.holdout_auc,
                                "holdout_precision": r.holdout_precision,
                                "n_windows": r.n_windows,
                                "model_path": r.model_path,
                            })

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
    label: str = typer.Option(
        "00_baseline", "--label", help="Filename prefix for the report"
    ),
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

    from tradingbot.config import AppConfig, BacktestConfig, RiskConfig, TradingConfig
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
            f"[red]No data for {symbol} {timeframe}. "
            "Run tradingbot download first.[/red]"
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
        console.print(
            f"  Target: {target_kind} (forward={forward_candles}, atr_mult={atr_mult})"
        )

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
        model_dir=Path(model_dir),
    )
    model_report = trainer.run(df, external_df=external_df)
    if not model_report.windows:
        console.print("[red]Training failed — insufficient data.[/red]")
        raise typer.Exit(1)

    # ---- Step 2: calibration + distribution metrics on the holdout eval half ----
    calibration = evaluate_calibration(
        y_true=model_report.holdout_y_true,
        raw_proba=model_report.holdout_raw_proba,
        calibrated_proba=model_report.holdout_calibrated_proba,
    ) if model_report.holdout_y_true is not None else None

    distribution = summarize_distribution(
        model_report.holdout_calibrated_proba
        if model_report.holdout_calibrated_proba is not None
        else (model_report.holdout_raw_proba
              if model_report.holdout_raw_proba is not None
              else None),
        thresholds=(exit_threshold, entry_threshold, 0.50),
    ) if (
        model_report.holdout_raw_proba is not None
        or model_report.holdout_calibrated_proba is not None
    ) else None

    feat_top10 = top_features(model_report.feature_importance, top_n=10)

    # ---- Step 3: strategy walk-forward (per-window backtest) ----
    strategy_report = None
    if not skip_backtest:
        config = AppConfig(
            trading=TradingConfig(
                symbols=[symbol], timeframe=timeframe, initial_balance=balance
            ),
            risk=RiskConfig(),
            backtest=BacktestConfig(),
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
            f"  Windows: {strategy_report.n_windows}  "
            f"(skipped: {strategy_report.n_skipped})"
        )
        console.print(f"  Avg Sharpe: {strategy_report.avg_sharpe:.2f}")
        console.print(
            f"  Cumulative return: {strategy_report.cumulative_return_pct:+.2f}%"
        )
        console.print(f"  Total trades: {strategy_report.total_trades}")
        console.print(
            f"  Avg win rate (traded windows): {strategy_report.avg_win_rate * 100:.1f}%"
        )

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
    threshold: float = typer.Option(
        0.006, "--threshold", help="Binary target return threshold"
    ),
    target_kind: str = typer.Option(
        "triple-barrier",
        "--target-kind",
        help="Target labelling strategy: binary | atr | triple-barrier",
    ),
    atr_mult: float = typer.Option(
        1.0, "--atr-mult", help="ATR multiplier for atr / triple-barrier targets"
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
    label: str = typer.Option(
        "02_tuned", "--label", help="Filename prefix for the tuner report"
    ),
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
    from tradingbot.ml.tuner import LGBMTuner
    from tradingbot.ml.walk_forward import MLWalkForwardTrainer

    try:
        df = load_candles(symbol, timeframe, Path(data_dir))
    except FileNotFoundError:
        console.print(
            f"[red]No data for {symbol} {timeframe}. "
            "Run tradingbot download first.[/red]"
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
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        balance=balance,
        external_data_dir=ext_dir_for_runner,
        objective=objective,
        seed=seed,
    )
    result = tuner.search(df, n_trials=trials, time_budget_sec=time_budget)

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
        if final_model_path is not None:
            symbol_key = symbol.replace("/", "_")
            meta_path = (
                Path(model_dir) / f"lgbm_{symbol_key}_{timeframe}_meta.json"
            )
            if meta_path.exists():
                meta_dict = json.loads(meta_path.read_text())
                meta_dict["tuning"] = {
                    "best_params": dict(result.best_params),
                    "best_value": result.best_value,
                    "objective": objective,
                    "n_trials_completed": result.n_trials_completed,
                    "elapsed_sec": result.elapsed_sec,
                }
                meta_path.write_text(json.dumps(meta_dict, indent=2, default=str))

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
        "final_holdout_auc": (
            final_report.holdout_auc if final_report.windows else None
        ),
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


if __name__ == "__main__":
    app()
