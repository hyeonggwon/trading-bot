from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from tradingbot.config import ExchangeConfig, load_config
from tradingbot.data.fetcher import DataFetcher
from tradingbot.data.storage import list_available_data, save_candles
from tradingbot.utils.logging import setup_logging
from tradingbot.utils.time import parse_date

app = typer.Typer(name="tradingbot", help="Algorithmic trading bot for Upbit")
console = Console()


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
    """Lazily load built-in strategies."""
    if STRATEGY_MAP:
        return
    from tradingbot.strategy.examples.bollinger_breakout import BollingerBreakoutStrategy
    from tradingbot.strategy.examples.macd_momentum import MacdMomentumStrategy
    from tradingbot.strategy.examples.rsi_mean_reversion import RsiMeanReversionStrategy
    from tradingbot.strategy.examples.sma_cross import SmaCrossStrategy

    STRATEGY_MAP["sma_cross"] = SmaCrossStrategy
    STRATEGY_MAP["rsi_mean_reversion"] = RsiMeanReversionStrategy
    STRATEGY_MAP["macd_momentum"] = MacdMomentumStrategy
    STRATEGY_MAP["bollinger_breakout"] = BollingerBreakoutStrategy


@app.command()
def backtest(
    strategy_name: str = typer.Option("sma_cross", "--strategy", "-S", help="Strategy name"),
    symbol: str = typer.Option("BTC/KRW", "--symbol", "-s", help="Trading pair"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Candle timeframe"),
    start: str = typer.Option(None, "--start", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(None, "--end", help="End date (YYYY-MM-DD)"),
    balance: float = typer.Option(1_000_000, "--balance", "-b", help="Initial balance (KRW)"),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
) -> None:
    """Run a backtest on historical data."""
    setup_logging()
    _load_strategies()

    if strategy_name not in STRATEGY_MAP:
        console.print(f"[red]Unknown strategy: {strategy_name}[/red]")
        console.print(f"Available: {', '.join(STRATEGY_MAP.keys())}")
        raise typer.Exit(1)

    from tradingbot.backtest.engine import BacktestEngine
    from tradingbot.data.storage import load_candles

    config = load_config(Path("config"), overrides={
        "trading": {"symbols": [symbol], "timeframe": timeframe, "initial_balance": balance},
        "backtest": {
            "start_date": start,
            "end_date": end,
        },
    })

    strategy_cls = STRATEGY_MAP[strategy_name]
    strategy = strategy_cls()
    strategy.symbols = [symbol]
    strategy.timeframe = timeframe

    console.print(f"[bold]Running backtest: {strategy_name} on {symbol} {timeframe}[/bold]")

    df = load_candles(symbol, timeframe, Path(data_dir))
    console.print(f"  Data: {len(df)} candles ({df.index.min()} ~ {df.index.max()})")

    engine = BacktestEngine(strategy=strategy, config=config)
    report = engine.run({symbol: df})
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
    _load_strategies()

    if strategy_name not in STRATEGY_MAP:
        console.print(f"[red]Unknown strategy: {strategy_name}[/red]")
        raise typer.Exit(1)

    from tradingbot.backtest.optimizer import GridSearchOptimizer
    from tradingbot.data.storage import load_candles

    config = load_config(Path("config"), overrides={
        "trading": {"symbols": [symbol], "timeframe": timeframe, "initial_balance": balance},
    })

    strategy_cls = STRATEGY_MAP[strategy_name]
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
    results = optimizer.optimize({symbol: df}, param_space=space, sort_by=sort_by)
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
    _load_strategies()

    if strategy_name not in STRATEGY_MAP:
        console.print(f"[red]Unknown strategy: {strategy_name}[/red]")
        raise typer.Exit(1)

    from tradingbot.backtest.walk_forward import WalkForwardValidator
    from tradingbot.data.storage import load_candles

    config = load_config(Path("config"), overrides={
        "trading": {"symbols": [symbol], "timeframe": timeframe, "initial_balance": balance},
    })

    strategy_cls = STRATEGY_MAP[strategy_name]

    df = load_candles(symbol, timeframe, Path(data_dir))
    console.print(
        f"[bold]Walk-forward: {strategy_name} on {symbol} "
        f"(train={train_months}m, test={test_months}m)[/bold]"
    )

    validator = WalkForwardValidator(
        strategy_cls=strategy_cls, config=config,
        train_months=train_months, test_months=test_months,
    )
    report = validator.validate({symbol: df})
    report.print_summary()


@app.command()
def paper(
    strategy_name: str = typer.Option("sma_cross", "--strategy", "-S", help="Strategy name"),
    symbol: str = typer.Option("BTC/KRW", "--symbol", "-s", help="Trading pair"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Candle timeframe"),
    balance: float = typer.Option(1_000_000, "--balance", "-b", help="Initial paper balance (KRW)"),
    exchange: str = typer.Option("upbit", "--exchange", "-e", help="Exchange for data feed"),
    state_file: str = typer.Option("state.json", "--state-file", help="State persistence file"),
) -> None:
    """Start paper trading with simulated execution."""
    import asyncio

    setup_logging()
    _load_strategies()

    if strategy_name not in STRATEGY_MAP:
        console.print(f"[red]Unknown strategy: {strategy_name}[/red]")
        raise typer.Exit(1)

    from tradingbot.config import EnvSettings, ExchangeConfig
    from tradingbot.exchange.ccxt_exchange import CcxtExchange
    from tradingbot.exchange.paper import PaperExchange
    from tradingbot.live.engine import LiveEngine
    from tradingbot.live.state import StateManager
    from tradingbot.notifications.telegram import TelegramNotifier

    config = load_config(Path("config"), overrides={
        "trading": {"symbols": [symbol], "timeframe": timeframe, "initial_balance": balance},
    })

    strategy_cls = STRATEGY_MAP[strategy_name]
    strategy = strategy_cls()
    strategy.symbols = [symbol]
    strategy.timeframe = timeframe

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

    console.print(f"[bold]Paper trading: {strategy_name} on {symbol} {timeframe}[/bold]")
    console.print(f"  Balance: {balance:,.0f} KRW")
    console.print(f"  Telegram: {'enabled' if notifier.enabled else 'disabled'}")
    console.print("[yellow]Press Ctrl+C to stop[/yellow]")

    engine = LiveEngine(
        strategy=strategy,
        exchange=paper_exchange,
        config=config,
        state_manager=state,
        notifier=notifier if notifier.enabled else None,
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
) -> None:
    """Start LIVE trading with real money. Use with caution."""
    import asyncio

    setup_logging()
    _load_strategies()

    if strategy_name not in STRATEGY_MAP:
        console.print(f"[red]Unknown strategy: {strategy_name}[/red]")
        raise typer.Exit(1)

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

    strategy_cls = STRATEGY_MAP[strategy_name]
    strategy = strategy_cls()
    strategy.symbols = [symbol]
    strategy.timeframe = timeframe

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


if __name__ == "__main__":
    app()
