"""Trading commands: paper, status, live, balance, dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.table import Table

from tradingbot.cli._shared import app, console
from tradingbot.cli.combine import _build_combined_strategy, _resolve_strategy
from tradingbot.config import load_config
from tradingbot.utils.logging import setup_logging

if TYPE_CHECKING:
    from tradingbot.strategy.base import Strategy


@app.command()
def paper(
    strategy_name: str = typer.Option("sma_cross", "--strategy", "-S", help="Strategy name"),
    symbol: str = typer.Option("BTC/KRW", "--symbol", "-s", help="Trading pair"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Candle timeframe"),
    balance: float = typer.Option(1_000_000, "--balance", "-b", help="Initial paper balance (KRW)"),
    exchange: str = typer.Option("upbit", "--exchange", "-e", help="Exchange for data feed"),
    state_file: str = typer.Option("state.json", "--state-file", help="State persistence file"),
    use_websocket: bool = typer.Option(
        False, "--websocket/--no-websocket", help="Use WebSocket for real-time prices"
    ),
    entry: str | None = typer.Option(
        None,
        "--entry",
        help="Combined entry filters (e.g., 'trend_up:4 + rsi_oversold:30 + lgbm_prob:0.35')",
    ),
    exit_: str | None = typer.Option(
        None, "--exit", help="Combined exit filters (e.g., 'rsi_overbought:70')"
    ),
) -> None:
    """Start paper trading with simulated execution."""
    import asyncio

    setup_logging()

    strategy: Strategy
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

    config = load_config(
        Path("config"),
        overrides={
            "trading": {"symbols": [symbol], "timeframe": timeframe, "initial_balance": balance},
        },
    )

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
    daily_loss_krw: float = typer.Option(
        200_000, "--daily-loss-limit", help="Daily loss limit (KRW)"
    ),
    use_websocket: bool = typer.Option(
        False, "--websocket/--no-websocket", help="Use WebSocket for real-time prices"
    ),
    entry: str | None = typer.Option(
        None,
        "--entry",
        help="Combined entry filters (e.g., 'trend_up:4 + rsi_oversold:30 + lgbm_prob:0.35')",
    ),
    exit_: str | None = typer.Option(
        None, "--exit", help="Combined exit filters (e.g., 'rsi_overbought:70')"
    ),
) -> None:
    """Start LIVE trading with real money. Use with caution."""
    import asyncio

    setup_logging()

    strategy: Strategy
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
        console.print(
            "[red]Upbit API keys not configured. "
            "Set UPBIT_ACCESS_KEY and UPBIT_SECRET_KEY in .env[/red]"
        )
        raise typer.Exit(1)

    config = load_config(
        Path("config"),
        overrides={
            "trading": {"symbols": [symbol], "timeframe": timeframe},
        },
    )

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

    async def _fetch() -> None:
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
    state_file: str = typer.Option(
        "state.json", "--state-file", help="State file for live monitor"
    ),
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
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_path),
            "--",
            f"--state-file={state_file}",
        ],
    )
