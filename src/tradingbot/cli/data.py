"""Data commands: download, data-list, symbols, download-external."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from tradingbot.cli._shared import app, console
from tradingbot.config import ExchangeConfig
from tradingbot.data.fetcher import DataFetcher
from tradingbot.data.storage import EXTERNAL_SUBDIR, list_available_data, save_candles
from tradingbot.utils.logging import setup_logging
from tradingbot.utils.time import parse_date


@app.command()
def download(
    symbol: str = typer.Option("BTC/KRW", "--symbol", "-s", help="Trading pair (e.g., BTC/KRW)"),
    timeframe: str = typer.Option(
        "1h", "--timeframe", "-t", help="Candle timeframe (e.g., 1m, 5m, 1h, 4h, 1d)"
    ),
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
