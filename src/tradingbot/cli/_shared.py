"""Shared Typer app, console, and cross-command CLI helpers."""

from __future__ import annotations

import logging
import os
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

from tradingbot.backtest.holdout import resolve_holdout_window as _resolve_holdout_window
from tradingbot.utils.time import parse_date

__all__ = ["_resolve_holdout_window"]


app = typer.Typer(name="tradingbot", help="Algorithmic trading bot for Upbit")
console = Console()


@app.callback()
def _app_root() -> None:
    """Anchor cwd-relative defaults before any command runs.

    Every default path in this CLI — ``config/``, ``data/``, ``models/``,
    ``state.json``, ``.env``, ``logs/`` — resolves against the current
    directory. When the console script is invoked outside the project root
    (e.g. from ``$HOME`` on a second machine), set ``TRADINGBOT_HOME`` to the
    project directory: we chdir there so every default keeps working.
    Without it a missing ``config/`` silently falls back to built-in
    defaults, so warn loudly instead of misbehaving in quiet.
    """
    home = os.environ.get("TRADINGBOT_HOME")
    if home:
        try:
            os.chdir(home)
        except OSError as e:
            console.print(f"[red]TRADINGBOT_HOME is not usable: {e}[/red]")
            raise typer.Exit(1)
    elif not Path("config").is_dir():
        console.print(
            "[yellow]Warning: no config/ in the current directory — built-in "
            "defaults will be used. Run from the project root or set "
            "TRADINGBOT_HOME.[/yellow]"
        )


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


STRATEGY_MAP: dict[str, type] = {}


def _load_strategies() -> None:
    """Lazily load built-in strategies from the shared registry."""
    if STRATEGY_MAP:
        return
    from tradingbot.strategy.registry import get_strategy_map

    STRATEGY_MAP.update(get_strategy_map())


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


def _write_scan_markdown_report(
    *,
    output: Path,
    title: str,
    metadata: list[str],
    section: str,
    columns: list[str],
    rows: list[list[str]],
) -> None:
    """Persist a scan / combine-scan result table as a markdown file.

    Mirrors the format of ``personal/scan_holdout_result.md`` /
    ``personal/combine_scan_holdout_result.md`` so reruns drop in cleanly:
    the user can diff today's run against the last one without first
    massaging Rich-rendered console output by hand.
    """
    lines: list[str] = [f"# {title}", ""]
    lines.extend(f"- {item}" for item in metadata)
    lines.extend(["", f"## {section}", ""])
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
