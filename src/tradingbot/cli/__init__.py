"""Typer CLI package — importing it registers every command on the shared app."""

# Command modules register their commands on the shared app at import time.
# Registration order mirrors the original single-file layout.
# isort: off
from tradingbot.cli import data  # noqa: F401
from tradingbot.cli import backtest  # noqa: F401
from tradingbot.cli import combine  # noqa: F401
from tradingbot.cli import trade  # noqa: F401
from tradingbot.cli import ml  # noqa: F401
from tradingbot.cli import pipeline  # noqa: F401

# isort: on
# ``app`` is the console-script entry point (pyproject.toml: tradingbot.cli:app)
# and is imported directly by dashboard/forms.py's click introspection.
from tradingbot.cli._shared import app

__all__ = ["app"]
