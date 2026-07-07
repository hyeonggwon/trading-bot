"""Typer CLI package — importing it registers every command on the shared app."""

# Command modules register their commands on the shared app at import time.
# Registration order mirrors the original single-file layout.
# isort: off
from tradingbot.cli import data  # noqa: F401
from tradingbot.cli import backtest  # noqa: F401
from tradingbot.cli import combine  # noqa: F401
from tradingbot.cli import trade  # noqa: F401
from tradingbot.cli import ml  # noqa: F401

# isort: on
from tradingbot.cli._shared import (
    _app_root,
    _resolve_holdout_window,
    _validate_date_range,
    app,
    console,
)
from tradingbot.cli.backtest import _walk_forward_combined
from tradingbot.cli.combine import (
    COMBINE_TEMPLATES,
    _build_combined_strategy,
    _find_combine_template,
    _resolve_strategy,
)
from tradingbot.cli.ml import ml_diagnostics, ml_walk_forward

__all__ = [
    "COMBINE_TEMPLATES",
    "_app_root",
    "_build_combined_strategy",
    "_find_combine_template",
    "_resolve_holdout_window",
    "_resolve_strategy",
    "_validate_date_range",
    "_walk_forward_combined",
    "app",
    "console",
    "ml_diagnostics",
    "ml_walk_forward",
]
