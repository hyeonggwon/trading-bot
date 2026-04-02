from __future__ import annotations

import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import structlog


def setup_logging(level: str = "INFO", log_dir: str | None = None) -> None:
    """Configure structlog for console and optional file output.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        log_dir: If set, also writes JSON logs to files with daily rotation.
                 Defaults to LOG_DIR env var if set.
    """
    log_dir = log_dir or os.environ.get("LOG_DIR")

    # Shared processors (run before final rendering)
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if log_dir:
        # Dual output: JSON file + console for docker logs
        _setup_file_logging(log_dir, level)

        structlog.configure(
            processors=[
                *shared_processors,
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, level.upper(), logging.INFO)
            ),
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
        )
    else:
        # Console output only: human-readable
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, level.upper(), logging.INFO)
            ),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
        )


def _setup_file_logging(log_dir: str, level: str) -> None:
    """Set up stdlib logging with file + console handlers."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, level.upper(), logging.INFO)

    # JSON formatter for file output
    json_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
    )

    # Console formatter for docker logs / terminal
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(),
    )

    # File handler: daily rotation, 7 days
    file_handler = TimedRotatingFileHandler(
        filename=log_path / "tradingbot.log",
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(json_formatter)

    # Console handler: keeps docker logs working
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)

    # Clear existing handlers to prevent duplicates on re-config
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(log_level)
