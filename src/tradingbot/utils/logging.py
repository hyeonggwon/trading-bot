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
        # File output: JSON format with rotation via stdlib integration
        _setup_file_logging(log_dir, level)

        structlog.configure(
            processors=[
                *shared_processors,
                # Route to stdlib for file output
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(structlog, level.upper(), structlog.INFO)  # type: ignore[arg-type]
            ),
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
        )
    else:
        # Console output: human-readable
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(structlog, level.upper(), structlog.INFO)  # type: ignore[arg-type]
            ),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
        )


def _setup_file_logging(log_dir: str, level: str) -> None:
    """Set up stdlib logging with daily rotating file handler + JSON formatter."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Use structlog's ProcessorFormatter for clean JSON output
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
    )

    handler = TimedRotatingFileHandler(
        filename=log_path / "tradingbot.log",
        when="midnight",
        interval=1,
        backupCount=7,  # Keep 7 days of logs
        encoding="utf-8",
    )
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
