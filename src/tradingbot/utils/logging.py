from __future__ import annotations

import structlog


def setup_logging(level: str = "INFO") -> None:
    """Configure structlog for console output."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(structlog, level.upper(), structlog.INFO)  # type: ignore[arg-type]
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )
