FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for LightGBM (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy source and install (single layer for hatchling compatibility)
# EXTRAS: 기본 ml — 대시보드 이미지는 --build-arg EXTRAS=ml,dashboard
ARG EXTRAS=ml
COPY pyproject.toml .
COPY src/ src/
COPY config/ config/
RUN pip install --no-cache-dir ".[${EXTRAS}]"

# Create non-root user with all needed directories
RUN useradd --create-home --shell /bin/bash botuser && \
    mkdir -p /app/data /app/logs /app/state /app/models && \
    chown -R botuser:botuser /app

# Copy health check script
COPY scripts/healthcheck.py /app/scripts/healthcheck.py

USER botuser

# Liveness probe: healthcheck.py exits non-zero if state.json is missing or
# stale (> HEALTHCHECK_MAX_STALE_SECONDS). STATE_FILE must match CMD --state-file.
ENV STATE_FILE=/app/state/state.json
HEALTHCHECK --interval=5m --timeout=10s --start-period=2m --retries=3 \
    CMD python /app/scripts/healthcheck.py || exit 1

# Default: paper trading with SMA cross on BTC/KRW
CMD ["tradingbot", "paper", "--strategy", "sma_cross", "--symbol", "BTC/KRW", "--state-file", "/app/state/state.json"]
