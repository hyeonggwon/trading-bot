FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy source and install (single layer for hatchling compatibility)
COPY pyproject.toml .
COPY src/ src/
COPY config/ config/
RUN pip install --no-cache-dir .

# Create non-root user with all needed directories
RUN useradd --create-home --shell /bin/bash botuser && \
    mkdir -p /app/data /app/logs /app/state && \
    chown -R botuser:botuser /app

# Copy health check script
COPY scripts/healthcheck.py /app/scripts/healthcheck.py

USER botuser

# Default: paper trading with SMA cross on BTC/KRW
CMD ["tradingbot", "paper", "--strategy", "sma_cross", "--symbol", "BTC/KRW", "--state-file", "/app/state/state.json"]
