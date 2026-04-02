from __future__ import annotations

import time
from datetime import datetime, timezone

import ccxt
import pandas as pd
import structlog

from tradingbot.config import ExchangeConfig

logger = structlog.get_logger()

# CCXT timeframe string -> milliseconds
TIMEFRAME_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
    "1w": 604_800_000,
}


class DataFetcher:
    """Fetches OHLCV data from exchanges via CCXT."""

    def __init__(self, exchange_config: ExchangeConfig | None = None):
        config = exchange_config or ExchangeConfig()
        exchange_class = getattr(ccxt, config.name)
        self.exchange: ccxt.Exchange = exchange_class({"enableRateLimit": True})
        self.rate_limit_per_sec = config.rate_limit_per_sec
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between API calls."""
        min_interval = 1.0 / self.rate_limit_per_sec
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 200,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles for a symbol.

        Paginates automatically to fetch all candles between since and until.
        Returns a DataFrame with columns: open, high, low, close, volume
        and a DatetimeIndex named 'timestamp'.
        """
        # Ensure timezone-aware UTC to avoid local-time misinterpretation (Bug #10)
        if since and since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
        if until and until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
        since_ms = int(since.timestamp() * 1000) if since else None
        until_ms = int(until.timestamp() * 1000) if until else None
        tf_ms = TIMEFRAME_MS.get(timeframe, 3_600_000)

        all_rows: list[list] = []

        while True:
            self._rate_limit()
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, timeframe=timeframe, since=since_ms, limit=limit
                )
            except ccxt.BaseError as e:
                logger.error("ccxt_error", symbol=symbol, error=str(e))
                raise

            if not ohlcv:
                break

            all_rows.extend(ohlcv)
            last_ts = ohlcv[-1][0]

            # Move past the last candle for next page
            since_ms = last_ts + tf_ms

            # Stop if we've passed the until boundary or reached present
            if until_ms and since_ms > until_ms:
                break
            now_ms = int(time.time() * 1000)
            if since_ms > now_ms:
                break
            # Upbit sometimes returns slightly fewer than limit (e.g., 199 instead of 200)
            # Only stop if we got significantly fewer, indicating end of available data
            if len(ohlcv) < limit // 2:
                break

            logger.debug(
                "fetching_page",
                symbol=symbol,
                fetched=len(all_rows),
                last_ts=datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).isoformat(),
            )

        if not all_rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

        # Filter to requested range
        if until:
            until_aware = until.replace(tzinfo=timezone.utc) if until.tzinfo is None else until
            df = df[df.index <= until_aware]

        return df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})

    def get_available_symbols(self) -> list[str]:
        """Get list of available trading symbols on the exchange."""
        self.exchange.load_markets()
        return list(self.exchange.symbols)
