from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import cast

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


def _paginate_ohlcv(
    fetch_page: Callable[[int | None], list[list[float]]],
    since_ms: int | None,
    until_ms: int | None,
    tf_ms: int,
    limit: int,
    on_continue: Callable[[list[list[float]], int], None] | None = None,
) -> list[list[float]]:
    """Accumulate OHLCV rows by repeatedly calling ``fetch_page``.

    Shared pagination skeleton for CCXT OHLCV endpoints: advance ``since_ms``
    from the last row's timestamp + ``tf_ms``, stop on an empty page, a short
    page (``len < limit // 2``, indicating end of available data), or
    ``since_ms`` passing ``until_ms``/now. Rate limiting and retries are the
    caller's responsibility inside ``fetch_page``. ``on_continue`` fires only
    when the loop is about to fetch another page (mirrors the per-page
    progress log / pacing sleep call sites had before extraction).
    """
    all_rows: list[list[float]] = []
    while True:
        page = fetch_page(since_ms)
        if not page:
            break

        all_rows.extend(page)
        last_ts = page[-1][0]
        since_ms = int(last_ts) + tf_ms

        if until_ms and since_ms > until_ms:
            break
        now_ms = int(time.time() * 1000)
        if since_ms > now_ms:
            break
        if len(page) < limit // 2:
            break

        if on_continue is not None:
            on_continue(all_rows, int(last_ts))

    return all_rows


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
            since = since.replace(tzinfo=UTC)
        if until and until.tzinfo is None:
            until = until.replace(tzinfo=UTC)
        since_ms = int(since.timestamp() * 1000) if since else None
        until_ms = int(until.timestamp() * 1000) if until else None
        tf_ms = TIMEFRAME_MS.get(timeframe, 3_600_000)

        def _fetch_page(page_since_ms: int | None) -> list[list[float]]:
            retries_429 = 0
            while True:
                self._rate_limit()
                try:
                    return cast(
                        "list[list[float]]",
                        self.exchange.fetch_ohlcv(
                            symbol, timeframe=timeframe, since=page_since_ms, limit=limit
                        ),
                    )
                except ccxt.RateLimitExceeded as e:
                    retries_429 += 1
                    if retries_429 > 5:
                        logger.error("ccxt_error", symbol=symbol, error=str(e))
                        raise
                    wait = min(5.0 * 2 ** (retries_429 - 1), 60.0)
                    logger.warning(
                        "rate_limited_backoff", symbol=symbol, wait_sec=wait, attempt=retries_429
                    )
                    time.sleep(wait)
                    continue
                except ccxt.BaseError as e:
                    logger.error("ccxt_error", symbol=symbol, error=str(e))
                    raise

        def _log_progress(rows: list[list[float]], last_ts: int) -> None:
            # Upbit sometimes returns slightly fewer than limit (e.g., 199 instead of 200);
            # _paginate_ohlcv only stops on a significantly shorter page, indicating
            # end of available data.
            logger.debug(
                "fetching_page",
                symbol=symbol,
                fetched=len(rows),
                last_ts=datetime.fromtimestamp(last_ts / 1000, tz=UTC).isoformat(),
            )

        all_rows = _paginate_ohlcv(
            _fetch_page, since_ms, until_ms, tf_ms, limit, on_continue=_log_progress
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
            until_aware = until.replace(tzinfo=UTC) if until.tzinfo is None else until
            df = df[df.index <= until_aware]

        return df.astype(
            {"open": float, "high": float, "low": float, "close": float, "volume": float}
        )

    def get_available_symbols(self) -> list[str]:
        """Get list of available trading symbols on the exchange."""
        self.exchange.load_markets()
        return list(self.exchange.symbols)
