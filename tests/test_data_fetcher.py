"""DataFetcher 429 backoff behavior."""

from typing import Any

import ccxt
import pytest

from tradingbot.data.fetcher import DataFetcher


class _FlakyExchange:
    """Raises 429 for the first `fail_count` calls, then returns one page."""

    def __init__(self, fail_count: int) -> None:
        self.fail_count = fail_count
        self.calls = 0

    def fetch_ohlcv(self, symbol: str, **kwargs: Any) -> list[list[float]]:
        self.calls += 1
        if self.calls <= self.fail_count:
            raise ccxt.RateLimitExceeded("429 Too Many Requests")
        return [[1704067200000 + i * 3_600_000, 1.0, 2.0, 0.5, 1.5, 10.0] for i in range(3)]


@pytest.fixture()
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tradingbot.data.fetcher.time.sleep", lambda s: None)


def test_fetch_ohlcv_retries_on_429(no_sleep: None) -> None:
    fetcher = DataFetcher()
    flaky = _FlakyExchange(fail_count=2)
    fetcher.exchange = flaky  # type: ignore[assignment]

    df = fetcher.fetch_ohlcv("BTC/KRW", timeframe="1h")

    assert len(df) == 3
    assert flaky.calls == 3  # 2 failures + 1 success


def test_fetch_ohlcv_raises_after_persistent_429(no_sleep: None) -> None:
    fetcher = DataFetcher()
    flaky = _FlakyExchange(fail_count=100)
    fetcher.exchange = flaky  # type: ignore[assignment]

    with pytest.raises(ccxt.RateLimitExceeded):
        fetcher.fetch_ohlcv("BTC/KRW", timeframe="1h")

    assert flaky.calls == 6  # initial + 5 retries
