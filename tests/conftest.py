from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime

import pandas as pd
import pytest

from tradingbot.core.models import Candle


@pytest.fixture
def sample_candles() -> list[Candle]:
    """Generate a list of sample candles for testing."""
    candles = []
    prices = [
        (100, 110, 95, 105, 1000),
        (105, 115, 100, 110, 1200),
        (110, 120, 105, 115, 1100),
        (115, 125, 108, 120, 1300),
        (120, 130, 112, 125, 1400),
        (125, 128, 110, 112, 900),
        (112, 118, 108, 115, 1000),
        (115, 122, 112, 120, 1100),
        (120, 126, 115, 118, 1050),
        (118, 124, 114, 122, 1150),
    ]
    for i, (o, h, lo, c, v) in enumerate(prices):
        ts = datetime(2024, 1, 1, i, tzinfo=UTC)
        candles.append(Candle(timestamp=ts, open=o, high=h, low=lo, close=c, volume=v))
    return candles


@pytest.fixture
def sample_df(sample_candles: list[Candle]) -> pd.DataFrame:
    """Convert sample candles to a DataFrame."""
    df = pd.DataFrame([asdict(c) for c in sample_candles])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp")


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Provide a temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir
