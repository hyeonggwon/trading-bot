from __future__ import annotations

from pathlib import Path

import pandas as pd
import structlog

logger = structlog.get_logger()

DEFAULT_DATA_DIR = Path("data")


def _symbol_to_dirname(symbol: str) -> str:
    """Convert symbol like 'BTC/KRW' to directory name 'BTC_KRW'."""
    return symbol.replace("/", "_")


def get_parquet_path(
    symbol: str, timeframe: str, data_dir: Path = DEFAULT_DATA_DIR
) -> Path:
    """Get the parquet file path for a symbol/timeframe pair."""
    dirname = _symbol_to_dirname(symbol)
    return data_dir / dirname / f"{timeframe}.parquet"


def save_candles(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> Path:
    """Save candle DataFrame to parquet, merging with existing data if present."""
    path = get_parquet_path(symbol, timeframe, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = load_candles(symbol, timeframe, data_dir)
        df = pd.concat([existing, df])
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

    df.to_parquet(path, engine="pyarrow")
    logger.info("saved_candles", symbol=symbol, timeframe=timeframe, rows=len(df), path=str(path))
    return path


def load_candles(
    symbol: str,
    timeframe: str,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """Load candle DataFrame from parquet."""
    path = get_parquet_path(symbol, timeframe, data_dir)
    if not path.exists():
        raise FileNotFoundError(f"No data file found: {path}")

    df = pd.read_parquet(path, engine="pyarrow")
    df = df.sort_index()
    logger.debug("loaded_candles", symbol=symbol, timeframe=timeframe, rows=len(df))
    return df


TIMEFRAME_FREQ: dict[str, str] = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
    "1w": "1W",
}


def detect_gaps(
    df: pd.DataFrame,
    timeframe: str = "1h",
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Detect gaps (missing candles) in OHLCV data.

    Returns list of (gap_start, gap_end) tuples where candles are missing.
    An empty list means no gaps.
    """
    if len(df) < 2:
        return []

    freq = TIMEFRAME_FREQ.get(timeframe, "1h")
    expected_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq)
    missing = expected_index.difference(df.index)

    if missing.empty:
        return []

    # Group consecutive missing timestamps into gap ranges
    gaps: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    gap_start = missing[0]
    prev = missing[0]

    for ts in missing[1:]:
        expected_delta = pd.tseries.frequencies.to_offset(freq)
        if ts - prev > expected_delta:
            gaps.append((gap_start, prev))
            gap_start = ts
        prev = ts
    gaps.append((gap_start, prev))

    if gaps:
        logger.warning("data_gaps_detected", count=len(gaps), missing_candles=len(missing))

    return gaps


def list_available_data(data_dir: Path = DEFAULT_DATA_DIR) -> list[dict[str, str]]:
    """List all available symbol/timeframe pairs in the data directory."""
    results = []
    if not data_dir.exists():
        return results

    for symbol_dir in sorted(data_dir.iterdir()):
        if not symbol_dir.is_dir() or symbol_dir.name.startswith("."):
            continue
        symbol = symbol_dir.name.replace("_", "/")
        for parquet_file in sorted(symbol_dir.glob("*.parquet")):
            timeframe = parquet_file.stem
            df = pd.read_parquet(parquet_file)
            results.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "rows": str(len(df)),
                "start": str(df.index.min()),
                "end": str(df.index.max()),
            })
    return results
