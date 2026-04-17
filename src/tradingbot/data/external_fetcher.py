"""External data fetchers for ML features.

Fetches kimchi premium components, funding rate, Fear & Greed Index,
and USD/KRW exchange rate. All sources are free and require no authentication.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import ccxt
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_EXTERNAL_DIR = Path("data/external")


# ---------------------------------------------------------------------------
# Binance BTC/USDT OHLCV (for kimchi premium calculation)
# ---------------------------------------------------------------------------


def fetch_binance_ohlcv(
    since: datetime,
    until: datetime | None = None,
    timeframe: str = "1h",
    limit: int = 500,
) -> pd.DataFrame:
    """Fetch BTC/USDT OHLCV from Binance via CCXT.

    Returns DataFrame with columns [open, high, low, close, volume]
    and DatetimeIndex named 'timestamp'.
    """
    exchange = ccxt.binance({"enableRateLimit": True})

    if since.tzinfo is None:
        since = since.replace(tzinfo=UTC)
    if until and until.tzinfo is None:
        until = until.replace(tzinfo=UTC)
    since_ms = int(since.timestamp() * 1000)
    until_ms = int(until.timestamp() * 1000) if until else None

    tf_ms_map = {
        "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
    }
    tf_ms = tf_ms_map.get(timeframe, 3_600_000)

    all_rows: list[list] = []
    max_pages = 5000
    for _ in range(max_pages):
        try:
            ohlcv = exchange.fetch_ohlcv(
                "BTC/USDT", timeframe=timeframe, since=since_ms, limit=limit
            )
        except ccxt.BaseError as e:
            log.warning(f"Binance OHLCV fetch error: {e}")
            break

        if not ohlcv:
            break

        all_rows.extend(ohlcv)
        last_ts = ohlcv[-1][0]
        since_ms = last_ts + tf_ms

        if until_ms and since_ms > until_ms:
            break
        if since_ms > int(time.time() * 1000):
            break
        if len(ohlcv) < limit // 2:
            break

        time.sleep(0.1)  # rate limit

    if not all_rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")].sort_index()

    if until:
        df = df[df.index <= until]

    return df.astype(float)


# ---------------------------------------------------------------------------
# Binance Funding Rate (perpetual futures)
# ---------------------------------------------------------------------------


def fetch_funding_rate(
    since: datetime,
    until: datetime | None = None,
) -> pd.DataFrame:
    """Fetch BTC/USDT perpetual funding rate from Binance.

    Returns DataFrame with column 'funding_rate' and DatetimeIndex.
    Funding rate settles every 8 hours (00:00, 08:00, 16:00 UTC).
    """
    exchange = ccxt.binance({"options": {"defaultType": "swap"}})

    if since.tzinfo is None:
        since = since.replace(tzinfo=UTC)
    since_ms = int(since.timestamp() * 1000)

    all_rows: list[dict] = []
    max_pages = 5000
    for _ in range(max_pages):
        try:
            rates = exchange.fetch_funding_rate_history(
                "BTC/USDT:USDT", since=since_ms, limit=500
            )
        except ccxt.BaseError as e:
            log.warning(f"Funding rate fetch error: {e}")
            break

        if not rates:
            break

        all_rows.extend(rates)
        last_ts = rates[-1]["timestamp"]
        since_ms = last_ts + 1

        now_ms = int(time.time() * 1000)
        if since_ms > now_ms:
            break
        if len(rates) < 250:
            break

        time.sleep(0.1)

    if not all_rows:
        return pd.DataFrame(columns=["funding_rate"])

    df = pd.DataFrame([
        {"timestamp": r["timestamp"], "funding_rate": r["fundingRate"]}
        for r in all_rows
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")].sort_index()

    if until:
        until_aware = until.replace(tzinfo=UTC) if until.tzinfo is None else until
        df = df[df.index <= until_aware]

    return df


# ---------------------------------------------------------------------------
# USD/KRW from FRED (daily)
# ---------------------------------------------------------------------------


def fetch_usd_krw(since: datetime | None = None) -> pd.DataFrame:
    """Fetch daily USD/KRW exchange rate from FRED (no auth required).

    Returns DataFrame with column 'usd_krw' and DatetimeIndex.
    """
    base_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXKOUS"
    if since:
        base_url += f"&cosd={since.strftime('%Y-%m-%d')}"

    try:
        with urllib.request.urlopen(base_url, timeout=30) as resp:
            text = resp.read().decode("utf-8")
    except Exception as e:
        log.warning(f"FRED USD/KRW fetch error: {e}")
        return pd.DataFrame(columns=["usd_krw"])

    # Shift by +1 day to prevent lookahead. DEXKOUS is observed at NY noon
    # (~17:00 UTC) and published to FRED later that same day, so assigning
    # the rate to DATE 00:00 UTC would expose the model to future info during
    # the 0–17h window. Treating it as available only on the *next* UTC day
    # is a conservative and simple guarantee for hourly crypto backtests.
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for row in reader:
        val = row.get("DEXKOUS", "").strip()
        if val and val != ".":
            try:
                rows.append({
                    "timestamp": pd.Timestamp(row["DATE"], tz="UTC") + pd.Timedelta(days=1),
                    "usd_krw": float(val),
                })
            except (ValueError, KeyError):
                continue

    if not rows:
        return pd.DataFrame(columns=["usd_krw"])

    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return df


# ---------------------------------------------------------------------------
# Fear & Greed Index (daily)
# ---------------------------------------------------------------------------


def fetch_fear_greed(limit: int = 0) -> pd.DataFrame:
    """Fetch Crypto Fear & Greed Index from alternative.me API.

    Args:
        limit: Number of days to fetch. 0 = full history.

    Returns DataFrame with column 'fng_value' (0-100) and DatetimeIndex.
    """
    url = f"https://api.alternative.me/fng/?limit={limit}&format=json"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "tradingbot/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        log.warning(f"Fear & Greed fetch error: {e}")
        return pd.DataFrame(columns=["fng_value"])

    entries = data.get("data", [])
    if not entries:
        return pd.DataFrame(columns=["fng_value"])

    # Shift by +1 day to prevent lookahead. Alternative.me's FNG is computed
    # from a full day of market data and published near day-end UTC, but each
    # entry's timestamp is the day's 00:00 UTC. Using it as-is would leak up
    # to 24h of future info during that UTC day.
    rows = []
    for entry in entries:
        try:
            raw_ts = pd.Timestamp(int(entry["timestamp"]), unit="s", tz="UTC")
            rows.append({
                "timestamp": raw_ts + pd.Timedelta(days=1),
                "fng_value": float(entry["value"]),
            })
        except (ValueError, KeyError):
            continue

    if not rows:
        return pd.DataFrame(columns=["fng_value"])

    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ---------------------------------------------------------------------------
# Kimchi Premium computation
# ---------------------------------------------------------------------------


def compute_kimchi_premium(
    upbit_df: pd.DataFrame,
    binance_df: pd.DataFrame,
    usd_krw_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute kimchi premium from Upbit KRW, Binance USDT, and USD/KRW rate.

    Formula: kimchi_pct = (upbit_close / (binance_close * usd_krw)) - 1

    Uses merge_asof(direction='backward') for anti-lookahead on daily USD/KRW.

    Returns DataFrame with column 'kimchi_pct' and DatetimeIndex.
    """
    # merge_asof requires both sides sorted on the key
    if not upbit_df.index.is_monotonic_increasing:
        upbit_df = upbit_df.sort_index()
    if not binance_df.index.is_monotonic_increasing:
        binance_df = binance_df.sort_index()
    if not usd_krw_df.index.is_monotonic_increasing:
        usd_krw_df = usd_krw_df.sort_index()

    # Align USD/KRW (daily) to hourly candles — use last known rate
    merged = pd.merge_asof(
        upbit_df[["close"]].rename(columns={"close": "upbit_close"}),
        usd_krw_df,
        left_index=True,
        right_index=True,
        direction="backward",
    )

    # Align Binance close to Upbit timestamps
    merged = pd.merge_asof(
        merged,
        binance_df[["close"]].rename(columns={"close": "binance_close"}),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    # Compute premium
    binance_krw = merged["binance_close"] * merged["usd_krw"]
    merged["kimchi_pct"] = (merged["upbit_close"] / binance_krw) - 1.0

    return merged[["kimchi_pct"]].dropna()


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def save_external(
    df: pd.DataFrame,
    name: str,
    data_dir: Path = DEFAULT_EXTERNAL_DIR,
) -> Path:
    """Save external data to parquet with auto-merge."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{name}.parquet"

    if path.exists():
        existing = pd.read_parquet(path, engine="pyarrow")
        df = pd.concat([existing, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()

    df.to_parquet(path, engine="pyarrow")
    log.info(f"External data saved: {path} ({len(df)} rows)")
    return path


def load_external(
    name: str,
    data_dir: Path = DEFAULT_EXTERNAL_DIR,
) -> pd.DataFrame | None:
    """Load external data from parquet. Returns None if not found."""
    path = data_dir / f"{name}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path, engine="pyarrow").sort_index()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def load_external_components(
    data_dir: Path = DEFAULT_EXTERNAL_DIR,
) -> dict[str, pd.DataFrame | None]:
    """Load raw external data components from parquet (unaligned).

    Returns a dict with keys {binance, usd_krw, funding, fng} where each
    value is the source DataFrame or None if the parquet file is missing.
    Keeping components separate lets callers align to different upbit
    DataFrames (e.g., one per symbol) without re-reading parquet.
    """
    return {
        "binance": load_external("binance_btc_usdt", data_dir),
        "usd_krw": load_external("usd_krw", data_dir),
        "funding": load_external("funding_rate", data_dir),
        "fng": load_external("fear_greed", data_dir),
    }


def align_external_to(
    upbit_df: pd.DataFrame,
    components: dict[str, pd.DataFrame | None],
) -> pd.DataFrame | None:
    """Align pre-loaded external components to a specific upbit_df index.

    Uses merge_asof(direction='backward') — only past external values are
    visible at each upbit timestamp (anti-lookahead preserved).
    """
    # merge_asof requires the left side sorted on the key
    if not upbit_df.index.is_monotonic_increasing:
        upbit_df = upbit_df.sort_index()

    binance_df = components.get("binance")
    usd_krw_df = components.get("usd_krw")
    funding_df = components.get("funding")
    fng_df = components.get("fng")

    frames: list[pd.DataFrame] = []

    if binance_df is not None and usd_krw_df is not None:
        try:
            kimchi_df = compute_kimchi_premium(upbit_df, binance_df, usd_krw_df)
            frames.append(kimchi_df)
        except Exception as e:
            log.warning(f"Kimchi premium computation failed: {e}")

    if funding_df is not None:
        frames.append(funding_df[["funding_rate"]])

    if fng_df is not None:
        frames.append(fng_df[["fng_value"]])

    if usd_krw_df is not None:
        frames.append(usd_krw_df[["usd_krw"]])

    if not frames:
        return None

    result = upbit_df[[]].copy()
    for frame in frames:
        result = pd.merge_asof(
            result,
            frame.sort_index(),
            left_index=True,
            right_index=True,
            direction="backward",
        )

    if result.dropna(how="all").empty:
        return None

    return result


def build_external_df(
    upbit_df: pd.DataFrame,
    data_dir: Path = DEFAULT_EXTERNAL_DIR,
) -> pd.DataFrame | None:
    """Load all external data and merge into a single DataFrame.

    Thin wrapper over ``load_external_components`` + ``align_external_to``.
    Callers needing to reuse components across multiple DataFrames should
    call the two helpers directly to avoid re-reading parquet each time.
    """
    components = load_external_components(data_dir)
    return align_external_to(upbit_df, components)


def fetch_all_external(
    since: datetime,
    until: datetime | None = None,
    data_dir: Path = DEFAULT_EXTERNAL_DIR,
) -> dict[str, int]:
    """Fetch all external data sources and save to parquet.

    Returns dict of {source_name: row_count} for successfully fetched sources.
    """
    results: dict[str, int] = {}

    # 1. Binance BTC/USDT
    log.info("Fetching Binance BTC/USDT OHLCV...")
    df = fetch_binance_ohlcv(since, until)
    if not df.empty:
        save_external(df, "binance_btc_usdt", data_dir)
        results["binance_btc_usdt"] = len(df)

    # 2. Funding rate
    log.info("Fetching Binance funding rate...")
    df = fetch_funding_rate(since, until)
    if not df.empty:
        save_external(df, "funding_rate", data_dir)
        results["funding_rate"] = len(df)

    # 3. USD/KRW
    log.info("Fetching FRED USD/KRW...")
    df = fetch_usd_krw(since)
    if not df.empty:
        save_external(df, "usd_krw", data_dir)
        results["usd_krw"] = len(df)

    # 4. Fear & Greed
    log.info("Fetching Fear & Greed Index...")
    df = fetch_fear_greed(limit=0)
    if not df.empty:
        save_external(df, "fear_greed", data_dir)
        results["fear_greed"] = len(df)

    return results
