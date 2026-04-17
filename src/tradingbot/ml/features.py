"""Feature engineering for LightGBM model.

Builds 10 technical features from 8 indicators (reduced from 15 to remove
multicollinearity: ADX 3→2, MACD 2→1, ROC 2→1).
All operations use only past data (no lookahead).
"""

from __future__ import annotations

import pandas as pd

from tradingbot.data.indicators import (
    add_adx,
    add_atr,
    add_ichimoku,
    add_macd,
    add_mfi,
    add_obv,
    add_roc,
    add_volume_sma,
)

WARMUP_CANDLES = 52  # Ichimoku window3 — longest lookback

# External feature columns (optional — present when external data is provided).
EXTERNAL_FEATURE_COLS = [
    "kimchi_pct",
    "kimchi_zscore_20",
    "funding_rate",
    "funding_rate_zscore_20",
    "fng_value",
    "usd_krw_change",
]

# Technical feature columns — 10 features (reduced from 15).
# Removed multicollinear features: adx_pos/neg (→ adx_diff), macd_hist (kept macd_norm),
# close_roc_5 (kept roc_12), close_std_20, atr_rank_50.
FEATURE_COLS = [
    # Raw indicators
    "adx_14",
    "adx_diff",  # adx_pos_14 - adx_neg_14 (directional strength)
    "mfi_14",
    "roc_12",
    # Derived
    "obv_roc_10",
    "atr_pct_14",
    "macd_norm",
    "ichi_dist",
    "volume_ratio",
    "hl_range_pct",
]


def build_feature_matrix(
    df: pd.DataFrame,
    external_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build feature matrix from OHLCV DataFrame with optional external features.

    Args:
        df: OHLCV DataFrame with DatetimeIndex. Should include warmup candles.
        external_df: Optional DataFrame with external data (kimchi_pct, funding_rate,
            fng_value, usd_krw). Merged via merge_asof(direction='backward').

    Returns:
        Tuple of (DataFrame with feature columns added, list of feature column names).
        If data is too short for indicators, feature columns will contain NaN.
    """
    # Guard: if data is too short, add NaN columns and return early
    if len(df) < WARMUP_CANDLES:
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = float("nan")
        return df, FEATURE_COLS

    # Step 1: Compute base indicators
    df = add_adx(df, period=14)
    df["adx_diff"] = df["adx_pos_14"] - df["adx_neg_14"]
    df = add_mfi(df, period=14)
    df = add_roc(df, period=12)
    df = add_macd(df, fast=12, slow=26, signal=9)
    df = add_atr(df, period=14)
    df = add_obv(df)
    df = add_volume_sma(df, period=20)
    df = add_ichimoku(df, window1=9, window2=26, window3=52)

    # Step 2: Derived features
    _nan = float("nan")

    # OBV momentum (normalized by rolling std)
    obv_std = df["obv"].rolling(50).std().replace(0, _nan)
    df["obv_roc_10"] = df["obv"].diff(10) / obv_std

    # ATR as % of price (normalized volatility)
    df["atr_pct_14"] = df["atr_14"] / df["close"]

    # MACD normalized by ATR
    df["macd_norm"] = df["macd_12_26_9"] / df["atr_14"].replace(0, _nan)

    # Ichimoku cloud distance
    ichi_top = df[["ichi_a_9_26_52", "ichi_b_9_26_52"]].max(axis=1)
    df["ichi_dist"] = (df["close"] - ichi_top) / df["close"]

    # Volume ratio
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"].replace(0, _nan)

    # Intra-candle range
    df["hl_range_pct"] = (df["high"] - df["low"]) / df["close"]

    # Step 3: Replace any inf values with NaN
    df[FEATURE_COLS] = df[FEATURE_COLS].replace([float("inf"), float("-inf")], float("nan"))

    # Step 4: External features (optional)
    active_cols = list(FEATURE_COLS)

    if external_df is not None and len(external_df) > 0:
        # Merge external data using backward lookup (anti-lookahead)
        _raw_ext = ("kimchi_pct", "funding_rate", "fng_value", "usd_krw")
        ext_cols_present = [c for c in external_df.columns if c in _raw_ext]
        if ext_cols_present:
            df = pd.merge_asof(
                df,
                external_df[ext_cols_present],
                left_index=True,
                right_index=True,
                direction="backward",
            )

        # Compute rolling z-scores from raw values
        _nan = float("nan")
        if "kimchi_pct" in df.columns and df["kimchi_pct"].notna().any():
            mu = df["kimchi_pct"].rolling(20, min_periods=5).mean()
            std = df["kimchi_pct"].rolling(20, min_periods=5).std().replace(0, _nan)
            df["kimchi_zscore_20"] = (df["kimchi_pct"] - mu) / std

        if "funding_rate" in df.columns and df["funding_rate"].notna().any():
            mu = df["funding_rate"].rolling(20, min_periods=5).mean()
            std = df["funding_rate"].rolling(20, min_periods=5).std().replace(0, _nan)
            df["funding_rate_zscore_20"] = (df["funding_rate"] - mu) / std

        if "usd_krw" in df.columns and df["usd_krw"].notna().any():
            # 24-period rolling change — USD/KRW is daily data merged into
            # hourly candles, so pct_change(1) is 0 for 23/24 rows. Using
            # pct_change(24) gives each hourly candle the day-over-day change.
            df["usd_krw_change"] = df["usd_krw"].pct_change(24)

        # Add available external features to active columns
        for col in EXTERNAL_FEATURE_COLS:
            if col in df.columns and df[col].notna().any():
                active_cols.append(col)

        # Clean inf values in external features
        ext_in_df = [c for c in EXTERNAL_FEATURE_COLS if c in df.columns]
        if ext_in_df:
            df[ext_in_df] = df[ext_in_df].replace([float("inf"), float("-inf")], float("nan"))

    return df, active_cols
