"""Feature engineering for LightGBM model.

Builds 15 features from 8 indicators (top by gain importance).
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

# Feature column names — top 15 by gain importance across 24 models.
# Reduced from 36 to remove multicollinear/low-importance features.
FEATURE_COLS = [
    # Raw indicators
    "adx_14",
    "adx_pos_14",
    "adx_neg_14",
    "mfi_14",
    "roc_12",
    "macd_hist_12_26_9",
    # Derived
    "obv_roc_10",
    "atr_pct_14",
    "macd_norm",
    "ichi_dist",
    "volume_ratio",
    "close_roc_5",
    "hl_range_pct",
    # Rolling stats
    "close_std_20",
    "atr_rank_50",
]


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build feature matrix from OHLCV DataFrame.

    Args:
        df: OHLCV DataFrame with DatetimeIndex. Should include warmup candles.

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

    # Step 1: Compute base indicators (only those needed for top-15 features)
    df = add_adx(df, period=14)
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

    # Price momentum
    df["close_roc_5"] = df["close"].pct_change(5)

    # Intra-candle range
    df["hl_range_pct"] = (df["high"] - df["low"]) / df["close"]

    # Step 3: Rolling statistical features
    df["close_std_20"] = df["close"].pct_change().rolling(20).std()
    df["atr_rank_50"] = df["atr_14"].rolling(50).rank(pct=True)

    # Step 4: Replace any inf values with NaN
    df[FEATURE_COLS] = df[FEATURE_COLS].replace([float("inf"), float("-inf")], float("nan"))

    return df, FEATURE_COLS
