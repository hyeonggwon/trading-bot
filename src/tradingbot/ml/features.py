"""Feature engineering for LightGBM model.

Builds ~33 features from existing indicators + derived calculations.
All operations use only past data (no lookahead).
"""

from __future__ import annotations

import pandas as pd

from tradingbot.data.indicators import (
    add_adx,
    add_aroon,
    add_atr,
    add_bollinger_bands,
    add_cci,
    add_donchian_channel,
    add_ema,
    add_ichimoku,
    add_keltner_channel,
    add_macd,
    add_mfi,
    add_obv,
    add_pct_from_ma,
    add_roc,
    add_rsi,
    add_stochastic,
    add_volume_sma,
    add_zscore,
)

WARMUP_CANDLES = 52  # Ichimoku window3 — longest lookback

# Feature column names (must match what build_feature_matrix produces)
FEATURE_COLS = [
    # Raw indicators
    "rsi_14",
    "adx_14",
    "adx_pos_14",
    "adx_neg_14",
    "stoch_k_14",
    "stoch_d_14_3",
    "aroon_up_25",
    "aroon_down_25",
    "mfi_14",
    "cci_20",
    "roc_12",
    "zscore_20",
    "pct_from_ma_20",
    "macd_hist_12_26_9",
    # Derived
    "obv_roc_10",
    "atr_pct_14",
    "bb_pos_20",
    "bb_kc_squeeze",
    "dc_pos_20",
    "macd_norm",
    "ichi_dist",
    "adx_di_diff",
    "volume_ratio",
    "close_roc_1",
    "close_roc_3",
    "close_roc_5",
    "hl_range_pct",
    "candle_body",
    "stoch_kd_diff",
    "rsi_roc_3",
    # Rolling stats
    "close_std_20",
    "atr_rank_50",
    "rsi_dist_from_50",
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

    # Step 1: Compute base indicators
    df = add_rsi(df, period=14)
    df = add_adx(df, period=14)
    df = add_stochastic(df, k_period=14, d_period=3)
    df = add_aroon(df, period=25)
    df = add_mfi(df, period=14)
    df = add_cci(df, period=20)
    df = add_roc(df, period=12)
    df = add_zscore(df, period=20)
    df = add_pct_from_ma(df, period=20)
    df = add_macd(df, fast=12, slow=26, signal=9)
    df = add_bollinger_bands(df, period=20, std=2.0)
    df = add_atr(df, period=14)
    df = add_obv(df)
    df = add_volume_sma(df, period=20)
    df = add_keltner_channel(df, period=20, atr_period=10)
    df = add_donchian_channel(df, period=20)
    df = add_ichimoku(df, window1=9, window2=26, window3=52)
    df = add_ema(df, period=20)

    # Step 2: Derived features
    _nan = float("nan")

    # OBV rate of change (raw OBV is non-stationary)
    df["obv_roc_10"] = df["obv"].pct_change(10)

    # ATR as % of price (normalized volatility)
    df["atr_pct_14"] = df["atr_14"] / df["close"]

    # Bollinger Band position (0=lower, 1=upper)
    bb_range = (df["bb_upper_20_2.0"] - df["bb_lower_20_2.0"]).replace(0, _nan)
    df["bb_pos_20"] = (df["close"] - df["bb_lower_20_2.0"]) / bb_range

    # Keltner squeeze (BB narrower than KC)
    df["bb_kc_squeeze"] = (
        (df["bb_upper_20_2.0"] - df["bb_lower_20_2.0"])
        < (df["kc_upper_20"] - df["kc_lower_20"])
    ).astype(int)

    # Donchian Channel position
    dc_range = (df["dc_upper_20"] - df["dc_lower_20"]).replace(0, _nan)
    df["dc_pos_20"] = (df["close"] - df["dc_lower_20"]) / dc_range

    # MACD normalized by ATR
    df["macd_norm"] = df["macd_12_26_9"] / df["atr_14"].replace(0, _nan)

    # Ichimoku cloud distance
    ichi_top = df[["ichi_a_9_26_52", "ichi_b_9_26_52"]].max(axis=1)
    df["ichi_dist"] = (df["close"] - ichi_top) / df["close"]

    # ADX direction differential
    df["adx_di_diff"] = df["adx_pos_14"] - df["adx_neg_14"]

    # Volume ratio
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"].replace(0, _nan)

    # Price momentum lags
    for lag in [1, 3, 5]:
        df[f"close_roc_{lag}"] = df["close"].pct_change(lag)

    # Intra-candle range
    df["hl_range_pct"] = (df["high"] - df["low"]) / df["close"]

    # Candle body (positive = green)
    df["candle_body"] = (df["close"] - df["open"]) / df["open"].replace(0, _nan)

    # Stochastic %K - %D diff
    df["stoch_kd_diff"] = df["stoch_k_14"] - df["stoch_d_14_3"]

    # RSI rate of change
    df["rsi_roc_3"] = df["rsi_14"].diff(3)

    # Step 3: Rolling statistical features
    df["close_std_20"] = df["close"].pct_change().rolling(20).std()
    df["atr_rank_50"] = df["atr_14"].rolling(50).rank(pct=True)
    df["rsi_dist_from_50"] = df["rsi_14"] - 50.0

    # Step 4: Replace any inf values with NaN (e.g., OBV pct_change through zero)
    import numpy as np

    df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], float("nan"))

    return df, FEATURE_COLS
