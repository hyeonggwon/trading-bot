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
    add_bollinger_bands,
    add_ichimoku,
    add_macd,
    add_mfi,
    add_obv,
    add_roc,
    add_rsi,
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

# Phase 4 extra features (opt-in via build_feature_matrix(..., include_extra=True)).
# Three groups: regime indicators, lag/diff, session.
EXTRA_FEATURE_COLS = [
    # Regime
    "adx_bucket",  # 0 (low), 1 (mid), 2 (high) — categorical from adx_14
    "bb_bandwidth_20",  # (BB upper - lower) / middle
    "bb_bandwidth_pct_50",  # rolling percentile of bb_bandwidth_20 over 50 bars
    "realized_vol_20",  # std of close.pct_change() over 20 bars
    "realized_vol_pct_50",  # rolling percentile of realized_vol_20 over 50 bars
    # Lag / diff
    "roc_12_lag1",
    "roc_12_lag2",
    "rsi_14",
    "rsi_14_diff",
    "obv_roc_10_lag1",
    # Session
    "hour_kst",  # (UTC hour + 9) % 24 — Korean market timing
    "day_of_week",  # 0=Mon..6=Sun
]


def build_feature_matrix(
    df: pd.DataFrame,
    external_df: pd.DataFrame | None = None,
    include_extra: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Build feature matrix from OHLCV DataFrame with optional external features.

    Args:
        df: OHLCV DataFrame with DatetimeIndex. Should include warmup candles.
        external_df: Optional DataFrame with external data (kimchi_pct, funding_rate,
            fng_value, usd_krw). Merged via merge_asof(direction='backward').
        include_extra: When True, also adds the Phase 4 extras
            (regime / lag / session). Default False keeps prior models
            backwards-compatible — meta.feature_names captures whichever set
            was used at training time.

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

    active_cols = list(FEATURE_COLS)

    # Step 3b: Phase 4 extras (opt-in)
    if include_extra:
        # Regime: ADX bucket (low/mid/high) — boundaries match common practice
        # (ADX < 20 = no trend, 20-40 = trending, > 40 = strong trend).
        adx = df["adx_14"]
        df["adx_bucket"] = (
            pd.cut(
                adx,
                bins=[-float("inf"), 20.0, 40.0, float("inf")],
                labels=[0, 1, 2],
            )
            .astype("Int64")
            .astype(float)
        )

        # Bollinger band width (volatility regime)
        df = add_bollinger_bands(df, period=20, std=2.0)
        bb_mid = df["bb_middle_20_2.0"].replace(0, _nan)
        df["bb_bandwidth_20"] = (df["bb_upper_20_2.0"] - df["bb_lower_20_2.0"]) / bb_mid
        df["bb_bandwidth_pct_50"] = df["bb_bandwidth_20"].rolling(50, min_periods=10).rank(pct=True)

        # Realized volatility regime
        ret_1 = df["close"].pct_change()
        df["realized_vol_20"] = ret_1.rolling(20, min_periods=10).std()
        df["realized_vol_pct_50"] = df["realized_vol_20"].rolling(50, min_periods=10).rank(pct=True)

        # Lag / diff features — give the model access to short-term momentum
        # changes without forcing it to relearn first differences from scratch.
        df["roc_12_lag1"] = df["roc_12"].shift(1)
        df["roc_12_lag2"] = df["roc_12"].shift(2)
        df = add_rsi(df, period=14)
        df["rsi_14_diff"] = df["rsi_14"].diff(1)
        df["obv_roc_10_lag1"] = df["obv_roc_10"].shift(1)

        # Session features — Korean market hours often differ from UTC traders.
        # Both are integer-coded; LightGBM handles them as ordinal/categorical.
        if isinstance(df.index, pd.DatetimeIndex):
            df["hour_kst"] = ((df.index.hour + 9) % 24).astype(float)
            df["day_of_week"] = df.index.dayofweek.astype(float)
        else:
            df["hour_kst"] = float("nan")
            df["day_of_week"] = float("nan")

        df[EXTRA_FEATURE_COLS] = df[EXTRA_FEATURE_COLS].replace(
            [float("inf"), float("-inf")], float("nan")
        )
        active_cols.extend(EXTRA_FEATURE_COLS)

    # Step 4: External features (optional)

    if external_df is not None and len(external_df) > 0:
        # Merge external data using backward lookup (anti-lookahead)
        _raw_ext = ("kimchi_pct", "funding_rate", "fng_value", "usd_krw")
        ext_cols_present = [c for c in external_df.columns if c in _raw_ext]
        if ext_cols_present:
            # Normalize datetime index precision — merge_asof rejects mixed
            # datetime64[us]/[ms] keys (external parquets are saved as [ms, UTC]).
            if (
                isinstance(df.index, pd.DatetimeIndex)
                and isinstance(external_df.index, pd.DatetimeIndex)
                and df.index.dtype != external_df.index.dtype
            ):
                df = df.copy()
                df.index = df.index.astype(external_df.index.dtype)
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
