"""Technical indicator wrappers using the `ta` library.

All functions take a DataFrame with OHLCV columns and return the same
DataFrame with indicator columns appended.
"""

from __future__ import annotations

import pandas as pd
import ta


def add_sma(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.DataFrame:
    """Simple Moving Average."""
    df[f"sma_{period}"] = df[column].rolling(window=period).mean()
    return df


def add_ema(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.DataFrame:
    """Exponential Moving Average."""
    df[f"ema_{period}"] = df[column].ewm(span=period, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14, column: str = "close") -> pd.DataFrame:
    """Relative Strength Index."""
    df[f"rsi_{period}"] = ta.momentum.RSIIndicator(close=df[column], window=period).rsi()
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "close",
) -> pd.DataFrame:
    """MACD (Moving Average Convergence Divergence)."""
    macd = ta.trend.MACD(close=df[column], window_fast=fast, window_slow=slow, window_sign=signal)
    df[f"macd_{fast}_{slow}_{signal}"] = macd.macd()
    df[f"macd_signal_{fast}_{slow}_{signal}"] = macd.macd_signal()
    df[f"macd_hist_{fast}_{slow}_{signal}"] = macd.macd_diff()
    return df


def add_bollinger_bands(
    df: pd.DataFrame, period: int = 20, std: float = 2.0, column: str = "close"
) -> pd.DataFrame:
    """Bollinger Bands."""
    bb = ta.volatility.BollingerBands(close=df[column], window=period, window_dev=std)
    df[f"bb_upper_{period}_{std}"] = bb.bollinger_hband()
    df[f"bb_middle_{period}_{std}"] = bb.bollinger_mavg()
    df[f"bb_lower_{period}_{std}"] = bb.bollinger_lband()
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range."""
    df[f"atr_{period}"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=period
    ).average_true_range()
    return df


def add_stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> pd.DataFrame:
    """Stochastic Oscillator."""
    stoch = ta.momentum.StochasticOscillator(
        high=df["high"], low=df["low"], close=df["close"],
        window=k_period, smooth_window=d_period,
    )
    df[f"stoch_k_{k_period}"] = stoch.stoch()
    df[f"stoch_d_{k_period}_{d_period}"] = stoch.stoch_signal()
    return df


def add_volume_sma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Volume Simple Moving Average."""
    df[f"volume_sma_{period}"] = df["volume"].rolling(window=period).mean()
    return df


# ── Trend indicators ──────────────────────────────────────────────


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index (ADX + ±DI)."""
    adx = ta.trend.ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=period)
    df[f"adx_{period}"] = adx.adx()
    df[f"adx_pos_{period}"] = adx.adx_pos()
    df[f"adx_neg_{period}"] = adx.adx_neg()
    return df


def add_ichimoku(
    df: pd.DataFrame, window1: int = 9, window2: int = 26, window3: int = 52
) -> pd.DataFrame:
    """Ichimoku Cloud (conversion, base, span A, span B)."""
    suffix = f"_{window1}_{window2}_{window3}"
    col_a = f"ichi_a{suffix}"
    if col_a in df.columns:
        return df
    ichi = ta.trend.IchimokuIndicator(
        high=df["high"], low=df["low"], window1=window1, window2=window2, window3=window3
    )
    df[f"ichi_conv{suffix}"] = ichi.ichimoku_conversion_line()
    df[f"ichi_base{suffix}"] = ichi.ichimoku_base_line()
    df[col_a] = ichi.ichimoku_a()
    df[f"ichi_b{suffix}"] = ichi.ichimoku_b()
    return df


def add_aroon(df: pd.DataFrame, period: int = 25) -> pd.DataFrame:
    """Aroon Up / Down indicator."""
    aroon = ta.trend.AroonIndicator(high=df["high"], low=df["low"], window=period)
    df[f"aroon_up_{period}"] = aroon.aroon_up()
    df[f"aroon_down_{period}"] = aroon.aroon_down()
    return df


# ── Momentum indicators ──────────────────────────────────────────


def add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Commodity Channel Index."""
    df[f"cci_{period}"] = ta.trend.CCIIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=period
    ).cci()
    return df


def add_roc(df: pd.DataFrame, period: int = 12) -> pd.DataFrame:
    """Rate of Change."""
    df[f"roc_{period}"] = ta.momentum.ROCIndicator(close=df["close"], window=period).roc()
    return df


# ── Volume indicators ─────────────────────────────────────────────


def add_mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Money Flow Index."""
    df[f"mfi_{period}"] = ta.volume.MFIIndicator(
        high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=period
    ).money_flow_index()
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """On-Balance Volume."""
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(
        close=df["close"], volume=df["volume"]
    ).on_balance_volume()
    return df


# ── Volatility indicators ────────────────────────────────────────


def add_keltner_channel(
    df: pd.DataFrame, period: int = 20, atr_period: int = 10
) -> pd.DataFrame:
    """Keltner Channel (upper, middle, lower)."""
    kc = ta.volatility.KeltnerChannel(
        high=df["high"], low=df["low"], close=df["close"],
        window=period, window_atr=atr_period,
    )
    df[f"kc_upper_{period}"] = kc.keltner_channel_hband()
    df[f"kc_middle_{period}"] = kc.keltner_channel_mband()
    df[f"kc_lower_{period}"] = kc.keltner_channel_lband()
    return df


def add_donchian_channel(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Donchian Channel (upper, middle, lower)."""
    dc = ta.volatility.DonchianChannel(
        high=df["high"], low=df["low"], close=df["close"], window=period
    )
    df[f"dc_upper_{period}"] = dc.donchian_channel_hband()
    df[f"dc_middle_{period}"] = dc.donchian_channel_mband()
    df[f"dc_lower_{period}"] = dc.donchian_channel_lband()
    return df


# ── Derived indicators (pandas) ──────────────────────────────────


def add_zscore(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.DataFrame:
    """Z-Score: (price - SMA) / StdDev."""
    sma = df[column].rolling(window=period).mean()
    std = df[column].rolling(window=period).std().replace(0, float("nan"))
    df[f"zscore_{period}"] = (df[column] - sma) / std
    return df


def add_pct_from_ma(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.DataFrame:
    """Percent distance from moving average."""
    sma = df[column].rolling(window=period).mean().replace(0, float("nan"))
    df[f"pct_from_ma_{period}"] = (df[column] - sma) / sma * 100
    return df
