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
    df[f"bb_upper_{period}"] = bb.bollinger_hband()
    df[f"bb_middle_{period}"] = bb.bollinger_mavg()
    df[f"bb_lower_{period}"] = bb.bollinger_lband()
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
