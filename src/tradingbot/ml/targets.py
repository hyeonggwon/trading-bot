"""Target variable construction for ML training.

WARNING: This module uses .shift(-N) which references FUTURE data.
Only call these functions in the offline training pipeline — NEVER inside
Strategy.indicators() or any code path that runs during backtesting/live trading.
"""

from __future__ import annotations

import pandas as pd


def build_target(
    df: pd.DataFrame,
    forward_candles: int = 4,
    threshold: float = 0.006,
) -> pd.Series:
    """Build binary classification target from forward returns.

    Target = 1 if forward return > threshold (profitable trade after fees).
    Target = 0 otherwise.
    Last `forward_candles` rows will be NaN (no future data available).

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        forward_candles: Number of candles to look ahead (e.g., 4 for 4h on 1h data).
        threshold: Minimum return to classify as positive (includes fee estimate).
                   Default 0.006 = 0.5% profit + 0.1% round-trip Upbit fee.

    Returns:
        pd.Series with binary labels (0/1) and NaN for last N rows.
    """
    fwd_return = df["close"].pct_change(forward_candles).shift(-forward_candles)
    target = pd.Series(index=df.index, dtype=float)
    valid = fwd_return.notna()
    target[valid] = (fwd_return[valid] > threshold).astype(float)
    return target
