"""Target variable construction for ML training.

WARNING: This module uses .shift(-N) and forward-window scans, which
reference FUTURE data. Only call these functions in the offline training
pipeline — NEVER inside Strategy.indicators() or any code path that runs
during backtesting/live trading.

Three label flavors are exposed:

- ``build_target`` (binary, fixed return threshold) — original baseline used
  in Phase 1.
- ``build_target_atr`` (binary, vol-normalized threshold) — same horizon but
  the threshold is ``atr_mult * ATR%`` per row, so the label adapts to local
  volatility.
- ``build_target_triple_barrier`` (binary, Lopez de Prado triple barrier) —
  scans up to ``forward_candles`` ahead and labels by which barrier (upper
  vs lower, scaled by ATR) is touched first; ties broken by vertical-barrier
  sign of the close.

All return a binary float Series indexed like ``df`` with NaN for rows
without enough lookahead or warmup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _atr_pct(df: pd.DataFrame, period: int) -> pd.Series:
    """ATR as a fraction of close, computed without mutating ``df``.

    Mirrors ``tradingbot.data.indicators.add_atr`` but avoids copying or
    appending columns to the caller's frame so callers don't have to defensively
    ``.copy()``.
    """
    import ta

    atr = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=period
    ).average_true_range()
    return atr / df["close"]


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


def build_target_atr(
    df: pd.DataFrame,
    forward_candles: int = 4,
    atr_mult: float = 1.0,
    atr_period: int = 14,
) -> pd.Series:
    """Binary target with per-row volatility-normalized threshold.

    Labels 1 when ``forward_return > atr_mult * (ATR / close)`` measured at
    the entry candle. This adapts the "what counts as profitable" bar to
    local volatility so the positive rate stays balanced across regimes.

    Rows with NaN ATR (warmup) or NaN forward return (tail) stay NaN.
    """
    atr_pct = _atr_pct(df, period=atr_period)
    fwd_return = df["close"].pct_change(forward_candles).shift(-forward_candles)

    threshold_per_row = atr_mult * atr_pct
    target = pd.Series(index=df.index, dtype=float)
    valid = fwd_return.notna() & atr_pct.notna()
    target[valid] = (fwd_return[valid] > threshold_per_row[valid]).astype(float)
    return target


def build_target_triple_barrier(
    df: pd.DataFrame,
    forward_candles: int = 4,
    atr_mult: float = 1.0,
    atr_period: int = 14,
    threshold: float = 0.006,
) -> pd.Series:
    """Lopez de Prado triple-barrier label, collapsed to binary.

    For each candle ``i`` we look at the next ``forward_candles`` bars and
    set:

    - upper_barrier = close[i] * (1 + atr_mult * ATR%[i])
    - lower_barrier = close[i] * (1 - atr_mult * ATR%[i])

    Label is:
        1 if upper barrier touched first (high[j] >= upper_barrier)
        0 if lower barrier touched first (low[j]  <= lower_barrier)
        if neither barrier hit, fall back to vertical barrier:
            1 if (close[i + forward_candles] / close[i] - 1) > threshold else 0

    The vertical-barrier ``threshold`` matches ``build_target``'s default
    (0.006 = 0.5% profit + 0.1% Upbit round-trip fee) so quiet rows where
    neither barrier triggers aren't labelled positive on sub-fee drift.

    Returns a binary float Series; rows with NaN ATR (warmup) or insufficient
    lookahead stay NaN.
    """
    atr_pct = _atr_pct(df, period=atr_period).to_numpy()
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    n = len(df)
    out = np.full(n, np.nan, dtype=float)
    horizon = int(forward_candles)
    if horizon < 1:
        return pd.Series(out, index=df.index)

    for i in range(n - horizon):
        atr_i = atr_pct[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue
        entry = close[i]
        upper = entry * (1.0 + atr_mult * atr_i)
        lower = entry * (1.0 - atr_mult * atr_i)
        upper_hit = -1
        lower_hit = -1
        for j in range(i + 1, i + horizon + 1):
            if upper_hit < 0 and high[j] >= upper:
                upper_hit = j
            if lower_hit < 0 and low[j] <= lower:
                lower_hit = j
            if upper_hit >= 0 and lower_hit >= 0:
                break
        if upper_hit >= 0 and (lower_hit < 0 or upper_hit <= lower_hit):
            # Tie (same bar touched both) is treated as upper-hit because
            # entries fill at the open and we want to avoid penalizing the
            # bar's high range.
            out[i] = 1.0
        elif lower_hit >= 0:
            out[i] = 0.0
        else:
            # Vertical barrier — neither barrier touched within the horizon.
            # Apply the same fee-aware threshold as build_target so quiet
            # rows don't get spurious positive labels on sub-fee drift.
            forward_ret = close[i + horizon] / entry - 1.0
            out[i] = 1.0 if forward_ret > threshold else 0.0

    return pd.Series(out, index=df.index)
