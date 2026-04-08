"""Vectorized screening engine for fast scan/combine-scan.

Computes entry/exit signals as boolean arrays across the full DataFrame,
then extracts trades in a single O(N) pass. ~100x faster than the
candle-by-candle engine for screening purposes.

NOT a replacement for the full BacktestEngine — simplified position sizing
and equity tracking. Use for ranking strategies, then re-verify top results
with the full engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd

from tradingbot.backtest.report import PERIODS_PER_YEAR
from tradingbot.strategy.filters.base import BaseFilter
from tradingbot.strategy.filters.exit import AtrTrailingExitFilter


@dataclass
class VectorizedResult:
    """Lightweight backtest result for screening."""

    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int


def vectorized_backtest(
    df: pd.DataFrame,
    entry_filters: list[BaseFilter],
    exit_filters: list[BaseFilter],
    initial_balance: float = 10_000_000,
    fee_rate: float = 0.0005,
    slippage_pct: float = 0.001,
    stop_loss_pct: float = 0.02,
    max_position_pct: float = 0.10,
    timeframe: str = "1h",
) -> VectorizedResult:
    """Run a vectorized backtest for screening.

    Args:
        df: DataFrame with indicator columns already computed.
        entry_filters: AND-combined entry filters.
        exit_filters: OR-combined exit filters.
    """
    if len(df) < 3:
        return VectorizedResult(0.0, 0.0, 0.0, 0.0, 0.0, 0)

    # --- Build signal arrays ---
    entry_mask = pd.Series(True, index=df.index)
    for f in entry_filters:
        if f.role == "exit":
            continue
        if not f.supports_vectorized:
            return VectorizedResult(0.0, 0.0, 0.0, 0.0, 0.0, 0)
        entry_mask = entry_mask & f.vectorized_entry(df)

    # Separate ATR trailing from other exit filters
    atr_filter: AtrTrailingExitFilter | None = None
    regular_exit_filters: list[BaseFilter] = []
    for f in exit_filters:
        if isinstance(f, AtrTrailingExitFilter):
            atr_filter = f
        else:
            regular_exit_filters.append(f)

    exit_mask = pd.Series(False, index=df.index)
    for f in regular_exit_filters:
        if not f.supports_vectorized:
            return VectorizedResult(0.0, 0.0, 0.0, 0.0, 0.0, 0)
        exit_mask = exit_mask | f.vectorized_exit(df)

    # Fill NaN with False
    entry_signals = entry_mask.fillna(False).values.astype(bool)
    exit_signals = exit_mask.fillna(False).values.astype(bool)

    # Extract numpy arrays
    opens = df["open"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)

    # ATR values for trailing stop
    atr_values: np.ndarray | None = None
    atr_multiplier = 0.0
    if atr_filter is not None:
        atr_col = f"atr_{atr_filter.period}"
        if atr_col in df.columns:
            atr_values = df[atr_col].values.astype(np.float64)
            atr_multiplier = atr_filter.multiplier

    # --- Extract trades ---
    trades, final_balance = _extract_trades(
        entry_signals=entry_signals,
        exit_signals=exit_signals,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        initial_balance=initial_balance,
        fee_rate=fee_rate,
        slippage_pct=slippage_pct,
        stop_loss_pct=stop_loss_pct,
        max_position_pct=max_position_pct,
        atr_values=atr_values,
        atr_multiplier=atr_multiplier,
    )

    # --- Compute metrics ---
    return _compute_metrics(
        trades=trades,
        initial_balance=initial_balance,
        final_balance=final_balance,
        closes=closes,
        index=df.index,
        timeframe=timeframe,
        fee_rate=fee_rate,
    )


def _extract_trades(
    entry_signals: np.ndarray,
    exit_signals: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    initial_balance: float,
    fee_rate: float,
    slippage_pct: float,
    stop_loss_pct: float,
    max_position_pct: float,
    atr_values: np.ndarray | None,
    atr_multiplier: float,
) -> tuple[list[tuple[int, int, float, float, float, float, float]], float]:
    """Extract trades from signal arrays in a single O(N) pass.

    Returns:
        trades: list of (entry_idx, exit_idx, entry_price, exit_price, qty, entry_fee, pnl)
        final_balance: cash after all trades
    """
    n = len(entry_signals)
    trades: list[tuple[int, int, float, float, float, float, float]] = []
    cash = initial_balance
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    quantity = 0.0
    entry_fee = 0.0
    highest_since_entry = 0.0

    for i in range(1, n):  # Start at 1 — need at least 1 prior candle
        if in_position:
            # Skip stop/exit checks on the entry candle itself
            if i == entry_idx:
                continue

            # 1. ATR trailing stop
            if atr_values is not None:
                if highs[i] > highest_since_entry:
                    highest_since_entry = highs[i]
                atr_val = atr_values[i]
                if not np.isnan(atr_val):
                    trailing_stop = highest_since_entry - atr_multiplier * atr_val
                    if lows[i] <= trailing_stop:
                        exit_price = trailing_stop * (1 - slippage_pct)
                        fee = exit_price * quantity * fee_rate
                        pnl = (exit_price - entry_price) * quantity - entry_fee - fee
                        cash += exit_price * quantity - fee
                        trades.append((entry_idx, i, entry_price, exit_price, quantity, entry_fee, pnl))
                        in_position = False
                        continue

            # 2. Fixed stop loss
            sl_price = entry_price * (1 - stop_loss_pct)
            if lows[i] <= sl_price:
                exit_price = sl_price * (1 - slippage_pct)
                fee = exit_price * quantity * fee_rate
                pnl = (exit_price - entry_price) * quantity - entry_fee - fee
                cash += exit_price * quantity - fee
                trades.append((entry_idx, i, entry_price, exit_price, quantity, entry_fee, pnl))
                in_position = False
                continue

            # 3. Exit signal → sell at next candle's open
            if exit_signals[i] and i + 1 < n:
                exit_price = opens[i + 1] * (1 - slippage_pct)
                fee = exit_price * quantity * fee_rate
                pnl = (exit_price - entry_price) * quantity - entry_fee - fee
                cash += exit_price * quantity - fee
                trades.append((entry_idx, i + 1, entry_price, exit_price, quantity, entry_fee, pnl))
                in_position = False
                continue

        else:
            # Entry signal → buy at next candle's open
            if entry_signals[i] and i + 1 < n:
                entry_price = opens[i + 1] * (1 + slippage_pct)
                if entry_price <= 0:
                    continue
                max_value = cash * max_position_pct
                quantity = max_value / entry_price
                entry_fee = entry_price * quantity * fee_rate
                cash -= entry_price * quantity + entry_fee
                entry_idx = i + 1
                highest_since_entry = highs[i + 1]
                in_position = True

    # Force close remaining position
    if in_position:
        exit_price = closes[-1]
        fee = exit_price * quantity * fee_rate
        pnl = (exit_price - entry_price) * quantity - entry_fee - fee
        cash += exit_price * quantity - fee
        trades.append((entry_idx, n - 1, entry_price, exit_price, quantity, entry_fee, pnl))

    return trades, cash


def _compute_metrics(
    trades: list[tuple[int, int, float, float, float, float, float]],
    initial_balance: float,
    final_balance: float,
    closes: np.ndarray,
    index: pd.DatetimeIndex,
    timeframe: str,
    fee_rate: float = 0.0005,
) -> VectorizedResult:
    """Compute screening metrics from trade list."""
    if not trades:
        return VectorizedResult(
            sharpe_ratio=0.0,
            total_return=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
        )

    # Build equity curve with mark-to-market during positions
    n = len(closes)
    equity = np.full(n, np.nan)
    equity[0] = initial_balance
    current_equity = initial_balance

    for entry_idx, exit_idx, entry_p, exit_p, qty, e_fee, pnl in trades:
        # Fill flat equity up to entry
        if entry_idx > 0:
            equity[entry_idx - 1] = current_equity
        # Mark-to-market during position (subtract both entry + estimated exit fee)
        for j in range(entry_idx, min(exit_idx + 1, n)):
            est_exit_fee = closes[j] * qty * fee_rate
            unrealized = (closes[j] - entry_p) * qty - e_fee - est_exit_fee
            equity[j] = current_equity + unrealized
        current_equity += pnl
        equity[exit_idx] = current_equity

    # Forward fill gaps
    for i in range(1, n):
        if np.isnan(equity[i]):
            equity[i] = equity[i - 1]

    equity_series = pd.Series(equity, index=index)

    # Sharpe ratio
    returns = equity_series.pct_change().dropna()
    std = returns.std(ddof=0)
    ann_factor = sqrt(PERIODS_PER_YEAR.get(timeframe, 8766))
    sharpe = float(returns.mean() / std * ann_factor) if len(returns) > 1 and std > 0 else 0.0

    # Max drawdown
    peak = equity_series.expanding().max()
    drawdown = (peak - equity_series) / peak
    max_dd = float(drawdown.max())

    # Win rate, profit factor
    pnls = [t[6] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(trades)
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    total_return = (final_balance - initial_balance) / initial_balance if initial_balance > 0 else 0.0

    return VectorizedResult(
        sharpe_ratio=sharpe,
        total_return=total_return,
        max_drawdown=max_dd,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_trades=len(trades),
    )
