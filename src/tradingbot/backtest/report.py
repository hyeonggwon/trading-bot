"""Backtest performance reporting.

Computes standard trading metrics: Sharpe, Sortino, max drawdown,
win rate, profit factor, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradingbot.core.models import Trade

# Periods per year by timeframe, for annualization
PERIODS_PER_YEAR: dict[str, float] = {
    "1m": 525_960,
    "3m": 175_320,
    "5m": 105_192,
    "15m": 35_064,
    "30m": 17_532,
    "1h": 8_766,
    "4h": 2_191.5,
    "1d": 365.25,
    "1w": 52.18,
}


@dataclass
class BacktestReport:
    """Performance report for a completed backtest."""

    trades: list[Trade]
    equity_curve: pd.Series  # DatetimeIndex → equity value
    initial_balance: float
    final_balance: float
    timeframe: str = "1h"

    @property
    def _annualization_factor(self) -> float:
        return np.sqrt(PERIODS_PER_YEAR.get(self.timeframe, 8766))

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.trades if t.is_win)

    @property
    def losing_trades(self) -> int:
        return self.total_trades - self.winning_trades

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def total_return(self) -> float:
        if self.initial_balance == 0:
            return 0.0
        return (self.final_balance - self.initial_balance) / self.initial_balance

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def avg_win(self) -> float:
        wins = [t.pnl for t in self.trades if t.is_win]
        return np.mean(wins) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [t.pnl for t in self.trades if not t.is_win]
        return np.mean(losses) if losses else 0.0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl for t in self.trades if t.is_win)
        gross_loss = abs(sum(t.pnl for t in self.trades if not t.is_win))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown as a positive fraction (e.g., 0.15 = 15%)."""
        if self.equity_curve.empty:
            return 0.0
        peak = self.equity_curve.expanding().max()
        drawdown = (peak - self.equity_curve) / peak
        return float(drawdown.max())

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio, scaled by timeframe.
        Uses population std (ddof=0) consistent with Sortino calculation."""
        returns = self.equity_curve.pct_change().dropna()
        std = returns.std(ddof=0)
        if len(returns) < 2 or std == 0:
            return 0.0
        return float(returns.mean() / std * self._annualization_factor)

    @property
    def sortino_ratio(self) -> float:
        """Annualized Sortino ratio using standard downside deviation.

        Downside deviation is computed over the full series, treating
        positive returns as zero (not excluded).
        """
        returns = self.equity_curve.pct_change().dropna()
        if len(returns) < 2:
            return 0.0
        # Standard Sortino: downside deviation over full series
        downside_returns = returns.copy()
        downside_returns[downside_returns > 0] = 0.0
        downside_std = np.sqrt((downside_returns**2).mean())
        if downside_std == 0:
            return float("inf") if returns.mean() > 0 else 0.0
        return float(returns.mean() / downside_std * self._annualization_factor)

    @property
    def avg_trade_duration_hours(self) -> float:
        durations = [t.duration for t in self.trades if t.duration is not None]
        return np.mean(durations) if durations else 0.0

    def summary(self) -> dict[str, str]:
        """Generate a summary dictionary for display."""
        return {
            "Total Trades": str(self.total_trades),
            "Win Rate": f"{self.win_rate:.1%}",
            "Total Return": f"{self.total_return:.2%}",
            "Total PnL": f"{self.total_pnl:,.0f} KRW",
            "Profit Factor": f"{self.profit_factor:.2f}",
            "Sharpe Ratio": f"{self.sharpe_ratio:.2f}",
            "Sortino Ratio": f"{self.sortino_ratio:.2f}",
            "Max Drawdown": f"{self.max_drawdown:.2%}",
            "Avg Win": f"{self.avg_win:,.0f} KRW",
            "Avg Loss": f"{self.avg_loss:,.0f} KRW",
            "Avg Duration": f"{self.avg_trade_duration_hours:.1f} hours",
            "Initial Balance": f"{self.initial_balance:,.0f} KRW",
            "Final Balance": f"{self.final_balance:,.0f} KRW",
        }

    def print_summary(self) -> None:
        """Print formatted summary to console."""
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Backtest Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        for key, value in self.summary().items():
            table.add_row(key, value)

        console.print(table)

        if self.trades:
            trades_table = Table(title=f"Trade Log ({self.total_trades} trades)")
            trades_table.add_column("#", justify="right")
            trades_table.add_column("Symbol")
            trades_table.add_column("Entry Price", justify="right")
            trades_table.add_column("Exit Price", justify="right")
            trades_table.add_column("PnL", justify="right")
            trades_table.add_column("PnL %", justify="right")
            trades_table.add_column("Duration", justify="right")

            for i, trade in enumerate(self.trades, 1):
                pnl_style = "green" if trade.is_win else "red"
                duration = f"{trade.duration:.1f}h" if trade.duration else "N/A"
                trades_table.add_row(
                    str(i),
                    trade.symbol,
                    f"{trade.entry_order.filled_price:,.0f}",
                    f"{trade.exit_order.filled_price:,.0f}",
                    f"[{pnl_style}]{trade.pnl:,.0f}[/{pnl_style}]",
                    f"[{pnl_style}]{trade.pnl_pct:.2%}[/{pnl_style}]",
                    duration,
                )

            console.print(trades_table)
