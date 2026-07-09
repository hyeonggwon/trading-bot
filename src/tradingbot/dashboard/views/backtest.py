"""Backtest page: interactive single-run viewer + background job forms."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import streamlit as st

from tradingbot.dashboard import forms
from tradingbot.dashboard.views import common

if TYPE_CHECKING:
    import numpy as np

    from tradingbot.backtest.report import BacktestReport
    from tradingbot.strategy.base import Strategy


def render() -> None:
    st.subheader("Backtest")
    _render_interactive()
    st.divider()
    st.markdown("### Run as background job")
    st.caption(
        "Full CLI surface (--start/--end, holdout default, multi-symbol, "
        "--output) — results land in the job log on the Jobs page."
    )
    common.command_expanders(forms.PAGE_COMMANDS["Backtest"])


def _render_interactive() -> None:
    """Quick single-symbol backtest with charts, rendered in-process."""
    st.caption(
        "Quick look over the FULL data range with fixed fee/slippage "
        "(0.05%/0.1%) — the CLI and the job form below default to the "
        "holdout-only window (last 20%), so numbers will differ."
    )
    # Strategy selection — from the registry so new strategies show up
    # automatically. lgbm needs saved models + per-symbol thresholds; the
    # proper path for it is `tradingbot ml-backtest`.
    names = sorted(n for n in _get_strategy_map() if n != "lgbm")
    col1, col2 = st.columns(2)
    with col1:
        strategy = st.selectbox(
            "Strategy",
            names,
            index=names.index("sma_cross") if "sma_cross" in names else 0,
        )
    with col2:
        symbol = st.text_input("Symbol", value="BTC/KRW")

    col3, col4, col5 = st.columns(3)
    with col3:
        timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d", "15m", "5m"])
    with col4:
        balance = st.number_input("Initial Balance (KRW)", value=1_000_000, step=100_000)
    with col5:
        data_dir = st.text_input("Data Directory", value="data")

    if st.button("Run Backtest", type="primary"):
        strategy_map = _get_strategy_map()
        strategy_obj = strategy_map[strategy]()
        run_and_display_backtest(strategy_obj, symbol, timeframe, balance, data_dir)


def run_and_display_backtest(
    strategy: Strategy,
    symbol: str,
    timeframe: str,
    balance: float,
    data_dir: str,
) -> None:
    """Execute a backtest for a ready strategy instance and render results.

    Shared with the Combine page, which passes a CombinedStrategy.
    """
    from tradingbot.backtest.engine import BacktestEngine
    from tradingbot.config import AppConfig, BacktestConfig, RiskConfig, TradingConfig
    from tradingbot.data.storage import load_candles

    try:
        df = load_candles(symbol, timeframe, Path(data_dir))
    except FileNotFoundError:
        st.error(
            f"No data for {symbol} {timeframe}. Run: `tradingbot download "
            f"--symbol {symbol} --timeframe {timeframe} --since 2024-01-01`"
        )
        return

    config = AppConfig(
        trading=TradingConfig(symbols=[symbol], timeframe=timeframe, initial_balance=balance),
        risk=RiskConfig(),
        backtest=BacktestConfig(fee_rate=0.0005, slippage_pct=0.001),
    )

    strategy.symbols = [symbol]
    strategy.timeframe = timeframe

    with st.spinner("Running backtest..."):
        engine = BacktestEngine(strategy=strategy, config=config)
        report = engine.run({symbol: df})

    # Summary metrics
    st.divider()
    cols = st.columns(6)
    cols[0].metric("Total Return", f"{report.total_return:.2%}")
    cols[1].metric("Sharpe Ratio", f"{report.sharpe_ratio:.2f}")
    cols[2].metric("Max Drawdown", f"{report.max_drawdown:.2%}")
    cols[3].metric("Win Rate", f"{report.win_rate:.1%}")
    cols[4].metric("Profit Factor", f"{report.profit_factor:.2f}")
    cols[5].metric("Total Trades", str(report.total_trades))

    # Equity curve with drawdown
    _render_backtest_equity(report)

    # Trade list
    _render_trade_list(report)


def _render_backtest_equity(report: BacktestReport) -> None:
    """Render equity curve with drawdown overlay and trade markers."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if report.equity_curve.empty:
        return

    equity = report.equity_curve
    peak = equity.expanding().max()
    drawdown = (peak - equity) / peak

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
    )

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            mode="lines",
            name="Equity",
            line=dict(color="#2196F3", width=2),
        ),
        row=1,
        col=1,
    )

    # Peak line
    fig.add_trace(
        go.Scatter(
            x=peak.index,
            y=peak.values,
            mode="lines",
            name="Peak",
            line=dict(color="#90CAF9", width=1, dash="dot"),
        ),
        row=1,
        col=1,
    )

    # Trade markers at equity curve values
    import pandas as pd

    for trade in report.trades:
        if trade.entry_order.filled_at:
            # Look up equity at entry time
            equity_at_entry = equity.asof(trade.entry_order.filled_at)
            if pd.isna(equity_at_entry):
                continue
            color = "#4CAF50" if trade.is_win else "#F44336"
            fig.add_trace(
                go.Scatter(
                    x=[trade.entry_order.filled_at],
                    y=[equity_at_entry],
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=10, color=color),
                    name=f"{'Win' if trade.is_win else 'Loss'} ({trade.symbol})",
                    showlegend=False,
                    hovertemplate=(
                        f"{trade.symbol}<br>PnL: {trade.pnl:,.0f} KRW<br>%{{x}}<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            # cast: float64 equity series — .values is numpy-backed
            y=cast("np.ndarray", drawdown.values) * -100,
            mode="lines",
            name="Drawdown %",
            line=dict(color="#F44336", width=1),
            fill="tozeroy",
            fillcolor="rgba(244, 67, 54, 0.2)",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="Equity (KRW)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def _render_trade_list(report: BacktestReport) -> None:
    """Render trade history table."""
    if not report.trades:
        return

    import pandas as pd

    st.subheader(f"Trade History ({report.total_trades} trades)")

    rows = []
    for i, trade in enumerate(report.trades, 1):
        rows.append(
            {
                "#": i,
                "Symbol": trade.symbol,
                "Entry Price": f"{trade.entry_order.filled_price:,.0f}"
                if trade.entry_order.filled_price
                else "N/A",
                "Exit Price": f"{trade.exit_order.filled_price:,.0f}"
                if trade.exit_order.filled_price
                else "N/A",
                "PnL": f"{trade.pnl:,.0f}",
                "PnL %": f"{trade.pnl_pct:.2%}",
                "Duration": f"{trade.duration:.1f}h" if trade.duration else "N/A",
                "Result": "Win" if trade.is_win else "Loss",
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _get_strategy_map() -> dict[str, type[Strategy]]:
    from tradingbot.strategy.registry import get_strategy_map

    return get_strategy_map()
