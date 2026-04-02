"""Streamlit dashboard for live monitoring and backtest visualization.

Run with: streamlit run src/tradingbot/dashboard/app.py
Or via CLI: tradingbot dashboard
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import streamlit as st

# KST timezone for Korean users
KST = timezone(timedelta(hours=9))

st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📊",
    layout="wide",
)


def _get_default_state_file() -> str:
    """Read --state-file from CLI args if provided."""
    for arg in sys.argv:
        if arg.startswith("--state-file="):
            return arg.split("=", 1)[1]
    return "state.json"


def main() -> None:
    st.title("📊 Trading Bot Dashboard")

    # Sidebar: mode selection
    mode = st.sidebar.radio("Mode", ["Live Monitor", "Backtest Viewer"])

    if mode == "Live Monitor":
        _render_live_monitor()
    else:
        _render_backtest_viewer()


# ── Live Monitor ──────────────────────────────────────────────────────


@st.fragment(run_every=10)
def _live_data_fragment(state_file: Path) -> None:
    """Auto-refreshing fragment for live data (does NOT block the whole page)."""
    if not state_file.exists():
        st.warning("State file not found. Start paper/live trading first.")
        st.code(
            "tradingbot paper --strategy sma_cross --symbol BTC/KRW",
            language="bash",
        )
        return

    try:
        data = json.loads(state_file.read_text())
    except (json.JSONDecodeError, OSError) as e:
        st.error(f"Cannot read state file: {e}")
        return

    # Header metrics
    _render_header_metrics(data)

    # Two columns: positions + equity chart
    col1, col2 = st.columns([1, 2])

    with col1:
        _render_positions(data)

    with col2:
        _render_equity_chart(data)


def _render_live_monitor() -> None:
    """Real-time monitoring of live/paper trading from state.json."""
    default_path = _get_default_state_file()
    state_path = st.sidebar.text_input("State file", value=default_path)

    state_file = Path(state_path)
    _live_data_fragment(state_file)


def _render_header_metrics(data: dict) -> None:
    """Show key metrics as big numbers."""
    equity_history = data.get("equity_history", [])
    positions = data.get("positions", {})
    saved_at = data.get("saved_at", "N/A")

    if equity_history:
        latest_equity = equity_history[-1].get("equity", 0)
        first_equity = equity_history[0].get("equity", latest_equity)
        total_return = (latest_equity - first_equity) / first_equity if first_equity else 0
    else:
        latest_equity = 0
        total_return = 0

    cols = st.columns(4)
    cols[0].metric("Equity", f"{latest_equity:,.0f} KRW")
    cols[1].metric("Return", f"{total_return:.2%}")
    cols[2].metric("Open Positions", str(len(positions)))
    cols[3].metric("Last Update", _format_timestamp(saved_at))


def _render_positions(data: dict) -> None:
    """Show open positions table."""
    st.subheader("Open Positions")
    positions = data.get("positions", {})

    if not positions:
        st.info("No open positions")
        return

    import pandas as pd

    rows = []
    for symbol, pos in positions.items():
        rows.append({
            "Symbol": symbol,
            "Side": pos.get("side", ""),
            "Size": f"{pos.get('size', 0):.8f}",
            "Entry Price": f"{pos.get('entry_price', 0):,.0f}",
            "Stop Loss": f"{pos.get('stop_loss', 0):,.0f}" if pos.get("stop_loss") else "N/A",
            "Entry Time": _format_timestamp(pos.get("entry_time", "")),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_equity_chart(data: dict) -> None:
    """Show equity curve chart."""
    st.subheader("Equity Curve")
    equity_history = data.get("equity_history", [])

    if len(equity_history) < 2:
        st.info("Not enough equity data yet")
        return

    import plotly.graph_objects as go

    timestamps = [e["timestamp"] for e in equity_history]
    equities = [e["equity"] for e in equity_history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=equities,
        mode="lines",
        name="Equity",
        line=dict(color="#2196F3", width=2),
        fill="tozeroy",
        fillcolor="rgba(33, 150, 243, 0.1)",
    ))
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Equity (KRW)",
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Backtest Viewer ──────────────────────────────────────────────────


def _render_backtest_viewer() -> None:
    """Backtest result visualization."""
    st.subheader("Run a backtest to visualize results")

    # Strategy selection
    col1, col2 = st.columns(2)
    with col1:
        strategy = st.selectbox(
            "Strategy",
            ["sma_cross", "rsi_mean_reversion", "macd_momentum", "bollinger_breakout"],
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
        _run_and_display_backtest(strategy, symbol, timeframe, balance, data_dir)


def _run_and_display_backtest(
    strategy_name: str,
    symbol: str,
    timeframe: str,
    balance: float,
    data_dir: str,
) -> None:
    """Execute backtest and render results."""
    from tradingbot.backtest.engine import BacktestEngine
    from tradingbot.config import AppConfig, BacktestConfig, RiskConfig, TradingConfig
    from tradingbot.data.storage import load_candles

    # Load strategy class
    strategy_map = _get_strategy_map()
    if strategy_name not in strategy_map:
        st.error(f"Unknown strategy: {strategy_name}")
        return

    # Load data
    try:
        df = load_candles(symbol, timeframe, Path(data_dir))
    except FileNotFoundError:
        st.error(f"No data for {symbol} {timeframe}. Run: `tradingbot download --symbol {symbol} --timeframe {timeframe} --since 2024-01-01`")
        return

    config = AppConfig(
        trading=TradingConfig(symbols=[symbol], timeframe=timeframe, initial_balance=balance),
        risk=RiskConfig(),
        backtest=BacktestConfig(fee_rate=0.0005, slippage_pct=0.001),
    )

    strategy_cls = strategy_map[strategy_name]
    strategy = strategy_cls()
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


def _render_backtest_equity(report) -> None:
    """Render equity curve with drawdown overlay and trade markers."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if report.equity_curve.empty:
        return

    equity = report.equity_curve
    peak = equity.expanding().max()
    drawdown = (peak - equity) / peak

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
    )

    # Equity curve
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        mode="lines", name="Equity",
        line=dict(color="#2196F3", width=2),
    ), row=1, col=1)

    # Peak line
    fig.add_trace(go.Scatter(
        x=peak.index, y=peak.values,
        mode="lines", name="Peak",
        line=dict(color="#90CAF9", width=1, dash="dot"),
    ), row=1, col=1)

    # Trade markers at equity curve values
    import pandas as pd_check

    for trade in report.trades:
        if trade.entry_order.filled_at:
            # Look up equity at entry time
            equity_at_entry = equity.asof(trade.entry_order.filled_at)
            if pd_check.isna(equity_at_entry):
                continue
            color = "#4CAF50" if trade.is_win else "#F44336"
            fig.add_trace(go.Scatter(
                x=[trade.entry_order.filled_at],
                y=[equity_at_entry],
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color=color),
                name=f"{'Win' if trade.is_win else 'Loss'} ({trade.symbol})",
                showlegend=False,
                hovertemplate=f"{trade.symbol}<br>PnL: {trade.pnl:,.0f} KRW<br>%{{x}}<extra></extra>",
            ), row=1, col=1)

    # Drawdown
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values * -100,
        mode="lines", name="Drawdown %",
        line=dict(color="#F44336", width=1),
        fill="tozeroy",
        fillcolor="rgba(244, 67, 54, 0.2)",
    ), row=2, col=1)

    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="Equity (KRW)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def _render_trade_list(report) -> None:
    """Render trade history table."""
    if not report.trades:
        return

    import pandas as pd

    st.subheader(f"Trade History ({report.total_trades} trades)")

    rows = []
    for i, trade in enumerate(report.trades, 1):
        rows.append({
            "#": i,
            "Symbol": trade.symbol,
            "Entry Price": f"{trade.entry_order.filled_price:,.0f}" if trade.entry_order.filled_price else "N/A",
            "Exit Price": f"{trade.exit_order.filled_price:,.0f}" if trade.exit_order.filled_price else "N/A",
            "PnL": f"{trade.pnl:,.0f}",
            "PnL %": f"{trade.pnl_pct:.2%}",
            "Duration": f"{trade.duration:.1f}h" if trade.duration else "N/A",
            "Result": "✅" if trade.is_win else "❌",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ── Helpers ──────────────────────────────────────────────────────────


def _get_strategy_map() -> dict:
    from tradingbot.strategy.registry import get_strategy_map
    return get_strategy_map()


def _format_timestamp(ts: str) -> str:
    """Format ISO timestamp for display in KST (UTC+9)."""
    if not ts:
        return "N/A"
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt_kst = dt.astimezone(KST)
        return dt_kst.strftime("%Y-%m-%d %H:%M KST")
    except (ValueError, TypeError):
        return str(ts)


if __name__ == "__main__":
    main()
