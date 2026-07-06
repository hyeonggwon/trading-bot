"""Streamlit dashboard for live monitoring and backtest visualization.

Run with: streamlit run src/tradingbot/dashboard/app.py
Or via CLI: tradingbot dashboard
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta, timezone
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
    st.title("Trading Bot Dashboard")

    # Sidebar: mode selection
    mode = st.sidebar.radio("Mode", ["Live Monitor", "Backtest Viewer", "Models"])

    if mode == "Live Monitor":
        _render_live_monitor()
    elif mode == "Backtest Viewer":
        _render_backtest_viewer()
    else:
        _render_models()


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
    _render_entry_controls(state_file)
    _live_data_fragment(state_file)


def _render_entry_controls(state_file: Path) -> None:
    """Operator kill-switch: pause/resume NEW entries via the control file.

    The engine polls the control file every tick and keeps managing existing
    positions (stops, take profits, exits, safety rails) while paused.
    """
    from tradingbot.live.control import control_path_for, read_pause, set_pause

    control_path = control_path_for(state_file)
    paused = read_pause(control_path)

    st.sidebar.divider()
    if paused:
        st.sidebar.error("⏸ Entries PAUSED")
        if st.sidebar.button("▶ Resume entries", use_container_width=True):
            set_pause(control_path, False)
            st.rerun()
    else:
        st.sidebar.success("▶ Entries active")
        if st.sidebar.button("⏸ Pause entries", use_container_width=True):
            set_pause(control_path, True)
            st.rerun()
    st.sidebar.caption(
        "Pause blocks new entries only — open positions keep their stops, "
        "take profits and safety rails."
    )


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

    peak = data.get("peak_equity") or 0
    # Clamped: after an external deposit raw equity can sit above the
    # ledger-tracked peak, which would read as a negative drawdown.
    drawdown = max(0.0, (peak - latest_equity) / peak) if peak > 0 else 0
    daily_pnl = data.get("daily_pnl") or 0
    cum_pnl = data.get("cum_realized_pnl") or 0

    cols = st.columns(4)
    cols[0].metric("Equity", f"{latest_equity:,.0f} KRW")
    cols[1].metric("Return", f"{total_return:.2%}")
    cols[2].metric("Drawdown vs Peak", f"{drawdown:.2%}")
    cols[3].metric("Daily PnL (realized)", f"{daily_pnl:+,.0f} KRW")

    cols = st.columns(4)
    cols[0].metric("Cum Realized PnL", f"{cum_pnl:+,.0f} KRW")
    cols[1].metric("Peak Equity", f"{peak:,.0f} KRW" if peak else "N/A")
    cols[2].metric("Open Positions", str(len(positions)))
    cols[3].metric("Last Update", _format_timestamp(saved_at))
    st.caption(
        "The breaker and daily-loss limit run on the engine's transfer-immune "
        "ledger, which matches these figures until an external transfer occurs."
    )


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
        rows.append(
            {
                "Symbol": symbol,
                "Side": pos.get("side", ""),
                "Size": f"{pos.get('size', 0):.8f}",
                "Entry Price": f"{pos.get('entry_price', 0):,.0f}",
                "Stop Loss": f"{pos.get('stop_loss', 0):,.0f}" if pos.get("stop_loss") else "N/A",
                "Stop %": (
                    f"{pos['stop_loss'] / pos['entry_price'] - 1:+.1%}"
                    if pos.get("stop_loss") and pos.get("entry_price")
                    else "N/A"
                ),
                "Entry Time": _format_timestamp(pos.get("entry_time", "")),
            }
        )

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
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=equities,
            mode="lines",
            name="Equity",
            line=dict(color="#2196F3", width=2),
        )
    )
    fig.update_layout(
        yaxis_title="Equity (KRW)",
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Backtest Viewer ──────────────────────────────────────────────────


def _render_backtest_viewer() -> None:
    """Backtest result visualization."""
    st.subheader("Backtest")

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
            y=drawdown.values * -100,
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


def _render_trade_list(report) -> None:
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


# ── Model Catalog ────────────────────────────────────────────────────


def _render_models() -> None:
    """Saved LightGBM model catalog from models/*_meta.json."""
    from tradingbot.ml.trainer import LGBMTrainer

    st.subheader("Model Catalog")
    model_dir = st.sidebar.text_input("Model directory", value="models")

    entries = LGBMTrainer.load_catalog(Path(model_dir))
    if not entries:
        st.info("No saved models found. Train them first: `tradingbot ml-train-all`")
        return

    import pandas as pd

    df = pd.DataFrame(entries)
    # Operator-relevant columns first; whatever else the meta grows stays behind.
    front = [
        c
        for c in (
            "symbol",
            "timeframe",
            "holdout_auc",
            "entry_threshold",
            "exit_threshold",
            "has_calibrator",
            "n_features",
            "trained_at",
        )
        if c in df.columns
    ]
    df = df[front + [c for c in df.columns if c not in front]]
    num_cols = df.select_dtypes("number").columns
    df[num_cols] = df[num_cols].round(3)

    st.caption(f"{len(entries)} models in `{model_dir}/`")
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
            dt = dt.replace(tzinfo=UTC)
        dt_kst = dt.astimezone(KST)
        # Year dropped so the value fits st.metric without clipping.
        return dt_kst.strftime("%m-%d %H:%M KST")
    except (ValueError, TypeError):
        return str(ts)


if __name__ == "__main__":
    main()
