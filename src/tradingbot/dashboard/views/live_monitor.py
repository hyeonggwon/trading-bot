"""Live Monitor page: real-time state.json view + entry pause/resume."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from tradingbot.dashboard.views.common import format_timestamp, get_default_state_file


def render() -> None:
    """Real-time monitoring of live/paper trading from state.json."""
    default_path = get_default_state_file()
    state_path = st.sidebar.text_input("State file", value=default_path)

    state_file = Path(state_path)
    _render_entry_controls(state_file)
    _live_data_fragment(state_file)


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


def _render_header_metrics(data: dict[str, Any]) -> None:
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
    cols[3].metric("Last Update", format_timestamp(saved_at))
    st.caption(
        "The breaker and daily-loss limit run on the engine's transfer-immune "
        "ledger, which matches these figures until an external transfer occurs."
    )


def _render_positions(data: dict[str, Any]) -> None:
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
                "Entry Time": format_timestamp(pos.get("entry_time", "")),
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_equity_chart(data: dict[str, Any]) -> None:
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
