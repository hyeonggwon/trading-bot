"""Combine page: filter-combination strategies (interactive + jobs)."""

from __future__ import annotations

import streamlit as st

from tradingbot.dashboard import forms
from tradingbot.dashboard.views import backtest, common


def render() -> None:
    st.subheader("Combine — filter combination strategies")
    _render_interactive()
    st.divider()
    st.markdown("### Run as background job")
    common.command_expanders(forms.PAGE_COMMANDS["Combine"])


def _render_interactive() -> None:
    """Quick combined-strategy backtest with charts, rendered in-process."""
    from tradingbot.cli.combine import COMBINE_TEMPLATES

    st.caption(
        "Quick look over the FULL data range with fixed fee/slippage "
        "(0.05%/0.1%) — the CLI and the job form below default to the "
        "holdout-only window (last 20%), so numbers will differ."
    )

    labels = ["(custom)"] + [str(t["label"]) for t in COMBINE_TEMPLATES]
    choice = st.selectbox(
        "Template",
        labels,
        help="Prefills entry/exit from the built-in combine templates",
    )
    initial_entry = "trend_up:4 + rsi_oversold:30"
    initial_exit = "rsi_overbought:70"
    if choice != "(custom)":
        tmpl = next(t for t in COMBINE_TEMPLATES if t["label"] == choice)
        initial_entry = str(tmpl["entry"])
        initial_exit = str(tmpl["exit"])

    # Template choice is baked into the widget keys so switching templates
    # remounts the inputs with the new prefill (Streamlit keeps keyed state).
    col1, col2 = st.columns(2)
    with col1:
        entry = st.text_input("Entry filters", value=initial_entry, key=f"combine_entry_{choice}")
    with col2:
        exit_ = st.text_input("Exit filters", value=initial_exit, key=f"combine_exit_{choice}")

    col3, col4, col5, col6 = st.columns(4)
    with col3:
        symbol = st.text_input("Symbol", value="BTC/KRW")
    with col4:
        timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d", "15m", "5m"])
    with col5:
        balance = st.number_input("Initial Balance (KRW)", value=1_000_000, step=100_000)
    with col6:
        data_dir = st.text_input("Data Directory", value="data")

    if st.button("Run Combine Backtest", type="primary"):
        from tradingbot.cli.combine import _build_combined_strategy

        try:
            strategy = _build_combined_strategy(entry, exit_, symbol, timeframe)
        except Exception as e:  # noqa: BLE001 — parse errors surface as-is in the GUI
            st.error(f"Invalid filter spec: {e}")
            return
        backtest.run_and_display_backtest(strategy, symbol, timeframe, balance, data_dir)
