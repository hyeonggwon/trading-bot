"""Data page: candle/external downloads (jobs) + data-list/symbols (inline)."""

from __future__ import annotations

import streamlit as st

from tradingbot.dashboard import forms
from tradingbot.dashboard.views import common


def render() -> None:
    st.subheader("Data")
    st.caption(
        "Candles land in `data/{SYMBOL}_{QUOTE}/{timeframe}.parquet`. "
        "Long --since ranges download a lot — keep ranges tight."
    )
    common.command_expanders(forms.PAGE_COMMANDS["Data"])
