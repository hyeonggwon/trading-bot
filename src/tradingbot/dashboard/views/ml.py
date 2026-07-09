"""ML page: training / walk-forward / tuning commands as background jobs."""

from __future__ import annotations

import importlib.util

import streamlit as st

from tradingbot.dashboard import forms
from tradingbot.dashboard.views import common


def render() -> None:
    st.subheader("Machine Learning")
    if importlib.util.find_spec("lightgbm") is None:
        st.warning('ML extra is not installed — these jobs will fail. `pip install -e ".[ml]"`')
    st.caption(
        "All ML commands run as background jobs (training and tuning take "
        "minutes to hours) — watch progress on the Jobs page."
    )
    common.command_expanders(forms.PAGE_COMMANDS["ML"])
