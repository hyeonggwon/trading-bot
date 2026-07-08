"""Streamlit dashboard: live monitoring + full CLI parity GUI.

Every CLI command is reachable from a page (parity is CI-enforced by
tests/test_dashboard_forms.py); long commands run as background jobs on
the Jobs page.

Run with: streamlit run src/tradingbot/dashboard/app.py
Or via CLI: tradingbot dashboard
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📊",
    layout="wide",
)

MODES = [
    "Live Monitor",
    "Trading",
    "Backtest",
    "Combine",
    "ML",
    "Pipeline",
    "Data",
    "Models",
    "Jobs",
]


def main() -> None:
    st.title("Trading Bot Dashboard")

    mode = st.sidebar.radio("Mode", MODES)

    from tradingbot.dashboard.views import (
        backtest,
        combine,
        data,
        jobs_page,
        live_monitor,
        ml,
        models,
        pipeline,
        trading,
    )

    renderers = {
        "Live Monitor": live_monitor.render,
        "Trading": trading.render,
        "Backtest": backtest.render,
        "Combine": combine.render,
        "ML": ml.render,
        "Pipeline": pipeline.render,
        "Data": data.render,
        "Models": models.render,
        "Jobs": jobs_page.render,
    }
    renderers[mode]()


if __name__ == "__main__":
    main()
