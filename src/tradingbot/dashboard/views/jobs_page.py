"""Jobs page: background job list, live log tail, graceful stop."""

from __future__ import annotations

import streamlit as st

from tradingbot.dashboard import jobs
from tradingbot.dashboard.views.common import format_timestamp


def render() -> None:
    st.subheader("Jobs")
    st.caption(
        "Jobs are detached CLI subprocesses — they keep running if you close "
        "the browser. Stop sends SIGINT (Ctrl+C equivalent), so engines save "
        "state and workers shut down cleanly."
    )
    _jobs_fragment()


@st.fragment(run_every=5)
def _jobs_fragment() -> None:
    all_jobs = jobs.list_jobs()
    if not all_jobs:
        st.info("No jobs yet. Start one from the Trading / Backtest / Combine / ML / Data pages.")
        return

    import pandas as pd

    rows = [
        {
            "Job": j.job_id,
            "Command": j.command,
            "Status": j.status,
            "Started": format_timestamp(j.started_at),
            "RC": "" if j.returncode is None else str(j.returncode),
            "Args": " ".join(j.args),
        }
        for j in all_jobs
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    job_ids = [j.job_id for j in all_jobs]
    selected = st.selectbox("Job detail", job_ids, key="jobs_detail_select")
    job = next(j for j in all_jobs if j.job_id == selected)

    if job.is_running and st.button("⏹ Stop job (SIGINT)", key="jobs_stop"):
        jobs.stop_job(job)
        st.rerun()

    st.code(jobs.read_log_tail(job) or "(no output yet)", language="text")
