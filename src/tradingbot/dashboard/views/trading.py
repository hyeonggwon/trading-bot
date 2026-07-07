"""Trading page: paper/live engines as background jobs + status/balance."""

from __future__ import annotations

import streamlit as st

from tradingbot.dashboard import forms, jobs
from tradingbot.dashboard.views import common


def render() -> None:
    st.subheader("Trading")
    _render_running_engines()

    cmd_map = forms.get_cli_commands()
    with st.expander(f"**paper** — {str(cmd_map['paper'].help or '').splitlines()[0]}"):
        _render_engine_form("paper")
    with st.expander(f"**live** — {str(cmd_map['live'].help or '').splitlines()[0]}"):
        st.warning(
            "⚠ LIVE trading places REAL orders on your Upbit account. "
            "Check --max-order and --daily-loss-limit before starting."
        )
        _render_engine_form("live", confirm_text="LIVE")

    st.divider()
    common.command_expanders(["status", "balance"])


def _render_engine_form(command: str, *, confirm_text: str | None = None) -> None:
    """Engine launch form with the duplicate-state-file guard."""
    result = forms.render_command_form(
        command, confirm_text=confirm_text, submit_label=f"Start {command}"
    )
    if result is None:
        return
    state_file = str(result.values.get("state_file") or "").strip() or "state.json"
    duplicate = jobs.running_job_for_state(state_file)
    if duplicate is not None:
        st.error(
            f"A `{duplicate.command}` job ({duplicate.job_id}) is already running "
            f"on state file `{state_file}` — two engines on one state file would "
            "corrupt it. Stop that job on the Jobs page first."
        )
        return
    common.submit_job(command, result.args, state_file=state_file)


def _render_running_engines() -> None:
    running = [j for j in jobs.list_jobs() if j.is_running and j.command in ("paper", "live")]
    if not running:
        return
    st.info(
        "Running engines: "
        + ", ".join(f"`{j.command}` ({j.job_id}, state={j.state_file})" for j in running)
        + " — stop them on the Jobs page."
    )
