"""Shared helpers for dashboard views (timestamps, sync runs, job submits)."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone

import streamlit as st

from tradingbot.dashboard import forms, jobs

# KST timezone for Korean users
KST = timezone(timedelta(hours=9))


def format_timestamp(ts: str) -> str:
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


def get_default_state_file() -> str:
    """Read --state-file from CLI args if provided."""
    for arg in sys.argv:
        if arg.startswith("--state-file="):
            return arg.split("=", 1)[1]
    return "state.json"


def run_sync_command(command: str, args: list[str]) -> None:
    """Run a fast CLI command inline and show its console output."""
    env = {**os.environ, "COLUMNS": "120"}
    try:
        with st.spinner(f"Running {command}..."):
            proc = subprocess.run(
                [*jobs.CLI_ARGV, command, *args],
                capture_output=True,
                text=True,
                env=env,
                timeout=180,
            )
    except subprocess.TimeoutExpired:
        st.error(f"{command} timed out after 180s — check network/exchange availability.")
        return
    output = proc.stdout.strip()
    if output:
        st.code(output, language="text")
    if proc.returncode != 0:
        st.error(f"{command} exited with code {proc.returncode}")
        if proc.stderr.strip():
            st.code(proc.stderr.strip(), language="text")
    elif not output:
        st.info("(no output)")


def submit_job(command: str, args: list[str], *, state_file: str | None = None) -> None:
    """Spawn a background job and point the user at the Jobs page."""
    job = jobs.start_job(command, args, state_file=state_file)
    st.success(f"Started background job `{job.job_id}` — track it on the **Jobs** page.")
    st.code(shlex.join(["tradingbot", command, *args]), language="bash")


def command_expanders(commands: list[str], *, key_prefix: str = "") -> None:
    """One expander + auto-form per command (sync commands run inline)."""
    cmd_map = forms.get_cli_commands()
    for name in commands:
        help_text = str(cmd_map[name].help or "")
        first_line = help_text.splitlines()[0] if help_text else ""
        label = f"**{name}** — {first_line}" if first_line else f"**{name}**"
        with st.expander(label):
            result = forms.render_command_form(name, key=f"{key_prefix}form_{name}")
            if result is None:
                continue
            if name in forms.SYNC_COMMANDS:
                run_sync_command(name, result.args)
            else:
                submit_job(name, result.args)
