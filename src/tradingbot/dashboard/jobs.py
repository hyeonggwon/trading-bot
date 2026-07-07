"""Background job manager for the dashboard GUI.

The GUI runs long CLI commands (scan, ml-train-all, paper, live, ...) as
detached subprocesses of the Streamlit server: ``python -m tradingbot.cli
<command> ...``. Job state lives on disk (``personal/gui_jobs/<job_id>/``)
so it survives Streamlit reruns and browser reloads — ``st.session_state``
does not.

Layout per job:
    job.json    — command, args, pid, started_at, stop flag (atomic writes)
    output.log  — merged stdout/stderr of the CLI process
    returncode  — written by a reaper thread when the process exits

Cancellation sends SIGINT (not SIGTERM) so the CLI takes its normal
KeyboardInterrupt path — the live/paper engines persist state every tick,
so an interrupted engine loses nothing.

This module is streamlit-free so it stays unit-testable without the
dashboard extra.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_JOBS_DIR = Path("personal/gui_jobs")

# Tests monkeypatch this to spawn a stub instead of the real CLI.
CLI_ARGV: list[str] = [sys.executable, "-m", "tradingbot.cli"]


@dataclass(frozen=True)
class Job:
    """A snapshot of one background job (status resolved at load time)."""

    job_id: str
    command: str
    args: list[str]
    pid: int
    started_at: str
    state_file: str | None
    stop_requested: bool
    status: str  # running | finished | failed | stopped | unknown
    returncode: int | None
    job_dir: Path

    @property
    def is_running(self) -> bool:
        return self.status == "running"


def start_job(
    command: str,
    args: list[str],
    *,
    jobs_dir: Path = DEFAULT_JOBS_DIR,
    state_file: str | None = None,
) -> Job:
    """Spawn ``python -m tradingbot.cli <command> <args>`` as a tracked job.

    ``state_file`` is recorded for paper/live so duplicate engines on the
    same state file can be refused (see :func:`running_job_for_state`).
    """
    started = datetime.now(UTC)
    job_id = f"{started.strftime('%Y%m%d_%H%M%S_%f')}_{command}"
    job_dir = jobs_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=False)

    log_path = job_dir / "output.log"
    log_path.touch(mode=0o600)  # job output may include balances — owner-only
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "COLUMNS": "160"}
    with log_path.open("wb") as log:
        # start_new_session: the job (and its ProcessPoolExecutor workers)
        # gets its own process group, so a Ctrl+C on the Streamlit server
        # doesn't kill running jobs, and stop_job can killpg the whole tree.
        proc = subprocess.Popen(
            [*CLI_ARGV, command, *args],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )

    meta: dict[str, Any] = {
        "job_id": job_id,
        "command": command,
        "args": list(args),
        "pid": proc.pid,
        "started_at": started.isoformat(),
        "state_file": state_file,
        "stop_requested": False,
    }
    _write_json(job_dir / "job.json", meta)

    # ponytail: the reaper dies with the dashboard process — a job that
    # outlives a dashboard restart keeps running but loses its exit code
    # ("unknown" after it ends). Re-attach logic only if that ever matters.
    threading.Thread(target=_reap, args=(proc, job_dir / "returncode"), daemon=True).start()
    return _to_job(meta, job_dir)


def list_jobs(jobs_dir: Path = DEFAULT_JOBS_DIR) -> list[Job]:
    """All jobs on disk, newest first. Unreadable job dirs are skipped."""
    if not jobs_dir.is_dir():
        return []
    jobs: list[Job] = []
    for job_dir in sorted(jobs_dir.iterdir(), reverse=True):
        if not job_dir.is_dir():
            continue
        job = _load_job(job_dir)
        if job is not None:
            jobs.append(job)
    return jobs


def stop_job(job: Job) -> None:
    """Request a graceful stop: SIGINT to the job's process group.

    The flag is written before the signal so a job that exits immediately
    still reads as "stopped" rather than "failed".
    """
    meta_path = job.job_dir / "job.json"
    try:
        meta: dict[str, Any] = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        meta = {
            "job_id": job.job_id,
            "command": job.command,
            "args": job.args,
            "pid": job.pid,
            "started_at": job.started_at,
            "state_file": job.state_file,
        }
    meta["stop_requested"] = True
    _write_json(meta_path, meta)

    try:
        os.killpg(job.pid, signal.SIGINT)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(job.pid, signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            pass  # already gone


def running_job_for_state(state_file: str, jobs_dir: Path = DEFAULT_JOBS_DIR) -> Job | None:
    """First running job bound to ``state_file`` (duplicate-engine guard)."""
    target = Path(state_file).resolve()
    for job in list_jobs(jobs_dir):
        if not job.is_running or not job.state_file:
            continue
        if Path(job.state_file).resolve() == target:
            return job
    return None


def read_log_tail(job: Job, max_bytes: int = 16_384) -> str:
    """Last ``max_bytes`` of the job's merged stdout/stderr."""
    log_path = job.job_dir / "output.log"
    try:
        with log_path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


# ── internals ────────────────────────────────────────────────────────


def _reap(proc: subprocess.Popen[bytes], rc_path: Path) -> None:
    """Wait on the child (prevents zombies) and persist its exit code."""
    rc = proc.wait()
    try:
        rc_path.write_text(str(rc))
    except OSError:
        pass


def _load_job(job_dir: Path) -> Job | None:
    try:
        meta: dict[str, Any] = json.loads((job_dir / "job.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return _to_job(meta, job_dir)


def _to_job(meta: dict[str, Any], job_dir: Path) -> Job:
    returncode = _read_returncode(job_dir)
    stop_requested = bool(meta.get("stop_requested", False))
    if returncode is not None:
        if stop_requested:
            status = "stopped"
        else:
            status = "finished" if returncode == 0 else "failed"
    elif _pid_alive(int(meta["pid"])):
        # ponytail: a recycled pid would read "running" forever — acceptable
        # for a single-user local GUI; a boot-id check is the upgrade path.
        status = "running"
    else:
        status = "unknown"  # exit code lost (e.g. dashboard restarted)
    return Job(
        job_id=str(meta.get("job_id", job_dir.name)),
        command=str(meta.get("command", "")),
        args=[str(a) for a in meta.get("args", [])],
        pid=int(meta["pid"]),
        started_at=str(meta.get("started_at", "")),
        state_file=meta.get("state_file"),
        stop_requested=stop_requested,
        status=status,
        returncode=returncode,
        job_dir=job_dir,
    )


def _read_returncode(job_dir: Path) -> int | None:
    try:
        return int((job_dir / "returncode").read_text().strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomic write — same tmp+replace pattern as live/control.py."""
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise
