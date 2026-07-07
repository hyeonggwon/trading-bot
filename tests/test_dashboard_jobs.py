"""Job manager lifecycle tests — stub subprocesses instead of the real CLI.

`jobs.CLI_ARGV` is monkeypatched to `python -c <script>` so no real
scan/train ever spawns; the command name/args become ignored extra argv.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable

import pytest

from tradingbot.dashboard import jobs


def _wait_for(cond: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def stub_cli(monkeypatch):
    """Make start_job spawn a tiny inline script instead of the real CLI."""

    def set_script(script: str) -> None:
        monkeypatch.setattr(jobs, "CLI_ARGV", [sys.executable, "-c", script])

    return set_script


class TestJobLifecycle:
    def test_finished_job(self, tmp_path, stub_cli):
        stub_cli("print('hello from job')")
        job = jobs.start_job("stub", ["--x", "1"], jobs_dir=tmp_path)
        assert _wait_for(lambda: not jobs.list_jobs(tmp_path)[0].is_running)
        done = jobs.list_jobs(tmp_path)[0]
        assert done.status == "finished"
        assert done.returncode == 0
        assert done.job_id == job.job_id
        assert "hello from job" in jobs.read_log_tail(done)

    def test_failed_job(self, tmp_path, stub_cli):
        stub_cli("raise SystemExit(3)")
        jobs.start_job("stub", [], jobs_dir=tmp_path)
        assert _wait_for(lambda: jobs.list_jobs(tmp_path)[0].returncode is not None)
        failed = jobs.list_jobs(tmp_path)[0]
        assert failed.status == "failed"
        assert failed.returncode == 3

    def test_stop_sends_graceful_sigint(self, tmp_path, stub_cli):
        """취소는 SIGINT — 엔진의 KeyboardInterrupt(state 저장) 경로를 탄다."""
        stub_cli("import time; time.sleep(60)")
        job = jobs.start_job("stub", [], jobs_dir=tmp_path)
        assert job.is_running
        jobs.stop_job(job)
        assert _wait_for(lambda: jobs.list_jobs(tmp_path)[0].returncode is not None)
        stopped = jobs.list_jobs(tmp_path)[0]
        assert stopped.status == "stopped"
        assert stopped.returncode != 0

    def test_jobs_survive_manager_reload(self, tmp_path, stub_cli):
        """잡 상태는 디스크 기반 — 세션(메모리)과 무관하게 재조회 가능."""
        stub_cli("print('ok')")
        job = jobs.start_job("stub", [], jobs_dir=tmp_path)
        assert _wait_for(lambda: jobs.list_jobs(tmp_path)[0].returncode is not None)
        reloaded = jobs.list_jobs(tmp_path)
        assert [j.job_id for j in reloaded] == [job.job_id]
        assert reloaded[0].args == []


class TestStateFileGuard:
    def test_running_job_found_by_state_file(self, tmp_path, stub_cli):
        stub_cli("import time; time.sleep(60)")
        state = str(tmp_path / "s1.json")
        job = jobs.start_job("paper", [], jobs_dir=tmp_path / "jobs", state_file=state)
        try:
            assert jobs.running_job_for_state(state, tmp_path / "jobs") is not None
            other = str(tmp_path / "other.json")
            assert jobs.running_job_for_state(other, tmp_path / "jobs") is None
        finally:
            jobs.stop_job(job)

    def test_finished_job_releases_state_file(self, tmp_path, stub_cli):
        stub_cli("print('done')")
        state = str(tmp_path / "s1.json")
        jobs.start_job("paper", [], jobs_dir=tmp_path / "jobs", state_file=state)
        assert _wait_for(lambda: jobs.list_jobs(tmp_path / "jobs")[0].returncode is not None)
        assert jobs.running_job_for_state(state, tmp_path / "jobs") is None
