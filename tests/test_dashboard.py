"""Dashboard smoke tests — skipped when the dashboard extra isn't installed.

CI installs only ``.[dev]`` so these skip there; on machines with
``pip install -e ".[dashboard]"`` they exercise the real Streamlit render
path via AppTest, which plain function tests can't cover.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

REPO_ROOT = Path(__file__).resolve().parent.parent
APP = str(REPO_ROOT / "src" / "tradingbot" / "dashboard" / "app.py")


class TestDashboardSmoke:
    def test_app_renders_all_modes(self, tmp_path, monkeypatch):
        """세 모드 전부 빈 환경(state.json·models/ 없음)에서 예외 없이 렌더링."""
        from streamlit.testing.v1 import AppTest

        monkeypatch.chdir(tmp_path)
        at = AppTest.from_file(APP, default_timeout=30)
        at.run()
        assert not at.exception

        at.sidebar.radio[0].set_value("Models").run()
        assert not at.exception

        at.sidebar.radio[0].set_value("Backtest Viewer").run()
        assert not at.exception

    def test_pause_button_writes_control_file(self, tmp_path, monkeypatch):
        """일시정지 버튼 클릭이 control 파일에 pause 플래그를 기록해야 한다.

        (파일→엔진 방향은 test_live_engine.TestEntryPauseControl 이 증명 —
        이 테스트가 대시보드→파일 방향을 닫아 e2e 체인이 완성된다.)
        """
        from streamlit.testing.v1 import AppTest

        from tradingbot.live.control import read_pause

        monkeypatch.chdir(tmp_path)
        at = AppTest.from_file(APP, default_timeout=30)
        at.run()
        assert not at.exception

        at.sidebar.button[0].click().run()
        assert not at.exception
        assert read_pause(tmp_path / "state.control.json") is True
