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

    def test_populated_state_renders_rails_metrics(self, tmp_path, monkeypatch):
        """채워진 state.json 에서 안전레일 메트릭(드로다운·일일 PnL)이 표시된다."""
        import json

        from streamlit.testing.v1 import AppTest

        state = {
            "positions": {
                "BTC/KRW": {
                    "symbol": "BTC/KRW",
                    "side": "long",
                    "size": 0.01,
                    "entry_price": 100_000_000.0,
                    "stop_loss": 98_000_000.0,
                    "entry_time": "2026-07-01T00:00:00+00:00",
                }
            },
            "equity_history": [
                {"timestamp": "2026-07-01T00:00:00+00:00", "equity": 1_000_000.0},
                {"timestamp": "2026-07-02T00:00:00+00:00", "equity": 1_050_000.0},
            ],
            "peak_equity": 1_100_000.0,
            "daily_pnl": -5_000.0,
            "cum_realized_pnl": 50_000.0,
            "ledger_baseline": 1_000_000.0,
            "saved_at": "2026-07-02T00:00:00+00:00",
        }
        (tmp_path / "state.json").write_text(json.dumps(state))
        monkeypatch.chdir(tmp_path)
        at = AppTest.from_file(APP, default_timeout=30)
        at.run()
        assert not at.exception

        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Drawdown vs Peak"] == f"{50_000 / 1_100_000:.2%}"
        assert metrics["Daily PnL (realized)"] == "-5,000 KRW"
        assert metrics["Cum Realized PnL"] == "+50,000 KRW"

    def test_models_catalog_renders_entries(self, tmp_path, monkeypatch):
        """models/ 메타가 있으면 카탈로그 테이블이 예외 없이 렌더링된다."""
        import json

        from streamlit.testing.v1 import AppTest

        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "lgbm_BTC_KRW_1h_meta.json").write_text(
            json.dumps({"symbol": "BTC/KRW", "timeframe": "1h", "holdout_auc": 0.5731})
        )
        monkeypatch.chdir(tmp_path)
        at = AppTest.from_file(APP, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Models").run()
        assert not at.exception
        assert len(at.dataframe) == 1
