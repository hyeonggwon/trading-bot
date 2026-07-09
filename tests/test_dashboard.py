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
        """전 모드가 빈 환경(state.json·models/·data/ 없음)에서 예외 없이 렌더링."""
        from streamlit.testing.v1 import AppTest

        monkeypatch.chdir(tmp_path)
        at = AppTest.from_file(APP, default_timeout=30)
        at.run()
        assert not at.exception

        for mode in at.sidebar.radio[0].options:
            at.sidebar.radio[0].set_value(mode).run()
            assert not at.exception, mode

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


class TestPipelinePage:
    def test_populated_run_renders_stage_tables(self, tmp_path, monkeypatch):
        """합성 런 디렉토리로 3개 탭(스캔·WF·랭킹) 표 + 우승 카드 렌더."""
        import json

        from streamlit.testing.v1 import AppTest

        run_dir = tmp_path / "results" / "pipeline" / "20260707_000000_000000"
        deploy = run_dir / "deploy"
        deploy.mkdir(parents=True)

        def _scan_row(name, sharpe, entry=""):
            return {
                "strategy": name,
                "symbol": "BTC/KRW",
                "timeframe": "1h",
                "sharpe_ratio": sharpe,
                "total_return": 0.1,
                "max_drawdown": 0.1,
                "win_rate": 0.5,
                "profit_factor": 1.5,
                "total_trades": 30,
                "entry": entry,
                "exit": "",
                "error": None,
            }

        def _rank_row(rank, name, scan_sharpe, oos_sharpe):
            return {
                "rank": rank,
                "name": name,
                "symbol": "BTC/KRW",
                "timeframe": "1h",
                "kind": "combined",
                "entry": "trend_up:4",
                "exit": "rsi_overbought:70",
                "validation": "walk_forward_combined",
                "scan_sharpe": scan_sharpe,
                "scan_return": 0.1,
                "oos_sharpe": oos_sharpe,
                "oos_cum_return": 0.05,
                "wf_efficiency": 0.6,
                "overfitting_ratio": 0.3,
                "oos_trades": 25,
                "num_windows": 3,
                "rank_value": oos_sharpe,
                "low_trades": False,
            }

        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "run_id": run_dir.name,
                    "created_at": "2026-07-07T00:00:00+00:00",
                    "status": "complete",
                    "options": {
                        "top": 2,
                        "sort_by": "sharpe_ratio",
                        "min_trades": 10,
                        "train_months": 3,
                        "test_months": 1,
                        "rank_by": "avg_test_sharpe",
                    },
                    "stages": {},
                }
            )
        )
        (run_dir / "stage1_scan.json").write_text(
            json.dumps(
                {
                    "window": "auto holdout",
                    "results": [_scan_row("A", 2.0), _scan_row("B", 1.0)],
                    "failures": [],
                }
            )
        )
        (run_dir / "selection.json").write_text(
            json.dumps(
                {
                    "sort_by": "sharpe_ratio",
                    "min_trades": 10,
                    "selected": [
                        {
                            "name": "A",
                            "symbol": "BTC/KRW",
                            "timeframe": "1h",
                            "kind": "strategy",
                            "entry": "",
                            "exit": "",
                        }
                    ],
                    "excluded": [{"name": "lgbm", "reason": "ml-candidate"}],
                }
            )
        )
        (run_dir / "stage2_walkforward.json").write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "candidate": {
                                "name": "A",
                                "symbol": "BTC/KRW",
                                "timeframe": "1h",
                                "kind": "strategy",
                                "entry": "",
                                "exit": "",
                            },
                            "validation": "walk_forward",
                            "summary": {
                                "num_windows": 2,
                                "avg_train_sharpe": 1.2,
                                "avg_test_sharpe": 0.9,
                                "avg_test_return": 0.02,
                                "walk_forward_efficiency": 0.75,
                                "overfitting_ratio": 0.25,
                                "cumulative_test_return": 0.04,
                                "total_test_trades": 22,
                            },
                            "windows": [
                                {
                                    "window_index": 0,
                                    "train_start": "2024-01-01T00:00:00+00:00",
                                    "train_end": "2024-04-01T00:00:00+00:00",
                                    "test_start": "2024-04-01T00:00:00+00:00",
                                    "test_end": "2024-05-01T00:00:00+00:00",
                                    "best_params": {"fast_period": "10"},
                                    "train_sharpe": 1.2,
                                    "train_return": 0.1,
                                    "test_sharpe": 0.9,
                                    "test_return": 0.02,
                                    "test_trades": 11,
                                    "test_max_drawdown": 0.06,
                                }
                            ],
                        }
                    ]
                }
            )
        )
        (run_dir / "ranking.json").write_text(
            json.dumps(
                {
                    "rank_by": "avg_test_sharpe",
                    "ranking": [_rank_row(1, "A", 2.0, 0.9), _rank_row(2, "B", 1.0, 0.4)],
                }
            )
        )
        (deploy / "paper.sh").write_text("#!/usr/bin/env bash\ntradingbot paper --strategy A\n")
        (deploy / "docker-compose.override.yml").write_text("services:\n  bot:\n    command: []\n")
        (run_dir / "stage0_ml_train.json").write_text(
            json.dumps(
                {
                    "trained": [{"symbol": "BTC/KRW", "timeframe": "1h", "holdout_auc": 0.55}],
                    "fresh": [{"symbol": "ETH/KRW", "timeframe": "1h"}],
                    "failed": [],
                }
            )
        )

        monkeypatch.chdir(tmp_path)
        at = AppTest.from_file(APP, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Pipeline").run()
        assert not at.exception
        # Stage0 trained/fresh + scan + excluded + WF summary + windows + ranking
        assert len(at.dataframe) >= 5
        assert any("Winner" in str(m.value) for m in at.markdown)
        assert any("ML Train" in str(e.label) for e in at.expander)

    def test_running_partial_run_renders_banners(self, tmp_path, monkeypatch):
        """진행 중 런(1단계만 기록)에서 running 배너 + 미기록 단계 경고 렌더."""
        import json

        from streamlit.testing.v1 import AppTest

        run_dir = tmp_path / "results" / "pipeline" / "20260708_000000_000000"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "run_id": run_dir.name,
                    "created_at": "2026-07-08T00:00:00+00:00",
                    "status": "running",
                    "options": {
                        "top": 5,
                        "sort_by": "sharpe_ratio",
                        "min_trades": 10,
                        "train_months": 3,
                        "test_months": 1,
                        "rank_by": "avg_test_sharpe",
                    },
                    "stages": {},
                }
            )
        )
        (run_dir / "stage1_scan.json").write_text(
            json.dumps({"window": "auto holdout", "results": [], "failures": []})
        )

        monkeypatch.chdir(tmp_path)
        at = AppTest.from_file(APP, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Pipeline").run()
        assert not at.exception
        assert any("progress" in str(i.value) for i in at.info)
        # stage 2 / ranking not written yet → warnings, no crash
        assert len(at.warning) >= 2


class TestLiveGate:
    def test_live_requires_typed_confirmation(self, tmp_path, monkeypatch):
        """확인 문구 없이 live 제출 → 에러 표시, 잡 스폰 없음 (실주문 게이트)."""
        from streamlit.testing.v1 import AppTest

        monkeypatch.chdir(tmp_path)
        at = AppTest.from_file(APP, default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("Trading").run()
        assert not at.exception

        submit = next(b for b in at.button if b.key == "FormSubmitter:form_live-Start live")
        submit.click().run()
        assert not at.exception
        assert any("Confirmation failed" in str(e.value) for e in at.error)
        assert not (tmp_path / "personal" / "gui_jobs").exists()
