"""Selection pipeline tests: stage logic units + a small synthetic end-to-end run."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tradingbot.backtest import pipeline as pl
from tradingbot.backtest.parallel import ScanResult


def _res(
    name: str = "s",
    sharpe: float = 1.0,
    trades: int = 50,
    entry: str = "",
    exit_: str = "",
    max_dd: float = 0.1,
    error: str | None = None,
) -> ScanResult:
    return ScanResult(
        strategy=name,
        symbol="BTC/KRW",
        timeframe="1h",
        sharpe_ratio=sharpe,
        total_return=0.1,
        max_drawdown=max_dd,
        win_rate=0.5,
        profit_factor=1.5,
        total_trades=trades,
        entry=entry,
        exit=exit_,
        error=error,
    )


def _wf(
    name: str,
    oos_sharpe: float,
    wf_eff: float = 0.5,
    trades: int = 30,
    kind: str = "strategy",
) -> dict[str, Any]:
    return {
        "candidate": {
            "name": name,
            "symbol": "BTC/KRW",
            "timeframe": "1h",
            "kind": kind,
            "entry": "",
            "exit": "",
        },
        "summary": {
            "num_windows": 3,
            "avg_train_sharpe": 1.0,
            "avg_test_sharpe": oos_sharpe,
            "avg_test_return": 0.01,
            "walk_forward_efficiency": wf_eff,
            "overfitting_ratio": 0.2,
            "cumulative_test_return": 0.05,
            "total_test_trades": trades,
        },
        "windows": [],
    }


class TestSelectCandidates:
    def test_top_n_with_gates(self):
        """min-trades·ML 게이트 통과 후 sort_by 상위 N — 제외는 사유와 함께."""
        results = [
            _res("a", sharpe=2.0),
            _res("b", sharpe=1.0),
            _res("few", sharpe=3.0, trades=2),
            _res("lgbm", sharpe=9.9),
            _res("MLT", sharpe=2.5, entry="lgbm_prob:0.45 + trend_up:4", exit_="rsi_overbought:70"),
            _res("T2", sharpe=1.5, entry="trend_up:4", exit_="rsi_overbought:70"),
        ]
        selected, excluded = pl.select_candidates(
            results, top=2, min_trades=10, sort_by="sharpe_ratio"
        )
        assert [c.name for c in selected] == ["a", "T2"]
        assert selected[0].kind == "strategy"
        assert selected[1].kind == "combined"
        assert selected[1].entry == "trend_up:4"
        reasons = {e["name"]: e["reason"] for e in excluded}
        assert "min_trades" in reasons["few"]
        assert "ml-candidate" in reasons["lgbm"]
        assert "ml-candidate" in reasons["MLT"]

    def test_max_drawdown_sorts_ascending(self):
        results = [_res("hi_dd", max_dd=0.5), _res("lo_dd", max_dd=0.1)]
        selected, _ = pl.select_candidates(results, top=1, min_trades=0, sort_by="max_drawdown")
        assert [c.name for c in selected] == ["lo_dd"]


class TestRankCandidates:
    def test_rank_by_oos_sharpe_with_low_trades_last(self):
        """OOS Sharpe가 높아도 저거래 후보는 항상 뒤로."""
        wf = [
            _wf("noise", oos_sharpe=9.0, trades=2),
            _wf("solid", oos_sharpe=1.5),
            _wf("meh", oos_sharpe=0.5),
        ]
        ranking = pl.rank_candidates(wf, [], rank_by="avg_test_sharpe", min_trades=10)
        assert [r["name"] for r in ranking] == ["solid", "meh", "noise"]
        assert ranking[-1]["low_trades"] is True
        assert [r["rank"] for r in ranking] == [1, 2, 3]

    def test_tiebreak_by_wf_efficiency_and_scan_join(self):
        wf = [_wf("lo_eff", oos_sharpe=1.0, wf_eff=0.3), _wf("hi_eff", oos_sharpe=1.0, wf_eff=0.8)]
        scans = [_res("hi_eff", sharpe=2.2)]
        ranking = pl.rank_candidates(wf, scans, rank_by="avg_test_sharpe", min_trades=10)
        assert [r["name"] for r in ranking] == ["hi_eff", "lo_eff"]
        assert ranking[0]["scan_sharpe"] == 2.2
        assert ranking[1]["scan_sharpe"] is None


class TestSerializeWfReport:
    def test_round_trips_through_json(self):
        from tradingbot.backtest.walk_forward import WalkForwardReport, WalkForwardWindow

        window = WalkForwardWindow(
            window_index=0,
            train_start=pd.Timestamp("2024-01-01", tz="UTC"),
            train_end=pd.Timestamp("2024-02-01", tz="UTC"),
            test_start=pd.Timestamp("2024-02-01", tz="UTC"),
            test_end=pd.Timestamp("2024-03-01", tz="UTC"),
            best_params={"fast_period": 10},
            train_sharpe=1.2,
            train_return=0.1,
            test_sharpe=0.8,
            test_return=0.05,
            test_trades=12,
            test_max_drawdown=0.07,
        )
        report = WalkForwardReport(windows=[window], strategy_name="x")
        payload = json.loads(json.dumps(pl.serialize_wf_report(report)))
        assert payload["summary"]["avg_test_sharpe"] == 0.8
        assert payload["windows"][0]["train_start"].startswith("2024-01-01")
        assert payload["windows"][0]["best_params"] == {"fast_period": "10"}


class TestDeployArtifacts:
    WINNER_COMBINED = {
        "name": "T",
        "symbol": "BTC/KRW",
        "timeframe": "1h",
        "kind": "combined",
        "entry": "trend_up:4 + rsi_oversold:30",
        "exit": "rsi_overbought:70",
        "rank_value": 1.234,
        "low_trades": False,
    }

    def test_combined_winner_artifacts(self, tmp_path):
        files = pl.write_deploy_artifacts(tmp_path, self.WINNER_COMBINED, pl.PipelineOptions())
        assert [f.name for f in files] == ["paper.sh", "live.sh", "docker-compose.override.yml"]

        paper = (tmp_path / "deploy" / "paper.sh").read_text()
        assert "tradingbot paper" in paper
        assert "--entry 'trend_up:4 + rsi_oversold:30'" in paper  # shlex quoting
        assert "--exit rsi_overbought:70" in paper
        assert (tmp_path / "deploy" / "paper.sh").stat().st_mode & stat.S_IXUSR

        live = (tmp_path / "deploy" / "live.sh").read_text()
        assert "tradingbot live" in live
        assert "REAL MONEY" in live

        compose = (tmp_path / "deploy" / "docker-compose.override.yml").read_text()
        assert "services:" in compose
        assert f"  {pl.COMPOSE_SERVICE}:" in compose
        assert '"--entry", "trend_up:4 + rsi_oversold:30"' in compose
        assert pl.DOCKER_STATE_FILE in compose
        # paper.sh ⇄ compose parity: both carry the run's --balance
        assert '"--balance", "1000000"' in compose

    def test_strategy_winner_uses_strategy_flag(self, tmp_path):
        winner = {
            **self.WINNER_COMBINED,
            "name": "sma_cross",
            "kind": "strategy",
            "entry": "",
            "exit": "",
        }
        pl.write_deploy_artifacts(tmp_path, winner, pl.PipelineOptions())
        paper = (tmp_path / "deploy" / "paper.sh").read_text()
        assert "--strategy sma_cross" in paper
        assert "--entry" not in paper


class TestCrashSemantics:
    def test_mid_run_crash_marks_manifest_failed(self, tmp_path, monkeypatch):
        """중간 크래시 시 manifest.status가 'running'으로 남지 않고 failed + error 기록."""
        monkeypatch.chdir(tmp_path)

        def _boom(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(pl, "_scan_stage", _boom)
        templates = [{"label": "T", "entry": "ema_above:20", "exit": "rsi_overbought:60"}]
        with pytest.raises(RuntimeError, match="boom"):
            pl.run_pipeline(pl.PipelineOptions(), templates=templates, log=lambda m: None)

        run_dirs = list((tmp_path / "results" / "pipeline").iterdir())
        assert len(run_dirs) == 1
        manifest = json.loads((run_dirs[0] / "manifest.json").read_text())
        assert manifest["status"] == "failed"
        assert "boom" in manifest["error"]


class TestPipelineOptionsValidation:
    def test_invalid_metrics_and_double_skip_raise(self, tmp_path):
        with pytest.raises(pl.PipelineError, match="sort-by"):
            pl.run_pipeline(pl.PipelineOptions(sort_by="nope"), templates=[])
        with pytest.raises(pl.PipelineError, match="rank-by"):
            pl.run_pipeline(pl.PipelineOptions(rank_by="nope"), templates=[])
        with pytest.raises(pl.PipelineError, match="nothing to scan"):
            pl.run_pipeline(pl.PipelineOptions(skip_rules=True, skip_combine=True), templates=[])

    def test_top_below_one_raises_clear_error(self, tmp_path):
        """--top 0은 min-trades 탓으로 오인되는 빈 선별 대신 명확한 에러."""
        with pytest.raises(pl.PipelineError, match="--top"):
            pl.run_pipeline(pl.PipelineOptions(top=0), templates=[])


class TestSummaryMd:
    def test_negative_oos_winner_flagged(self, tmp_path):
        """전 후보 OOS 음수면 summary.md에 '배포 신호 아님' 경고가 남는다."""
        manifest = {
            "run_id": "r1",
            "created_at": "2026-07-08T00:00:00+00:00",
            "options": {
                "include_train": False,
                "top": 1,
                "sort_by": "sharpe_ratio",
                "min_trades": 10,
                "train_months": 3,
                "test_months": 1,
                "rank_by": "avg_test_sharpe",
            },
        }
        ranking = [
            {
                "rank": 1,
                "name": "least-bad",
                "symbol": "BTC/KRW",
                "timeframe": "1h",
                "scan_sharpe": 1.0,
                "oos_sharpe": -0.4,
                "wf_efficiency": 0.0,
                "oos_cum_return": -0.05,
                "oos_trades": 30,
                "low_trades": False,
            }
        ]
        pl._write_summary_md(tmp_path, manifest, ranking, [])
        text = (tmp_path / "summary.md").read_text()
        assert "positive out-of-sample" in text


class TestPipelineEndToEnd:
    def test_small_synthetic_run(self, tmp_path, monkeypatch):
        """합성 3.5개월 캔들로 전체 5단계 실행 — 산출물·매니페스트·아티팩트 검증."""
        from tradingbot.data.storage import save_candles

        monkeypatch.chdir(tmp_path)

        rng = np.random.default_rng(42)
        n = 24 * 105  # ~3.5 months hourly
        idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
        close = 100 * np.exp(np.linspace(0, 0.3, n) + rng.normal(0, 0.008, n).cumsum())
        df = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": rng.uniform(1, 10, n),
            },
            index=idx,
        )
        save_candles(df, "BTC/KRW", "1h", Path("data"))

        templates = [
            {"label": "T-Test", "entry": "ema_above:20", "exit": "rsi_overbought:60"},
        ]
        options = pl.PipelineOptions(
            top=1,
            min_trades=0,
            train_months=1,
            test_months=1,
            workers=1,
            skip_rules=True,  # registry grid-optimization is too slow for a unit test
        )
        result = pl.run_pipeline(options, templates=templates, log=lambda m: None)

        run_dir = Path(result["run_dir"])
        for fname in (
            "manifest.json",
            "stage1_scan.json",
            "selection.json",
            "stage2_walkforward.json",
            "ranking.json",
            "summary.md",
        ):
            assert (run_dir / fname).exists(), fname

        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["status"] == "complete"
        assert set(manifest["stages"]) == {
            "scan",
            "select",
            "walk_forward",
            "rank",
            "deploy_artifacts",
        }

        assert result["winner"] is not None
        assert result["winner"]["name"] == "T-Test"
        wf_doc = json.loads((run_dir / "stage2_walkforward.json").read_text())
        assert wf_doc["results"][0]["summary"]["num_windows"] >= 1

        paper = (run_dir / "deploy" / "paper.sh").read_text()
        assert "ema_above:20" in paper
        compose = (run_dir / "deploy" / "docker-compose.override.yml").read_text()
        assert "ema_above:20" in compose
