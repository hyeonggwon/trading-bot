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
    _RESULTS = [
        _res("a", sharpe=2.0),
        _res("b", sharpe=1.0),
        _res("few", sharpe=3.0, trades=2),
        _res("lgbm", sharpe=9.9),
        _res("MLT", sharpe=2.5, entry="lgbm_prob:0.45 + trend_up:4", exit_="rsi_overbought:70"),
        _res("T2", sharpe=1.5, entry="trend_up:4", exit_="rsi_overbought:70"),
    ]

    def test_ml_candidate_competes_when_included(self):
        """include_ml=True: lgbm은 kind='ml', lgbm_prob 템플릿은 kind='combined'로 경쟁."""
        selected, excluded = pl.select_candidates(
            self._RESULTS, top=2, min_trades=10, sort_by="sharpe_ratio"
        )
        assert [c.name for c in selected] == ["lgbm", "MLT"]
        assert selected[0].kind == "ml"
        assert selected[1].kind == "combined"
        assert selected[1].entry == "lgbm_prob:0.45 + trend_up:4"
        reasons = {e["name"]: e["reason"] for e in excluded}
        assert "MLT" not in reasons  # 더는 scan-only 제외 아님
        assert "min_trades" in reasons["few"]

    def test_no_ml_excludes_all_ml_candidates(self):
        """include_ml=False: lgbm·lgbm_prob 전부 제외 (현행 --no-ml 동작)."""
        selected, excluded = pl.select_candidates(
            self._RESULTS, top=2, min_trades=10, sort_by="sharpe_ratio", include_ml=False
        )
        assert [c.name for c in selected] == ["a", "T2"]
        assert selected[0].kind == "strategy"
        assert selected[1].kind == "combined"
        reasons = {e["name"]: e["reason"] for e in excluded}
        assert "min_trades" in reasons["few"]
        assert "no-ml" in reasons["lgbm"]
        assert "no-ml" in reasons["MLT"]

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
            pl.run_pipeline(pl.PipelineOptions(ml=False), templates=templates, log=lambda m: None)

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
            pl.run_pipeline(
                pl.PipelineOptions(skip_rules=True, skip_combine=True, ml=False), templates=[]
            )

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
                "wf_train_months": 6,
                "wf_test_months": 2,
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
            wf_train_months=1,
            wf_test_months=1,
            workers=1,
            skip_rules=True,  # registry grid-optimization is too slow for a unit test
            ml=False,  # ML flow has its own e2e (TestPipelineMlEndToEnd)
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


class TestNeedsTraining:
    LAST = pd.Timestamp("2026-07-08", tz="UTC")

    def test_missing_model_or_meta_fields_retrain(self):
        assert pl._needs_training(None, last_candle=self.LAST, stale_days=7, retrain_all=False)
        assert pl._needs_training({}, last_candle=self.LAST, stale_days=7, retrain_all=False)
        assert pl._needs_training(
            {"data_end": "not-a-date"}, last_candle=self.LAST, stale_days=7, retrain_all=False
        )

    def test_fresh_model_skipped_and_stale_retrained(self):
        fresh = {"data_end": "2026-07-05T00:00:00+00:00"}
        stale = {"data_end": "2026-06-01T00:00:00+00:00"}
        assert (
            pl._needs_training(fresh, last_candle=self.LAST, stale_days=7, retrain_all=False)
            is None
        )
        reason = pl._needs_training(stale, last_candle=self.LAST, stale_days=7, retrain_all=False)
        assert reason is not None and "stale" in reason

    def test_retrain_all_overrides_freshness(self):
        fresh = {"data_end": "2026-07-08T00:00:00+00:00"}
        assert (
            pl._needs_training(fresh, last_candle=self.LAST, stale_days=7, retrain_all=True)
            == "retrain-all"
        )

    def test_corrupt_meta_counts_as_missing_and_retrains(self, tmp_path):
        """손상된 meta.json은 런 중단 대신 '모델 없음'으로 취급 → 재학습 방향."""
        from tradingbot.ml.trainer import LGBMTrainer

        (tmp_path / "lgbm_BTC_KRW_4h_meta.json").write_text("{corrupt json!!")
        meta = LGBMTrainer.load_meta("BTC/KRW", "4h", tmp_path)
        assert meta is None
        assert pl._needs_training(meta, last_candle=self.LAST, stale_days=7, retrain_all=False)

    def test_empty_parquet_recorded_as_failed_not_crash(self, tmp_path, monkeypatch):
        """0행 parquet는 IndexError 크래시 대신 failed 항목으로 기록."""
        monkeypatch.chdir(tmp_path)
        pair_dir = Path("data/BTC_KRW")
        pair_dir.mkdir(parents=True)
        pd.DataFrame({"open": [], "high": [], "low": [], "close": [], "volume": []}).to_parquet(
            pair_dir / "1h.parquet"
        )
        summary = pl._ml_train_stage(pl.PipelineOptions(), log=lambda m: None, run_dir=Path("run"))
        assert summary["trained"] == []
        assert summary["fresh"] == []
        assert summary["failed"] == [
            {"symbol": "BTC/KRW", "timeframe": "1h", "reason": "empty data file"}
        ]


class TestMlTuneRouting:
    """Stage 0 tune/threshold 라우팅 — spawn 풀을 인라인 실행으로 치환해 캡처."""

    class _InlineFuture:
        def __init__(self, fn, args):
            self._fn, self._args = fn, args

        def result(self, timeout=None):
            return self._fn(*self._args)

    class _InlinePool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def submit(self, fn, *args):
            return TestMlTuneRouting._InlineFuture(fn, args)

    def _setup(self, tmp_path, monkeypatch, meta):
        from tradingbot.data.storage import save_candles
        from tradingbot.ml.trainer import LGBMTrainer

        monkeypatch.chdir(tmp_path)
        idx = pd.date_range("2026-01-01", periods=48, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx
        )
        save_candles(df, "BTC/KRW", "1h", Path("data"))
        monkeypatch.setattr(LGBMTrainer, "load_meta", lambda *a, **k: meta)
        monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", self._InlinePool)
        monkeypatch.setattr("concurrent.futures.as_completed", lambda fs: list(fs))

    def test_ml_tune_routes_to_tune_pair_with_meta_targets(self, tmp_path, monkeypatch):
        """--ml-tune: 재학습 대상이 tune_pair로 가고 타깃 설정은 meta에서 상속."""
        from tradingbot.ml.parallel import TunePairResult

        # data_end가 오래됨 → stale → 재학습 대상
        meta = {"data_end": "2020-01-01T00:00:00+00:00", "target_kind": "atr", "atr_mult": 1.5}
        self._setup(tmp_path, monkeypatch, meta)

        captured = {}

        def fake_tune_pair(*args):
            captured["args"] = args
            return TunePairResult(
                symbol=args[0], timeframe=args[1], best_value=1.23, objective=args[17]
            )

        monkeypatch.setattr("tradingbot.ml.parallel.tune_pair", fake_tune_pair)
        options = pl.PipelineOptions(
            ml_tune=True, ml_tune_trials=7, ml_tune_budget_sec=99.0, workers=1
        )
        summary = pl._ml_train_stage(options, log=lambda m: None, run_dir=tmp_path)

        args = captured["args"]
        # tune_pair(sym, tf, data, model, ext, train_m, test_m, fwd, thr, target_kind,
        #           atr_mult, extra, entry, exit, balance, trials, budget, objective, ...)
        assert args[9] == "atr" and args[10] == 1.5  # meta 상속
        assert args[15] == 7 and args[16] == 99.0
        assert args[17] == "holdout_sharpe"
        row = summary["trained"][0]
        assert row["tuned"] is True and row["best_value"] == 1.23

    def test_smart_refresh_preserves_tuned_params(self, tmp_path, monkeypatch):
        """--ml-tune 없이도 stale 재학습이 meta의 best_params를 잃지 않는다."""
        from tradingbot.ml.parallel import PairTrainResult

        best = {"num_leaves": 31, "learning_rate": 0.02}
        meta = {"data_end": "2020-01-01T00:00:00+00:00", "tuning": {"best_params": best}}
        self._setup(tmp_path, monkeypatch, meta)

        captured = {}

        def fake_train_pair(*args):
            captured["args"] = args
            return PairTrainResult(
                symbol=args[0],
                timeframe=args[1],
                avg_auc=0.5,
                avg_precision=0.5,
                holdout_auc=0.6,
                holdout_precision=0.5,
                n_windows=1,
                model_path="m.lgb",
            )

        monkeypatch.setattr("tradingbot.ml.parallel.train_pair", fake_train_pair)
        summary = pl._ml_train_stage(
            pl.PipelineOptions(workers=1), log=lambda m: None, run_dir=tmp_path
        )

        assert captured["args"][-1] == best  # lgbm_params로 전달됨
        assert summary["trained"][0]["tuned"] is False

    def test_threshold_sweep_runs_for_trained_pairs(self, tmp_path, monkeypatch):
        """--ml-tune-thresholds: 이번 런 학습 pair에 스윕이 돌고 승자가 기록된다."""
        from tradingbot.ml.parallel import PairTrainResult, ThresholdTunePairResult

        self._setup(tmp_path, monkeypatch, None)  # meta 없음 → no model → 학습 대상

        def fake_train_pair(*args):
            return PairTrainResult(
                symbol=args[0],
                timeframe=args[1],
                avg_auc=0.5,
                avg_precision=0.5,
                holdout_auc=0.6,
                holdout_precision=0.5,
                n_windows=1,
                model_path="m.lgb",
            )

        captured = {}

        def fake_thresholds_pair(*args):
            captured["args"] = args
            return ThresholdTunePairResult(
                symbol=args[0], timeframe=args[1], best_entry=0.5, best_exit=0.25
            )

        monkeypatch.setattr("tradingbot.ml.parallel.train_pair", fake_train_pair)
        monkeypatch.setattr("tradingbot.ml.parallel.tune_thresholds_pair", fake_thresholds_pair)
        summary = pl._ml_train_stage(
            pl.PipelineOptions(ml_tune_thresholds=True, workers=1),
            log=lambda m: None,
            run_dir=tmp_path,
        )

        assert captured["args"][10] is True  # write_meta — 검증·배포로 자동 상속
        row = summary["trained"][0]
        assert row["best_entry"] == 0.5 and row["best_exit"] == 0.25


class TestMlMetaInheritance:
    def test_meta_values_reach_ml_walk_forward(self, tmp_path, monkeypatch):
        """저장 모델 meta의 비기본값이 검증 러너에 실제로 전달된다 (검증=배포 설정)."""
        from tradingbot.ml.strategy_walk_forward import MLStrategyWalkForwardReport
        from tradingbot.ml.trainer import LGBMTrainer

        captured: dict[str, Any] = {}

        class FakeRunner:
            def __init__(self, symbol: str, timeframe: str, **kwargs: Any) -> None:
                captured.update({"symbol": symbol, "timeframe": timeframe, **kwargs})

            def run(self, df: Any) -> MLStrategyWalkForwardReport:
                return MLStrategyWalkForwardReport(
                    windows=[],
                    avg_sharpe=0.0,
                    cumulative_return_pct=0.0,
                    total_trades=0,
                    avg_win_rate=0.0,
                    final_equity_multiple=1.0,
                    n_windows=0,
                    n_skipped=0,
                )

        monkeypatch.setattr("tradingbot.ml.strategy_walk_forward.MLStrategyWalkForward", FakeRunner)
        meta = {
            "forward_candles": 8,
            "threshold": 0.01,
            "target_kind": "atr",
            "atr_mult": 1.5,
            "include_extra": True,
            "entry_threshold": 0.5,
            "exit_threshold": 0.25,
        }
        monkeypatch.setattr(LGBMTrainer, "load_meta", lambda *a, **k: dict(meta))
        monkeypatch.chdir(tmp_path)  # no external data dir

        cand = pl.Candidate(name="lgbm", symbol="BTC/KRW", timeframe="4h", kind="ml")
        options = pl.PipelineOptions(wf_train_months=4, wf_test_months=2)
        result = pl._run_ml_walk_forward(cand, df=None, config=None, options=options)

        assert captured["forward_candles"] == 8
        assert captured["threshold"] == 0.01
        assert captured["target_kind"] == "atr"
        assert captured["atr_mult"] == 1.5
        assert captured["include_extra"] is True
        assert captured["entry_threshold"] == 0.5
        assert captured["exit_threshold"] == 0.25
        assert captured["train_months"] == 4
        assert captured["test_months"] == 2
        assert captured["external_data_dir"] is None
        assert result["summary"]["num_windows"] == 0


class TestMlCombinedWfWiring:
    def test_factory_injects_fresh_model_and_extras_forced_off(self, tmp_path, monkeypatch):
        """lgbm_prob 템플릿 검증: 팩토리가 윈도우 모델을 필터에 주입, include_extra는
        meta가 True여도 False 고정(필터가 extra 피처를 못 만들므로)."""
        from tradingbot.ml.strategy_walk_forward import MLStrategyWalkForwardReport
        from tradingbot.ml.trainer import LGBMTrainer
        from tradingbot.strategy.filters.ml import LgbmProbFilter

        captured: dict[str, Any] = {}

        class FakeRunner:
            def __init__(self, symbol: str, timeframe: str, **kwargs: Any) -> None:
                captured.update({"symbol": symbol, "timeframe": timeframe, **kwargs})

            def run(self, df: Any) -> MLStrategyWalkForwardReport:
                return MLStrategyWalkForwardReport()

        monkeypatch.setattr("tradingbot.ml.strategy_walk_forward.MLStrategyWalkForward", FakeRunner)
        meta = {"forward_candles": 8, "target_kind": "atr", "include_extra": True}
        monkeypatch.setattr(LGBMTrainer, "load_meta", lambda *a, **k: dict(meta))
        monkeypatch.chdir(tmp_path)  # no external data dir

        cand = pl.Candidate(
            name="MLT",
            symbol="BTC/KRW",
            timeframe="4h",
            kind="combined",
            entry="lgbm_prob:0.45 + trend_up:4",
            exit="rsi_overbought:70",
        )
        options = pl.PipelineOptions(wf_train_months=4, wf_test_months=2)
        result = pl._run_ml_walk_forward_combined(cand, df=None, config=None, options=options)

        assert captured["forward_candles"] == 8
        assert captured["target_kind"] == "atr"
        assert captured["include_extra"] is False  # meta True여도 강제 False
        assert captured["train_months"] == 4
        assert captured["test_months"] == 2
        assert captured["warmup_candles"] >= 300  # 콤바인 룰 필터 워밍업

        factory = captured["strategy_factory"]
        stub_model = object()
        strategy = factory(stub_model, None, ["close"], 1.7)
        assert strategy.symbols == ["BTC/KRW"]
        ml_filters = [f for f in strategy.entry_filters if isinstance(f, LgbmProbFilter)]
        assert len(ml_filters) == 1
        f = ml_filters[0]
        assert f._loaded and f._model is stub_model
        assert f._feature_names == ["close"]
        assert f._win_loss_ratio == 1.7
        assert f.threshold == 0.45  # 임계값은 템플릿 스펙에서

        # 윈도우마다 새 인스턴스 (상태 오염 없음)
        assert factory(stub_model, None, ["close"], 1.7) is not strategy
        assert result["summary"]["num_windows"] == 0


class TestMlWfAdapter:
    def _report(self):
        from tradingbot.ml.strategy_walk_forward import MLStrategyWalkForwardReport

        return MLStrategyWalkForwardReport(
            windows=[
                {
                    "window": 1,
                    "train_start": "2024-01-01",
                    "train_end": "2024-03-01",
                    "test_start": "2024-03-01",
                    "test_end": "2024-04-01",
                    "n_train": 900,
                    "n_test": 300,
                    "sharpe": 1.1,
                    "return_pct": 5.0,
                    "trades": 9,
                    "win_rate": 0.5,
                    "max_dd_pct": 3.0,
                    "final_balance": 1_050_000.0,
                    "win_loss_ratio_used": 1.2,
                }
            ],
            avg_sharpe=1.1,
            cumulative_return_pct=5.0,
            total_trades=9,
            avg_win_rate=0.5,
            final_equity_multiple=1.05,
            n_windows=1,
            n_skipped=0,
        )

    def test_percent_fields_normalized_to_fractions(self):
        """% 단위(ML 보고서) → fraction(룰 WF와 동일 척도) 정규화가 랭킹 오염을 막는다."""
        s = pl.serialize_ml_wf_report(self._report())
        assert s["summary"]["cumulative_test_return"] == pytest.approx(0.05)
        assert s["windows"][0]["test_return"] == pytest.approx(0.05)
        assert s["windows"][0]["test_max_drawdown"] == pytest.approx(0.03)

    def test_no_train_side_metrics(self):
        s = pl.serialize_ml_wf_report(self._report())
        assert s["summary"]["walk_forward_efficiency"] is None
        assert s["summary"]["avg_train_sharpe"] is None
        assert json.dumps(s)  # JSON-safe


class TestRankWithMlCandidates:
    def _ml_wf(self, name, oos_sharpe, trades=30):
        wf = _wf(name, oos_sharpe=oos_sharpe, trades=trades, kind="ml")
        wf["validation"] = "ml_walk_forward"
        wf["summary"]["walk_forward_efficiency"] = None
        wf["summary"]["avg_train_sharpe"] = None
        wf["summary"]["overfitting_ratio"] = None
        return wf

    def test_mixed_ranking_carries_validation(self):
        wf = [self._ml_wf("ml", 1.2), {**_wf("rule", oos_sharpe=0.9), "validation": "walk_forward"}]
        ranking = pl.rank_candidates(wf, [], rank_by="avg_test_sharpe", min_trades=10)
        assert [(r["name"], r["validation"]) for r in ranking] == [
            ("ml", "ml_walk_forward"),
            ("rule", "walk_forward"),
        ]

    def test_rank_by_wf_efficiency_demotes_ml_without_crash(self):
        """ML은 train측 지표가 없어 eff=None — efficiency 랭킹에서 뒤로, 예외 없음."""
        wf = [self._ml_wf("ml", 9.0), {**_wf("rule", oos_sharpe=0.5), "validation": "walk_forward"}]
        ranking = pl.rank_candidates(wf, [], rank_by="walk_forward_efficiency", min_trades=10)
        assert [r["name"] for r in ranking] == ["rule", "ml"]


class TestMlDeployArtifacts:
    def test_ml_winner_uses_strategy_lgbm_with_model_note(self, tmp_path):
        winner = {
            "name": "lgbm",
            "symbol": "BTC/KRW",
            "timeframe": "4h",
            "kind": "ml",
            "entry": "",
            "exit": "",
            "rank_value": 1.5,
            "low_trades": False,
        }
        pl.write_deploy_artifacts(tmp_path, winner, pl.PipelineOptions())
        paper = (tmp_path / "deploy" / "paper.sh").read_text()
        assert "--strategy lgbm" in paper
        assert "models/lgbm_BTC_KRW_4h.lgb" in paper
        assert "--entry" not in paper
        compose = (tmp_path / "deploy" / "docker-compose.override.yml").read_text()
        assert '"--strategy", "lgbm"' in compose
        assert "models/lgbm_BTC_KRW_4h.lgb" in compose

    def test_lgbm_prob_winner_keeps_entry_argv_with_model_note(self, tmp_path):
        """lgbm_prob 템플릿 승자: --entry argv 유지 + 모델 의존성 주석."""
        winner = {
            "name": "MLT",
            "symbol": "BTC/KRW",
            "timeframe": "4h",
            "kind": "combined",
            "entry": "lgbm_prob:0.45 + trend_up:4",
            "exit": "rsi_overbought:70",
            "rank_value": 1.2,
            "low_trades": False,
        }
        pl.write_deploy_artifacts(tmp_path, winner, pl.PipelineOptions())
        paper = (tmp_path / "deploy" / "paper.sh").read_text()
        assert "--entry" in paper and "lgbm_prob:0.45" in paper
        assert "--strategy" not in paper
        assert "models/lgbm_BTC_KRW_4h.lgb" in paper


class TestPipelineMlEndToEnd:
    def test_ml_only_run_trains_validates_and_deploys(self, tmp_path, monkeypatch):
        """ML 전체 흐름 e2e: stage0 실학습 → lgbm + lgbm_prob 템플릿 후보 →
        ml_walk_forward / ml_walk_forward_combined → 배포 아티팩트.

        4h 캔들(월 180개)로 축소 — 단독 실측 ~12초 (tests/CLAUDE.md 2분 예산 내).
        """
        pytest.importorskip("lightgbm")
        from tradingbot.data.storage import save_candles

        monkeypatch.chdir(tmp_path)

        rng = np.random.default_rng(7)
        n = 1400  # ~233 days of 4h candles
        idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
        close = 100 * np.exp(np.linspace(0, 0.4, n) + rng.normal(0, 0.012, n).cumsum())
        df = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.006,
                "low": close * 0.994,
                "close": close,
                "volume": rng.uniform(1, 10, n),
            },
            index=idx,
        )
        save_candles(df, "BTC/KRW", "4h", Path("data"))

        options = pl.PipelineOptions(
            top=2,
            min_trades=0,
            workers=1,
            skip_rules=True,
            ml=True,
            ml_train_months=2,
            ml_test_months=1,
            wf_train_months=2,
            wf_test_months=1,
            ml_tune_thresholds=True,  # 실 스윕 경로 — 재학습 없어 저렴
        )
        templates = [
            {"label": "MLT", "entry": "lgbm_prob:0.35 + trend_up:4", "exit": "rsi_overbought:70"}
        ]
        result = pl.run_pipeline(options, templates=templates, log=lambda m: None)

        run_dir = Path(result["run_dir"])
        stage0 = json.loads((run_dir / "stage0_ml_train.json").read_text())
        assert len(stage0["trained"]) == 1  # smart refresh trained the missing model
        assert stage0["trained"][0]["tuned"] is False
        # threshold sweep: 승자 임계값이 stage0에 기록되고 meta로 영속
        assert "best_entry" in stage0["trained"][0]
        assert (Path("models") / "lgbm_BTC_KRW_4h.lgb").exists()
        meta = json.loads((Path("models") / "lgbm_BTC_KRW_4h_meta.json").read_text())
        assert "entry_threshold" in meta and "exit_threshold" in meta

        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["status"] == "complete"
        assert "ml_train" in manifest["stages"]

        ranking = json.loads((run_dir / "ranking.json").read_text())["ranking"]
        by_name = {r["name"]: r for r in ranking}
        assert by_name["lgbm"]["validation"] == "ml_walk_forward"
        assert by_name["lgbm"]["wf_efficiency"] is None
        # lgbm_prob 템플릿이 시간정직 콤바인 경로로 검증되어 랭킹에 나타남
        assert by_name["MLT"]["validation"] == "ml_walk_forward_combined"
        assert by_name["MLT"]["wf_efficiency"] is None

        wf_doc = json.loads((run_dir / "stage2_walkforward.json").read_text())
        assert all(r["summary"]["num_windows"] >= 1 for r in wf_doc["results"])

        paper = (run_dir / "deploy" / "paper.sh").read_text()
        assert "models/lgbm_BTC_KRW_4h.lgb" in paper  # 어느 승자든 모델 의존 주석
