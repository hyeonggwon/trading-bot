"""Smoke tests for ml/parallel.py workers — focus on error-path contracts.

The happy path runs end-to-end Optuna / threshold sweeps and is exercised
by the higher-level CLI tests + manual smoke runs. These tests just pin
the contract that a missing candle file yields a structured result with
``error="no_data"`` rather than a raised exception (otherwise the parent
batch's broad except would still catch it but the user wouldn't see the
specific reason).
"""

from __future__ import annotations

from tradingbot.ml.parallel import (
    ThresholdTunePairResult,
    TunePairResult,
    tune_pair,
    tune_thresholds_pair,
)


class TestDataclassDefaults:
    def test_threshold_pair_defaults(self):
        r = ThresholdTunePairResult(symbol="BTC/KRW", timeframe="4h")
        assert r.symbol == "BTC/KRW"
        assert r.timeframe == "4h"
        assert r.best_sharpe == float("-inf")
        assert r.best_entry == 0.45
        assert r.best_exit == 0.30
        assert r.error is None

    def test_tune_pair_defaults(self):
        r = TunePairResult(symbol="BTC/KRW", timeframe="4h")
        assert r.symbol == "BTC/KRW"
        assert r.timeframe == "4h"
        assert r.best_value == float("-inf")
        assert r.best_params == {}
        assert r.objective == "holdout_sharpe"
        assert r.error is None


class TestTuneThresholdsPairNoData:
    def test_returns_no_data_when_missing_candles(self, tmp_path):
        result = tune_thresholds_pair(
            symbol="ZZZ/KRW",
            timeframe="1h",
            data_dir=str(tmp_path),
            model_dir=str(tmp_path),
            external_data_dir=None,
            entry_grid=(0.45,),
            exit_grid=(0.30,),
            baseline_entry=0.45,
            baseline_exit=0.30,
            balance=1_000_000,
            write_meta=False,
            output_dir=str(tmp_path / "out"),
            label="test",
        )
        assert isinstance(result, ThresholdTunePairResult)
        assert result.error == "no_data"
        assert result.best_sharpe == float("-inf")


class TestTunePairNoData:
    def test_returns_no_data_when_missing_candles(self, tmp_path):
        result = tune_pair(
            symbol="ZZZ/KRW",
            timeframe="1h",
            data_dir=str(tmp_path),
            model_dir=str(tmp_path),
            external_data_dir=None,
            train_months=1,
            test_months=1,
            forward_candles=4,
            threshold=0.006,
            target_kind="binary",
            atr_mult=1.0,
            include_extra=False,
            entry_threshold=0.45,
            exit_threshold=0.30,
            balance=1_000_000,
            trials=1,
            time_budget_sec=10.0,
            objective="holdout_sharpe",
            seed=42,
            output_dir=str(tmp_path / "out"),
            label="test",
            num_threads=1,
            config_dump=None,
        )
        assert isinstance(result, TunePairResult)
        assert result.error == "no_data"
        assert result.best_value == float("-inf")
        assert result.best_params == {}
