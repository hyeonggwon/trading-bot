"""Smoke tests for LGBMTuner — validates the wiring without long runtimes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingbot.ml.tuner import LGBMTuner, LGBMTunerResult


def _synthetic_ohlcv(n: int = 1500) -> pd.DataFrame:
    """Generate enough hourly candles for one strategy walk-forward window.

    The trainer needs train_months * 720 + test_months * 720 + warmup at
    minimum. Use 1500 candles to comfortably cover a 1m/1m WF on 1h.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    t = np.linspace(0, 12 * np.pi, n)
    close = 50_000_000 + 5_000_000 * np.sin(t) + rng.normal(0, 200_000, n)
    high = close + np.abs(rng.normal(500_000, 200_000, n))
    low = close - np.abs(rng.normal(500_000, 200_000, n))
    open_ = close + rng.normal(0, 300_000, n)
    volume = rng.uniform(100, 1000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class TestLGBMTuner:
    def test_invalid_objective_raises(self):
        with pytest.raises(ValueError):
            LGBMTuner(symbol="BTC/KRW", timeframe="1h", objective="bogus")

    def test_search_returns_best_params(self):
        df = _synthetic_ohlcv(1500)
        tuner = LGBMTuner(
            symbol="BTC/KRW",
            timeframe="1h",
            train_months=1,
            test_months=1,
            objective="holdout_sharpe",
            seed=7,
        )
        # Two trials with a tight time budget so the test stays fast.
        result = tuner.search(df, n_trials=2, time_budget_sec=120.0)
        assert isinstance(result, LGBMTunerResult)
        assert result.n_trials_completed >= 1
        assert isinstance(result.best_params, dict)
        # Best params (when found) include the tuned keys
        if result.best_params:
            for key in (
                "num_leaves",
                "max_depth",
                "min_data_in_leaf",
                "reg_alpha",
                "reg_lambda",
                "feature_fraction",
                "bagging_fraction",
                "learning_rate",
                "n_estimators",
            ):
                assert key in result.best_params, f"missing tuned key: {key}"

    def test_trial_log_records_each_trial(self):
        df = _synthetic_ohlcv(1500)
        tuner = LGBMTuner(
            symbol="BTC/KRW",
            timeframe="1h",
            train_months=1,
            test_months=1,
            seed=11,
        )
        result = tuner.search(df, n_trials=2, time_budget_sec=120.0)
        # Trials list must mirror the count Optuna ran (errors are still logged
        # because we catch=Exception in optimize).
        assert len(result.trials) == result.n_trials_completed
        for t in result.trials:
            assert "trial" in t
            assert "score" in t
            assert "params" in t
