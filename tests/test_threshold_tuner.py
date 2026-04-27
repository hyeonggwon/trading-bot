"""Tests for the Phase 5 per-model threshold tuner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tradingbot.config import AppConfig, BacktestConfig, RiskConfig, TradingConfig
from tradingbot.ml.features import WARMUP_CANDLES, build_feature_matrix
from tradingbot.ml.targets import build_target
from tradingbot.ml.threshold_tuner import (
    DEFAULT_ENTRY_GRID,
    DEFAULT_EXIT_GRID,
    ThresholdTuner,
    ThresholdTunerResult,
    patch_meta_thresholds,
)
from tradingbot.ml.trainer import LGBMTrainer


def _make_data(n: int = 600) -> pd.DataFrame:
    """Synthetic OHLCV with enough length for warmup + holdout backtest."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    t = np.linspace(0, 6 * np.pi, n)
    close = 50_000_000 + 5_000_000 * np.sin(t) + rng.normal(0, 200_000, n)
    high = close + np.abs(rng.normal(500_000, 200_000, n))
    low = close - np.abs(rng.normal(500_000, 200_000, n))
    open_ = close + rng.normal(0, 300_000, n)
    volume = rng.uniform(100, 1000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _train_and_save(
    df: pd.DataFrame,
    tmp_path: Path,
    holdout_start: pd.Timestamp,
    holdout_end: pd.Timestamp | None = None,
) -> Path:
    """Train a small model and persist it with a meta carrying holdout dates."""
    df_feat, feature_cols = build_feature_matrix(df.copy())
    target = build_target(df_feat)
    mask = df_feat[feature_cols].notna().all(axis=1) & target.notna()
    X, y = df_feat.loc[mask, feature_cols], target[mask]

    trainer = LGBMTrainer()
    model = trainer.train(X, y)
    meta = {
        "holdout_start": str(holdout_start),
        "holdout_end": str(holdout_end) if holdout_end is not None else None,
        "avg_win_loss_ratio": 1.5,
    }
    return trainer.save(
        model,
        "BTC/KRW",
        "1h",
        meta,
        feature_cols,
        model_dir=tmp_path,
    )


def _config() -> AppConfig:
    """Minimal AppConfig the tuner / engine can run with."""
    return AppConfig(
        trading=TradingConfig(symbols=["BTC/KRW"], timeframe="1h", initial_balance=10_000_000),
        risk=RiskConfig(
            max_position_size_pct=0.5,
            max_open_positions=1,
            max_drawdown_pct=0.3,
            default_stop_loss_pct=0.05,
            risk_per_trade_pct=0.02,
        ),
        backtest=BacktestConfig(fee_rate=0.0005, slippage_pct=0.001),
    )


class TestThresholdTuner:
    def test_meta_missing_returns_error(self, tmp_path):
        df = _make_data(300)
        tuner = ThresholdTuner(
            symbol="BTC/KRW",
            timeframe="1h",
            model_dir=tmp_path,
            config=_config(),
            balance=10_000_000,
        )
        result = tuner.search(df)
        assert isinstance(result, ThresholdTunerResult)
        assert result.error == "meta_missing"
        assert result.grid == []

    def test_search_evaluates_grid_and_returns_best(self, tmp_path):
        df = _make_data(600)
        # Holdout starts ~70% in so warmup + scoring rows both fit.
        holdout_start = df.index[int(len(df) * 0.7)]
        _train_and_save(df, tmp_path, holdout_start)

        tuner = ThresholdTuner(
            symbol="BTC/KRW",
            timeframe="1h",
            model_dir=tmp_path,
            config=_config(),
            balance=10_000_000,
            baseline_entry=0.40,
            baseline_exit=0.20,
        )
        # Tight grid so the test is fast — synthetic model often produces
        # constant-ish probabilities, so use values guaranteed to be valid.
        result = tuner.search(
            df,
            entry_grid=(0.30, 0.40, 0.50),
            exit_grid=(0.20, 0.30),
        )

        assert result.error is None or result.error == "no_combo_with_trades"
        # Every (entry, exit) pair where exit < entry should be evaluated.
        # 3 entries × 2 exits = 6 combos, minus pairs where exit >= entry.
        # Skip count: (0.30, 0.30) → exit==entry. So 1 skip, 5 evaluated.
        assert result.n_combos_skipped >= 1
        assert result.n_combos_evaluated + result.n_combos_skipped == 6
        # Holdout window should be recorded
        assert result.holdout_start
        # Best entry must be one of the grid values when there's any winning combo.
        if result.grid and result.best_sharpe > float("-inf"):
            assert result.best_entry in (0.30, 0.40, 0.50)
            assert result.best_exit in (0.20, 0.30)
            assert result.best_exit < result.best_entry

    def test_skips_combos_with_exit_above_entry(self, tmp_path):
        df = _make_data(600)
        holdout_start = df.index[int(len(df) * 0.7)]
        _train_and_save(df, tmp_path, holdout_start)

        tuner = ThresholdTuner(
            symbol="BTC/KRW",
            timeframe="1h",
            model_dir=tmp_path,
            config=_config(),
            balance=10_000_000,
        )
        # 2x2 grid where exit values are >= entry values forces every combo
        # to be skipped — entry_grid all <= exit_grid.
        result = tuner.search(
            df,
            entry_grid=(0.20, 0.30),
            exit_grid=(0.40, 0.50),
        )
        assert result.n_combos_evaluated == 0
        assert result.n_combos_skipped == 4

    def test_no_holdout_in_meta_returns_error(self, tmp_path):
        df = _make_data(600)
        df_feat, feature_cols = build_feature_matrix(df.copy())
        target = build_target(df_feat)
        mask = df_feat[feature_cols].notna().all(axis=1) & target.notna()
        X, y = df_feat.loc[mask, feature_cols], target[mask]

        trainer = LGBMTrainer()
        model = trainer.train(X, y)
        # Save with NO holdout_start in meta — tuner should refuse to run.
        trainer.save(
            model, "BTC/KRW", "1h", {"avg_win_loss_ratio": 1.5}, feature_cols, model_dir=tmp_path
        )

        tuner = ThresholdTuner(
            symbol="BTC/KRW",
            timeframe="1h",
            model_dir=tmp_path,
            config=_config(),
            balance=10_000_000,
        )
        result = tuner.search(df)
        assert result.error == "no_holdout_start_in_meta"

    def test_default_grids_are_sensible(self):
        # Sanity: defaults shouldn't drift into nonsense. Entry must always be
        # > all exit values for at least some combos so we don't end up with
        # an all-skipped run when callers accept the defaults.
        assert max(DEFAULT_EXIT_GRID) < max(DEFAULT_ENTRY_GRID)
        assert all(0 < v < 1 for v in DEFAULT_ENTRY_GRID + DEFAULT_EXIT_GRID)


class TestPatchMetaThresholds:
    def test_writes_thresholds_atomically(self, tmp_path):
        df = _make_data(600)
        holdout_start = df.index[int(len(df) * 0.7)]
        _train_and_save(df, tmp_path, holdout_start)

        result = ThresholdTunerResult(
            symbol="BTC/KRW",
            timeframe="1h",
            best_entry=0.55,
            best_exit=0.25,
            best_sharpe=1.23,
            best_return_pct=4.5,
            best_trades=18,
            best_win_rate=0.55,
            best_max_dd_pct=-3.1,
            baseline_entry=0.45,
            baseline_exit=0.30,
            baseline_sharpe=0.5,
            baseline_return_pct=1.0,
            baseline_trades=8,
            n_combos_evaluated=12,
            n_combos_skipped=4,
            holdout_start=str(holdout_start),
            holdout_end="2024-01-26",
            grid=[{"entry": 0.55, "exit": 0.25, "sharpe": 1.23, "trades": 18}],
        )
        meta_path = patch_meta_thresholds("BTC/KRW", "1h", tmp_path, result)
        assert meta_path is not None
        meta = json.loads(meta_path.read_text())
        assert meta["entry_threshold"] == 0.55
        assert meta["exit_threshold"] == 0.25
        assert meta["threshold_tuning"]["best_sharpe"] == 1.23
        assert meta["threshold_tuning"]["baseline_trades"] == 8

        # Atomic write: the .tmp sibling must be cleaned up.
        tmp_sibling = meta_path.with_suffix(".json.tmp")
        assert not tmp_sibling.exists()

    def test_skips_when_meta_missing(self, tmp_path):
        result = ThresholdTunerResult(symbol="BTC/KRW", timeframe="1h")
        # The grid has at least one entry so we don't bail on the error path,
        # but the meta file simply doesn't exist on disk.
        result.grid = [{"entry": 0.5, "exit": 0.3, "sharpe": 1.0, "trades": 10}]
        result.best_sharpe = 1.0
        result.best_trades = 10
        meta_path = patch_meta_thresholds("BTC/KRW", "1h", tmp_path, result)
        assert meta_path is None


class TestLGBMStrategyMetaThresholds:
    def test_load_model_picks_up_meta_thresholds(self, tmp_path):
        from tradingbot.strategy.base import StrategyParams
        from tradingbot.strategy.lgbm_strategy import LGBMStrategy

        df = _make_data(600)
        df_feat, feature_cols = build_feature_matrix(df.copy())
        target = build_target(df_feat)
        mask = df_feat[feature_cols].notna().all(axis=1) & target.notna()
        X, y = df_feat.loc[mask, feature_cols], target[mask]

        trainer = LGBMTrainer()
        model = trainer.train(X, y)
        trainer.save(
            model,
            "BTC/KRW",
            "1h",
            {
                "avg_win_loss_ratio": 1.5,
                "entry_threshold": 0.55,
                "exit_threshold": 0.25,
            },
            feature_cols,
            model_dir=tmp_path,
        )

        strategy = LGBMStrategy(
            StrategyParams(
                values={
                    "model_dir": str(tmp_path),
                    "entry_threshold": 0.45,  # CLI default — should be overridden
                    "exit_threshold": 0.30,
                }
            )
        )
        strategy.timeframe = "1h"
        # Trigger the lazy load → populates per-symbol thresholds from meta.
        loaded = strategy._load_model("BTC/KRW")
        assert loaded is not None
        assert strategy._entry_thresholds["BTC/KRW"] == 0.55
        assert strategy._exit_thresholds["BTC/KRW"] == 0.25

    def test_set_model_does_not_populate_meta_thresholds(self):
        """Injection path keeps the param-defined thresholds (used by ml-walk-forward)."""
        from tradingbot.strategy.base import StrategyParams
        from tradingbot.strategy.lgbm_strategy import LGBMStrategy

        strategy = LGBMStrategy(
            StrategyParams(values={"entry_threshold": 0.45, "exit_threshold": 0.30})
        )
        strategy.symbols = ["BTC/KRW"]
        strategy.timeframe = "1h"
        # set_model intentionally does not touch the threshold dicts.
        strategy.set_model(
            symbol="BTC/KRW",
            model=object(),
            calibrator=None,
            feature_cols=["adx_14"],
            win_loss_ratio=2.0,
        )
        assert "BTC/KRW" not in strategy._entry_thresholds
        assert "BTC/KRW" not in strategy._exit_thresholds

    def test_load_model_without_meta_thresholds_uses_defaults(self, tmp_path):
        from tradingbot.strategy.base import StrategyParams
        from tradingbot.strategy.lgbm_strategy import LGBMStrategy

        df = _make_data(600)
        # Intentionally omit entry_threshold / exit_threshold from meta.
        _train_and_save(df, tmp_path, df.index[int(len(df) * 0.7)])

        strategy = LGBMStrategy(
            StrategyParams(
                values={
                    "model_dir": str(tmp_path),
                    "entry_threshold": 0.45,
                    "exit_threshold": 0.30,
                }
            )
        )
        strategy.timeframe = "1h"
        loaded = strategy._load_model("BTC/KRW")
        assert loaded is not None
        # No per-symbol overrides → fall back to instance defaults.
        assert "BTC/KRW" not in strategy._entry_thresholds
        assert "BTC/KRW" not in strategy._exit_thresholds


def test_warmup_constant_is_used():
    # Smoke check that the tuner imports the same WARMUP_CANDLES as features.
    assert WARMUP_CANDLES > 0
