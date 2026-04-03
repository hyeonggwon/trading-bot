"""Tests for ML module — features, targets, trainer, strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingbot.ml.features import FEATURE_COLS, WARMUP_CANDLES, build_feature_matrix
from tradingbot.ml.targets import build_target


def _make_data(n: int = 300) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    t = np.linspace(0, 6 * np.pi, n)
    close = 50_000_000 + 5_000_000 * np.sin(t) + np.random.normal(0, 200_000, n)
    high = close + np.abs(np.random.normal(500_000, 200_000, n))
    low = close - np.abs(np.random.normal(500_000, 200_000, n))
    open_ = close + np.random.normal(0, 300_000, n)
    volume = np.random.uniform(100, 1000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class TestFeatures:
    def test_build_feature_matrix_columns(self):
        df = _make_data(200)
        df_feat, feature_cols = build_feature_matrix(df)
        assert feature_cols == FEATURE_COLS
        assert len(feature_cols) == 36
        for col in feature_cols:
            assert col in df_feat.columns, f"Missing column: {col}"

    def test_features_have_valid_values_after_warmup(self):
        df = _make_data(200)
        df_feat, feature_cols = build_feature_matrix(df)
        # After warmup period, most features should have non-NaN values
        valid = df_feat[feature_cols].iloc[WARMUP_CANDLES + 10 :]
        for col in feature_cols:
            assert valid[col].notna().any(), f"All NaN after warmup: {col}"

    def test_no_future_leakage(self):
        """Features at row N should be identical whether computed on df[:N+1] or df[:N+5]."""
        df = _make_data(200)
        df_full, feature_cols = build_feature_matrix(df.copy())

        # Compute on truncated data (missing last 5 candles)
        df_partial, _ = build_feature_matrix(df.iloc[:-5].copy())

        # Compare row -6 in full vs row -1 in partial (same candle)
        compared = 0
        for col in feature_cols:
            if col in df_partial.columns and col in df_full.columns:
                val_full = df_full[col].iloc[-6]
                val_partial = df_partial[col].iloc[-1]
                if pd.notna(val_full) and pd.notna(val_partial):
                    assert abs(val_full - val_partial) < 1e-6, f"Leakage in {col}"
                    compared += 1
        # Ensure a meaningful number of columns were actually compared
        assert compared >= 20, f"Only {compared} columns compared — too many NaN"


class TestTargets:
    def test_build_target_binary(self):
        df = _make_data(100)
        target = build_target(df, forward_candles=4, threshold=0.006)
        assert set(target.dropna().unique()).issubset({0.0, 1.0})

    def test_build_target_tail_nan(self):
        """Last N rows should be NaN (no future data)."""
        df = _make_data(100)
        target = build_target(df, forward_candles=4)
        # shift(-4) makes last 4 rows NaN
        assert target.iloc[-4:].isna().all()
        # Middle rows should have valid values
        assert target.iloc[10:90].notna().any()

    def test_build_target_custom_threshold(self):
        df = _make_data(200)
        # Very high threshold → fewer positives
        target_high = build_target(df, forward_candles=4, threshold=0.1)
        target_low = build_target(df, forward_candles=4, threshold=0.001)
        assert target_high.sum() <= target_low.sum()


class TestTrainer:
    def test_train_and_predict(self):
        from tradingbot.ml.trainer import LGBMTrainer

        df = _make_data(500)
        df_feat, feature_cols = build_feature_matrix(df)
        target = build_target(df_feat)

        mask = df_feat[feature_cols].notna().all(axis=1) & target.notna()
        X, y = df_feat.loc[mask, feature_cols], target[mask]
        assert len(X) > 100

        split = int(len(X) * 0.7)
        trainer = LGBMTrainer()
        model = trainer.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])

        proba = model.predict(X.iloc[split:])
        assert len(proba) == len(X) - split
        assert all(0 <= p <= 1 for p in proba)

    def test_evaluate_metrics(self):
        from tradingbot.ml.trainer import LGBMTrainer

        df = _make_data(500)
        df_feat, feature_cols = build_feature_matrix(df)
        target = build_target(df_feat)

        mask = df_feat[feature_cols].notna().all(axis=1) & target.notna()
        X, y = df_feat.loc[mask, feature_cols], target[mask]
        split = int(len(X) * 0.7)

        trainer = LGBMTrainer()
        model = trainer.train(X.iloc[:split], y.iloc[:split])
        metrics = trainer.evaluate(model, X.iloc[split:], y.iloc[split:])

        assert "auc" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["auc"] <= 1

    def test_save_and_load(self, tmp_path):
        from tradingbot.ml.trainer import LGBMTrainer

        df = _make_data(300)
        df_feat, feature_cols = build_feature_matrix(df)
        target = build_target(df_feat)

        mask = df_feat[feature_cols].notna().all(axis=1) & target.notna()
        X, y = df_feat.loc[mask, feature_cols], target[mask]

        trainer = LGBMTrainer()
        model = trainer.train(X, y)
        trainer.save(model, "BTC/KRW", "1h", {"test": True}, feature_cols, model_dir=tmp_path)

        loaded = LGBMTrainer.load("BTC/KRW", "1h", model_dir=tmp_path)
        assert loaded is not None

        # Same predictions
        orig_pred = model.predict(X.iloc[:5])
        loaded_pred = loaded.predict(X.iloc[:5])
        np.testing.assert_array_almost_equal(orig_pred, loaded_pred)

        # Metadata
        meta = LGBMTrainer.load_meta("BTC/KRW", "1h", model_dir=tmp_path)
        assert meta is not None
        assert meta["symbol"] == "BTC/KRW"
        assert meta["test"] is True


class TestLGBMStrategy:
    def test_backtest_runs(self, tmp_path):
        """Full pipeline: train → save → backtest with LGBMStrategy."""
        from tradingbot.backtest.engine import BacktestEngine
        from tradingbot.config import AppConfig, BacktestConfig, RiskConfig, TradingConfig
        from tradingbot.ml.trainer import LGBMTrainer
        from tradingbot.strategy.base import StrategyParams
        from tradingbot.strategy.lgbm_strategy import LGBMStrategy

        df = _make_data(500)

        # Train
        df_feat, feature_cols = build_feature_matrix(df.copy())
        target = build_target(df_feat)
        mask = df_feat[feature_cols].notna().all(axis=1) & target.notna()
        X, y = df_feat.loc[mask, feature_cols], target[mask]

        trainer = LGBMTrainer()
        model = trainer.train(X, y)
        trainer.save(model, "BTC/KRW", "1h", {}, feature_cols, model_dir=tmp_path)

        # Backtest
        strategy = LGBMStrategy(StrategyParams(values={
            "model_dir": str(tmp_path),
            "entry_threshold": 0.30,  # Below base rate for synthetic data (model outputs ~0.35)
            "exit_threshold": 0.25,
        }))
        strategy.timeframe = "1h"

        config = AppConfig(
            trading=TradingConfig(symbols=["BTC/KRW"], timeframe="1h", initial_balance=10_000_000),
            risk=RiskConfig(max_position_size_pct=0.5, max_open_positions=1,
                           max_drawdown_pct=0.3, default_stop_loss_pct=0.05, risk_per_trade_pct=0.02),
            backtest=BacktestConfig(fee_rate=0.0005, slippage_pct=0.001),
        )

        engine = BacktestEngine(strategy=strategy, config=config)
        report = engine.run({"BTC/KRW": df})
        assert report.final_balance > 0
        # Note: synthetic data produces flat probabilities (~base rate 0.35),
        # so Half-Kelly yields strength=0 → no trades. This is correct behavior.
        # The test verifies the full pipeline runs without errors.

    def test_lgbm_prob_filter_combined_backtest(self, tmp_path):
        """Full pipeline: train → save → CombinedStrategy with lgbm_prob + rule filters."""
        from tradingbot.backtest.engine import BacktestEngine
        from tradingbot.config import AppConfig, BacktestConfig, RiskConfig, TradingConfig
        from tradingbot.ml.trainer import LGBMTrainer
        from tradingbot.strategy.combined import CombinedStrategy
        from tradingbot.strategy.filters.ml import LgbmProbFilter
        from tradingbot.strategy.filters.momentum import RsiOverboughtFilter, RsiOversoldFilter

        df = _make_data(500)

        # Train
        df_feat, feature_cols = build_feature_matrix(df.copy())
        target = build_target(df_feat)
        mask = df_feat[feature_cols].notna().all(axis=1) & target.notna()
        X, y = df_feat.loc[mask, feature_cols], target[mask]

        trainer = LGBMTrainer()
        model = trainer.train(X, y)
        trainer.save(model, "BTC/KRW", "1h", {}, feature_cols, model_dir=tmp_path)

        # Combined: RSI + ML filter
        ml_filter = LgbmProbFilter(
            threshold=0.01,  # Low threshold for synthetic data
            symbol="BTC/KRW",
            timeframe="1h",
            model_dir=str(tmp_path),
        )
        entry = [RsiOversoldFilter(threshold=40), ml_filter]
        exit_ = [RsiOverboughtFilter(threshold=60)]

        strategy = CombinedStrategy(entry_filters=entry, exit_filters=exit_)
        strategy.timeframe = "1h"

        config = AppConfig(
            trading=TradingConfig(symbols=["BTC/KRW"], timeframe="1h", initial_balance=10_000_000),
            risk=RiskConfig(max_position_size_pct=0.5, max_open_positions=1,
                           max_drawdown_pct=0.3, default_stop_loss_pct=0.05, risk_per_trade_pct=0.02),
            backtest=BacktestConfig(fee_rate=0.0005, slippage_pct=0.001),
        )

        engine = BacktestEngine(strategy=strategy, config=config)
        report = engine.run({"BTC/KRW": df})
        assert report.final_balance > 0

    def test_no_model_no_trades(self, tmp_path):
        """Without a model file, strategy should generate no trades."""
        from tradingbot.backtest.engine import BacktestEngine
        from tradingbot.config import AppConfig, BacktestConfig, RiskConfig, TradingConfig
        from tradingbot.strategy.base import StrategyParams
        from tradingbot.strategy.lgbm_strategy import LGBMStrategy

        df = _make_data(300)

        strategy = LGBMStrategy(StrategyParams(values={
            "model_dir": str(tmp_path),  # Empty dir — no model
        }))
        strategy.timeframe = "1h"

        config = AppConfig(
            trading=TradingConfig(symbols=["BTC/KRW"], timeframe="1h", initial_balance=10_000_000),
            risk=RiskConfig(),
            backtest=BacktestConfig(fee_rate=0.0005, slippage_pct=0.001),
        )

        engine = BacktestEngine(strategy=strategy, config=config)
        report = engine.run({"BTC/KRW": df})
        assert report.total_trades == 0
