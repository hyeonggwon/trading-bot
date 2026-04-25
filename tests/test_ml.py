"""Tests for ML module — features, targets, trainer, strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingbot.ml.features import (
    EXTERNAL_FEATURE_COLS,
    FEATURE_COLS,
    WARMUP_CANDLES,
    build_feature_matrix,
)
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
        assert len(feature_cols) == 10
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
        assert compared >= 7, f"Only {compared} columns compared — too many NaN"

    def test_build_feature_matrix_with_external(self):
        """External features should be merged and z-scores computed."""
        df = _make_data(200)
        external_df = pd.DataFrame(
            {
                "kimchi_pct": np.random.normal(0.02, 0.01, 200),
                "funding_rate": np.random.normal(0.0001, 0.00005, 200),
                "fng_value": np.random.uniform(20, 80, 200),
                "usd_krw": 1300 + np.random.normal(0, 10, 200),
            },
            index=df.index,
        )
        df_feat, feature_cols = build_feature_matrix(df, external_df=external_df)

        # Should have 10 technical + 6 external = 16
        assert len(feature_cols) == 16
        for col in FEATURE_COLS:
            assert col in feature_cols
        # Z-scores and derived features
        assert "kimchi_zscore_20" in feature_cols
        assert "funding_rate_zscore_20" in feature_cols
        assert "usd_krw_change" in feature_cols

    def test_build_feature_matrix_without_external(self):
        """Without external data, should return only technical features."""
        df = _make_data(200)
        df_feat, feature_cols = build_feature_matrix(df, external_df=None)
        assert len(feature_cols) == 10
        assert feature_cols == FEATURE_COLS


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
        # Baseline comparison fields
        assert "auc_p_value" in metrics
        assert "auc_significant" in metrics
        assert isinstance(metrics["auc_significant"], bool)

    def test_calibration(self, tmp_path):
        """Calibrator should fit and transform probabilities."""
        from tradingbot.ml.trainer import LGBMTrainer

        df = _make_data(500)
        df_feat, feature_cols = build_feature_matrix(df)
        target = build_target(df_feat)

        mask = df_feat[feature_cols].notna().all(axis=1) & target.notna()
        X, y = df_feat.loc[mask, feature_cols], target[mask]
        split = int(len(X) * 0.7)

        trainer = LGBMTrainer()
        model = trainer.train(X.iloc[:split], y.iloc[:split])

        # Calibrate on test set
        calibrator = trainer.calibrate(model, X.iloc[split:], y.iloc[split:])
        assert calibrator is not None

        # Save and load calibrator
        trainer.save(
            model, "BTC/KRW", "1h", {}, feature_cols,
            model_dir=tmp_path, calibrator=calibrator,
        )
        loaded_cal = LGBMTrainer.load_calibrator("BTC/KRW", "1h", model_dir=tmp_path)
        assert loaded_cal is not None

        # Calibrated probabilities should be valid
        raw = model.predict(X.iloc[split:split + 5])
        calibrated = loaded_cal.transform(raw)
        assert len(calibrated) == 5
        assert all(0 <= p <= 1 for p in calibrated)

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


class TestWalkForward:
    def test_holdout_report(self, tmp_path):
        """Walk-forward should produce holdout metrics."""
        from tradingbot.ml.walk_forward import MLWalkForwardTrainer

        df = _make_data(4000)  # Need enough for holdout split + embargo=150 + windows
        trainer = MLWalkForwardTrainer(
            symbol="BTC/KRW",
            timeframe="1h",
            train_months=1,
            test_months=1,
            model_dir=tmp_path,
        )
        report = trainer.run(df)

        assert len(report.windows) > 0, "Expected at least 1 walk-forward window"
        # Holdout metrics should be populated
        assert report.holdout_auc >= 0
        assert report.holdout_precision >= 0
        # Model should be saved with calibrator
        from tradingbot.ml.trainer import LGBMTrainer

        meta = LGBMTrainer.load_meta("BTC/KRW", "1h", model_dir=tmp_path)
        assert meta is not None
        assert "holdout_auc" in meta
        assert "has_calibrator" in meta
        assert meta["has_calibrator"] is True


class TestMLStrategyWalkForward:
    def test_run_produces_time_honest_windows(self):
        """End-to-end: runner produces multiple windows with train_end < test_start."""
        from tradingbot.config import AppConfig, BacktestConfig, RiskConfig, TradingConfig
        from tradingbot.ml.strategy_walk_forward import MLStrategyWalkForward

        df = _make_data(1500)
        config = AppConfig(
            trading=TradingConfig(
                symbols=["BTC/KRW"], timeframe="4h", initial_balance=1_000_000
            ),
            risk=RiskConfig(),
            backtest=BacktestConfig(),
        )
        runner = MLStrategyWalkForward(
            symbol="BTC/KRW",
            timeframe="4h",
            train_months=2,
            test_months=1,
            entry_threshold=0.30,  # synthetic data — drop below typical 0.60
            exit_threshold=0.25,
            config=config,
        )
        report = runner.run(df)

        assert report.n_windows >= 2, f"expected >=2 windows, got {report.n_windows}"
        assert len(report.windows) == report.n_windows

        for w in report.windows:
            train_end = pd.Timestamp(w["train_end"])
            test_start = pd.Timestamp(w["test_start"])
            assert train_end < test_start, (
                f"window {w['window']}: train_end {train_end} >= test_start {test_start}"
            )
            assert w["n_test"] > 0
            assert -1.0 <= w["return_pct"] / 100 < 100.0  # sanity bounds

    def test_run_is_deterministic(self):
        """Two consecutive runs over the same data must produce identical reports."""
        from tradingbot.config import AppConfig, BacktestConfig, RiskConfig, TradingConfig
        from tradingbot.ml.strategy_walk_forward import MLStrategyWalkForward

        df = _make_data(1500)
        config = AppConfig(
            trading=TradingConfig(
                symbols=["BTC/KRW"], timeframe="4h", initial_balance=1_000_000
            ),
            risk=RiskConfig(),
            backtest=BacktestConfig(),
        )

        def _run() -> list[dict]:
            runner = MLStrategyWalkForward(
                symbol="BTC/KRW",
                timeframe="4h",
                train_months=2,
                test_months=1,
                entry_threshold=0.30,
                exit_threshold=0.25,
                config=config,
            )
            return runner.run(df).windows

        first = _run()
        second = _run()
        assert first == second


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

    def test_set_model_bypasses_file_io(self, tmp_path):
        """set_model() should inject model into strategy without reading from disk."""
        from tradingbot.ml.trainer import LGBMTrainer
        from tradingbot.strategy.base import StrategyParams
        from tradingbot.strategy.lgbm_strategy import LGBMStrategy

        df = _make_data(500)
        df_feat, feature_cols = build_feature_matrix(df.copy())
        target = build_target(df_feat)
        mask = df_feat[feature_cols].notna().all(axis=1) & target.notna()
        X, y = df_feat.loc[mask, feature_cols], target[mask]

        trainer = LGBMTrainer()
        model = trainer.train(X, y)
        # Note: NOT calling trainer.save — tmp_path stays empty.

        strategy = LGBMStrategy(StrategyParams(values={"model_dir": str(tmp_path)}))
        strategy.timeframe = "1h"
        strategy.set_model(
            symbol="BTC/KRW",
            model=model,
            calibrator=None,
            feature_cols=feature_cols,
            win_loss_ratio=2.0,
        )

        # _load_model should short-circuit to the injected model
        assert strategy._load_model("BTC/KRW") is model
        assert strategy._feature_cols["BTC/KRW"] == feature_cols
        assert strategy._win_loss_ratios["BTC/KRW"] == 2.0
        # Calibrator stored as None (not auto-loaded from disk)
        assert strategy._calibrators["BTC/KRW"] is None
        # The strategy must not have read or written anything to model_dir.
        # Verifies set_model truly bypasses file I/O.
        assert list(tmp_path.iterdir()) == []

        # _predict should run without raising
        prob = strategy._predict(df_feat.dropna(subset=feature_cols), "BTC/KRW")
        assert prob is not None
        assert 0.0 <= prob <= 1.0

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

    def test_backtest_with_external_data_dir(self, tmp_path):
        """End-to-end: train 16-feature model, backtest with external_data_dir configured.

        Regression test for a bug where ``build_external_df(external_data_dir)``
        was called with a Path instead of (upbit_df, data_dir), silently
        disabling external features.
        """
        from tradingbot.backtest.engine import BacktestEngine
        from tradingbot.config import AppConfig, BacktestConfig, RiskConfig, TradingConfig
        from tradingbot.data.external_fetcher import save_external
        from tradingbot.ml.trainer import LGBMTrainer
        from tradingbot.strategy.base import StrategyParams
        from tradingbot.strategy.lgbm_strategy import LGBMStrategy

        df = _make_data(500)

        # Persist synthetic external components so build_external_df can load them
        external_dir = tmp_path / "external"
        external_dir.mkdir()
        rng = np.random.default_rng(7)
        # binance BTC/USDT aligned to the upbit cadence
        save_external(
            pd.DataFrame({"close": 45000 + rng.normal(0, 500, len(df))}, index=df.index),
            "binance_btc_usdt",
            external_dir,
        )
        save_external(
            pd.DataFrame({"usd_krw": 1300 + rng.normal(0, 5, len(df))}, index=df.index),
            "usd_krw",
            external_dir,
        )
        save_external(
            pd.DataFrame({"funding_rate": rng.normal(0.0001, 0.00005, len(df))}, index=df.index),
            "funding_rate",
            external_dir,
        )
        save_external(
            pd.DataFrame({"fng_value": rng.uniform(20, 80, len(df))}, index=df.index),
            "fear_greed",
            external_dir,
        )

        # Train 16-feature model (technical + external)
        from tradingbot.data.external_fetcher import build_external_df
        ext_df = build_external_df(df, external_dir)
        assert ext_df is not None and len(ext_df.columns) >= 3

        df_feat, feature_cols = build_feature_matrix(df.copy(), external_df=ext_df)
        target = build_target(df_feat)
        mask = df_feat[feature_cols].notna().all(axis=1) & target.notna()
        X, y = df_feat.loc[mask, feature_cols], target[mask]
        assert len(feature_cols) > 10, "Expected external features in training set"

        trainer = LGBMTrainer()
        model = trainer.train(X, y)
        trainer.save(model, "BTC/KRW", "1h", {}, feature_cols, model_dir=tmp_path)

        # Backtest with external_data_dir — must match the 16-column model
        strategy = LGBMStrategy(StrategyParams(values={
            "model_dir": str(tmp_path),
            "external_data_dir": str(external_dir),
            "entry_threshold": 0.30,
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
        # External components must have loaded — if C1 regression returns, stays None
        assert strategy._external_load_tried is True
        assert strategy._external_components is not None
        assert strategy._external_components.get("binance") is not None
        # Feature cols from meta should include the external features
        assert len(strategy._feature_cols.get("BTC/KRW", [])) > 10

    def test_multi_symbol_external_alignment(self, tmp_path):
        """Each symbol must receive external features aligned to its own index.

        Regression test for H1: previously the strategy cached an external_df
        aligned to the first symbol's timestamps and reused it for all symbols,
        silently misaligning external features on subsequent symbols.
        """
        from tradingbot.data.external_fetcher import (
            align_external_to,
            load_external_components,
            save_external,
        )
        from tradingbot.strategy.base import StrategyParams
        from tradingbot.strategy.lgbm_strategy import LGBMStrategy

        # Two symbols with *disjoint* date ranges — worst case for a shared cache
        idx_a = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
        idx_b = pd.date_range("2024-03-01", periods=200, freq="h", tz="UTC")
        full_idx = idx_a.union(idx_b)
        rng = np.random.default_rng(11)

        external_dir = tmp_path / "external"
        external_dir.mkdir()
        # External data spanning both date ranges
        save_external(
            pd.DataFrame({"close": 45000 + rng.normal(0, 500, len(full_idx))}, index=full_idx),
            "binance_btc_usdt",
            external_dir,
        )
        save_external(
            pd.DataFrame({"usd_krw": 1300 + rng.normal(0, 5, len(full_idx))}, index=full_idx),
            "usd_krw",
            external_dir,
        )
        save_external(
            pd.DataFrame({"funding_rate": rng.normal(0.0001, 0.00005, len(full_idx))}, index=full_idx),
            "funding_rate",
            external_dir,
        )
        save_external(
            pd.DataFrame({"fng_value": rng.uniform(20, 80, len(full_idx))}, index=full_idx),
            "fear_greed",
            external_dir,
        )

        strategy = LGBMStrategy(StrategyParams(values={
            "model_dir": str(tmp_path),
            "external_data_dir": str(external_dir),
        }))
        strategy.timeframe = "1h"

        # Call indicators() for each symbol (as BacktestEngine does)
        df_a = _make_data(200)
        df_a.index = idx_a
        df_b = _make_data(200)
        df_b.index = idx_b

        out_a = strategy.indicators(df_a.copy())
        out_b = strategy.indicators(df_b.copy())

        # Each output's index must match its own symbol's timestamps
        assert out_a.index.equals(df_a.index)
        assert out_b.index.equals(df_b.index)

        # Each output must carry kimchi_pct populated from its OWN date range.
        # Shared-cache regression would leave the later symbol with mostly NaN.
        assert "kimchi_pct" in out_a.columns
        assert "kimchi_pct" in out_b.columns
        assert out_a["kimchi_pct"].notna().sum() > 100, (
            "Symbol A missing kimchi_pct — external alignment broken"
        )
        assert out_b["kimchi_pct"].notna().sum() > 100, (
            "Symbol B missing kimchi_pct — shared-cache regression"
        )

        # Sanity: the two symbols' kimchi series should differ (different underlying
        # upbit close values on different dates → different premium)
        common_len = min(len(out_a), len(out_b))
        a_vals = out_a["kimchi_pct"].dropna().iloc[:common_len].values
        b_vals = out_b["kimchi_pct"].dropna().iloc[:common_len].values
        if len(a_vals) > 0 and len(b_vals) > 0:
            n = min(len(a_vals), len(b_vals))
            assert not np.allclose(a_vals[:n], b_vals[:n]), (
                "Symbols A and B produced identical kimchi_pct — shared aligned df reused"
            )

        # Components cache untouched by alignment (raw only)
        components = strategy._external_components
        assert components is not None
        assert components.get("binance") is not None
        aligned_a = align_external_to(df_a, components)
        aligned_b = align_external_to(df_b, components)
        assert aligned_a is not None and aligned_b is not None
        assert aligned_a.index.equals(df_a.index)
        assert aligned_b.index.equals(df_b.index)
