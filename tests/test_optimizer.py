"""Tests for the grid search optimizer and walk-forward validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingbot.backtest.optimizer import (
    GridSearchOptimizer,
    generate_param_combinations,
)
from tradingbot.backtest.walk_forward import (
    WalkForwardValidator,
    create_walk_forward_windows,
)
from tradingbot.config import AppConfig, BacktestConfig, RiskConfig, TradingConfig
from tradingbot.strategy.examples.sma_cross import SmaCrossStrategy


def _make_long_data(months: int = 12) -> pd.DataFrame:
    """Generate synthetic data spanning several months for optimization tests."""
    np.random.seed(42)
    hours = months * 30 * 24
    dates = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")

    # Multiple sine waves for varied regimes
    t = np.linspace(0, 8 * np.pi, hours)
    base = 50_000_000 + 5_000_000 * np.sin(t) + 2_000_000 * np.sin(3 * t)
    noise = np.random.normal(0, 200_000, hours)
    close = base + noise

    high = close + np.abs(np.random.normal(500_000, 200_000, hours))
    low = close - np.abs(np.random.normal(500_000, 200_000, hours))
    open_ = close + np.random.normal(0, 300_000, hours)
    volume = np.random.uniform(100, 1000, hours)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _make_config() -> AppConfig:
    return AppConfig(
        trading=TradingConfig(symbols=["BTC/KRW"], timeframe="1h", initial_balance=10_000_000),
        risk=RiskConfig(
            max_position_size_pct=0.5,
            max_open_positions=1,
            max_drawdown_pct=0.30,
            default_stop_loss_pct=0.05,
            risk_per_trade_pct=0.02,
        ),
        backtest=BacktestConfig(fee_rate=0.0005, slippage_pct=0.001),
    )


class TestParamCombinations:
    def test_empty_space(self):
        combos = generate_param_combinations({})
        assert combos == [{}]

    def test_single_param(self):
        combos = generate_param_combinations({"a": [1, 2, 3]})
        assert len(combos) == 3
        assert combos[0] == {"a": 1}

    def test_two_params(self):
        combos = generate_param_combinations({"a": [1, 2], "b": [10, 20]})
        assert len(combos) == 4
        assert {"a": 1, "b": 10} in combos
        assert {"a": 2, "b": 20} in combos


class TestGridSearchOptimizer:
    def test_basic_optimization(self):
        """Grid search should return results sorted by Sharpe."""
        df = _make_long_data(6)
        config = _make_config()

        optimizer = GridSearchOptimizer(
            strategy_cls=SmaCrossStrategy,
            config=config,
            max_workers=1,
        )
        results = optimizer.optimize(
            {config.trading.symbols[0]: df},
            param_space={"fast_period": [5, 10], "slow_period": [20, 30]},
            sort_by="sharpe_ratio",
        )

        assert len(results) == 4  # 2 x 2 combinations
        # Results should be sorted by Sharpe (descending)
        for i in range(len(results) - 1):
            assert results[i].sharpe_ratio >= results[i + 1].sharpe_ratio

    def test_results_have_params(self):
        """Each result should contain the parameters used."""
        df = _make_long_data(4)
        config = _make_config()

        optimizer = GridSearchOptimizer(
            strategy_cls=SmaCrossStrategy,
            config=config,
            max_workers=1,
        )
        results = optimizer.optimize(
            {config.trading.symbols[0]: df},
            param_space={"fast_period": [10], "slow_period": [30]},
        )

        assert len(results) == 1
        assert results[0].params == {"fast_period": 10, "slow_period": 30}

    def test_to_dataframe(self):
        """Results should convert to a valid DataFrame."""
        df = _make_long_data(4)
        config = _make_config()

        optimizer = GridSearchOptimizer(
            strategy_cls=SmaCrossStrategy,
            config=config,
            max_workers=1,
        )
        results = optimizer.optimize(
            {config.trading.symbols[0]: df},
            param_space={"fast_period": [5, 10], "slow_period": [20, 30]},
        )

        results_df = optimizer.results_to_dataframe(results)
        assert len(results_df) == 4
        assert "fast_period" in results_df.columns
        assert "sharpe_ratio" in results_df.columns


class TestWalkForwardWindows:
    def test_window_creation_expanding_with_embargo(self):
        """Expanding train + 150-candle embargo gap (ML 프레임과 동일 구조)."""
        from tradingbot.backtest.walk_forward import EMBARGO_CANDLES

        dates = pd.date_range("2024-01-01", periods=365 * 24, freq="h", tz="UTC")
        df = pd.DataFrame({"close": range(len(dates))}, index=dates)

        windows = create_walk_forward_windows(df, train_months=3, test_months=1)
        assert len(windows) >= 3

        for train_start, train_end, test_start, test_end in windows:
            # Expanding: train always pinned at the first candle
            assert train_start == df.index[0]
            # Embargo: exactly EMBARGO_CANDLES rows between train_end and test_start
            pos = int(df.index.searchsorted(train_end))
            assert df.index[pos + EMBARGO_CANDLES] == test_start
            assert train_start < train_end < test_start < test_end
            assert test_end <= df.index.max()

        # Train grows by one test span per window
        train_ends = [w[1] for w in windows]
        assert train_ends[1] == train_ends[0] + pd.DateOffset(months=1)

    def test_embargo_single_source(self):
        """룰/ML 윈도우가 같은 embargo 정본을 공유한다 (드리프트 방지)."""
        from tradingbot.backtest import walk_forward as rule_wf
        from tradingbot.ml import walk_forward as ml_wf

        assert rule_wf.EMBARGO_CANDLES == ml_wf.EMBARGO_CANDLES == 150

    def test_insufficient_data(self):
        """Too little data should produce no windows."""
        dates = pd.date_range("2024-01-01", periods=30 * 24, freq="h", tz="UTC")
        df = pd.DataFrame({"close": range(len(dates))}, index=dates)

        windows = create_walk_forward_windows(df, train_months=3, test_months=1)
        assert len(windows) == 0

    def test_data_gap_never_scores_same_test_window_twice(self):
        """test span보다 큰 데이터 공백에서도 같은 test 창을 중복 채점하지 않는다."""
        first = pd.date_range("2024-01-01", periods=120 * 24, freq="h", tz="UTC")
        second = pd.date_range("2024-07-15", periods=120 * 24, freq="h", tz="UTC")
        idx = first.append(second)  # ~2.5개월 공백
        df = pd.DataFrame({"close": range(len(idx))}, index=idx)

        windows = create_walk_forward_windows(df, train_months=3, test_months=1)
        test_starts = [w[2] for w in windows]
        assert len(test_starts) == len(set(test_starts))
        assert test_starts == sorted(test_starts)


class TestWalkForwardValidator:
    def test_basic_validation(self):
        """Walk-forward should produce a report with windows."""
        df = _make_long_data(12)
        config = _make_config()

        validator = WalkForwardValidator(
            strategy_cls=SmaCrossStrategy,
            config=config,
            train_months=3,
            test_months=1,
        )
        report = validator.validate(
            {config.trading.symbols[0]: df},
            param_space={"fast_period": [5, 10], "slow_period": [20, 30]},
        )

        assert report.num_windows >= 1
        assert report.strategy_name == "sma_cross"

        # Each window should have valid data
        for w in report.windows:
            assert w.train_start < w.train_end
            assert w.test_start < w.test_end
            assert len(w.best_params) > 0

    def test_report_metrics(self):
        """Report should compute aggregate metrics."""
        df = _make_long_data(12)
        config = _make_config()

        validator = WalkForwardValidator(
            strategy_cls=SmaCrossStrategy,
            config=config,
            train_months=3,
            test_months=1,
        )
        report = validator.validate(
            {config.trading.symbols[0]: df},
            param_space={"fast_period": [5, 10], "slow_period": [20, 30]},
        )

        assert isinstance(report.walk_forward_efficiency, float)
        assert isinstance(report.overfitting_ratio, float)
        assert isinstance(report.cumulative_test_return, float)
        assert isinstance(report.avg_test_sharpe, float)
