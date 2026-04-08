"""Tests for CLI helper functions — template lookup, strategy resolution, combined building,
parallel batch worker, and walk-forward combined."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingbot.cli import (
    COMBINE_TEMPLATES,
    _build_combined_strategy,
    _find_combine_template,
    _resolve_strategy,
)
from tradingbot.strategy.filters.registry import parse_filter_string


class TestFindCombineTemplate:
    def test_exact_match(self):
        result = _find_combine_template("Trend+RSI")
        assert result is not None
        assert result["label"] == "Trend+RSI"

    def test_case_insensitive(self):
        r1 = _find_combine_template("trend+rsi")
        r2 = _find_combine_template("TREND+RSI")
        r3 = _find_combine_template("Trend+RSI")
        assert r1 == r2 == r3

    def test_unknown_returns_none(self):
        assert _find_combine_template("NonExistentStrategy") is None

    def test_empty_string_returns_none(self):
        assert _find_combine_template("") is None

    def test_partial_match_returns_none(self):
        assert _find_combine_template("Trend") is None


class TestBuildCombinedStrategy:
    def test_returns_combined(self):
        from tradingbot.strategy.combined import CombinedStrategy

        strategy = _build_combined_strategy(
            "rsi_oversold:30", "rsi_overbought:70", "BTC/KRW", "1h",
        )
        assert isinstance(strategy, CombinedStrategy)
        assert len(strategy.entry_filters) == 1
        assert len(strategy.exit_filters) == 1

    def test_sets_strategy_attrs(self):
        strategy = _build_combined_strategy(
            "rsi_oversold:30", "rsi_overbought:70", "ETH/KRW", "4h",
        )
        assert strategy.symbols == ["ETH/KRW"]
        assert strategy.timeframe == "4h"

    def test_sets_ml_filter_symbol_timeframe(self):
        strategy = _build_combined_strategy(
            "rsi_oversold:30 + lgbm_prob:0.35", "rsi_overbought:70", "SOL/KRW", "1h",
        )
        ml_filter = [f for f in strategy.entry_filters if f.name == "lgbm_prob"][0]
        assert ml_filter.symbol == "SOL/KRW"
        assert ml_filter.timeframe == "1h"

    def test_invalid_filter_raises(self):
        with pytest.raises(ValueError):
            _build_combined_strategy(
                "nonexistent_filter:30", "rsi_overbought:70", "BTC/KRW", "1h",
            )


class TestResolveStrategy:
    def test_registered_strategy(self):
        strategy, name, cls = _resolve_strategy("sma_cross", "BTC/KRW", "1h")
        assert name == "sma_cross"
        assert cls is not None
        assert strategy.timeframe == "1h"

    def test_combine_template(self):
        strategy, name, cls = _resolve_strategy("Trend+RSI", "BTC/KRW", "1h")
        assert name == "Trend+RSI"
        assert cls is None  # combined templates have no class

    def test_combine_template_case_insensitive(self):
        _, name, _ = _resolve_strategy("trend+rsi", "BTC/KRW", "1h")
        assert name == "Trend+RSI"

    def test_unknown_raises_exit(self):
        from click.exceptions import Exit

        with pytest.raises((SystemExit, Exit)):
            _resolve_strategy("NonExistentStrategy", "BTC/KRW", "1h")

    def test_sets_symbols(self):
        strategy, _, _ = _resolve_strategy(
            "Trend+RSI", "BTC/KRW", "1h", symbols=["BTC/KRW", "ETH/KRW"],
        )
        assert strategy.symbols == ["BTC/KRW", "ETH/KRW"]

    def test_sets_timeframe(self):
        strategy, _, _ = _resolve_strategy("ML+TrendEMA", "BTC/KRW", "4h")
        assert strategy.timeframe == "4h"

    def test_ml_template_rejects_multi_symbol(self):
        from click.exceptions import Exit

        with pytest.raises((SystemExit, Exit)):
            _resolve_strategy(
                "ML+TrendEMA", "BTC/KRW", "1h",
                symbols=["BTC/KRW", "ETH/KRW"],
            )

    def test_non_ml_template_allows_multi_symbol(self):
        strategy, _, _ = _resolve_strategy(
            "Trend+RSI", "BTC/KRW", "1h",
            symbols=["BTC/KRW", "ETH/KRW"],
        )
        assert strategy.symbols == ["BTC/KRW", "ETH/KRW"]


class TestCombineTemplates:
    def test_all_templates_parse_successfully(self):
        """Every template's entry and exit strings must parse without error."""
        for tmpl in COMBINE_TEMPLATES:
            try:
                parse_filter_string(tmpl["entry"])
                parse_filter_string(tmpl["exit"])
            except Exception as e:
                pytest.fail(f"Template '{tmpl['label']}' failed to parse: {e}")

    def test_unique_labels(self):
        labels = [t["label"].lower() for t in COMBINE_TEMPLATES]
        assert len(labels) == len(set(labels)), f"Duplicate labels: {[l for l in labels if labels.count(l) > 1]}"

    def test_required_keys(self):
        for tmpl in COMBINE_TEMPLATES:
            assert "entry" in tmpl, f"Missing 'entry' in {tmpl}"
            assert "exit" in tmpl, f"Missing 'exit' in {tmpl}"
            assert "label" in tmpl, f"Missing 'label' in {tmpl}"

    def test_ml_templates_use_threshold_035(self):
        """All ML templates should use 0.35 threshold (veto mode)."""
        for tmpl in COMBINE_TEMPLATES:
            entry = tmpl["entry"]
            if "lgbm_prob" in entry:
                assert "lgbm_prob:0.35" in entry, (
                    f"Template '{tmpl['label']}' uses wrong ML threshold: {entry}"
                )


# ── Helpers for batch worker and walk-forward tests ──


def _make_cyclic_df(n: int = 500) -> pd.DataFrame:
    """Create synthetic OHLCV data with cyclic price pattern."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    t = np.linspace(0, 6 * np.pi, n)
    base = 50_000_000 + 5_000_000 * np.sin(t)
    noise = np.random.normal(0, 200_000, n)
    close = base + noise
    high = close + np.abs(np.random.normal(500_000, 200_000, n))
    low = close - np.abs(np.random.normal(500_000, 200_000, n))
    open_ = close + np.random.normal(0, 300_000, n)
    volume = np.random.uniform(100, 1000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _make_config():
    from tradingbot.config import AppConfig, BacktestConfig, RiskConfig, TradingConfig

    return AppConfig(
        trading=TradingConfig(symbols=["BTC/KRW"], timeframe="1h", initial_balance=10_000_000),
        risk=RiskConfig(
            max_position_size_pct=0.5, max_open_positions=1,
            max_drawdown_pct=0.30, default_stop_loss_pct=0.05, risk_per_trade_pct=0.02,
        ),
        backtest=BacktestConfig(fee_rate=0.0005, slippage_pct=0.001),
    )


class TestRunBatch:
    """Tests for the _run_batch parallel worker."""

    def test_basic_scan_returns_results(self, tmp_path):
        """_run_batch should return one ScanResult per job."""
        from tradingbot.backtest.parallel import _run_batch

        # Save test data as parquet
        df = _make_cyclic_df(300)
        data_dir = tmp_path / "data" / "BTC_KRW"
        data_dir.mkdir(parents=True)
        df.to_parquet(data_dir / "1h.parquet")

        # Write minimal config
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "default.yaml").write_text(
            "exchange:\n  name: upbit\ntrading:\n  symbols: ['BTC/KRW']\n"
            "  timeframe: '1h'\n  initial_balance: 10000000\n"
            "risk:\n  max_position_size_pct: 0.5\n  max_open_positions: 1\n"
            "  max_drawdown_pct: 0.3\n  default_stop_loss_pct: 0.05\n  risk_per_trade_pct: 0.02\n"
            "backtest:\n  fee_rate: 0.0005\n  slippage_pct: 0.001\n"
        )

        jobs = [
            ("sma_cross", "", ""),
            ("bollinger_breakout", "", ""),
        ]
        results = _run_batch(
            "BTC/KRW", "1h", jobs,
            str(data_dir.parent), 10_000_000, str(config_dir),
        )

        assert len(results) == 2
        assert results[0].strategy == "sma_cross"
        assert results[1].strategy == "bollinger_breakout"
        assert all(r.error is None for r in results)
        assert all(r.symbol == "BTC/KRW" for r in results)
        assert all(r.timeframe == "1h" for r in results)

    def test_combined_filters_in_batch(self, tmp_path):
        """_run_batch should handle combined filter jobs."""
        from tradingbot.backtest.parallel import _run_batch

        df = _make_cyclic_df(300)
        data_dir = tmp_path / "data" / "BTC_KRW"
        data_dir.mkdir(parents=True)
        df.to_parquet(data_dir / "1h.parquet")

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "default.yaml").write_text(
            "exchange:\n  name: upbit\ntrading:\n  symbols: ['BTC/KRW']\n"
            "  timeframe: '1h'\n  initial_balance: 10000000\n"
            "risk:\n  max_position_size_pct: 0.5\n  max_open_positions: 1\n"
            "  max_drawdown_pct: 0.3\n  default_stop_loss_pct: 0.05\n  risk_per_trade_pct: 0.02\n"
            "backtest:\n  fee_rate: 0.0005\n  slippage_pct: 0.001\n"
        )

        jobs = [
            ("Trend+RSI", "trend_up:4 + rsi_oversold:30", "rsi_overbought:70"),
            ("BB+Vol", "bb_upper_break:20 + volume_spike:2.0", "ema_above:20"),
        ]
        results = _run_batch(
            "BTC/KRW", "1h", jobs,
            str(data_dir.parent), 10_000_000, str(config_dir),
        )

        assert len(results) == 2
        assert results[0].entry == "trend_up:4 + rsi_oversold:30"
        assert results[1].entry == "bb_upper_break:20 + volume_spike:2.0"
        assert all(r.error is None for r in results)

    def test_no_data_returns_errors(self, tmp_path):
        """_run_batch should return error results when data file missing."""
        from tradingbot.backtest.parallel import _run_batch

        jobs = [("sma_cross", "", "")]
        results = _run_batch(
            "BTC/KRW", "1h", jobs,
            str(tmp_path / "nodata"), 10_000_000,
        )

        assert len(results) == 1
        assert results[0].error == "no data"

    def test_invalid_strategy_returns_error(self, tmp_path):
        """_run_batch should catch errors for invalid strategies."""
        from tradingbot.backtest.parallel import _run_batch

        df = _make_cyclic_df(300)
        data_dir = tmp_path / "data" / "BTC_KRW"
        data_dir.mkdir(parents=True)
        df.to_parquet(data_dir / "1h.parquet")

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "default.yaml").write_text(
            "exchange:\n  name: upbit\ntrading:\n  symbols: ['BTC/KRW']\n"
            "  timeframe: '1h'\n  initial_balance: 10000000\n"
            "risk:\n  max_position_size_pct: 0.5\n  max_open_positions: 1\n"
            "  max_drawdown_pct: 0.3\n  default_stop_loss_pct: 0.05\n  risk_per_trade_pct: 0.02\n"
            "backtest:\n  fee_rate: 0.0005\n  slippage_pct: 0.001\n"
        )

        jobs = [
            ("sma_cross", "", ""),
            ("nonexistent_strategy", "", ""),
        ]
        results = _run_batch(
            "BTC/KRW", "1h", jobs,
            str(data_dir.parent), 10_000_000, str(config_dir),
        )

        assert len(results) == 2
        assert results[0].error is None  # sma_cross should succeed
        assert results[1].error is not None  # nonexistent should fail

    def test_df_not_mutated_across_jobs(self, tmp_path):
        """Each job should get a fresh copy of data."""
        from tradingbot.backtest.parallel import _run_batch

        df = _make_cyclic_df(300)
        data_dir = tmp_path / "data" / "BTC_KRW"
        data_dir.mkdir(parents=True)
        df.to_parquet(data_dir / "1h.parquet")

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "default.yaml").write_text(
            "exchange:\n  name: upbit\ntrading:\n  symbols: ['BTC/KRW']\n"
            "  timeframe: '1h'\n  initial_balance: 10000000\n"
            "risk:\n  max_position_size_pct: 0.5\n  max_open_positions: 1\n"
            "  max_drawdown_pct: 0.3\n  default_stop_loss_pct: 0.05\n  risk_per_trade_pct: 0.02\n"
            "backtest:\n  fee_rate: 0.0005\n  slippage_pct: 0.001\n"
        )

        # Run same strategy twice — results should be identical
        jobs = [("sma_cross", "", ""), ("sma_cross", "", "")]
        results = _run_batch(
            "BTC/KRW", "1h", jobs,
            str(data_dir.parent), 10_000_000, str(config_dir),
        )

        assert results[0].total_trades == results[1].total_trades
        assert results[0].sharpe_ratio == results[1].sharpe_ratio
        assert results[0].total_return == results[1].total_return


class TestWalkForwardCombined:
    """Tests for _walk_forward_combined (combined template walk-forward)."""

    def test_runs_without_error(self):
        """Walk-forward combined should complete and print results."""
        from tradingbot.cli import _walk_forward_combined

        strategy = _build_combined_strategy(
            "rsi_oversold:30", "rsi_overbought:70", "BTC/KRW", "1h",
        )
        df = _make_cyclic_df(2000)  # Need enough data for multiple windows
        config = _make_config()

        # Should not raise
        _walk_forward_combined(
            strategy, "TestStrategy", "BTC/KRW", df, config,
            train_months=2, test_months=1,
        )

    def test_insufficient_data_handled(self):
        """Walk-forward combined should handle too-short data gracefully."""
        from tradingbot.cli import _walk_forward_combined

        strategy = _build_combined_strategy(
            "rsi_oversold:30", "rsi_overbought:70", "BTC/KRW", "1h",
        )
        df = _make_cyclic_df(50)  # Too short for any windows
        config = _make_config()

        # Should not raise, just print warning
        _walk_forward_combined(
            strategy, "TestStrategy", "BTC/KRW", df, config,
            train_months=3, test_months=1,
        )
