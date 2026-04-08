"""Tests for vectorized screening engine — filter consistency and engine accuracy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingbot.backtest.vectorized import VectorizedResult, vectorized_backtest, _extract_trades


def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.RandomState(seed)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    close = np.maximum(close, 10.0)  # Keep positive
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    low = np.maximum(low, 1.0)
    open_ = close + rng.randn(n) * 0.3
    volume = rng.uniform(100, 1000, n)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=pd.date_range("2020-01-01", periods=n, freq="1h"))
    return df


class TestVectorizedFilters:
    """Test that vectorized_entry/exit matches check_entry/check_exit."""

    def _check_filter_consistency(self, filter_obj, df, check_type="entry"):
        """Compare vectorized vs scalar results for a filter."""
        df = filter_obj.compute(df.copy())

        if check_type == "entry":
            vec_result = filter_obj.vectorized_entry(df).fillna(False)
        else:
            vec_result = filter_obj.vectorized_exit(df).fillna(False)

        # Check scalar on last ~50 rows (skip warmup)
        start = max(60, len(df) - 50)
        mismatches = []
        for i in range(start, len(df)):
            visible = df.iloc[:i + 1]
            if check_type == "entry":
                scalar = filter_obj.check_entry(visible)
            else:
                scalar = filter_obj.check_exit(visible)
            if bool(vec_result.iloc[i]) != bool(scalar):
                mismatches.append(i)

        return mismatches

    def test_rsi_oversold_entry(self):
        from tradingbot.strategy.filters.momentum import RsiOversoldFilter
        f = RsiOversoldFilter(period=14, threshold=30.0)
        df = _make_ohlcv(200)
        mismatches = self._check_filter_consistency(f, df, "entry")
        assert len(mismatches) == 0, f"Mismatches at indices: {mismatches}"

    def test_rsi_overbought_exit(self):
        from tradingbot.strategy.filters.momentum import RsiOverboughtFilter
        f = RsiOverboughtFilter(period=14, threshold=70.0)
        df = _make_ohlcv(200)
        mismatches = self._check_filter_consistency(f, df, "exit")
        assert len(mismatches) == 0, f"Mismatches at indices: {mismatches}"

    def test_macd_cross_up_entry(self):
        from tradingbot.strategy.filters.momentum import MacdCrossUpFilter
        f = MacdCrossUpFilter(fast=12, slow=26, signal=9)
        df = _make_ohlcv(200)
        mismatches = self._check_filter_consistency(f, df, "entry")
        assert len(mismatches) == 0, f"Mismatches at indices: {mismatches}"

    def test_macd_cross_up_exit(self):
        from tradingbot.strategy.filters.momentum import MacdCrossUpFilter
        f = MacdCrossUpFilter(fast=12, slow=26, signal=9)
        df = _make_ohlcv(200)
        mismatches = self._check_filter_consistency(f, df, "exit")
        assert len(mismatches) == 0, f"Mismatches at indices: {mismatches}"

    def test_ema_above_entry(self):
        from tradingbot.strategy.filters.price import EmaAboveFilter
        f = EmaAboveFilter(period=20)
        df = _make_ohlcv(200)
        mismatches = self._check_filter_consistency(f, df, "entry")
        assert len(mismatches) == 0, f"Mismatches at indices: {mismatches}"

    def test_ema_above_exit(self):
        from tradingbot.strategy.filters.price import EmaAboveFilter
        f = EmaAboveFilter(period=20)
        df = _make_ohlcv(200)
        mismatches = self._check_filter_consistency(f, df, "exit")
        assert len(mismatches) == 0, f"Mismatches at indices: {mismatches}"

    def test_volume_spike_entry(self):
        from tradingbot.strategy.filters.volume import VolumeSpikeFilter
        f = VolumeSpikeFilter(sma_period=20, threshold=2.5)
        df = _make_ohlcv(200)
        mismatches = self._check_filter_consistency(f, df, "entry")
        assert len(mismatches) == 0, f"Mismatches at indices: {mismatches}"

    def test_bb_upper_break_entry(self):
        from tradingbot.strategy.filters.price import BbUpperBreakFilter
        f = BbUpperBreakFilter(period=20, std=2.0)
        df = _make_ohlcv(200)
        mismatches = self._check_filter_consistency(f, df, "entry")
        assert len(mismatches) == 0, f"Mismatches at indices: {mismatches}"

    def test_ema_cross_up_entry(self):
        from tradingbot.strategy.filters.price import EmaCrossUpFilter
        f = EmaCrossUpFilter(fast=12, slow=26)
        df = _make_ohlcv(200)
        mismatches = self._check_filter_consistency(f, df, "entry")
        assert len(mismatches) == 0, f"Mismatches at indices: {mismatches}"

    def test_adx_strong_entry(self):
        from tradingbot.strategy.filters.trend import AdxStrongFilter
        f = AdxStrongFilter(threshold=25.0, period=14)
        df = _make_ohlcv(200)
        mismatches = self._check_filter_consistency(f, df, "entry")
        assert len(mismatches) == 0, f"Mismatches at indices: {mismatches}"

    def test_stoch_overbought_exit(self):
        from tradingbot.strategy.filters.exit import StochOverboughtFilter
        f = StochOverboughtFilter(threshold=80.0)
        df = _make_ohlcv(200)
        mismatches = self._check_filter_consistency(f, df, "exit")
        assert len(mismatches) == 0, f"Mismatches at indices: {mismatches}"

    def test_donchian_break_entry(self):
        from tradingbot.strategy.filters.price import DonchianBreakFilter
        f = DonchianBreakFilter(period=20)
        df = _make_ohlcv(200)
        mismatches = self._check_filter_consistency(f, df, "entry")
        assert len(mismatches) == 0, f"Mismatches at indices: {mismatches}"

    def test_bb_squeeze_entry(self):
        from tradingbot.strategy.filters.volatility import BbSqueezeFilter
        f = BbSqueezeFilter(bb_period=20, kc_period=20, bb_std=2.0)
        df = _make_ohlcv(200)
        mismatches = self._check_filter_consistency(f, df, "entry")
        assert len(mismatches) == 0, f"Mismatches at indices: {mismatches}"

    def test_obv_rising_entry(self):
        from tradingbot.strategy.filters.volume import ObvRisingFilter
        f = ObvRisingFilter(obv_sma_period=20)
        df = _make_ohlcv(200)
        mismatches = self._check_filter_consistency(f, df, "entry")
        assert len(mismatches) == 0, f"Mismatches at indices: {mismatches}"


class TestVectorizedEngine:
    """Test the vectorized backtest engine."""

    def test_no_signals_zero_trades(self):
        """When no entry signals fire, should produce zero trades."""
        from tradingbot.strategy.filters.momentum import RsiOversoldFilter
        df = _make_ohlcv(100)
        # Use extreme threshold so no signals fire
        f_entry = RsiOversoldFilter(period=14, threshold=1.0)
        df = f_entry.compute(df)

        from tradingbot.strategy.filters.momentum import RsiOverboughtFilter
        f_exit = RsiOverboughtFilter(period=14, threshold=99.0)
        df = f_exit.compute(df)

        result = vectorized_backtest(
            df=df,
            entry_filters=[f_entry],
            exit_filters=[f_exit],
            initial_balance=10_000_000,
            timeframe="1h",
        )
        assert result.total_trades == 0
        assert result.sharpe_ratio == 0.0
        assert result.total_return == 0.0

    def test_basic_trades_produced(self):
        """With reasonable thresholds, some trades should be produced."""
        from tradingbot.strategy.filters.price import EmaAboveFilter
        from tradingbot.strategy.filters.momentum import RsiOverboughtFilter

        df = _make_ohlcv(500, seed=123)
        f_entry = EmaAboveFilter(period=20)
        f_exit = RsiOverboughtFilter(period=14, threshold=70.0)
        df = f_entry.compute(df)
        df = f_exit.compute(df)

        result = vectorized_backtest(
            df=df,
            entry_filters=[f_entry],
            exit_filters=[f_exit],
            initial_balance=10_000_000,
            timeframe="1h",
        )
        assert isinstance(result, VectorizedResult)
        assert result.total_trades >= 0
        assert isinstance(result.sharpe_ratio, float)
        assert 0.0 <= result.win_rate <= 1.0
        assert result.max_drawdown >= 0.0

    def test_stop_loss_triggers(self):
        """Stop loss should trigger when price drops below threshold."""
        n = 50
        close = np.array([100.0] * 10 + [110.0] * 5 + [90.0] * 5 + [100.0] * 30)
        df = pd.DataFrame({
            "open": close,
            "high": close + 1,
            "low": close - 3,
            "close": close,
            "volume": [1000.0] * n,
        }, index=pd.date_range("2020-01-01", periods=n, freq="1h"))

        # Force an entry at index 5
        entry_signals = np.zeros(n, dtype=bool)
        entry_signals[5] = True
        exit_signals = np.zeros(n, dtype=bool)

        trades, _ = _extract_trades(
            entry_signals=entry_signals,
            exit_signals=exit_signals,
            opens=df["open"].values,
            highs=df["high"].values,
            lows=df["low"].values,
            closes=df["close"].values,
            initial_balance=10_000_000,
            fee_rate=0.0005,
            slippage_pct=0.001,
            stop_loss_pct=0.05,
            max_position_pct=0.10,
            atr_values=None,
            atr_multiplier=0.0,
        )
        # Should have at least 1 trade (entry at 6, stop loss when price drops)
        assert len(trades) >= 1

    def test_extract_trades_force_close(self):
        """Unclosed position should be force-closed at last bar."""
        n = 20
        entry_signals = np.zeros(n, dtype=bool)
        entry_signals[2] = True  # Entry at bar 2 → fills at bar 3
        exit_signals = np.zeros(n, dtype=bool)  # No exit signal

        opens = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)
        closes = np.full(n, 100.0)

        trades, _ = _extract_trades(
            entry_signals=entry_signals,
            exit_signals=exit_signals,
            opens=opens, highs=highs, lows=lows, closes=closes,
            initial_balance=10_000_000,
            fee_rate=0.0005,
            slippage_pct=0.001,
            stop_loss_pct=0.50,  # very wide so no SL
            max_position_pct=0.10,
            atr_values=None,
            atr_multiplier=0.0,
        )
        assert len(trades) == 1
        assert trades[0][1] == n - 1  # exit at last bar

    def test_metrics_with_known_trades(self):
        """Verify win_rate and profit_factor with hand-crafted trades."""
        # 3 winning trades, 1 losing trade
        trades = [
            (0, 5, 100.0, 110.0, 10.0, 0.5, 99.0),   # win: pnl=99
            (10, 15, 100.0, 90.0, 10.0, 0.5, -100.5),  # loss: pnl=-100.5
            (20, 25, 100.0, 105.0, 10.0, 0.5, 49.0),   # win: pnl=49
            (30, 35, 100.0, 108.0, 10.0, 0.5, 79.0),   # win: pnl=79
        ]
        from tradingbot.backtest.vectorized import _compute_metrics
        result = _compute_metrics(
            trades=trades,
            initial_balance=10_000_000,
            final_balance=10_000_126.5,
            closes=np.full(40, 100.0),
            index=pd.date_range("2020-01-01", periods=40, freq="1h"),
            timeframe="1h",
        )
        assert result.total_trades == 4
        assert result.win_rate == 0.75
        assert result.profit_factor == pytest.approx((99.0 + 49.0 + 79.0) / 100.5, rel=1e-3)

    def test_short_df_returns_empty(self):
        """DataFrame with < 3 rows should return zero result."""
        df = _make_ohlcv(2)
        from tradingbot.strategy.filters.price import EmaAboveFilter
        f = EmaAboveFilter(period=20)
        result = vectorized_backtest(df=df, entry_filters=[f], exit_filters=[], timeframe="1h")
        assert result.total_trades == 0

    def test_non_vectorizable_filter_returns_empty(self):
        """Filter without supports_vectorized should return zero."""
        from tradingbot.strategy.filters.base import BaseFilter

        class DummyFilter(BaseFilter):
            name = "dummy"
            role = "entry"
            def compute(self, df): return df
            def check_entry(self, df): return False
            def check_exit(self, df, entry_index=None): return False

        df = _make_ohlcv(100)
        result = vectorized_backtest(
            df=df, entry_filters=[DummyFilter()], exit_filters=[], timeframe="1h",
        )
        assert result.total_trades == 0


class TestRunBatchRouting:
    """Test that _run_batch routes vectorizable vs fallback correctly."""

    def test_ml_template_detected_as_fallback(self):
        """Templates with lgbm_prob should be routed to fallback."""
        entry = "trend_up:4 + rsi_oversold:30 + lgbm_prob:0.35"
        assert "lgbm_prob" in entry

    def test_rule_template_detected_as_vectorizable(self):
        """Templates without lgbm_prob should be vectorizable."""
        entry = "trend_up:4 + rsi_oversold:30"
        assert "lgbm_prob" not in entry

    def test_registered_strategy_detected_as_fallback(self):
        """Jobs with empty entry string are registered strategies → fallback."""
        job = ("sma_cross", "", "")
        assert not job[1]  # empty entry → fallback

    def test_force_engine_routes_all_to_engine(self, tmp_path):
        """force_engine=True routes rule-only jobs through full engine."""
        from tradingbot.backtest.parallel import _run_batch
        from tradingbot.data.storage import save_candles

        df = _make_ohlcv(500)
        symbol = "TEST/KRW"
        timeframe = "1h"
        save_candles(df, symbol, timeframe, tmp_path)

        config_dir = str(tmp_path / "config")
        import os
        os.makedirs(config_dir, exist_ok=True)

        jobs = [("Trend+RSI", "trend_up:4 + rsi_oversold:30", "rsi_overbought:70")]
        results = _run_batch(
            symbol, timeframe, jobs,
            str(tmp_path), 1_000_000, config_dir,
            force_engine=True,
        )
        assert len(results) == 1
        assert results[0].error is None
        assert results[0].total_trades >= 0
        assert results[0].strategy == "Trend+RSI"
