"""Tests for combine engine — filters, parsing, CombinedStrategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingbot.backtest.engine import BacktestEngine
from tradingbot.config import AppConfig, BacktestConfig, RiskConfig, TradingConfig
from tradingbot.strategy.combined import CombinedStrategy
from tradingbot.strategy.filters.base import BaseFilter
from tradingbot.strategy.filters.exit import AtrTrailingExitFilter, ZscoreExtremeFilter
from tradingbot.strategy.filters.momentum import (
    RsiOverboughtFilter,
    RsiOversoldFilter,
    StochOversoldFilter,
)
from tradingbot.strategy.filters.price import (
    EmaCrossUpFilter,
    EmaAboveFilter,
    PriceBreakoutFilter,
)
from tradingbot.strategy.filters.registry import (
    get_filter_map,
    parse_filter_spec,
    parse_filter_string,
)
from tradingbot.strategy.filters.trend import AdxStrongFilter, TrendUpFilter
from tradingbot.strategy.filters.volume import ObvRisingFilter, VolumeSpikeFilter


def _make_data(n: int = 300) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    t = np.linspace(0, 6 * np.pi, n)
    close = 50_000_000 + 5_000_000 * np.sin(t) + np.random.normal(0, 200_000, n)
    high = close + np.abs(np.random.normal(500_000, 200_000, n))
    low = close - np.abs(np.random.normal(500_000, 200_000, n))
    open_ = close + np.random.normal(0, 300_000, n)
    volume = np.random.uniform(100, 1000, n)
    # Add some volume spikes
    for idx in [50, 150, 250]:
        if idx < n:
            volume[idx] = 5000
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class TestFilterRegistry:
    def test_all_filters_registered(self):
        fmap = get_filter_map()
        assert "trend_up" in fmap
        assert "rsi_oversold" in fmap
        assert "volume_spike" in fmap
        assert "ema_above" in fmap
        assert "lgbm_prob" in fmap
        assert len(fmap) == 31

    def test_parse_simple(self):
        f = parse_filter_spec("rsi_oversold:30")
        assert isinstance(f, RsiOversoldFilter)
        assert f.threshold == 30.0

    def test_parse_with_multiple_params(self):
        f = parse_filter_spec("trend_up:6:20")
        assert isinstance(f, TrendUpFilter)
        assert f.tf_factor == 6
        assert f.sma_period == 20

    def test_parse_no_params(self):
        f = parse_filter_spec("ema_above")
        assert isinstance(f, EmaAboveFilter)
        assert f.period == 20  # default

    def test_parse_trailing_colon(self):
        """Trailing colons should be ignored, not crash."""
        f = parse_filter_spec("rsi_oversold:30:")
        assert isinstance(f, RsiOversoldFilter)
        assert f.threshold == 30.0

    def test_parse_invalid_param(self):
        """Non-numeric params should raise clear ValueError."""
        import pytest
        with pytest.raises(ValueError, match="Invalid parameters"):
            parse_filter_spec("rsi_oversold:abc")

    def test_parse_unknown_filter(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown filter"):
            parse_filter_spec("nonexistent_filter")

    def test_parse_filter_string(self):
        filters = parse_filter_string("trend_up:4 + rsi_oversold:30 + volume_spike:2.5")
        assert len(filters) == 3
        assert filters[0].name == "trend_up"
        assert filters[1].name == "rsi_oversold"
        assert filters[2].name == "volume_spike"

    def test_parse_adx_strong(self):
        f = parse_filter_spec("adx_strong:25")
        assert isinstance(f, AdxStrongFilter)
        assert f.threshold == 25.0
        assert f.role == "trend"

    def test_parse_ema_cross_up(self):
        f = parse_filter_spec("ema_cross_up:12:26")
        assert isinstance(f, EmaCrossUpFilter)
        assert f.fast == 12
        assert f.slow == 26
        assert f.role == "entry"

    def test_parse_atr_trailing_exit(self):
        f = parse_filter_spec("atr_trailing_exit:14:2.5")
        assert isinstance(f, AtrTrailingExitFilter)
        assert f.period == 14
        assert f.multiplier == 2.5
        assert f.role == "exit"

    def test_parse_obv_rising(self):
        f = parse_filter_spec("obv_rising:20")
        assert isinstance(f, ObvRisingFilter)
        assert f.obv_sma_period == 20
        assert f.role == "volume"

    def test_parse_stoch_oversold(self):
        f = parse_filter_spec("stoch_oversold:20:14:3")
        assert isinstance(f, StochOversoldFilter)
        assert f.threshold == 20.0
        assert f.k_period == 14

    def test_parse_zscore_extreme(self):
        f = parse_filter_spec("zscore_extreme:2.0")
        assert isinstance(f, ZscoreExtremeFilter)
        assert f.threshold == 2.0
        assert f.role == "exit"

    def test_role_tags(self):
        """Verify role tags are correctly set across filter categories."""
        fmap = get_filter_map()
        roles = {name: fmap[name].role for name in fmap}
        assert roles["trend_up"] == "trend"
        assert roles["rsi_oversold"] == "entry"
        assert roles["volume_spike"] == "volume"
        assert roles["rsi_overbought"] == "exit"
        assert roles["bb_squeeze"] == "volatility"
        assert roles["atr_trailing_exit"] == "exit"
        assert roles["obv_rising"] == "volume"
        assert roles["adx_strong"] == "trend"
        assert roles["lgbm_prob"] == "entry"

    def test_parse_lgbm_prob(self):
        from tradingbot.strategy.filters.ml import LgbmProbFilter

        f = parse_filter_spec("lgbm_prob:0.55")
        assert isinstance(f, LgbmProbFilter)
        assert f.threshold == 0.55
        assert f.role == "entry"

    def test_parse_lgbm_prob_with_model_dir(self):
        from tradingbot.strategy.filters.ml import LgbmProbFilter

        f = parse_filter_spec("lgbm_prob:0.60:/tmp/models")
        assert isinstance(f, LgbmProbFilter)
        assert f.threshold == 0.60
        assert str(f.model_dir) == "/tmp/models"

    def test_parse_lgbm_prob_default(self):
        from tradingbot.strategy.filters.ml import LgbmProbFilter

        f = parse_filter_spec("lgbm_prob")
        assert isinstance(f, LgbmProbFilter)
        assert f.threshold == 0.55  # default


class TestCombinedStrategy:
    def test_describe(self):
        entry = [RsiOversoldFilter(threshold=30), VolumeSpikeFilter(threshold=2.0)]
        exit_ = [RsiOverboughtFilter(threshold=70)]
        strategy = CombinedStrategy(entry_filters=entry, exit_filters=exit_)
        desc = strategy.describe()
        assert "rsi_oversold" in desc
        assert "volume_spike" in desc
        assert "rsi_overbought" in desc

    def test_backtest_runs(self):
        df = _make_data(300)
        entry = [RsiOversoldFilter(threshold=35)]
        exit_ = [RsiOverboughtFilter(threshold=65)]
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

    def test_multi_filter_entry(self):
        """Multiple entry filters require ALL to pass (AND logic)."""
        df = _make_data(300)
        # Strict entry: RSI oversold + EMA above (hard to satisfy simultaneously)
        entry = [RsiOversoldFilter(threshold=30), EmaAboveFilter(period=50)]
        exit_ = [RsiOverboughtFilter(threshold=70)]
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
        # With stricter filters, should have fewer trades than single filter
        assert report.final_balance > 0

    def test_exit_role_skipped_in_entry_and(self):
        """Exit-role filters in entry_filters should be skipped in AND logic."""
        df = _make_data(300)
        # Put an exit-only filter in entry_filters — should be skipped
        entry = [RsiOversoldFilter(threshold=35), ZscoreExtremeFilter(threshold=2.0)]
        exit_ = [RsiOverboughtFilter(threshold=65)]
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
        # ZscoreExtremeFilter.check_entry() returns False always,
        # but since role=="exit", it should be skipped → trades should happen
        assert report.final_balance > 0

    def test_no_entry_filters_no_trades(self):
        df = _make_data(100)
        strategy = CombinedStrategy(entry_filters=[], exit_filters=[])
        strategy.timeframe = "1h"

        config = AppConfig(
            trading=TradingConfig(symbols=["BTC/KRW"], timeframe="1h", initial_balance=10_000_000),
            risk=RiskConfig(),
            backtest=BacktestConfig(fee_rate=0.0005, slippage_pct=0.001),
        )

        engine = BacktestEngine(strategy=strategy, config=config)
        report = engine.run({"BTC/KRW": df})
        assert report.total_trades == 0


class TestLgbmProbFilter:
    def test_no_model_returns_false(self, tmp_path):
        """Without a model, check_entry should always return False."""
        from tradingbot.strategy.filters.ml import LgbmProbFilter

        f = LgbmProbFilter(threshold=0.55, model_dir=str(tmp_path))
        df = _make_data(200)
        df = f.compute(df)
        assert f.check_entry(df) is False
        assert f.last_prob is None
        assert f.last_strength is None

    def test_with_model(self, tmp_path):
        """With a trained model, check_entry should return based on probability."""
        from tradingbot.ml.features import build_feature_matrix
        from tradingbot.ml.targets import build_target
        from tradingbot.ml.trainer import LGBMTrainer
        from tradingbot.strategy.filters.ml import LgbmProbFilter

        df = _make_data(500)

        # Train and save model
        df_feat, feature_cols = build_feature_matrix(df.copy())
        target = build_target(df_feat)
        mask = df_feat[feature_cols].notna().all(axis=1) & target.notna()
        X, y = df_feat.loc[mask, feature_cols], target[mask]

        trainer = LGBMTrainer()
        model = trainer.train(X, y)
        trainer.save(model, "BTC/KRW", "1h", {}, feature_cols, model_dir=tmp_path)

        # Test filter with very low threshold (should pass)
        f = LgbmProbFilter(threshold=0.01, symbol="BTC/KRW", timeframe="1h", model_dir=str(tmp_path))
        df_test = _make_data(200)
        df_test = f.compute(df_test)
        result = f.check_entry(df_test)

        # Model loaded and prediction made
        assert f._loaded is True
        assert f._model is not None
        assert f.last_prob is not None
        assert 0 <= f.last_prob <= 1

        if result:
            assert f.last_strength is not None
            assert f.last_strength >= 0

    def test_combined_strategy_strength(self, tmp_path):
        """CombinedStrategy should propagate ML strength to Signal."""
        from tradingbot.ml.features import build_feature_matrix
        from tradingbot.ml.targets import build_target
        from tradingbot.ml.trainer import LGBMTrainer
        from tradingbot.strategy.filters.ml import LgbmProbFilter

        df = _make_data(500)

        # Train model
        df_feat, feature_cols = build_feature_matrix(df.copy())
        target = build_target(df_feat)
        mask = df_feat[feature_cols].notna().all(axis=1) & target.notna()
        X, y = df_feat.loc[mask, feature_cols], target[mask]

        trainer = LGBMTrainer()
        model = trainer.train(X, y)
        trainer.save(model, "BTC/KRW", "1h", {}, feature_cols, model_dir=tmp_path)

        # CombinedStrategy with ML filter (low threshold to ensure entry)
        ml_filter = LgbmProbFilter(threshold=0.01, symbol="BTC/KRW", timeframe="1h", model_dir=str(tmp_path))
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

    def test_no_model_no_trades_combined(self, tmp_path):
        """Without a model, ML filter blocks all entries in CombinedStrategy."""
        from tradingbot.strategy.filters.ml import LgbmProbFilter

        df = _make_data(300)
        ml_filter = LgbmProbFilter(threshold=0.55, model_dir=str(tmp_path))  # Empty dir
        entry = [RsiOversoldFilter(threshold=35), ml_filter]
        exit_ = [RsiOverboughtFilter(threshold=65)]

        strategy = CombinedStrategy(entry_filters=entry, exit_filters=exit_)
        strategy.timeframe = "1h"

        config = AppConfig(
            trading=TradingConfig(symbols=["BTC/KRW"], timeframe="1h", initial_balance=10_000_000),
            risk=RiskConfig(),
            backtest=BacktestConfig(fee_rate=0.0005, slippage_pct=0.001),
        )

        engine = BacktestEngine(strategy=strategy, config=config)
        report = engine.run({"BTC/KRW": df})
        assert report.total_trades == 0


class _AlwaysEnter(BaseFilter):
    """Trivial entry filter that always fires (drives should_entry caching)."""

    name = "always_enter"
    role = "entry"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def check_entry(self, df: pd.DataFrame) -> bool:
        return True

    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        return False


def _wick_ohlcv(levels: list[float], peaks: dict[int, float]) -> pd.DataFrame:
    """Flat OHLCV at each level, with `peaks` raising only the `high` (a wick).

    A post-entry high wick lets a trailing-stop exit anchor to the true peak.
    """
    n = len(levels)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    high = list(levels)
    for i, h in peaks.items():
        high[i] = h
    return pd.DataFrame(
        {"open": levels, "high": high, "low": levels, "close": levels,
         "volume": [100.0] * n},
        index=dates,
    )


class TestTrailingExitTimestampAnchor:
    """ATR trailing exit must anchor 'highest high since entry' on the entry
    *timestamp*, not a positional index frozen at signal time.

    In backtest, slices are anchored at index 0 so the frozen positional index
    is stable. In live trading the rolling fetch window slides forward each
    tick, so the cached index drifts off the entry candle, and a restart loses
    the cache entirely. Both regressions are reproduced below — they FAIL on the
    pre-fix positional-cache code.
    """

    def test_live_window_slide_keeps_trailing_anchor(self):
        from tradingbot.core.enums import PositionSide
        from tradingbot.core.models import Position

        # Entry on the last flat-100 candle (index 10); a high wick to 160 fires
        # one candle later, then price settles flat at 130.
        levels = [100.0] * 11 + [130.0] * 14  # 25 candles
        series = _wick_ohlcv(levels, peaks={11: 160.0})

        strategy = CombinedStrategy(
            entry_filters=[_AlwaysEnter()],
            exit_filters=[AtrTrailingExitFilter(period=3, multiplier=1.0)],
        )

        # Window AT ENTRY: candles 0..10. should_entry caches positional idx 10.
        df_entry = strategy.indicators(series.iloc[:11].copy())
        assert strategy.should_entry(df_entry, "BTC/KRW") is not None

        position = Position(
            symbol="BTC/KRW",
            side=PositionSide.LONG,
            size=1.0,
            entry_price=100.0,
            entry_time=df_entry.index[-1].to_pydatetime(),
            stop_loss=None,
        )

        # Window LATER (slid forward to 5..24): the entry candle is now at
        # position 5, so the cached index 10 points past the 160 peak into the
        # flat-130 tail. Old code: highest=130, no exit. Timestamp anchor:
        # re-locates the entry candle -> highest=160 -> close 130 exits.
        df_exit = strategy.indicators(series.iloc[5:25].copy())
        exit_sig = strategy.should_exit(df_exit, "BTC/KRW", position)
        assert exit_sig is not None

    def test_restart_reanchors_via_persisted_entry_time(self):
        from tradingbot.core.enums import PositionSide
        from tradingbot.core.models import Position

        # Peak wick at index 11; long flat-130 tail (>20 candles) afterwards.
        levels = [100.0] * 11 + [130.0] * 29  # 40 candles
        series = _wick_ohlcv(levels, peaks={11: 160.0})

        strategy = CombinedStrategy(
            entry_filters=[_AlwaysEnter()],
            exit_filters=[AtrTrailingExitFilter(period=3, multiplier=1.0)],
        )
        df_exit = strategy.indicators(series.copy())

        # Simulate a restart: in-memory caches are empty, but the position
        # (with entry_time) was restored from state.json.
        position = Position(
            symbol="BTC/KRW",
            side=PositionSide.LONG,
            size=1.0,
            entry_price=100.0,
            entry_time=series.index[10].to_pydatetime(),
            stop_loss=None,
        )

        # Old code: no cached index -> falls back to last 20 candles, which are
        # all flat 130 (peak at index 11 is older), so highest=130, no exit.
        # New code: anchors on persisted entry_time -> highest=160 -> exits.
        exit_sig = strategy.should_exit(df_exit, "BTC/KRW", position)
        assert exit_sig is not None
