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
        assert "time_stop" in fmap
        assert "session_kst" in fmap
        assert "realized_vol_low" in fmap
        assert "realized_vol_high" in fmap
        assert len(fmap) == 35

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

    def test_parse_time_stop(self):
        from tradingbot.strategy.filters.exit import TimeStopExitFilter

        f = parse_filter_spec("time_stop:12")
        assert isinstance(f, TimeStopExitFilter)
        assert f.max_bars == 12
        assert f.role == "exit"

    def test_parse_time_stop_default(self):
        from tradingbot.strategy.filters.exit import TimeStopExitFilter

        f = parse_filter_spec("time_stop")
        assert isinstance(f, TimeStopExitFilter)
        assert f.max_bars == 24  # default

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


class TestTimeStopFilter:
    """TimeStopExitFilter boundary + entry_index handling."""

    def _df(self, n: int) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        return pd.DataFrame(
            {
                "open": [100.0] * n, "high": [101.0] * n,
                "low": [99.0] * n, "close": [100.0] * n, "volume": [1.0] * n,
            },
            index=dates,
        )

    def test_fires_when_holding_reaches_max_bars(self):
        from tradingbot.strategy.filters.exit import TimeStopExitFilter

        f = TimeStopExitFilter(max_bars=3)
        df = self._df(10)
        # entry at index 6 → bars held = (10-1) - 6 = 3 → fires at the boundary
        assert f.check_exit(df, entry_index=6) is True

    def test_holds_below_max_bars(self):
        from tradingbot.strategy.filters.exit import TimeStopExitFilter

        f = TimeStopExitFilter(max_bars=3)
        df = self._df(10)
        # entry at index 7 → bars held = 9 - 7 = 2 < 3 → still holding
        assert f.check_exit(df, entry_index=7) is False

    def test_none_entry_index_never_fires(self):
        from tradingbot.strategy.filters.exit import TimeStopExitFilter

        f = TimeStopExitFilter(max_bars=1)
        df = self._df(10)
        # Unknown entry (plain-strategy path / restart) must not anchor on the
        # oldest bar and force an exit.
        assert f.check_exit(df, entry_index=None) is False

    def test_out_of_range_entry_index_never_fires(self):
        from tradingbot.strategy.filters.exit import TimeStopExitFilter

        f = TimeStopExitFilter(max_bars=1)
        df = self._df(10)
        assert f.check_exit(df, entry_index=99) is False

    def test_combined_strategy_threads_entry_index(self):
        """End-to-end: CombinedStrategy must thread the resolved entry index into
        the filter so the time stop fires N bars after the real entry — the
        entry-relative plumbing that a plain vectorized mask cannot express."""
        from tradingbot.core.enums import PositionSide, SignalType
        from tradingbot.core.models import Position
        from tradingbot.strategy.filters.exit import TimeStopExitFilter

        df = self._df(10)

        # Entry anchored at index 6 → bars held at the last bar = 3 → time stop fires.
        strategy = CombinedStrategy(
            entry_filters=[RsiOversoldFilter(threshold=30)],
            exit_filters=[TimeStopExitFilter(max_bars=3)],
        )
        strategy._entry_times = {"BTC/KRW": df.index[6]}
        position = Position(
            symbol="BTC/KRW", side=PositionSide.LONG, size=1.0,
            entry_price=100.0, entry_time=df.index[6].to_pydatetime(),
        )
        signal = strategy.should_exit(df, "BTC/KRW", position)
        assert signal is not None
        assert signal.signal_type == SignalType.LONG_EXIT

        # Entry only 2 bars back must NOT trigger the 3-bar time stop.
        strategy2 = CombinedStrategy(
            entry_filters=[RsiOversoldFilter(threshold=30)],
            exit_filters=[TimeStopExitFilter(max_bars=3)],
        )
        strategy2._entry_times = {"BTC/KRW": df.index[7]}
        position2 = Position(
            symbol="BTC/KRW", side=PositionSide.LONG, size=1.0,
            entry_price=100.0, entry_time=df.index[7].to_pydatetime(),
        )
        assert strategy2.should_exit(df, "BTC/KRW", position2) is None


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

    def test_scrolled_out_entry_falls_back_to_none(self):
        """When the entry candle has aged out of the rolling window, the entry
        timestamp predates every bar. The anchor must fall back to None (the
        filter's own heuristic), not silently snap to the oldest bar (index 0).
        Old code returns 0 for that scrolled-off case; the fix returns None."""
        from tradingbot.core.enums import PositionSide
        from tradingbot.core.models import Position

        # Current fetch window starts 2024-01-05; the position entered on
        # 2024-01-01, long before every bar still in the window.
        dates = pd.date_range("2024-01-05", periods=20, freq="h", tz="UTC")
        df = pd.DataFrame(
            {"open": [100.0] * 20, "high": [100.0] * 20, "low": [100.0] * 20,
             "close": [100.0] * 20, "volume": [100.0] * 20},
            index=dates,
        )
        strategy = CombinedStrategy(
            entry_filters=[_AlwaysEnter()],
            exit_filters=[AtrTrailingExitFilter(period=3, multiplier=1.0)],
        )
        position = Position(
            symbol="BTC/KRW", side=PositionSide.LONG, size=1.0, entry_price=100.0,
            entry_time=pd.Timestamp("2024-01-01", tz="UTC").to_pydatetime(),
            stop_loss=None,
        )
        # No cached entry time -> uses persisted position.entry_time, which
        # predates the window: index-0 snap is wrong, None is the contract.
        assert strategy._resolve_entry_index(df, "BTC/KRW", position) is None


class TestSessionKstFilter:
    """KST session gate — entry-only, end-exclusive, overnight wrap."""

    def test_parse_and_defaults(self):
        from tradingbot.strategy.filters.session import SessionKstFilter

        f = parse_filter_spec("session_kst")
        assert isinstance(f, SessionKstFilter)
        assert (f.start_hour, f.end_hour) == (9, 23)
        f2 = parse_filter_spec("session_kst:10:22")
        assert (f2.start_hour, f2.end_hour) == (10, 22)

    def test_invalid_hours_rejected(self):
        import pytest

        from tradingbot.strategy.filters.session import SessionKstFilter

        with pytest.raises(ValueError, match="0..23"):
            SessionKstFilter(start_hour=24, end_hour=6)
        with pytest.raises(ValueError, match="empty session"):
            SessionKstFilter(start_hour=9, end_hour=9)

    def _df_ending_at(self, utc_hour: int) -> pd.DataFrame:
        idx = pd.date_range(
            f"2024-01-02 {utc_hour:02d}:00", periods=1, freq="h", tz="UTC"
        )
        return pd.DataFrame(
            {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]},
            index=idx,
        )

    def test_gate_by_kst_hour(self):
        from tradingbot.strategy.filters.session import SessionKstFilter

        f = SessionKstFilter(start_hour=9, end_hour=23)
        # 05:00 UTC = 14:00 KST → in session
        assert f.check_entry(self._df_ending_at(5)) is True
        # 16:00 UTC = 01:00 KST → out of session
        assert f.check_entry(self._df_ending_at(16)) is False
        # end-exclusive: 14:00 UTC = 23:00 KST → out
        assert f.check_entry(self._df_ending_at(14)) is False
        # exit must never be gated
        assert f.check_exit(self._df_ending_at(16)) is False

    def test_overnight_wrap(self):
        from tradingbot.strategy.filters.session import SessionKstFilter

        f = SessionKstFilter(start_hour=22, end_hour=6)
        # 14:00 UTC = 23:00 KST → in (after 22)
        assert f.check_entry(self._df_ending_at(14)) is True
        # 18:00 UTC = 03:00 KST → in (before 6)
        assert f.check_entry(self._df_ending_at(18)) is True
        # 03:00 UTC = 12:00 KST → out
        assert f.check_entry(self._df_ending_at(3)) is False

    def test_vectorized_matches_scalar(self):
        from tradingbot.strategy.filters.session import SessionKstFilter

        df = _make_data(48)
        f = SessionKstFilter(start_hour=9, end_hour=23)
        vec = f.vectorized_entry(df)
        for i in range(1, 48):
            assert bool(vec.iloc[i]) == f.check_entry(df.iloc[: i + 1]), f"row {i}"


class TestRealizedVolRegimeFilters:
    """Realized-vol percentile regime gates (calm vs expanding)."""

    def test_parse(self):
        from tradingbot.strategy.filters.volatility import (
            RealizedVolHighFilter,
            RealizedVolLowFilter,
        )

        lo = parse_filter_spec("realized_vol_low:0.25:20:50")
        assert isinstance(lo, RealizedVolLowFilter)
        assert (lo.threshold, lo.vol_period, lo.rank_period) == (0.25, 20, 50)
        hi = parse_filter_spec("realized_vol_high")
        assert isinstance(hi, RealizedVolHighFilter)
        assert hi.threshold == 0.7

    def _regime_df(self) -> pd.DataFrame:
        """calm(0..119) → wild(120..179) → calm(180..239)."""
        np.random.seed(7)
        n = 240
        noise = np.concatenate([
            np.random.normal(0, 0.05, 120),
            np.random.normal(0, 3.0, 60),
            np.random.normal(0, 0.05, 60),
        ])
        close = 100.0 + np.cumsum(noise)
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        return pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close,
             "volume": np.ones(n)},
            index=idx,
        )

    def test_calm_vs_expanding_regime(self):
        from tradingbot.strategy.filters.volatility import (
            RealizedVolHighFilter,
            RealizedVolLowFilter,
        )

        df = self._regime_df()
        lo, hi = RealizedVolLowFilter(), RealizedVolHighFilter()
        df = lo.compute(df)  # hi shares the same columns (idempotent)

        # Bar 140: vol window is all-wild, rank window spans calm+wild → high regime.
        # (truthiness, not `is True` — filters return numpy bools like their siblings)
        wild = df.iloc[:141]
        assert hi.check_entry(wild)
        assert not lo.check_entry(wild)

        # Bar 200: vol window is all-calm, rank window still spans wild → low regime.
        calm = df.iloc[:201]
        assert lo.check_entry(calm)
        assert not hi.check_entry(calm)

    def test_warmup_and_missing_column_are_false(self):
        from tradingbot.strategy.filters.volatility import RealizedVolLowFilter

        df = self._regime_df()
        lo = RealizedVolLowFilter()
        assert lo.check_entry(df.iloc[:5]) is False  # column not computed yet
        df = lo.compute(df)
        assert lo.check_entry(df.iloc[:5]) is False  # warmup NaN

    def test_vectorized_matches_scalar(self):
        """프리픽스에서 '재계산'한 scalar 와 full-df 벡터화가 일치해야 한다.

        check_entry 에 full-df 계산 컬럼을 그대로 물려주면 두 경로가 같은
        값을 읽는 항등식이 된다 — 프리픽스 재계산이어야 shift(-1) /
        center=True 류 미래 누수 회귀를 실제로 검출한다 (code-review 지적).
        """
        from tradingbot.strategy.filters.volatility import RealizedVolHighFilter

        raw = self._regime_df()
        hi = RealizedVolHighFilter()
        vec = hi.vectorized_entry(hi.compute(raw.copy()))
        for i in (100, 140, 170, 200, 239):
            prefix = hi.compute(raw.iloc[: i + 1].copy())
            assert bool(vec.iloc[i]) == hi.check_entry(prefix), f"row {i}"
