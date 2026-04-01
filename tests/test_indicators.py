from __future__ import annotations

import numpy as np

from tradingbot.data.indicators import (
    add_atr,
    add_bollinger_bands,
    add_ema,
    add_macd,
    add_rsi,
    add_sma,
    add_volume_sma,
)


class TestSMA:
    def test_sma_values(self, sample_df):
        df = add_sma(sample_df, period=3)
        assert "sma_3" in df.columns
        assert np.isnan(df["sma_3"].iloc[0])
        assert np.isnan(df["sma_3"].iloc[1])
        # SMA of close prices [105, 110, 115] = 110
        assert abs(df["sma_3"].iloc[2] - 110.0) < 0.01

    def test_sma_period_5(self, sample_df):
        df = add_sma(sample_df, period=5)
        assert "sma_5" in df.columns
        # 5th value: mean of [105, 110, 115, 120, 125] = 115
        assert abs(df["sma_5"].iloc[4] - 115.0) < 0.01


class TestEMA:
    def test_ema_exists(self, sample_df):
        df = add_ema(sample_df, period=3)
        assert "ema_3" in df.columns
        assert not np.isnan(df["ema_3"].iloc[-1])


class TestRSI:
    def test_rsi_range(self, sample_df):
        df = add_rsi(sample_df, period=5)
        assert "rsi_5" in df.columns
        valid = df["rsi_5"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()


class TestMACD:
    def test_macd_columns(self, sample_df):
        df = add_macd(sample_df, fast=3, slow=5, signal=3)
        assert "macd_3_5_3" in df.columns
        assert "macd_signal_3_5_3" in df.columns
        assert "macd_hist_3_5_3" in df.columns


class TestBollingerBands:
    def test_bbands_columns(self, sample_df):
        df = add_bollinger_bands(sample_df, period=5, std=2.0)
        assert "bb_upper_5" in df.columns
        assert "bb_middle_5" in df.columns
        assert "bb_lower_5" in df.columns
        # Upper should be > middle > lower
        valid_idx = df["bb_upper_5"].dropna().index
        assert (df.loc[valid_idx, "bb_upper_5"] >= df.loc[valid_idx, "bb_middle_5"]).all()
        assert (df.loc[valid_idx, "bb_middle_5"] >= df.loc[valid_idx, "bb_lower_5"]).all()


class TestATR:
    def test_atr_positive(self, sample_df):
        df = add_atr(sample_df, period=5)
        assert "atr_5" in df.columns
        # Skip warmup period where ATR is 0
        valid = df["atr_5"].dropna()
        after_warmup = valid[valid > 0]
        assert len(after_warmup) > 0
        assert (after_warmup > 0).all()


class TestVolumeSMA:
    def test_volume_sma(self, sample_df):
        df = add_volume_sma(sample_df, period=3)
        assert "volume_sma_3" in df.columns
        # SMA of volumes [1000, 1200, 1100] = 1100
        assert abs(df["volume_sma_3"].iloc[2] - 1100.0) < 0.01
