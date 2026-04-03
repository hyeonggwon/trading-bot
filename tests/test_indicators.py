from __future__ import annotations

import numpy as np

from tradingbot.data.indicators import (
    add_adx,
    add_aroon,
    add_atr,
    add_bollinger_bands,
    add_cci,
    add_donchian_channel,
    add_ema,
    add_ichimoku,
    add_keltner_channel,
    add_macd,
    add_mfi,
    add_obv,
    add_pct_from_ma,
    add_roc,
    add_rsi,
    add_sma,
    add_stochastic,
    add_volume_sma,
    add_zscore,
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
        assert "bb_upper_5_2.0" in df.columns
        assert "bb_middle_5_2.0" in df.columns
        assert "bb_lower_5_2.0" in df.columns
        # Upper should be > middle > lower
        valid_idx = df["bb_upper_5_2.0"].dropna().index
        assert (df.loc[valid_idx, "bb_upper_5_2.0"] >= df.loc[valid_idx, "bb_middle_5_2.0"]).all()
        assert (df.loc[valid_idx, "bb_middle_5_2.0"] >= df.loc[valid_idx, "bb_lower_5_2.0"]).all()


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


class TestStochastic:
    def test_stochastic_columns(self, sample_df):
        df = add_stochastic(sample_df, k_period=5, d_period=3)
        assert "stoch_k_5" in df.columns
        assert "stoch_d_5_3" in df.columns
        valid = df["stoch_k_5"].dropna()
        assert len(valid) > 0
        assert (valid >= 0).all() and (valid <= 100).all()


class TestADX:
    def test_adx_columns(self, sample_df):
        df = add_adx(sample_df, period=5)
        assert "adx_5" in df.columns
        assert "adx_pos_5" in df.columns
        assert "adx_neg_5" in df.columns
        valid = df["adx_5"].dropna()
        assert len(valid) > 0


class TestIchimoku:
    def test_ichimoku_columns(self, sample_df):
        df = add_ichimoku(sample_df, window1=3, window2=5, window3=7)
        assert "ichi_conv_3_5_7" in df.columns
        assert "ichi_base_3_5_7" in df.columns
        assert "ichi_a_3_5_7" in df.columns
        assert "ichi_b_3_5_7" in df.columns


class TestAroon:
    def test_aroon_columns(self, sample_df):
        df = add_aroon(sample_df, period=5)
        assert "aroon_up_5" in df.columns
        assert "aroon_down_5" in df.columns
        valid = df["aroon_up_5"].dropna()
        assert len(valid) > 0
        assert (valid >= 0).all() and (valid <= 100).all()


class TestCCI:
    def test_cci_exists(self, sample_df):
        df = add_cci(sample_df, period=5)
        assert "cci_5" in df.columns
        assert df["cci_5"].dropna().shape[0] > 0


class TestROC:
    def test_roc_exists(self, sample_df):
        df = add_roc(sample_df, period=3)
        assert "roc_3" in df.columns
        assert df["roc_3"].dropna().shape[0] > 0


class TestMFI:
    def test_mfi_range(self, sample_df):
        df = add_mfi(sample_df, period=5)
        assert "mfi_5" in df.columns
        valid = df["mfi_5"].dropna()
        assert len(valid) > 0
        assert (valid >= 0).all() and (valid <= 100).all()


class TestOBV:
    def test_obv_exists(self, sample_df):
        df = add_obv(sample_df)
        assert "obv" in df.columns
        # OBV should have no NaN values
        assert df["obv"].notna().all()


class TestKeltnerChannel:
    def test_keltner_columns(self, sample_df):
        df = add_keltner_channel(sample_df, period=5, atr_period=3)
        assert "kc_upper_5" in df.columns
        assert "kc_middle_5" in df.columns
        assert "kc_lower_5" in df.columns
        valid_idx = df["kc_upper_5"].dropna().index
        assert len(valid_idx) > 0
        assert (df.loc[valid_idx, "kc_upper_5"] >= df.loc[valid_idx, "kc_lower_5"]).all()


class TestDonchianChannel:
    def test_donchian_columns(self, sample_df):
        df = add_donchian_channel(sample_df, period=5)
        assert "dc_upper_5" in df.columns
        assert "dc_middle_5" in df.columns
        assert "dc_lower_5" in df.columns
        valid_idx = df["dc_upper_5"].dropna().index
        assert len(valid_idx) > 0
        assert (df.loc[valid_idx, "dc_upper_5"] >= df.loc[valid_idx, "dc_lower_5"]).all()


class TestZscore:
    def test_zscore_exists(self, sample_df):
        df = add_zscore(sample_df, period=5)
        assert "zscore_5" in df.columns
        valid = df["zscore_5"].dropna()
        assert len(valid) > 0


class TestPctFromMa:
    def test_pct_from_ma_exists(self, sample_df):
        df = add_pct_from_ma(sample_df, period=5)
        assert "pct_from_ma_5" in df.columns
        valid = df["pct_from_ma_5"].dropna()
        assert len(valid) > 0
