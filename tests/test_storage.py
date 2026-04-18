from __future__ import annotations

import pandas as pd
import pytest

from tradingbot.core.models import candles_to_dataframe
from tradingbot.data.storage import (
    detect_gaps,
    get_parquet_path,
    list_available_data,
    load_candles,
    save_candles,
)


class TestParquetPath:
    def test_symbol_conversion(self, tmp_data_dir):
        path = get_parquet_path("BTC/KRW", "1h", tmp_data_dir)
        assert "BTC_KRW" in str(path)
        assert path.name == "1h.parquet"


class TestSaveLoad:
    def test_save_and_load(self, sample_df, tmp_data_dir):
        save_candles(sample_df, "BTC/KRW", "1h", tmp_data_dir)
        loaded = load_candles("BTC/KRW", "1h", tmp_data_dir)

        assert len(loaded) == len(sample_df)
        pd.testing.assert_frame_equal(loaded, sample_df)

    def test_merge_on_save(self, sample_df, tmp_data_dir):
        # Save first half
        first_half = sample_df.iloc[:5]
        save_candles(first_half, "BTC/KRW", "1h", tmp_data_dir)

        # Save second half (overlapping last candle)
        second_half = sample_df.iloc[4:]
        save_candles(second_half, "BTC/KRW", "1h", tmp_data_dir)

        loaded = load_candles("BTC/KRW", "1h", tmp_data_dir)
        assert len(loaded) == 10  # No duplicates

    def test_load_nonexistent(self, tmp_data_dir):
        with pytest.raises(FileNotFoundError):
            load_candles("XRP/KRW", "1h", tmp_data_dir)


class TestListData:
    def test_list_empty(self, tmp_data_dir):
        result = list_available_data(tmp_data_dir)
        assert result == []

    def test_list_after_save(self, sample_df, tmp_data_dir):
        save_candles(sample_df, "BTC/KRW", "1h", tmp_data_dir)
        result = list_available_data(tmp_data_dir)
        assert len(result) == 1
        assert result[0]["symbol"] == "BTC/KRW"
        assert result[0]["timeframe"] == "1h"

    def test_list_excludes_external(self, sample_df, tmp_data_dir):
        """external/ holds ML feature data (FNG, funding rate, etc.), not tradeable symbols."""
        save_candles(sample_df, "BTC/KRW", "1h", tmp_data_dir)
        ext_dir = tmp_data_dir / "external"
        ext_dir.mkdir()
        pd.DataFrame({"fng_value": [30.0, 15.0]}).to_parquet(ext_dir / "fear_greed.parquet")

        result = list_available_data(tmp_data_dir)
        symbols = {r["symbol"] for r in result}
        assert symbols == {"BTC/KRW"}


class TestDetectGaps:
    def test_no_gaps(self):
        """Continuous hourly data should have no gaps."""
        dates = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({"close": range(24)}, index=dates)
        gaps = detect_gaps(df, "1h")
        assert gaps == []

    def test_with_gap(self):
        """Missing candles should be detected as gaps."""
        dates = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        # Remove hours 5-8 (4 missing candles)
        df = pd.DataFrame({"close": range(24)}, index=dates)
        df = df.drop(dates[5:9])
        gaps = detect_gaps(df, "1h")
        assert len(gaps) == 1
        assert gaps[0][0] == dates[5]
        assert gaps[0][1] == dates[8]

    def test_multiple_gaps(self):
        """Multiple separate gaps should be detected."""
        dates = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({"close": range(24)}, index=dates)
        # Remove hours 3-4 and 15-16
        df = df.drop(dates[3:5])
        df = df.drop(dates[15:17])
        gaps = detect_gaps(df, "1h")
        assert len(gaps) == 2

    def test_empty_df(self):
        df = pd.DataFrame({"close": []})
        gaps = detect_gaps(df, "1h")
        assert gaps == []

    def test_single_row(self):
        dates = pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC")
        df = pd.DataFrame({"close": [100]}, index=dates)
        gaps = detect_gaps(df, "1h")
        assert gaps == []
