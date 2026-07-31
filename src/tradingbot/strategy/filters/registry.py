"""Filter registry — maps filter names to classes."""

from __future__ import annotations

from collections.abc import Callable

from tradingbot.strategy.filters.base import BaseFilter


def get_filter_map() -> dict[str, type[BaseFilter]]:
    """Return map of filter name → class."""
    from tradingbot.strategy.filters.exit import (
        AtrTrailingExitFilter,
        CciOverboughtFilter,
        MfiOverboughtFilter,
        PctFromMaExitFilter,
        StochOverboughtFilter,
        TimeStopExitFilter,
        ZscoreExtremeFilter,
    )
    from tradingbot.strategy.filters.ml import LgbmProbFilter
    from tradingbot.strategy.filters.momentum import (
        CciOversoldFilter,
        MacdCrossUpFilter,
        MfiOversoldFilter,
        RocPositiveFilter,
        RsiOverboughtFilter,
        RsiOversoldFilter,
        StochOversoldFilter,
    )
    from tradingbot.strategy.filters.price import (
        BbUpperBreakFilter,
        DonchianBreakFilter,
        EmaAboveFilter,
        EmaCrossUpFilter,
        PriceBreakoutFilter,
    )
    from tradingbot.strategy.filters.session import SessionKstFilter
    from tradingbot.strategy.filters.trend import (
        AdxStrongFilter,
        AroonUpFilter,
        IchimokuAboveFilter,
        TrendDownFilter,
        TrendUpFilter,
    )
    from tradingbot.strategy.filters.volatility import (
        AtrBreakoutFilter,
        BbBandwidthLowFilter,
        BbSqueezeFilter,
        KeltnerBreakFilter,
        RealizedVolHighFilter,
        RealizedVolLowFilter,
    )
    from tradingbot.strategy.filters.volume import (
        MfiConfirmFilter,
        ObvRisingFilter,
        VolumeSpikeFilter,
    )

    return {
        # Trend filters
        "trend_up": TrendUpFilter,
        "trend_down": TrendDownFilter,
        "adx_strong": AdxStrongFilter,
        "ichimoku_above": IchimokuAboveFilter,
        "aroon_up": AroonUpFilter,
        # Entry signals
        "rsi_oversold": RsiOversoldFilter,
        "macd_cross_up": MacdCrossUpFilter,
        "stoch_oversold": StochOversoldFilter,
        "cci_oversold": CciOversoldFilter,
        "roc_positive": RocPositiveFilter,
        "mfi_oversold": MfiOversoldFilter,
        "ema_cross_up": EmaCrossUpFilter,
        "donchian_break": DonchianBreakFilter,
        "price_breakout": PriceBreakoutFilter,
        "ema_above": EmaAboveFilter,
        "bb_upper_break": BbUpperBreakFilter,
        # Volatility filters
        "atr_breakout": AtrBreakoutFilter,
        "keltner_break": KeltnerBreakFilter,
        "bb_squeeze": BbSqueezeFilter,
        "bb_bandwidth_low": BbBandwidthLowFilter,
        "realized_vol_low": RealizedVolLowFilter,
        "realized_vol_high": RealizedVolHighFilter,
        # Session gate
        "session_kst": SessionKstFilter,
        # Volume confirm
        "volume_spike": VolumeSpikeFilter,
        "obv_rising": ObvRisingFilter,
        "mfi_confirm": MfiConfirmFilter,
        # Exit signals
        "rsi_overbought": RsiOverboughtFilter,
        "stoch_overbought": StochOverboughtFilter,
        "cci_overbought": CciOverboughtFilter,
        "mfi_overbought": MfiOverboughtFilter,
        "zscore_extreme": ZscoreExtremeFilter,
        "pct_from_ma_exit": PctFromMaExitFilter,
        "atr_trailing_exit": AtrTrailingExitFilter,
        "time_stop": TimeStopExitFilter,
        # ML filter
        "lgbm_prob": LgbmProbFilter,
    }


def parse_filter_spec(spec: str, base_timeframe: str = "1h") -> BaseFilter:
    """Parse a filter spec string like 'trend_up:4' or 'rsi_oversold:30'.

    Format: 'filter_name' or 'filter_name:param' or 'filter_name:p1:p2'
    """
    filter_map = get_filter_map()
    parts = spec.strip().split(":")
    name = parts[0]

    if name not in filter_map:
        raise ValueError(f"Unknown filter: {name}. Available: {', '.join(sorted(filter_map))}")

    filter_cls = filter_map[name]
    kwargs: dict[str, int | float | str] = {}

    # Filter empty parts from trailing colons
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError("Empty filter spec")
    name = parts[0]

    try:
        _parse_filter_params(name, parts, kwargs, base_timeframe)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid parameters for '{spec}': {e}")

    return filter_cls(**kwargs)


_FILTER_ARGS: dict[str, list[tuple[str, Callable[[str], int | float | str]]]] = {
    "trend_up": [("tf_factor", int), ("sma_period", int)],
    "trend_down": [("tf_factor", int), ("sma_period", int)],
    "rsi_oversold": [("threshold", float), ("period", int)],
    "rsi_overbought": [("threshold", float), ("period", int)],
    "macd_cross_up": [("fast", int), ("slow", int), ("signal", int)],
    "volume_spike": [("threshold", float), ("sma_period", int)],
    "price_breakout": [("lookback", int)],
    "ema_above": [("period", int)],
    "bb_upper_break": [("period", int), ("std", float)],
    "stoch_oversold": [("threshold", float), ("k_period", int), ("d_period", int)],
    "stoch_overbought": [("threshold", float), ("k_period", int), ("d_period", int)],
    "cci_oversold": [("threshold", float), ("period", int)],
    "cci_overbought": [("threshold", float), ("period", int)],
    "roc_positive": [("period", int)],
    "mfi_oversold": [("threshold", float), ("period", int)],
    "mfi_overbought": [("threshold", float), ("period", int)],
    "mfi_confirm": [("threshold", float), ("period", int)],
    "ema_cross_up": [("fast", int), ("slow", int)],
    "donchian_break": [("period", int)],
    "adx_strong": [("threshold", float), ("period", int)],
    "ichimoku_above": [("window1", int), ("window2", int), ("window3", int)],
    "aroon_up": [("threshold", float), ("period", int)],
    "atr_breakout": [("period", int), ("multiplier", float), ("ema_period", int)],
    "keltner_break": [("period", int), ("atr_period", int)],
    "bb_squeeze": [("bb_period", int), ("kc_period", int)],
    "bb_bandwidth_low": [("threshold", float), ("period", int)],
    "realized_vol_low": [("threshold", float), ("vol_period", int), ("rank_period", int)],
    "realized_vol_high": [("threshold", float), ("vol_period", int), ("rank_period", int)],
    "session_kst": [("start_hour", int), ("end_hour", int)],
    "obv_rising": [("obv_sma_period", int)],
    "zscore_extreme": [("threshold", float), ("period", int)],
    "pct_from_ma_exit": [("period", int), ("threshold", float)],
    "atr_trailing_exit": [("period", int), ("multiplier", float)],
    "time_stop": [("max_bars", int)],
    "lgbm_prob": [("threshold", float), ("model_dir", str)],
}


def _parse_filter_params(
    name: str, parts: list[str], kwargs: dict[str, int | float | str], base_timeframe: str
) -> None:
    """Parse filter-specific parameters into kwargs dict."""
    for (key, conv), raw in zip(_FILTER_ARGS.get(name, []), parts[1:]):
        kwargs[key] = conv(raw)
    if name in ("trend_up", "trend_down"):
        kwargs["base_timeframe"] = base_timeframe


def parse_filter_string(filter_string: str, base_timeframe: str = "1h") -> list[BaseFilter]:
    """Parse a combined filter string like 'trend_up:4 + rsi_oversold:30'."""
    specs = [s.strip() for s in filter_string.split("+")]
    return [parse_filter_spec(s, base_timeframe) for s in specs if s]
