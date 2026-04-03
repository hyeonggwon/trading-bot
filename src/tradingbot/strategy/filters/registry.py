"""Filter registry — maps filter names to classes."""

from __future__ import annotations

from tradingbot.strategy.filters.base import BaseFilter


def get_filter_map() -> dict[str, type[BaseFilter]]:
    """Return map of filter name → class."""
    from tradingbot.strategy.filters.exit import (
        AtrTrailingExitFilter,
        CciOverboughtFilter,
        MfiOverboughtFilter,
        PctFromMaExitFilter,
        StochOverboughtFilter,
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
    kwargs: dict = {}

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


def _parse_filter_params(
    name: str, parts: list[str], kwargs: dict, base_timeframe: str
) -> None:
    """Parse filter-specific parameters into kwargs dict."""
    if name in ("trend_up", "trend_down"):
        if len(parts) >= 2:
            kwargs["tf_factor"] = int(parts[1])
        if len(parts) >= 3:
            kwargs["sma_period"] = int(parts[2])
        kwargs["base_timeframe"] = base_timeframe

    elif name == "rsi_oversold":
        if len(parts) >= 2:
            kwargs["threshold"] = float(parts[1])
        if len(parts) >= 3:
            kwargs["period"] = int(parts[2])

    elif name == "rsi_overbought":
        if len(parts) >= 2:
            kwargs["threshold"] = float(parts[1])
        if len(parts) >= 3:
            kwargs["period"] = int(parts[2])

    elif name == "macd_cross_up":
        if len(parts) >= 2:
            kwargs["fast"] = int(parts[1])
        if len(parts) >= 3:
            kwargs["slow"] = int(parts[2])
        if len(parts) >= 4:
            kwargs["signal"] = int(parts[3])

    elif name == "volume_spike":
        if len(parts) >= 2:
            kwargs["threshold"] = float(parts[1])
        if len(parts) >= 3:
            kwargs["sma_period"] = int(parts[2])

    elif name == "price_breakout":
        if len(parts) >= 2:
            kwargs["lookback"] = int(parts[1])

    elif name == "ema_above":
        if len(parts) >= 2:
            kwargs["period"] = int(parts[1])

    elif name == "bb_upper_break":
        if len(parts) >= 2:
            kwargs["period"] = int(parts[1])
        if len(parts) >= 3:
            kwargs["std"] = float(parts[2])

    # ── New filters ──

    elif name in ("stoch_oversold", "stoch_overbought"):
        if len(parts) >= 2:
            kwargs["threshold"] = float(parts[1])
        if len(parts) >= 3:
            kwargs["k_period"] = int(parts[2])
        if len(parts) >= 4:
            kwargs["d_period"] = int(parts[3])

    elif name in ("cci_oversold", "cci_overbought"):
        if len(parts) >= 2:
            kwargs["threshold"] = float(parts[1])
        if len(parts) >= 3:
            kwargs["period"] = int(parts[2])

    elif name == "roc_positive":
        if len(parts) >= 2:
            kwargs["period"] = int(parts[1])

    elif name in ("mfi_oversold", "mfi_overbought"):
        if len(parts) >= 2:
            kwargs["threshold"] = float(parts[1])
        if len(parts) >= 3:
            kwargs["period"] = int(parts[2])

    elif name == "mfi_confirm":
        if len(parts) >= 2:
            kwargs["threshold"] = float(parts[1])
        if len(parts) >= 3:
            kwargs["period"] = int(parts[2])

    elif name == "ema_cross_up":
        if len(parts) >= 2:
            kwargs["fast"] = int(parts[1])
        if len(parts) >= 3:
            kwargs["slow"] = int(parts[2])

    elif name == "donchian_break":
        if len(parts) >= 2:
            kwargs["period"] = int(parts[1])

    elif name == "adx_strong":
        if len(parts) >= 2:
            kwargs["threshold"] = float(parts[1])
        if len(parts) >= 3:
            kwargs["period"] = int(parts[2])

    elif name == "ichimoku_above":
        if len(parts) >= 2:
            kwargs["window1"] = int(parts[1])
        if len(parts) >= 3:
            kwargs["window2"] = int(parts[2])
        if len(parts) >= 4:
            kwargs["window3"] = int(parts[3])

    elif name == "aroon_up":
        if len(parts) >= 2:
            kwargs["threshold"] = float(parts[1])
        if len(parts) >= 3:
            kwargs["period"] = int(parts[2])

    elif name == "atr_breakout":
        if len(parts) >= 2:
            kwargs["period"] = int(parts[1])
        if len(parts) >= 3:
            kwargs["multiplier"] = float(parts[2])
        if len(parts) >= 4:
            kwargs["ema_period"] = int(parts[3])

    elif name == "keltner_break":
        if len(parts) >= 2:
            kwargs["period"] = int(parts[1])
        if len(parts) >= 3:
            kwargs["atr_period"] = int(parts[2])

    elif name == "bb_squeeze":
        if len(parts) >= 2:
            kwargs["bb_period"] = int(parts[1])
        if len(parts) >= 3:
            kwargs["kc_period"] = int(parts[2])

    elif name == "bb_bandwidth_low":
        if len(parts) >= 2:
            kwargs["threshold"] = float(parts[1])
        if len(parts) >= 3:
            kwargs["period"] = int(parts[2])

    elif name == "obv_rising":
        if len(parts) >= 2:
            kwargs["obv_sma_period"] = int(parts[1])

    elif name == "zscore_extreme":
        if len(parts) >= 2:
            kwargs["threshold"] = float(parts[1])
        if len(parts) >= 3:
            kwargs["period"] = int(parts[2])

    elif name == "pct_from_ma_exit":
        if len(parts) >= 2:
            kwargs["period"] = int(parts[1])
        if len(parts) >= 3:
            kwargs["threshold"] = float(parts[2])

    elif name == "atr_trailing_exit":
        if len(parts) >= 2:
            kwargs["period"] = int(parts[1])
        if len(parts) >= 3:
            kwargs["multiplier"] = float(parts[2])

    elif name == "lgbm_prob":
        if len(parts) >= 2:
            kwargs["threshold"] = float(parts[1])
        if len(parts) >= 3:
            kwargs["model_dir"] = parts[2]


def parse_filter_string(filter_string: str, base_timeframe: str = "1h") -> list[BaseFilter]:
    """Parse a combined filter string like 'trend_up:4 + rsi_oversold:30'."""
    specs = [s.strip() for s in filter_string.split("+")]
    return [parse_filter_spec(s, base_timeframe) for s in specs if s]
