"""Filter registry — maps filter names to classes."""

from __future__ import annotations

from tradingbot.strategy.filters.base import BaseFilter


def get_filter_map() -> dict[str, type[BaseFilter]]:
    """Return map of filter name → class."""
    from tradingbot.strategy.filters.momentum import (
        MacdCrossUpFilter,
        RsiOverboughtFilter,
        RsiOversoldFilter,
    )
    from tradingbot.strategy.filters.price import (
        BbUpperBreakFilter,
        EmaAboveFilter,
        PriceBreakoutFilter,
    )
    from tradingbot.strategy.filters.trend import TrendDownFilter, TrendUpFilter
    from tradingbot.strategy.filters.volume import VolumeSpikeFilter

    return {
        "trend_up": TrendUpFilter,
        "trend_down": TrendDownFilter,
        "rsi_oversold": RsiOversoldFilter,
        "rsi_overbought": RsiOverboughtFilter,
        "macd_cross_up": MacdCrossUpFilter,
        "volume_spike": VolumeSpikeFilter,
        "price_breakout": PriceBreakoutFilter,
        "ema_above": EmaAboveFilter,
        "bb_upper_break": BbUpperBreakFilter,
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
        raise ValueError(f"Empty filter spec")
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


def parse_filter_string(filter_string: str, base_timeframe: str = "1h") -> list[BaseFilter]:
    """Parse a combined filter string like 'trend_up:4 + rsi_oversold:30'."""
    specs = [s.strip() for s in filter_string.split("+")]
    return [parse_filter_spec(s, base_timeframe) for s in specs if s]
