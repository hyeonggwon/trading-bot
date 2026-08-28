"""Signal-triggered pyramiding: adding tranches to an already-open position.

Home of the shared gate used by both the backtest engine and the live engine.
The two must agree: a gate that diverges would make live trade a setup the
walk-forward never vetted.
"""

from __future__ import annotations

from typing import Any

from tradingbot.config import PyramidingConfig
from tradingbot.core.models import Position

# Floor for an additional tranche — below this an "add" is dust once fees and
# the exchange minimum order size are accounted for.
MIN_ADD_CASH_KRW = 50_000

# Strategy-side caches pinning the entry candle for "since entry" exit filters
# (CombinedStrategy). Snapshotted around an add, see snapshot_entry_anchor.
_ENTRY_ANCHOR_ATTRS = ("_entry_indices", "_entry_times")


def can_add_tranche(
    position: Position,
    free_cash: float,
    equity: float,
    config: PyramidingConfig,
) -> bool:
    """Whether an open position may take another signal-triggered tranche.

    The budget is free cash, not equity: the open position is already spent,
    so measuring the add against total equity would double-count it.
    """
    if not config.enabled:
        return False
    return free_cash >= max(MIN_ADD_CASH_KRW, equity * config.min_add_cash_pct)


def snapshot_entry_anchor(strategy: Any, symbol: str) -> dict[str, Any]:
    """Capture the strategy's entry-candle anchors for ``symbol``.

    ``CombinedStrategy.should_entry`` re-pins them every time it fires, so an
    add would restart trailing exits ("highest high since entry") from the
    add's candle. Pass the result to ``restore_entry_anchor`` afterwards.
    """
    saved: dict[str, Any] = {}
    for attr in _ENTRY_ANCHOR_ATTRS:
        cache = getattr(strategy, attr, None)
        if isinstance(cache, dict) and symbol in cache:
            saved[attr] = cache[symbol]
    return saved


def restore_entry_anchor(strategy: Any, symbol: str, saved: dict[str, Any]) -> None:
    """Restore the caches to their snapshot state — absence included.

    An anchor the snapshot did not capture must be *removed* again, not left at
    whatever the add's ``should_entry`` just pinned. That is the state right
    after a restart: the position survives in state.json but the strategy cache
    is empty, so re-pinning would anchor trailing exits on the add's candle
    instead of falling back to ``position.entry_time`` (the first entry).
    """
    for attr in _ENTRY_ANCHOR_ATTRS:
        cache = getattr(strategy, attr, None)
        if not isinstance(cache, dict):
            continue
        if attr in saved:
            cache[symbol] = saved[attr]
        else:
            cache.pop(symbol, None)


def clear_entry_anchor(strategy: Any, symbol: str | None = None) -> None:
    """Drop cached anchors for ``symbol``, or for every symbol when None.

    Shares ``_ENTRY_ANCHOR_ATTRS`` with snapshot/restore so a new cache can't
    be added to one and forgotten in the others.
    """
    for attr in _ENTRY_ANCHOR_ATTRS:
        cache = getattr(strategy, attr, None)
        if not isinstance(cache, dict):
            continue
        if symbol is None:
            cache.clear()
        else:
            cache.pop(symbol, None)
