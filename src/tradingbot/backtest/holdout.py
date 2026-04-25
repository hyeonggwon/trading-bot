"""Holdout window resolution for evaluation commands.

Shared between the CLI single-run commands (``backtest``/``combine``) and
the parallel scan workers (``scan``/``combine-scan``). Kept in a small
module so the spawn-safe parallel worker can import it without pulling in
``cli.py``'s Typer/Console/strategy registry imports.
"""
from __future__ import annotations


def resolve_holdout_window(
    df_or_dfs,
    start: str | None,
    end: str | None,
    include_train: bool,
    holdout_pct: float = 0.2,
) -> tuple[str | None, str | None, str]:
    """Decide the evaluation window for rule-based backtests.

    Mirrors ``ml-backtest`` behaviour: by default we slice to the data's
    last ``holdout_pct`` fraction so single/combined rule-based results
    can be compared against ML results on roughly the same period. The
    ratio matches ``MLWalkForwardTrainer``'s ``int(len(df_valid) * 0.8)``
    split. Multi-symbol mode picks the cutoff inside the symbols' common
    timestamp window (max start, min end), so every symbol gets the same
    global cutoff timestamp; per-symbol slicing still happens downstream
    in ``BacktestEngine``.

    Precedence: ``--start``/``--end`` > ``--include-train`` > auto holdout.

    Returns ``(effective_start, effective_end, period_note)``. ``None``
    bounds let ``BacktestEngine`` (or the caller) keep the data edge.
    """
    if start is not None or end is not None:
        return start, end, "user-specified range"

    if include_train:
        return None, None, "full data range (--include-train)"

    if isinstance(df_or_dfs, dict):
        non_empty = [d for d in df_or_dfs.values() if not d.empty]
        if not non_empty:
            return None, None, "full data range (no data)"
        common_start = max(d.index[0] for d in non_empty)
        common_end = min(d.index[-1] for d in non_empty)
        if common_start >= common_end:
            return None, None, "full data range (no overlap)"
        cutoff_ts = common_start + (common_end - common_start) * (1 - holdout_pct)
    else:
        df = df_or_dfs
        if len(df) == 0:
            return None, None, "full data range (empty)"
        cutoff_idx = int(len(df) * (1 - holdout_pct))
        cutoff_ts = df.index[cutoff_idx]

    return str(cutoff_ts), None, f"holdout window (last {holdout_pct:.0%})"
