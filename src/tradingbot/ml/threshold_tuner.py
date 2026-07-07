"""Per-model entry/exit threshold tuner for the LGBM strategy.

Phase 4 surfaced a tension: extras lifted AUC and cumulative return on the
sandbox 3 but Sharpe slipped because the calibrated probability landscape
shifts per (symbol, timeframe) — the same global ``entry_threshold=0.45``
fires too eagerly on some models and not at all on others.

This module tunes (entry_threshold, exit_threshold) per saved model on its
recorded holdout window. It does **not** retrain — that would be ~16x more
expensive and would defeat the point of separating decision thresholds
from model fit. Instead, it loads the existing booster + calibrator, then
runs a cheap grid of holdout backtests (one per (entry, exit) combo) and
writes the winners back into the meta file.

``LGBMStrategy._load_model`` consumes the persisted thresholds — when
present, they override the defaults the CLI / param_space provides.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, cast

import pandas as pd

from tradingbot.backtest.engine import BacktestEngine
from tradingbot.config import AppConfig
from tradingbot.ml.features import WARMUP_CANDLES
from tradingbot.ml.trainer import LGBMTrainer
from tradingbot.strategy.base import StrategyParams
from tradingbot.strategy.lgbm_strategy import LGBMStrategy

log = logging.getLogger(__name__)

# Defaults centred on the 0.45/0.30 priors but stretched to cover the
# tighter calibrators (e.g., BTC/KRW 4h max ~0.48) and more permissive ones.
DEFAULT_ENTRY_GRID: tuple[float, ...] = (0.40, 0.45, 0.50, 0.55, 0.60)
DEFAULT_EXIT_GRID: tuple[float, ...] = (0.20, 0.25, 0.30, 0.35)


@dataclass
class ThresholdTunerResult:
    """Outcome of a single (symbol, timeframe) threshold sweep."""

    symbol: str = ""
    timeframe: str = ""
    best_entry: float = 0.45
    best_exit: float = 0.30
    best_sharpe: float = float("-inf")
    best_return_pct: float = 0.0
    best_trades: int = 0
    best_win_rate: float = 0.0
    best_max_dd_pct: float = 0.0
    baseline_entry: float = 0.45
    baseline_exit: float = 0.30
    baseline_sharpe: float = float("-inf")
    baseline_return_pct: float = 0.0
    baseline_trades: int = 0
    n_combos_evaluated: int = 0
    n_combos_skipped: int = 0
    holdout_start: str = ""
    holdout_end: str = ""
    # Disjoint sub-windows: the grid is graded on [selection_start, selection_end]
    # and the winning combo + baseline are re-scored on [validation_start,
    # validation_end]. ``best_*`` / ``baseline_*`` are validation-window metrics.
    selection_start: str = ""
    selection_end: str = ""
    validation_start: str = ""
    validation_end: str = ""
    min_trades_applied: int = 1
    grid: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def _select_best(grid: list[dict[str, Any]], min_trades: int = 1) -> dict[str, Any] | None:
    """Pick the best (entry, exit) by lexicographic (sharpe, trades).

    A tie on Sharpe should break in favour of the combo with more trades —
    a Sharpe number from 1 trade is statistically meaningless even if
    nominally high, while Sharpe from 30+ trades is much more credible.

    ``min_trades`` filters out high-precision overfits before argmax. Sandbox
    runs surfaced this: XRP picked a 5-trade combo (Sharpe 1.46) over a
    44-trade combo (Sharpe 1.29) because the tuner was blind to trade count.
    Default 1 keeps the historical "any combo with trades" floor; callers
    typically pass ``baseline_trades`` to enforce "must not regress".
    """
    floor = max(int(min_trades), 1)
    # ``pd.notna`` rejects NaN sharpe values that otherwise sneak past an
    # ``is not None`` check — backtests with constant equity (zero-trade
    # combos squeak through if trade-count guard ever loosens) report Sharpe
    # as NaN and would poison ``max`` with non-deterministic ordering.
    valid = [g for g in grid if g.get("trades", 0) >= floor and pd.notna(g.get("sharpe"))]
    if not valid:
        # Degrade gracefully: if no combo meets ``min_trades`` (e.g. the
        # caller passed an aspirational floor), fall back to "any positive
        # trade count" so we still record the best available combo. Callers
        # that care about the floor can inspect ``min_trades_applied`` /
        # ``best_trades`` on the result.
        if floor > 1:
            valid = [g for g in grid if g.get("trades", 0) > 0 and pd.notna(g.get("sharpe"))]
        if not valid:
            return None
    return max(valid, key=lambda g: (g["sharpe"], g["trades"]))


class ThresholdTuner:
    """Search a grid of (entry, exit) thresholds against the holdout window.

    The tuner reuses the *saved* model + calibrator on disk. Each combo
    instantiates a fresh ``LGBMStrategy`` (so its baseline thresholds and
    feature cache start clean), points it at the same ``model_dir``, and
    runs the engine on the holdout slice. Total cost: O(|entry_grid| ·
    |exit_grid|) cheap backtests — typically ~20 runs in tens of seconds.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        model_dir: Path,
        external_data_dir: Path | None = None,
        config: AppConfig | None = None,
        balance: float = 1_000_000,
        baseline_entry: float = 0.45,
        baseline_exit: float = 0.30,
        min_trades: int | None = None,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_dir = Path(model_dir)
        self.external_data_dir = Path(external_data_dir) if external_data_dir else None
        # Clone happens per-evaluation in ``_evaluate`` so per-trial mutations
        # don't leak between trials or back to the caller.
        self.config = config or AppConfig()
        self.balance = balance
        self.baseline_entry = baseline_entry
        self.baseline_exit = baseline_exit
        # ``None`` → "auto" (use baseline_trades as the floor at search time).
        # Explicit int → use that floor verbatim. ``0`` is normalised to ``1``
        # because zero-trade combos must never win regardless.
        self.min_trades = min_trades

    def search(
        self,
        df: pd.DataFrame,
        entry_grid: tuple[float, ...] = DEFAULT_ENTRY_GRID,
        exit_grid: tuple[float, ...] = DEFAULT_EXIT_GRID,
    ) -> ThresholdTunerResult:
        """Run the threshold grid against the model's recorded holdout window.

        Returns a populated :class:`ThresholdTunerResult`. If the meta file
        is missing or has no holdout window the result carries an ``error``
        string and an empty grid — callers should check ``result.error``
        before treating the output as authoritative.
        """
        result = ThresholdTunerResult(
            symbol=self.symbol,
            timeframe=self.timeframe,
            baseline_entry=self.baseline_entry,
            baseline_exit=self.baseline_exit,
        )

        meta = LGBMTrainer.load_meta(self.symbol, self.timeframe, self.model_dir)
        if meta is None:
            result.error = "meta_missing"
            return result

        holdout_start = meta.get("holdout_start")
        holdout_end = meta.get("holdout_end")
        if not holdout_start:
            result.error = "no_holdout_start_in_meta"
            return result

        df = df[~df.index.duplicated(keep="last")].sort_index()
        slice_df = self._slice_holdout(df, holdout_start, holdout_end)
        if len(slice_df) < WARMUP_CANDLES + 10:
            result.error = f"holdout_slice_too_short ({len(slice_df)} rows)"
            return result

        result.holdout_start = str(holdout_start)
        result.holdout_end = str(holdout_end or df.index[-1])

        # Pre-compute indicators ONCE on the warmup-prefixed slice so each
        # trial reuses them. Without this, the engine recomputes indicators
        # *after* its own start_date filter has dropped the warmup prefix —
        # leaving rolling windows NaN at the start of the holdout. That bug
        # also wastes O(|grid|) duplicate indicator passes; passing the
        # precomputed frame via ``precomputed_indicators`` fixes both.
        eval_df, eval_indicators = self._precompute_indicators(slice_df, holdout_start)
        if eval_df is None or eval_indicators is None:
            result.error = "indicator_precompute_failed"
            return result

        # Disjoint selection / validation split. The grid is graded on the
        # first half of the holdout; the winning combo and the baseline are
        # then re-scored on the second (validation) half. This removes the
        # in-sample selection bias of grading the recommended thresholds on
        # the very bars that chose them — ``best_*`` / ``baseline_*`` are
        # therefore out-of-sample relative to the grid search.
        split = len(eval_df) // 2
        select_df, select_ind = eval_df.iloc[:split], eval_indicators.iloc[:split]
        valid_df, valid_ind = eval_df.iloc[split:], eval_indicators.iloc[split:]
        result.selection_start = str(select_df.index[0])
        result.selection_end = str(select_df.index[-1])
        result.validation_start = str(valid_df.index[0])
        result.validation_end = str(valid_df.index[-1])

        # Baseline is scored on the *validation* window so the best-vs-baseline
        # comparison is apples-to-apples (both out-of-sample).
        baseline_metrics = self._evaluate(
            valid_df, valid_ind, self.baseline_entry, self.baseline_exit
        )
        if baseline_metrics is not None:
            result.baseline_sharpe = baseline_metrics["sharpe"]
            result.baseline_return_pct = baseline_metrics["return_pct"]
            result.baseline_trades = baseline_metrics["trades"]

        # Grid search runs on the selection window only.
        for entry_thr, exit_thr in product(entry_grid, exit_grid):
            # Skip nonsensical combos: an exit threshold above entry would
            # close a position the moment it opened.
            if exit_thr >= entry_thr:
                result.n_combos_skipped += 1
                continue

            metrics = self._evaluate(select_df, select_ind, entry_thr, exit_thr)
            if metrics is None:
                result.n_combos_skipped += 1
                continue

            result.grid.append(
                {
                    "entry": float(entry_thr),
                    "exit": float(exit_thr),
                    **metrics,
                }
            )
            result.n_combos_evaluated += 1

        # Resolve the trade-count floor. None means "auto" — pin to the
        # baseline so the tuner never recommends fewer trades than the
        # current default. We also floor at 1 because zero-trade combos
        # have no meaningful Sharpe even if the user passes ``min_trades=0``.
        if self.min_trades is None:
            min_trades_resolved = max(int(result.baseline_trades), 1)
        else:
            min_trades_resolved = max(int(self.min_trades), 1)
        result.min_trades_applied = min_trades_resolved

        best = _select_best(result.grid, min_trades=min_trades_resolved)
        if best is not None:
            result.best_entry = float(best["entry"])
            result.best_exit = float(best["exit"])
            # Re-score the chosen combo out-of-sample on the validation window;
            # the reported best_* metrics come from there, not the selection grid.
            val_metrics = self._evaluate(valid_df, valid_ind, result.best_entry, result.best_exit)
            if val_metrics is not None:
                result.best_sharpe = float(val_metrics["sharpe"])
                result.best_return_pct = float(val_metrics["return_pct"])
                result.best_trades = int(val_metrics["trades"])
                result.best_win_rate = float(val_metrics["win_rate"])
                result.best_max_dd_pct = float(val_metrics["max_dd_pct"])
            else:
                # Validation re-eval hit an engine error; fall back to the
                # selection-window metrics so the combo is still recorded,
                # and flag that the report is in-sample for this run.
                result.best_sharpe = float(best["sharpe"])
                result.best_return_pct = float(best["return_pct"])
                result.best_trades = int(best["trades"])
                result.best_win_rate = float(best["win_rate"])
                result.best_max_dd_pct = float(best["max_dd_pct"])
                result.error = result.error or "validation_eval_failed"
        else:
            result.error = result.error or "no_combo_with_trades"

        return result

    def _precompute_indicators(
        self,
        slice_df: pd.DataFrame,
        holdout_start: str,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """Compute indicators on warmup+holdout, then strip the warmup prefix.

        Returns ``(eval_df, eval_indicators)`` aligned on the holdout-only
        index — ready to feed straight to ``BacktestEngine.run`` via
        ``precomputed_indicators`` without any further slicing. Returns
        ``(None, None)`` if the strategy fails to build the feature matrix
        (e.g. external data missing for an extras-trained model).
        """
        params = StrategyParams(
            values={
                "model_dir": str(self.model_dir),
                "external_data_dir": (
                    str(self.external_data_dir) if self.external_data_dir else None
                ),
                # Indicator pre-compute path doesn't actually load thresholds,
                # but stay consistent with ``_evaluate`` so any future feature
                # build that does is also unaffected by stale meta overrides.
                "ignore_meta_thresholds": True,
            }
        )
        indicator_strategy = LGBMStrategy(params)
        indicator_strategy.symbols = [self.symbol]
        indicator_strategy.timeframe = self.timeframe

        try:
            full_indicators = indicator_strategy.indicators(slice_df.copy())
        except Exception as exc:  # noqa: BLE001 — log + bail any indicator failure
            log.warning(
                "ThresholdTuner[%s %s]: indicator precompute failed: %s",
                self.symbol,
                self.timeframe,
                exc,
            )
            return None, None

        start_ts = pd.Timestamp(holdout_start, tz="UTC")
        # Strip warmup prefix from both frames so they share the holdout-only
        # index. The reindex check inside the engine becomes a no-op.
        eval_df = slice_df[slice_df.index >= start_ts]
        eval_indicators = full_indicators[full_indicators.index >= start_ts]
        if not eval_df.index.equals(eval_indicators.index):
            eval_indicators = eval_indicators.reindex(eval_df.index)
        return eval_df, eval_indicators

    def _slice_holdout(
        self,
        df: pd.DataFrame,
        holdout_start: str,
        holdout_end: str | None,
    ) -> pd.DataFrame:
        """Return holdout window plus a WARMUP_CANDLES prefix.

        The prefix lets ``LGBMStrategy.indicators()`` come up from NaN
        before the first scoring bar; predictions on the warmup rows return
        None so they don't contaminate trade counts. ``_precompute_indicators``
        strips the warmup back off after computation so the holdout index
        feeds straight into the engine via ``precomputed_indicators``.
        """
        start_ts = pd.Timestamp(holdout_start, tz="UTC")
        end_ts = pd.Timestamp(holdout_end, tz="UTC") if holdout_end else None

        if start_ts not in df.index:
            # Find the first bar at or after holdout_start.
            after = df.index[df.index >= start_ts]
            if len(after) == 0:
                return df.iloc[0:0]
            start_ts = after[0]

        # cast: unique DatetimeIndex — get_loc returns a scalar position
        start_pos = cast(int, df.index.get_loc(start_ts))
        warmup_pos = max(0, start_pos - WARMUP_CANDLES)

        if end_ts is not None:
            sliced = df.iloc[warmup_pos:]
            return sliced[sliced.index <= end_ts]
        return df.iloc[warmup_pos:]

    def _evaluate(
        self,
        eval_df: pd.DataFrame,
        eval_indicators: pd.DataFrame,
        entry_threshold: float,
        exit_threshold: float,
    ) -> dict[str, Any] | None:
        """Run one backtest at the given thresholds, return metrics or None on failure."""
        # Per-trial config clone — initial_balance & symbol must reflect this
        # tuner instance, but we shouldn't mutate the caller's AppConfig. We
        # do *not* set ``backtest.start_date`` here: ``eval_df`` is already
        # holdout-only, and setting it would re-trigger the same slicing
        # path that strips the warmup-aware indicator alignment.
        config = self.config.model_copy(deep=True)
        config.trading.symbols = [self.symbol]
        config.trading.timeframe = self.timeframe
        config.trading.initial_balance = self.balance

        params = StrategyParams(
            values={
                "entry_threshold": float(entry_threshold),
                "exit_threshold": float(exit_threshold),
                "model_dir": str(self.model_dir),
                "external_data_dir": (
                    str(self.external_data_dir) if self.external_data_dir else None
                ),
                # Critical: without this opt-out, ``_load_model`` would
                # overwrite the per-trial entry/exit with whatever meta
                # currently has, collapsing every grid cell onto the same
                # thresholds. This was discovered when re-running the XRP
                # tuner on a model whose meta had already been patched with
                # 0.55/0.35 — every combo produced 5 trades.
                "ignore_meta_thresholds": True,
            }
        )
        strategy = LGBMStrategy(params)
        strategy.symbols = [self.symbol]
        strategy.timeframe = self.timeframe

        try:
            engine = BacktestEngine(strategy=strategy, config=config)
            report = engine.run(
                {self.symbol: eval_df},
                precomputed_indicators={self.symbol: eval_indicators},
            )
        except Exception as exc:
            log.warning(
                "ThresholdTuner[%s %s] entry=%s exit=%s failed: %s",
                self.symbol,
                self.timeframe,
                entry_threshold,
                exit_threshold,
                exc,
            )
            return None

        return {
            "sharpe": float(report.sharpe_ratio),
            "return_pct": float(report.total_return * 100),
            "trades": int(report.total_trades),
            "win_rate": float(report.win_rate),
            "max_dd_pct": float(report.max_drawdown * 100),
        }


def patch_meta_thresholds(
    symbol: str,
    timeframe: str,
    model_dir: Path,
    result: ThresholdTunerResult,
) -> Path | None:
    """Persist the tuned thresholds into the model meta file atomically.

    Writes ``entry_threshold`` / ``exit_threshold`` plus a ``threshold_tuning``
    audit dict (best metrics, baseline metrics, holdout window, grid stats).
    ``LGBMStrategy._load_model`` reads ``entry_threshold`` / ``exit_threshold``
    when populating per-symbol overrides. Returns the meta path on success or
    ``None`` if there was nothing to patch (missing meta or no winning combo).

    Atomic write: tmp file + ``os.replace`` so an interrupted process never
    leaves a partial JSON behind.
    """
    if result.error and not result.grid:
        log.info(
            "ThresholdTuner[%s %s]: skipping meta patch (%s)",
            symbol,
            timeframe,
            result.error,
        )
        return None

    symbol_key = symbol.replace("/", "_")
    meta_path = Path(model_dir) / f"lgbm_{symbol_key}_{timeframe}_meta.json"
    if not meta_path.exists():
        log.warning("ThresholdTuner: meta missing at %s — cannot patch", meta_path)
        return None

    # A corrupt or unreadable meta should not crash the whole tune sweep —
    # log and bail so the caller can report on the rest of the symbols.
    try:
        meta_dict = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning(
            "ThresholdTuner: meta read failed at %s (%s) — cannot patch",
            meta_path,
            exc,
        )
        return None
    meta_dict["entry_threshold"] = result.best_entry
    meta_dict["exit_threshold"] = result.best_exit
    meta_dict["threshold_tuning"] = {
        "best_entry": result.best_entry,
        "best_exit": result.best_exit,
        "best_sharpe": result.best_sharpe,
        "best_return_pct": result.best_return_pct,
        "best_trades": result.best_trades,
        "best_win_rate": result.best_win_rate,
        "best_max_dd_pct": result.best_max_dd_pct,
        "baseline_entry": result.baseline_entry,
        "baseline_exit": result.baseline_exit,
        "baseline_sharpe": result.baseline_sharpe,
        "baseline_return_pct": result.baseline_return_pct,
        "baseline_trades": result.baseline_trades,
        "n_combos_evaluated": result.n_combos_evaluated,
        "n_combos_skipped": result.n_combos_skipped,
        "min_trades_applied": result.min_trades_applied,
        "holdout_start": result.holdout_start,
        "holdout_end": result.holdout_end,
        "selection_start": result.selection_start,
        "selection_end": result.selection_end,
        "validation_start": result.validation_start,
        "validation_end": result.validation_end,
    }

    tmp_path = meta_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(meta_dict, indent=2, default=str))
    os.replace(tmp_path, meta_path)
    return meta_path
