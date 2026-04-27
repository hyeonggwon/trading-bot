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
    grid: list[dict] = field(default_factory=list)
    error: str | None = None


def _select_best(grid: list[dict]) -> dict | None:
    """Pick the best (entry, exit) by lexicographic (sharpe, trades).

    A tie on Sharpe should break in favour of the combo with more trades —
    a Sharpe number from 1 trade is statistically meaningless even if
    nominally high, while Sharpe from 30+ trades is much more credible.
    """
    valid = [g for g in grid if g.get("trades", 0) > 0 and g.get("sharpe") is not None]
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

        # Baseline: evaluate the user-specified defaults so we always have a
        # comparison point even when the grid happens to skip those values.
        baseline_metrics = self._evaluate(
            slice_df, holdout_start, self.baseline_entry, self.baseline_exit
        )
        if baseline_metrics is not None:
            result.baseline_sharpe = baseline_metrics["sharpe"]
            result.baseline_return_pct = baseline_metrics["return_pct"]
            result.baseline_trades = baseline_metrics["trades"]

        for entry_thr, exit_thr in product(entry_grid, exit_grid):
            # Skip nonsensical combos: an exit threshold above entry would
            # close a position the moment it opened.
            if exit_thr >= entry_thr:
                result.n_combos_skipped += 1
                continue

            metrics = self._evaluate(slice_df, holdout_start, entry_thr, exit_thr)
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

        best = _select_best(result.grid)
        if best is not None:
            result.best_entry = float(best["entry"])
            result.best_exit = float(best["exit"])
            result.best_sharpe = float(best["sharpe"])
            result.best_return_pct = float(best["return_pct"])
            result.best_trades = int(best["trades"])
            result.best_win_rate = float(best["win_rate"])
            result.best_max_dd_pct = float(best["max_dd_pct"])
        else:
            result.error = result.error or "no_combo_with_trades"

        return result

    def _slice_holdout(
        self,
        df: pd.DataFrame,
        holdout_start: str,
        holdout_end: str | None,
    ) -> pd.DataFrame:
        """Return holdout window plus a WARMUP_CANDLES prefix.

        The prefix lets ``LGBMStrategy.indicators()`` come up from NaN
        before the first scoring bar; predictions on the warmup rows return
        None so they don't contaminate trade counts. The engine's
        ``start_date`` filter gets the actual holdout boundary back later
        (see ``_evaluate``).
        """
        start_ts = pd.Timestamp(holdout_start, tz="UTC")
        end_ts = pd.Timestamp(holdout_end, tz="UTC") if holdout_end else None

        if start_ts not in df.index:
            # Find the first bar at or after holdout_start.
            after = df.index[df.index >= start_ts]
            if len(after) == 0:
                return df.iloc[0:0]
            start_ts = after[0]

        start_pos = df.index.get_loc(start_ts)
        warmup_pos = max(0, start_pos - WARMUP_CANDLES)

        if end_ts is not None:
            sliced = df.iloc[warmup_pos:]
            return sliced[sliced.index <= end_ts]
        return df.iloc[warmup_pos:]

    def _evaluate(
        self,
        slice_df: pd.DataFrame,
        holdout_start: str,
        entry_threshold: float,
        exit_threshold: float,
    ) -> dict | None:
        """Run one backtest at the given thresholds, return metrics or None on failure."""
        # Per-trial config clone — initial_balance & symbol must reflect this
        # tuner instance, but we shouldn't mutate the caller's AppConfig.
        # The engine uses backtest.start_date to drop the warmup prefix from
        # equity / trade accounting so the metrics line up with the holdout.
        config = self.config.model_copy(deep=True)
        config.trading.symbols = [self.symbol]
        config.trading.timeframe = self.timeframe
        config.trading.initial_balance = self.balance
        config.backtest.start_date = str(holdout_start)
        # Leave end_date alone — slice_df is already truncated to holdout_end.

        params = StrategyParams(
            values={
                "entry_threshold": float(entry_threshold),
                "exit_threshold": float(exit_threshold),
                "model_dir": str(self.model_dir),
                "external_data_dir": (
                    str(self.external_data_dir) if self.external_data_dir else None
                ),
            }
        )
        strategy = LGBMStrategy(params)
        strategy.symbols = [self.symbol]
        strategy.timeframe = self.timeframe

        try:
            engine = BacktestEngine(strategy=strategy, config=config)
            report = engine.run({self.symbol: slice_df})
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

    meta_dict = json.loads(meta_path.read_text())
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
        "holdout_start": result.holdout_start,
        "holdout_end": result.holdout_end,
    }

    tmp_path = meta_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(meta_dict, indent=2, default=str))
    os.replace(tmp_path, meta_path)
    return meta_path
