"""Optuna-driven hyperparameter tuner for the LGBM strategy.

Each Optuna trial reuses ``MLStrategyWalkForward`` to evaluate the candidate
hyperparams the same way Phase 2 did — by running the time-honest per-window
backtest and aggregating Sharpe / cumulative return / AUC. This keeps the
tuning objective aligned with what we actually report, instead of overfitting
to a model-only metric (the AUC pitfall surfaced in Phase 2).

Budget: ``study.optimize(n_trials=..., timeout=...)`` stops at whichever fires
first. Sandbox plan calls for 50 trials × 60 min per (symbol, timeframe).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from tradingbot.config import AppConfig
from tradingbot.ml.strategy_walk_forward import MLStrategyWalkForward

log = logging.getLogger(__name__)

VALID_OBJECTIVES = ("holdout_sharpe", "holdout_cum_return", "holdout_auc")


@dataclass
class LGBMTunerResult:
    """Outcome of an LGBMTuner.search run."""

    best_params: dict = field(default_factory=dict)
    best_value: float = float("-inf")
    n_trials_completed: int = 0
    n_trials_pruned: int = 0
    elapsed_sec: float = 0.0
    objective: str = "holdout_sharpe"
    trials: list[dict] = field(default_factory=list)


class LGBMTuner:
    """Optuna wrapper around MLStrategyWalkForward.

    The tuner trains/evaluates one full strategy walk-forward per trial. That
    is the expensive operation but also the one whose result we trust — Phase 2
    showed AUC alone is misleading when the label distribution changes.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        train_months: int = 6,
        test_months: int = 2,
        forward_candles: int = 4,
        threshold: float = 0.006,
        target_kind: str = "binary",
        atr_mult: float = 1.0,
        include_extra: bool = False,
        entry_threshold: float = 0.45,
        exit_threshold: float = 0.30,
        balance: float = 1_000_000,
        external_data_dir: str | Path | None = None,
        config: AppConfig | None = None,
        objective: str = "holdout_sharpe",
        seed: int = 42,
    ) -> None:
        if objective not in VALID_OBJECTIVES:
            raise ValueError(f"Unknown objective={objective!r}; expected one of {VALID_OBJECTIVES}")
        self.symbol = symbol
        self.timeframe = timeframe
        self.train_months = train_months
        self.test_months = test_months
        self.forward_candles = forward_candles
        self.threshold = threshold
        self.target_kind = target_kind
        self.atr_mult = atr_mult
        self.include_extra = include_extra
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.balance = balance
        self.external_data_dir = Path(external_data_dir) if external_data_dir else None
        # Caller's AppConfig carries fee rate, slippage, risk settings, etc.
        # We clone per trial so mutation of trading.symbols / initial_balance
        # doesn't bleed between trials or back to the caller.
        self.config = config or AppConfig()
        self.objective = objective
        self.seed = seed

    def _suggest_params(self, trial) -> dict:
        """Sample one set of LightGBM hyperparams from the Phase 3 search space."""
        return {
            "num_leaves": trial.suggest_int("num_leaves", 8, 64, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, log=True),
        }

    def _score(self, lgbm_params: dict, df: pd.DataFrame) -> tuple[float, dict]:
        """Run one strategy walk-forward and extract the objective score.

        Returns ``(score, summary_dict)``. On any exception we log it and
        return ``(-inf, {"error": ...})`` so the trial still gets recorded
        in the trial log and Optuna keeps moving — losing a trial silently
        would desync ``trial_log`` from the study's trial count.
        """
        # Clone the user's AppConfig so per-trial mutations (symbols,
        # timeframe, balance) don't leak across trials or back to the CLI.
        config = self.config.model_copy(deep=True)
        config.trading.symbols = [self.symbol]
        config.trading.timeframe = self.timeframe
        config.trading.initial_balance = self.balance

        runner = MLStrategyWalkForward(
            symbol=self.symbol,
            timeframe=self.timeframe,
            train_months=self.train_months,
            test_months=self.test_months,
            forward_candles=self.forward_candles,
            threshold=self.threshold,
            target_kind=self.target_kind,
            atr_mult=self.atr_mult,
            include_extra=self.include_extra,
            entry_threshold=self.entry_threshold,
            exit_threshold=self.exit_threshold,
            external_data_dir=self.external_data_dir,
            config=config,
            lgbm_params=lgbm_params,
        )
        try:
            report = runner.run(df)
        except Exception as exc:  # don't let one bad trial kill the study
            log.exception("LGBMTuner trial failed: %s", exc)
            return float("-inf"), {"error": str(exc)}

        if not report.windows:
            return float("-inf"), {"error": "no_windows"}

        summary = {
            "avg_sharpe": float(report.avg_sharpe),
            "cumulative_return_pct": float(report.cumulative_return_pct),
            "total_trades": int(report.total_trades),
            "n_windows": int(report.n_windows),
            "n_skipped": int(report.n_skipped),
        }

        if self.objective == "holdout_sharpe":
            score = float(report.avg_sharpe)
        elif self.objective == "holdout_cum_return":
            score = float(report.cumulative_return_pct)
        elif self.objective == "holdout_auc":
            # Per-window AUC isn't recorded by MLStrategyWalkForward; derive
            # an AUC-shaped proxy from win rate. Mapping:
            #   random win-rate 0.5 → score 0.0
            #   perfect win-rate 1.0 → score 1.0
            # Use holdout_sharpe for serious tuning; this proxy is for fast
            # iterations only.
            score = float(report.avg_win_rate * 2 - 1)
            summary["proxy_auc"] = score
        else:  # pragma: no cover (validated in __init__)
            score = float("-inf")

        return score, summary

    def search(
        self,
        df: pd.DataFrame,
        n_trials: int = 50,
        time_budget_sec: float = 3600.0,
        verbose: bool = False,
    ) -> LGBMTunerResult:
        """Run an Optuna study and return the best params + score.

        Stops at whichever limit fires first (n_trials or time_budget_sec).
        """
        import optuna

        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        trial_log: list[dict] = []
        start = time.time()

        def _objective(trial) -> float:
            params = self._suggest_params(trial)
            score, summary = self._score(params, df)
            trial_log.append(
                {
                    "trial": trial.number,
                    "score": score,
                    "params": params,
                    **summary,
                }
            )
            return score

        study.optimize(
            _objective,
            n_trials=n_trials,
            timeout=time_budget_sec,
            gc_after_trial=True,
            show_progress_bar=False,
            catch=(Exception,),  # don't let one bad trial kill the study
        )

        elapsed = time.time() - start

        result = LGBMTunerResult(
            objective=self.objective,
            elapsed_sec=round(elapsed, 2),
            n_trials_completed=len(study.trials),
            n_trials_pruned=sum(
                1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
            ),
            trials=trial_log,
        )

        # Best trial may be missing if every trial errored
        try:
            best = study.best_trial
            result.best_params = dict(best.params)
            result.best_value = float(best.value) if best.value is not None else float("-inf")
        except ValueError:
            log.warning("LGBMTuner: no successful trial — best_params empty")

        return result
