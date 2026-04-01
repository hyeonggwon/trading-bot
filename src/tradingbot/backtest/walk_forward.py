"""Walk-forward validation.

Splits data into rolling train/test windows, optimizes parameters on each
training window, then evaluates on the subsequent test window. This measures
how well optimized parameters generalize to unseen data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import structlog

from tradingbot.backtest.engine import BacktestEngine
from tradingbot.backtest.optimizer import GridSearchOptimizer, OptimizationResult
from tradingbot.backtest.report import BacktestReport
from tradingbot.config import AppConfig
from tradingbot.strategy.base import Strategy, StrategyParams

logger = structlog.get_logger()


@dataclass
class WalkForwardWindow:
    """Result from a single walk-forward window."""

    window_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict[str, Any]
    train_sharpe: float
    train_return: float
    test_sharpe: float
    test_return: float
    test_trades: int
    test_max_drawdown: float


@dataclass
class WalkForwardReport:
    """Aggregated walk-forward validation results."""

    windows: list[WalkForwardWindow]
    strategy_name: str

    @property
    def num_windows(self) -> int:
        return len(self.windows)

    @property
    def avg_test_sharpe(self) -> float:
        if not self.windows:
            return 0.0
        return sum(w.test_sharpe for w in self.windows) / len(self.windows)

    @property
    def avg_test_return(self) -> float:
        if not self.windows:
            return 0.0
        return sum(w.test_return for w in self.windows) / len(self.windows)

    @property
    def avg_train_sharpe(self) -> float:
        if not self.windows:
            return 0.0
        return sum(w.train_sharpe for w in self.windows) / len(self.windows)

    @property
    def walk_forward_efficiency(self) -> float:
        """Ratio of out-of-sample to in-sample Sharpe. Higher is better.
        > 0.5 is generally acceptable, > 0.7 is good.
        Only meaningful when avg_train_sharpe > 0."""
        if self.avg_train_sharpe <= 0:
            return 0.0
        return self.avg_test_sharpe / self.avg_train_sharpe

    @property
    def overfitting_ratio(self) -> float:
        """(train_sharpe - test_sharpe) / train_sharpe. Lower is better.
        < 0.5 means the strategy is not heavily overfit.
        Only meaningful when avg_train_sharpe > 0."""
        if self.avg_train_sharpe <= 0:
            return 0.0
        return (self.avg_train_sharpe - self.avg_test_sharpe) / self.avg_train_sharpe

    @property
    def total_test_trades(self) -> int:
        return sum(w.test_trades for w in self.windows)

    @property
    def cumulative_test_return(self) -> float:
        """Cumulative return across all test windows (compounded)."""
        cumulative = 1.0
        for w in self.windows:
            cumulative *= (1 + w.test_return)
        return cumulative - 1.0

    def print_summary(self) -> None:
        """Print walk-forward results."""
        from rich.console import Console
        from rich.table import Table

        console = Console()

        # Summary table
        summary = Table(title="Walk-Forward Validation Summary")
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="green", justify="right")

        summary.add_row("Strategy", self.strategy_name)
        summary.add_row("Windows", str(self.num_windows))
        summary.add_row("Avg Train Sharpe", f"{self.avg_train_sharpe:.2f}")
        summary.add_row("Avg Test Sharpe", f"{self.avg_test_sharpe:.2f}")
        summary.add_row("WF Efficiency", f"{self.walk_forward_efficiency:.2%}")
        summary.add_row("Overfitting Ratio", f"{self.overfitting_ratio:.2%}")
        summary.add_row("Cumulative Test Return", f"{self.cumulative_test_return:.2%}")
        summary.add_row("Total Test Trades", str(self.total_test_trades))

        console.print(summary)

        # Window detail table
        detail = Table(title="Window Details")
        detail.add_column("#", justify="right")
        detail.add_column("Train Period")
        detail.add_column("Test Period")
        detail.add_column("Best Params")
        detail.add_column("Train Sharpe", justify="right")
        detail.add_column("Test Sharpe", justify="right")
        detail.add_column("Test Return", justify="right")
        detail.add_column("Test Trades", justify="right")

        for w in self.windows:
            params_str = ", ".join(f"{k}={v}" for k, v in w.best_params.items())
            train_period = f"{w.train_start.date()} ~ {w.train_end.date()}"
            test_period = f"{w.test_start.date()} ~ {w.test_end.date()}"

            test_style = "green" if w.test_sharpe > 0 else "red"
            detail.add_row(
                str(w.window_index + 1),
                train_period,
                test_period,
                params_str,
                f"{w.train_sharpe:.2f}",
                f"[{test_style}]{w.test_sharpe:.2f}[/{test_style}]",
                f"[{test_style}]{w.test_return:.2%}[/{test_style}]",
                str(w.test_trades),
            )

        console.print(detail)


def create_walk_forward_windows(
    df: pd.DataFrame,
    train_months: int = 3,
    test_months: int = 1,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Create rolling train/test window boundaries.

    Returns list of (train_start, train_end, test_start, test_end) tuples.
    """
    start = df.index.min()
    end = df.index.max()

    windows = []
    current = start

    while True:
        train_start = current
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_end > end:
            break

        windows.append((
            pd.Timestamp(train_start),
            pd.Timestamp(train_end),
            pd.Timestamp(test_start),
            pd.Timestamp(test_end),
        ))

        # Slide forward by test_months
        current = test_start

    return windows


class WalkForwardValidator:
    """Walk-forward validation runner."""

    def __init__(
        self,
        strategy_cls: type[Strategy],
        config: AppConfig,
        train_months: int = 3,
        test_months: int = 1,
    ):
        self.strategy_cls = strategy_cls
        self.config = config
        self.train_months = train_months
        self.test_months = test_months

    def validate(
        self,
        data: dict[str, pd.DataFrame],
        param_space: dict[str, list[Any]] | None = None,
    ) -> WalkForwardReport:
        """Run walk-forward validation.

        For each window:
        1. Optimize parameters on training data
        2. Test best parameters on out-of-sample data
        """
        if param_space is None:
            param_space = self.strategy_cls.param_space()

        # Clear date filters — walk-forward pre-slices data per window,
        # so engine must not re-filter with stale start/end dates.
        wf_config = self.config.model_copy(deep=True)
        wf_config.backtest.start_date = None
        wf_config.backtest.end_date = None

        symbol = wf_config.trading.symbols[0]
        df = data[symbol]

        windows = create_walk_forward_windows(df, self.train_months, self.test_months)

        if not windows:
            logger.warning("insufficient_data_for_walk_forward")
            return WalkForwardReport(windows=[], strategy_name=self.strategy_cls.name)

        logger.info(
            "walk_forward_start",
            strategy=self.strategy_cls.name,
            windows=len(windows),
            train_months=self.train_months,
            test_months=self.test_months,
        )

        results: list[WalkForwardWindow] = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(
                "walk_forward_window",
                window=i + 1,
                train=f"{train_start.date()} ~ {train_end.date()}",
                test=f"{test_start.date()} ~ {test_end.date()}",
            )

            # Step 1: Optimize on training data
            train_df = df[(df.index >= train_start) & (df.index < train_end)]
            train_data = {symbol: train_df}

            optimizer = GridSearchOptimizer(
                strategy_cls=self.strategy_cls,
                config=wf_config,
                max_workers=1,  # Sequential within each window
            )
            opt_results = optimizer.optimize(train_data, param_space, sort_by="sharpe_ratio")

            if not opt_results:
                continue

            best = opt_results[0]

            # Step 2: Test on out-of-sample data
            test_df = df[(df.index >= test_start) & (df.index < test_end)]
            test_data = {symbol: test_df}

            test_result = _run_test(
                self.strategy_cls, best.params, test_data, wf_config
            )

            results.append(WalkForwardWindow(
                window_index=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_params=best.params,
                train_sharpe=best.sharpe_ratio,
                train_return=best.total_return,
                test_sharpe=test_result.sharpe_ratio,
                test_return=test_result.total_return,
                test_trades=test_result.total_trades,
                test_max_drawdown=test_result.max_drawdown,
            ))

        report = WalkForwardReport(
            windows=results,
            strategy_name=self.strategy_cls.name,
        )

        logger.info(
            "walk_forward_complete",
            windows=report.num_windows,
            wf_efficiency=f"{report.walk_forward_efficiency:.2%}",
            overfitting_ratio=f"{report.overfitting_ratio:.2%}",
        )

        return report


def _run_test(
    strategy_cls: type[Strategy],
    params: dict[str, Any],
    data: dict[str, pd.DataFrame],
    config: AppConfig,
) -> BacktestReport:
    """Run a single backtest and return the report."""
    strategy = strategy_cls(StrategyParams(params))
    strategy.symbols = config.trading.symbols
    strategy.timeframe = config.trading.timeframe

    engine = BacktestEngine(strategy=strategy, config=config)
    return engine.run(data)
