"""Grid search parameter optimizer.

Runs backtests across all combinations of strategy parameters and ranks
results by selected metrics. Supports parallel execution.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd
import structlog

from tradingbot.backtest.engine import BacktestEngine
from tradingbot.config import AppConfig
from tradingbot.strategy.base import Strategy

if TYPE_CHECKING:
    from rich.progress import Progress

logger = structlog.get_logger()


@dataclass
class OptimizationResult:
    """Result of a single parameter combination backtest."""

    params: dict[str, Any]
    sharpe_ratio: float
    sortino_ratio: float
    total_return: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    profit_factor: float
    final_balance: float


def _run_single_backtest(
    strategy_cls: type[Strategy],
    params: dict[str, Any],
    data: dict[str, pd.DataFrame],
    config: AppConfig,
) -> OptimizationResult:
    """Run a single backtest with given parameters. Designed for parallel execution."""
    strategy = strategy_cls(params)
    strategy.symbols = config.trading.symbols
    strategy.timeframe = config.trading.timeframe

    engine = BacktestEngine(strategy=strategy, config=config)
    report = engine.run(data)

    return OptimizationResult(
        params=params,
        sharpe_ratio=report.sharpe_ratio,
        sortino_ratio=report.sortino_ratio,
        total_return=report.total_return,
        max_drawdown=report.max_drawdown,
        total_trades=report.total_trades,
        win_rate=report.win_rate,
        profit_factor=report.profit_factor,
        final_balance=report.final_balance,
    )


def generate_param_combinations(param_space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Generate all combinations from a parameter space."""
    if not param_space:
        return [{}]

    keys = list(param_space.keys())
    values = list(param_space.values())
    combinations = list(itertools.product(*values))

    return [dict(zip(keys, combo)) for combo in combinations]


class GridSearchOptimizer:
    """Exhaustive grid search over strategy parameter space."""

    def __init__(
        self,
        strategy_cls: type[Strategy],
        config: AppConfig,
    ):
        self.strategy_cls = strategy_cls
        self.config = config

    def optimize(
        self,
        data: dict[str, pd.DataFrame],
        param_space: dict[str, list[Any]] | None = None,
        sort_by: str = "sharpe_ratio",
        progress: Progress | None = None,
    ) -> list[OptimizationResult]:
        """Run grid search optimization.

        Args:
            data: Historical OHLCV data keyed by symbol.
            param_space: Parameter search space. If None, uses strategy's default.
            sort_by: Metric to sort results by (descending).
            progress: Optional Rich Progress instance for progress bar display.

        Returns:
            List of OptimizationResult sorted by the chosen metric.
        """
        if param_space is None:
            param_space = self.strategy_cls.param_space()

        combinations = generate_param_combinations(param_space)
        total = len(combinations)

        logger.info("optimization_start", strategy=self.strategy_cls.name, combinations=total)

        results: list[OptimizationResult] = []
        opt_task = progress.add_task("Optimizing", total=total) if progress else None

        try:
            for i, params in enumerate(combinations):
                result = _run_single_backtest(self.strategy_cls, params, data, self.config)
                results.append(result)
                if progress and opt_task is not None:
                    progress.advance(opt_task)
                elif (i + 1) % 10 == 0 or i + 1 == total:
                    logger.debug("optimization_progress", completed=i + 1, total=total)
        finally:
            if progress and opt_task is not None:
                progress.remove_task(opt_task)

        # Sort by chosen metric (descending, except max_drawdown which is ascending)
        reverse = sort_by != "max_drawdown"
        results.sort(key=lambda r: getattr(r, sort_by), reverse=reverse)

        logger.info(
            "optimization_complete",
            combinations=total,
            best_sharpe=f"{results[0].sharpe_ratio:.2f}" if results else "N/A",
        )

        return results

    @staticmethod
    def print_results(results: list[OptimizationResult], top_n: int = 10) -> None:
        """Print top N results as a Rich table."""
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=f"Optimization Results (Top {min(top_n, len(results))})")

        table.add_column("#", justify="right")
        table.add_column("Parameters")
        table.add_column("Sharpe", justify="right")
        table.add_column("Return", justify="right")
        table.add_column("MaxDD", justify="right")
        table.add_column("Trades", justify="right")
        table.add_column("Win%", justify="right")
        table.add_column("PF", justify="right")

        for i, r in enumerate(results[:top_n], 1):
            params_str = ", ".join(f"{k}={v}" for k, v in r.params.items())
            table.add_row(
                str(i),
                params_str,
                f"{r.sharpe_ratio:.2f}",
                f"{r.total_return:.2%}",
                f"{r.max_drawdown:.2%}",
                str(r.total_trades),
                f"{r.win_rate:.1%}",
                f"{r.profit_factor:.2f}",
            )

        console.print(table)
