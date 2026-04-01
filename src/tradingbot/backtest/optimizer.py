"""Grid search parameter optimizer.

Runs backtests across all combinations of strategy parameters and ranks
results by selected metrics. Supports parallel execution.
"""

from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import pandas as pd
import structlog

from tradingbot.backtest.engine import BacktestEngine
from tradingbot.backtest.report import BacktestReport
from tradingbot.config import AppConfig
from tradingbot.strategy.base import Strategy, StrategyParams

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
    strategy = strategy_cls(StrategyParams(params))
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
        max_workers: int | None = None,
    ):
        self.strategy_cls = strategy_cls
        self.config = config
        self.max_workers = max_workers

    def optimize(
        self,
        data: dict[str, pd.DataFrame],
        param_space: dict[str, list[Any]] | None = None,
        sort_by: str = "sharpe_ratio",
    ) -> list[OptimizationResult]:
        """Run grid search optimization.

        Args:
            data: Historical OHLCV data keyed by symbol.
            param_space: Parameter search space. If None, uses strategy's default.
            sort_by: Metric to sort results by (descending).

        Returns:
            List of OptimizationResult sorted by the chosen metric.
        """
        if param_space is None:
            param_space = self.strategy_cls.param_space()

        combinations = generate_param_combinations(param_space)
        total = len(combinations)

        logger.info("optimization_start", strategy=self.strategy_cls.name, combinations=total)

        results: list[OptimizationResult] = []

        if self.max_workers == 1 or total <= 4:
            # Sequential execution for small searches or debugging
            for i, params in enumerate(combinations):
                result = _run_single_backtest(
                    self.strategy_cls, params, data, self.config
                )
                results.append(result)
                if (i + 1) % 10 == 0 or i + 1 == total:
                    logger.debug("optimization_progress", completed=i + 1, total=total)
        else:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        _run_single_backtest,
                        self.strategy_cls, params, data, self.config,
                    ): params
                    for params in combinations
                }
                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    results.append(result)
                    if (i + 1) % 10 == 0 or i + 1 == total:
                        logger.debug("optimization_progress", completed=i + 1, total=total)

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
    def results_to_dataframe(results: list[OptimizationResult]) -> pd.DataFrame:
        """Convert optimization results to a DataFrame for analysis."""
        rows = []
        for r in results:
            row = {**r.params}
            row["sharpe_ratio"] = r.sharpe_ratio
            row["sortino_ratio"] = r.sortino_ratio
            row["total_return"] = r.total_return
            row["max_drawdown"] = r.max_drawdown
            row["total_trades"] = r.total_trades
            row["win_rate"] = r.win_rate
            row["profit_factor"] = r.profit_factor
            row["final_balance"] = r.final_balance
            rows.append(row)
        return pd.DataFrame(rows)

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
