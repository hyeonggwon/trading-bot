"""Walk-forward validation.

Splits data into expanding train/test windows with an embargo gap — the same
frame as the ML walk-forward (``ml.walk_forward.make_expanding_windows``), so
rule and ML validation numbers are directly comparable. Parameters are
optimized on each training window, then evaluated on the subsequent test
window. This measures how well optimized parameters generalize to unseen data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd
import structlog

from tradingbot.backtest.engine import BacktestEngine
from tradingbot.backtest.optimizer import GridSearchOptimizer
from tradingbot.backtest.report import BacktestReport
from tradingbot.config import AppConfig
from tradingbot.strategy.base import Strategy, StrategyParams

if TYPE_CHECKING:
    from rich.progress import Progress

logger = structlog.get_logger()

# Canonical embargo for BOTH rule and ML walk-forward windows (~3x max
# indicator lookback (52) for safer purging). ml.walk_forward re-exports this.
EMBARGO_CANDLES = 150


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
            cumulative *= 1 + w.test_return
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
    embargo_candles: int = EMBARGO_CANDLES,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Create expanding train/test window boundaries with an embargo gap.

    Mirrors ``ml.walk_forward.make_expanding_windows``: train always starts at
    the first candle and ``train_end`` advances by ``test_months`` per window;
    ``embargo_candles`` rows are skipped between train_end and test_start so
    the spans used for fitting and scoring never touch.

    Returns list of (train_start, train_end, test_start, test_end) tuples.
    """
    if test_months < 1:
        return []  # train_end would never advance — no valid frame exists

    start = df.index.min()
    end = df.index.max()

    windows: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    train_start = pd.Timestamp(start)
    train_end = train_start + pd.DateOffset(months=train_months)

    while True:
        # Embargo applied in candle counts (index positions), matching the
        # ML frame's integer gap regardless of timeframe.
        pos = int(df.index.searchsorted(train_end)) + embargo_candles
        if pos >= len(df.index):
            break
        test_start = pd.Timestamp(df.index[pos])
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_end > end:
            break

        # A data gap larger than the test span can pin test_start in place
        # while train_end advances — the same candles would be scored twice.
        # Skip forward until the frame actually moves.
        if not windows or test_start > windows[-1][2]:
            windows.append((train_start, pd.Timestamp(train_end), test_start, test_end))

        # Expand: train grows by one test span per window
        train_end = train_end + pd.DateOffset(months=test_months)

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
        progress: Progress | None = None,
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
        wf_task = progress.add_task("Walk-Forward", total=len(windows)) if progress else None

        try:
            for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
                if progress and wf_task is not None:
                    desc = f"WF {i + 1}/{len(windows)}: {train_start.date()}~{test_end.date()}"
                    progress.update(wf_task, description=desc)
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
                    if progress and wf_task is not None:
                        progress.advance(wf_task)
                    continue

                best = opt_results[0]

                # Step 2: Test on out-of-sample data
                test_df = df[(df.index >= test_start) & (df.index < test_end)]
                test_data = {symbol: test_df}

                test_result = _run_test(self.strategy_cls, best.params, test_data, wf_config)

                results.append(
                    WalkForwardWindow(
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
                    )
                )

                if progress and wf_task is not None:
                    progress.advance(wf_task)
        finally:
            if progress and wf_task is not None:
                progress.remove_task(wf_task)

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


def walk_forward_combined(
    strategy: Strategy,
    strategy_name: str,
    symbol: str,
    df: pd.DataFrame,
    config: AppConfig,
    train_months: int = 3,
    test_months: int = 1,
    progress: Progress | None = None,
) -> WalkForwardReport:
    """Walk-forward for fixed-filter (combined) strategies — no optimization.

    Same expanding+embargo windows as :class:`WalkForwardValidator`, but the strategy
    is fixed: each window backtests the train and test spans with a warmup
    buffer so indicator values at the window edge match full-history
    computation. Returns an empty-windows report when the data cannot fit
    a single window.
    """
    import copy

    # Warmup buffer: enough for the most demanding indicators
    # (e.g., trend_up:4 with SMA_50 at 4x = 200 bars, plus margin)
    warmup_bars = 300

    wf_config = config.model_copy(deep=True)
    wf_config.backtest.start_date = None
    wf_config.backtest.end_date = None

    windows = create_walk_forward_windows(df, train_months, test_months)
    if not windows:
        return WalkForwardReport(windows=[], strategy_name=strategy_name)

    results: list[WalkForwardWindow] = []
    task = progress.add_task("Walk-Forward (combined)", total=len(windows)) if progress else None

    for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
        if progress and task is not None:
            progress.update(
                task,
                description=f"WF {i + 1}/{len(windows)}: {train_start.date()}~{test_end.date()}",
            )

        # Train window — include warmup buffer for indicator computation
        train_start_idx = df.index.searchsorted(train_start)
        train_warmup_idx = max(0, train_start_idx - warmup_bars)
        train_with_warmup = df.iloc[train_warmup_idx:].copy()
        train_with_warmup = train_with_warmup[train_with_warmup.index < train_end]

        train_strategy = copy.deepcopy(strategy)
        engine = BacktestEngine(strategy=train_strategy, config=wf_config)
        full_train_report = engine.run({symbol: train_with_warmup})

        # Filter to train period only
        train_start_dt = (
            train_start.to_pydatetime() if hasattr(train_start, "to_pydatetime") else train_start
        )
        train_trades = [
            t
            for t in full_train_report.trades
            if t.entry_order.created_at is not None and t.entry_order.created_at >= train_start_dt
        ]
        train_equity = full_train_report.equity_curve[
            full_train_report.equity_curve.index >= train_start
        ]

        if len(train_equity) < 2:
            train_report = full_train_report
        else:
            train_report = BacktestReport(
                trades=train_trades,
                equity_curve=train_equity,
                initial_balance=float(train_equity.iloc[0]),
                final_balance=float(train_equity.iloc[-1]),
                timeframe=wf_config.trading.timeframe,
            )

        # Test window — include warmup buffer for indicator computation
        test_start_idx = df.index.searchsorted(test_start)
        warmup_idx = max(0, test_start_idx - warmup_bars)
        test_with_warmup = df.iloc[warmup_idx:].copy()
        test_with_warmup = test_with_warmup[test_with_warmup.index < test_end]

        test_strategy = copy.deepcopy(strategy)
        engine = BacktestEngine(strategy=test_strategy, config=wf_config)
        full_report = engine.run({symbol: test_with_warmup})

        # Filter to test period only (exclude warmup trades)
        test_start_dt = (
            test_start.to_pydatetime() if hasattr(test_start, "to_pydatetime") else test_start
        )
        test_trades = [
            t
            for t in full_report.trades
            if t.entry_order.created_at is not None and t.entry_order.created_at >= test_start_dt
        ]
        test_equity = full_report.equity_curve[full_report.equity_curve.index >= test_start]

        if len(test_equity) < 2:
            test_sharpe = 0.0
            test_return_val = 0.0
            test_dd = 0.0
            test_trade_count = 0
        else:
            filtered_report = BacktestReport(
                trades=test_trades,
                equity_curve=test_equity,
                initial_balance=float(test_equity.iloc[0]),
                final_balance=float(test_equity.iloc[-1]),
                timeframe=wf_config.trading.timeframe,
            )
            test_sharpe = filtered_report.sharpe_ratio
            test_return_val = filtered_report.total_return
            test_dd = filtered_report.max_drawdown
            test_trade_count = filtered_report.total_trades

        results.append(
            WalkForwardWindow(
                window_index=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_params={"filters": "fixed"},
                train_sharpe=train_report.sharpe_ratio,
                train_return=train_report.total_return,
                test_sharpe=test_sharpe,
                test_return=test_return_val,
                test_trades=test_trade_count,
                test_max_drawdown=test_dd,
            )
        )

        if progress and task is not None:
            progress.advance(task)

    return WalkForwardReport(windows=results, strategy_name=strategy_name)
