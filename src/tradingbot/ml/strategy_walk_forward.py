"""Walk-forward evaluation for the LGBM strategy.

Trains a fresh model per window (Path B: single training with inner train/val
split + early stopping), injects it into LGBMStrategy, and runs the backtest
engine on the test window. Unlike the rule-based ``walk-forward`` command —
which uses a single saved model for every test window — each window here has
no exposure to its test data during training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from tradingbot.backtest.engine import BacktestEngine
from tradingbot.backtest.report import BacktestReport
from tradingbot.config import AppConfig
from tradingbot.data.external_fetcher import build_external_df
from tradingbot.ml.features import WARMUP_CANDLES, build_feature_matrix
from tradingbot.ml.targets import build_target
from tradingbot.ml.trainer import LGBMTrainer
from tradingbot.ml.walk_forward import (
    EMBARGO_CANDLES,
    MIN_VAL_FOR_EARLY_STOPPING,
    candles_per_month,
    make_expanding_windows,
)
from tradingbot.strategy.base import StrategyParams
from tradingbot.strategy.lgbm_strategy import LGBMStrategy

log = logging.getLogger(__name__)


class _SkipWindowError(Exception):
    """Internal signal that a window cannot be trained (e.g., pos rate too low)."""


@dataclass
class MLStrategyWalkForwardReport:
    """Per-window backtest results for ML walk-forward evaluation."""

    windows: list[dict] = field(default_factory=list)
    avg_sharpe: float = 0.0
    cumulative_return_pct: float = 0.0
    total_trades: int = 0
    avg_win_rate: float = 0.0
    final_equity_multiple: float = 1.0  # product of (1 + window_return)
    n_windows: int = 0
    n_skipped: int = 0


class MLStrategyWalkForward:
    """Per-window LGBM training + strategy backtest.

    The runner trains a model on every available window using Path B
    (single training, inner train/val split for early stopping, isotonic
    calibration on a held-out tail of the train slice), then injects that
    model into a fresh ``LGBMStrategy`` instance and runs the backtest
    engine on just the test window.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        train_months: int = 6,
        test_months: int = 2,
        forward_candles: int = 4,
        threshold: float = 0.006,
        entry_threshold: float = 0.45,
        exit_threshold: float = 0.30,
        external_data_dir: str | Path | None = None,
        config: AppConfig | None = None,
        lgbm_params: dict | None = None,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.train_months = train_months
        self.test_months = test_months
        self.forward_candles = forward_candles
        self.threshold = threshold
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.external_data_dir = (
            Path(external_data_dir) if external_data_dir else None
        )
        self.config = config or AppConfig()
        self.trainer = LGBMTrainer(lgbm_params)

    def run(self, df: pd.DataFrame) -> MLStrategyWalkForwardReport:
        """Run walk-forward and return per-window backtest metrics."""
        # Dedup + sort up front so index lookups in _slice_test_ohlcv always
        # return integer positions; otherwise duplicate timestamps cause
        # df.index.get_loc to return a slice and the iloc call below crashes.
        df = df[~df.index.duplicated(keep="last")].sort_index()
        external_df = (
            build_external_df(df, self.external_data_dir)
            if self.external_data_dir
            else None
        )

        df_feat, feature_cols = build_feature_matrix(df, external_df=external_df)
        target = build_target(df_feat, self.forward_candles, self.threshold)
        fwd_return = (
            df_feat["close"].pct_change(self.forward_candles).shift(-self.forward_candles)
        )

        valid_mask = df_feat[feature_cols].notna().all(axis=1) & target.notna()
        df_valid = df_feat[valid_mask]
        target_valid = target[valid_mask]
        fwd_return_valid = fwd_return[valid_mask]

        if len(df_valid) < 200:
            log.warning(f"ML strategy WF: insufficient data ({len(df_valid)} valid rows)")
            return MLStrategyWalkForwardReport()

        cpm = candles_per_month(self.timeframe)
        train_size = self.train_months * cpm
        test_size = self.test_months * cpm
        windows = make_expanding_windows(len(df_valid), train_size, test_size)

        if not windows:
            log.warning(
                f"ML strategy WF: no windows produced "
                f"(train_size={train_size}, test_size={test_size}, valid={len(df_valid)})"
            )
            return MLStrategyWalkForwardReport()

        log.info(
            f"ML strategy WF: {len(windows)} windows, "
            f"train_size={train_size}, test_size={test_size}, embargo={EMBARGO_CANDLES}"
        )

        results: list[dict] = []
        equity_multiple = 1.0
        n_skipped = 0

        for window_idx, (train_end_idx, test_start_idx, test_end_idx) in enumerate(windows):
            train_data = df_valid.iloc[:train_end_idx]
            target_train = target_valid.iloc[:train_end_idx]
            fwd_return_train = fwd_return_valid.iloc[:train_end_idx]

            try:
                model, calibrator, win_loss_ratio = self._train_one_window(
                    train_data, target_train, fwd_return_train, feature_cols
                )
            except _SkipWindowError as exc:
                log.warning(f"ML strategy WF window {window_idx}: skipped — {exc}")
                n_skipped += 1
                continue

            strategy = LGBMStrategy(
                StrategyParams(
                    values={
                        "entry_threshold": self.entry_threshold,
                        "exit_threshold": self.exit_threshold,
                        "external_data_dir": (
                            str(self.external_data_dir)
                            if self.external_data_dir
                            else None
                        ),
                    }
                )
            )
            strategy.timeframe = self.timeframe
            strategy.symbols = [self.symbol]
            strategy.set_model(
                symbol=self.symbol,
                model=model,
                calibrator=calibrator,
                feature_cols=feature_cols,
                win_loss_ratio=win_loss_ratio,
            )

            test_ohlcv = self._slice_test_ohlcv(
                df, df_valid, test_start_idx, test_end_idx
            )
            if test_ohlcv.empty:
                log.warning(f"ML strategy WF window {window_idx}: empty OHLCV slice")
                n_skipped += 1
                continue

            engine = BacktestEngine(strategy=strategy, config=self.config)
            full_report = engine.run({self.symbol: test_ohlcv})

            # Strip the warmup prefix from the report. The backtest sees
            # WARMUP_CANDLES rows before test_start_ts to let indicators
            # come up from NaN; equity sits flat there (no trades), but
            # those flat-zero returns deflate Sharpe and skew win-rate.
            # Pattern mirrors _walk_forward_combined in cli.py.
            test_start_ts = df_valid.index[test_start_idx]
            backtest_report = self._filter_to_test_window(full_report, test_start_ts)

            window_result = {
                "window": window_idx,
                "train_start": str(df_valid.index[0]),
                "train_end": str(df_valid.index[train_end_idx - 1]),
                "test_start": str(test_start_ts),
                "test_end": str(df_valid.index[test_end_idx - 1]),
                "n_train": int(train_end_idx),
                "n_test": int(test_end_idx - test_start_idx),
                "sharpe": float(backtest_report.sharpe_ratio),
                "return_pct": float(backtest_report.total_return * 100),
                "trades": int(backtest_report.total_trades),
                "win_rate": float(backtest_report.win_rate),
                "max_dd_pct": float(backtest_report.max_drawdown * 100),
                "final_balance": float(backtest_report.final_balance),
                "win_loss_ratio_used": float(win_loss_ratio),
            }
            results.append(window_result)
            equity_multiple *= 1.0 + backtest_report.total_return

            log.info(
                f"ML strategy WF window {window_idx}: "
                f"sharpe={window_result['sharpe']:.2f}, "
                f"return={window_result['return_pct']:.2f}%, "
                f"trades={window_result['trades']}, "
                f"win_rate={window_result['win_rate']:.2f}"
            )

        report = MLStrategyWalkForwardReport(
            windows=results,
            n_windows=len(results),
            n_skipped=n_skipped,
        )
        if results:
            report.avg_sharpe = round(
                sum(r["sharpe"] for r in results) / len(results), 4
            )
            report.cumulative_return_pct = round((equity_multiple - 1.0) * 100, 4)
            report.final_equity_multiple = round(equity_multiple, 6)
            report.total_trades = sum(r["trades"] for r in results)
            traded_windows = [r for r in results if r["trades"] > 0]
            report.avg_win_rate = round(
                sum(r["win_rate"] for r in traded_windows) / len(traded_windows), 4
            ) if traded_windows else 0.0

        return report

    def _train_one_window(
        self,
        train_data: pd.DataFrame,
        target_train: pd.Series,
        fwd_return_train: pd.Series,
        feature_cols: list[str],
    ) -> tuple[Any, Any, float]:
        """Path B training for a single window. Returns (model, calibrator, win_loss_ratio)."""
        pos_rate = float(target_train.mean())
        if pos_rate < 0.01:
            raise _SkipWindowError(f"positive rate too low ({pos_rate:.4f})")

        spw = min(2.0, max(1.0, 0.5 / pos_rate))
        self.trainer.params["scale_pos_weight"] = round(spw, 2)

        wins = fwd_return_train[fwd_return_train > self.threshold]
        losses = fwd_return_train[fwd_return_train <= self.threshold]
        if len(wins) > 0 and len(losses) > 0:
            avg_win = float(wins.mean())
            avg_loss = float(abs(losses.mean()))
            ratio = avg_win / avg_loss if avg_loss > 0 else 1.5
            win_loss_ratio = max(0.5, min(round(ratio, 3), 5.0))
        else:
            win_loss_ratio = 1.5

        # Path B: inner train/val split with embargo for early stopping.
        # Calibrator is fit on whichever held-out slice exists — either the
        # inner_val (when early stopping ran) or the last ~15% of train_data
        # (fixed_rounds fallback). Both keep the calibrator's data disjoint
        # from the slice used to fit tree weights.
        val_split = int(len(train_data) * 0.8)
        val_start = val_split + EMBARGO_CANDLES
        val_size = len(train_data) - val_start

        if val_size >= MIN_VAL_FOR_EARLY_STOPPING:
            X_tr = train_data[feature_cols].iloc[:val_split]
            X_val = train_data[feature_cols].iloc[val_start:]
            y_tr = target_train.iloc[:val_split]
            y_val = target_train.iloc[val_start:]
            model = self.trainer.train(X_tr, y_tr, X_val, y_val)
            X_cal, y_cal = X_val, y_val
        else:
            X_train_all = train_data[feature_cols]
            y_train_all = target_train
            model = self.trainer.train(X_train_all, y_train_all, fixed_rounds=300)
            cal_start = int(len(train_data) * 0.85)
            X_cal = train_data[feature_cols].iloc[cal_start:]
            y_cal = target_train.iloc[cal_start:]

        calibrator = None
        if len(X_cal) >= 30:
            calibrator = self.trainer.calibrate(model, X_cal, y_cal)

        return model, calibrator, win_loss_ratio

    def _filter_to_test_window(
        self,
        full_report: BacktestReport,
        test_start_ts: pd.Timestamp,
    ) -> BacktestReport:
        """Drop the warmup prefix from the engine's report.

        Returns a fresh ``BacktestReport`` whose equity_curve and trades only
        cover ``test_start_ts`` onwards. If the equity curve has fewer than
        2 points after filtering, returns the original report (rare — would
        indicate the warmup consumed the whole window).
        """
        test_equity = full_report.equity_curve[
            full_report.equity_curve.index >= test_start_ts
        ]
        if len(test_equity) < 2:
            return full_report

        test_start_dt = test_start_ts.to_pydatetime()
        test_trades = [
            t for t in full_report.trades
            if t.entry_order.created_at is not None
            and t.entry_order.created_at >= test_start_dt
        ]
        return BacktestReport(
            trades=test_trades,
            equity_curve=test_equity,
            initial_balance=float(test_equity.iloc[0]),
            final_balance=float(test_equity.iloc[-1]),
            timeframe=full_report.timeframe,
        )

    def _slice_test_ohlcv(
        self,
        df: pd.DataFrame,
        df_valid: pd.DataFrame,
        test_start_idx: int,
        test_end_idx: int,
    ) -> pd.DataFrame:
        """Slice raw OHLCV for the test window plus warmup before it.

        ``df`` is the original OHLCV; ``df_valid`` indexes into rows where
        all features and target are non-NaN. We map the test window endpoints
        to positions in ``df`` and prepend ``WARMUP_CANDLES`` rows so the
        engine can pre-compute indicators correctly. Predictions on those
        warmup candles return None (NaN features) so they generate no trades.
        """
        test_start_ts = df_valid.index[test_start_idx]
        test_end_ts = df_valid.index[test_end_idx - 1]

        df_pos_start = df.index.get_loc(test_start_ts)
        df_pos_end = df.index.get_loc(test_end_ts)

        warmup_start = max(0, df_pos_start - WARMUP_CANDLES)
        return df.iloc[warmup_start : df_pos_end + 1]
