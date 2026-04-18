"""ML Walk-Forward trainer — purged expanding window with embargo."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tradingbot.ml.features import build_feature_matrix
from tradingbot.ml.targets import build_target
from tradingbot.ml.trainer import LGBMTrainer

log = logging.getLogger(__name__)

EMBARGO_CANDLES = 150  # ~3x max indicator lookback (52) for safer purging


@dataclass
class MLWalkForwardReport:
    """Results from ML walk-forward validation."""

    windows: list[dict] = field(default_factory=list)
    avg_auc: float = 0.0
    avg_precision: float = 0.0
    holdout_auc: float = 0.0
    holdout_precision: float = 0.0
    model_path: Path | None = None
    feature_importance: dict = field(default_factory=dict)


class MLWalkForwardTrainer:
    """Walk-forward training with purged expanding windows and embargo."""

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        train_months: int = 3,
        test_months: int = 1,
        forward_candles: int = 4,
        threshold: float = 0.006,
        model_dir: Path = Path("models"),
        lgbm_params: dict | None = None,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.train_months = train_months
        self.test_months = test_months
        self.forward_candles = forward_candles
        self.threshold = threshold
        self.model_dir = model_dir
        self.trainer = LGBMTrainer(lgbm_params)

    def run(
        self,
        df: pd.DataFrame,
        external_df: pd.DataFrame | None = None,
    ) -> MLWalkForwardReport:
        """Run walk-forward ML training and evaluation.

        Args:
            df: Full OHLCV DataFrame for the symbol.
            external_df: Optional external features (kimchi, funding rate, etc.).

        Returns:
            MLWalkForwardReport with per-window metrics and final model path.
        """
        # Build features and target on full data
        df_feat, feature_cols = build_feature_matrix(df.copy(), external_df=external_df)
        target = build_target(df_feat, self.forward_candles, self.threshold)

        # Forward return per candle — needed later for empirical win/loss ratio
        fwd_return = df_feat["close"].pct_change(self.forward_candles).shift(-self.forward_candles)

        # Valid rows: features not NaN and target not NaN
        valid_mask = df_feat[feature_cols].notna().all(axis=1) & target.notna()
        df_valid = df_feat[valid_mask]
        target_valid = target[valid_mask]
        fwd_return_valid = fwd_return[valid_mask]

        if len(df_valid) < 200:
            log.warning(f"ML WF: insufficient data ({len(df_valid)} valid rows)")
            return MLWalkForwardReport()

        # Dynamic scale_pos_weight based on actual positive rate
        pos_rate = float(target_valid.mean())
        if pos_rate < 0.01:
            log.warning(f"ML WF: positive rate too low ({pos_rate:.4f}), skipping")
            return MLWalkForwardReport()
        spw = min(2.0, max(1.0, 0.5 / pos_rate))
        self.trainer.params["scale_pos_weight"] = round(spw, 2)
        log.info(f"ML WF: positive_rate={pos_rate:.4f}, scale_pos_weight={spw:.2f}")

        # Reserve last 20% as holdout (never seen during walk-forward).
        # Embargo + forward_candles at the boundary prevents target leakage:
        # the last rows of train_pool have targets that depend on prices
        # inside the holdout window.
        holdout_split = int(len(df_valid) * 0.8)
        boundary_gap = EMBARGO_CANDLES + self.forward_candles
        df_train_pool = df_valid.iloc[:holdout_split]
        target_train_pool = target_valid.iloc[:holdout_split]
        fwd_return_train_pool = fwd_return_valid.iloc[:holdout_split]
        df_holdout = df_valid.iloc[holdout_split + boundary_gap:]
        target_holdout = target_valid.iloc[holdout_split + boundary_gap:]

        # Empirical avg_win/avg_loss for Kelly sizing — uses train_pool only
        # so the ratio shipped with the model never sees holdout data.
        # "Loss" here means "non-win per target definition" (return <= threshold),
        # so it includes small positive returns below the fee/profit threshold.
        wins = fwd_return_train_pool[fwd_return_train_pool > self.threshold]
        losses = fwd_return_train_pool[fwd_return_train_pool <= self.threshold]
        if len(wins) > 0 and len(losses) > 0:
            avg_win = float(wins.mean())
            avg_loss = float(abs(losses.mean()))
            avg_win_loss_ratio = round(avg_win / avg_loss, 3) if avg_loss > 0 else 1.5
            # Cap to avoid overconfident Kelly when distribution is skewed by outliers
            avg_win_loss_ratio = max(0.5, min(avg_win_loss_ratio, 5.0))
        else:
            avg_win_loss_ratio = 1.5
        log.info(f"ML WF: empirical avg_win_loss_ratio={avg_win_loss_ratio}")

        log.info(
            f"ML WF: train_pool={len(df_train_pool)}, holdout={len(df_holdout)} "
            f"(split at {holdout_split}, boundary_gap={boundary_gap})"
        )

        # Create expanding windows on train pool only
        windows = self._create_windows(df_train_pool)
        if not windows:
            log.warning("ML WF: no valid windows created")
            return MLWalkForwardReport()

        report = MLWalkForwardReport()
        es_iterations: list[int] = []  # best_iteration from early-stopped windows

        for i, (train_end_idx, test_start_idx, test_end_idx) in enumerate(windows):
            X_train = df_train_pool[feature_cols].iloc[:train_end_idx]
            y_train = target_train_pool.iloc[:train_end_idx]
            X_test = df_train_pool[feature_cols].iloc[test_start_idx:test_end_idx]
            y_test = target_train_pool.iloc[test_start_idx:test_end_idx]

            if len(X_train) < 100 or len(X_test) < 30:
                continue

            # Split train into train/val for early stopping (80/20 with embargo)
            # Need at least 1000 val rows for stable early stopping signal
            MIN_VAL_FOR_EARLY_STOPPING = 1000
            val_split = int(len(X_train) * 0.8)
            val_start = val_split + EMBARGO_CANDLES
            val_size = len(X_train) - val_start

            used_early_stopping = False
            if val_size >= MIN_VAL_FOR_EARLY_STOPPING:
                X_tr = X_train.iloc[:val_split]
                X_val = X_train.iloc[val_start:]
                y_tr = y_train.iloc[:val_split]
                y_val = y_train.iloc[val_start:]
                model = self.trainer.train(X_tr, y_tr, X_val, y_val)
                used_early_stopping = True
            else:
                # Val set too small — use fixed rounds, no early stopping
                model = self.trainer.train(X_train, y_train, fixed_rounds=300)
            metrics = self.trainer.evaluate(model, X_test, y_test)
            metrics["window"] = i
            metrics["n_train"] = len(X_train)
            metrics["best_iteration"] = model.best_iteration
            report.windows.append(metrics)

            if used_early_stopping and model.best_iteration > 0:
                es_iterations.append(model.best_iteration)

            log.info(f"ML WF window {i}: AUC={metrics['auc']:.4f}, precision={metrics['precision']:.4f}, best_iter={model.best_iteration}, train={len(X_train)}, test={len(X_test)}")

        if report.windows:
            report.avg_auc = round(np.mean([w["auc"] for w in report.windows]), 4)
            report.avg_precision = round(np.mean([w["precision"] for w in report.windows]), 4)

        # Train final model on train pool (NOT all data — holdout reserved)
        if es_iterations:
            final_rounds = int(np.median(es_iterations))
            log.info(
                f"ML WF: final model rounds={final_rounds} "
                f"(median of {len(es_iterations)} early-stopped windows)"
            )
        else:
            final_rounds = 300
            log.info(f"ML WF: no early stopping data, using default rounds={final_rounds}")

        X_train_all = df_train_pool[feature_cols]
        y_train_all = target_train_pool
        final_model = self.trainer.train(X_train_all, y_train_all, fixed_rounds=final_rounds)

        # Evaluate on holdout set. Split holdout in half: first half is a
        # true evaluation set (never touches the model), second half fits the
        # calibrator. Prevents calibrator-via-holdout leakage in reported metrics.
        calibrator = None
        if len(df_holdout) >= 60:
            mid = len(df_holdout) // 2
            X_eval = df_holdout[feature_cols].iloc[:mid]
            y_eval = target_holdout.iloc[:mid]
            X_cal = df_holdout[feature_cols].iloc[mid:]
            y_cal = target_holdout.iloc[mid:]

            holdout_metrics = self.trainer.evaluate(final_model, X_eval, y_eval)
            report.holdout_auc = holdout_metrics["auc"]
            report.holdout_precision = holdout_metrics["precision"]

            if report.avg_auc > 0 and holdout_metrics["auc"] < report.avg_auc - 0.05:
                log.warning(
                    f"ML WF: holdout AUC ({holdout_metrics['auc']:.4f}) is significantly "
                    f"lower than walk-forward avg ({report.avg_auc:.4f}) — possible overfitting"
                )

            log.info(
                f"ML WF holdout: AUC={holdout_metrics['auc']:.4f}, "
                f"precision={holdout_metrics['precision']:.4f}, "
                f"p_value={holdout_metrics.get('auc_p_value', 'N/A')}, "
                f"n_eval={len(X_eval)}, n_cal={len(X_cal)}"
            )

            # Fit probability calibrator on the calibration half only
            calibrator = self.trainer.calibrate(final_model, X_cal, y_cal)

        # Feature importance
        importance = final_model.feature_importance(importance_type="gain")
        report.feature_importance = dict(
            sorted(zip(feature_cols, importance), key=lambda x: -x[1])
        )

        # Save final model + calibrator
        meta = {
            "avg_auc": report.avg_auc,
            "avg_precision": report.avg_precision,
            "holdout_auc": report.holdout_auc,
            "holdout_precision": report.holdout_precision,
            "n_windows": len(report.windows),
            "n_train_pool": len(df_train_pool),
            "n_holdout": len(df_holdout),
            "n_total_samples": len(df_valid),
            "forward_candles": self.forward_candles,
            "threshold": self.threshold,
            "avg_win_loss_ratio": avg_win_loss_ratio,
            "train_start": str(df_valid.index[0]),
            "train_end": str(df_valid.index[-1]),
        }
        report.model_path = self.trainer.save(
            final_model, self.symbol, self.timeframe, meta, feature_cols,
            self.model_dir, calibrator=calibrator,
        )

        return report

    def _create_windows(
        self, df: pd.DataFrame
    ) -> list[tuple[int, int, int]]:
        """Create expanding window splits with embargo.

        Returns list of (train_end_idx, test_start_idx, test_end_idx).
        """
        n = len(df)
        # Estimate candles per month from timeframe
        candles_per_month = self._candles_per_month()
        test_size = self.test_months * candles_per_month
        min_train = self.train_months * candles_per_month

        windows = []
        train_end = min_train

        while train_end + EMBARGO_CANDLES + test_size <= n:
            test_start = train_end + EMBARGO_CANDLES
            test_end = min(test_start + test_size, n)
            windows.append((train_end, test_start, test_end))
            train_end += test_size  # Expand by one test window

        return windows

    def _candles_per_month(self) -> int:
        """Approximate candles per month based on timeframe."""
        tf_map = {
            "1m": 43200, "5m": 8640, "15m": 2880, "30m": 1440,
            "1h": 720, "2h": 360, "4h": 180, "6h": 120,
            "12h": 60, "1d": 30,
        }
        return tf_map.get(self.timeframe, 720)
