"""ML Walk-Forward trainer — purged expanding window with embargo."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tradingbot.ml.features import WARMUP_CANDLES, build_feature_matrix
from tradingbot.ml.targets import build_target
from tradingbot.ml.trainer import LGBMTrainer

log = logging.getLogger(__name__)

EMBARGO_CANDLES = 52  # Same as max indicator lookback


@dataclass
class MLWalkForwardReport:
    """Results from ML walk-forward validation."""

    windows: list[dict] = field(default_factory=list)
    avg_auc: float = 0.0
    avg_precision: float = 0.0
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

    def run(self, df: pd.DataFrame) -> MLWalkForwardReport:
        """Run walk-forward ML training and evaluation.

        Args:
            df: Full OHLCV DataFrame for the symbol.

        Returns:
            MLWalkForwardReport with per-window metrics and final model path.
        """
        # Build features and target on full data
        df_feat, feature_cols = build_feature_matrix(df.copy())
        target = build_target(df_feat, self.forward_candles, self.threshold)

        # Valid rows: features not NaN and target not NaN
        valid_mask = df_feat[feature_cols].notna().all(axis=1) & target.notna()
        df_valid = df_feat[valid_mask]
        target_valid = target[valid_mask]

        if len(df_valid) < 200:
            log.warning(f"ML WF: insufficient data ({len(df_valid)} valid rows)")
            return MLWalkForwardReport()

        # Dynamic scale_pos_weight based on actual positive rate
        pos_rate = float(target_valid.mean())
        spw = min(2.0, max(1.0, 0.5 / pos_rate))
        self.trainer.params["scale_pos_weight"] = round(spw, 2)
        log.info(f"ML WF: positive_rate={pos_rate:.4f}, scale_pos_weight={spw:.2f}")

        # Create expanding windows
        windows = self._create_windows(df_valid)
        if not windows:
            log.warning("ML WF: no valid windows created")
            return MLWalkForwardReport()

        report = MLWalkForwardReport()

        for i, (train_end_idx, test_start_idx, test_end_idx) in enumerate(windows):
            X_train = df_valid[feature_cols].iloc[:train_end_idx]
            y_train = target_valid.iloc[:train_end_idx]
            X_test = df_valid[feature_cols].iloc[test_start_idx:test_end_idx]
            y_test = target_valid.iloc[test_start_idx:test_end_idx]

            if len(X_train) < 100 or len(X_test) < 30:
                continue

            # Split train into train/val for early stopping (80/20 with embargo)
            # Need at least 1000 val rows for stable early stopping signal
            MIN_VAL_FOR_EARLY_STOPPING = 1000
            val_split = int(len(X_train) * 0.8)
            val_start = val_split + EMBARGO_CANDLES
            val_size = len(X_train) - val_start

            if val_size >= MIN_VAL_FOR_EARLY_STOPPING:
                X_tr = X_train.iloc[:val_split]
                X_val = X_train.iloc[val_start:]
                y_tr = y_train.iloc[:val_split]
                y_val = y_train.iloc[val_start:]
                model = self.trainer.train(X_tr, y_tr, X_val, y_val)
            else:
                # Val set too small — use fixed rounds, no early stopping
                model = self.trainer.train(X_train, y_train, fixed_rounds=300)
            metrics = self.trainer.evaluate(model, X_test, y_test)
            metrics["window"] = i
            metrics["n_train"] = len(X_train)
            metrics["best_iteration"] = model.best_iteration
            report.windows.append(metrics)

            log.info(f"ML WF window {i}: AUC={metrics['auc']:.4f}, precision={metrics['precision']:.4f}, best_iter={model.best_iteration}, train={len(X_train)}, test={len(X_test)}")

        if report.windows:
            report.avg_auc = round(np.mean([w["auc"] for w in report.windows]), 4)
            report.avg_precision = round(np.mean([w["precision"] for w in report.windows]), 4)

        # Train final model on ALL valid data — no early stopping, fixed 300 rounds
        # (median best_iteration from windows is unreliable when most use fixed_rounds)
        final_rounds = 300
        log.info(f"ML WF: final model fixed_rounds={final_rounds}")

        X_all = df_valid[feature_cols]
        y_all = target_valid
        final_model = self.trainer.train(X_all, y_all, fixed_rounds=final_rounds)

        # Feature importance
        importance = final_model.feature_importance(importance_type="gain")
        report.feature_importance = dict(
            sorted(zip(feature_cols, importance), key=lambda x: -x[1])
        )

        # Save final model
        meta = {
            "avg_auc": report.avg_auc,
            "avg_precision": report.avg_precision,
            "n_windows": len(report.windows),
            "n_total_samples": len(df_valid),
            "forward_candles": self.forward_candles,
            "threshold": self.threshold,
            "train_start": str(df_valid.index[0]),
            "train_end": str(df_valid.index[-1]),
        }
        report.model_path = self.trainer.save(
            final_model, self.symbol, self.timeframe, meta, feature_cols, self.model_dir
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
