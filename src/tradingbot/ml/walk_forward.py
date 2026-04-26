"""ML Walk-Forward trainer — purged expanding window with embargo."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tradingbot.ml.features import build_feature_matrix
from tradingbot.ml.targets import (
    build_target,
    build_target_atr,
    build_target_triple_barrier,
)
from tradingbot.ml.trainer import LGBMTrainer

VALID_TARGET_KINDS = ("binary", "atr", "triple-barrier")


def _build_target_dispatch(
    df: pd.DataFrame,
    target_kind: str,
    forward_candles: int,
    threshold: float,
    atr_mult: float,
    atr_period: int = 14,
) -> pd.Series:
    """Pick the right target builder based on ``target_kind``.

    Centralizes the dispatch so MLWalkForwardTrainer / MLStrategyWalkForward
    stay aligned (binary uses ``threshold``; atr / triple-barrier use
    ``atr_mult``).
    """
    if target_kind == "binary":
        return build_target(df, forward_candles=forward_candles, threshold=threshold)
    if target_kind == "atr":
        return build_target_atr(
            df,
            forward_candles=forward_candles,
            atr_mult=atr_mult,
            atr_period=atr_period,
        )
    if target_kind == "triple-barrier":
        return build_target_triple_barrier(
            df,
            forward_candles=forward_candles,
            atr_mult=atr_mult,
            atr_period=atr_period,
            threshold=threshold,
        )
    raise ValueError(
        f"Unknown target_kind={target_kind!r}; "
        f"expected one of {VALID_TARGET_KINDS}"
    )

log = logging.getLogger(__name__)

EMBARGO_CANDLES = 150  # ~3x max indicator lookback (52) for safer purging
MIN_VAL_FOR_EARLY_STOPPING = 1000  # below this, fall back to fixed rounds

CANDLES_PER_MONTH = {
    "1m": 43200, "5m": 8640, "15m": 2880, "30m": 1440,
    "1h": 720, "2h": 360, "4h": 180, "6h": 120,
    "12h": 60, "1d": 30,
}


def candles_per_month(timeframe: str) -> int:
    """Approximate candles per month for a timeframe."""
    return CANDLES_PER_MONTH.get(timeframe, 720)


def make_expanding_windows(
    n: int,
    train_size: int,
    test_size: int,
    embargo: int = EMBARGO_CANDLES,
) -> list[tuple[int, int, int]]:
    """Expanding-window splits with embargo.

    Returns list of (train_end_idx, test_start_idx, test_end_idx).
    """
    windows: list[tuple[int, int, int]] = []
    train_end = train_size
    while train_end + embargo + test_size <= n:
        test_start = train_end + embargo
        test_end = min(test_start + test_size, n)
        windows.append((train_end, test_start, test_end))
        train_end += test_size  # expand by one test window
    return windows


@dataclass
class MLWalkForwardReport:
    """Results from ML walk-forward validation.

    The ``holdout_*_proba`` / ``holdout_y_true`` arrays cover the *eval half*
    of the holdout — the same slice ``holdout_auc`` / ``holdout_precision``
    were computed on. ``holdout_calibrated_proba`` is None when the
    calibrator could not be fit (single-class cal half). They exist so the
    diagnostics command can compute calibration error / distribution stats
    without re-deriving the holdout split.
    """

    windows: list[dict] = field(default_factory=list)
    avg_auc: float = 0.0
    avg_precision: float = 0.0
    holdout_auc: float = 0.0
    holdout_precision: float = 0.0
    model_path: Path | None = None
    feature_importance: dict = field(default_factory=dict)
    holdout_y_true: np.ndarray | None = None
    holdout_raw_proba: np.ndarray | None = None
    holdout_calibrated_proba: np.ndarray | None = None


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
        target_kind: str = "binary",
        atr_mult: float = 1.0,
        model_dir: Path = Path("models"),
        lgbm_params: dict | None = None,
    ):
        if target_kind not in VALID_TARGET_KINDS:
            raise ValueError(
                f"Unknown target_kind={target_kind!r}; "
                f"expected one of {VALID_TARGET_KINDS}"
            )
        self.symbol = symbol
        self.timeframe = timeframe
        self.train_months = train_months
        self.test_months = test_months
        self.forward_candles = forward_candles
        self.threshold = threshold
        self.target_kind = target_kind
        self.atr_mult = atr_mult
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
        target = _build_target_dispatch(
            df_feat,
            target_kind=self.target_kind,
            forward_candles=self.forward_candles,
            threshold=self.threshold,
            atr_mult=self.atr_mult,
        )

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

        report = MLWalkForwardReport()

        # Path B: single training on train_pool with inner train/val split.
        # Inner val drives early stopping; no fold loop, no median heuristic.
        val_split = int(len(df_train_pool) * 0.8)
        val_start = val_split + EMBARGO_CANDLES
        val_size = len(df_train_pool) - val_start

        if val_size >= MIN_VAL_FOR_EARLY_STOPPING:
            X_tr = df_train_pool[feature_cols].iloc[:val_split]
            X_val = df_train_pool[feature_cols].iloc[val_start:]
            y_tr = target_train_pool.iloc[:val_split]
            y_val = target_train_pool.iloc[val_start:]
            final_model = self.trainer.train(X_tr, y_tr, X_val, y_val)
            val_metrics = self.trainer.evaluate(final_model, X_val, y_val)
            report.avg_auc = round(val_metrics["auc"], 4)
            report.avg_precision = round(val_metrics["precision"], 4)
            report.windows.append({
                "split": "inner_val",
                "n_train": len(X_tr),
                "n_val": len(X_val),
                "best_iteration": final_model.best_iteration,
                **val_metrics,
            })
            log.info(
                f"ML WF: trained with early stopping — "
                f"inner_val AUC={val_metrics['auc']:.4f}, "
                f"precision={val_metrics['precision']:.4f}, "
                f"best_iter={final_model.best_iteration}, "
                f"n_train={len(X_tr)}, n_val={len(X_val)}"
            )
        else:
            # Val set too small — fixed rounds without early stopping
            X_train_all = df_train_pool[feature_cols]
            y_train_all = target_train_pool
            final_model = self.trainer.train(X_train_all, y_train_all, fixed_rounds=300)
            report.windows.append({
                "split": "fixed_rounds",
                "n_train": len(X_train_all),
                "n_val": 0,
                "best_iteration": 300,
            })
            log.info(
                f"ML WF: trained with fixed_rounds=300 "
                f"(val_size={val_size} below {MIN_VAL_FOR_EARLY_STOPPING})"
            )

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

            # Expose eval-half predictions so the diagnostics CLI can compute
            # calibration error + distribution stats without re-deriving the
            # holdout split. Calibrated probs are produced via the calibrator
            # we just fit (which itself only saw the cal half — no leakage).
            raw_eval = np.asarray(final_model.predict(X_eval), dtype=float)
            report.holdout_y_true = y_eval.to_numpy(dtype=float)
            report.holdout_raw_proba = raw_eval
            if calibrator is not None:
                report.holdout_calibrated_proba = np.asarray(
                    calibrator.transform(raw_eval), dtype=float
                )

        # Feature importance
        importance = final_model.feature_importance(importance_type="gain")
        report.feature_importance = dict(
            sorted(zip(feature_cols, importance), key=lambda x: -x[1])
        )

        # Save final model + calibrator. train_start / train_end track the
        # *actual* slice the model fit on (df_train_pool), so downstream tools
        # like ml-backtest can avoid evaluating on candles the model has seen.
        # data_end keeps the original "last valid candle" for reference;
        # holdout_* are None when no holdout was reserved.
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
            "target_kind": self.target_kind,
            "atr_mult": self.atr_mult,
            "avg_win_loss_ratio": avg_win_loss_ratio,
            "train_start": str(df_train_pool.index[0]),
            "train_end": str(df_train_pool.index[-1]),
            "holdout_start": (
                str(df_holdout.index[0]) if len(df_holdout) > 0 else None
            ),
            "holdout_end": (
                str(df_holdout.index[-1]) if len(df_holdout) > 0 else None
            ),
            "data_end": str(df_valid.index[-1]),
        }
        report.model_path = self.trainer.save(
            final_model, self.symbol, self.timeframe, meta, feature_cols,
            self.model_dir, calibrator=calibrator,
        )

        return report
