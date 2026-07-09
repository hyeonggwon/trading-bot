"""LightGBM trainer — train, evaluate, save, load models."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression

log = logging.getLogger(__name__)

DEFAULT_LGBM_PARAMS = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "verbose": -1,
    # Tree structure — conservative to prevent overfitting
    "num_leaves": 15,
    "max_depth": 4,
    "min_data_in_leaf": 50,
    "min_sum_hessian_in_leaf": 1.0,
    # Regularization — stronger than before
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    # Class imbalance — walk_forward.py dynamically computes scale_pos_weight
    # based on actual positive rate. Default 1.0 avoids probability distortion.
    "scale_pos_weight": 1.0,
    # Learning
    "learning_rate": 0.05,
    "n_estimators": 300,
    # Speed
    "num_threads": -1,
    "seed": 42,
}


class LGBMTrainer:
    """Train and manage LightGBM models for trading."""

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = {**DEFAULT_LGBM_PARAMS, **(params or {})}

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        early_stopping_rounds: int = 100,
        fixed_rounds: int | None = None,
    ) -> lgb.Booster:
        """Train a LightGBM model.

        Args:
            fixed_rounds: If set, override n_estimators and disable early stopping.

        Returns:
            lgb.Booster
        """
        import lightgbm as lgb

        # Extract n_estimators from params (used as num_boost_round)
        params = {k: v for k, v in self.params.items() if k != "n_estimators"}
        num_boost_round = fixed_rounds or self.params.get("n_estimators", 500)

        train_set = lgb.Dataset(X_train, label=y_train)

        callbacks: list[Callable[..., Any]] = [
            lgb.log_evaluation(period=0)
        ]  # suppress per-iteration logs
        valid_sets = [train_set]
        valid_names = ["train"]

        if X_val is not None and y_val is not None and len(X_val) > 0 and not fixed_rounds:
            val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
            valid_sets.append(val_set)
            valid_names.append("val")
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

        model = lgb.train(
            params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        n_val = len(X_val) if X_val is not None else 0
        log.info(
            f"LightGBM training complete: n_train={len(X_train)}, n_val={n_val}, "
            f"best_iter={model.best_iteration}"
        )
        return model

    def evaluate(
        self, model: lgb.Booster, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict[str, Any]:
        """Evaluate model on test data.

        Returns dict with auc, precision, recall, f1, n_test, positive_rate,
        auc_p_value (vs random 0.5), and auc_significant (p < 0.05).
        """
        from sklearn.metrics import (
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        proba = cast(np.ndarray, model.predict(X_test))
        y_pred = (proba > 0.5).astype(int)

        auc = float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else 0.5
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        # Baseline comparison: p-value for AUC vs random (0.5)
        # Uses Hanley-McNeil SE approximation for AUC
        n_pos = int(y_test.sum())
        n_neg = len(y_test) - n_pos
        p_value = 1.0
        if n_pos > 0 and n_neg > 0 and auc != 0.5:
            from scipy.stats import norm

            q1 = auc / (2 - auc)
            q2 = 2 * auc**2 / (1 + auc)
            se = (
                (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2))
                / (n_pos * n_neg)
            ) ** 0.5
            z = (auc - 0.5) / se if se > 0 else 0.0
            p_value = float(1 - norm.cdf(z))

        return {
            "auc": round(auc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "n_test": len(y_test),
            "positive_rate": round(float(y_test.mean()), 4),
            "auc_p_value": round(p_value, 6),
            "auc_significant": p_value < 0.05,
        }

    def calibrate(
        self, model: lgb.Booster, X_cal: pd.DataFrame, y_cal: pd.Series
    ) -> IsotonicRegression | None:
        """Fit isotonic calibrator on calibration data.

        Args:
            model: Trained LightGBM Booster.
            X_cal: Calibration feature matrix.
            y_cal: Calibration target.

        Returns:
            Fitted IsotonicRegression calibrator, or None if the calibration
            set has fewer than 2 classes (a constant mapping would silently
            crush all predictions to the majority class).
        """
        from sklearn.isotonic import IsotonicRegression

        if y_cal.nunique() < 2:
            log.warning(
                f"Calibrator skipped: calibration set has single class "
                f"(n={len(y_cal)}, pos_rate={float(y_cal.mean()):.4f})"
            )
            return None

        raw_proba = model.predict(X_cal)
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_proba, y_cal)
        return calibrator

    def save(
        self,
        model: lgb.Booster,
        symbol: str,
        timeframe: str,
        meta: dict[str, Any],
        feature_cols: list[str],
        model_dir: Path = Path("models"),
        calibrator: IsotonicRegression | None = None,
    ) -> Path:
        """Save model (.lgb), metadata (_meta.json), and optional calibrator (_cal.json).

        Returns path to saved model file.
        """
        model_dir.mkdir(parents=True, exist_ok=True)
        symbol_key = symbol.replace("/", "_")
        model_path = model_dir / f"lgbm_{symbol_key}_{timeframe}.lgb"
        meta_path = model_dir / f"lgbm_{symbol_key}_{timeframe}_meta.json"

        model.save_model(str(model_path))

        has_calibrator = False
        if calibrator is not None:
            cal_path = model_dir / f"lgbm_{symbol_key}_{timeframe}_cal.json"
            cal_data = {
                "x": calibrator.X_thresholds_.tolist(),
                "y": calibrator.y_thresholds_.tolist(),
            }
            cal_path.write_text(json.dumps(cal_data))
            has_calibrator = True
            log.info(f"Calibrator saved: {cal_path}")

        full_meta = {
            "trained_at": datetime.now(UTC).isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "n_features": len(feature_cols),
            "feature_names": feature_cols,
            "best_iteration": model.best_iteration,
            "has_calibrator": has_calibrator,
            **meta,
        }
        # Preserve tuner-produced keys across retrains: the trainers never put
        # these in ``meta``, so a plain refresh would otherwise silently revert
        # a tuned model to defaults on its NEXT retrain (the booster keeps the
        # params for one cycle, the meta record must survive indefinitely).
        # An explicit key in the new ``meta`` still wins.
        prior = self.load_meta(symbol, timeframe, model_dir) or {}
        for key in ("tuning", "entry_threshold", "exit_threshold"):
            if key not in full_meta and key in prior:
                full_meta[key] = prior[key]
        meta_path.write_text(json.dumps(full_meta, indent=2, default=str))

        log.info(f"LightGBM model saved: {model_path}")
        return model_path

    @staticmethod
    def load(symbol: str, timeframe: str, model_dir: Path = Path("models")) -> lgb.Booster | None:
        """Load a saved model. Returns lgb.Booster or None if not found."""
        import lightgbm as lgb

        symbol_key = symbol.replace("/", "_")
        model_path = model_dir / f"lgbm_{symbol_key}_{timeframe}.lgb"

        if not model_path.exists():
            log.warning(f"LightGBM model not found: {model_path}")
            return None

        return lgb.Booster(model_file=str(model_path))

    @staticmethod
    def load_calibrator(
        symbol: str, timeframe: str, model_dir: Path = Path("models")
    ) -> IsotonicRegression | None:
        """Load a saved calibrator. Returns IsotonicRegression or None if not found."""
        from scipy.interpolate import interp1d
        from sklearn.isotonic import IsotonicRegression

        symbol_key = symbol.replace("/", "_")
        cal_path = model_dir / f"lgbm_{symbol_key}_{timeframe}_cal.json"

        if not cal_path.exists():
            return None

        cal_data = json.loads(cal_path.read_text())
        x = np.array(cal_data["x"])
        y = np.array(cal_data["y"])

        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.X_thresholds_ = x
        calibrator.y_thresholds_ = y
        calibrator.X_min_ = float(x[0])
        calibrator.X_max_ = float(x[-1])
        calibrator.increasing_ = True
        # interp1d requires >=2 distinct points. In a low-signal regime the
        # model can predict a constant raw probability, so the saved isotonic
        # fit collapses to a single (x, y) pair — fall back to a constant map.
        if len(x) > 1:
            calibrator.f_ = interp1d(
                x,
                y,
                kind="linear",
                bounds_error=False,
                fill_value=(y[0], y[-1]),
            )
        else:
            const_y = float(y[0])
            calibrator.f_ = lambda val, _c=const_y: np.full_like(
                np.asarray(val, dtype=float),
                _c,
                dtype=float,
            )
        return calibrator

    @staticmethod
    def load_meta(
        symbol: str, timeframe: str, model_dir: Path = Path("models")
    ) -> dict[str, Any] | None:
        """Load model metadata. Returns dict or None if not found/unreadable.

        Corrupt meta counts as missing (same defensive read as load_catalog):
        every caller already handles None, and the pipeline's staleness check
        then retrains instead of aborting the whole run.
        """
        symbol_key = symbol.replace("/", "_")
        meta_path = model_dir / f"lgbm_{symbol_key}_{timeframe}_meta.json"

        if not meta_path.exists():
            return None

        try:
            meta: dict[str, Any] = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Unreadable model meta treated as missing: {meta_path} ({e})")
            return None
        return meta

    @staticmethod
    def load_catalog(model_dir: Path = Path("models")) -> list[dict[str, Any]]:
        """Summarize every saved model's meta for the dashboard catalog.

        Meta contents vary by pipeline phase (walk-forward, Optuna tuner and
        threshold tuner each add keys), so fields are extracted defensively —
        absent keys become None. Unreadable metas are skipped with a warning.
        Returns [] when the directory holds no models.
        """
        entries: list[dict[str, Any]] = []
        for meta_path in sorted(model_dir.glob("lgbm_*_meta.json")):
            try:
                meta = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                log.warning(f"Unreadable model meta skipped: {meta_path} ({e})")
                continue
            entries.append(
                {
                    "symbol": meta.get("symbol"),
                    "timeframe": meta.get("timeframe"),
                    "trained_at": meta.get("trained_at"),
                    "n_features": meta.get("n_features"),
                    "has_calibrator": meta.get("has_calibrator"),
                    "holdout_start": meta.get("holdout_start"),
                    "holdout_auc": meta.get("holdout_auc"),
                    "entry_threshold": meta.get("entry_threshold"),
                    "exit_threshold": meta.get("exit_threshold"),
                    "avg_win_loss_ratio": meta.get("avg_win_loss_ratio"),
                }
            )
        return entries
