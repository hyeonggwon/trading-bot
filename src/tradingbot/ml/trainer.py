"""LightGBM trainer — train, evaluate, save, load models."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

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

    def __init__(self, params: dict | None = None):
        self.params = {**DEFAULT_LGBM_PARAMS, **(params or {})}

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        early_stopping_rounds: int = 100,
        fixed_rounds: int | None = None,
    ):
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

        callbacks = [lgb.log_evaluation(period=0)]  # suppress per-iteration logs
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
        log.info(f"LightGBM training complete: n_train={len(X_train)}, n_val={n_val}, best_iter={model.best_iteration}")
        return model

    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
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

        proba = model.predict(X_test)
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
            se = ((auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2))
                  / (n_pos * n_neg)) ** 0.5
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

    def calibrate(self, model, X_cal: pd.DataFrame, y_cal: pd.Series):
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
        model,
        symbol: str,
        timeframe: str,
        meta: dict,
        feature_cols: list[str],
        model_dir: Path = Path("models"),
        calibrator=None,
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
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "n_features": len(feature_cols),
            "feature_names": feature_cols,
            "best_iteration": model.best_iteration,
            "has_calibrator": has_calibrator,
            **meta,
        }
        meta_path.write_text(json.dumps(full_meta, indent=2, default=str))

        log.info(f"LightGBM model saved: {model_path}")
        return model_path

    @staticmethod
    def load(symbol: str, timeframe: str, model_dir: Path = Path("models")):
        """Load a saved model. Returns lgb.Booster or None if not found."""
        import lightgbm as lgb

        symbol_key = symbol.replace("/", "_")
        model_path = model_dir / f"lgbm_{symbol_key}_{timeframe}.lgb"

        if not model_path.exists():
            log.warning(f"LightGBM model not found: {model_path}")
            return None

        return lgb.Booster(model_file=str(model_path))

    @staticmethod
    def load_calibrator(symbol: str, timeframe: str, model_dir: Path = Path("models")):
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
                x, y, kind="linear", bounds_error=False,
                fill_value=(y[0], y[-1]),
            )
        else:
            const_y = float(y[0])
            calibrator.f_ = lambda val, _c=const_y: np.full_like(
                np.asarray(val, dtype=float), _c, dtype=float,
            )
        return calibrator

    @staticmethod
    def load_meta(symbol: str, timeframe: str, model_dir: Path = Path("models")) -> dict | None:
        """Load model metadata. Returns dict or None if not found."""
        symbol_key = symbol.replace("/", "_")
        meta_path = model_dir / f"lgbm_{symbol_key}_{timeframe}_meta.json"

        if not meta_path.exists():
            return None

        return json.loads(meta_path.read_text())
