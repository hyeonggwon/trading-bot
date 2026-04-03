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
    "metric": ["binary_logloss", "auc"],
    "verbose": -1,
    # Tree structure — shallow to prevent overfitting
    "num_leaves": 31,
    "max_depth": 6,
    "min_data_in_leaf": 200,
    "min_sum_hessian_in_leaf": 10.0,
    # Regularization
    "reg_alpha": 0.1,
    "reg_lambda": 0.5,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    # Class imbalance
    "is_unbalance": True,
    # Learning
    "learning_rate": 0.02,
    "n_estimators": 2000,
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
    ):
        """Train a LightGBM model.

        Returns:
            lgb.Booster
        """
        import lightgbm as lgb

        # Extract n_estimators from params (used as num_boost_round)
        params = {k: v for k, v in self.params.items() if k != "n_estimators"}
        num_boost_round = self.params.get("n_estimators", 2000)

        train_set = lgb.Dataset(X_train, label=y_train)

        callbacks = [lgb.log_evaluation(period=0)]  # suppress per-iteration logs
        valid_sets = [train_set]
        valid_names = ["train"]

        if X_val is not None and y_val is not None and len(X_val) > 0:
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

        Returns dict with auc, precision, recall, f1, n_test, positive_rate.
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

        return {
            "auc": round(auc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "n_test": len(y_test),
            "positive_rate": round(float(y_test.mean()), 4),
        }

    def save(
        self,
        model,
        symbol: str,
        timeframe: str,
        meta: dict,
        feature_cols: list[str],
        model_dir: Path = Path("models"),
    ) -> Path:
        """Save model (.lgb) and metadata (_meta.json).

        Returns path to saved model file.
        """
        model_dir.mkdir(parents=True, exist_ok=True)
        symbol_key = symbol.replace("/", "_")
        model_path = model_dir / f"lgbm_{symbol_key}_{timeframe}.lgb"
        meta_path = model_dir / f"lgbm_{symbol_key}_{timeframe}_meta.json"

        model.save_model(str(model_path))

        full_meta = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "n_features": len(feature_cols),
            "feature_names": feature_cols,
            "best_iteration": model.best_iteration,
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
    def load_meta(symbol: str, timeframe: str, model_dir: Path = Path("models")) -> dict | None:
        """Load model metadata. Returns dict or None if not found."""
        symbol_key = symbol.replace("/", "_")
        meta_path = model_dir / f"lgbm_{symbol_key}_{timeframe}_meta.json"

        if not meta_path.exists():
            return None

        return json.loads(meta_path.read_text())
