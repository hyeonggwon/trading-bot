"""Diagnostic: ETH/KRW 4h calibrator output distribution on holdout window.

Loads the saved model + calibrator, builds the same feature matrix as
LGBMStrategy, runs inference on each holdout candle, and reports the
distribution of calibrated probabilities. Goal: confirm whether thresholds
0.25-0.45 are reachable, or if the model is producing a near-constant output.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tradingbot.data.external_fetcher import (
    align_external_to,
    load_external_components,
    resolve_external_data_dir,
)
from tradingbot.data.storage import load_candles
from tradingbot.ml.features import build_feature_matrix
from tradingbot.ml.trainer import LGBMTrainer

ROOT = Path("/Users/matthew/develop/Projects/trading-bot")
SYMBOL = "ETH/KRW"
TIMEFRAME = "4h"
MODEL_DIR = ROOT / "models"


def main() -> None:
    meta = LGBMTrainer.load_meta(SYMBOL, TIMEFRAME, MODEL_DIR)
    model = LGBMTrainer.load(SYMBOL, TIMEFRAME, MODEL_DIR)
    calibrator = LGBMTrainer.load_calibrator(SYMBOL, TIMEFRAME, MODEL_DIR)
    feature_cols = meta["feature_names"]
    holdout_start = pd.Timestamp(meta["holdout_start"])
    holdout_end = pd.Timestamp(meta["holdout_end"])

    df = load_candles(SYMBOL, TIMEFRAME, ROOT / "data")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    include_extra = bool(meta.get("include_extra", False))
    ext_dir = resolve_external_data_dir(None)
    external_df = None
    if ext_dir is not None:
        components = load_external_components(ext_dir)
        external_df = align_external_to(df, components)
        print(f"External data dir: {ext_dir}")
    feat_df, _ = build_feature_matrix(df, external_df=external_df, include_extra=include_extra)

    holdout = feat_df.loc[holdout_start:holdout_end]
    print(f"Holdout window: {holdout_start} → {holdout_end} ({len(holdout)} rows)")

    missing = [c for c in feature_cols if c not in holdout.columns]
    if missing:
        print(f"Missing feature columns: {missing}")

    available = [c for c in feature_cols if c in holdout.columns]
    X = holdout[available].dropna()
    print(f"Rows after dropna on {len(available)} features: {len(X)}")

    raw_probs = model.predict(X)
    cal_probs = (
        np.array(calibrator.transform(raw_probs)) if calibrator is not None else raw_probs
    )

    def stats(arr, label):
        if len(arr) == 0:
            print(f"{label}: empty")
            return
        print(
            f"{label}: n={len(arr)} min={arr.min():.4f} p10={np.quantile(arr,0.1):.4f} "
            f"p50={np.quantile(arr,0.5):.4f} mean={arr.mean():.4f} "
            f"p90={np.quantile(arr,0.9):.4f} max={arr.max():.4f} std={arr.std():.4f}"
        )

    stats(raw_probs, "Raw probs    ")
    stats(cal_probs, "Calibrated   ")

    print()
    print("Calibrated exceedance:")
    for t in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        frac = (cal_probs > t).mean()
        n = (cal_probs > t).sum()
        print(f"  > {t:.2f}  →  {frac*100:6.2f}%  ({n} candles)")

    print()
    print(f"Calibrator type: {type(calibrator).__name__}")
    cal_path = MODEL_DIR / f"lgbm_{SYMBOL.replace('/', '_')}_{TIMEFRAME}_cal.json"
    if cal_path.exists():
        cal_json = json.loads(cal_path.read_text())
        if "X" in cal_json and "y" in cal_json:
            cx = np.array(cal_json["X"])
            cy = np.array(cal_json["y"])
            print(
                f"Calibrator fit data: n={len(cx)} "
                f"X range [{cx.min():.4f}, {cx.max():.4f}], "
                f"y mean (base rate) {cy.mean():.4f}"
            )


if __name__ == "__main__":
    main()
