"""Feature ablation harness — holdout AUC with bootstrap CI across configs.

For each (symbol, timeframe, config) it runs the SAME walk-forward trainer the
CLI uses (MLWalkForwardTrainer, Path B single-split holdout), to a temp model
dir so models/ is never touched, and captures:
  - holdout_auc          : out-of-sample AUC on the holdout eval half
  - holdout_auc_ci       : 95% bootstrap CI on that AUC (judges single-window noise)
  - avg_auc              : inner-val AUC (early-stopping set)
  - best_iteration       : trees kept (shallow => weak signal)
  - holdout_pos_rate     : positive rate on the eval half
  - n_eval               : eval-half sample count

Configs:
  tech10 : technical only            (external_df=None,   include_extra=False)
  ext16  : + 6 external features      (external_df=loaded, include_extra=False)
  full28 : + 6 external + 12 extra    (external_df=loaded, include_extra=True)

Usage: .venv/bin/python scripts/feature_ablation.py
Writes results/feature_expansion/ablation.json
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from tradingbot.data.external_fetcher import build_external_df
from tradingbot.data.storage import EXTERNAL_SUBDIR, load_candles
from tradingbot.ml.walk_forward import MLWalkForwardTrainer

logging.getLogger().setLevel(logging.WARNING)

SYMBOLS = ["BTC/KRW", "ETH/KRW", "XRP/KRW", "SOL/KRW", "DOGE/KRW"]
TIMEFRAMES = ["1h", "4h"]
CONFIGS = ["tech10", "ext16", "full28"]
DATA_DIR = Path("data")
OUT = Path("results/feature_expansion/ablation.json")
B = 2000  # bootstrap resamples
RNG = np.random.default_rng(42)


def bootstrap_auc_ci(
    y_true: np.ndarray, proba: np.ndarray, b: int = B
) -> tuple[float, float] | None:
    """95% percentile bootstrap CI for AUC. None if a class is absent."""
    if y_true is None or proba is None or len(np.unique(y_true)) < 2:
        return None
    n = len(y_true)
    aucs = []
    for _ in range(b):
        idx = RNG.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, proba[idx]))
    if not aucs:
        return None
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def run_one(symbol: str, timeframe: str, config: str, ext_dir: Path) -> dict:
    df = load_candles(symbol, timeframe, DATA_DIR)
    include_extra = config == "full28"
    external_df = None if config == "tech10" else build_external_df(df, ext_dir)

    with tempfile.TemporaryDirectory() as tmp:
        trainer = MLWalkForwardTrainer(
            symbol=symbol,
            timeframe=timeframe,
            include_extra=include_extra,
            model_dir=Path(tmp),
        )
        report = trainer.run(df, external_df=external_df)

    yt = report.holdout_y_true
    raw = report.holdout_raw_proba
    ci = bootstrap_auc_ci(yt, raw) if yt is not None and raw is not None else None
    best_iter = report.windows[0].get("best_iteration") if report.windows else None
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "config": config,
        "holdout_auc": round(float(report.holdout_auc), 4),
        "holdout_auc_ci": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
        "avg_auc": round(float(report.avg_auc), 4),
        "best_iteration": best_iter,
        "n_eval": int(len(yt)) if yt is not None else 0,
        "holdout_pos_rate": round(float(yt.mean()), 4) if yt is not None and len(yt) else None,
    }


def main() -> None:
    ext_dir = DATA_DIR / EXTERNAL_SUBDIR
    results = []
    for tf in TIMEFRAMES:
        for symbol in SYMBOLS:
            for config in CONFIGS:
                r = run_one(symbol, timeframe=tf, config=config, ext_dir=ext_dir)
                ci = r["holdout_auc_ci"]
                ci_s = f"[{ci[0]:.3f},{ci[1]:.3f}]" if ci else "n/a"
                print(
                    f"{symbol:9s} {tf:3s} {config:7s} "
                    f"holdout_auc={r['holdout_auc']:.4f} ci={ci_s} "
                    f"best_iter={r['best_iteration']} n_eval={r['n_eval']}",
                    flush=True,
                )
                results.append(r)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT} ({len(results)} rows)")


if __name__ == "__main__":
    main()
