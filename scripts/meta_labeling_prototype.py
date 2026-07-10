"""Meta-labeling prototype — does conditioning the ML on a rule's triggers help?

The current bot trains LGBM on an UNCONDITIONAL target ("fwd 4-candle return >
0.6%" over *every* candle) and then bolts it onto a rule as a veto. That is a
population mismatch: the model learns from all candles but is only ever applied
at the rule's trigger candles.

Meta-labeling (Lopez de Prado) fixes the mismatch: a PRIMARY rule decides WHEN
to enter; a SECONDARY model, trained ONLY on the primary's trigger candles and
labeled by whether THAT trade won (triple-barrier), predicts P(this trade wins).

This prototype answers two questions on real data, time-honestly:

  1. THESIS — does a model trained on trigger rows (meta) beat the SAME model
     trained on all rows (unconditional), both evaluated on the held-out test
     triggers? (compare AUC on the identical test set)
  2. TRADEABLE — does filtering the rule's entries by the meta probability lift
     win-rate / net return per trade vs. taking every trigger?

Reuses: CombinedStrategy (rule -> filters + indicators), filter.vectorized_entry
(triggers), build_target_triple_barrier (labels), build_feature_matrix (tech10
features, external OFF to avoid the ext16 footgun), LGBMTrainer (train/calibrate).

Usage:
  .venv/bin/python scripts/meta_labeling_prototype.py \
      --entry "rsi_oversold:35 + trend_up:4" --symbol BTC/KRW --timeframe 1h
  .venv/bin/python scripts/meta_labeling_prototype.py --self-check

Writes results/meta_labeling/<symbol>_<tf>.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tradingbot.data.storage import load_candles
from tradingbot.ml.features import build_feature_matrix
from tradingbot.ml.targets import build_target_triple_barrier
from tradingbot.ml.trainer import LGBMTrainer
from tradingbot.strategy.combined import CombinedStrategy
from tradingbot.strategy.filters.registry import parse_filter_string

logging.getLogger().setLevel(logging.WARNING)

DATA_DIR = Path("data")
# 2 x (Upbit fee 0.05% + slippage 0.1%) — round-trip cost applied to the PnL proxy.
ROUND_TRIP_COST = 0.003
# Keep the top-q fraction by meta prob. Quantile (rank) cutoffs, not absolute probs:
# isotonic calibration squashes probs toward the base rate (~0.32), so an absolute
# 0.45+ grid keeps ~nothing — and the real consumer (lgbm_prob veto) top-slices by rank.
QUANTILES = [0.50, 0.33, 0.25, 0.15, 0.10]


def trigger_mask(df: pd.DataFrame, entry: str, symbol: str, tf: str) -> pd.Series:
    """Boolean mask of the primary rule's entry-trigger candles (AND of filters)."""
    entry_filters = parse_filter_string(entry, base_timeframe=tf)
    for f in entry_filters:
        if hasattr(f, "symbol"):
            f.symbol = symbol
        if hasattr(f, "timeframe"):
            f.timeframe = tf
    strat = CombinedStrategy(entry_filters=entry_filters, exit_filters=[])
    ind = strat.indicators(df.copy())

    mask = pd.Series(True, index=ind.index)
    used = 0
    for f in entry_filters:
        if f.role == "exit":
            continue
        if not f.supports_vectorized:
            print(f"  [warn] non-vectorized filter skipped: {f.name}")
            continue
        mask &= f.vectorized_entry(ind).fillna(False).astype(bool)
        used += 1
    if used == 0:
        raise SystemExit("no vectorizable entry filters in rule")
    return mask.fillna(False).astype(bool)


def _fit_val_split(
    idx: pd.DatetimeIndex, frac: float = 0.8
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    k = max(1, int(len(idx) * frac))
    return idx[:k], idx[k:]


def _train_calibrated(
    feat: pd.DataFrame, y_all: pd.Series, train_idx: pd.DatetimeIndex, cols: list[str]
) -> tuple[object, object]:
    """Train an LGBM booster on `train_idx` with a time-ordered early-stop/cal split."""
    fit_idx, val_idx = _fit_val_split(train_idx)
    has_val = len(val_idx) > 0
    trainer = LGBMTrainer()
    model = trainer.train(
        feat.loc[fit_idx, cols],
        y_all.loc[fit_idx],
        feat.loc[val_idx, cols] if has_val else None,
        y_all.loc[val_idx] if has_val else None,
        early_stopping_rounds=30,
    )
    # ponytail: val slice reused as calibration set (triggers are scarce); split a
    # third slice if calibration fidelity ever matters more than a prototype needs.
    calibrator = None
    if has_val:
        calibrator = trainer.calibrate(model, feat.loc[val_idx, cols], y_all.loc[val_idx])
    return model, calibrator


def _proba(model: object, calibrator: object, feat: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict(feat))  # type: ignore[attr-defined]
    return np.asarray(calibrator.predict(raw)) if calibrator is not None else raw


def run(
    df: pd.DataFrame,
    entry: str,
    symbol: str,
    tf: str,
    forward: int,
    atr_mult: float,
    train_frac: float,
) -> dict:
    triggers = trigger_mask(df, entry, symbol, tf)
    labels = build_target_triple_barrier(df, forward_candles=forward, atr_mult=atr_mult)
    feat_df, cols = build_feature_matrix(df.copy())  # tech10, external OFF

    valid = feat_df[cols].notna().all(axis=1) & labels.notna()
    trig_valid = valid & triggers
    trig_idx = df.index[trig_valid.to_numpy()]
    if len(trig_idx) < 120:
        raise SystemExit(
            f"only {len(trig_idx)} valid triggers — rule too sparse for a train/test split; "
            f"loosen --entry or use a longer history"
        )

    # --- Purged time-honest split (barrier window of last train row must end
    # before the first test trigger, else its label peeks into the test region) ---
    pos = pd.Series(np.arange(len(df)), index=df.index)
    split = int(len(trig_idx) * train_frac)
    test_idx = trig_idx[split:]
    first_test_pos = int(pos.loc[test_idx[0]])
    cutoff = first_test_pos - forward
    train_rows = valid & (pos < cutoff)

    meta_train_idx = df.index[(train_rows & triggers).to_numpy()]
    uncond_train_idx = df.index[train_rows.to_numpy()]

    meta_model, meta_cal = _train_calibrated(feat_df, labels, meta_train_idx, cols)
    uncond_model, _ = _train_calibrated(feat_df, labels, uncond_train_idx, cols)

    x_test = feat_df.loc[test_idx, cols]
    y_test = labels.loc[test_idx].astype(int)
    meta_p = _proba(meta_model, meta_cal, x_test)
    uncond_p = _proba(uncond_model, None, x_test)

    two_class = y_test.nunique() > 1
    meta_auc = float(roc_auc_score(y_test, meta_p)) if two_class else float("nan")
    uncond_auc = float(roc_auc_score(y_test, uncond_p)) if two_class else float("nan")

    # PnL proxy: realized fwd return over the horizon, net of round-trip cost.
    fwd = df["close"].pct_change(forward).shift(-forward)
    net = (fwd.loc[test_idx] - ROUND_TRIP_COST).to_numpy()
    base_rate = float(y_test.mean())

    order = np.argsort(-meta_p)  # test triggers ranked by meta prob, descending
    yt = y_test.to_numpy()
    sweep = []
    for q in QUANTILES:
        k = max(1, int(len(meta_p) * q))
        sel = order[:k]
        yk = yt[sel]
        rk = net[sel]
        sweep.append(
            {
                "top_q": q,
                "kept": k,
                "coverage": round(k / len(meta_p), 3),
                "prob_cutoff": round(float(meta_p[sel[-1]]), 4),
                "win_rate": round(float(yk.mean()), 4),
                "win_lift": round(float(yk.mean()) - base_rate, 4),
                "mean_net_ret": round(float(np.nanmean(rk)), 5),
                "total_net_ret": round(float(np.nansum(rk)), 4),
                "ret_per_risk": round(float(np.nanmean(rk) / (np.nanstd(rk) + 1e-9)), 3),
            }
        )

    take_all_total = round(float(np.nansum(net)), 4)
    take_all_mean = round(float(np.nanmean(net)), 5)

    # Verdict: meta helps if some top-q slice (>=30 trades) lifts BOTH win-rate
    # (>= +3pp) AND mean net return vs. take-all.
    helped = [
        s
        for s in sweep
        if s["kept"] >= 30 and s["win_lift"] >= 0.03 and s["mean_net_ret"] > take_all_mean
    ]
    verdict = (
        f"META HELPS @top_q={max(helped, key=lambda s: s['mean_net_ret'])['top_q']}"
        if helped
        else "NO LIFT — meta filter does not beat take-all on this rule/window"
    )
    thesis = (
        "meta > unconditional (conditioning helps)"
        if meta_auc > uncond_auc
        else "meta <= unconditional (no conditioning gain)"
    )

    return {
        "symbol": symbol,
        "timeframe": tf,
        "entry_rule": entry,
        "forward_candles": forward,
        "atr_mult": atr_mult,
        "n_triggers_total": len(trig_idx),
        "n_train_triggers": len(meta_train_idx),
        "n_uncond_train_rows": len(uncond_train_idx),
        "n_test_triggers": len(test_idx),
        "test_base_win_rate": round(base_rate, 4),
        "meta_test_auc": round(meta_auc, 4),
        "uncond_test_auc": round(uncond_auc, 4),
        "thesis_meta_vs_uncond": thesis,
        "take_all_mean_net_ret": take_all_mean,
        "take_all_total_net_ret": take_all_total,
        "threshold_sweep": sweep,
        "verdict": verdict,
    }


def _print_report(r: dict) -> None:
    print(f"\n=== Meta-labeling: {r['symbol']} {r['timeframe']} | rule: {r['entry_rule']} ===")
    print(
        f"triggers: {r['n_triggers_total']} total "
        f"(train {r['n_train_triggers']} / test {r['n_test_triggers']}) | "
        f"uncond train rows: {r['n_uncond_train_rows']} | "
        f"barrier fwd={r['forward_candles']} atr_mult={r['atr_mult']}"
    )
    print("\nTHESIS — same model, different training population, evaluated on test triggers:")
    print(f"  meta (trigger-trained)   test AUC = {r['meta_test_auc']}")
    print(f"  uncond (all-row-trained) test AUC = {r['uncond_test_auc']}")
    print(f"  -> {r['thesis_meta_vs_uncond']}")
    print(
        f"\nTRADEABLE — keep top-q rule entries by meta prob rank "
        f"(base win-rate {r['test_base_win_rate']}, "
        f"take-all mean/total net {r['take_all_mean_net_ret']}/{r['take_all_total_net_ret']}):"
    )
    hdr = f"  {'topQ':>5} {'kept':>5} {'cutoff':>7} {'win%':>6} {'lift':>6}"
    print(f"{hdr} {'meanNet':>8} {'totNet':>7} {'r/risk':>6}")
    for s in r["threshold_sweep"]:
        print(
            f"  {s['top_q']:>5.0%} {s['kept']:>5} {s['prob_cutoff']:>7.3f} "
            f"{s['win_rate']:>6.1%} {s['win_lift']:>+6.1%} {s['mean_net_ret']:>8.4f} "
            f"{s['total_net_ret']:>7.3f} {s['ret_per_risk']:>6.2f}"
        )
    print(f"\nVERDICT: {r['verdict']}")


def self_check() -> None:
    """Two invariants: (1) planted signal is learnable; (2) sub-min threshold == take-all."""
    rng = np.random.default_rng(0)
    n = 4000
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame(
        {"open": price, "high": price * 1.003, "low": price * 0.997, "close": price, "volume": 1.0},
        index=idx,
    )
    labels = build_target_triple_barrier(df, forward_candles=4, atr_mult=1.0)
    feat_df, cols = build_feature_matrix(df.copy())
    valid = feat_df[cols].notna().all(axis=1) & labels.notna()
    vidx = df.index[valid.to_numpy()]
    # Plant a perfectly-predictive feature = label.
    feat_df = feat_df.copy()
    feat_df.loc[vidx, "adx_14"] = labels.loc[vidx].to_numpy()
    k = int(len(vidx) * 0.7)
    model, cal = _train_calibrated(feat_df, labels, vidx[:k], cols)
    p = _proba(model, cal, feat_df.loc[vidx[k:], cols])
    auc = roc_auc_score(labels.loc[vidx[k:]].astype(int), p)
    assert auc > 0.9, f"planted-signal AUC too low: {auc:.3f}"

    # Threshold below any prob keeps everything and reproduces the base rate.
    yt = labels.loc[vidx[k:]].astype(int).to_numpy()
    keep = p >= -1.0
    assert keep.all() and abs(yt[keep].mean() - yt.mean()) < 1e-12
    print(f"self-check OK (planted AUC={auc:.3f}, take-all invariant holds)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", default="trend_up:4")  # canonical wide-net primary (fires often)
    ap.add_argument("--symbol", default="BTC/KRW")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--forward", type=int, default=4)
    ap.add_argument("--atr-mult", type=float, default=1.0)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        self_check()
        return

    df = load_candles(args.symbol, args.timeframe, DATA_DIR)
    r = run(
        df, args.entry, args.symbol, args.timeframe, args.forward, args.atr_mult, args.train_frac
    )
    _print_report(r)

    out = Path("results/meta_labeling") / f"{args.symbol.replace('/', '_')}_{args.timeframe}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
