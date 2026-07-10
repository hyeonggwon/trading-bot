"""Action 1 — 비대칭 배리어 × 호라이즌 스윕 (심의 합의 1위 액션).

가설(architect/critic 합의): 봇의 병목은 "약한 신호"가 아니라 **거래 기하**다.
대칭 배리어(TP=SL)에서 손익분기 승률 p* = (SL+c)/(TP+SL) = 0.5 + c/(2·barrier)
이라 50% 위에 못박히고, AUC 0.5~0.55짜리 신호로는 영영 못 넘는다. reward:risk를
3:1로 벌리면 p* ≈ 0.30으로 내려가 **이미 확보된 상위-decile 승률(≈45%)이 통과**할 수
있다. 즉 신호를 고정한 채 라벨/청산 기하만 바꿔 손익 부등식을 바꾼다.

이 스크립트는 신호(무조건부 tech10 LGBM)를 고정하고 (TP:SL, horizon) 격자만 흔들어,
각 셀에서 상위-decile 진입의 **비용 차감 후 per-trade 실현 EV**가 양수로 가는지 5심볼
실측한다. 손익과 라벨은 동일한 bracket 시뮬레이션 한 곳에서 나온다(단일 진실원).

정직성 장치:
- bracket 결과는 오프라인 라벨/평가 전용(피처로 절대 안 들어감) — anti-lookahead 유지.
- 같은 바에서 TP·SL 동시 터치 = **SL 우선(손실)** 보수 회계. 실현 EV의 하한.
- train/test purged split, **embargo = horizon** (지평에 비례 — 고정 150이 아님).
- 비용 = 왕복 0.3%(수수료 0.1% + 슬리피지 0.2%), prototype과 동일.
- 진입가 = close[i] 프록시(실엔진은 다음 봉 open+슬리피지; 비용의 슬리피지분이 그 갭을 흡수).

Usage:
  .venv/bin/python scripts/barrier_geometry_sweep.py            # 5심볼 1h 전체
  .venv/bin/python scripts/barrier_geometry_sweep.py --symbol XRP/KRW
  .venv/bin/python scripts/barrier_geometry_sweep.py --self-check
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tradingbot.data.storage import load_candles
from tradingbot.ml.features import build_feature_matrix
from tradingbot.ml.targets import _atr_pct

# Reuse the prototype's trainer/predict/cost so signal + methodology match exactly.
# Import the sibling by dir (works under `python scripts/x.py` and repo-root cwd alike).
sys.path.insert(0, os.path.dirname(__file__))
from meta_labeling_prototype import ROUND_TRIP_COST, _proba, _train_calibrated  # noqa: E402

logging.getLogger().setLevel(logging.WARNING)

DATA_DIR = Path("data")
SYMBOLS = ["BTC/KRW", "ETH/KRW", "XRP/KRW", "SOL/KRW", "DOGE/KRW"]
HORIZONS = [4, 12, 24]
TP_MULTS = [1.0, 2.0, 3.0]  # SL_mult fixed at 1.0 → reward:risk = 1:1, 2:1, 3:1
SL_MULT = 1.0
ATR_PERIOD = 14
TRAIN_FRAC = 0.7
TOP_QS = [0.10, 0.05]


def simulate_bracket(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_pct: np.ndarray,
    tp_mult: float,
    sl_mult: float,
    horizon: int,
) -> np.ndarray:
    """Realized fractional return of a TP/SL bracket entered at close[i].

    Scans bars i+1..i+horizon. SL checked before TP within a bar → same-bar
    double-touch counts as SL (conservative). Neither touched → vertical
    (timeout) return close[i+horizon]/close[i]-1. Tail without a full window,
    and warmup rows with non-finite ATR, stay NaN.
    """
    n = len(close)
    ret = np.full(n, np.nan)
    valid = np.isfinite(atr_pct) & (atr_pct > 0)
    tp_level = close * (1.0 + tp_mult * atr_pct)
    sl_level = close * (1.0 - sl_mult * atr_pct)
    done = np.zeros(n, dtype=bool)

    for k in range(1, horizon + 1):
        fh = np.full(n, np.nan)
        fl = np.full(n, np.nan)
        fh[: n - k] = high[k:]
        fl[: n - k] = low[k:]
        newly_sl = valid & ~done & (fl <= sl_level)  # SL first (conservative)
        ret[newly_sl] = -sl_mult * atr_pct[newly_sl]
        done |= newly_sl
        newly_tp = valid & ~done & (fh >= tp_level)
        ret[newly_tp] = tp_mult * atr_pct[newly_tp]
        done |= newly_tp

    fc = np.full(n, np.nan)
    fc[: n - horizon] = close[horizon:]
    timeout = valid & ~done & np.isfinite(fc)
    ret[timeout] = fc[timeout] / close[timeout] - 1.0
    ret[n - horizon :] = np.nan  # incomplete forward window
    ret[~valid] = np.nan
    return ret


def _breakeven_winrate(
    atr_pct_test: np.ndarray, tp_mult: float, sl_mult: float, cost: float
) -> float:
    """Approx p* = (SL+c)/(TP+SL) using median barrier width on the test slice."""
    b = float(np.nanmedian(atr_pct_test))
    tp, sl = tp_mult * b, sl_mult * b
    return (sl + cost) / (tp + sl)


def _eval_cell(
    feat_df: pd.DataFrame,
    cols: list[str],
    df: pd.DataFrame,
    atr_pct: np.ndarray,
    tp_mult: float,
    horizon: int,
) -> dict | None:
    bracket = simulate_bracket(
        df["high"].to_numpy(float),
        df["low"].to_numpy(float),
        df["close"].to_numpy(float),
        atr_pct,
        tp_mult,
        SL_MULT,
        horizon,
    )
    bracket_s = pd.Series(bracket, index=df.index)
    label = (bracket_s > 0).astype(float)
    label[bracket_s.isna()] = np.nan

    valid = feat_df[cols].notna().all(axis=1) & bracket_s.notna()
    vidx = df.index[valid.to_numpy()]
    if len(vidx) < 400:
        return None

    pos = pd.Series(np.arange(len(df)), index=df.index)
    split = int(len(vidx) * TRAIN_FRAC)
    test_idx = vidx[split:]
    first_test_pos = int(pos.loc[test_idx[0]])
    cutoff = first_test_pos - horizon  # embargo scaled to the label horizon
    train_idx = df.index[(valid & (pos < cutoff)).to_numpy()]
    if len(train_idx) < 300 or len(test_idx) < 100:
        return None

    model, cal = _train_calibrated(feat_df, label, train_idx, cols)
    p = _proba(model, cal, feat_df.loc[test_idx, cols])

    bt = bracket_s.loc[test_idx].to_numpy()
    yt = label.loc[test_idx].to_numpy()
    atr_test = atr_pct[pos.loc[test_idx].to_numpy()]
    cost = ROUND_TRIP_COST
    order = np.argsort(-p)

    slices = {}
    for q in TOP_QS:
        k = max(1, int(len(p) * q))
        sel = order[:k]
        slices[f"top{int(q * 100)}"] = {
            "kept": k,
            "win_rate": round(float(yt[sel].mean()), 4),
            "gross_ev": round(float(np.nanmean(bt[sel])), 5),
            "net_ev": round(float(np.nanmean(bt[sel]) - cost), 5),
        }
    return {
        "tp_mult": tp_mult,
        "sl_mult": SL_MULT,
        "reward_risk": f"{tp_mult:g}:{SL_MULT:g}",
        "horizon": horizon,
        "test_n": len(test_idx),
        "breakeven_winrate": round(_breakeven_winrate(atr_test, tp_mult, SL_MULT, cost), 4),
        "base_win_rate": round(float(yt.mean()), 4),
        "base_net_ev": round(float(np.nanmean(bt) - cost), 5),
        **slices,
    }


def run_symbol(symbol: str, tf: str) -> dict:
    df = load_candles(symbol, tf, DATA_DIR)
    atr_pct = _atr_pct(df, period=ATR_PERIOD).to_numpy()
    feat_df, cols = build_feature_matrix(df.copy())  # tech10, external OFF
    cells = []
    for h in HORIZONS:
        for tp in TP_MULTS:
            cell = _eval_cell(feat_df, cols, df, atr_pct, tp, h)
            if cell is not None:
                cells.append(cell)
    return {"symbol": symbol, "timeframe": tf, "rows": len(df), "cells": cells}


def _print_symbol(r: dict) -> None:
    print(f"\n=== {r['symbol']} {r['timeframe']} ({r['rows']} candles) ===")
    print(
        f"  {'R:R':>5} {'horiz':>5} {'testN':>6} {'p*':>6} {'baseWin':>7} | "
        f"{'top10Win':>8} {'top10Net':>9} {'top5Net':>9}"
    )
    for c in r["cells"]:
        flag = "  <== NET+" if c["top10"]["net_ev"] > 0 else ""
        print(
            f"  {c['reward_risk']:>5} {c['horizon']:>5} {c['test_n']:>6} "
            f"{c['breakeven_winrate']:>6.1%} {c['base_win_rate']:>7.1%} | "
            f"{c['top10']['win_rate']:>8.1%} {c['top10']['net_ev']:>9.5f} "
            f"{c['top5']['net_ev']:>9.5f}{flag}"
        )


def self_check() -> None:
    # 1) bracket sanity: monotonic rise -> all TP; monotonic fall -> all SL.
    #    Ranges are asymmetric so the intended barrier is reachable while the
    #    opposite one stays untouched (else conservative SL-first tie would fire).
    n = 200
    atr = np.full(n, 0.01)  # TP@2xATR = +2%, SL@1xATR = -1%
    rise = np.linspace(100, 130, n)
    rb = simulate_bracket(rise * 1.03, rise * 0.999, rise, atr, 2.0, 1.0, 5)
    got = rb[~np.isnan(rb)]
    assert np.all(got > 0), "rising series should hit TP (positive)"
    top = float(np.nanmax(rb))
    assert abs(top - 0.02) < 1e-9, f"TP return should be tp_mult*atr=0.02, got {top}"
    fall = np.linspace(130, 100, n)
    rb2 = simulate_bracket(fall * 1.001, fall * 0.97, fall, atr, 2.0, 1.0, 5)
    got2 = rb2[~np.isnan(rb2)]
    assert np.all(got2 < 0), "falling series should hit SL (negative)"
    assert abs(float(np.nanmin(rb2)) + 0.01) < 1e-9, "SL return should be -sl_mult*atr=-0.01"

    # 2) asymmetric breakeven drops below 0.5 as reward:risk widens.
    p_sym = _breakeven_winrate(atr, 1.0, 1.0, ROUND_TRIP_COST)
    p_3to1 = _breakeven_winrate(atr, 3.0, 1.0, ROUND_TRIP_COST)
    assert p_sym > 0.5 and p_3to1 < 0.4, f"breakeven: sym={p_sym:.3f} 3:1={p_3to1:.3f}"

    # 3) planted signal: a feature equals the bracket win -> top decile net EV > 0 at 3:1.
    rng = np.random.default_rng(0)
    m = 5000
    idx = pd.date_range("2022-01-01", periods=m, freq="h", tz="UTC")
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, m)))
    df = pd.DataFrame(
        {"open": price, "high": price * 1.006, "low": price * 0.994, "close": price, "volume": 1.0},
        index=idx,
    )
    atr_pct = _atr_pct(df, ATR_PERIOD).to_numpy()
    bracket = simulate_bracket(
        df["high"].to_numpy(float),
        df["low"].to_numpy(float),
        df["close"].to_numpy(float),
        atr_pct,
        3.0,
        1.0,
        12,
    )
    bs = pd.Series(bracket, index=idx)
    label = (bs > 0).astype(float)
    label[bs.isna()] = np.nan
    feat_df, cols = build_feature_matrix(df.copy())
    valid = feat_df[cols].notna().all(axis=1) & bs.notna()
    vidx = idx[valid.to_numpy()]
    feat_df = feat_df.copy()
    feat_df.loc[vidx, "adx_14"] = label.loc[vidx].to_numpy()  # plant
    k = int(len(vidx) * 0.7)
    model, cal = _train_calibrated(feat_df, label, vidx[:k], cols)
    test = vidx[k:]
    p = _proba(model, cal, feat_df.loc[test, cols])
    order = np.argsort(-p)
    sel = order[: int(len(p) * 0.1)]
    net = float(np.nanmean(bs.loc[test].to_numpy()[sel]) - ROUND_TRIP_COST)
    assert net > 0, f"planted signal @3:1 should give positive top-decile net EV, got {net}"
    print("self-check OK (bracket sanity, asym breakeven, planted-signal net EV+)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None, help="single symbol; default = all 5")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        self_check()
        return

    symbols = [args.symbol] if args.symbol else SYMBOLS
    results = []
    wins = []
    for sym in symbols:
        r = run_symbol(sym, args.timeframe)
        _print_symbol(r)
        results.append(r)
        for c in r["cells"]:
            if c["top10"]["net_ev"] > 0:
                wins.append((sym, c["reward_risk"], c["horizon"], c["top10"]["net_ev"]))

    print("\n===== NET-POSITIVE top-decile cells (success = TP:SL>=2 on >=2 symbols) =====")
    if wins:
        for w in sorted(wins, key=lambda x: -x[3]):
            print(f"  {w[0]:>9}  R:R={w[1]}  horizon={w[2]}  top10 net EV/trade={w[3]:+.5f}")
    else:
        print("  (none — direction prediction stays below cost across all geometries)")

    out = Path("results/next_steps/barrier_geometry_sweep.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
