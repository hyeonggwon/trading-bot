"""Item 2 (백테스트 전용) — 배포 후보 KC+Trend+Vol 의 nested-holdout 재검증.

Action 2 는 배포 후보(XRP KC+Trend+Vol)가 M=825 스캔에서 선택됐지만 (a) 스캔 Sharpe
선택은 잡음 지배, (b) 11창 OOS alpha t=1.62(유의성 미달)·R²=0.83(레짐 동조)임을 보였다.
남은 구멍: **선택 절차가 접촉하지 않은 forward 슬라이스**에서 이 후보가 살아남는가?

garden-of-forking-paths 의 핵심은 *선택*이 과최적된다는 것 — 그래서 재검증은 선택을
reserve 를 가린 채 다시 돌려야 한다. 단일 컷오프 nested holdout:

  전체 = [start .. data_end]
  IN-SAMPLE = [start .. T]     (T = CUTOFF)         ← 선택은 여기서만
  RESERVE   = (T .. data_end]                        ← 선택이 못 본 forward 슬라이스

- **Stage A (reserve 가린 재선택)**: IN-SAMPLE 의 마지막 20% 홀드아웃 Sharpe 로 48 combine
  템플릿 × (심볼,tf) 그리드를 파이프라인 stage-1 과 동일 기준(min_trades≥10, sort=sharpe)
  으로 랭크. → KC+Trend+Vol 이 여전히 상위로 뽑히나? (선택 안정성)
- **Stage B (untouched OOS)**: Stage A 상위 K + KC+Trend+Vol 를 RESERVE 에서 평가.
  선택이 못 본 구간에서 상위 후보가 돈을 버나? KC 는?
- **Stage C (유의성)**: KC+Trend+Vol 를 RESERVE 에서 풀엔진 실행 → 트레이드별 순수익률
  (`pnl_pct`, 수수료 포함) 의 평균·t·이동블록 부트스트랩 95% CI + 같은 구간 buy&hold 대비.

정직성 장치:
- 지표는 인과적(과거만) → `_run_batch` 이 full df 로 지표를 계산해도 [C,T] 트레이드는
  T 이후 데이터의 영향을 받지 않는다(구조적 anti-lookahead). 그래도 선택 평가창은 ≤T.
- RESERVE 는 T 이후만 — Stage A 선택 기준에 절대 안 들어감.
- KC+Trend+Vol 는 원래 full-history 파이프라인이 뽑았으므로 이미 reserve 를 봤다. Stage A
  가 reserve 를 가리고도 같은 후보를 뽑으면, 그 선택은 reserve 에 의존하지 않았다는 뜻
  → RESERVE 성과가 깨끗한 OOS 다.

Usage:
  .venv/bin/python scripts/nested_holdout_revalidation.py
  .venv/bin/python scripts/nested_holdout_revalidation.py --cutoff 2025-09-30 --top 10
  .venv/bin/python scripts/nested_holdout_revalidation.py --self-check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from tradingbot.backtest.engine import BacktestEngine
from tradingbot.backtest.parallel import ScanResult, _run_batch
from tradingbot.cli.combine import COMBINE_TEMPLATES
from tradingbot.config import load_config
from tradingbot.data.storage import list_available_data, load_candles

sys.path.insert(0, os.path.dirname(__file__))
from cost_sensitivity_sweep import _build_strategy  # noqa: E402  (sibling script reuse)

DATA_DIR = Path("data")
BALANCE = 1_000_000
HOLDOUT_PCT = 0.2
MIN_TRADES = 10  # pipeline stage-1 gate
CANDIDATE = {  # the deploy candidate under scrutiny
    "name": "KC+Trend+Vol",
    "symbol": "XRP/KRW",
    "timeframe": "1h",
    "entry": "keltner_break + trend_up:4 + volume_spike:2.0",
    "exit": "keltner_break",
}
BLOCK = 3  # moving-block bootstrap block length (trade returns are regime-clustered)
N_BOOT = 5000


def _insample_holdout_start(df_le_t) -> str | None:
    """Mirror resolve_holdout_window's single-df split on the ≤T frame."""
    if len(df_le_t) < 50:
        return None
    return str(df_le_t.index[int(len(df_le_t) * (1 - HOLDOUT_PCT))])


def _grid() -> list[tuple[str, str]]:
    av = list_available_data(DATA_DIR)
    # combined candidate peers live on the same timeframes it was scanned on.
    return sorted({(x["symbol"], x["timeframe"]) for x in av})


def _template_jobs() -> list[tuple[str, str, str]]:
    return [(t["label"], t["entry"], t["exit"]) for t in COMBINE_TEMPLATES]


def stage_a_reselect(cutoff: str) -> list[dict]:
    """Rank the combine-template universe on IN-SAMPLE holdout, reserve hidden."""
    jobs = _template_jobs()
    rows: list[dict] = []
    for sym, tf in _grid():
        df = load_candles(sym, tf, DATA_DIR)
        df_le = df[df.index <= cutoff]
        c = _insample_holdout_start(df_le)
        if c is None:
            continue
        batch: list[ScanResult] = _run_batch(
            sym, tf, jobs, str(DATA_DIR), BALANCE, "config", False, c, cutoff, False
        )
        for r in batch:
            if r.error or r.total_trades < MIN_TRADES:
                continue
            rows.append(
                {
                    "name": r.strategy,
                    "symbol": r.symbol,
                    "timeframe": r.timeframe,
                    "entry": r.entry,
                    "exit": r.exit,
                    "insample_sharpe": round(r.sharpe_ratio, 4),
                    "insample_return": round(r.total_return, 5),
                    "insample_trades": r.total_trades,
                }
            )
    rows.sort(key=lambda x: x["insample_sharpe"], reverse=True)
    for i, x in enumerate(rows, 1):
        x["insample_rank"] = i
    return rows


def _reserve_metrics(name, sym, tf, entry, exit_, cutoff) -> dict:
    """Evaluate one candidate on RESERVE = (cutoff, data_end], untouched by selection."""
    job = (name, entry, exit_)
    res = _run_batch(sym, tf, [job], str(DATA_DIR), BALANCE, "config", False, cutoff, None, False)[
        0
    ]
    n = res.total_trades
    return {
        "name": name,
        "symbol": sym,
        "timeframe": tf,
        "reserve_sharpe": round(res.sharpe_ratio, 4),
        "reserve_return": round(res.total_return, 5),
        "reserve_trades": n,
        "reserve_ret_per_trade": round(res.total_return / n, 5) if n else None,
        "error": res.error,
    }


def _block_bootstrap_mean(x: np.ndarray, block: int, n_boot: int, seed: int = 0) -> tuple:
    """Moving-block bootstrap CI for the mean of a (autocorrelated) series."""
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < block + 1:
        return float(x.mean()), float("nan"), float("nan")
    n_blocks = int(np.ceil(n / block))
    starts_pool = np.arange(0, n - block + 1)
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.choice(starts_pool, size=n_blocks)
        sample = np.concatenate([x[s : s + block] for s in starts])[:n]
        means[b] = sample.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def stage_c_significance(cutoff: str) -> dict:
    """Per-trade significance of the candidate on the untouched RESERVE slice."""
    sym, tf = CANDIDATE["symbol"], CANDIDATE["timeframe"]
    strat = _build_strategy(CANDIDATE["entry"], CANDIDATE["exit"], sym, tf)
    config = load_config(
        Path("config"),
        overrides={
            "trading": {"symbols": [sym], "timeframe": tf, "initial_balance": BALANCE},
            "backtest": {"start_date": cutoff, "end_date": None},
        },
    )
    report = BacktestEngine(strategy=strat, config=config).run(
        {sym: load_candles(sym, tf, DATA_DIR)}
    )
    rets = np.array([t.pnl_pct for t in report.trades], dtype=float)
    n = len(rets)
    df_res = load_candles(sym, tf, DATA_DIR)
    df_res = df_res[df_res.index > cutoff]
    bh = (
        float(df_res["close"].iloc[-1] / df_res["close"].iloc[0] - 1.0) if len(df_res) > 1 else None
    )
    out: dict = {
        "symbol": sym,
        "timeframe": tf,
        "reserve_trades": n,
        "reserve_win_rate": round(report.win_rate, 4),
        "reserve_total_return": round(report.total_return, 5),
        "buy_hold_reserve_return": round(bh, 5) if bh is not None else None,
    }
    if n >= 2:
        mean = float(rets.mean())
        se = float(rets.std(ddof=1) / np.sqrt(n))
        t = mean / se if se else float("nan")
        m, lo, hi = _block_bootstrap_mean(rets, BLOCK, N_BOOT)
        out.update(
            {
                "mean_ret_per_trade": round(mean, 5),
                "t_stat_per_trade": round(t, 3),
                "boot_ci95_low": round(lo, 5),
                "boot_ci95_high": round(hi, 5),
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
                "verdict": (
                    "OOS edge distinguishable from zero (CI excludes 0)"
                    if (lo > 0)
                    else "OOS edge NOT distinguishable from zero (CI spans 0)"
                ),
            }
        )
    else:
        out["verdict"] = "too few reserve trades for significance"
    return out


def run(cutoff: str, top: int) -> dict:
    ranked = stage_a_reselect(cutoff)
    kc = next(
        (
            r
            for r in ranked
            if r["symbol"] == CANDIDATE["symbol"]
            and r["timeframe"] == CANDIDATE["timeframe"]
            and "keltner_break" in r["entry"]
            and "trend_up" in r["entry"]
            and "volume_spike" in r["entry"]
        ),
        None,
    )
    winners = ranked[:top]
    # Reserve-evaluate the top-K winners plus the candidate (dedup on identity).
    eval_set = {(w["name"], w["symbol"], w["timeframe"], w["entry"], w["exit"]) for w in winners}
    eval_set.add(
        (
            CANDIDATE["name"],
            CANDIDATE["symbol"],
            CANDIDATE["timeframe"],
            CANDIDATE["entry"],
            CANDIDATE["exit"],
        )
    )
    reserve = [_reserve_metrics(*e, cutoff) for e in sorted(eval_set)]
    return {
        "cutoff": cutoff,
        "n_ranked_insample": len(ranked),
        "candidate_insample": kc,
        "top_insample": winners,
        "reserve_eval": sorted(reserve, key=lambda x: x["reserve_sharpe"], reverse=True),
        "candidate_significance": stage_c_significance(cutoff),
    }


def _print(res: dict) -> None:
    print(f"\n=== nested-holdout 재검증 (cutoff {res['cutoff']}) ===")
    kc = res["candidate_insample"]
    n = res["n_ranked_insample"]
    print(f"\n[Stage A] reserve 가린 재선택 — {n} combos 통과(min_trades≥{MIN_TRADES})")
    if kc:
        print(
            f"  KC+Trend+Vol XRP 1h: in-sample rank {kc['insample_rank']}/{n}"
            f"  (sharpe {kc['insample_sharpe']}, ret {kc['insample_return']:.2%}, "
            f"{kc['insample_trades']} trades)"
        )
    else:
        print("  KC+Trend+Vol: min_trades 게이트 미통과 또는 재선택 그리드에서 탈락")
    print("  in-sample 상위 5:")
    for r in res["top_insample"][:5]:
        print(
            f"    {r['insample_rank']:>2}. {r['name']:<18} {r['symbol']:<8} {r['timeframe']:<3} "
            f"sharpe {r['insample_sharpe']:>6.2f}  {r['insample_trades']:>3}tr"
        )
    print("\n[Stage B] RESERVE (untouched forward) 평가:")
    h = ("candidate", "sym", "tf", "sharpe", "return", "trades", "ret/tr")
    print(f"    {h[0]:<18}{h[1]:<8}{h[2]:<4}{h[3]:>7}{h[4]:>9}{h[5]:>7}{h[6]:>9}")
    for r in res["reserve_eval"]:
        rpt = r["reserve_ret_per_trade"] or 0
        print(
            f"    {r['name'][:17]:<18}{r['symbol']:<8}{r['timeframe']:<4}"
            f"{r['reserve_sharpe']:>7.2f}{r['reserve_return']:>9.2%}"
            f"{r['reserve_trades']:>7}{rpt:>9.3%}"
        )
    s = res["candidate_significance"]
    print("\n[Stage C] KC+Trend+Vol RESERVE 유의성:")
    print(
        f"    trades {s['reserve_trades']}  win% {s.get('reserve_win_rate')}  "
        f"total {s['reserve_total_return']:.2%}  buy&hold {s.get('buy_hold_reserve_return')}"
    )
    if "mean_ret_per_trade" in s:
        print(
            f"    per-trade mean {s['mean_ret_per_trade']:.3%}  t={s['t_stat_per_trade']}  "
            f"boot95%CI[{s['boot_ci95_low']:.3%}, {s['boot_ci95_high']:.3%}]"
        )
    print(f"    → {s['verdict']}")


def self_check() -> None:
    # in-sample holdout start must land at the 80% mark of the ≤T frame.
    import pandas as pd

    idx = pd.date_range("2024-01-01", periods=1000, freq="h", tz="UTC")
    df = pd.DataFrame({"close": np.arange(1000.0)}, index=idx)
    c = _insample_holdout_start(df)
    assert c == str(idx[800]), f"holdout start wrong: {c}"
    # block bootstrap CI must bracket a strongly-positive mean and exclude zero.
    rng = np.random.default_rng(1)
    x = rng.normal(0.02, 0.01, 200)  # clearly positive edge
    m, lo, hi = _block_bootstrap_mean(x, BLOCK, 1000)
    assert lo > 0 and lo < m < hi, f"boot CI failed: {lo},{m},{hi}"
    # exactly-zero-mean series (demeaned) CI must span zero.
    z = rng.normal(0.0, 0.01, 300)
    z = z - z.mean()  # pin sample mean to 0 so "spans zero" isn't a lucky draw
    _, lz, hz = _block_bootstrap_mean(z, BLOCK, 1000)
    assert lz < 0 < hz, f"zero-mean CI should span 0: {lz},{hz}"
    print("self-check OK (holdout split @80%, boot CI brackets positive edge & spans zero-mean)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2025-09-30", help="reserve everything after this date")
    ap.add_argument("--top", type=int, default=10, help="top-K in-sample winners to reserve-eval")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        self_check()
        return
    res = run(args.cutoff, args.top)
    _print(res)
    out = Path("results/next_steps/nested_holdout_revalidation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
