"""Action 3 (1단계, 백테스트 전용) — 규칙 top-5의 비용 민감도 스윕 (scientist#1 + critic#3a).

라이브 지정가 계측 코드를 짜기 전에 먼저 **백테스트로** 답할 것: 슬리피지를 바꾸면 후보의
net 성과가 어떻게 움직이나? 두 방향 다 본다.
- 아래로(0 → 실집행 개선분): 비용에 눌린 한계 후보가 지정가 체결로 살아나는가?
- 위로(가정보다 나쁜 슬리피지): 생존 후보(#1)가 실슬리피지가 가정의 2~3배여도 버티는가(취약성)?

fee는 고정(Upbit 메이커=테이커 0.05%/side = 왕복 0.1%, 불가피). slippage만 스윕한다.
왕복비용 = 2×(fee 0.0005 + slippage). baseline slippage=0.001 → 왕복 0.3%.

정직성 장치:
- 슬리피지는 체결가만 바꾸고 진입 트리거(종가 로직)는 안 바꾼다 → 같은 트레이드가 더 좋은/나쁜
  가격에 체결. self-check가 "슬리피지 0 net ≥ 높은 슬리피지 net, 그리고 값이 실제로 다름"으로
  노브 배선을 검증.
- 창은 combine 기본과 동일(auto holdout, 마지막 20%) — 절대 Sharpe는 파이프라인 WF-OOS와
  다르지만 슬리피지 간 *상대 변화*가 답이다.
- 후보/entry/exit는 최신 파이프라인 ranking.json에서 그대로 읽는다(재선택 없음).

Usage:
  .venv/bin/python scripts/cost_sensitivity_sweep.py            # 최신 런 top-5
  .venv/bin/python scripts/cost_sensitivity_sweep.py --run <run_id>
  .venv/bin/python scripts/cost_sensitivity_sweep.py --self-check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tradingbot.backtest.engine import BacktestEngine
from tradingbot.backtest.holdout import resolve_holdout_window
from tradingbot.config import load_config
from tradingbot.data.storage import load_candles
from tradingbot.strategy.combined import CombinedStrategy
from tradingbot.strategy.filters.registry import parse_filter_string

PIPE_DIR = Path("results/pipeline")
DATA_DIR = Path("data")
FEE_RATE = 0.0005  # Upbit maker=taker per side, fixed
# per-side slippage grid: 0 (perfect limit) → 0.003 (3x the flat assumption)
SLIPPAGE_GRID = [0.0, 0.0005, 0.001, 0.002, 0.003]
BASELINE_SLIP = 0.001
BALANCE = 1_000_000


def _build_strategy(entry: str, exit_: str, symbol: str, tf: str) -> CombinedStrategy:
    ef = parse_filter_string(entry, base_timeframe=tf)
    xf = parse_filter_string(exit_, base_timeframe=tf)
    for f in ef + xf:  # ML filters need symbol/timeframe; rule filters ignore
        if hasattr(f, "symbol"):
            f.symbol = symbol
        if hasattr(f, "timeframe"):
            f.timeframe = tf
    s = CombinedStrategy(entry_filters=ef, exit_filters=xf)
    s.symbols = [symbol]
    s.timeframe = tf
    return s


def _run_once(entry: str, exit_: str, symbol: str, tf: str, slippage: float, df) -> dict:
    strat = _build_strategy(entry, exit_, symbol, tf)
    config = load_config(
        Path("config"),
        overrides={
            "trading": {"symbols": [symbol], "timeframe": tf, "initial_balance": BALANCE},
            "backtest": {"slippage_pct": slippage, "fee_rate": FEE_RATE},
        },
    )
    report = BacktestEngine(strategy=strat, config=config).run({symbol: df})
    n = report.total_trades
    return {
        "slippage": slippage,
        "roundtrip_cost": round(2 * (FEE_RATE + slippage), 4),
        "total_return": round(report.total_return, 5),
        "sharpe": round(report.sharpe_ratio, 4),
        "trades": n,
        "net_ret_per_trade": round(report.total_return / n, 5) if n else None,
        "win_rate": round(report.win_rate, 4),
    }


def _holdout_df(symbol: str, tf: str):
    df = load_candles(symbol, tf, DATA_DIR)
    s, e, _ = resolve_holdout_window(df, None, None, include_train=False)
    if s:
        df = df[df.index >= s]
    if e:
        df = df[df.index <= e]
    return df


def sweep_candidate(cand: dict) -> dict:
    sym, tf = cand["symbol"], cand["timeframe"]
    df = _holdout_df(sym, tf)
    rows = [_run_once(cand["entry"], cand["exit"], sym, tf, sl, df) for sl in SLIPPAGE_GRID]
    return {
        "name": cand["name"],
        "symbol": sym,
        "timeframe": tf,
        "entry": cand["entry"],
        "exit": cand["exit"],
        "holdout_candles": len(df),
        "rows": rows,
    }


def run(run_id: str | None) -> dict:
    run = PIPE_DIR / run_id if run_id else max(PIPE_DIR.iterdir(), key=lambda p: p.name)
    ranking = json.loads((run / "ranking.json").read_text())["ranking"]
    cands = [c for c in ranking if c.get("kind") == "combined" and c.get("entry")]
    return {"run": run.name, "candidates": [sweep_candidate(c) for c in cands]}


def _print(res: dict) -> None:
    print(f"\n=== 비용 민감도 스윕 (run {res['run']}, holdout window) ===")
    print("  slippage/side: 0=완전지정가 · 0.001=baseline(왕복0.3%) · 0.003=가정의3배")
    for c in res["candidates"]:
        print(f"\n{c['name']}  {c['symbol']} {c['timeframe']}  ({c['holdout_candles']} candles)")
        print(f"  entry: {c['entry']}   exit: {c['exit']}")
        h = ("slip/side", "RTcost", "return", "sharpe", "trades", "net/trade")
        print(f"  {h[0]:>10}{h[1]:>8}{h[2]:>10}{h[3]:>8}{h[4]:>7}{h[5]:>10}")
        for r in c["rows"]:
            flag = "  *base" if r["slippage"] == BASELINE_SLIP else ""
            print(
                f"  {r['slippage']:>10.4f}{r['roundtrip_cost']:>8.4f}"
                f"{r['total_return']:>10.2%}{r['sharpe']:>8.2f}{r['trades']:>7}"
                f"{(r['net_ret_per_trade'] or 0):>10.4%}{flag}"
            )


def self_check() -> None:
    # Cost knob must be wired: same candidate, zero slippage nets >= high slippage,
    # and the two differ (else slippage_pct isn't reaching the simulator).
    run = max(PIPE_DIR.iterdir(), key=lambda p: p.name)
    ranking = json.loads((run / "ranking.json").read_text())["ranking"]
    c = next(x for x in ranking if x.get("kind") == "combined" and x.get("entry"))
    df = _holdout_df(c["symbol"], c["timeframe"])
    lo = _run_once(c["entry"], c["exit"], c["symbol"], c["timeframe"], 0.0, df)
    hi = _run_once(c["entry"], c["exit"], c["symbol"], c["timeframe"], 0.01, df)
    assert lo["trades"] > 0, "candidate must trade in holdout for a valid check"
    assert lo["total_return"] >= hi["total_return"] - 1e-9, (
        f"zero slippage should not net worse: {lo['total_return']} vs {hi['total_return']}"
    )
    assert abs(lo["total_return"] - hi["total_return"]) > 1e-9, (
        "slippage knob has no effect — not wired to simulator"
    )
    print(
        f"self-check OK (cost knob wired: return {hi['total_return']:.2%}@1%slip "
        f"→ {lo['total_return']:.2%}@0slip, {lo['trades']} trades)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="pipeline run_id; default = latest")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        self_check()
        return
    res = run(args.run)
    _print(res)
    out = Path("results/next_steps/cost_sensitivity_sweep.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
