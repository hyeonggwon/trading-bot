"""Action 2 — 규칙 top-5 강건성: 다중검정 신기루 + 베타/알파 분해 (심의 critic#2 + scientist).

두 개의 독립 질문을 판정한다:

1) **다중검정 (garden-of-forking-paths).** 파이프라인은 스캔에서 M개 조합을 태워 in-sample
   holdout Sharpe로 top-N을 뽑는다. M이 크면 최고 Sharpe는 *실력이 아니라 M번 뽑기의 극단값*
   일 수 있다. 스캔 Sharpe 분포(μ, σ)로 **귀무가설 하 기대 최대 Sharpe**(Gumbel 근사, Bailey &
   López de Prado deflated-Sharpe 정신)를 구해 실측 최대와 비교한다. 실측 최대가 기대 최대에
   못 미치면 in-sample Sharpe 선택은 정보가 없다는 뜻 — 배포 근거는 OOS 워크포워드 생존뿐.

2) **베타/알파 분해.** OOS를 통과한 후보의 수익이 진짜 alpha인지, 아니면 특정 상승 레짐의
   beta(롱 노출)를 레버 없이 잘 탄 것인지 분리한다. 후보의 11개 워크포워드 test 창별 수익을
   같은 창의 buy&hold 수익에 회귀 → alpha(절편)·beta(기울기)·R²·alpha t-stat. alpha≈0이면
   "그냥 롱 XRP"이고 배포 엣지가 아니다.

정직성 장치:
- 창은 겹치지 않는 2개월 OOS 슬라이스(파이프라인 stage2 산출) — 재계산 없이 그대로 사용.
- buy&hold 수익 = 창 [test_start, test_end] 내 첫/마지막 종가 비율. 전략과 동일 창.
- 표본 11개(창 수)라 얇다 — 방향 판정용, 유의성은 t-stat과 함께 보수적으로 읽을 것.

Usage:
  .venv/bin/python scripts/rule_robustness_check.py            # 최신 파이프라인 런
  .venv/bin/python scripts/rule_robustness_check.py --run <run_id>
  .venv/bin/python scripts/rule_robustness_check.py --self-check
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tradingbot.data.storage import load_candles

PIPE_DIR = Path("results/pipeline")
DATA_DIR = Path("data")
EULER = 0.5772156649015329  # Euler-Mascheroni, for the Gumbel max correction


def expected_max_under_null(mu: float, sigma: float, m: int) -> float:
    """E[max of m iid N(mu, sigma^2)] via the standard Gumbel approximation.

    Under the null that all m scan Sharpes are noise draws with this dispersion,
    the best one you'd *expect* to see by chance. If the observed max <= this,
    in-sample Sharpe selection carries no signal.
    """
    if m < 2:
        return mu
    a = math.sqrt(2.0 * math.log(m))
    b = a - (math.log(math.log(m)) + math.log(4.0 * math.pi)) / (2.0 * a)
    return mu + sigma * (b + EULER / a)


def multiple_testing(run: Path) -> dict:
    r = json.loads((run / "stage1_scan.json").read_text())["results"]
    sh = np.array([x["sharpe_ratio"] for x in r if x.get("sharpe_ratio") is not None], dtype=float)
    m = len(sh)
    mu, sigma, obs_max = float(sh.mean()), float(sh.std(ddof=1)), float(sh.max())
    exp_max_null = expected_max_under_null(mu, sigma, m)
    exp_max_zero = expected_max_under_null(0.0, sigma, m)  # null centered at zero edge
    # z-score of the observed max against the noise dispersion (how many sigma out)
    z_obs = (obs_max - mu) / sigma if sigma else float("nan")
    return {
        "m_scanned": m,
        "scan_sharpe_mean": round(mu, 4),
        "scan_sharpe_std": round(sigma, 4),
        "observed_max_scan_sharpe": round(obs_max, 4),
        "expected_max_under_null_centered": round(exp_max_null, 4),
        "expected_max_under_null_zero_mean": round(exp_max_zero, 4),
        "observed_max_z": round(z_obs, 4),
        "verdict": (
            "in-sample Sharpe selection is NOISE (obs max <= expected-max-under-null)"
            if obs_max <= exp_max_null
            else "observed max EXCEEDS noise expectation — some in-sample signal"
        ),
    }


def _bh_return(symbol: str, tf: str, start: str, end: str) -> float | None:
    df = _bh_return.cache.get((symbol, tf))  # type: ignore[attr-defined]
    if df is None:
        df = load_candles(symbol, tf, DATA_DIR)
        _bh_return.cache[(symbol, tf)] = df  # type: ignore[attr-defined]
    win = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    if len(win) < 2:
        return None
    return float(win["close"].iloc[-1] / win["close"].iloc[0] - 1.0)


_bh_return.cache = {}  # type: ignore[attr-defined]


def beta_alpha(candidate: dict) -> dict:
    """Regress a candidate's per-window OOS returns on same-window buy&hold."""
    c = candidate["candidate"]
    sym, tf = c["symbol"], c["timeframe"]
    strat, mkt = [], []
    for w in candidate["windows"]:
        bh = _bh_return(sym, tf, w["test_start"], w["test_end"])
        if bh is None:
            continue
        strat.append(w["test_return"])
        mkt.append(bh)
    strat, mkt = np.array(strat), np.array(mkt)
    n = len(strat)
    out = {
        "name": c["name"],
        "symbol": sym,
        "timeframe": tf,
        "n_windows": n,
        "strat_mean_ret": round(float(strat.mean()), 5),
        "market_mean_ret": round(float(mkt.mean()), 5),
        "oos_sharpe": round(candidate["summary"]["avg_test_sharpe"], 4),
    }
    if n >= 3 and np.std(mkt) > 0:
        lin = stats.linregress(mkt, strat)  # strat = alpha + beta*market
        # two-sided t-stat on alpha (intercept) via its standard error
        resid = strat - (lin.intercept + lin.slope * mkt)
        sse = float(np.sum(resid**2))
        se_alpha = math.sqrt(
            sse / (n - 2) * (1.0 / n + mkt.mean() ** 2 / np.sum((mkt - mkt.mean()) ** 2))
        )
        t_alpha = lin.intercept / se_alpha if se_alpha else float("nan")
        out.update(
            {
                "alpha_per_window": round(float(lin.intercept), 5),
                "beta": round(float(lin.slope), 4),
                "r_squared": round(float(lin.rvalue**2), 4),
                "alpha_t_stat": round(float(t_alpha), 3),
                "alpha_p_value": round(float(2 * stats.t.sf(abs(t_alpha), n - 2)), 4),
                "verdict": (
                    "ALPHA present (t>2, intercept>0)"
                    if (t_alpha > 2 and lin.intercept > 0)
                    else "no significant alpha — return is mostly beta/regime"
                ),
            }
        )
    else:
        out["verdict"] = "too few windows for regression"
    return out


def run(run_id: str | None) -> dict:
    run = PIPE_DIR / run_id if run_id else max(PIPE_DIR.iterdir(), key=lambda p: p.name)
    mt = multiple_testing(run)
    wf = json.loads((run / "stage2_walkforward.json").read_text())["results"]
    ba = [beta_alpha(c) for c in wf]
    return {"run": run.name, "multiple_testing": mt, "beta_alpha": ba}


def _print(res: dict) -> None:
    mt = res["multiple_testing"]
    print(f"\n=== 다중검정 (run {res['run']}) ===")
    print(f"  조합 스캔 수 M            : {mt['m_scanned']}")
    print(
        f"  스캔 Sharpe 분포          : mean {mt['scan_sharpe_mean']}  std {mt['scan_sharpe_std']}"
    )
    print(
        f"  실측 최대 스캔 Sharpe     : {mt['observed_max_scan_sharpe']}  ({mt['observed_max_z']}σ)"
    )
    print(f"  귀무 기대 최대(중심)      : {mt['expected_max_under_null_centered']}")
    print(f"  귀무 기대 최대(zero-mean) : {mt['expected_max_under_null_zero_mean']}")
    print(f"  → {mt['verdict']}")
    print("\n=== 베타/알파 분해 (11 OOS 창 회귀) ===")
    hdr = ("candidate", "sym", "OOSsh", "stratμ", "mktμ", "beta", "alpha", "t", "R²")
    print(
        f"  {hdr[0]:<22}{hdr[1]:<9}{hdr[2]:>6}{hdr[3]:>8}{hdr[4]:>8}"
        f"{hdr[5]:>7}{hdr[6]:>8}{hdr[7]:>6}{hdr[8]:>6}"
    )
    for b in res["beta_alpha"]:
        if "beta" in b:
            print(
                f"  {b['name'][:21]:<22}{b['symbol']:<9}{b['oos_sharpe']:>6.2f}"
                f"{b['strat_mean_ret']:>8.4f}{b['market_mean_ret']:>8.4f}"
                f"{b['beta']:>7.3f}{b['alpha_per_window']:>8.4f}"
                f"{b['alpha_t_stat']:>6.2f}{b['r_squared']:>6.2f}"
            )
        else:
            print(f"  {b['name'][:21]:<22}{b['symbol']:<9}  {b['verdict']}")
    for b in res["beta_alpha"]:
        tag = b.get("verdict", "")
        print(f"    - {b['name']}: {tag}")


def self_check() -> None:
    # Gumbel expected-max is increasing in M and in sigma; centered null shifts by mu.
    assert expected_max_under_null(0, 1, 1000) > expected_max_under_null(0, 1, 10)
    assert expected_max_under_null(0, 2, 100) > expected_max_under_null(0, 1, 100)
    assert abs(expected_max_under_null(5, 1, 100) - expected_max_under_null(0, 1, 100) - 5) < 1e-9
    # Planted alpha: strat = 0.05 + 0.0*market + tiny noise -> alpha≈0.05, beta≈0, t large.
    rng = np.random.default_rng(0)
    mkt = rng.normal(0, 0.1, 40)
    strat = 0.05 + 0.0 * mkt + rng.normal(0, 0.002, 40)
    lin = stats.linregress(mkt, strat)
    assert abs(lin.intercept - 0.05) < 0.01 and abs(lin.slope) < 0.05, "planted alpha"
    # Planted pure beta: strat = 0 + 1.2*market -> alpha≈0, beta≈1.2.
    strat2 = 1.2 * mkt + rng.normal(0, 0.001, 40)
    lin2 = stats.linregress(mkt, strat2)
    assert abs(lin2.intercept) < 0.01 and abs(lin2.slope - 1.2) < 0.05, "planted beta"
    print("self-check OK (Gumbel monotonicity + mean-shift, planted alpha & beta)")


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
    out = Path("results/next_steps/rule_robustness_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
