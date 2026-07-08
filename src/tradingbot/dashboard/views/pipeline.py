"""Pipeline page: run the selection pipeline + browse run results stage by stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from tradingbot.dashboard import forms
from tradingbot.dashboard.views import common

RUNS_ROOT = Path("results/pipeline")


def render() -> None:
    st.subheader("Pipeline — scan → select → walk-forward → rank → deploy")
    st.caption(
        "One run automates the recurring workflow: scan everything, keep the "
        "top candidates, walk-forward them, rank by out-of-sample results and "
        "generate deploy artifacts (never auto-executed)."
    )
    with st.expander("**pipeline** — start a new run (background job)"):
        result = forms.render_command_form("pipeline")
        if result is not None:
            common.submit_job("pipeline", result.args)
    st.divider()
    _render_run_browser()


def _render_run_browser() -> None:
    runs = (
        sorted((d for d in RUNS_ROOT.iterdir() if d.is_dir()), reverse=True)
        if RUNS_ROOT.is_dir()
        else []
    )
    if not runs:
        st.info("No pipeline runs yet — start one above; progress shows on the Jobs page.")
        return

    run_name = st.selectbox("Run", [d.name for d in runs])
    run_dir = RUNS_ROOT / str(run_name)
    manifest = _load_json(run_dir / "manifest.json") or {}
    opts: dict[str, Any] = manifest.get("options", {})
    status = manifest.get("status", "?")
    st.caption(
        f"status: **{status}** · select top {opts.get('top')} by {opts.get('sort_by')} "
        f"(min {opts.get('min_trades')} trades) · WF {opts.get('train_months')}m/"
        f"{opts.get('test_months')}m · rank by {opts.get('rank_by')}"
    )
    if status == "running":
        st.info("Run in progress — tables appear as stages finish (log on the Jobs page).")
    elif status == "failed":
        st.error(f"Run failed: {manifest.get('error', 'see Jobs page log')}")

    stage0 = _load_json(run_dir / "stage0_ml_train.json")
    if stage0 is not None:
        _render_stage0(stage0)

    tab_scan, tab_wf, tab_rank = st.tabs(["1 · Scan", "2 · Validation", "3 · Ranking & Deploy"])
    with tab_scan:
        _render_scan_tab(run_dir, opts)
    with tab_wf:
        _render_wf_tab(run_dir)
    with tab_rank:
        _render_ranking_tab(run_dir)


def _render_stage0(stage0: dict[str, Any]) -> None:
    import pandas as pd

    trained = stage0.get("trained", [])
    fresh = stage0.get("fresh", [])
    failed = stage0.get("failed", [])
    label = f"0 · ML Train — trained {len(trained)} · fresh {len(fresh)} · failed {len(failed)}"
    with st.expander(label):
        for title, rows in (("Trained", trained), ("Failed", failed), ("Fresh (skipped)", fresh)):
            if rows:
                st.markdown(f"**{title}**")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_scan_tab(run_dir: Path, opts: dict[str, Any]) -> None:
    stage1 = _load_json(run_dir / "stage1_scan.json")
    if stage1 is None:
        st.warning("Stage 1 not written yet.")
        return
    selection = _load_json(run_dir / "selection.json") or {}
    selected_keys = {
        (c["name"], c["symbol"], c["timeframe"]) for c in selection.get("selected", [])
    }

    import pandas as pd

    rows = [
        {
            "Selected": "✔"
            if (r["strategy"], r["symbol"], r["timeframe"]) in selected_keys
            else "",
            "Candidate": r["strategy"],
            "Symbol": r["symbol"],
            "TF": r["timeframe"],
            "Sharpe": r["sharpe_ratio"],
            "Return": r["total_return"],
            "MaxDD": r["max_drawdown"],
            "Win%": r["win_rate"],
            "PF": r["profit_factor"],
            "Trades": r["total_trades"],
        }
        for r in stage1.get("results", [])
    ]
    if not rows:
        st.info("Scan produced no results.")
        return
    sort_col = {
        "sharpe_ratio": "Sharpe",
        "total_return": "Return",
        "max_drawdown": "MaxDD",
        "win_rate": "Win%",
        "profit_factor": "PF",
        "total_trades": "Trades",
    }.get(str(opts.get("sort_by")), "Sharpe")
    df = pd.DataFrame(rows).sort_values(sort_col, ascending=sort_col == "MaxDD").round(3)
    st.caption(f"{len(rows)} scan results ({stage1.get('window', '')}) — sorted by {sort_col}")
    st.dataframe(df, use_container_width=True, hide_index=True)

    excluded = selection.get("excluded", [])
    if excluded:
        with st.expander(f"Excluded candidates ({len(excluded)})"):
            st.dataframe(pd.DataFrame(excluded), use_container_width=True, hide_index=True)


def _render_wf_tab(run_dir: Path) -> None:
    stage2 = _load_json(run_dir / "stage2_walkforward.json")
    if stage2 is None:
        st.warning("Stage 2 not written yet.")
        return

    import pandas as pd

    results = stage2.get("results", [])
    if not results:
        st.info("No walk-forward results.")
        return
    rows = [
        {
            "Candidate": r["candidate"]["name"],
            "Symbol": r["candidate"]["symbol"],
            "TF": r["candidate"]["timeframe"],
            "Validation": r.get("validation", "walk_forward"),
            "Windows": r["summary"]["num_windows"],
            "Train Sharpe (IS)": r["summary"]["avg_train_sharpe"],
            "Test Sharpe (OOS)": r["summary"]["avg_test_sharpe"],
            "WF Efficiency": r["summary"]["walk_forward_efficiency"],
            "OOS Cum Return": r["summary"]["cumulative_test_return"],
            "OOS Trades": r["summary"]["total_test_trades"],
        }
        for r in results
    ]
    df = pd.DataFrame(rows).sort_values("Test Sharpe (OOS)", ascending=False).round(3)
    st.dataframe(df, use_container_width=True, hide_index=True)

    labels = [
        f"{r['candidate']['name']} · {r['candidate']['symbol']} {r['candidate']['timeframe']}"
        for r in results
    ]
    pick = st.selectbox("Per-window detail", labels)
    detail = results[labels.index(pick)]
    win_rows = [
        {
            "#": w["window_index"] + 1,
            "Train": f"{w['train_start'][:10]} ~ {w['train_end'][:10]}",
            "Test": f"{w['test_start'][:10]} ~ {w['test_end'][:10]}",
            "Params": ", ".join(f"{k}={v}" for k, v in w["best_params"].items()),
            "Train Sharpe": w["train_sharpe"],
            "Test Sharpe": w["test_sharpe"],
            "Test Return": w["test_return"],
            "Trades": w["test_trades"],
        }
        for w in detail.get("windows", [])
    ]
    if win_rows:
        st.dataframe(pd.DataFrame(win_rows).round(3), use_container_width=True, hide_index=True)


def _render_ranking_tab(run_dir: Path) -> None:
    ranking_doc = _load_json(run_dir / "ranking.json")
    if ranking_doc is None:
        st.warning("Ranking not written yet.")
        return
    ranking = ranking_doc.get("ranking", [])
    if not ranking:
        st.info("No ranked candidates.")
        return

    import pandas as pd

    rows = [
        {
            "Rank": r["rank"],
            "Candidate": r["name"] + (" ⚠low-trades" if r["low_trades"] else ""),
            "Symbol": r["symbol"],
            "TF": r["timeframe"],
            "Validation": r.get("validation", "-"),
            "Scan Sharpe (holdout)": r["scan_sharpe"],
            "OOS Sharpe": r["oos_sharpe"],
            "Holdout→OOS Δ": (
                r["oos_sharpe"] - r["scan_sharpe"] if r["scan_sharpe"] is not None else None
            ),
            "WF Efficiency": r["wf_efficiency"],
            "OOS Cum Return": r["oos_cum_return"],
            "OOS Trades": r["oos_trades"],
        }
        for r in ranking
    ]
    st.caption(
        f"Ranked by {ranking_doc.get('rank_by')} — a large negative Holdout→OOS Δ "
        "means the scan number didn't hold up across walk-forward windows "
        "(regime/parameter sensitivity). ML-WF rows compare a tuned disk model (scan) "
        "against fresh default-param per-window models (OOS), so some gap is expected."
    )
    st.dataframe(pd.DataFrame(rows).round(3), use_container_width=True, hide_index=True)

    _render_is_oos_scatter(ranking)
    _render_winner_card(run_dir, ranking[0])


def _render_is_oos_scatter(ranking: list[dict[str, Any]]) -> None:
    points = [r for r in ranking if r["scan_sharpe"] is not None]
    if len(points) < 2:
        return
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Scatter(
            x=[r["scan_sharpe"] for r in points],
            y=[r["oos_sharpe"] for r in points],
            mode="markers+text",
            text=[r["name"] for r in points],
            textposition="top center",
            marker=dict(size=10, color="#2196F3"),
        )
    )
    lo = min(min(r["scan_sharpe"] for r in points), min(r["oos_sharpe"] for r in points))
    hi = max(max(r["scan_sharpe"] for r in points), max(r["oos_sharpe"] for r in points))
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line=dict(color="#90CAF9", dash="dot"),
            showlegend=False,
        )
    )
    fig.update_layout(
        xaxis_title="Scan Sharpe (holdout window)",
        yaxis_title="Walk-forward Sharpe (out-of-sample)",
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    st.caption(
        "Below the dotted line = the scan number didn't hold up out-of-sample — "
        "treat that candidate's scan Sharpe with suspicion."
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_winner_card(run_dir: Path, winner: dict[str, Any]) -> None:
    st.markdown("### 🏆 Winner — deploy artifacts (generated only, review before running)")
    st.success(
        f"{winner['name']} on {winner['symbol']} {winner['timeframe']}"
        + (" — ⚠ low OOS trade count, treat with suspicion" if winner["low_trades"] else "")
    )
    if winner["oos_sharpe"] <= 0:
        st.warning(
            "No candidate showed a positive out-of-sample edge — these artifacts "
            "are for inspection, not deployment."
        )
    deploy = run_dir / "deploy"
    for fname, lang in (
        ("paper.sh", "bash"),
        ("live.sh", "bash"),
        ("docker-compose.override.yml", "yaml"),
    ):
        path = deploy / fname
        if path.exists():
            st.markdown(f"**`{fname}`**")
            st.code(path.read_text(), language=lang)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data: dict[str, Any] = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data
