"""Strategy selection pipeline: scan → select → walk-forward → rank → deploy artifacts.

Automates the operator's recurring workflow as one run:

1. **Scan** registry strategies + combine templates (reuses ``parallel._run_batch``)
2. **Select** top-N candidates (min-trades gate; ML candidates routed out —
   their time-honest validation is ``ml-walk-forward``)
3. **Walk-forward** each candidate (registry → per-window param optimization,
   combined → fixed filters)
4. **Rank** by an out-of-sample metric (scan numbers are in-sample-ish;
   the ranking uses only walk-forward OOS results)
5. **Deploy artifacts**: paper/live commands + docker-compose override for
   the winner — generated, NEVER executed (operator reviews and runs them)

Every stage persists JSON into ``results/pipeline/<run_id>/`` so the
dashboard Pipeline page can render stage-by-stage comparison tables.
Typer-free: the CLI wrapper (cli/pipeline.py) and tests share this module.
"""

from __future__ import annotations

import json
import shlex
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tradingbot.utils.io import atomic_write_json

SORT_METRICS = frozenset(
    {"sharpe_ratio", "total_return", "max_drawdown", "win_rate", "profit_factor", "total_trades"}
)
RANK_METRICS = frozenset({"avg_test_sharpe", "cumulative_test_return", "walk_forward_efficiency"})

# ponytail: service name matches docker-compose.yml; parsing compose to
# discover it is overkill for a single-service repo.
COMPOSE_SERVICE = "bot"
DOCKER_STATE_FILE = "/app/state/state.json"

_ML_REASON = "ml-candidate: time-honest validation is `tradingbot ml-walk-forward`"


class PipelineError(RuntimeError):
    """Pipeline cannot proceed (no data, empty selection, ...)."""


@dataclass(frozen=True)
class PipelineOptions:
    """Knobs for one pipeline run (CLI options mirror these 1:1)."""

    top: int = 5
    min_trades: int = 10
    sort_by: str = "sharpe_ratio"
    rank_by: str = "avg_test_sharpe"
    train_months: int = 3
    test_months: int = 1
    workers: int = 0
    balance: float = 1_000_000
    data_dir: str = "data"
    include_train: bool = False
    output_dir: str = "results/pipeline"
    skip_rules: bool = False
    skip_combine: bool = False


@dataclass(frozen=True)
class Candidate:
    """One walk-forward candidate distilled from a scan result."""

    name: str
    symbol: str
    timeframe: str
    kind: str  # "strategy" | "combined"
    entry: str = ""
    exit: str = ""


def run_pipeline(
    options: PipelineOptions,
    templates: list[dict[str, str]],
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run all five stages; returns a summary dict (run_dir, ranking, winner)."""
    if options.sort_by not in SORT_METRICS:
        raise PipelineError(
            f"invalid --sort-by {options.sort_by!r} (use one of {sorted(SORT_METRICS)})"
        )
    if options.rank_by not in RANK_METRICS:
        raise PipelineError(
            f"invalid --rank-by {options.rank_by!r} (use one of {sorted(RANK_METRICS)})"
        )
    if options.top < 1:
        raise PipelineError(f"--top must be >= 1 (got {options.top})")
    if options.skip_rules and options.skip_combine:
        raise PipelineError("--skip-rules and --skip-combine together leave nothing to scan")

    started = datetime.now(UTC)
    run_id = started.strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(options.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at": started.isoformat(),
        "options": asdict(options),
        "status": "running",
        "stages": {},
    }
    atomic_write_json(run_dir / "manifest.json", manifest)

    try:
        return _run_stages(options, templates, log, run_dir, manifest)
    except Exception as e:
        # The Jobs page reports the process exit code, but the Pipeline run
        # browser reads this status — without it a crashed run would show
        # "running" forever.
        manifest["status"] = "failed"
        manifest["error"] = str(e)
        atomic_write_json(run_dir / "manifest.json", manifest)
        raise


def _run_stages(
    options: PipelineOptions,
    templates: list[dict[str, str]],
    log: Callable[[str], None],
    run_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    run_id = str(manifest["run_id"])

    def _stage_done(name: str, t0: float, **info: Any) -> None:
        manifest["stages"][name] = {"duration_sec": round(time.monotonic() - t0, 1), **info}
        atomic_write_json(run_dir / "manifest.json", manifest)

    # ── Stage 1: scan ────────────────────────────────────────────────
    t0 = time.monotonic()
    log(f"[1/5] scan: rules={not options.skip_rules} combine={not options.skip_combine}")
    scan_results, failures, skipped_ml = _scan_stage(options, templates, log)
    atomic_write_json(
        run_dir / "stage1_scan.json",
        {
            "window": "full range"
            if options.include_train
            else "auto holdout (last 20% per batch)",
            "results": [asdict(r) for r in scan_results],
            "failures": failures,
        },
    )
    _stage_done("scan", t0, results=len(scan_results), failures=len(failures))

    # ── Stage 2: select ──────────────────────────────────────────────
    t0 = time.monotonic()
    selected, excluded = select_candidates(
        scan_results, top=options.top, min_trades=options.min_trades, sort_by=options.sort_by
    )
    excluded = skipped_ml + excluded
    log(f"[2/5] select: {len(selected)} candidates (excluded {len(excluded)})")
    atomic_write_json(
        run_dir / "selection.json",
        {
            "sort_by": options.sort_by,
            "min_trades": options.min_trades,
            "selected": [asdict(c) for c in selected],
            "excluded": excluded,
        },
    )
    _stage_done("select", t0, selected=len(selected), excluded=len(excluded))
    if not selected:
        raise PipelineError(
            f"no candidates survived selection (min_trades={options.min_trades}) "
            f"— see {run_dir}/selection.json"
        )

    # ── Stage 3: walk-forward ────────────────────────────────────────
    t0 = time.monotonic()
    wf_results = _walk_forward_stage(selected, options, log)
    atomic_write_json(run_dir / "stage2_walkforward.json", {"results": wf_results})
    _stage_done("walk_forward", t0, candidates=len(wf_results))

    # ── Stage 4: rank ────────────────────────────────────────────────
    t0 = time.monotonic()
    ranking = rank_candidates(
        wf_results, scan_results, rank_by=options.rank_by, min_trades=options.min_trades
    )
    winner = ranking[0] if ranking else None
    log(f"[4/5] rank: winner = {winner['name'] if winner else 'none'} (by {options.rank_by})")
    atomic_write_json(run_dir / "ranking.json", {"rank_by": options.rank_by, "ranking": ranking})
    _stage_done("rank", t0, candidates=len(ranking))

    # ── Stage 5: deploy artifacts ────────────────────────────────────
    t0 = time.monotonic()
    artifacts: list[str] = []
    if winner is not None:
        artifacts = [str(p) for p in write_deploy_artifacts(run_dir, winner, options)]
        log(f"[5/5] deploy artifacts: {len(artifacts)} files (generated only — review first)")
    _stage_done("deploy_artifacts", t0, files=len(artifacts))

    _write_summary_md(run_dir, manifest, ranking, artifacts)
    manifest["status"] = "complete"
    atomic_write_json(run_dir / "manifest.json", manifest)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "scanned": len(scan_results),
        "selected": len(selected),
        "ranking": ranking,
        "winner": winner,
        "artifacts": artifacts,
    }


# ── Stage 1: scan ────────────────────────────────────────────────────


def _scan_stage(
    options: PipelineOptions,
    templates: list[dict[str, str]],
    log: Callable[[str], None],
) -> tuple[list[Any], list[str], list[dict[str, Any]]]:
    """Scan all (symbol, timeframe) batches; returns (results, failures, skipped_ml)."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from tradingbot.backtest.parallel import ScanResult, _run_batch
    from tradingbot.data.storage import list_available_data

    available = list_available_data(Path(options.data_dir))
    if not available:
        raise PipelineError(f"no data in {options.data_dir!r} — run `tradingbot download` first")

    symbol_timeframes: dict[str, list[str]] = {}
    for item in available:
        symbol_timeframes.setdefault(item["symbol"], []).append(item["timeframe"])

    jobs: list[tuple[str, str, str]] = []
    skipped_ml: list[dict[str, Any]] = []
    if not options.skip_rules:
        from tradingbot.strategy.registry import get_strategy_map

        for name in get_strategy_map():
            if name == "lgbm":
                # symbol/timeframe "*": skipped before any per-batch scan ran
                skipped_ml.append(
                    {"name": name, "symbol": "*", "timeframe": "*", "reason": _ML_REASON}
                )
            else:
                jobs.append((name, "", ""))
    if not options.skip_combine:
        for tmpl in templates:
            if "lgbm_prob" in tmpl["entry"]:
                skipped_ml.append(
                    {"name": tmpl["label"], "symbol": "*", "timeframe": "*", "reason": _ML_REASON}
                )
            else:
                jobs.append((tmpl["label"], tmpl["entry"], tmpl["exit"]))
    if not jobs:
        raise PipelineError("nothing to scan after ML exclusions")

    batches = {(sym, tf): jobs for sym, tfs in symbol_timeframes.items() for tf in tfs}
    cpu = multiprocessing.cpu_count() or 4
    n_workers = options.workers if options.workers > 0 else min(cpu, 8)
    log(f"      {len(jobs)} jobs × {len(batches)} (symbol, timeframe) batches, {n_workers} workers")

    results: list[ScanResult] = []
    failures: list[str] = []
    with ProcessPoolExecutor(
        max_workers=n_workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {
            pool.submit(
                _run_batch,
                sym,
                tf,
                batch_jobs,
                str(Path(options.data_dir).resolve()),
                options.balance,
                str(Path("config").resolve()),
                False,
                None,
                None,
                options.include_train,
            ): (sym, tf)
            for (sym, tf), batch_jobs in batches.items()
        }
        for future in as_completed(futures):
            sym, tf = futures[future]
            try:
                batch = future.result(timeout=600)
            except Exception as exc:
                failures.append(f"{sym}/{tf}: worker crashed: {exc}")
                continue
            for res in batch:
                if res.error:
                    failures.append(f"{res.strategy}/{res.symbol}/{res.timeframe}: {res.error}")
                else:
                    results.append(res)
            log(f"      scanned {sym} {tf}")
    return results, failures, skipped_ml


# ── Stage 2: select ──────────────────────────────────────────────────


def select_candidates(
    scan_results: list[Any],
    *,
    top: int,
    min_trades: int,
    sort_by: str,
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    """Top-N candidates by ``sort_by`` after the min-trades / ML gates.

    Returns (selected, excluded-with-reason). Results below top-N are
    simply not selected — only gate failures land in ``excluded``.
    """
    excluded: list[dict[str, Any]] = []
    eligible = []
    for r in scan_results:
        ident = {"name": r.strategy, "symbol": r.symbol, "timeframe": r.timeframe}
        if r.strategy == "lgbm" or "lgbm_prob" in r.entry:
            excluded.append({**ident, "reason": _ML_REASON})
        elif r.total_trades < min_trades:
            excluded.append(
                {**ident, "reason": f"trades {r.total_trades} < min_trades {min_trades}"}
            )
        else:
            eligible.append(r)

    reverse = sort_by != "max_drawdown"
    eligible.sort(key=lambda r: getattr(r, sort_by), reverse=reverse)
    selected = [
        Candidate(
            name=r.strategy,
            symbol=r.symbol,
            timeframe=r.timeframe,
            kind="combined" if r.entry else "strategy",
            entry=r.entry,
            exit=r.exit,
        )
        for r in eligible[: max(0, top)]
    ]
    return selected, excluded


# ── Stage 3: walk-forward ────────────────────────────────────────────


def _walk_forward_stage(
    candidates: list[Candidate],
    options: PipelineOptions,
    log: Callable[[str], None],
) -> list[dict[str, Any]]:
    """Walk-forward each candidate sequentially, serializing the reports.

    # ponytail: sequential over ≤top candidates; parallelize per-candidate
    # if wall-clock ever hurts.
    """
    from tradingbot.backtest.walk_forward import WalkForwardValidator, walk_forward_combined
    from tradingbot.config import load_config
    from tradingbot.data.storage import load_candles

    out: list[dict[str, Any]] = []
    for i, cand in enumerate(candidates, 1):
        log(f"[3/5] walk-forward {i}/{len(candidates)}: {cand.name} {cand.symbol} {cand.timeframe}")
        df = load_candles(cand.symbol, cand.timeframe, Path(options.data_dir))
        config = load_config(
            Path("config"),
            overrides={
                "trading": {
                    "symbols": [cand.symbol],
                    "timeframe": cand.timeframe,
                    "initial_balance": options.balance,
                },
            },
        )
        if cand.kind == "strategy":
            from tradingbot.strategy.registry import get_strategy_map

            validator = WalkForwardValidator(
                strategy_cls=get_strategy_map()[cand.name],
                config=config,
                train_months=options.train_months,
                test_months=options.test_months,
            )
            report = validator.validate({cand.symbol: df})
        else:
            from tradingbot.strategy.combined import CombinedStrategy

            strategy = CombinedStrategy.from_filter_strings(
                cand.entry, cand.exit, cand.symbol, cand.timeframe
            )
            report = walk_forward_combined(
                strategy,
                cand.name,
                cand.symbol,
                df,
                config,
                train_months=options.train_months,
                test_months=options.test_months,
            )
        out.append({"candidate": asdict(cand), **serialize_wf_report(report)})
    return out


def serialize_wf_report(report: Any) -> dict[str, Any]:
    """WalkForwardReport → JSON-safe dict (summary + per-window rows)."""
    return {
        "summary": {
            "num_windows": report.num_windows,
            "avg_train_sharpe": report.avg_train_sharpe,
            "avg_test_sharpe": report.avg_test_sharpe,
            "avg_test_return": report.avg_test_return,
            "walk_forward_efficiency": report.walk_forward_efficiency,
            "overfitting_ratio": report.overfitting_ratio,
            "cumulative_test_return": report.cumulative_test_return,
            "total_test_trades": report.total_test_trades,
        },
        "windows": [
            {
                "window_index": w.window_index,
                "train_start": w.train_start.isoformat(),
                "train_end": w.train_end.isoformat(),
                "test_start": w.test_start.isoformat(),
                "test_end": w.test_end.isoformat(),
                "best_params": {k: str(v) for k, v in w.best_params.items()},
                "train_sharpe": w.train_sharpe,
                "train_return": w.train_return,
                "test_sharpe": w.test_sharpe,
                "test_return": w.test_return,
                "test_trades": w.test_trades,
                "test_max_drawdown": w.test_max_drawdown,
            }
            for w in report.windows
        ],
    }


# ── Stage 4: rank ────────────────────────────────────────────────────


def rank_candidates(
    wf_results: list[dict[str, Any]],
    scan_results: list[Any],
    *,
    rank_by: str,
    min_trades: int,
) -> list[dict[str, Any]]:
    """Sort candidates by OOS ``rank_by`` (tiebreak: WF efficiency).

    Candidates whose OOS trade count is below ``min_trades`` are flagged
    ``low_trades`` and always rank below the rest — a great Sharpe on
    three trades is noise, not signal.
    """
    scan_by_key = {(r.strategy, r.symbol, r.timeframe): r for r in scan_results}
    rows: list[dict[str, Any]] = []
    for wf in wf_results:
        cand = wf["candidate"]
        summary = wf["summary"]
        scan = scan_by_key.get((cand["name"], cand["symbol"], cand["timeframe"]))
        rows.append(
            {
                **cand,
                "scan_sharpe": scan.sharpe_ratio if scan else None,
                "scan_return": scan.total_return if scan else None,
                "oos_sharpe": summary["avg_test_sharpe"],
                "oos_cum_return": summary["cumulative_test_return"],
                "wf_efficiency": summary["walk_forward_efficiency"],
                "overfitting_ratio": summary["overfitting_ratio"],
                "oos_trades": summary["total_test_trades"],
                "num_windows": summary["num_windows"],
                "rank_value": summary[rank_by],
                "low_trades": summary["total_test_trades"] < min_trades,
            }
        )
    rows.sort(key=lambda r: (r["low_trades"], -r["rank_value"], -r["wf_efficiency"]))
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return rows


# ── Stage 5: deploy artifacts ────────────────────────────────────────


def _winner_argv(
    command: str, winner: dict[str, Any], *, state_file: str | None = None
) -> list[str]:
    argv = ["tradingbot", command]
    if winner["kind"] == "strategy":
        argv += ["--strategy", winner["name"]]
    else:
        argv += ["--entry", winner["entry"], "--exit", winner["exit"]]
    argv += ["--symbol", winner["symbol"], "--timeframe", winner["timeframe"]]
    if state_file is not None:
        argv += ["--state-file", state_file]
    return argv


def write_deploy_artifacts(
    run_dir: Path,
    winner: dict[str, Any],
    options: PipelineOptions,
) -> list[Path]:
    """Generate paper.sh / live.sh / docker-compose.override.yml for the winner.

    Artifacts are generated only — the pipeline never starts an engine or
    a container. The operator reviews and runs them.
    """
    deploy = run_dir / "deploy"
    deploy.mkdir(exist_ok=True)
    run_id = run_dir.name

    # Comments can only be escaped by a newline — strip them so no value can
    # smuggle a shell line into the script (argv itself is shlex-quoted).
    def _comment_safe(value: Any) -> str:
        return str(value).replace("\n", " ").replace("\r", " ")

    winner_desc = (
        f"{_comment_safe(winner['name'])} on "
        f"{_comment_safe(winner['symbol'])} {_comment_safe(winner['timeframe'])}"
    )
    paper_argv = _winner_argv("paper", winner) + ["--balance", str(int(options.balance))]
    paper_sh = deploy / "paper.sh"
    paper_sh.write_text(
        "#!/usr/bin/env bash\n"
        f"# Generated by `tradingbot pipeline` — run {run_id}\n"
        f"# Winner: {winner_desc} (rank_value={winner['rank_value']:.3f})\n"
        f"{shlex.join(paper_argv)}\n"
    )
    paper_sh.chmod(0o755)

    live_argv = _winner_argv("live", winner)
    live_sh = deploy / "live.sh"
    live_sh.write_text(
        "#!/usr/bin/env bash\n"
        f"# Generated by `tradingbot pipeline` — run {run_id}\n"
        "# ⚠ LIVE TRADING — REAL MONEY. Review before running.\n"
        "# Defaults apply: --max-order 500000 KRW, --daily-loss-limit 200000 KRW.\n"
        f"{shlex.join(live_argv)}\n"
    )
    live_sh.chmod(0o755)

    compose_argv = _winner_argv("paper", winner, state_file=DOCKER_STATE_FILE) + [
        "--balance",
        str(int(options.balance)),
    ]
    compose = deploy / "docker-compose.override.yml"
    compose.write_text(
        f"# Generated by `tradingbot pipeline` — run {run_id}\n"
        "# Drop next to docker-compose.yml and run: docker compose up -d\n"
        "# Switch to live by editing the command below (and review risk limits!).\n"
        "services:\n"
        f"  {COMPOSE_SERVICE}:\n"
        f"    command: {json.dumps(compose_argv)}\n"
    )
    return [paper_sh, live_sh, compose]


# ── summary.md ───────────────────────────────────────────────────────


def _write_summary_md(
    run_dir: Path,
    manifest: dict[str, Any],
    ranking: list[dict[str, Any]],
    artifacts: list[str],
) -> None:
    opts = manifest["options"]
    lines = [
        f"# Pipeline Run — {manifest['run_id']}",
        "",
        f"- created: {manifest['created_at']}",
        f"- stage-1 window: {'full range' if opts['include_train'] else 'auto holdout (last 20%)'}",
        f"- selection: top {opts['top']} by {opts['sort_by']} (min_trades {opts['min_trades']})",
        f"- walk-forward: train {opts['train_months']}m / test {opts['test_months']}m",
        f"- ranking: {opts['rank_by']}",
        "",
        "## Final Ranking (walk-forward OOS)",
        "",
        "| # | Candidate | Symbol | TF | Scan Sharpe (holdout) | OOS Sharpe | WF Eff "
        "| OOS Cum Return | OOS Trades |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in ranking:
        scan_sharpe = f"{r['scan_sharpe']:.2f}" if r["scan_sharpe"] is not None else "-"
        flag = " ⚠low-trades" if r["low_trades"] else ""
        lines.append(
            f"| {r['rank']} | {r['name']}{flag} | {r['symbol']} | {r['timeframe']} "
            f"| {scan_sharpe} | {r['oos_sharpe']:.2f} | {r['wf_efficiency']:.2f} "
            f"| {r['oos_cum_return']:.2%} | {r['oos_trades']} |"
        )
    if ranking and ranking[0]["oos_sharpe"] <= 0:
        lines += [
            "",
            "> ⚠ No candidate showed a positive out-of-sample edge — the deploy "
            "artifacts below are for inspection only, not a deploy signal.",
        ]
    if artifacts:
        lines += ["", "## Deploy artifacts (generated only — review, then run)", ""]
        lines += [f"- `{a}`" for a in artifacts]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
