"""Strategy selection pipeline: ML train → scan → validate → rank → deploy artifacts.

Automates the operator's recurring workflow as one run:

0. **ML train** (smart refresh): retrain lgbm models that are missing or whose
   training data went stale (``--retrain-all`` forces, ``--no-ml-train`` skips)
1. **Scan** registry strategies + combine templates + lgbm (reuses
   ``parallel._run_batch``); lgbm_prob templates compare here too
2. **Select** top-N candidates (min-trades gate). lgbm rows compete as
   kind="ml"; lgbm_prob templates stay scan-only — LgbmProbFilter has no
   per-window model-injection path yet, so a walk-forward would leak
3. **Validate** each candidate: registry → per-window param optimization,
   combined → fixed filters, ml → ``MLStrategyWalkForward`` (fresh model per
   window — same time-honest path as ``ml-walk-forward``)
4. **Rank** by an out-of-sample metric; every row carries a ``validation``
   provenance column (walk_forward / walk_forward_combined / ml_walk_forward)
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

_ML_DISABLED_REASON = (
    "ml-candidate: excluded by --no-ml (time-honest path: `tradingbot ml-walk-forward`)"
)
_ML_TEMPLATE_REASON = (
    "ml-filter template: scan-only — LgbmProbFilter has no per-window model "
    "injection yet, so a walk-forward here would leak future data"
)


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
    # ML integration (stage 0 training + lgbm candidates)
    ml: bool = True
    ml_train: bool = True
    retrain_all: bool = False
    ml_stale_days: int = 7
    ml_train_months: int = 3  # stage-0 training (mirrors ml-train-all defaults)
    ml_test_months: int = 1
    ml_wf_train_months: int = 6  # stage-3 validation (mirrors ml-walk-forward defaults)
    ml_wf_test_months: int = 2


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
    if options.skip_rules and options.skip_combine and not options.ml:
        raise PipelineError("--skip-rules and --skip-combine with --no-ml leave nothing to scan")

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

    # ── Stage 0: ML train (smart refresh) ────────────────────────────
    if options.ml and options.ml_train:
        t0 = time.monotonic()
        log("[0/6] ml-train: smart refresh (missing/stale models)")
        train_summary = _ml_train_stage(options, log)
        atomic_write_json(run_dir / "stage0_ml_train.json", train_summary)
        _stage_done(
            "ml_train",
            t0,
            trained=len(train_summary["trained"]),
            fresh=len(train_summary["fresh"]),
            failed=len(train_summary["failed"]),
        )

    # ── Stage 1: scan ────────────────────────────────────────────────
    t0 = time.monotonic()
    log(
        f"[1/6] scan: rules={not options.skip_rules} "
        f"combine={not options.skip_combine} ml={options.ml}"
    )
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
        scan_results,
        top=options.top,
        min_trades=options.min_trades,
        sort_by=options.sort_by,
        include_ml=options.ml,
    )
    excluded = skipped_ml + excluded
    log(f"[2/6] select: {len(selected)} candidates (excluded {len(excluded)})")
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
    log(f"[4/6] rank: winner = {winner['name'] if winner else 'none'} (by {options.rank_by})")
    atomic_write_json(run_dir / "ranking.json", {"rank_by": options.rank_by, "ranking": ranking})
    _stage_done("rank", t0, candidates=len(ranking))

    # ── Stage 5: deploy artifacts ────────────────────────────────────
    t0 = time.monotonic()
    artifacts: list[str] = []
    if winner is not None:
        artifacts = [str(p) for p in write_deploy_artifacts(run_dir, winner, options)]
        log(f"[5/6] deploy artifacts: {len(artifacts)} files (generated only — review first)")
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


# ── Stage 0: ML train (smart refresh) ────────────────────────────────


def _needs_training(
    meta: dict[str, Any] | None,
    *,
    last_candle: Any,
    stale_days: int,
    retrain_all: bool,
) -> str | None:
    """Reason to (re)train this pair's model, or None when it is fresh.

    Fail-safe direction: an unreadable/incomplete meta counts as stale.
    """
    import pandas as pd

    if retrain_all:
        return "retrain-all"
    if meta is None:
        return "no model"
    data_end = meta.get("data_end")
    if not data_end:
        return "meta missing data_end"
    try:
        trained_end = pd.Timestamp(data_end)
    except (ValueError, TypeError):
        return f"unparseable data_end {data_end!r}"
    if trained_end.tzinfo is None:
        trained_end = trained_end.tz_localize("UTC")
    last = pd.Timestamp(last_candle)
    if pd.isna(last):
        return "unreadable data end"
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    age = last - trained_end
    if age > pd.Timedelta(days=stale_days):
        return f"stale: {age.days}d of new data (> {stale_days}d)"
    return None


def _ml_train_stage(
    options: PipelineOptions,
    log: Callable[[str], None],
) -> dict[str, list[dict[str, Any]]]:
    """Retrain models that are missing or stale (smart refresh).

    Reuses the ``ml-train-all`` worker (``ml/parallel.py:train_pair``) and
    its spawn-pool pattern; fresh models are skipped so a routine run only
    pays for pairs with meaningful new data.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from tradingbot.data.storage import EXTERNAL_SUBDIR, list_available_data
    from tradingbot.ml.parallel import train_pair
    from tradingbot.ml.trainer import LGBMTrainer

    available = list_available_data(Path(options.data_dir))
    if not available:
        raise PipelineError(f"no data in {options.data_dir!r} — run `tradingbot download` first")

    summary: dict[str, list[dict[str, Any]]] = {"trained": [], "fresh": [], "failed": []}
    to_train: list[tuple[str, str]] = []
    for item in available:
        symbol, tf = item["symbol"], item["timeframe"]
        # list_available_data already read each parquet — reuse its "end"
        # instead of a second full read; an empty file yields rows="0"/"NaT".
        if item["rows"] == "0" or item["end"] in ("NaT", ""):
            summary["failed"].append(
                {"symbol": symbol, "timeframe": tf, "reason": "empty data file"}
            )
            continue
        meta = LGBMTrainer.load_meta(symbol, tf, Path("models"))
        reason = _needs_training(
            meta,
            last_candle=item["end"],
            stale_days=options.ml_stale_days,
            retrain_all=options.retrain_all,
        )
        if reason is None:
            summary["fresh"].append({"symbol": symbol, "timeframe": tf})
        else:
            to_train.append((symbol, tf))
            log(f"      train {symbol} {tf} ({reason})")
    if not to_train:
        log("      all models fresh — nothing to train")
        return summary

    ext_dir = Path(options.data_dir) / EXTERNAL_SUBDIR
    external = str(ext_dir) if ext_dir.exists() and any(ext_dir.iterdir()) else None
    cpu = multiprocessing.cpu_count() or 4
    n_workers = options.workers if options.workers > 0 else max(1, cpu // 2)
    n_workers = max(1, min(n_workers, len(to_train)))  # never more workers than pairs
    num_threads = max(1, cpu // n_workers)
    with ProcessPoolExecutor(
        max_workers=n_workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {
            pool.submit(
                train_pair,
                sym,
                tf,
                str(Path(options.data_dir).resolve()),
                str(Path("models").resolve()),
                options.ml_train_months,
                options.ml_test_months,
                num_threads,
                external,
            ): (sym, tf)
            for sym, tf in to_train
        }
        for future in as_completed(futures):
            sym, tf = futures[future]
            try:
                result = future.result(timeout=3600)
            except Exception as exc:
                summary["failed"].append({"symbol": sym, "timeframe": tf, "reason": str(exc)})
                continue
            if result.error:
                summary["failed"].append({"symbol": sym, "timeframe": tf, "reason": result.error})
            else:
                summary["trained"].append(
                    {"symbol": sym, "timeframe": tf, "holdout_auc": result.holdout_auc}
                )
            log(f"      trained {sym} {tf}")
    return summary


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

        jobs.extend((name, "", "") for name in get_strategy_map() if name != "lgbm")
    # lgbm is ML, not "rules" — included whenever ML is on, independent of
    # --skip-rules. symbol/timeframe "*": decided before any per-batch scan.
    if options.ml:
        jobs.append(("lgbm", "", ""))
    else:
        skipped_ml.append(
            {"name": "lgbm", "symbol": "*", "timeframe": "*", "reason": _ML_DISABLED_REASON}
        )
    if not options.skip_combine:
        for tmpl in templates:
            if "lgbm_prob" in tmpl["entry"] and not options.ml:
                skipped_ml.append(
                    {
                        "name": tmpl["label"],
                        "symbol": "*",
                        "timeframe": "*",
                        "reason": _ML_DISABLED_REASON,
                    }
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
    include_ml: bool = True,
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    """Top-N candidates by ``sort_by`` after the min-trades / ML gates.

    lgbm rows compete as kind="ml" when ``include_ml``; lgbm_prob templates
    are always scan-only (no time-honest walk-forward path for the filter).
    Returns (selected, excluded-with-reason). Results below top-N are
    simply not selected — only gate failures land in ``excluded``.
    """
    excluded: list[dict[str, Any]] = []
    eligible = []
    for r in scan_results:
        ident = {"name": r.strategy, "symbol": r.symbol, "timeframe": r.timeframe}
        if "lgbm_prob" in r.entry:
            excluded.append(
                {**ident, "reason": _ML_TEMPLATE_REASON if include_ml else _ML_DISABLED_REASON}
            )
        elif r.strategy == "lgbm" and not include_ml:
            excluded.append({**ident, "reason": _ML_DISABLED_REASON})
        elif r.total_trades < min_trades:
            excluded.append(
                {**ident, "reason": f"trades {r.total_trades} < min_trades {min_trades}"}
            )
        else:
            eligible.append(r)

    reverse = sort_by != "max_drawdown"
    eligible.sort(key=lambda r: getattr(r, sort_by), reverse=reverse)

    def _kind(r: Any) -> str:
        if r.strategy == "lgbm":
            return "ml"
        return "combined" if r.entry else "strategy"

    selected = [
        Candidate(
            name=r.strategy,
            symbol=r.symbol,
            timeframe=r.timeframe,
            kind=_kind(r),
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
        log(
            f"[3/6] validate {i}/{len(candidates)}: {cand.name} {cand.symbol} "
            f"{cand.timeframe} ({cand.kind})"
        )
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
        if cand.kind == "ml":
            out.append(
                {
                    "candidate": asdict(cand),
                    "validation": "ml_walk_forward",
                    **_run_ml_walk_forward(cand, df, config, options),
                }
            )
            continue
        if cand.kind == "strategy":
            from tradingbot.strategy.registry import get_strategy_map

            validator = WalkForwardValidator(
                strategy_cls=get_strategy_map()[cand.name],
                config=config,
                train_months=options.train_months,
                test_months=options.test_months,
            )
            report = validator.validate({cand.symbol: df})
            validation = "walk_forward"
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
            validation = "walk_forward_combined"
        out.append(
            {"candidate": asdict(cand), "validation": validation, **serialize_wf_report(report)}
        )
    return out


def _run_ml_walk_forward(
    cand: Candidate,
    df: Any,
    config: Any,
    options: PipelineOptions,
) -> dict[str, Any]:
    """Time-honest ML validation: fresh model per window (ml-walk-forward path).

    Target/threshold settings are inherited from the saved model's meta so
    the validated configuration matches what stage-0 training (and the
    threshold tuner) produced; sensible CLI defaults apply when absent.
    """
    from tradingbot.data.storage import EXTERNAL_SUBDIR
    from tradingbot.ml.strategy_walk_forward import MLStrategyWalkForward
    from tradingbot.ml.trainer import LGBMTrainer

    # Hyperparameters (meta["tuning"]["best_params"]) are intentionally NOT
    # inherited: the WF validates the methodology with default params, not
    # tuned weights — reusing tuned params per window is its own honesty
    # question (deferred with the ml-tune pipeline integration).
    meta = LGBMTrainer.load_meta(cand.symbol, cand.timeframe, Path("models")) or {}
    ext_dir = Path(options.data_dir) / EXTERNAL_SUBDIR
    has_external = ext_dir.exists() and any(ext_dir.iterdir())
    runner = MLStrategyWalkForward(
        cand.symbol,
        cand.timeframe,
        train_months=options.ml_wf_train_months,
        test_months=options.ml_wf_test_months,
        forward_candles=int(meta.get("forward_candles", 4)),
        threshold=float(meta.get("threshold", 0.006)),
        target_kind=str(meta.get("target_kind", "binary")),
        atr_mult=float(meta.get("atr_mult", 1.0)),
        include_extra=bool(meta.get("include_extra", False)),
        entry_threshold=float(meta.get("entry_threshold", 0.45)),
        exit_threshold=float(meta.get("exit_threshold", 0.30)),
        external_data_dir=ext_dir if has_external else None,
        config=config,
    )
    report = runner.run(df)
    return serialize_ml_wf_report(report)


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


def serialize_ml_wf_report(report: Any) -> dict[str, Any]:
    """MLStrategyWalkForwardReport → the common summary/window schema.

    MLStrategyWalkForward reports returns in PERCENT (``return_pct``,
    ``cumulative_return_pct``) while WalkForwardReport uses fractions —
    normalized here so mixed-kind ranking compares like with like.
    ML walk-forward has no train-side metrics, so walk_forward_efficiency /
    overfitting_ratio are None (rendered "-", demoted when ranking by them).
    """
    return {
        "summary": {
            "num_windows": report.n_windows,
            "n_skipped": report.n_skipped,
            "avg_train_sharpe": None,
            "avg_test_sharpe": report.avg_sharpe,
            "avg_test_return": None,
            "walk_forward_efficiency": None,
            "overfitting_ratio": None,
            "cumulative_test_return": report.cumulative_return_pct / 100.0,
            "total_test_trades": report.total_trades,
        },
        "windows": [
            {
                "window_index": int(w["window"]),
                "train_start": str(w["train_start"]),
                "train_end": str(w["train_end"]),
                "test_start": str(w["test_start"]),
                "test_end": str(w["test_end"]),
                "best_params": {"model": "fresh-per-window"},
                "train_sharpe": None,
                "train_return": None,
                "test_sharpe": w["sharpe"],
                "test_return": w["return_pct"] / 100.0,
                "test_trades": w["trades"],
                "test_max_drawdown": w["max_dd_pct"] / 100.0,
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
                "validation": wf.get("validation", "walk_forward"),
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

    def _sort_key(r: dict[str, Any]) -> tuple[bool, bool, float, float]:
        # None metrics (ML rows have no walk_forward_efficiency; rank_value is
        # None when ranking by it) are demoted like low_trades — never crash.
        rank_value = r["rank_value"]
        efficiency = r["wf_efficiency"]
        return (
            bool(r["low_trades"]),
            rank_value is None,
            -(rank_value if rank_value is not None else 0.0),
            -(efficiency if efficiency is not None else float("-inf")),
        )

    rows.sort(key=_sort_key)
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return rows


# ── Stage 5: deploy artifacts ────────────────────────────────────────


def _winner_argv(
    command: str, winner: dict[str, Any], *, state_file: str | None = None
) -> list[str]:
    argv = ["tradingbot", command]
    if winner["kind"] in ("strategy", "ml"):
        # ml winner: --strategy lgbm — model + tuned thresholds load from
        # models/ at engine start (compose already mounts ./models).
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
    rank_value = winner["rank_value"]
    rank_note = f"rank_value={rank_value:.3f}" if rank_value is not None else "rank_value=-"
    model_note = ""
    if winner["kind"] == "ml":
        sym_key = _comment_safe(winner["symbol"]).replace("/", "_")
        model_note = (
            f"# Requires models/lgbm_{sym_key}_{_comment_safe(winner['timeframe'])}.lgb "
            "(+_meta.json) — thresholds load from meta.\n"
        )

    paper_argv = _winner_argv("paper", winner) + ["--balance", str(int(options.balance))]
    paper_sh = deploy / "paper.sh"
    paper_sh.write_text(
        "#!/usr/bin/env bash\n"
        f"# Generated by `tradingbot pipeline` — run {run_id}\n"
        f"# Winner: {winner_desc} ({rank_note})\n"
        f"{model_note}"
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
        f"{model_note}"
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
        f"{model_note}"
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
        "## Final Ranking (out-of-sample)",
        "",
        "| # | Candidate | Symbol | TF | Validation | Scan Sharpe (holdout) | OOS Sharpe "
        "| WF Eff | OOS Cum Return | OOS Trades |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in ranking:
        scan_sharpe = f"{r['scan_sharpe']:.2f}" if r["scan_sharpe"] is not None else "-"
        efficiency = f"{r['wf_efficiency']:.2f}" if r["wf_efficiency"] is not None else "-"
        flag = " ⚠low-trades" if r["low_trades"] else ""
        lines.append(
            f"| {r['rank']} | {r['name']}{flag} | {r['symbol']} | {r['timeframe']} "
            f"| {r.get('validation', '-')} | {scan_sharpe} | {r['oos_sharpe']:.2f} "
            f"| {efficiency} | {r['oos_cum_return']:.2%} | {r['oos_trades']} |"
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
