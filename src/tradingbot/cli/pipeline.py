"""Pipeline command: scan → select → walk-forward → rank → deploy artifacts."""

from __future__ import annotations

import typer
from rich.table import Table

from tradingbot.cli._shared import app, console
from tradingbot.utils.logging import setup_logging


@app.command()
def pipeline(
    top: int = typer.Option(5, "--top", help="Walk-forward candidate count"),
    min_trades: int = typer.Option(
        10, "--min-trades", help="Minimum trade count gate (scan selection + OOS ranking)"
    ),
    sort_by: str = typer.Option("sharpe_ratio", "--sort-by", help="Stage-1 selection metric"),
    rank_by: str = typer.Option(
        "avg_test_sharpe",
        "--rank-by",
        help=(
            "Final ranking metric: avg_test_sharpe | cumulative_test_return | "
            "walk_forward_efficiency"
        ),
    ),
    train_months: int = typer.Option(
        3, "--train-months", help="Walk-forward training window (months)"
    ),
    test_months: int = typer.Option(1, "--test-months", help="Walk-forward test window (months)"),
    workers: int = typer.Option(0, "--workers", "-w", help="Scan workers (0=auto)"),
    balance: float = typer.Option(1_000_000, "--balance", "-b", help="Initial balance (KRW)"),
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
    include_train: bool = typer.Option(
        False,
        "--include-train",
        help="Scan the full data range instead of the last-20% holdout",
    ),
    output_dir: str = typer.Option(
        "results/pipeline", "--output-dir", help="Root directory for run outputs"
    ),
    skip_rules: bool = typer.Option(False, "--skip-rules", help="Skip registry-strategy scan"),
    skip_combine: bool = typer.Option(False, "--skip-combine", help="Skip combine-template scan"),
    ml: bool = typer.Option(
        True, "--ml/--no-ml", help="Include ML: stage-0 training + lgbm candidates"
    ),
    ml_train: bool = typer.Option(
        True,
        "--ml-train/--no-ml-train",
        help="Stage-0 smart model refresh (skip = use existing models only)",
    ),
    retrain_all: bool = typer.Option(
        False, "--retrain-all", help="Force retraining every (symbol, timeframe) model"
    ),
    ml_stale_days: int = typer.Option(
        7,
        "--ml-stale-days",
        help="Retrain when this many days of new candles accumulated since the model's data_end",
    ),
    ml_train_months: int = typer.Option(
        3, "--ml-train-months", help="Stage-0 training window (months, mirrors ml-train-all)"
    ),
    ml_test_months: int = typer.Option(
        1, "--ml-test-months", help="Stage-0 test window (months, mirrors ml-train-all)"
    ),
    ml_wf_train_months: int = typer.Option(
        6,
        "--ml-wf-train-months",
        help="ML validation training window (months, mirrors ml-walk-forward)",
    ),
    ml_wf_test_months: int = typer.Option(
        2,
        "--ml-wf-test-months",
        help="ML validation test window (months, mirrors ml-walk-forward)",
    ),
) -> None:
    """Run the full selection pipeline: ML train → scan → validate → rank → deploy artifacts.

    lgbm candidates are validated via the time-honest ml-walk-forward path
    (fresh model per window); lgbm_prob templates compare in the scan stage
    only. Deploy artifacts (paper/live commands, docker-compose override)
    are GENERATED ONLY — nothing is executed. Review them under the run
    directory and start engines yourself.
    """
    setup_logging()

    from tradingbot.backtest.pipeline import PipelineError, PipelineOptions, run_pipeline
    from tradingbot.cli.combine import COMBINE_TEMPLATES

    options = PipelineOptions(
        top=top,
        min_trades=min_trades,
        sort_by=sort_by,
        rank_by=rank_by,
        train_months=train_months,
        test_months=test_months,
        workers=workers,
        balance=balance,
        data_dir=data_dir,
        include_train=include_train,
        output_dir=output_dir,
        skip_rules=skip_rules,
        skip_combine=skip_combine,
        ml=ml,
        ml_train=ml_train,
        retrain_all=retrain_all,
        ml_stale_days=ml_stale_days,
        ml_train_months=ml_train_months,
        ml_test_months=ml_test_months,
        ml_wf_train_months=ml_wf_train_months,
        ml_wf_test_months=ml_wf_test_months,
    )
    try:
        result = run_pipeline(
            options,
            templates=COMBINE_TEMPLATES,
            log=lambda m: console.print(m, markup=False),
        )
    except PipelineError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    ranking = result["ranking"]
    table = Table(title=f"Pipeline Ranking — run {result['run_id']} (by {rank_by})")
    table.add_column("#", justify="right")
    table.add_column("Candidate")
    table.add_column("Symbol")
    table.add_column("TF")
    table.add_column("Val")
    table.add_column("Scan Sharpe (holdout)", justify="right")
    table.add_column("OOS Sharpe", justify="right")
    table.add_column("WF Eff", justify="right")
    table.add_column("OOS CumRet", justify="right")
    table.add_column("OOS Trades", justify="right")
    val_short = {
        "walk_forward": "WF",
        "walk_forward_combined": "WF-C",
        "ml_walk_forward": "ML-WF",
        "ml_walk_forward_combined": "ML-WF-C",
    }
    for r in ranking:
        style = "green" if r["oos_sharpe"] > 0 else "red"
        name = f"{r['name']} ⚠" if r["low_trades"] else r["name"]
        table.add_row(
            str(r["rank"]),
            name,
            r["symbol"],
            r["timeframe"],
            val_short.get(r.get("validation", ""), "-"),
            f"{r['scan_sharpe']:.2f}" if r["scan_sharpe"] is not None else "-",
            f"[{style}]{r['oos_sharpe']:.2f}[/{style}]",
            f"{r['wf_efficiency']:.2f}" if r["wf_efficiency"] is not None else "-",
            f"{r['oos_cum_return']:.2%}",
            str(r["oos_trades"]),
        )
    console.print(table)
    if any(str(r.get("validation", "")).startswith("ml_walk_forward") for r in ranking):
        console.print(
            "[dim]ML-WF rows: scan infers with the tuned disk model, OOS trains fresh "
            "default-param models per window — some scan→OOS gap is expected there.[/dim]"
        )

    winner = result["winner"]
    if winner is not None:
        console.print(
            f"[bold green]Winner:[/bold green] {winner['name']} on "
            f"{winner['symbol']} {winner['timeframe']}"
        )
        if winner["oos_sharpe"] <= 0:
            console.print(
                "[yellow]⚠ No candidate showed a positive out-of-sample edge — "
                "treat the artifacts as inspection material, not a deploy signal.[/yellow]"
            )
        console.print("[bold]Deploy artifacts (generated only — review, then run):[/bold]")
        for path in result["artifacts"]:
            console.print(f"  {path}")
    console.print(f"Run directory: {result['run_dir']}")
