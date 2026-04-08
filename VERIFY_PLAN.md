# --verify-top 구현 계획

## 변경 파일

| 파일 | 변경 | 규모 |
|------|------|------|
| `src/tradingbot/backtest/parallel.py` | `_run_batch()`에 `force_engine` 파라미터 추가 | ~5줄 |
| `src/tradingbot/cli.py` | `--verify-top` 옵션 + Phase 2 재검증 로직 | ~60줄 |
| `tests/test_vectorized.py` | `force_engine=True` 라우팅 테스트 | ~15줄 |

---

## Step 1: `_run_batch()`에 `force_engine` 파라미터 추가

### 파일: `src/tradingbot/backtest/parallel.py`

```python
def _run_batch(
    symbol: str,
    timeframe: str,
    jobs: list[tuple[str, str, str]],
    data_dir: str,
    balance: float,
    config_dir: str = "config",
    force_engine: bool = False,      # ← 추가
) -> list[ScanResult]:
```

`force_engine=True`이면 vectorized/fallback 분기를 건너뛰고 모든 job을 `_run_engine_batch()`로 전달:

```python
    if force_engine:
        # 재검증: 모든 job을 풀 엔진으로 강제
        results.extend(_run_engine_batch(df, symbol, timeframe, jobs, config, balance))
    else:
        # 기존 로직: vectorizable vs fallback 분기
        vectorizable_jobs = [...]
        fallback_jobs = [...]
        if vectorizable_jobs:
            results.extend(_run_vectorized_batch(...))
        if fallback_jobs:
            results.extend(_run_engine_batch(...))
```

---

## Step 2: CLI `--verify-top` 옵션 추가

### 파일: `src/tradingbot/cli.py`

### 2-1. 파라미터 추가

```python
@app.command(name="combine-scan")
def combine_scan(
    top_n: int = typer.Option(10, "--top", help="Show top N results"),
    verify_top: int = typer.Option(0, "--verify-top", help="Re-verify top N with full engine"),
    ...
) -> None:
```

### 2-2. Phase 2 재검증 로직

기존 Phase 1 완료 후 (results 수집, Sharpe 정렬 이후, 테이블 출력 이전):

```python
    results.sort(key=lambda r: r["sharpe_ratio"], reverse=True)

    # Phase 2: 재검증
    verified_set: set[tuple[str, str, str]] = set()  # (template, symbol, timeframe)
    if verify_top > 0 and results:
        n_verify = min(verify_top, len(results))
        to_verify = results[:n_verify]

        # 이미 풀 엔진으로 돌린 ML 템플릿은 스킵
        verify_jobs: list[dict] = [r for r in to_verify if "lgbm_prob" not in r["entry"]]

        if verify_jobs:
            # (sym, tf)별 그룹핑
            verify_batches: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
            for r in verify_jobs:
                key = (r["symbol"], r["timeframe"])
                verify_batches.setdefault(key, []).append((r["template"], r["entry"], r["exit"]))

            console.print(f"\n[bold]Verifying top {len(verify_jobs)} with full engine...[/bold]")

            verified_results: dict[tuple[str, str, str], dict] = {}
            with _progress_context() as progress:
                task = progress.add_task("Verifying", total=len(verify_jobs))
                with ProcessPoolExecutor(...) as pool:
                    futures = {
                        pool.submit(_run_batch, sym, tf, batch_jobs, abs_data_dir, balance, abs_config_dir, True): (sym, tf)
                        for (sym, tf), batch_jobs in verify_batches.items()
                    }
                    for future in as_completed(futures):
                        batch_results = future.result(timeout=1800)
                        for r in batch_results:
                            if not r.error:
                                verified_results[(r.strategy, r.symbol, r.timeframe)] = {
                                    "sharpe_ratio": r.sharpe_ratio,
                                    "total_return": r.total_return,
                                    "max_drawdown": r.max_drawdown,
                                    "win_rate": r.win_rate,
                                    "profit_factor": r.profit_factor,
                                    "total_trades": r.total_trades,
                                }
                                verified_set.add((r.strategy, r.symbol, r.timeframe))
                        progress.advance(task, advance=len(batch_results))

            # 결과 교체
            for r in results:
                key = (r["template"], r["symbol"], r["timeframe"])
                if key in verified_results:
                    r.update(verified_results[key])

            # 재정렬
            results.sort(key=lambda r: r["sharpe_ratio"], reverse=True)

        # ML 템플릿도 verified로 마크 (이미 풀 엔진 경유)
        for r in to_verify:
            if "lgbm_prob" in r["entry"]:
                verified_set.add((r["template"], r["symbol"], r["timeframe"]))
```

### 2-3. 테이블에 V 컬럼 추가

`--verify-top > 0`일 때만 V 컬럼 표시:

```python
    if verify_top > 0:
        table.add_column("V", justify="center")  # Verified 마크

    for i, r in enumerate(results[:top_n], 1):
        row = [str(i), r["template"], r["symbol"], r["timeframe"], ...]
        if verify_top > 0:
            key = (r["template"], r["symbol"], r["timeframe"])
            row.append("[green]✓[/green]" if key in verified_set else "")
        table.add_row(*row)
```

---

## Step 3: 테스트 추가

### 파일: `tests/test_vectorized.py`

`TestRunBatchRouting` 클래스에 테스트 추가:

```python
def test_force_engine_routes_all_to_engine(self):
    """force_engine=True면 vectorizable job도 풀 엔진으로 가야 함."""
    # rule-only job (normally vectorized)
    jobs = [("Trend+RSI", "trend_up:4 + rsi_oversold:30", "rsi_overbought:70")]
    # force_engine=True로 호출
    results = _run_batch(sym, tf, jobs, data_dir, balance, config_dir, force_engine=True)
    # 결과가 ScanResult이고 에러 없음 확인
    assert len(results) == 1
    assert results[0].error is None
    assert results[0].total_trades >= 0
```

---

## 체크리스트

- [ ] Step 1: `parallel.py` — `force_engine` 파라미터 추가
- [ ] Step 2: `cli.py` — `--verify-top` 옵션 + Phase 2 로직 + V 컬럼
- [ ] Step 3: `tests/test_vectorized.py` — force_engine 테스트
- [ ] pytest 통과 확인
- [ ] `tradingbot combine-scan --verify-top 5` 실행 테스트
