# Progress Bar 구현 계획

## 개요

5개 CLI 명령에 Rich Progress 바를 적용한다. 기존 `console.print(end="\r")` 해킹을 제거하고, 고정된 프로그레스바 + ETA + 현재 작업 표시로 대체.

## 공통 설계

### Progress 인스턴스

```python
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeRemainingColumn, MofNCompleteColumn,
)

progress = Progress(
    SpinnerColumn(),
    TextColumn("[bold blue]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeRemainingColumn(),
    console=console,  # 기존 cli.py의 Console 인스턴스 재사용
)
```

### 로그 억제

Progress 컨텍스트 내에서 structlog 콘솔 출력이 바를 깨뜨리므로:

```python
import logging

with progress:
    logging.getLogger("tradingbot").setLevel(logging.WARNING)
    task = progress.add_task(...)
    # ... 루프 ...

# Progress 종료 후 자동 복원 (컨텍스트 밖)
```

로그 레벨 복원은 필요 없음 — 프로그레스바 이후 결과 테이블 출력만 남으므로.

## 명령별 구현

### 1. `combine-scan` (cli.py line 805-848)

**Before:**
```python
for sym, timeframes in symbol_timeframes.items():
    for tf in timeframes:
        for tmpl in COMBINE_TEMPLATES:
            count += 1
            if count % 10 == 0:
                console.print(f"  Progress: {count}/{total}...", end="\r")
            # ... 백테스트 ...
console.print(f"  Completed {count}/{total} combinations.     ")
```

**After:**
```python
import logging

with Progress(
    SpinnerColumn(),
    TextColumn("[bold blue]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeRemainingColumn(),
    console=console,
) as progress:
    logging.getLogger("tradingbot").setLevel(logging.WARNING)
    task = progress.add_task("Scanning combinations", total=total)

    for sym, timeframes in symbol_timeframes.items():
        for tf in timeframes:
            for tmpl in COMBINE_TEMPLATES:
                progress.update(task, description=f"{tmpl['label']} · {sym} {tf}")
                # ... 백테스트 ...
                progress.advance(task)

# "Progress: X/Y" 라인 제거, "Completed" 라인 제거
```

**제거할 코드:**
- line 807-808: `if count % 10 == 0: console.print(...)`
- line 848: `console.print(f"  Completed {count}/{total} combinations.     ")`
- `count` 변수 불필요

---

### 2. `scan` (cli.py line 570-606)

**Before:**
```python
for sym, timeframes in symbol_timeframes.items():
    for tf in timeframes:
        for strat_name in strategies:
            count += 1
            if count % 10 == 0:
                console.print(f"  Progress: {count}/{total}...", end="\r")
            # ... 백테스트 ...
console.print(f"  Completed {count}/{total} combinations.     ")
```

**After:**
```python
with Progress(..., console=console) as progress:
    logging.getLogger("tradingbot").setLevel(logging.WARNING)
    task = progress.add_task("Scanning strategies", total=total)

    for sym, timeframes in symbol_timeframes.items():
        for tf in timeframes:
            for strat_name in strategies:
                progress.update(task, description=f"{strat_name} · {sym} {tf}")
                # ... 백테스트 ...
                progress.advance(task)
```

**제거할 코드:** 동일 패턴 — count, if count % 10, Completed 라인

---

### 3. `ml-train-all` (cli.py line 1061-1098)

**Before:**
```python
for i, (sym, tf) in enumerate(pairs, 1):
    console.print(f"[{i}/{len(pairs)}] {sym} {tf}...", end=" ")
    # ... 학습 ...
    console.print(f"[green]AUC={report.avg_auc:.4f} ...")
```

**After:**
```python
with Progress(..., console=console) as progress:
    logging.getLogger("tradingbot").setLevel(logging.WARNING)
    task = progress.add_task("Training models", total=len(pairs))

    for sym, tf in pairs:
        progress.update(task, description=f"Training {sym} {tf}")
        # ... 학습 ...
        if report.windows:
            progress.log(
                f"[green]{sym} {tf}: AUC={report.avg_auc:.4f} "
                f"precision={report.avg_precision:.4f}[/green]"
            )
        progress.advance(task)
```

**`progress.log()`**: 프로그레스바 위에 결과를 한 줄씩 출력. 바를 깨뜨리지 않음.

**제거할 코드:** `console.print(f"[{i}/{len(pairs)}] ...")`, `enumerate(pairs, 1)` → `for sym, tf in pairs`

---

### 4. `optimize` (optimizer.py line 118-141)

optimizer.py는 CLI가 아니라 라이브러리 코드. Progress 객체를 외부에서 전달받도록 설계.

**optimizer.py 변경:**

```python
def optimize(
    self,
    data: dict[str, pd.DataFrame],
    param_space: dict[str, list[Any]] | None = None,
    sort_by: str = "sharpe_ratio",
    progress=None,  # 추가: Rich Progress 객체 (optional)
) -> list[OptimizationResult]:
```

**순차 경로 (line 120-126):**
```python
if self.max_workers == 1 or total <= 4:
    task = progress.add_task("Optimizing", total=total) if progress else None
    for i, params in enumerate(combinations):
        result = _run_single_backtest(...)
        results.append(result)
        if progress and task is not None:
            progress.advance(task)
    if progress and task is not None:
        progress.remove_task(task)
```

**병렬 경로 (line 137-141):**
```python
else:
    task = progress.add_task("Optimizing (parallel)", total=total) if progress else None
    with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
        futures = {...}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            if progress and task is not None:
                progress.advance(task)
    if progress and task is not None:
        progress.remove_task(task)
```

**CLI에서 호출 시** (cli.py optimize 명령):
```python
with Progress(..., console=console) as progress:
    logging.getLogger("tradingbot").setLevel(logging.WARNING)
    results = optimizer.optimize(data, param_grid, sort_by=sort_by, progress=progress)
```

**기존 logger.debug 제거:** progress가 있으면 불필요.

---

### 5. `walk-forward` (walk_forward.py line 247-292)

walk_forward.py도 라이브러리 코드. Progress를 외부에서 전달.

**walk_forward.py 변경:**

```python
def validate(
    self,
    data: dict[str, pd.DataFrame],
    progress=None,  # 추가
) -> WalkForwardReport:
```

**루프 (line 247-292):**
```python
task = progress.add_task("Walk-Forward", total=len(windows)) if progress else None

for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
    if progress and task is not None:
        progress.update(
            task,
            description=f"WF {i+1}/{len(windows)}: train {train_start.date()}~{train_end.date()}"
        )

    # Step 1: Optimize
    optimizer = GridSearchOptimizer(...)
    opt_results = optimizer.optimize(train_data, param_space, progress=progress)

    # Step 2: Test
    test_result = _run_test(...)

    if progress and task is not None:
        progress.advance(task)

if progress and task is not None:
    progress.remove_task(task)
```

**이중 프로그레스바**: walk-forward가 내부에서 optimizer.optimize(progress=progress)를 호출하면, optimizer가 하위 task를 추가/제거. Rich Progress가 여러 task를 동시 표시:

```
⠋ WF 2/5: train 2024-01-01~2024-03-31    ████████████████████ 5/5  0:00:00
⠋ Optimizing                              ████████░░░░░░░░░░░ 45/100  0:00:12
```

**CLI에서 호출 시** (cli.py walk-forward 명령):
```python
with Progress(..., console=console) as progress:
    logging.getLogger("tradingbot").setLevel(logging.WARNING)
    report = validator.validate(data, progress=progress)
```

## 파일별 변경 요약

| 파일 | 변경 |
|------|------|
| `cli.py` | combine-scan, scan, ml-train-all에 Progress 적용. optimize, walk-forward 호출 시 progress 전달 |
| `backtest/optimizer.py` | optimize()에 `progress=None` 파라미터 추가, 기존 logger.debug 유지 (progress 없을 때 폴백) |
| `backtest/walk_forward.py` | validate()에 `progress=None` 파라미터 추가, 내부 optimizer에 progress 전달 |

## 구현 순서

1. **cli.py — combine-scan**: 가장 자주 사용, 패턴 확립
2. **cli.py — scan**: combine-scan과 거의 동일 패턴
3. **cli.py — ml-train-all**: progress.log() 패턴 (결과 인라인 출력)
4. **optimizer.py**: progress 파라미터 추가 (하위 호환)
5. **walk_forward.py**: progress 파라미터 추가 + 이중 프로그레스바
6. **cli.py — optimize, walk-forward 호출부**: progress 전달

## 검증 기준

1. 기존 159 테스트 전부 통과
2. `combine-scan --top 5` 실행 시 프로그레스바 정상 표시 + ETA
3. `scan --top 5` 실행 시 프로그레스바 정상 표시
4. `ml-train-all --timeframe 1h` 실행 시 프로그레스바 + 결과 로그
5. `optimize` 실행 시 프로그레스바 표시
6. `walk-forward` 실행 시 이중 프로그레스바 (WF + Optimize)
7. 프로그레스바가 structlog 로그에 의해 깨지지 않음
8. progress=None (기존 호출) 시 동작 변경 없음 (하위 호환)
