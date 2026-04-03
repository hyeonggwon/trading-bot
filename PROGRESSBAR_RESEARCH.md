# Progress Bar 리서치

## 대상 CLI 명령 및 루프

### 1. `combine-scan` — 가장 긴 작업 [HIGH]

**위치**: cli.py line 769 (함수 정의), line 805-865 (루프)

**구조**: 3중 루프 (심볼 → 타임프레임 → 템플릿). 루프 자체는 단순 조합 순회이고, 병목은 각 iteration의 BacktestEngine.run() 호출.
```python
for sym, timeframes in symbol_timeframes.items():
    for tf in timeframes:
        for tmpl in COMBINE_TEMPLATES:
            count += 1
            if count % 10 == 0:
                console.print(f"  Progress: {count}/{total}...", end="\r")
```

**총 개수**: 사전 계산됨 (line 801)
```python
total = len(COMBINE_TEMPLATES) * sum(len(tfs) for tfs in symbol_timeframes.values())
```

**표시 가능 정보**: 템플릿 이름(`tmpl["label"]`), 심볼, 타임프레임, count/total

**현재 표시**: `console.print(f"  Progress: {count}/{total}...", end="\r")` (10건마다)

---

### 2. `scan` — 전략 스캔 [HIGH]

**위치**: cli.py line 526 (함수 정의), line 576-627 (루프)

**구조**: 3중 루프 (심볼 → 타임프레임 → 전략). 동일하게 병목은 각 iteration의 BacktestEngine.run().
```python
for sym, timeframes in symbol_timeframes.items():
    for tf in timeframes:
        for strat_name in strategies:
            count += 1
            if count % 10 == 0:
                console.print(f"  Progress: {count}/{total}...", end="\r")
```

**총 개수**: 사전 계산됨 (line 571-572)
```python
total = sum(len(tfs) for tfs in symbol_timeframes.values()) * len(strategies)
```

**표시 가능 정보**: 전략명, 심볼, 타임프레임, count/total

**현재 표시**: `console.print(f"  Progress: {count}/{total}...", end="\r")` (10건마다)

---

### 3. `ml-train-all` — ML 학습 [MEDIUM]

**위치**: cli.py line 1008 (함수 정의), line 1047-1089 (루프)

**구조**: 단일 루프 (심볼 × 타임프레임 pairs)
```python
for i, (sym, tf) in enumerate(pairs, 1):
    console.print(f"[{i}/{len(pairs)}] {sym} {tf}...", end=" ")
```

**총 개수**: `len(pairs)` 사전 계산

**표시 가능 정보**: 심볼, 타임프레임, AUC, precision (학습 후)

**현재 표시**: `[i/total] SYM TF... AUC=X precision=Y` (인라인)

---

### 4. `optimize` — 그리드 서치 [HIGH]

**위치**: backtest/optimizer.py line 92 (함수 정의)

**순차 경로** (line 120-126): `max_workers == 1 or total <= 4`
```python
for i, params in enumerate(combinations):
    result = _run_single_backtest(...)
    if (i + 1) % 10 == 0 or i + 1 == total:
        logger.debug("optimization_progress", completed=i + 1, total=total)
```

**병렬 경로** (line 137-141): `max_workers > 1 and total > 4`
```python
for i, future in enumerate(as_completed(futures)):
    result = future.result()
    if (i + 1) % 10 == 0 or i + 1 == total:
        logger.debug("optimization_progress", completed=i + 1, total=total)
```

**총 개수**: `total = len(combinations)` (line 112)

**표시 가능 정보**: 파라미터 값, 완료 수/전체, sharpe 등 결과

**현재 표시**: logger.debug만 (사용자 미표시)

---

### 5. `walk-forward` — Walk-Forward 검증 [HIGH]

**위치**: backtest/walk_forward.py line 208 (함수 정의), line 247-292 (루프)

```python
for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
    logger.info("walk_forward_window", window=i + 1, ...)
    opt_results = optimizer.optimize(...)  # 내부에 또 루프
    test_result = _run_test(...)
```

**총 개수**: `len(windows)` 사전 계산

**표시 가능 정보**: 윈도우 번호, train/test 기간, best params, sharpe

**현재 표시**: logger.info만

**주의**: 내부에서 optimizer.optimize() 호출 → 이중 프로그레스바 가능

---

### 6. `backtest` — 데이터 로딩 [LOW]

**위치**: cli.py line 113, line 165-170

```python
for sym in symbols:
    data[sym] = load_candles(sym, timeframe, Path(data_dir))
```

**총 개수**: `len(symbols)` 사전 계산

**표시 가능 정보**: 심볼명, 캔들 수

**현재 표시**: `console.print(f"  {sym}: {len(data[sym])} candles")`

**판단**: 보통 빠름 (심볼 수가 적음). 프로그레스바 불필요할 수 있음.

---

## Rich Progress 사용 방법

### 기본 패턴
```python
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

with Progress(
    SpinnerColumn(),
    TextColumn("[bold blue]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TextColumn("{task.completed}/{task.total}"),
    TimeRemainingColumn(),
) as progress:
    task = progress.add_task("Scanning...", total=total)
    for item in items:
        progress.update(task, description=f"Scanning {item.name}...")
        do_work(item)
        progress.advance(task)
```

### 주의사항

1. **console.print와 충돌**: Progress 컨텍스트 내에서 `console.print()`를 쓰면 프로그레스바가 깨짐. `progress.console.print()` 또는 `progress.log()` 사용.

2. **로그 출력 억제**: 백테스트 엔진이 `structlog`으로 로그를 출력하면 프로그레스바가 깨짐. 프로그레스바 사용 시 로그 레벨을 WARNING 이상으로 올려야 함.

3. **병렬 실행 (optimizer)**: `ProcessPoolExecutor`와 함께 쓸 때는 `as_completed` 결과를 `progress.advance()`와 연결.

4. **이중 프로그레스바**: walk-forward 내부에서 optimizer를 호출하면 이중 프로그레스바 가능. Rich Progress는 여러 task를 동시에 표시할 수 있음.
```python
with Progress(...) as progress:
    wf_task = progress.add_task("Walk-Forward", total=len(windows))
    for window in windows:
        opt_task = progress.add_task(f"  Optimizing W{i}", total=total_params)
        for params in combinations:
            ...
            progress.advance(opt_task)
        progress.remove_task(opt_task)
        progress.advance(wf_task)
```

5. **기존 Rich console 재사용**: cli.py에 이미 `console = Console()`이 있음. Progress에서 같은 console 인스턴스 사용해야 충돌 방지.
```python
with Progress(..., console=console) as progress:
```

## 구현 범위

| 명령 | 파일 | 현재 표시 | 변경 |
|------|------|-----------|------|
| `combine-scan` | cli.py | carriage return | Rich Progress |
| `scan` | cli.py | carriage return | Rich Progress |
| `ml-train-all` | cli.py | inline print | Rich Progress |
| `optimize` | optimizer.py | logger.debug | Rich Progress (CLI에서 전달) |
| `walk-forward` | walk_forward.py | logger.info | Rich Progress (CLI에서 전달) |
| `backtest` | cli.py | console.print | 그대로 유지 (빠름) |

## 로그 억제 전략

프로그레스바 사용 시 `structlog` 로그가 바를 깨뜨리므로:

**방안 A**: Progress 컨텍스트 진입 시 로그 레벨 올리기
```python
import logging
logging.getLogger("tradingbot").setLevel(logging.WARNING)
```

**방안 B**: structlog의 ConsoleRenderer를 비활성화하고 Progress.log()로 대체

**채택**: 방안 A — 가장 단순. 프로그레스바 완료 후 레벨 복원.

## 성능 영향

Rich Progress 오버헤드는 무시 가능. `progress.advance()`는 터미널 출력 갱신일 뿐이고 실제 작업(백테스트, ML 학습) 대비 마이크로초 수준. Rich가 터미널 갱신 빈도를 자동 조절(refresh_per_second)하므로 현재 `console.print(end="\r")`보다 오히려 효율적.
