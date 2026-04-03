# ML Performance Plan — 병렬 학습 구현

## Context

`ml-train-all`이 24개 모델을 순차 학습하여 시간이 오래 걸림. `--workers N` 옵션으로 ProcessPoolExecutor 기반 병렬 학습을 추가하여 학습 시간을 N분의 1로 줄임.

## Step 1: `src/tradingbot/ml/parallel.py` 생성

spawn 방식 ProcessPoolExecutor에서 pickle by reference로 호출할 최상위 워커 함수.

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PairTrainResult:
    """단일 symbol×timeframe 학습 결과."""
    symbol: str
    timeframe: str
    avg_auc: float
    avg_precision: float
    n_windows: int
    model_path: str
    error: str | None = None


def train_pair(
    symbol: str,
    timeframe: str,
    data_dir: str,
    model_dir: str,
    train_months: int,
    test_months: int,
    num_threads: int,
) -> PairTrainResult:
    """spawn-safe 워커. import는 함수 내부에서 수행."""
    from tradingbot.data.storage import load_candles
    from tradingbot.ml.walk_forward import MLWalkForwardTrainer

    try:
        df = load_candles(symbol, timeframe, Path(data_dir))
    except FileNotFoundError:
        return PairTrainResult(
            symbol=symbol, timeframe=timeframe,
            avg_auc=0.0, avg_precision=0.0, n_windows=0,
            model_path="", error="no data",
        )

    try:
        trainer = MLWalkForwardTrainer(
            symbol=symbol, timeframe=timeframe,
            train_months=train_months, test_months=test_months,
            model_dir=Path(model_dir),
            lgbm_params={"num_threads": num_threads},
        )
        report = trainer.run(df)
        return PairTrainResult(
            symbol=symbol, timeframe=timeframe,
            avg_auc=report.avg_auc, avg_precision=report.avg_precision,
            n_windows=len(report.windows),
            model_path=str(report.model_path) if report.model_path else "",
        )
    except Exception as exc:
        return PairTrainResult(
            symbol=symbol, timeframe=timeframe,
            avg_auc=0.0, avg_precision=0.0, n_windows=0,
            model_path="", error=str(exc),
        )
```

## Step 2: `src/tradingbot/cli.py` — `ml-train-all` 수정

### 2-1. `--workers` 옵션 추가

```python
workers: int = typer.Option(
    0, "--workers", "-w",
    help="Parallel workers (0=auto: cpu_count//2, 1=sequential)",
),
```

### 2-2. 워커 수 계산 + 출력

```python
import multiprocessing as mp

cpu_count = mp.cpu_count()
if workers <= 0:
    workers = max(1, min(cpu_count // 2, len(pairs)))
workers = min(workers, len(pairs))
threads_per_worker = max(1, cpu_count // workers)

console.print(f"  Workers: {workers}  (threads/worker: {threads_per_worker})\n")
```

### 2-3. workers == 1 → 기존 순차 코드 유지

기존 for loop 그대로. `lgbm_params={"num_threads": threads_per_worker}` 만 추가.

### 2-4. workers >= 2 → 병렬 실행

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
from tradingbot.ml.parallel import train_pair, PairTrainResult

ctx = mp.get_context("spawn")

with _progress_context() as progress:
    task = progress.add_task("Training models", total=len(pairs))

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        futures = {
            executor.submit(
                train_pair, sym, tf, data_dir, model_dir,
                train_months, test_months, threads_per_worker,
            ): (sym, tf)
            for sym, tf in pairs
        }

        for future in as_completed(futures):
            sym, tf = futures[future]
            r = future.result()  # PairTrainResult (예외는 내부에서 처리됨)

            if r.error:
                color = "red"
                msg = f"{sym} {tf}: {r.error}"
            elif r.n_windows == 0:
                color = "yellow"
                msg = f"{sym} {tf}: insufficient data"
            else:
                color = "green"
                msg = f"{sym} {tf}: AUC={r.avg_auc:.4f} precision={r.avg_precision:.4f} windows={r.n_windows}"
                results.append({...})  # 기존과 동일

            progress.log(f"[{color}]{msg}[/{color}]")
            progress.advance(task)
```

### 2-5. Summary table — 변경 없음

기존 코드 그대로 유지.

## Step 3: `src/tradingbot/ml/walk_forward.py` — `lgbm_params` 전달

`MLWalkForwardTrainer.__init__`에 `lgbm_params` 파라미터 추가하여 `LGBMTrainer`에 전달.

```python
def __init__(self, ..., lgbm_params: dict | None = None):
    ...
    self.trainer = LGBMTrainer(lgbm_params)
```

현재 이미 `lgbm_params` 파라미터가 있으므로 변경 불필요할 수 있음 → 확인 필요.

## 변경 파일 요약

| 파일 | 작업 |
|------|------|
| `src/tradingbot/ml/parallel.py` | **새로 생성** — train_pair 워커 + PairTrainResult |
| `src/tradingbot/cli.py` | `--workers` 옵션, 병렬/순차 분기 |
| `src/tradingbot/ml/walk_forward.py` | lgbm_params 전달 확인 (이미 있으면 변경 없음) |

## 변경하지 않는 파일

- `ml/trainer.py` — 이미 params 오버라이드 지원
- `ml/features.py`, `ml/targets.py` — 독립적, 변경 불필요
- `backtest/` — ML 학습과 무관

## 검증

```bash
# 1. 순차 모드 (기존 동작 동일 확인)
tradingbot ml-train-all --workers 1

# 2. 병렬 2워커
tradingbot ml-train-all --workers 2

# 3. 자동 워커 수
tradingbot ml-train-all

# 4. 타임프레임 필터 + 병렬
tradingbot ml-train-all --timeframe 1h --workers 4

# 5. 기존 테스트 통과
pytest tests/ -v
```

체크리스트:
- [ ] 하나의 페어 실패 시 나머지 계속 실행
- [ ] Progress bar에 완료 즉시 표시
- [ ] 모델 파일 충돌 없음 (각 페어별 고유 경로)
- [ ] workers=1일 때 기존과 동일한 동작
- [ ] spawn 컨텍스트 명시
