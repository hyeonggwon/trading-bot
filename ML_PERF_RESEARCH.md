# ML Performance Research — 병렬 학습

## 현재 상태

`ml-train-all`은 symbol×timeframe 페어를 **순차 실행**. 24개 모델(8심볼×3TF) 학습 시 전체 시간 = 각 모델 시간의 합.

### 현재 병렬화 패턴 (코드베이스)

| 컴포넌트 | 파일 | 패턴 | 상태 |
|----------|------|------|------|
| Backtest Optimizer | `backtest/optimizer.py` | ProcessPoolExecutor + as_completed | 정의됨 (CLI에서 max_workers=1) |
| Walk-Forward | `backtest/walk_forward.py` | 순차 (윈도우별) | optimizer에 max_workers=1 전달 |
| ML Walk-Forward | `ml/walk_forward.py` | 순차 (윈도우별) | 병렬화 없음 |
| ML Trainer | `ml/trainer.py` | LightGBM 내부 스레드 | `num_threads: -1` (전체 코어) |
| CLI ml-train-all | `cli.py:1065` | for loop 순차 | 병렬화 없음 |

### 메모리 사용량 (페어당)

- OHLCV: ~8,640행 × 6열 (1h, 12개월)
- 피처 매트릭스: 36열 추가 → 총 ~42열
- 추정 메모리: <100MB per job
- 동시 4개 실행 시: <400MB → 문제 없음

## 병렬화 설계

### 1. 실행기 선택: `ProcessPoolExecutor`

| 옵션 | 장점 | 단점 | 판정 |
|------|------|------|------|
| ProcessPoolExecutor | GIL 우회, 기존 패턴과 일치, as_completed로 Progress 연동 | pickle 직렬화 비용 | ✅ 채택 |
| ThreadPoolExecutor | 오버헤드 적음 | GIL 때문에 pandas/numpy 연산 병렬화 안 됨 | ❌ |
| multiprocessing.Pool | ProcessPoolExecutor와 동일 | Progress 연동 불편, 에러 처리 불편 | ❌ |

### 2. LightGBM 스레드 할당

**문제**: LightGBM `num_threads=-1`은 모든 코어 사용. N개 프로세스가 동시에 전체 코어 점유 → 스레드 경합.

**해결**: `threads_per_worker = max(1, cpu_count // workers)`

```
8코어 + 2워커 → 4스레드/워커 (총 8, 경합 없음)
8코어 + 4워커 → 2스레드/워커 (총 8, 경합 없음)
```

`MLWalkForwardTrainer`에 `lgbm_params={"num_threads": threads_per_worker}` 전달.
기존 `LGBMTrainer.__init__`이 `{**DEFAULT_LGBM_PARAMS, **(params or {})}` 로 머지하므로 내부 수정 불필요.

### 3. 기본 워커 수

```python
workers = max(1, min(cpu_count() // 2, len(pairs)))
```

- `cpu_count() // 2`: 인터랙티브 머신에서 여유 확보 (터미널, Progress 렌더링)
- `len(pairs)` 캡: 3개 페어에 8개 프로세스 생성 방지
- `--workers 1`: 명시적 순차 모드 (디버깅용)
- `--workers 0` 또는 미지정: 자동

### 4. fork vs spawn

**반드시 `spawn` 사용.** LightGBM은 OpenMP를 링크. `fork()` 후 OpenMP parallel region에서 데드락 가능.

```python
ctx = mp.get_context("spawn")
ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
```

macOS는 Python 3.8+에서 기본 spawn이지만, Linux 호환을 위해 명시.
spawn 비용: 프로세스당 ~2-3초 → 학습 30-300초 대비 무시 가능.

### 5. 워커 함수 제약

`spawn`은 함수를 pickle by reference. 따라서 워커 함수는:
- 모듈 최상위 레벨 함수여야 함 (lambda, closure, 중첩 함수 불가)
- `ml/parallel.py`에 독립 모듈로 분리

## 구현 계획

### 파일 변경

| 파일 | 작업 |
|------|------|
| `src/tradingbot/ml/parallel.py` | **새로 생성** — `train_pair()` 워커 함수 + `PairTrainResult` 데이터클래스 |
| `src/tradingbot/cli.py` | `ml-train-all`에 `--workers` 옵션 추가, 병렬/순차 분기 |

### 변경하지 않는 파일

- `ml/walk_forward.py` — 변경 없음. num_threads는 lgbm_params로 전달
- `ml/trainer.py` — 변경 없음. 이미 params 오버라이드 지원
- `ml/features.py`, `ml/targets.py` — 변경 없음

### 새 파일: `ml/parallel.py`

```python
@dataclass
class PairTrainResult:
    symbol: str
    timeframe: str
    avg_auc: float
    avg_precision: float
    n_windows: int
    model_path: str
    error: str | None = None

def train_pair(symbol, timeframe, data_dir, model_dir,
               train_months, test_months, num_threads) -> PairTrainResult:
    """spawn-safe 최상위 워커 함수. import는 함수 내부에서."""
    ...
```

### CLI 변경: `ml-train-all`

```
workers == 1 → 기존 순차 코드 (오버헤드 0)
workers >= 2 → ProcessPoolExecutor + as_completed + Progress
```

## 검증

```bash
# 순차 (기존 동작 동일)
tradingbot ml-train-all --workers 1

# 병렬 2워커
tradingbot ml-train-all --workers 2

# 자동 (cpu_count // 2)
tradingbot ml-train-all

# 타임프레임 필터 + 병렬
tradingbot ml-train-all --timeframe 1h --workers 4
```

- 하나의 페어 실패 시 나머지 계속 실행
- Progress bar에 완료된 페어 즉시 표시
- 기존 159개 테스트 통과