# LgbmProbFilter 구현 계획

## 개요

LightGBM 확률 출력을 BaseFilter로 감싸서 기존 30개 룰 기반 필터와 AND 조합 가능하게 한다.
추가로 `ml-train-all` CLI 명령으로 모든 심볼×타임프레임 모델을 일괄 학습한다.

## Phase 1: LgbmProbFilter 구현

### 1-1. `strategy/filters/ml.py` 신규 생성

```python
class LgbmProbFilter(BaseFilter):
    name = "lgbm_prob"
    role = "entry"  # ML 거부권 필터 (AND 조합에서 veto)

    def __init__(self, threshold=0.55, symbol="BTC/KRW", timeframe="1h", model_dir="models"):
        super().__init__(threshold=threshold)
        self.threshold = threshold
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_dir = Path(model_dir)
        self._model = None        # lazy load
        self._loaded = False       # 로드 시도 여부
        self.last_prob = None      # 마지막 예측 확률
        self.last_strength = None  # Half-Kelly 결과

    def compute(self, df):
        # build_feature_matrix() 호출 → 33개 피처 컬럼 추가
        # 인디케이터 중복 가드(if col not in df.columns)로 기존 필터와 충돌 없음
        df, _ = build_feature_matrix(df)
        return df

    def check_entry(self, df):
        # 1. lazy model load
        # 2. FEATURE_COLS의 마지막 행 NaN 체크
        # 3. model.predict() → prob
        # 4. prob >= threshold → True + last_strength = half_kelly(prob)
        # 5. prob < threshold → False
        ...

    def check_exit(self, df, entry_index=None):
        return False  # 진입 전용 필터
```

핵심:
- lightgbm import는 try/except로 감싸서 ML 미설치 환경에서도 import 에러 없이 동작
- 모델 없으면 check_entry() = False (보수적, 로그 warning 1회)
- `_half_kelly()` 함수는 lgbm_strategy.py에서 가져오거나 공유 유틸로 추출

### 1-2. `strategy/filters/registry.py` 수정

```python
# get_filter_map()에 추가
"lgbm_prob": LgbmProbFilter,

# _parse_filter_params()에 추가
elif name == "lgbm_prob":
    if len(parts) >= 2:
        kwargs["threshold"] = float(parts[1])
    if len(parts) >= 3:
        kwargs["model_dir"] = parts[2]
```

- symbol과 timeframe은 파싱 시점에 알 수 없음 → CLI에서 필터 생성 후 설정

### 1-3. `strategy/combined.py` 수정

should_entry()에서 Signal.strength 수집 로직 추가:

```python
def should_entry(self, df, symbol):
    ...
    strength = 1.0  # 기본값
    checked = 0
    for f in self.entry_filters:
        if f.role == "exit":
            continue
        checked += 1
        if not f.check_entry(df):
            return None
        # strength 수집
        if hasattr(f, 'last_strength') and f.last_strength is not None:
            strength = f.last_strength

    if checked == 0:
        return None

    return Signal(..., strength=strength)
```

- 기존 필터는 last_strength 속성이 없으므로 영향 없음 (하위 호환)
- LgbmProbFilter가 있으면 그 strength 사용, 없으면 1.0 유지

### 1-4. `cli.py` 수정 — combine/combine-scan에서 symbol/timeframe 전달

```python
# combine 명령
entry_filters = parse_filter_string(entry, base_timeframe=timeframe)
for f in entry_filters:
    if hasattr(f, 'symbol'):
        f.symbol = symbol
    if hasattr(f, 'timeframe'):
        f.timeframe = timeframe
# exit_filters도 동일
```

combine-scan에서도 루프 내에서 동일하게 설정:
```python
for f in entry_filters + exit_filters:
    if hasattr(f, 'symbol'):
        f.symbol = sym
    if hasattr(f, 'timeframe'):
        f.timeframe = tf
```

COMBINE_TEMPLATES에 lgbm_prob 포함 템플릿 3개 추가:
```python
{"entry": "trend_up:4 + rsi_oversold:30 + lgbm_prob:0.55", "exit": "rsi_overbought:70", "label": "Trend+RSI+ML"},
{"entry": "ema_cross_up:12:26 + lgbm_prob:0.50", "exit": "atr_trailing_exit:14:2.5", "label": "EMACross+ML→ATR"},
{"entry": "volume_spike:2.0 + adx_strong:25 + lgbm_prob:0.55", "exit": "rsi_overbought:70", "label": "Vol+ADX+ML"},
```

## Phase 2: ml-train-all CLI 명령

### 2-1. `cli.py`에 `ml-train-all` 명령 추가

```python
@app.command(name="ml-train-all")
def ml_train_all(
    timeframe: str | None = Option(None, "--timeframe", "-t"),
    train_months: int = Option(3, "--train-months"),
    test_months: int = Option(1, "--test-months"),
    data_dir: str = Option("data", "--data-dir"),
    model_dir: str = Option("models", "--model-dir"),
):
```

동작:
1. `list_available_data()`로 심볼×타임프레임 목록 조회
2. `--timeframe` 지정 시 해당 타임프레임만 필터링
3. 각 조합에 대해 `MLWalkForwardTrainer(symbol, tf, ...).run(df)` 실행
4. Rich 테이블로 결과 출력: symbol, timeframe, avg_auc, avg_precision, n_windows, model_path
5. 실패 시 에러 로그 + 다음 조합 계속 (전체 중단 안 함)

## Phase 3: 테스트

### 3-1. `tests/test_combine.py` 추가

- `test_lgbm_prob_filter_with_model`: 모델 있을 때 check_entry/strength 동작
- `test_lgbm_prob_filter_no_model`: 모델 없을 때 항상 False
- `test_lgbm_prob_combined_strength`: CombinedStrategy에서 strength 전달 확인
- `test_lgbm_prob_parse`: `parse_filter_spec("lgbm_prob:0.55")` 파싱

### 3-2. `tests/test_ml.py` 추가

- `test_lgbm_prob_filter_backtest`: train → save → CombinedStrategy(lgbm_prob + rule) → BacktestEngine → 파이프라인 정상 동작

## 파일별 변경 요약

| 순서 | 파일 | 작업 |
|------|------|------|
| 1 | `strategy/filters/ml.py` | 신규 — LgbmProbFilter |
| 2 | `strategy/filters/registry.py` | lgbm_prob 등록 + 파싱 |
| 3 | `strategy/combined.py` | strength 수집 로직 |
| 4 | `cli.py` | symbol/timeframe 전달 + 템플릿 + ml-train-all |
| 5 | `tests/test_combine.py` | 필터 단위 테스트 |
| 6 | `tests/test_ml.py` | 통합 테스트 |
| 7 | `CLAUDE.md`, `README.md` | 문서 업데이트 |

## 검증 기준

1. 기존 150개 테스트 전부 통과
2. `lgbm_prob:0.55` 파싱 → LgbmProbFilter(threshold=0.55) 생성
3. 모델 있을 때: check_entry() = prob >= threshold, last_strength = half_kelly(prob)
4. 모델 없을 때: check_entry() = False, 경고 로그 1회
5. CombinedStrategy에서 Signal.strength가 Half-Kelly 값으로 설정
6. `ml-train-all`이 모든 다운로드된 데이터에 대해 모델 생성
7. lightgbm 미설치 시 나머지 30개 필터 정상 동작
