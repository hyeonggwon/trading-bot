# LgbmProbFilter 리서치

## 목표

LightGBM 모델의 확률 출력을 BaseFilter로 감싸서 기존 30개 룰 기반 필터와 CombinedStrategy에서 조합 가능하게 만든다.

```bash
tradingbot combine --entry "trend_up:4 + rsi_oversold:30 + lgbm_prob:0.55" --exit "rsi_overbought:70"
```

## 현재 시스템 분석

### 1. BaseFilter 인터페이스 (`strategy/filters/base.py`)

```python
class BaseFilter(ABC):
    name: str = "base"
    role: str = "entry"  # "entry" | "trend" | "volatility" | "volume" | "exit"

    def __init__(self, **kwargs):
        self.params = kwargs

    def compute(self, df) -> pd.DataFrame:    # 인디케이터 컬럼 추가
    def check_entry(self, df) -> bool:         # 마지막 캔들 기준 진입 조건
    def check_exit(self, df, entry_index=None) -> bool:  # 청산 조건
```

- `compute()`: CombinedStrategy.indicators()에서 호출, 중복 방지(seen 딕셔너리)
- `check_entry()`: 마지막 확정 캔들 기준, bool 반환
- `check_exit()`: 청산 조건, entry_index로 트레일링 가능

### 2. CombinedStrategy 동작 (`strategy/combined.py`)

- `indicators(df)`: 모든 필터의 compute() 호출 (중복 제거)
- `should_entry(df, symbol)`: 모든 entry 필터 AND (role=="exit" 건너뜀, checked==0 가드)
- `should_exit(df, symbol, position)`: exit 필터 중 하나라도 True면 청산 (OR)
- **Signal.strength = 1.0 (고정)** — 현재 CombinedStrategy는 strength를 설정하지 않음

### 3. LGBMStrategy 동작 (`strategy/lgbm_strategy.py`)

- `_load_model(symbol)`: 심볼별 lazy 로딩 (self._models 딕셔너리 캐시)
- `_predict(df, symbol)`: build_feature_matrix() → model.predict() → float 확률
- `should_entry()`: prob > entry_threshold → Half-Kelly strength
- Signal.strength = Half-Kelly(prob) → backtest engine이 포지션 크기 조절

### 4. ML Feature Pipeline (`ml/features.py`)

- `build_feature_matrix(df)`: 19개 인디케이터 → 33개 피처 컬럼 추가
- `WARMUP_CANDLES = 52`: 최소 필요 캔들 수
- `FEATURE_COLS`: 33개 피처 이름 리스트
- 기존 필터의 compute()와 **일부 인디케이터 중복** (RSI, MACD, BB, Stochastic 등)
  - 필터: `add_rsi(df, 14)` → `rsi_14` 컬럼
  - ML: `add_rsi(df, period=14)` → 동일 `rsi_14` 컬럼
  - 동일 컬럼명이면 두 번째 호출이 무시되므로 충돌 없음 (if col not in df.columns 가드)

### 5. 필터 파싱 (`strategy/filters/registry.py`)

- `parse_filter_spec("lgbm_prob:0.55")` → filter_map에서 클래스 조회 → kwargs 파싱
- `_parse_filter_params()`: 필터별 파라미터 매핑 로직 (elif 체인)
- `get_filter_map()`: 이름→클래스 매핑 (lazy import)

### 6. BacktestEngine의 strength 활용 (`backtest/engine.py:274`)

```python
quantity = quantity * signal.strength  # ML probability-based sizing
```

- strength=1.0이면 risk manager가 계산한 100% 포지션
- strength=0.5면 50% 포지션
- CombinedStrategy는 현재 strength=1.0 고정

## 핵심 설계 이슈

### Issue 1: compute()에서 모델 추론을 매 캔들마다 실행하면 성능 문제

**문제**: CombinedStrategy.indicators()가 매 캔들마다 호출됨. LgbmProbFilter.compute()에서 build_feature_matrix() + model.predict()를 매번 실행하면 극도로 느림.

**분석**:
- build_feature_matrix()는 19개 인디케이터를 계산 → 이미 필터들이 일부 계산한 것과 중복
- model.predict()는 마지막 행 1개만 예측 → 가벼움 (< 1ms)
- 핵심 병목은 build_feature_matrix()의 인디케이터 재계산

**해결 방안**:
- compute()에서는 build_feature_matrix()만 호출 (인디케이터 + 파생 피처 컬럼 추가)
- check_entry()에서 model.predict()만 실행 (마지막 행)
- build_feature_matrix 내부의 `if col not in df.columns` 가드로 중복 인디케이터 스킵됨
- **결론: 성능 영향 허용 가능** — 추가 비용은 파생 피처 계산(df 연산) + predict(< 1ms)

### Issue 2: 모델 로딩 시점

**문제**: BaseFilter.__init__()에서 모델을 로드할 수 없음 — symbol 정보가 __init__ 시점에 없음.

**분석**:
- parse_filter_spec("lgbm_prob:0.55")는 threshold만 파싱
- symbol은 CombinedStrategy.should_entry(df, symbol)에서 전달됨
- 하지만 BaseFilter는 symbol을 받지 않음

**해결 방안**:
- `__init__(threshold, symbol="BTC/KRW", timeframe="1h", model_dir="models")`
- symbol/timeframe을 파라미터로 받음 — CLI에서 전달 가능
- 모델은 첫 compute() 또는 check_entry()에서 lazy 로드 (LGBMStrategy 패턴 동일)
- parse_filter_spec에서 `lgbm_prob:0.55` → threshold=0.55, symbol/timeframe은 외부에서 설정

**대안**: CombinedStrategy에서 symbol을 필터에 전달하도록 수정
- 인터페이스 변경 범위가 큼 (30개 필터 모두 영향)
- **비추천**

**채택**: 필터 생성 후 외부에서 symbol/timeframe 설정. CLI combine 명령에서:
```python
for f in entry_filters:
    if hasattr(f, 'symbol'):
        f.symbol = symbol
    if hasattr(f, 'timeframe'):
        f.timeframe = timeframe
```

### Issue 3: Signal.strength 전달

**문제**: CombinedStrategy.should_entry()는 strength=1.0 고정. ML 확률 기반 포지션 사이징을 하려면 strength를 전달해야 함.

**방안 A — LgbmProbFilter에 strength 속성 추가**:
- check_entry() 통과 시 self.last_strength = half_kelly(prob) 저장
- CombinedStrategy.should_entry()에서 LgbmProbFilter가 있으면 strength 가져옴

**방안 B — CombinedStrategy 수정 없이 strength=1.0 유지**:
- ML 필터는 순수 거부권(veto) 역할만 수행
- 포지션 사이징은 risk manager에게 위임 (현재 고정비율법)
- 단순하지만 Half-Kelly 사이징 이점 포기

**방안 C — CombinedStrategy가 필터에서 strength를 수집**:
- BaseFilter에 `get_strength() -> float | None` 옵셔널 메서드 추가
- 대부분 필터는 None 반환, LgbmProbFilter만 Half-Kelly 반환
- CombinedStrategy는 strength가 있는 필터의 값을 Signal에 전달

**채택: 방안 A** — 가장 단순, 인터페이스 변경 최소
- LgbmProbFilter에 `last_strength` 속성 추가
- CombinedStrategy에서 entry 필터 중 `last_strength` 있는 것 사용

### Issue 4: check_entry()에서 feature NaN 처리

**문제**: build_feature_matrix()는 warmup 기간에 NaN 생성. model.predict()에 NaN 입력 시 LightGBM은 자체적으로 처리하지만 결과 불안정.

**해결**: LGBMStrategy._predict()와 동일한 로직 사용
```python
X = df[FEATURE_COLS].iloc[[-1]]
if X.isna().any(axis=1).iloc[0]:
    return False  # NaN → 조건 불충족
```

### Issue 5: 모델이 없는 경우

**문제**: 모델 파일이 없으면(학습 안 됨) 어떻게 동작해야 하는가?

**해결**:
- 모델 로드 실패 → check_entry() always False (보수적)
- warning 로그 출력 (1회)
- LGBMStrategy의 기존 패턴과 동일

### Issue 6: combine-scan 템플릿에 lgbm_prob 추가

**문제**: combine-scan은 모든 심볼×타임프레임에 템플릿을 적용. ML 모델은 심볼/타임프레임별로 별도 학습 필요.

**해결**:
- lgbm_prob 포함 템플릿은 모델이 존재하는 경우에만 실행됨 (모델 없으면 0 trades)
- 사용자가 먼저 `ml-train`으로 모델을 학습한 후 combine-scan 실행
- 템플릿 2~3개 추가:
  - `"trend_up:4 + rsi_oversold:30 + lgbm_prob:0.55" → "rsi_overbought:70"`
  - `"ema_cross_up:12:26 + lgbm_prob:0.50" → "atr_trailing_exit:14:2.5"`
  - `"volume_spike:2.0 + lgbm_prob:0.60" → "rsi_overbought:70"`

## 구현에 필요한 변경 사항 요약

| 파일 | 변경 |
|------|------|
| `strategy/filters/ml.py` | **신규** — LgbmProbFilter 클래스 |
| `strategy/filters/registry.py` | `lgbm_prob` 등록 + 파라미터 파싱 |
| `strategy/combined.py` | entry 필터에서 strength 수집 (last_strength) |
| `cli.py` | combine/combine-scan에서 symbol/timeframe을 lgbm_prob 필터에 설정 + 템플릿 추가 + `ml-train-all` 명령 추가 |
| `tests/test_combine.py` | LgbmProbFilter 테스트 |
| `tests/test_ml.py` | ML+Rule 조합 통합 테스트 |

### ml-train-all CLI 명령

다운로드된 모든 심볼×타임프레임 조합에 대해 자동으로 ML 모델을 학습하는 명령.

```bash
tradingbot ml-train-all                          # 모든 다운로드된 데이터
tradingbot ml-train-all --timeframe 1h           # 특정 타임프레임만
tradingbot ml-train-all --train-months 3 --test-months 1  # WF 파라미터 지정
```

**동작**:
1. `list_available_data()`로 다운로드된 심볼×타임프레임 목록 조회
2. 각 조합에 대해 `MLWalkForwardTrainer.run()` 실행
3. 진행률 + 결과(AUC, precision) 테이블 출력
4. 이미 모델이 있는 조합은 재학습 (덮어쓰기)

**이점**: combine-scan에서 lgbm_prob 포함 템플릿이 모든 심볼에서 즉시 동작

## 의존성

- lightgbm (이미 `[ml]` optional deps에 포함)
- 기존 ml/features.py, ml/trainer.py 변경 없음
- BaseFilter 인터페이스 변경 없음 (하위 호환)

## 유의사항

1. **Anti-lookahead**: build_feature_matrix()는 과거 데이터만 사용 → 안전. model.predict()도 현재 시점까지의 피처만 사용.

2. **Optional import**: lightgbm이 없으면 LgbmProbFilter 사용 불가 → import 시 try/except로 graceful degradation. ML 없는 환경에서도 나머지 30개 필터는 정상 동작해야 함.

3. **모델 캐싱**: 한 번 로드된 모델은 재사용. build_feature_matrix()는 매번 호출되지만 인디케이터 중복 가드로 추가 비용 최소화.

4. **심볼별 모델**: 현재 ML 시스템은 심볼별 개별 모델 (lgbm_BTC_KRW_1h.lgb). 필터도 동일하게 심볼별 모델 로드.
