# ML_RESEARCH.md

LightGBM 메타 모델 전략 구현을 위한 리서치 결과.

---

## 1. Feature Engineering

### 1.1 기존 인디케이터 값 (Raw Features)

LightGBM은 스케일 불변이므로 정규화 없이 그대로 사용 가능:

```
rsi_14                          # 0-100
adx_14, adx_pos_14, adx_neg_14 # 0-100
stoch_k_14, stoch_d_14_3       # 0-100
aroon_up_25, aroon_down_25      # 0-100
mfi_14                          # 0-100
cci_20                          # -200..+200 범위
roc_12                          # % 변화
zscore_20                       # 표준화됨, -3..+3
pct_from_ma_20                  # MA 대비 % 괴리
macd_hist_12_26_9               # 부호 중요
```

### 1.2 파생 피처 (Derived Features)

피처 빌드 시점에 계산 (indicators.py가 아닌 별도 모듈에서):

```python
# OBV 변화율 (raw OBV는 비정상 시계열이라 직접 쓰면 안 됨)
df["obv_roc_10"] = df["obv"].pct_change(10)

# ATR을 가격 대비 % (정규화된 변동성)
df["atr_pct_14"] = df["atr_14"] / df["close"]

# 볼린저 밴드 내 위치 (0=하단, 1=상단)
df["bb_pos_20"] = (df["close"] - df["bb_lower_20_2.0"]) / (
    df["bb_upper_20_2.0"] - df["bb_lower_20_2.0"]
)

# 켈트너 스퀴즈 (BB < KC = 스퀴즈 활성)
df["bb_kc_squeeze"] = (
    (df["bb_upper_20_2.0"] - df["bb_lower_20_2.0"]) <
    (df["kc_upper_20"] - df["kc_lower_20"])
).astype(int)

# 돈치안 채널 내 위치
df["dc_pos_20"] = (df["close"] - df["dc_lower_20"]) / (
    df["dc_upper_20"] - df["dc_lower_20"]
)

# MACD를 ATR로 정규화 (심볼/가격 수준 간 비교 가능)
df["macd_norm"] = df["macd_12_26_9"] / df["atr_14"]

# 이치모쿠 구름 거리 (양수=구름 위, 음수=구름 아래)
df["ichi_dist"] = (df["close"] - df[["ichi_a_9_26_52", "ichi_b_9_26_52"]].max(axis=1)) / df["close"]

# ADX 방향 차이 (+DI vs -DI)
df["adx_di_diff"] = df["adx_pos_14"] - df["adx_neg_14"]

# 거래량 비율 (현재 vs 20봉 평균)
df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

# 가격 모멘텀 래그 (1, 3, 5봉)
for lag in [1, 3, 5]:
    df[f"close_roc_{lag}"] = df["close"].pct_change(lag)

# 봉 내 변동폭 (High-Low range as %)
df["hl_range_pct"] = (df["high"] - df["low"]) / df["close"]

# 캔들 바디 (양봉/음봉 + 크기)
df["candle_body"] = (df["close"] - df["open"]) / df["open"]

# Stochastic %K - %D (모멘텀의 모멘텀)
df["stoch_kd_diff"] = df["stoch_k_14"] - df["stoch_d_14_3"]

# RSI 변화율 (가속도)
df["rsi_roc_3"] = df["rsi_14"].diff(3)
```

### 1.3 롤링 통계 피처

레짐 수준의 컨텍스트 캡처:

```python
# 롤링 변동성 레짐
df["close_std_20"] = df["close"].pct_change().rolling(20).std()

# ATR 백분위 순위 (현재 변동성이 역사적으로 높은가/낮은가)
df["atr_rank_50"] = df["atr_14"].rolling(50).rank(pct=True)

# RSI 평균회귀 포텐셜
df["rsi_dist_from_50"] = df["rsi_14"] - 50.0
```

### 1.4 Anti-Lookahead 규칙

- `.shift()`, `.diff()`, `.pct_change()`, `.rolling()` 연산은 `visible_df` (엔진이 `df.iloc[:idx]`로 자른 슬라이스)에서만 안전
- `.shift(-1)` 또는 미래 캔들 참조는 절대 피처에 사용 금지
- **타겟 변수(forward return)는 절대 `indicators()` 안에서 계산하지 않음** — 오프라인 학습 파이프라인에서만
- 기존 엔진이 `visible_df = df.iloc[:idx].copy()`로 구조적으로 보장

**총 피처: ~25개** (Raw ~12 + Derived ~13)

---

## 2. Target Variable Design

### 추천: 4h Forward Return 이진 분류

```python
# 오프라인 학습 데이터 준비에서만 — indicators() 안에서 절대 사용 금지
FORWARD_CANDLES = 4       # 1h 데이터 기준 4h 호라이즌
RETURN_THRESHOLD = 0.005  # 0.5% 최소 수익
FEE_ROUNDTRIP = 0.001    # Upbit 0.05% × 2

df["fwd_return"] = df["close"].pct_change(FORWARD_CANDLES).shift(-FORWARD_CANDLES)
df["target"] = (df["fwd_return"] > RETURN_THRESHOLD + FEE_ROUNDTRIP).astype(int)
```

**왜 4h인가:**
- 1h: 잡음 과다, 신호 대 잡음비 붕괴
- 24h: 학습 샘플 부족, 야간 변동성 노출
- 4h: Upbit KRW 크립토에서 일중 모멘텀 사이클에 적합

**왜 이진 분류인가:**
- 회귀는 크립토에서 분산이 너무 커서 수렴 안 됨
- 확률 출력이 `Signal.strength` (0.0-1.0)에 자연스럽게 매핑
- 진입(prob > 0.60)과 청산(prob < 0.45) 임계값을 다르게 설정 가능

**클래스 불균형 처리:**
- 4h + 0.5% 임계값 → BTC/KRW 1h 데이터에서 ~35-42% 양성 클래스
- LightGBM `is_unbalance=True` 사용 (SMOTE 금지 — 합성 금융 시계열은 무의미)

---

## 3. LightGBM 하이퍼파라미터

```python
lgbm_params = {
    # 목적 함수
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "verbose": -1,

    # 트리 구조 — 오버피팅 방지를 위해 얕게
    "num_leaves": 31,              # 범위: 15-63
    "max_depth": 6,                # 범위: 4-8
    "min_data_in_leaf": 200,       # 금융에서 핵심: 높게 설정 (범위: 100-500)
    "min_sum_hessian_in_leaf": 10.0,

    # 정규화 — 시계열에서 가장 중요
    "reg_alpha": 0.1,              # L1 (범위: 0.0-1.0)
    "reg_lambda": 0.5,             # L2 (범위: 0.1-2.0)
    "feature_fraction": 0.7,       # 트리당 피처 70% 사용 (범위: 0.5-0.9)
    "bagging_fraction": 0.8,       # 행 서브샘플링 (범위: 0.6-0.9)
    "bagging_freq": 5,

    # 클래스 불균형
    "is_unbalance": True,

    # 학습
    "learning_rate": 0.02,         # 기본값 0.1보다 낮게 (범위: 0.01-0.05)
    "n_estimators": 2000,          # early stopping이 잘라줌
    "early_stopping_rounds": 100,

    # 속도
    "n_jobs": -1,
    "device": "cpu",               # 10만 행 이하 테이블 데이터에서 GPU 이점 거의 없음
    "seed": 42,
}
```

**핵심: `min_data_in_leaf=200`이 금융 데이터에서 가장 중요한 정규화 장치.** 각 리프에 최소 200개 샘플을 요구해서 한 번만 나타나는 희귀 가격 패턴 암기를 방지.

**Early stopping**: Walk-Forward 다음 윈도우의 첫 절반을 eval set으로 사용. AUC가 100라운드 동안 개선 안 되면 중단. 보통 300-600 트리에서 수렴.

---

## 4. Train/Test Split (시계열)

### Purged Walk-Forward with Embargo

표준 k-fold CV는 롤링 윈도우 피처(예: rsi_14의 row 100과 101이 13봉 공유) 때문에 정보 누출 발생.

**Embargo 기간 계산 (1h 데이터):**
```
max_lookback = max(ichimoku_window3=52, aroon=25, ...) = 52 candles
embargo_candles = 52  # train과 test 사이에 52봉 드롭
```

**Expanding window 사용** (Sliding 아님):
- 항상 처음부터 학습 시작 — 오래된 데이터도 레짐 정보 보유
- Expanding + early stopping으로 오래된 레짐 암기 방지

---

## 5. Walk-Forward 통합

기존 `WalkForwardValidator`는 `GridSearchOptimizer`를 호출하지만, ML은 별도의 학습 루프 필요:

```
MLWalkForwardTrainer (신규 클래스)
├── create_walk_forward_windows() 재사용
├── 윈도우별:
│   ├── build_feature_matrix(train_df) → X_train, y_train
│   ├── train_lgbm(X_train, y_train) → lgb.Booster
│   ├── model.save_model("models/{symbol}_{window}.lgb")
│   └── test 윈도우 평가 → AUC, profit_factor
└── 최종 모델 = 전체 데이터로 학습 → paper/live에서 사용
```

**LGBMStrategy.indicators()**: 모델 파일을 `__init__`에서 로드하고, `indicators()` 에서는 추론(`.predict()`)만 수행 → `df["lgbm_prob"]` 컬럼 추가.

---

## 6. Position Sizing: Half-Kelly

모델 확률 출력을 `Signal.strength`에 매핑:

```python
def _half_kelly(p: float, avg_win_loss_ratio: float = 1.5) -> float:
    """Half-Kelly criterion for position sizing."""
    q = 1.0 - p
    b = avg_win_loss_ratio
    full_kelly = (p * b - q) / b
    return max(0.0, full_kelly * 0.5)  # half-Kelly

# LGBMStrategy.should_entry():
prob = float(df["lgbm_prob"].iloc[-1])
if prob < 0.60:
    return None
strength = min(_half_kelly(prob), 1.0)
return Signal(..., strength=strength)
```

**BacktestEngine 수정**: `_handle_signal()`에서 `quantity *= signal.strength` (1줄 변경)

---

## 7. Python Dependencies

`pyproject.toml`에 `ml` 옵셔널 그룹 추가:

```toml
[project.optional-dependencies]
ml = [
    "lightgbm>=4.3.0",       # Apple Silicon 네이티브 wheel (4.3+)
    "scikit-learn>=1.4.0",    # metrics, split utilities
    "optuna>=3.5.0",          # 하이퍼파라미터 탐색
    "shap>=0.45.0",           # 피처 중요도 시각화
]
```

설치: `pip install -e ".[ml]"`

**M4 24GB에서의 성능 예상:**
- 학습 (5000행 × 25피처): ~2초
- Walk-Forward 5 윈도우: ~15초
- 추론 (1행): <1ms (실시간 거래 가능)

---

## 8. Model Persistence

### 파일 형식과 위치

```python
# 저장 (.lgb 텍스트 형식 — 버전 안정적, git diff 가능)
MODEL_DIR = Path("models")
model_path = MODEL_DIR / f"lgbm_{symbol_key}_{timeframe}.lgb"
booster.save_model(str(model_path))

# 메타데이터 (학습 조건 기록)
meta = {
    "trained_at": datetime.utcnow().isoformat(),
    "train_start": str(train_start),
    "train_end": str(train_end),
    "n_features": len(feature_cols),
    "feature_names": feature_cols,
    "threshold_entry": 0.60,
    "threshold_exit": 0.45,
    "val_auc": float(val_auc),
    "forward_candles": 4,
}
meta_path = MODEL_DIR / f"lgbm_{symbol_key}_{timeframe}_meta.json"
```

**왜 `.lgb`인가:** pickle은 lightgbm 버전 변경 시 깨짐. `.lgb`는 텍스트 기반이라 버전 간 안정적.

---

## 9. 성능 평가 지표

### 모델 레벨 (백테스트 전)

| 지표 | 목표 | 의미 |
|------|------|------|
| AUC-ROC | > 0.55 | 크립토에서 통계적으로 유의미, > 0.60이면 좋음 |
| Precision @ threshold | > 0.50 | 진입 시그널의 절반 이상이 수익 |
| Calibration | 직선에 가까움 | prob=0.65가 실제 65% 맞는지 |

### 트레이딩 레벨 (백테스트 후)

| 지표 | 목표 | 의미 |
|------|------|------|
| Sharpe Ratio | > 1.5 | 리스크 대비 수익 |
| Profit Factor | > 1.5 | Upbit 0.1% 왕복 수수료 극복 |
| Max Drawdown | < 15% | 리스크 관리 |
| Avg Hold Hours | > 4h | 틱 잡음 오버피팅 아님 |
| Fee Drag | < 5% annual | 거래 빈도 지속 가능 |

---

## 10. Regime Detection (2단계, Optional)

### HMM 3-State Model

```python
# hmmlearn>=0.3.0 추가 필요
from hmmlearn import hmm

regime_features = np.column_stack([
    df["close"].pct_change().fillna(0).values,  # 수익률
    df["close_std_20"].fillna(0).values,         # 변동성
    df["adx_14"].fillna(25).values / 100,        # 추세 강도
])

model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=200, random_state=42)
model.fit(regime_features)
regimes = model.predict(regime_features)  # 0, 1, 2
```

**레짐 별 해석:**
- 낮은 변동성 + ADX < 25 = 횡보장 (ranging)
- 양의 수익률 + ADX > 25 = 상승 추세 (trending-up)
- 음의 수익률 + 높은 변동성 = 하락/폭락 (trending-down)

**활용**: 레짐을 LightGBM 카테고리 피처로 추가하거나, 레짐별 별도 모델 학습.

---

## 11. 파일별 구현 계획

| 파일 | 작업 |
|------|------|
| `pyproject.toml` | `ml` 옵셔널 의존성 추가 |
| `src/tradingbot/ml/__init__.py` | 패키지 초기화 |
| `src/tradingbot/ml/features.py` | `build_feature_matrix(df)` — 25개 피처 생성, warmup=52 |
| `src/tradingbot/ml/targets.py` | `build_target(df, forward=4, threshold=0.005)` — 타겟 생성 |
| `src/tradingbot/ml/trainer.py` | `LGBMTrainer` — 학습, 저장, 평가 |
| `src/tradingbot/ml/walk_forward.py` | `MLWalkForwardTrainer` — purged WF 학습 |
| `src/tradingbot/strategy/lgbm_strategy.py` | `LGBMStrategy` — 추론 + Signal.strength |
| `src/tradingbot/backtest/engine.py` | `quantity *= signal.strength` (1줄 변경) |
| `src/tradingbot/cli.py` | `tradingbot ml-train`, `tradingbot ml-backtest` CLI 명령 |
| `models/` | 학습된 `.lgb` + `_meta.json` 저장 디렉토리 |
| `tests/test_ml.py` | 피처 빌드, 타겟 생성, 학습 파이프라인 테스트 |

### 구현 순서

1. `pyproject.toml` — 의존성 추가
2. `ml/features.py` + `ml/targets.py` — 피처/타겟 엔지니어링
3. `ml/trainer.py` — LightGBM 학습 파이프라인
4. `strategy/lgbm_strategy.py` — Strategy 인터페이스 구현
5. `backtest/engine.py` — signal.strength 반영 (1줄)
6. `ml/walk_forward.py` — ML Walk-Forward 검증
7. `cli.py` — CLI 명령 추가
8. `tests/` — 테스트
9. 검증 — 학습 → 백테스트 → 기존 전략 대비 비교

### Anti-Lookahead 체크리스트

- [ ] `build_target()`의 `.shift(-N)`은 오프라인 학습에서만 호출
- [ ] `LGBMStrategy.indicators()`에서 `.shift(-N)` 절대 사용 안 함
- [ ] 피처 빌드 시 warmup 버퍼 52봉 확보
- [ ] Embargo 52봉이 train/test 사이에 적용
- [ ] 모델 학습은 BacktestEngine 외부에서 수행 (엔진은 추론만)

---

## 12. 학습 데이터와 학습 시점, 온라인 모델 업데이트

### 학습 데이터는 언제 구하나?

이미 갖고 있는 데이터를 사용합니다. `tradingbot download`로 받아둔 Parquet 파일(8심볼 × 3타임프레임, 최대 5년치)이 그대로 학습 데이터입니다. 별도의 데이터 수집 작업은 필요 없습니다.

```
data/BTC_KRW/1h.parquet  → 이 데이터에서 피처 + 타겟을 빌드
```

### 학습 시점은 언제인가?

**거래 전에 오프라인으로 학습합니다.** 실시간 거래 중에 학습하는 게 아닙니다.

```
[오프라인] tradingbot ml-train --symbol BTC/KRW --timeframe 1h
  → Parquet 로드 → 피처 빌드 → LightGBM 학습 → models/lgbm_BTC_KRW_1h.lgb 저장
  → Walk-Forward 검증 → AUC, Sharpe 등 성능 리포트

[온라인] tradingbot paper --strategy lgbm --symbol BTC/KRW
  → 저장된 .lgb 모델 로드 → 매 봉마다 추론(.predict())만 수행
  → 학습은 안 함, 추론만 함
```

학습은 M4에서 ~2초면 끝나고, 추론은 <1ms라 실시간 거래에 영향 없습니다.

### 실시간 거래 중 모델 업데이트가 가능한가?

**가능합니다. 두 가지 방식이 있습니다:**

#### 방식 1: 주기적 재학습 (추천)

```
매주 일요일 새벽 (크론잡 또는 수동)
  1. tradingbot download --symbol BTC/KRW --timeframe 1h  # 최신 데이터 받기
  2. tradingbot ml-train --symbol BTC/KRW --timeframe 1h  # 새 모델 학습
  3. 라이브 엔진 재시작 → 새 모델 자동 로드
```

- 가장 간단하고 안전한 방식
- 새 데이터가 포함되므로 시장 레짐 변화에 적응
- 기존 모델과 새 모델의 Walk-Forward 성능 비교 후 교체 결정 가능

#### 방식 2: 핫 리로드 (고급)

라이브 엔진을 멈추지 않고 모델 파일만 교체:

```python
# LGBMStrategy에 모델 리로드 로직 추가
class LGBMStrategy(Strategy):
    def indicators(self, df):
        # 모델 파일의 수정 시간이 바뀌었으면 자동 리로드
        if self._model_path.stat().st_mtime > self._model_loaded_at:
            self._model = lgb.Booster(model_file=str(self._model_path))
            self._model_loaded_at = self._model_path.stat().st_mtime
        ...
```

- 엔진 재시작 없이 모델 교체
- 구현이 좀 더 복잡하지만 다운타임 없음
- 1차에서는 방식 1로 시작하고, 필요하면 나중에 추가

### 핵심 포인트

| 항목 | 내용 |
|------|------|
| 학습 데이터 | 이미 다운로드한 Parquet (추가 수집 불필요) |
| 학습 시점 | 거래 전 오프라인 (`tradingbot ml-train`) |
| 학습 빈도 | 주 1회 재학습 권장 (시장 레짐 적응) |
| 실시간 중 학습 | 안 함 — 추론만 수행 (<1ms) |
| 모델 업데이트 | 재학습 후 엔진 재시작 또는 핫 리로드 |
| 오버피팅 방지 | 재학습마다 Walk-Forward 검증 → 성능 하락 시 이전 모델 유지 |
