# ML_PLAN.md

LightGBM 메타 모델 전략 구현 계획. ML_RESEARCH.md 기반.

---

## Phase 1: 의존성 + 프로젝트 구조

### 1-1. `pyproject.toml` — ml 옵셔널 의존성 추가

```toml
[project.optional-dependencies]
ml = [
    "lightgbm>=4.3.0",
    "scikit-learn>=1.4.0",
    "optuna>=3.5.0",
    "shap>=0.45.0",
]
```

### 1-2. 디렉토리 구조

```
src/tradingbot/ml/
├── __init__.py
├── features.py       # 피처 엔지니어링
├── targets.py        # 타겟 변수 생성
├── trainer.py        # LGBMTrainer (학습 + 저장 + 평가)
└── walk_forward.py   # MLWalkForwardTrainer

src/tradingbot/strategy/
└── lgbm_strategy.py  # LGBMStrategy (추론 전용)

models/                # .lgb + _meta.json 저장 (gitignore)
└── .gitkeep
```

---

## Phase 2: 피처 엔지니어링 (`src/tradingbot/ml/features.py`)

### 2-1. `build_feature_matrix(df) -> tuple[pd.DataFrame, list[str]]`

기존 `indicators.py`의 함수들을 호출하여 인디케이터를 먼저 계산한 뒤, 파생 피처를 추가.

**입력**: OHLCV DataFrame (warmup 포함)
**출력**: (피처가 추가된 df, feature_cols 리스트)

```python
WARMUP_CANDLES = 52  # Ichimoku window3

def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # 1단계: 기존 인디케이터 계산
    df = add_rsi(df, period=14)
    df = add_adx(df, period=14)
    df = add_stochastic(df, k_period=14, d_period=3)
    df = add_aroon(df, period=25)
    df = add_mfi(df, period=14)
    df = add_cci(df, period=20)
    df = add_roc(df, period=12)
    df = add_zscore(df, period=20)
    df = add_pct_from_ma(df, period=20)
    df = add_macd(df, fast=12, slow=26, signal=9)
    df = add_bollinger_bands(df, period=20, std=2.0)
    df = add_atr(df, period=14)
    df = add_obv(df)
    df = add_volume_sma(df, period=20)
    df = add_keltner_channel(df, period=20, atr_period=10)
    df = add_donchian_channel(df, period=20)
    df = add_ichimoku(df, window1=9, window2=26, window3=52)
    df = add_ema(df, period=20)

    # 2단계: 파생 피처
    df["obv_roc_10"] = df["obv"].pct_change(10)
    df["atr_pct_14"] = df["atr_14"] / df["close"]
    df["bb_pos_20"] = (df["close"] - df["bb_lower_20_2.0"]) / (
        (df["bb_upper_20_2.0"] - df["bb_lower_20_2.0"]).replace(0, float("nan"))
    )
    df["bb_kc_squeeze"] = (
        (df["bb_upper_20_2.0"] - df["bb_lower_20_2.0"]) <
        (df["kc_upper_20"] - df["kc_lower_20"])
    ).astype(int)
    df["dc_pos_20"] = (df["close"] - df["dc_lower_20"]) / (
        (df["dc_upper_20"] - df["dc_lower_20"]).replace(0, float("nan"))
    )
    df["macd_norm"] = df["macd_12_26_9"] / df["atr_14"].replace(0, float("nan"))
    df["ichi_dist"] = (
        df["close"] - df[["ichi_a_9_26_52", "ichi_b_9_26_52"]].max(axis=1)
    ) / df["close"]
    df["adx_di_diff"] = df["adx_pos_14"] - df["adx_neg_14"]
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"].replace(0, float("nan"))
    for lag in [1, 3, 5]:
        df[f"close_roc_{lag}"] = df["close"].pct_change(lag)
    df["hl_range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["candle_body"] = (df["close"] - df["open"]) / df["open"].replace(0, float("nan"))
    df["stoch_kd_diff"] = df["stoch_k_14"] - df["stoch_d_14_3"]
    df["rsi_roc_3"] = df["rsi_14"].diff(3)
    df["close_std_20"] = df["close"].pct_change().rolling(20).std()
    df["atr_rank_50"] = df["atr_14"].rolling(50).rank(pct=True)
    df["rsi_dist_from_50"] = df["rsi_14"] - 50.0

    # 3단계: 피처 컬럼 목록 반환
    feature_cols = [
        # Raw indicators
        "rsi_14", "adx_14", "adx_pos_14", "adx_neg_14",
        "stoch_k_14", "stoch_d_14_3", "aroon_up_25", "aroon_down_25",
        "mfi_14", "cci_20", "roc_12", "zscore_20", "pct_from_ma_20",
        "macd_hist_12_26_9",
        # Derived
        "obv_roc_10", "atr_pct_14", "bb_pos_20", "bb_kc_squeeze",
        "dc_pos_20", "macd_norm", "ichi_dist", "adx_di_diff",
        "volume_ratio", "close_roc_1", "close_roc_3", "close_roc_5",
        "hl_range_pct", "candle_body", "stoch_kd_diff", "rsi_roc_3",
        "close_std_20", "atr_rank_50", "rsi_dist_from_50",
    ]
    return df, feature_cols
```

**div-by-zero 보호**: 모든 나눗셈에서 `.replace(0, float("nan"))` 적용.

---

## Phase 3: 타겟 변수 (`src/tradingbot/ml/targets.py`)

### 3-1. `build_target(df, forward_candles=4, threshold=0.006) -> pd.Series`

```python
def build_target(
    df: pd.DataFrame,
    forward_candles: int = 4,
    threshold: float = 0.006,  # 0.5% 수익 + 0.1% 수수료
) -> pd.Series:
    fwd_return = df["close"].pct_change(forward_candles).shift(-forward_candles)
    return (fwd_return > threshold).astype(int)
```

**주의**: 이 함수는 오프라인 학습 파이프라인에서만 호출. `.shift(-N)` 사용 → 절대 indicators() 안에서 호출 금지.

---

## Phase 4: LGBMTrainer (`src/tradingbot/ml/trainer.py`)

### 4-1. 클래스 구조

```python
class LGBMTrainer:
    def __init__(self, params: dict | None = None):
        self.params = params or DEFAULT_LGBM_PARAMS

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> lgb.Booster:
        """LightGBM 학습. early stopping은 val set으로."""

    def evaluate(
        self,
        model: lgb.Booster,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict:
        """AUC, precision, recall, f1 등 평가 지표 반환."""

    def save(
        self,
        model: lgb.Booster,
        symbol: str,
        timeframe: str,
        meta: dict,
        model_dir: Path = Path("models"),
    ) -> Path:
        """모델(.lgb) + 메타데이터(_meta.json) 저장."""

    @staticmethod
    def load(symbol: str, timeframe: str, model_dir: Path = Path("models")) -> lgb.Booster | None:
        """저장된 모델 로드. 없으면 None."""
```

### 4-2. DEFAULT_LGBM_PARAMS

ML_RESEARCH.md 섹션 3의 파라미터 그대로 사용:
- `objective: binary`, `metric: [binary_logloss, auc]`
- `num_leaves: 31`, `max_depth: 6`, `min_data_in_leaf: 200`
- `learning_rate: 0.02`, `n_estimators: 2000`, `early_stopping_rounds: 100`
- `feature_fraction: 0.7`, `bagging_fraction: 0.8`
- `is_unbalance: True`

### 4-3. train() 내부 흐름

```
1. lgb.Dataset(X_train, label=y_train)
2. lgb.Dataset(X_val, label=y_val) as eval set (있으면)
3. lgb.train(params, train_set, valid_sets=[val_set], callbacks=[early_stopping, log_evaluation])
4. return booster
```

### 4-4. evaluate() 출력

```python
{
    "auc": 0.58,
    "precision": 0.52,
    "recall": 0.61,
    "f1": 0.56,
    "n_test": 500,
    "positive_rate": 0.38,
}
```

---

## Phase 5: LGBMStrategy (`src/tradingbot/strategy/lgbm_strategy.py`)

### 5-1. Strategy 인터페이스 구현

```python
class LGBMStrategy(Strategy):
    name = "lgbm"
    timeframe = "1h"
    symbols = ["BTC/KRW"]  # 기본값. CLI/config에서 오버라이드됨

    def __init__(self, params: StrategyParams | None = None):
        super().__init__(params)
        self.entry_threshold = self.params.get("entry_threshold", 0.60)
        self.exit_threshold = self.params.get("exit_threshold", 0.45)
        self.model_dir = Path(self.params.get("model_dir", "models"))

        # 모델 로드 (심볼별)
        self._models: dict[str, lgb.Booster] = {}
        self._feature_cols: list[str] | None = None

    def _load_model(self, symbol: str) -> lgb.Booster | None:
        """심볼별 모델 lazy 로드."""
        if symbol not in self._models:
            model = LGBMTrainer.load(symbol, self.timeframe, self.model_dir)
            if model:
                self._models[symbol] = model
        return self._models.get(symbol)

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """피처 계산 + 모델 추론 → df["lgbm_prob"] 컬럼 추가."""
        df, self._feature_cols = build_feature_matrix(df)

        # 모델이 로드되어 있으면 추론
        # (심볼 정보는 should_entry/should_exit에서 처리)
        return df

    def should_entry(self, df: pd.DataFrame, symbol: str) -> Signal | None:
        if len(df) < WARMUP_CANDLES + 2:
            return None

        model = self._load_model(symbol)
        if model is None or self._feature_cols is None:
            return None

        # 마지막 봉의 피처로 추론
        X = df[self._feature_cols].iloc[[-1]]
        if X.isna().any(axis=1).iloc[0]:
            return None

        prob = model.predict(X)[0]
        if prob < self.entry_threshold:
            return None

        strength = min(_half_kelly(prob), 1.0)
        return Signal(
            timestamp=df.index[-1].to_pydatetime(),
            symbol=symbol,
            signal_type=SignalType.LONG_ENTRY,
            price=df["close"].iloc[-1],
            strength=strength,
        )

    def should_exit(self, df: pd.DataFrame, symbol: str, position: Position) -> Signal | None:
        if len(df) < WARMUP_CANDLES + 2:
            return None

        model = self._load_model(symbol)
        if model is None or self._feature_cols is None:
            return None

        X = df[self._feature_cols].iloc[[-1]]
        if X.isna().any(axis=1).iloc[0]:
            return None

        prob = model.predict(X)[0]
        if prob < self.exit_threshold:
            return Signal(
                timestamp=df.index[-1].to_pydatetime(),
                symbol=symbol,
                signal_type=SignalType.LONG_EXIT,
                price=df["close"].iloc[-1],
            )
        return None
```

### 5-2. Half-Kelly 함수

```python
def _half_kelly(p: float, avg_win_loss_ratio: float = 1.5) -> float:
    q = 1.0 - p
    b = avg_win_loss_ratio
    full_kelly = (p * b - q) / b
    return max(0.0, full_kelly * 0.5)
```

### 5-3. strategy registry 등록

`strategy/registry.py`에 추가:
```python
"lgbm": LGBMStrategy,
```

---

## Phase 6: BacktestEngine signal.strength 반영

### 6-1. `backtest/engine.py` 수정 (1줄)

`_handle_signal()` 메서드에서 quantity 계산 후:

```python
# 기존 (line 271-273):
quantity = self.risk_manager.calculate_position_size(
    fill.fill_price, stop_loss, equity
)

# 추가 (line 274):
quantity = quantity * signal.strength  # ML 확률 기반 포지션 사이징
```

**하위 호환**: 기존 전략들의 `Signal.strength` 기본값은 1.0이므로 기존 동작 변경 없음.

---

## Phase 7: ML Walk-Forward (`src/tradingbot/ml/walk_forward.py`)

### 7-1. MLWalkForwardTrainer

```python
class MLWalkForwardTrainer:
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        train_months: int = 3,
        test_months: int = 1,
        embargo_candles: int = 52,
    ):
        ...

    def run(self, df: pd.DataFrame) -> MLWalkForwardReport:
        """
        1. 윈도우 분할 (expanding window + embargo)
        2. 윈도우별 학습 → 평가
        3. 최종 모델 = 전체 데이터로 학습
        4. 성능 리포트 반환
        """
```

### 7-2. 윈도우 분할 로직

```python
def _create_windows(self, df):
    # expanding window: train은 항상 처음부터
    # embargo: train_end와 test_start 사이에 52봉 갭
    # warmup: 피처 계산을 위해 train_start 이전 52봉 버퍼
    windows = []
    ...
    return windows  # list of (train_slice, val_slice, test_slice)
```

### 7-3. MLWalkForwardReport

```python
@dataclass
class MLWalkForwardReport:
    windows: list[dict]      # 윈도우별 {auc, precision, n_train, n_test}
    avg_auc: float
    avg_precision: float
    model_path: Path         # 최종 모델 경로
    feature_importance: dict # {feature_name: importance_score}
```

---

## Phase 8: CLI 명령 (`src/tradingbot/cli.py`)

### 8-1. `tradingbot ml-train`

```bash
tradingbot ml-train --symbol BTC/KRW --timeframe 1h --train-months 6
```

흐름:
1. Parquet 로드
2. `build_feature_matrix()` → 피처 생성
3. `build_target()` → 타겟 생성
4. `MLWalkForwardTrainer.run()` → Walk-Forward 학습 + 검증
5. 최종 모델 저장 → `models/lgbm_BTC_KRW_1h.lgb`
6. 성능 리포트 출력 (AUC, Precision, Feature Importance Top 10)

### 8-2. `tradingbot ml-backtest`

```bash
tradingbot ml-backtest --symbol BTC/KRW --timeframe 1h
```

흐름:
1. 저장된 모델 로드
2. `LGBMStrategy` 생성 → 기존 `BacktestEngine.run()` 실행
3. 기존 BacktestReport 출력 (Sharpe, Drawdown, Win Rate 등)

---

## Phase 9: 테스트 (`tests/test_ml.py`)

### 9-1. 피처 빌드 테스트

```python
class TestFeatures:
    def test_build_feature_matrix(self):
        df = _make_data(200)  # test_combine.py의 헬퍼 재사용
        df, feature_cols = build_feature_matrix(df)
        assert len(feature_cols) == 33
        for col in feature_cols:
            assert col in df.columns
        # warmup 이후에는 NaN이 아닌 값이 있어야 함
        valid = df[feature_cols].iloc[WARMUP_CANDLES:]
        assert valid.notna().any().all()

    def test_no_future_leakage(self):
        """피처가 미래 데이터를 참조하지 않는지 확인."""
        df = _make_data(200)
        df, feature_cols = build_feature_matrix(df)
        # 마지막 봉의 피처가 마지막 봉 이전 데이터만으로 계산 가능한지
        df_partial = df.iloc[:-1].copy()
        df_partial, _ = build_feature_matrix(df_partial)
        # 마지막-1번째 봉의 피처 값이 전체 df와 동일해야 함
        for col in feature_cols:
            if col in df_partial.columns and col in df.columns:
                val_full = df[col].iloc[-2]
                val_partial = df_partial[col].iloc[-1]
                if pd.notna(val_full) and pd.notna(val_partial):
                    assert abs(val_full - val_partial) < 1e-10
```

### 9-2. 타겟 빌드 테스트

```python
class TestTargets:
    def test_build_target(self):
        df = _make_data(100)
        target = build_target(df, forward_candles=4, threshold=0.006)
        assert target.dtype == int
        assert set(target.dropna().unique()).issubset({0, 1})
        # 마지막 4봉은 NaN (forward return 계산 불가)
        assert target.iloc[-4:].isna().all()
```

### 9-3. Trainer 테스트

```python
class TestTrainer:
    def test_train_and_predict(self):
        df = _make_data(500)
        df, feature_cols = build_feature_matrix(df)
        target = build_target(df)
        # warmup + NaN 제거
        mask = df[feature_cols].notna().all(axis=1) & target.notna()
        X, y = df.loc[mask, feature_cols], target[mask]

        # 학습
        split = int(len(X) * 0.7)
        trainer = LGBMTrainer()
        model = trainer.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])

        # 추론
        proba = model.predict(X.iloc[split:])
        assert len(proba) == len(X) - split
        assert all(0 <= p <= 1 for p in proba)

    def test_save_and_load(self, tmp_path):
        # 학습 → 저장 → 로드 → 같은 예측값
        ...
```

### 9-4. LGBMStrategy 백테스트 테스트

```python
class TestLGBMStrategy:
    def test_backtest_runs(self, tmp_path):
        """모델 학습 → 저장 → LGBMStrategy로 백테스트."""
        df = _make_data(500)
        # 학습
        df_feat, feature_cols = build_feature_matrix(df.copy())
        target = build_target(df_feat)
        mask = df_feat[feature_cols].notna().all(axis=1) & target.notna()
        trainer = LGBMTrainer()
        model = trainer.train(df_feat.loc[mask, feature_cols], target[mask])
        trainer.save(model, "BTC/KRW", "1h", {}, model_dir=tmp_path)

        # 백테스트
        strategy = LGBMStrategy(StrategyParams({"model_dir": str(tmp_path)}))
        strategy.timeframe = "1h"
        config = AppConfig(...)
        engine = BacktestEngine(strategy=strategy, config=config)
        report = engine.run({"BTC/KRW": df})
        assert report.final_balance > 0
```

---

## Phase 10: 검증

### 10-1. 단위 테스트

```bash
pip install -e ".[ml]"
pytest tests/test_ml.py -v
```

### 10-2. 전체 테스트

```bash
pytest tests/ -v
```

기존 139개 + 신규 ML 테스트 전부 통과 확인.

### 10-3. 실제 데이터 통합 테스트

```bash
# 1. 모델 학습
tradingbot ml-train --symbol BTC/KRW --timeframe 1h

# 2. ML 백테스트
tradingbot ml-backtest --symbol BTC/KRW --timeframe 1h

# 3. 기존 전략과 비교
tradingbot backtest --strategy sma_cross --symbol BTC/KRW
tradingbot backtest --strategy lgbm --symbol BTC/KRW
```

### ML과 기존 Rule-Base 전략의 관계

**기존 rule-base 위에 ML을 얹는 구조입니다.** 별개가 아닙니다.

```
기존 indicators.py (19개 인디케이터: RSI, ADX, MACD, BB, ...)
    ↓ 그대로 재사용 (피처로 변환)
ml/features.py (인디케이터 값 + 파생 피처 = ~33개)
    ↓
LightGBM 모델 → "상승 확률 73%" 출력
    ↓
LGBMStrategy (기존 Strategy 인터페이스 준수)
    ↓
기존 BacktestEngine / LiveEngine / RiskManager 그대로 사용
```

차이점:
- **CombinedStrategy** (rule-base): "RSI < 30 AND ADX > 25" → 하드 규칙, 사람이 정의
- **LGBMStrategy** (ML): "RSI 33이어도 ADX와 OBV 조합에 따라 매수 확률 73%" → 모델이 학습

`LGBMStrategy`는 기존 6개 전략과 **동일한 레벨**로 동작. `tradingbot backtest --strategy lgbm`으로 실행하고, 기존 엔진/리스크 관리를 그대로 사용.

### 멀티 심볼 지원

ML 모델은 **심볼별로 따로 학습**합니다 (BTC와 DOGE의 가격 패턴이 다름):

```bash
# 심볼별 모델 학습 (8개 심볼 각각)
tradingbot ml-train --symbol BTC/KRW --timeframe 1h
tradingbot ml-train --symbol ETH/KRW --timeframe 1h
tradingbot ml-train --symbol SOL/KRW --timeframe 1h
# ...

# 모델 파일도 심볼별 저장
models/lgbm_BTC_KRW_1h.lgb
models/lgbm_ETH_KRW_1h.lgb
models/lgbm_SOL_KRW_1h.lgb
```

`LGBMStrategy._load_model(symbol)`이 심볼에 맞는 모델을 자동 로드하므로 멀티 심볼 백테스트/라이브 정상 동작.

---

## 실행 순서 요약

| 순서 | Phase | 파일 | 작업 |
|------|-------|------|------|
| 1 | Phase 1 | `pyproject.toml` | ml 의존성 추가 |
| 2 | Phase 1 | `src/tradingbot/ml/__init__.py` | 패키지 생성 |
| 3 | Phase 2 | `src/tradingbot/ml/features.py` | 피처 엔지니어링 (~33개 피처) |
| 4 | Phase 3 | `src/tradingbot/ml/targets.py` | 타겟 변수 생성 |
| 5 | Phase 4 | `src/tradingbot/ml/trainer.py` | LGBMTrainer (학습/저장/평가/로드) |
| 6 | Phase 5 | `src/tradingbot/strategy/lgbm_strategy.py` | LGBMStrategy (추론 + Kelly sizing) |
| 7 | Phase 5 | `src/tradingbot/strategy/registry.py` | "lgbm" 등록 |
| 8 | Phase 6 | `src/tradingbot/backtest/engine.py` | `quantity *= signal.strength` (1줄) |
| 9 | Phase 7 | `src/tradingbot/ml/walk_forward.py` | MLWalkForwardTrainer |
| 10 | Phase 8 | `src/tradingbot/cli.py` | `ml-train`, `ml-backtest` 명령 |
| 11 | Phase 9 | `tests/test_ml.py` | 피처/타겟/학습/전략 테스트 |
| 12 | Phase 10 | — | pytest + 실데이터 검증 |

**총 신규 파일**: 6개 (`ml/` 4개 + `lgbm_strategy.py` + `test_ml.py`)
**총 수정 파일**: 3개 (`pyproject.toml`, `strategy/registry.py`, `backtest/engine.py`)
**기존 코드 변경 최소화**: engine.py 1줄, registry.py 2줄

### Anti-Lookahead 체크리스트

- [ ] `build_target()`의 `.shift(-N)`은 `ml/targets.py`에만 존재
- [ ] `LGBMStrategy.indicators()`에서 `.shift(-N)` 사용 안 함
- [ ] `build_feature_matrix()`의 모든 연산이 과거 데이터만 참조
- [ ] Embargo 52봉이 WF train/test 사이에 적용
- [ ] 모델 학습은 BacktestEngine 외부 (`ml/trainer.py`)에서 수행
- [ ] 추론(`model.predict()`)만 엔진 내부에서 실행
