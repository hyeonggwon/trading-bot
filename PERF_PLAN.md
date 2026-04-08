# Backtest Engine Performance Optimization Plan

## 목표
combine-scan 1.5시간 → 3~5분 (20~40x 개선)

---

## Change 1: O(N²) copy() 제거 + read-only flag

### 변경 파일: `src/tradingbot/backtest/engine.py`

#### Step 1-1: 인디케이터 DataFrame에 read-only flag 설정

`run()` 메서드에서 인디케이터 사전 계산 직후 (라인 138 이후):

```python
# 현재:
if use_precompute:
    for sym, df in symbol_data.items():
        indicator_data[sym] = self.strategy.indicators(df.copy())

# 변경:
if use_precompute:
    for sym, df in symbol_data.items():
        indicator_data[sym] = self.strategy.indicators(df.copy())
        # read-only로 설정 — 전략이 수정 시 ValueError 발생
        indicator_data[sym].values.flags.writeable = False
```

#### Step 1-2: 메인 루프 copy() 제거

라인 202:
```python
# 현재:
visible_df = indicator_data[sym].iloc[:idx].copy()

# 변경:
visible_df = indicator_data[sym].iloc[:idx]
```

#### Step 1-3: per-iteration path (supports_precompute=False) 처리

라인 204: `MultiTimeframeStrategy`만 사용하는 경로.
`indicators()`가 df에 컬럼을 추가하므로 copy 유지.

```python
# 현재 (유지):
visible_df = symbol_data[sym].iloc[:idx].copy()
visible_df = self.strategy.indicators(visible_df)
```

#### Step 1-4: Phase 3의 .loc 접근 최적화

라인 230:
```python
# 현재:
self._last_known_prices[sym] = float(symbol_data[sym].loc[ts, "close"])

# 변경 (Step 3에서 numpy 배열로 전환):
# 여기서는 일단 유지, Change 3에서 함께 처리
```

### 테스트 방법
- 기존 180개 테스트 전부 통과 확인
- 전략이 visible_df 수정 시도하면 ValueError 발생 확인하는 테스트 추가
- sma_cross BTC/KRW 4h 백테스트 결과가 수정 전후 동일한지 비교

### 예상 효과
- 백테스트당 30~120초 → ~1~5초 (8~15x)
- scan/combine-scan 모두 개선

---

## Change 2: 인디케이터 사전 계산 공유 (combine-scan 전용)

### 변경 파일: `src/tradingbot/backtest/engine.py`, `src/tradingbot/backtest/parallel.py`

#### Step 2-1: engine.run()에 precomputed_indicators 파라미터 추가

```python
# engine.py run() 시그니처:
def run(
    self,
    data: dict[str, pd.DataFrame],
    precomputed_indicators: dict[str, pd.DataFrame] | None = None,
) -> BacktestReport:
```

#### Step 2-2: 사전 계산된 인디케이터 주입 로직

인디케이터 계산 부분 (라인 134~138):
```python
use_precompute = self.strategy.supports_precompute
indicator_data: dict[str, pd.DataFrame] = {}
if use_precompute:
    if precomputed_indicators:
        # 외부에서 사전 계산된 인디케이터 사용
        indicator_data = precomputed_indicators
    else:
        for sym, df in symbol_data.items():
            indicator_data[sym] = self.strategy.indicators(df.copy())
            indicator_data[sym].values.flags.writeable = False
```

주의: precomputed_indicators가 주입되면 현재 strategy의 `indicators()`를 호출하지 않음.
하지만 CombinedStrategy의 각 필터가 필요로 하는 컬럼이 모두 포함되어 있어야 함.

#### Step 2-3: _run_batch에서 union indicator 계산

`parallel.py` `_run_batch()` 수정:
```python
def _run_batch(symbol, timeframe, jobs, data_dir, balance, config_dir):
    ...
    df = load_candles(symbol, timeframe, Path(data_dir))
    config = load_config(...)

    # combined 전략 job이 있는지 확인
    combined_jobs = [(name, entry, exit_) for name, entry, exit_ in jobs if entry]
    registered_jobs = [(name, entry, exit_) for name, entry, exit_ in jobs if not entry]

    # combined job이 있으면 union indicator 사전 계산
    precomputed = None
    if combined_jobs:
        all_filters = []
        for name, entry, exit_ in combined_jobs:
            all_filters += parse_filter_string(entry, base_timeframe=timeframe)
            all_filters += parse_filter_string(exit_, base_timeframe=timeframe)
        # 중복 제거를 위해 unique 컬럼만 계산
        union_strategy = CombinedStrategy(
            entry_filters=all_filters, exit_filters=[]
        )
        union_strategy.symbols = [symbol]
        union_strategy.timeframe = timeframe
        precomputed_df = union_strategy.indicators(df.copy())
        precomputed_df.values.flags.writeable = False
        precomputed = {symbol: precomputed_df}

    # 등록된 전략 실행 (각자 indicators 호출)
    for name, entry, exit_ in registered_jobs:
        strategy_cls = get_strategy_map()[name]
        strategy = strategy_cls()
        ...
        engine = BacktestEngine(strategy=strategy, config=config)
        report = engine.run({symbol: df.copy()})  # 개별 indicators 사용
        ...

    # combined 전략 실행 (공유 indicators 사용)
    for name, entry, exit_ in combined_jobs:
        ...
        engine = BacktestEngine(strategy=strategy, config=config)
        report = engine.run({symbol: df.copy()}, precomputed_indicators=precomputed)
        ...
```

#### Step 2-4: CombinedStrategy indicators() 호환성 확인

`combined.py`의 `indicators()`:
- 모든 entry_filters + exit_filters의 `compute(df)` 호출
- 각 필터가 필요한 인디케이터 컬럼 추가
- union 전략에 모든 필터를 넣으면 모든 컬럼이 계산됨 ✅

주의: 필터의 `compute()`에 idempotency guard 있음 (`if col in df.columns: return df`)
→ 중복 필터가 있어도 안전 ✅

### 테스트 방법
- combine-scan 결과가 수정 전후 동일한지 비교 (top 10)
- precomputed_indicators=None일 때 기존 동작 유지 확인
- 등록된 전략 + combined 전략 혼합 batch 테스트

### 예상 효과
- combine-scan: 48회 indicator 계산 → 1회 (배치당 ~10~24초 절약)
- scan: 효과 없음 (전략마다 indicators가 다름)

---

## Change 3: pandas index lookup → dict/numpy

### 변경 파일: `src/tradingbot/backtest/engine.py`

#### Step 3-1: 타임스탬프 → 인덱스 dict 사전 구축

메인 루프 전 (라인 148 이후):
```python
# 현재:
symbol_ts_sets = {sym: set(df.index) for sym, df in symbol_data.items()}

# 추가:
symbol_ts_to_idx: dict[str, dict] = {
    sym: {ts: i for i, ts in enumerate(df.index)}
    for sym, df in symbol_data.items()
}
```

#### Step 3-2: 메인 루프 내 get_loc 대체

라인 160-166:
```python
# 현재:
if ts not in symbol_ts_sets[sym]:
    continue
df = symbol_data[sym]
idx = df.index.get_loc(ts)

# 변경:
idx = symbol_ts_to_idx[sym].get(ts)
if idx is None:
    continue
```

`symbol_ts_sets` 제거 가능 (dict.get으로 대체).

#### Step 3-3: Phase 3 close 가격 numpy 배열로 접근

라인 228-230:
```python
# 현재:
for sym in symbol_data:
    if ts in symbol_ts_sets[sym]:
        self._last_known_prices[sym] = float(symbol_data[sym].loc[ts, "close"])

# 변경 (사전 구축):
close_arrays = {sym: df["close"].values for sym, df in symbol_data.items()}

# 루프 내:
for sym in symbol_data:
    idx = symbol_ts_to_idx[sym].get(ts)
    if idx is not None:
        self._last_known_prices[sym] = float(close_arrays[sym][idx])
```

#### Step 3-4: Phase 1 fill_row도 numpy로 최적화

라인 172:
```python
# 현재:
fill_row = df.iloc[idx]

# 변경 (사전 구축):
ohlcv_arrays = {
    sym: {
        "open": df["open"].values,
        "high": df["high"].values,
        "low": df["low"].values,
        "close": df["close"].values,
        "volume": df["volume"].values,
    }
    for sym, df in symbol_data.items()
}

# 루프 내:
arrays = ohlcv_arrays[sym]
fill_candle = Candle(
    timestamp=ts.to_pydatetime(),
    open=float(arrays["open"][idx]),
    high=float(arrays["high"][idx]),
    low=float(arrays["low"][idx]),
    close=float(arrays["close"][idx]),
    volume=float(arrays["volume"][idx]),
)
```

### 테스트 방법
- 기존 180개 테스트 전부 통과
- 백테스트 결과 비트 단위 동일 확인

### 예상 효과
- 매 캔들 pandas 인덱스/Series 접근 → 순수 Python dict + numpy 배열
- 추가 2~3x 개선

---

## 구현 순서

```
1. Change 1 (copy 제거) → 테스트 → 벤치마크
2. Change 3 (dict/numpy) → 테스트 → 벤치마크
3. Change 2 (인디케이터 공유) → 테스트 → 벤치마크
```

Change 2보다 Change 3을 먼저 하는 이유:
- Change 3은 engine 내부 최적화라 외부 API 변경 없음
- Change 2는 `engine.run()` API 변경 + `_run_batch` 수정 필요 → 더 복잡

## 벤치마크 방법

각 Change 적용 후:
```bash
# 단일 백테스트 시간 측정
time tradingbot backtest --strategy sma_cross --symbol BTC/KRW --timeframe 1h

# scan 시간 측정
time tradingbot scan --top 10 --workers 8

# combine-scan 시간 측정 (subset으로 빠르게)
time tradingbot combine-scan --top 10 --workers 8
```
