# 백테스트 엔진 성능 최적화 구현 계획

## 개요

7개 병목을 해결하여 ML 백테스트를 10분+ → 15초 이내로 개선한다.
변경 범위: engine.py (핵심), combined.py, core/models.py

## Phase 1: indicators 사전 계산 + copy 제거 [병목 1+2]

가장 큰 임팩트. O(N²) → O(N).

### 1-1. engine.py — 인디케이터 사전 계산 추가

루프 직전(line 125 부근), symbol_data 준비 후에 인디케이터를 한 번만 계산:

```python
# Pre-compute indicators on full DataFrame per symbol (O(N) total)
indicator_data: dict[str, pd.DataFrame] = {}
for sym, df in symbol_data.items():
    indicator_data[sym] = self.strategy.indicators(df.copy())
```

- `df.copy()`로 원본 symbol_data 보호 (fill_candle에서 원본 OHLCV 필요)
- strategy.indicators()는 모든 필터의 compute() 호출 (기존과 동일)
- 결과를 indicator_data에 저장

### 1-2. engine.py Phase 2 — visible_df 변경

기존 (line 170-171):
```python
visible_df = df.iloc[:idx].copy()
visible_df = self.strategy.indicators(visible_df)
```

변경:
```python
visible_df = indicator_data[sym].iloc[:idx]
```

- copy 제거: iloc[:idx]는 view 반환, 변경 없으므로 안전
- indicators() 호출 제거: 이미 계산됨
- anti-lookahead 유지: iloc[:idx]가 미래 캔들 차단

### 1-3. anti-lookahead assertion 수정

기존 (line 174-176):
```python
assert set(df.columns) == original_columns[sym], (
    f"Anti-lookahead violation on {sym}: indicator columns leaked"
)
```

변경: 제거 또는 주석 처리.
- 원본 symbol_data[sym]을 더 이상 indicators()에 전달하지 않으므로 의미 없음
- indicator_data는 별도 dict이라 원본 오염 불가
- `original_columns` 변수도 제거

## Phase 2: fill_candle 캐시 [병목 3]

### 2-1. engine.py — Phase 1에서 생성한 fill_candle 재사용

Phase 1 시작 전에 캐시 dict 선언:
```python
fill_candles: dict[str, Candle] = {}
```

Phase 1 (line 142-150) 후에 캐시:
```python
fill_candles[sym] = fill_candle
```

Phase 2 (line 178-186) 교체:
```python
fill_candle = fill_candles[sym]
```

## Phase 3: entry_index 캐시 [병목 4]

### 3-1. combined.py — _entry_indices dict 추가

```python
class CombinedStrategy(Strategy):
    def __init__(self, ...):
        ...
        self._entry_indices: dict[str, int] = {}
```

### 3-2. should_entry()에서 entry 시점 인덱스 저장

should_entry()에서 Signal 반환 직전:
```python
# 현재 visible_df의 마지막 인덱스 = entry 캔들 위치
self._entry_indices[symbol] = len(df) - 1
```

### 3-3. should_exit()에서 get_indexer 대신 캐시 사용

기존 (line 90-97):
```python
entry_index = None
if position and position.entry_time:
    try:
        idx = df.index.get_indexer([position.entry_time], method="ffill")[0]
        if idx >= 0:
            entry_index = idx
    except (KeyError, IndexError) as e:
        log.warning(...)
```

변경:
```python
entry_index = self._entry_indices.get(symbol)
```

### 3-4. 포지션 청산 시 캐시 제거

should_exit()에서 exit Signal 반환 시:
```python
self._entry_indices.pop(symbol, None)
```

## Phase 4: timestamp set 사전 생성 [병목 5]

### 4-1. engine.py — 루프 전에 set 생성

```python
symbol_ts_sets: dict[str, set] = {
    sym: set(df.index) for sym, df in symbol_data.items()
}
```

### 4-2. 3개 Phase의 `ts in df.index` → `ts in symbol_ts_sets[sym]`

Phase 1 (line 132): `if ts not in symbol_ts_sets[sym]: continue`
Phase 2 (line 162): `if ts not in symbol_ts_sets[sym]: continue`
Phase 3 (line 205): `if ts in symbol_ts_sets[sym]:`

## Phase 5: _calculate_equity 최적화 [병목 6]

### 5-1. engine.py — 포지션 변경 시에만 value 갱신

`_position_value` 추적 변수 추가:
```python
# __init__
self._position_value: float = 0.0
```

_execute_buy()에서:
```python
self._position_value += fill_price * order.quantity
```

_execute_sell()에서:
```python
self._position_value -= self._last_known_prices.get(order.symbol, fill_price) * order.quantity
```

Phase 3 가격 업데이트 시:
```python
# 가격 변동에 따른 position_value 갱신
if sym in self.positions:
    pos = self.positions[sym]
    old_price = self._last_known_prices.get(sym, pos.entry_price)
    new_price = float(df.loc[ts, "close"])
    self._position_value += (new_price - old_price) * pos.size
    self._last_known_prices[sym] = new_price
```

_calculate_equity() 간소화:
```python
def _calculate_equity(self, prices: dict[str, float]) -> float:
    return self.cash + self._position_value
```

**주의**: _handle_signal()에서도 _calculate_equity를 호출하므로 (line 269),
해당 시점의 _position_value가 정확해야 함. 신중하게 구현.

→ 복잡도 대비 이득이 적으면 스킵 가능. 포지션 수가 적어서 현재 구현으로도 충분할 수 있음.

## Phase 6: indicators() 키 사전 생성 [병목 7]

### 6-1. combined.py — __init__에서 중복 제거

```python
def __init__(self, entry_filters=None, exit_filters=None):
    ...
    self._unique_filters = self._deduplicate_filters()

def _deduplicate_filters(self) -> list[BaseFilter]:
    seen: set[tuple] = set()
    unique = []
    for f in self.entry_filters + self.exit_filters:
        key = (f.__class__.__name__, tuple(sorted(f.params.items())))
        if key not in seen:
            unique.append(f)
            seen.add(key)
    return unique
```

### 6-2. indicators()에서 사전 생성된 리스트 사용

```python
def indicators(self, df):
    for f in self._unique_filters:
        df = f.compute(df)
    return df
```

## 파일별 변경 요약

| 순서 | 파일 | 작업 |
|------|------|------|
| 1 | `backtest/engine.py` | indicator 사전 계산, copy 제거, fill_candle 캐시, ts set, (equity 최적화) |
| 2 | `strategy/combined.py` | entry_index 캐시, 필터 중복 제거 사전 계산 |
| 3 | `tests/` | 기존 158 테스트 통과 확인 + 성능 검증 |

## 구현 순서

1. Phase 1 (indicators + copy) — 가장 큰 효과, 반드시 먼저
2. Phase 6 (키 사전 생성) — Phase 1과 함께 적용하면 깔끔
3. Phase 2 (fill_candle 캐시) — 단순, 즉시 적용
4. Phase 4 (ts set) — 단순, 즉시 적용
5. Phase 3 (entry_index 캐시) — 로직 변경 있음, 신중하게
6. Phase 5 (equity) — 가장 복잡, 이득 적으면 스킵

## 검증 기준

1. 기존 158 테스트 전부 통과
2. ML combine 백테스트 결과가 이전과 동일 (동일 trades, 동일 PnL)
3. 8751 캔들 BTC combine 백테스트 < 30초
4. 54000+ 캔들 전체 기간 백테스트 실행 가능
5. combine-scan 36템플릿 × 8심볼 실행 가능

## 리스크

- **anti-lookahead 위반**: indicators 사전 계산이 안전한지 재확인 필요
  - 모든 ta 인디케이터는 과거 참조만 (rolling, shift, cumsum)
  - visible_df = indicator_df.iloc[:idx]로 미래 차단
  - ✅ 안전 확인됨

- **view vs copy**: iloc[:idx] 반환값이 view인데, strategy가 df를 변경하면?
  - should_entry(), should_exit()는 읽기만 수행 (마지막 행 check)
  - check_entry(), check_exit()도 읽기만
  - ✅ 안전

- **_position_value 정합성**: buy/sell/stop_loss 모든 경로에서 정확히 갱신해야 함
  - force_close_position도 포함
  - 버그 여지 있음 → Phase 5는 선택적
