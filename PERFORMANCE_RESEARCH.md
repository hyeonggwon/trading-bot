# 백테스트 엔진 성능 최적화 리서치

## 문제

ML 필터(lgbm_prob) 포함 combine 백테스트 시 8751 캔들에 10분+ 소요.
전체 기간(54000+ 캔들)은 사실상 실행 불가.

## 현재 엔진 구조 (engine.py)

### 메인 루프 (3 Phase)

```
초기화:
  symbol_data = {sym: df.copy()} per symbol
  all_timestamps = sorted(union of all symbol timestamps)
  original_columns = {sym: set(df.columns)}

for ts in all_timestamps:

  Phase 1 — 인덱스 업데이트 + 스탑로스 + 주문 체결 (line 129-157):
    for sym in symbol_data:
      idx = df.index.get_loc(ts)          # O(log N)
      symbol_indices[sym] = idx
      fill_candle = Candle(df.iloc[idx])   # 현재 캔들
      _check_stop_losses(sym, fill_candle)
      _process_pending_orders(fill_candle, sym)

  Phase 2 — 전략 평가 (line 159-200):
    for sym in symbol_data:
      idx = symbol_indices[sym]
      visible_df = df.iloc[:idx].copy()       # ← 병목 1: 매번 복사
      visible_df = strategy.indicators(visible_df)  # ← 병목 2: 매번 재계산
      assert columns == original_columns[sym]  # anti-lookahead 검증
      fill_candle = Candle(df.iloc[idx])       # ← 병목 3: Phase 1과 중복
      should_exit(visible_df, sym, position)
      should_entry(visible_df, sym)

  Phase 3 — 가격 업데이트 + 에쿼티 기록 (line 202-210):
    for sym in symbol_data:
      _last_known_prices[sym] = df.loc[ts, "close"]
    equity = _calculate_equity(prices)         # ← 병목 6: 매 틱
    equity_snapshots.append((ts, equity))
```

## 병목 분석

### 병목 1: indicators() 매 캔들 재계산 — O(N²) [CRITICAL]

**위치**: engine.py line 171
```python
visible_df = self.strategy.indicators(visible_df)
```

**비용**: 캔들 idx에서 idx개 행에 인디케이터 재계산.
- ML 필터: build_feature_matrix() → 19개 ta 인디케이터 + 36개 파생 피처
- 총 작업량: Σ(i=1..N) × indicator_cost(i) ≈ O(N²)
- 8751 캔들 × 평균 4375행 × 인디케이터 = 수천만 연산

**안전성**: 인디케이터는 과거 데이터만 사용 (shift, rolling, cumsum 등).
row i의 인디케이터 값은 row 0..i에만 의존. 전체 df에서 한 번 계산해도 동일한 결과.

**해결**: 루프 전에 전체 df에 인디케이터 사전 계산.
```python
# 루프 전
indicator_data = {}
for sym, df in symbol_data.items():
    indicator_data[sym] = self.strategy.indicators(df.copy())

# 루프 내 Phase 2
visible_df = indicator_data[sym].iloc[:idx]  # 슬라이싱만, copy 불필요
```

### 병목 2: .copy() 매 캔들 — GB 수준 메모리 할당 [HIGH]

**위치**: engine.py line 170
```python
visible_df = df.iloc[:idx].copy()
```

**비용**: 매 캔들마다 growing DataFrame 딥카피.
- 캔들 8750에서: 8750행 × 50컬럼 × 8바이트 ≈ 3.5MB per copy
- 전체: Σ(i=1..8751) × row_size ≈ 수 GB GC 압력

**이유**: indicators()가 df를 변경(컬럼 추가)하므로 원본 보호 필요.

**해결**: 인디케이터 사전 계산 후에는 copy 불필요. `df.iloc[:idx]`는 view 반환.

### 병목 3: fill_candle 이중 생성 [LOW]

**위치**:
- Phase 1 (line 142-150): fill_candle 생성
- Phase 2 (line 178-186): 동일한 fill_candle 재생성

**해결**: Phase 1에서 생성한 fill_candle을 dict에 캐시, Phase 2에서 재사용.
```python
fill_candles: dict[str, Candle] = {}
# Phase 1
fill_candles[sym] = fill_candle
# Phase 2
fill_candle = fill_candles[sym]
```

### 병목 4: get_indexer() exit마다 호출 [MEDIUM]

**위치**: combined.py line 93
```python
idx = df.index.get_indexer([position.entry_time], method="ffill")[0]
```

**비용**: 매 캔들 × 오픈 포지션마다 get_indexer 호출.
포지션이 길면 수천 번 동일한 entry_time 검색.

**해결 방안**:
- CombinedStrategy에서 `_entry_indices: dict[str, int]` 관리
- should_entry()에서 entry 시 인덱스 저장
- should_exit()에서 저장된 인덱스 사용

**주의**: visible_df의 인덱스가 매번 바뀌므로 (길이 증가),
entry_time의 절대 인덱스(integer offset)는 고정. iloc 기준으로 저장하면 OK.

### 병목 5: `ts in df.index` 3중 체크 [LOW]

**위치**: engine.py line 133, 163, 204

**비용**: DatetimeIndex 해싱 × 3 × 심볼 수 × 타임스탬프 수

**해결**: 루프 전 set 사전 생성.
```python
symbol_ts_sets = {sym: set(df.index) for sym, df in symbol_data.items()}
# 루프 내
if ts not in symbol_ts_sets[sym]: continue
```

### 병목 6: _calculate_equity() 매 틱 [LOW]

**위치**: engine.py line 208
```python
equity = self._calculate_equity(self._last_known_prices)
```

**현재 구현** (line 411-416):
```python
def _calculate_equity(self, prices):
    position_value = sum(
        prices.get(p.symbol, p.entry_price) * p.size
        for p in self.positions.values()
    )
    return self.cash + position_value
```

**비용**: 매 틱마다 모든 오픈 포지션 순회. 포지션이 적으면 거의 무시 가능.

**해결**: running total 관리.
- 포지션 오픈/클로즈 시에만 position_value 갱신
- 가격 변동 시에만 해당 포지션의 value 업데이트
- 복잡도 증가 대비 이득 적음 → 낮은 우선순위

### 병목 7: indicators() 키 정렬 매 캔들 [LOW]

**위치**: combined.py line 49
```python
key = (f.__class__.__name__, tuple(sorted(f.params.items())))
```

**비용**: 필터 수 × 캔들 수만큼 sort 호출.

**해결**: `__init__()`에서 중복 제거된 필터 리스트 사전 생성.
```python
# __init__에서
self._unique_filters = self._deduplicate_filters(entry_filters + exit_filters)

# indicators()에서
for f in self._unique_filters:
    df = f.compute(df)
```

## anti-lookahead 안전성 검증

### 현재 메커니즘
1. `visible_df = df.iloc[:idx].copy()` — 과거 캔들만 복사
2. `strategy.indicators(visible_df)` — 복사본에 인디케이터 추가
3. `assert columns == original_columns` — 원본 변경 없음 확인

### 사전 계산 후
1. `indicator_df = strategy.indicators(df.copy())` — 전체 df에 한 번 계산
2. `visible_df = indicator_df.iloc[:idx]` — view로 과거만 슬라이싱

**안전한 이유**:
- 모든 인디케이터 함수 (SMA, EMA, RSI, MACD, BB, ATR 등)는 `rolling()`, `shift()`, `cumsum()` 등 과거 참조 연산만 사용
- row i의 값은 row 0..i에만 의존 → 전체 계산 vs 부분 계산 결과 동일
- `visible_df.iloc[:idx]`가 미래 차단 (strategy는 idx 이후 행 접근 불가)
- build_feature_matrix()의 파생 피처도 동일 (pct_change, diff, rolling 등)

**assertion 제거 가능**: 원본 df를 더 이상 indicators()에 전달하지 않으므로,
`original_columns` 체크는 불필요. 하지만 안전장치로 남겨둬도 됨.

## 기존 코드 핵심 위치 (변경 대상)

| 파일 | 라인 | 내용 |
|------|------|------|
| `engine.py` | 79-92 | symbol_data 구축 |
| `engine.py` | 119-123 | symbol_indices, original_columns 초기화 |
| `engine.py` | 142-150 | Phase 1 fill_candle 생성 |
| `engine.py` | 170-171 | **visible_df copy + indicators** |
| `engine.py` | 174-176 | anti-lookahead assertion |
| `engine.py` | 178-186 | Phase 2 fill_candle 재생성 |
| `engine.py` | 202-210 | Phase 3 가격/에쿼티 |
| `engine.py` | 411-416 | _calculate_equity() |
| `combined.py` | 45-53 | indicators() 중복 제거 |
| `combined.py` | 91-97 | should_exit() get_indexer |
| `core/models.py` | 141-164 | Position dataclass |

## 예상 효과

| 변경 | Before | After |
|------|--------|-------|
| indicators 사전 계산 | O(N²) | O(N) |
| copy 제거 | ~GB GC | ~0 |
| fill_candle 캐시 | 2× 생성 | 1× |
| entry_index 캐시 | O(N) per exit | O(1) |
| ts set 사전 생성 | hash×3 | hash×1 |

**총 예상**: 8751 캔들 ML 백테스트 10분+ → **15초 이내**
