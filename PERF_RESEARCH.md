# Backtest Engine Performance Optimization Research

## 현재 상황

- combine-scan (1152 조합 × 6년 데이터): **1.5시간 이상** 소요
- 목표: **3~5분**으로 단축 (20~40x 개선)

## 병목 분석

### 병목 1: O(N²) visible_df.copy() — 전체 시간의 ~50%

`engine.py` 라인 202:
```python
visible_df = indicator_data[sym].iloc[:idx].copy()
```

- 54k 캔들 루프에서 매 반복마다 증가하는 크기의 DataFrame 복사
- 복사량: 1+2+3+...+54k = **약 15억 셀**
- 메모리: 25 컬럼 × 8바이트 = **~292GB** 할당/해제 (per backtest)
- copy()는 순수 방어적 조치 — **현재 모든 전략/필터는 read-only**

### 병목 2: 인디케이터 중복 계산 — 배치 처리량 제한

`_run_batch`에서 48개 템플릿 각각 `engine.run()` 호출:
- 매번 `strategy.indicators(df.copy())` 실행
- 같은 (symbol, timeframe)에 대해 RSI/EMA/ATR 등 **48번 중복 계산**
- 인디케이터 계산: 54k행 × 20개 지표 ≈ 200~500ms/회 → 48회 = **10~24초** 낭비

### 병목 3: pandas index lookup 오버헤드

`engine.py` 라인 166:
```python
idx = df.index.get_loc(ts)
```
- 메인 루프에서 매 타임스탬프 × 매 심볼마다 호출
- pandas DatetimeIndex lookup은 Python 오버헤드 큼

## DataFrame 수정 감사 결과

모든 전략/필터의 `should_entry`, `should_exit`, `check_entry`, `check_exit` 메서드 확인:

| 메서드 | 읽기 패턴 | 쓰기 | 결론 |
|--------|-----------|------|------|
| 7개 전략 should_entry/exit | `.iloc[-1]`, `.iloc[-2]` | ❌ | read-only |
| 31개 필터 check_entry/exit | `.iloc[-1]`, 서브슬라이스 | ❌ | read-only |
| AtrTrailingExitFilter | `df["high"].iloc[entry_index:].max()` | ❌ | read-only |
| LgbmProbFilter | `df[FEATURE_COLS].iloc[[-1]]` | ❌ | read-only |
| indicators() / compute() | 전체 df | ✅ 컬럼 추가 | precompute 시에만 (copy 위에서) |

**결론: copy() 제거 안전. 추가 안전장치로 read-only flag 설정.**

## 구현 계획

### Change 1: copy() 제거 + read-only flag (8~15x)

**파일**: `engine.py`

```python
# 인디케이터 사전 계산 후, 메인 루프 전:
for sym in indicator_data:
    indicator_data[sym].values.flags.writeable = False

# 메인 루프 내 (라인 202):
# Before:
visible_df = indicator_data[sym].iloc[:idx].copy()
# After:
visible_df = indicator_data[sym].iloc[:idx]  # zero-copy view
```

- O(N²) 메모리 할당 → O(1)
- 54k 캔들 기준: 백테스트당 **30~120초 절약**
- 전략이 df 수정 시도하면 `ValueError: read-only` → 안전

**per-iteration path (supports_precompute=False)도 동일 적용:**
```python
# 라인 204:
visible_df = symbol_data[sym].iloc[:idx]  # copy 제거
visible_df = self.strategy.indicators(visible_df)  # indicators()는 새 컬럼 추가하므로 copy 필요할 수 있음
```
→ 이 경로는 MultiTimeframeStrategy만 사용. indicators()가 df를 수정하므로 copy 유지 또는 별도 처리.

### Change 2: 인디케이터 사전 계산 공유 (배치 처리량 10~20x)

**파일**: `parallel.py`, `engine.py`

1. `_run_batch`에서 모든 job의 필터를 union으로 합쳐 인디케이터 1회 계산
2. `engine.run()`에 `precomputed_indicators` 파라미터 추가
3. 전달받으면 `strategy.indicators()` 호출 스킵

```python
# parallel.py - _run_batch 내:
# 1) 모든 job에서 사용하는 필터를 합쳐 union strategy 생성
all_entry_filters, all_exit_filters = [], []
for name, entry, exit_ in jobs:
    if entry:
        all_entry_filters += parse_filter_string(entry, base_timeframe=timeframe)
        all_exit_filters += parse_filter_string(exit_, base_timeframe=timeframe)

if all_entry_filters:
    union_strategy = CombinedStrategy(entry_filters=all_entry_filters, exit_filters=all_exit_filters)
    union_strategy.symbols = [symbol]
    union_strategy.timeframe = timeframe
    precomputed = union_strategy.indicators(df.copy())
    precomputed.values.flags.writeable = False
else:
    precomputed = None

# 2) 각 job에 precomputed 주입
engine.run({symbol: df.copy()}, precomputed_indicators={symbol: precomputed})
```

```python
# engine.py - run() 메서드:
def run(self, data, precomputed_indicators=None):
    ...
    if precomputed_indicators and sym in precomputed_indicators:
        indicator_data[sym] = precomputed_indicators[sym]
    else:
        indicator_data[sym] = self.strategy.indicators(df.copy())
```

**효과**: 48개 템플릿이 인디케이터를 공유 → 계산 48회 → 1회

**주의사항**:
- 등록된 전략(sma_cross 등)은 각자 다른 indicators() 구현 → scan에서는 공유 불가
- combine-scan 전용 최적화 (scan은 전략마다 indicators가 다름)
- scan도 같은 전략 클래스끼리는 공유 가능하지만 효과 미미 (7개뿐)

### Change 3: pandas index lookup → dict/numpy (2~3x 추가)

**파일**: `engine.py`

```python
# 메인 루프 전:
symbol_ts_to_idx = {
    sym: {ts: i for i, ts in enumerate(df.index)}
    for sym, df in symbol_data.items()
}
# close 가격도 numpy로 추출
close_arrays = {
    sym: df["close"].values for sym, df in symbol_data.items()
}

# 메인 루프 내:
# Before:
idx = df.index.get_loc(ts)
# After:
idx = symbol_ts_to_idx[sym].get(ts)
if idx is None:
    continue

# Before:
current_price = symbol_data[sym].loc[ts, "close"]
# After:
current_price = close_arrays[sym][idx]
```

## 예상 결과

| 변경 | 단일 백테스트 속도 | combine-scan 전체 |
|------|-------------------|-------------------|
| 현재 | ~30초 (1h 기준) | ~1.5시간 |
| Change 1 (copy 제거) | ~3~5초 | ~15~25분 |
| Change 1+2 (+ 인디케이터 공유) | ~2~3초 | ~5~10분 |
| Change 1+2+3 (+ dict lookup) | ~1~2초 | **~3~5분** |

## 구현 순서

1. **Change 1** (copy 제거) — 가장 적은 변경으로 가장 큰 효과. 기존 테스트로 검증.
2. **Change 2** (인디케이터 공유) — `engine.run()` API 추가, `_run_batch` 수정.
3. **Change 3** (dict lookup) — engine 내부 최적화, API 변경 없음.

## 핵심 파일

| 파일 | 변경 내용 |
|------|----------|
| `engine.py` 202행 | copy() 제거, read-only flag |
| `engine.py` 166행 | get_loc → dict lookup |
| `engine.py` run() | precomputed_indicators 파라미터 |
| `parallel.py` _run_batch | union indicator 계산, 주입 |
| `strategy/base.py` | (변경 없음) |
