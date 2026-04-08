# Vectorized Screening Engine Research

## 목표

scan/combine-scan 전용 고속 백테스트 엔진 구현.
기존 candle-by-candle 엔진(정밀 백테스트, live/paper용)은 그대로 유지.

- **현재**: scan 168조합 ~10분, combine-scan 1152조합 ~1.5시간
- **목표**: scan ~30초, combine-scan ~2~3분

## 핵심 아이디어

기존 엔진은 매 캔들마다 `should_entry(visible_df)` / `should_exit(visible_df)` 호출 → **O(N)번의 Python 함수 호출**.
벡터화 엔진은 **한 번에 전체 DataFrame에 대한 boolean 배열**을 생성 → 거래 시뮬레이션도 배열 연산.

```
기존: for ts in timestamps → strategy.should_entry(df[:idx]) → True/False
벡터: entry_mask = filter1_bool & filter2_bool & ...  (단일 numpy 연산)
```

## 필터 벡터화 가능성 분석

### 현재 필터 인터페이스

```python
class BaseFilter(ABC):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """인디케이터 컬럼 추가 (이미 있으면 스킵)"""
    def check_entry(self, df: pd.DataFrame) -> bool:
        """df.iloc[-1] 기준 진입 조건 판단"""
    def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
        """df.iloc[-1] 기준 종료 조건 판단"""
```

### 벡터화 인터페이스 (신규)

```python
class BaseFilter(ABC):
    def compute(self, df): ...       # 기존 유지
    def check_entry(self, df): ...   # 기존 유지
    def check_exit(self, df): ...    # 기존 유지

    # 신규: 벡터화 메서드
    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        """전체 DataFrame에 대한 entry boolean 배열 반환"""
    def vectorized_exit(self, df: pd.DataFrame) -> pd.Series:
        """전체 DataFrame에 대한 exit boolean 배열 반환"""
```

### 31개 필터 벡터화 가능성

| # | 필터 | 룩백 패턴 | 벡터화 | 구현 방식 |
|---|------|----------|--------|----------|
| 1 | TrendUpFilter | iloc[-1] | O | `close > sma` (Series 비교) |
| 2 | TrendDownFilter | iloc[-1] | O | `close < sma` |
| 3 | AdxStrongFilter | iloc[-1] | O | `adx > threshold` |
| 4 | IchimokuAboveFilter | iloc[-1] | O | `close > max(ichi_a, ichi_b)` |
| 5 | AroonUpFilter | iloc[-1] | O | `aroon_up > threshold & aroon_up > aroon_down` |
| 6 | RsiOversoldFilter | iloc[-1,-2] (crossover) | O | `(prev <= threshold) & (curr > threshold)` via `.shift(1)` |
| 7 | RsiOverboughtFilter | iloc[-1] | O | `rsi > threshold` |
| 8 | MacdCrossUpFilter | iloc[-1,-2] | O | `(hist_prev < 0) & (hist_curr >= 0)` via `.shift(1)` |
| 9 | StochOversoldFilter | iloc[-1,-2] | O | `(k_prev < threshold) & (k_curr >= threshold)` |
| 10 | CciOversoldFilter | iloc[-1,-2] | O | `(cci_prev < -threshold) & (cci_curr >= -threshold)` |
| 11 | RocPositiveFilter | iloc[-1,-2] | O | `(roc_prev < 0) & (roc_curr >= 0)` |
| 12 | MfiOversoldFilter | iloc[-1,-2] | O | `(mfi_prev < threshold) & (mfi_curr >= threshold)` |
| 13 | PriceBreakoutFilter | iloc[-1,-2] | O | `close > recent_high.shift(1)` |
| 14 | EmaAboveFilter | iloc[-1] | O | `close > ema` |
| 15 | BbUpperBreakFilter | iloc[-1,-2] | O | `(close_prev <= upper_prev) & (close_curr > upper_curr)` |
| 16 | EmaCrossUpFilter | iloc[-1,-2] | O | `(fast_prev <= slow_prev) & (fast_curr > slow_curr)` |
| 17 | DonchianBreakFilter | iloc[-1,-2] | O | `close > dc_upper.shift(1)` |
| 18 | AtrBreakoutFilter | iloc[-1] | O | `close > ema + multiplier * atr` |
| 19 | KeltnerBreakFilter | iloc[-1] | O | `close > kc_upper` |
| 20 | BbSqueezeFilter | iloc[-1,-2] | O | `(bb_prev_inside_kc) & (bb_curr_outside_kc)` |
| 21 | BbBandwidthLowFilter | iloc[-1] | O | `bandwidth < threshold` |
| 22 | VolumeSpikeFilter | iloc[-1] | O | `vol_ratio > threshold` |
| 23 | ObvRisingFilter | iloc[-1] | O | `obv > obv_sma` |
| 24 | MfiConfirmFilter | iloc[-1] | O | `mfi > threshold` |
| 25 | StochOverboughtFilter | iloc[-1] | O | `stoch_k > threshold` |
| 26 | CciOverboughtFilter | iloc[-1] | O | `cci > threshold` |
| 27 | MfiOverboughtFilter | iloc[-1] | O | `mfi > threshold` |
| 28 | ZscoreExtremeFilter | iloc[-1] | O | `abs(zscore) > threshold` |
| 29 | PctFromMaExitFilter | iloc[-1] | O | `pct_from_ma > threshold` |
| 30 | AtrTrailingExitFilter | iloc[entry_index:] | **△** | entry_index 필요 → 거래 루프에서 처리 |
| 31 | LgbmProbFilter | iloc[-1] + 모델 상태 | **X** | ML 모델 호출 필요 → fallback |

**결론: 29/31 완전 벡터화, 1개 부분 벡터화, 1개 불가 (fallback)**

## 벡터화 백테스트 시뮬레이션 설계

### Step 1: 인디케이터 계산 (1회)

기존 `CombinedStrategy.indicators(df)` 재사용. 이미 `_run_batch`에서 union으로 1회 계산.

### Step 2: 시그널 배열 생성

```python
# Entry: AND 조건
entry_mask = filter1.vectorized_entry(df) & filter2.vectorized_entry(df) & ...

# Exit: OR 조건
exit_mask = filter1.vectorized_exit(df) | filter2.vectorized_exit(df) | ...
```

결과: 54,000행 DataFrame에 대해 `entry_mask[i]`, `exit_mask[i]` boolean 배열 2개.

### Step 3: 거래 추출 (trade extraction)

시그널 배열에서 거래 쌍을 추출하는 로직:

```python
def extract_trades(entry_mask, exit_mask, open_prices, low_prices, stop_loss_pct, fee_rate, slippage_pct):
    """
    규칙:
    1. entry_mask[i] = True → 다음 캔들(i+1)의 open에 매수
    2. 포지션 보유 중 exit_mask[j] = True → 다음 캔들(j+1)의 open에 매도
    3. 포지션 보유 중 low[k] <= stop_loss → stop_loss 가격에 매도
    4. 동시에 exit + entry → exit 먼저, 같은 캔들에 재진입 안 함
    5. 마지막까지 미청산 → 마지막 캔들 close에 청산
    """
    trades = []
    in_position = False
    entry_price = 0.0
    entry_idx = 0

    for i in range(len(entry_mask)):
        if in_position:
            # Stop loss 체크 (현재 캔들의 low)
            sl_price = entry_price * (1 - stop_loss_pct)
            if low_prices[i] <= sl_price:
                exit_price = sl_price * (1 - slippage_pct)
                trades.append((entry_idx, i, entry_price, exit_price))
                in_position = False
                continue

            # Exit signal 체크
            if exit_mask[i]:
                # 다음 캔들 open에 매도
                if i + 1 < len(open_prices):
                    exit_price = open_prices[i + 1] * (1 - slippage_pct)
                    trades.append((entry_idx, i + 1, entry_price, exit_price))
                    in_position = False
                continue
        else:
            if entry_mask[i]:
                # 다음 캔들 open에 매수
                if i + 1 < len(open_prices):
                    entry_price = open_prices[i + 1] * (1 + slippage_pct)
                    entry_idx = i + 1
                    in_position = True

    return trades
```

**이 루프는 O(N)이지만 순수 Python이므로 Numba `@njit`으로 가속 가능**.

### Step 4: 메트릭 계산

```python
# 거래 결과에서 직접 계산
pnl_per_trade = [(exit_p - entry_p) * qty - fees for ...]
equity_curve = initial_balance + cumsum(pnl_per_trade)  # 거래 기준
sharpe = annualized(mean(returns) / std(returns))
max_dd = max((peak - equity) / peak)
win_rate = sum(pnl > 0) / len(trades)
profit_factor = sum(wins) / abs(sum(losses))
```

## 성능 예측

### 현재 (candle-by-candle)

| 단계 | 시간 (54k 캔들, 1h, 6년) |
|------|--------------------------|
| 인디케이터 계산 | ~200ms (1회, 공유) |
| 메인 루프 (per template) | ~11초 |
| **48 템플릿 합계** | ~11 × 48 = **~530초** (per batch) |
| **combine-scan (24 batch × 8 workers)** | ~530 × 3 = **~26분** |

### 벡터화 (신규)

| 단계 | 시간 (54k 캔들) |
|------|-----------------|
| 인디케이터 계산 | ~200ms (1회, 공유) |
| 시그널 배열 생성 (per template) | ~1~2ms (numpy 비교 연산) |
| 거래 추출 + 메트릭 (per template) | ~5~10ms (O(N) 루프 or njit) |
| **48 템플릿 합계** | ~200ms + 48 × 10ms = **~0.7초** |
| **combine-scan (24 batch × 8 workers)** | ~0.7 × 3 = **~2초** + 오버헤드 |

**예상: combine-scan ~1.5시간 → ~30초~1분 (100x 이상 개선)**

## 정밀도 차이

벡터화 엔진은 스크리닝 전용이므로 일부 단순화 허용:

| 항목 | 기존 엔진 | 벡터화 엔진 | 차이 |
|------|----------|------------|------|
| 포지션 사이징 | fixed-fractional, ML strength | 고정 비율 (equity의 10%) | 약간 |
| 스톱 로스 | stop_loss 가격에 매도 | 동일 | 없음 |
| 슬리피지/수수료 | 동일 | 동일 | 없음 |
| 멀티 심볼 | 지원 | 단일 심볼 (scan용) | scan은 단일 심볼 |
| Equity 추적 | 매 캔들 스냅샷 | 거래 시점만 | Sharpe 약간 차이 |
| AtrTrailingExit | entry_index 기반 max(high) | 거래 루프에서 처리 | 없음 |
| ML veto (LgbmProb) | 지원 | **미지원 → 기존 엔진 fallback** | ML 템플릿만 |
| Anti-lookahead | visible_df[:idx] | 시그널 자체가 현재/이전 바만 사용 | 동일 |

**상위 N개 결과는 기존 `_run_batch` (→ `engine.run()`)로 재검증 → 정밀도 보장**

## AtrTrailingExitFilter 처리

벡터화에서 유일한 까다로운 필터. entry_index에 따라 trailing max가 달라짐.

**해결 방법**: 거래 추출 루프(Step 3)에서 처리:
```python
if atr_trailing_used:
    # 포지션 보유 중 매 캔들에서:
    highest_since_entry = max(high[entry_idx:current_idx+1])
    trailing_stop = highest_since_entry - multiplier * atr[current_idx]
    if low[current_idx] <= trailing_stop:
        exit at trailing_stop
```

이미 O(N) 루프이므로 추가 비용 미미. expanding max를 유지하면 O(1) per iteration.

## LgbmProbFilter 처리

ML 필터가 포함된 템플릿(15/48)은 벡터화 불가 → **기존 엔진으로 fallback**.

```python
def is_vectorizable(template):
    return "lgbm_prob" not in template["entry"]

vectorizable = [t for t in COMBINE_TEMPLATES if is_vectorizable(t)]  # 33개
fallback = [t for t in COMBINE_TEMPLATES if not is_vectorizable(t)]   # 15개
```

ML 템플릿은 기존 `_run_batch` 경로로 처리. 성능상 문제없음:
- ML AUC가 낮아 실전 사용 안 하는 중
- 15개 × 기존 속도 = ~3분 (허용 범위)

## 구현 범위

### 신규 파일

```
src/tradingbot/backtest/vectorized.py  — VectorizedEngine 클래스
```

### 수정 파일

```
src/tradingbot/strategy/filters/base.py      — vectorized_entry/exit 기본 구현 추가
src/tradingbot/strategy/filters/trend.py      — vectorized_entry/exit 구현
src/tradingbot/strategy/filters/momentum.py   — vectorized_entry/exit 구현
src/tradingbot/strategy/filters/price.py      — vectorized_entry/exit 구현
src/tradingbot/strategy/filters/volatility.py — vectorized_entry/exit 구현
src/tradingbot/strategy/filters/volume.py     — vectorized_entry/exit 구현
src/tradingbot/strategy/filters/exit.py       — vectorized_entry/exit 구현
src/tradingbot/backtest/parallel.py           — _run_batch에서 벡터화 엔진 분기
src/tradingbot/cli.py                         — scan/combine-scan에서 벡터화 사용
```

### 변경하지 않는 파일

```
src/tradingbot/backtest/engine.py  — 기존 정밀 엔진 (live/paper/backtest CLI용)
src/tradingbot/strategy/filters/ml.py  — fallback 경로로 처리
```

## Equity Curve 계산 방식

벡터화 엔진은 **scan/combine-scan 전용**. live/paper 트레이딩은 `live/engine.py`(실시간 폴링 루프)가 담당하며 백테스트 엔진을 사용하지 않음. 벡터화 변경이 live/paper에 영향 없음.

### 기존 엔진: 매 캔들 스냅샷
```
equity[t] = cash + position_value[t]  (54,000개 포인트)
```

### 벡터화 엔진: 거래 기반 스냅샷
```
equity[trade_exit_t] = equity[trade_entry_t] + pnl  (거래 수만큼 포인트)
```

Sharpe 계산 시 차이 발생 가능 (거래 없는 기간의 수익률이 빠짐).
**해결**: 거래 기반 equity를 전체 타임라인으로 ffill하여 full equity curve 생성.

```python
# 거래 발생 시점에 equity 기록 → 나머지는 forward fill
equity_series = pd.Series(index=df.index, dtype=float)
equity_series.iloc[0] = initial_balance
for entry_idx, exit_idx, pnl in trades:
    equity_series.iloc[exit_idx] = current_equity
equity_series = equity_series.ffill()
```

## 등록된 전략(scan) 처리

scan은 등록된 전략(sma_cross, bollinger 등)을 사용. 이들은 `should_entry()`/`should_exit()` 인터페이스.

**방법 A**: 등록된 전략에도 vectorized 인터페이스 추가 (작업량 큼, 전략 7개)
**방법 B**: 등록된 전략은 기존 엔진 그대로, combined 템플릿만 벡터화

→ **방법 B 선택**: scan은 168조합(7전략 × 8심볼 × 3TF) × ~11초 = 이미 최적화된 엔진으로 ~10분.
  combine-scan이 1152조합으로 병목. combine-scan만 벡터화해도 충분.

## Numba 사용 여부

거래 추출 루프(Step 3)는 O(N) Python 루프. 옵션:

- **순수 Python**: 54k 캔들 루프 ~5~10ms. 48 템플릿 × 10ms = ~0.5초. 충분히 빠름.
- **Numba @njit**: ~0.5~1ms로 단축 가능. 하지만 초기 JIT 컴파일 ~2초, 의존성 추가.

→ **순수 Python으로 시작, 필요 시 Numba 추가**. numpy 배열 연산이 대부분이므로 Python 루프는 전체의 ~10%.

## 핵심 리스크

1. **벡터화 시그널 ≠ 기존 시그널**: 일부 필터의 미묘한 edge case (NaN 처리, 첫 N행 warmup)
   - 해결: 기존 엔진으로 상위 N개 재검증
2. **AtrTrailingExit 정확도**: expanding max 계산이 기존 엔진과 다를 수 있음
   - 해결: 거래 루프에서 정확히 구현
3. **포지션 사이징 무시**: 벡터화는 고정 비율 → 실제 수익률/Sharpe와 차이
   - 해결: 스크리닝 용도이므로 순위(ranking)만 정확하면 됨
