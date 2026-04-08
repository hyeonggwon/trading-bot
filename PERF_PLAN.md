# Vectorized Screening Engine Implementation Plan

## 목표
combine-scan ~1.5시간 → ~1분 (100x 개선)
scan은 기존 엔진 유지 (등록된 전략 7개, 벡터화 대상 아님)

---

## Step 1: BaseFilter에 vectorized 메서드 추가

### 파일: `src/tradingbot/strategy/filters/base.py`

```python
class BaseFilter(ABC):
    # 기존 메서드 유지
    def compute(self, df): ...
    def check_entry(self, df): ...
    def check_exit(self, df, entry_index=None): ...

    # 신규: 벡터화 메서드 (기본 구현)
    def vectorized_entry(self, df: pd.DataFrame) -> pd.Series:
        """전체 DataFrame에 대한 entry boolean Series 반환.
        기본 구현은 NotImplementedError — 벡터화 가능한 필터만 override."""
        raise NotImplementedError

    def vectorized_exit(self, df: pd.DataFrame) -> pd.Series:
        """전체 DataFrame에 대한 exit boolean Series 반환."""
        raise NotImplementedError

    @property
    def supports_vectorized(self) -> bool:
        """이 필터가 벡터화를 지원하는지 여부."""
        return False
```

---

## Step 2: 29개 필터에 vectorized_entry/exit 구현

### 패턴별 구현 방식

**패턴 A: 단일 바 비교 (iloc[-1] 패턴)**
```python
# 예: RsiOverboughtFilter
def vectorized_exit(self, df):
    col = f"rsi_{self.period}"
    return df[col] >= self.threshold
```

**패턴 B: 크로스오버 (iloc[-1,-2] 패턴)**
```python
# 예: RsiOversoldFilter
def vectorized_entry(self, df):
    col = f"rsi_{self.period}"
    prev = df[col].shift(1)
    curr = df[col]
    return (prev <= self.threshold) & (curr > self.threshold)
```

**패턴 C: 두 시리즈 크로스 (EmaCrossUp 등)**
```python
# 예: EmaCrossUpFilter
def vectorized_entry(self, df):
    fast_prev = df[f"ema_{self.fast}"].shift(1)
    fast_curr = df[f"ema_{self.fast}"]
    slow_prev = df[f"ema_{self.slow}"].shift(1)
    slow_curr = df[f"ema_{self.slow}"]
    return (fast_prev <= slow_prev) & (fast_curr > slow_curr)
```

### 파일별 변경

#### `filters/trend.py` (5개)
| 필터 | 패턴 | vectorized_entry | vectorized_exit |
|------|------|-----------------|----------------|
| TrendUpFilter | A | `close > sma` | `close < sma` |
| TrendDownFilter | A | `close < sma` | `close > sma` |
| AdxStrongFilter | A | `adx > threshold` | `False` |
| IchimokuAboveFilter | A | `close > max(ichi_a, ichi_b)` | `close < min(ichi_a, ichi_b)` |
| AroonUpFilter | A | `aroon_up > threshold & aroon_up > aroon_down` | `False` |

#### `filters/momentum.py` (7개)
| 필터 | 패턴 | vectorized_entry | vectorized_exit |
|------|------|-----------------|----------------|
| RsiOversoldFilter | B | `prev <= th & curr > th` | `False` |
| RsiOverboughtFilter | A | `False` | `rsi >= threshold` |
| MacdCrossUpFilter | B | `hist_prev < 0 & hist_curr >= 0` | `hist_prev >= 0 & hist_curr < 0` |
| StochOversoldFilter | B | `k_prev < th & k_curr >= th` | `False` |
| CciOversoldFilter | B | `cci_prev < -th & cci_curr >= -th` | `False` |
| RocPositiveFilter | B | `roc_prev < 0 & roc_curr >= 0` | `False` |
| MfiOversoldFilter | B | `mfi_prev < th & mfi_curr >= th` | `False` |

#### `filters/price.py` (5개)
| 필터 | 패턴 | vectorized_entry | vectorized_exit |
|------|------|-----------------|----------------|
| PriceBreakoutFilter | B | `close > recent_high.shift(1)` | `False` |
| EmaAboveFilter | A | `close > ema` | `close < ema` |
| BbUpperBreakFilter | B | `prev <= upper_prev & curr > upper_curr` | `close < middle` |
| EmaCrossUpFilter | C | `fast crosses above slow` | `False` |
| DonchianBreakFilter | B | `close > dc_upper.shift(1)` | `close < dc_middle` |

#### `filters/volatility.py` (4개)
| 필터 | 패턴 | vectorized_entry | vectorized_exit |
|------|------|-----------------|----------------|
| AtrBreakoutFilter | A | `close > ema + mult * atr` | `close < ema - mult * atr` |
| KeltnerBreakFilter | A | `close > kc_upper` | `close < kc_lower` |
| BbSqueezeFilter | B | `bb was inside kc & now outside` | `False` |
| BbBandwidthLowFilter | A | `bandwidth < threshold` | `False` |

#### `filters/volume.py` (3개)
| 필터 | 패턴 | vectorized_entry | vectorized_exit |
|------|------|-----------------|----------------|
| VolumeSpikeFilter | A | `vol_ratio > threshold` | `False` |
| ObvRisingFilter | A | `obv > obv_sma` | `obv < obv_sma` |
| MfiConfirmFilter | A | `mfi > threshold` | `mfi < threshold` |

#### `filters/exit.py` (5개, AtrTrailingExit 제외)
| 필터 | 패턴 | vectorized_entry | vectorized_exit |
|------|------|-----------------|----------------|
| StochOverboughtFilter | A | `False` | `stoch_k >= threshold` |
| CciOverboughtFilter | A | `False` | `cci > threshold` |
| MfiOverboughtFilter | A | `False` | `mfi >= threshold` |
| ZscoreExtremeFilter | A | `False` | `zscore > threshold` |
| PctFromMaExitFilter | A | `False` | `pct_from_ma > threshold` |

#### NaN 처리
모든 vectorized 메서드에서 NaN은 자동으로 False 처리:
```python
# pandas 비교 연산에서 NaN은 False 반환 → 별도 처리 불필요
# 단, shift(1)의 첫 행은 NaN → 크로스오버 패턴에서 첫 행은 자동으로 False
```

### vectorized_entry/exit 미추가 (2개) — 별도 경로로 처리
- **AtrTrailingExitFilter**: exit 판단에 진입 시점(entry_index) 필요 → 단일 boolean 배열 불가 → `_extract_trades` 루프에서 직접 처리 (Step 3-4)
- **LgbmProbFilter**: ML 모델 로드 + predict() + last_strength 상태 → 벡터화 불가 → 이 필터 포함 템플릿(15개)은 기존 엔진 fallback

---

## Step 3: VectorizedEngine 구현

### 신규 파일: `src/tradingbot/backtest/vectorized.py`

```python
@dataclass
class VectorizedResult:
    """벡터화 백테스트 결과 (ScanResult와 호환)."""
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int


def vectorized_backtest(
    df: pd.DataFrame,
    entry_filters: list[BaseFilter],
    exit_filters: list[BaseFilter],
    initial_balance: float = 10_000_000,
    fee_rate: float = 0.0005,
    slippage_pct: float = 0.001,
    stop_loss_pct: float = 0.02,
    max_position_pct: float = 0.10,
    timeframe: str = "1h",
) -> VectorizedResult:
```

### 내부 흐름

#### 3-1: 시그널 배열 생성

```python
# Entry: AND 조건 (exit role 필터 제외)
entry_mask = pd.Series(True, index=df.index)
for f in entry_filters:
    if f.role == "exit":
        continue
    entry_mask &= f.vectorized_entry(df)

# Exit: OR 조건
exit_mask = pd.Series(False, index=df.index)
for f in exit_filters:
    exit_mask |= f.vectorized_exit(df)
```

#### 3-2: 거래 추출 (O(N) 루프)

numpy 배열로 변환 후 순회:

```python
def _extract_trades(
    entry_signals: np.ndarray,  # bool
    exit_signals: np.ndarray,   # bool
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    initial_balance: float,
    fee_rate: float,
    slippage_pct: float,
    stop_loss_pct: float,
    max_position_pct: float,
    atr_trailing: np.ndarray | None,  # ATR 값 (AtrTrailingExit용)
    atr_multiplier: float,
) -> tuple[list[tuple], float]:
    """
    Returns:
        trades: list of (entry_idx, exit_idx, entry_price, exit_price, quantity, pnl)
        final_balance: float
    """
    n = len(entry_signals)
    trades = []
    cash = initial_balance
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    quantity = 0.0
    highest_since_entry = 0.0
    entry_fee = 0.0

    for i in range(n):
        if in_position:
            # 1. ATR trailing stop 체크
            if atr_trailing is not None:
                if highs[i] > highest_since_entry:
                    highest_since_entry = highs[i]
                trailing_stop = highest_since_entry - atr_multiplier * atr_trailing[i]
                if not np.isnan(trailing_stop) and lows[i] <= trailing_stop:
                    exit_price = trailing_stop * (1 - slippage_pct)
                    fee = exit_price * quantity * fee_rate
                    pnl = (exit_price - entry_price) * quantity - entry_fee - fee
                    cash += exit_price * quantity - fee
                    trades.append((entry_idx, i, entry_price, exit_price, quantity, pnl))
                    in_position = False
                    continue

            # 2. 고정 스톱 로스 체크
            sl_price = entry_price * (1 - stop_loss_pct)
            if lows[i] <= sl_price:
                exit_price = sl_price * (1 - slippage_pct)
                fee = exit_price * quantity * fee_rate
                pnl = (exit_price - entry_price) * quantity - entry_fee - fee
                cash += exit_price * quantity - fee
                trades.append((entry_idx, i, entry_price, exit_price, quantity, pnl))
                in_position = False
                continue

            # 3. Exit 시그널 체크 → 다음 캔들 open에 매도
            if exit_signals[i] and i + 1 < n:
                exit_price = opens[i + 1] * (1 - slippage_pct)
                fee = exit_price * quantity * fee_rate
                pnl = (exit_price - entry_price) * quantity - entry_fee - fee
                cash += exit_price * quantity - fee
                trades.append((entry_idx, i + 1, entry_price, exit_price, quantity, pnl))
                in_position = False
                continue

        else:
            # Entry 시그널 → 다음 캔들 open에 매수
            if entry_signals[i] and i + 1 < n:
                entry_price = opens[i + 1] * (1 + slippage_pct)
                max_value = cash * max_position_pct
                quantity = max_value / entry_price
                entry_fee = entry_price * quantity * fee_rate
                cash -= entry_price * quantity + entry_fee
                entry_idx = i + 1
                highest_since_entry = highs[i + 1]
                in_position = True

    # 미청산 포지션 강제 청산
    if in_position:
        exit_price = closes[-1]
        fee = exit_price * quantity * fee_rate
        pnl = (exit_price - entry_price) * quantity - entry_fee - fee
        cash += exit_price * quantity - fee
        trades.append((entry_idx, n - 1, entry_price, exit_price, quantity, pnl))

    return trades, cash
```

#### 3-3: 메트릭 계산

```python
def _compute_metrics(
    trades: list[tuple],
    initial_balance: float,
    final_balance: float,
    df_index: pd.DatetimeIndex,
    closes: np.ndarray,
    timeframe: str,
) -> VectorizedResult:
    # Equity curve 구축
    # 포지션 미보유 구간: 직전 equity 유지
    # 포지션 보유 구간: entry equity + unrealized PnL
    equity = np.full(len(df_index), np.nan)
    equity[0] = initial_balance
    current_equity = initial_balance

    for entry_idx, exit_idx, entry_p, exit_p, qty, pnl in trades:
        # entry 이전까지 현재 equity로 채우기
        equity[entry_idx] = current_equity - (entry_p * qty + entry_fee)  # 현금만
        # 보유 중: 시가평가
        for j in range(entry_idx, min(exit_idx + 1, len(equity))):
            unrealized = (closes[j] - entry_p) * qty
            equity[j] = current_equity + unrealized - entry_fee_approx
        current_equity += pnl

    # NaN → forward fill
    equity_series = pd.Series(equity, index=df_index)
    equity_series = equity_series.ffill().bfill()

    # 메트릭 계산 (report.py와 동일 공식)
    returns = equity_series.pct_change().dropna()
    std = returns.std(ddof=0)
    annualization = np.sqrt(PERIODS_PER_YEAR.get(timeframe, 8766))
    sharpe = float(returns.mean() / std * annualization) if std > 0 else 0.0

    peak = equity_series.expanding().max()
    drawdown = (peak - equity_series) / peak
    max_dd = float(drawdown.max())

    wins = [t for t in trades if t[5] > 0]
    losses = [t for t in trades if t[5] <= 0]
    win_rate = len(wins) / len(trades) if trades else 0.0
    gross_profit = sum(t[5] for t in wins)
    gross_loss = abs(sum(t[5] for t in losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    total_return = (final_balance - initial_balance) / initial_balance if initial_balance > 0 else 0.0

    return VectorizedResult(
        sharpe_ratio=sharpe, total_return=total_return, max_drawdown=max_dd,
        win_rate=win_rate, profit_factor=pf, total_trades=len(trades),
    )
```

#### 3-4: AtrTrailingExit 처리

AtrTrailingExitFilter가 exit_filters에 있는지 확인:
```python
atr_filter = None
for f in exit_filters:
    if isinstance(f, AtrTrailingExitFilter):
        atr_filter = f
        break

if atr_filter:
    atr_col = f"atr_{atr_filter.period}"
    atr_values = df[atr_col].values
    atr_multiplier = atr_filter.multiplier
    # 나머지 exit_filters에서 vectorized_exit OR 마스크 생성 (AtrTrailing 제외)
    # atr trailing은 _extract_trades 루프에서 별도 처리
else:
    atr_values = None
    atr_multiplier = 0.0
```

---

## Step 4: _run_batch에서 벡터화 분기

### 파일: `src/tradingbot/backtest/parallel.py`

```python
def _run_batch(symbol, timeframe, jobs, data_dir, balance, config_dir="config"):
    df = load_candles(symbol, timeframe, Path(data_dir))
    config = load_config(...)

    # 1. ML 포함 여부로 분리
    vectorizable_jobs = []
    fallback_jobs = []
    for name, entry, exit_ in jobs:
        if not entry:
            fallback_jobs.append((name, entry, exit_))  # 등록된 전략
        elif "lgbm_prob" in entry:
            fallback_jobs.append((name, entry, exit_))  # ML 포함
        else:
            vectorizable_jobs.append((name, entry, exit_))

    results = []

    # 2. 벡터화 경로: 인디케이터 1회 계산 → 템플릿별 시그널 배열 → 거래 추출
    if vectorizable_jobs:
        all_filters = []
        for _, entry, exit_ in vectorizable_jobs:
            all_filters += parse_filter_string(entry, base_timeframe=timeframe)
            all_filters += parse_filter_string(exit_, base_timeframe=timeframe)
        union_strategy = CombinedStrategy(entry_filters=all_filters, exit_filters=[])
        union_strategy.symbols = [symbol]
        union_strategy.timeframe = timeframe
        indicator_df = union_strategy.indicators(df.copy())

        for name, entry, exit_ in vectorizable_jobs:
            entry_filters = parse_filter_string(entry, base_timeframe=timeframe)
            exit_filters = parse_filter_string(exit_, base_timeframe=timeframe)
            result = vectorized_backtest(indicator_df, entry_filters, exit_filters, ...)
            results.append(ScanResult(...))

    # 3. Fallback 경로: 기존 엔진 (등록된 전략 + ML 포함 템플릿)
    # 기존 _run_batch 로직 유지
    ...

    return results
```

---

## Step 5: CLI에서 재검증 옵션 추가

### 파일: `src/tradingbot/cli.py`

combine-scan에 `--verify-top` 옵션 추가:

```python
@app.command()
def combine_scan(
    top_n: int = 15,
    verify_top: int = 0,  # 상위 N개를 기존 엔진으로 재검증
    ...
):
    # 1. 벡터화 엔진으로 전체 스캔 (빠름)
    # 2. verify_top > 0이면 상위 N개를 기존 엔진으로 재검증
    # 3. 재검증 결과로 최종 순위 표시
```

`--verify-top N`을 주면 벡터화 결과 상위 N개만 기존 엔진(`_run_batch`)으로 재검증하여 정밀 수치 확인.
예: `--top 50 --verify-top 10` → 벡터화로 50개 표시, 상위 10개만 기존 엔진으로 재검증.
재검증은 선택사항. 벡터화 결과의 순위(ranking)가 기존 엔진과 거의 동일할 것으로 예상.

---

## Step 6: 테스트

### 신규 테스트 파일: `tests/test_vectorized.py`

```python
class TestVectorizedFilters:
    """각 필터의 vectorized_entry/exit가 check_entry/exit와 동일한 결과를 내는지 확인"""

    def test_rsi_oversold_vectorized_matches_scalar(self):
        # 1. 인디케이터 계산된 df 준비
        # 2. 전체 df에 대해 vectorized_entry 호출 → bool Series
        # 3. 각 행에 대해 check_entry(df[:i]) 호출 → bool
        # 4. 두 결과 비교 (warmup 기간 제외)

    def test_all_filters_vectorized_consistency(self):
        # 모든 29개 필터에 대해 위 패턴 반복


class TestVectorizedEngine:
    """벡터화 엔진 결과가 기존 엔진과 유사한지 확인"""

    def test_simple_template_matches_engine(self):
        # Trend+RSI 템플릿으로 양쪽 엔진 실행 → 거래 수, 방향 비교

    def test_atr_trailing_exit(self):
        # AtrTrailingExit 포함 템플릿 → 거래 추출 정확성

    def test_stop_loss_trigger(self):
        # 스톱 로스 트리거 → 정확한 가격에 청산

    def test_no_trades_produces_zero_sharpe(self):
        # 시그널 없는 경우 → Sharpe 0, trades 0

    def test_metrics_calculation(self):
        # 알려진 거래 목록 → Sharpe, MaxDD, WinRate 직접 검증


class TestRunBatchVectorized:
    """_run_batch에서 벡터화/fallback 분기 테스트"""

    def test_ml_templates_use_fallback(self):
        # lgbm_prob 포함 → 기존 엔진

    def test_rule_templates_use_vectorized(self):
        # lgbm_prob 미포함 → 벡터화 엔진

    def test_mixed_batch(self):
        # ML + Rule 혼합 → 각각 올바른 경로
```

---

## 구현 순서

```
Step 1: BaseFilter에 vectorized 메서드 + supports_vectorized 추가
Step 2: 29개 필터에 vectorized_entry/exit 구현
Step 3: VectorizedEngine (vectorized.py) 구현
Step 4: _run_batch에서 벡터화 분기
Step 5: CLI verify-top 옵션 (선택)
Step 6: 테스트
```

Step 1~2는 기존 코드에 영향 없음 (메서드 추가만).
Step 3은 신규 파일.
Step 4가 실제 통합 지점 — 기존 fallback 경로 유지.

## 예상 결과

| 명령어 | 현재 | 벡터화 후 |
|--------|------|----------|
| scan (168조합) | ~10분 | ~10분 (변경 없음, 등록된 전략) |
| combine-scan (1152조합) | ~1.5시간 | **~1분** (룰 33개 벡터화 + ML 15개 fallback ~3분) |
| combine-scan 룰만 (33 × 24 batch) | ~50분 | **~30초** |
| combine-scan ML만 (15 × 24 batch) | ~40분 | ~40분 (fallback) |

## 핵심 파일

| 파일 | 변경 내용 |
|------|----------|
| `strategy/filters/base.py` | vectorized_entry/exit, supports_vectorized 추가 |
| `strategy/filters/trend.py` | 5개 필터 vectorized 구현 |
| `strategy/filters/momentum.py` | 7개 필터 vectorized 구현 |
| `strategy/filters/price.py` | 5개 필터 vectorized 구현 |
| `strategy/filters/volatility.py` | 4개 필터 vectorized 구현 |
| `strategy/filters/volume.py` | 3개 필터 vectorized 구현 |
| `strategy/filters/exit.py` | 5개 필터 vectorized 구현 (AtrTrailing 제외) |
| `backtest/vectorized.py` | **신규** — vectorized_backtest(), _extract_trades(), _compute_metrics() |
| `backtest/parallel.py` | _run_batch에서 벡터화/fallback 분기 |
| `cli.py` | combine-scan에 verify-top 옵션 (선택) |
| `tests/test_vectorized.py` | **신규** — 필터 일관성 + 엔진 정확성 테스트 |
