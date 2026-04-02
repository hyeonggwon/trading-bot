# REFACTOR_RESEARCH.md

새로운 필터/인디케이터 추가를 위한 리서치 결과 정리.

---

## 1. 현재 구현 상태

### 1.1 기존 인디케이터 (`src/tradingbot/data/indicators.py`)

| 함수 | 출력 컬럼 | ta 라이브러리 사용 |
|------|-----------|-------------------|
| `add_sma(period=20)` | `sma_{period}` | X (pandas rolling) |
| `add_ema(period=20)` | `ema_{period}` | X (pandas ewm) |
| `add_rsi(period=14)` | `rsi_{period}` | `ta.momentum.RSIIndicator` |
| `add_macd(fast=12, slow=26, signal=9)` | `macd_*`, `macd_signal_*`, `macd_hist_*` | `ta.trend.MACD` |
| `add_bollinger_bands(period=20, std=2.0)` | `bb_upper_*`, `bb_middle_*`, `bb_lower_*` | `ta.volatility.BollingerBands` |
| `add_atr(period=14)` | `atr_{period}` | `ta.volatility.AverageTrueRange` |
| `add_stochastic(k_period=14, d_period=3)` | `stoch_k_*`, `stoch_d_*` | `ta.momentum.StochasticOscillator` |
| `add_volume_sma(period=20)` | `volume_sma_{period}` | X (pandas rolling) |

**총 8개**, ta 버전: `ta>=0.11`

### 1.2 기존 필터 (9개, `src/tradingbot/strategy/filters/`)

| 필터 | 파일 | 파라미터 |
|------|------|----------|
| `trend_up` | trend.py | tf_factor, sma_period, base_timeframe |
| `trend_down` | trend.py | tf_factor, sma_period, base_timeframe |
| `rsi_oversold` | momentum.py | period, threshold |
| `rsi_overbought` | momentum.py | period, threshold |
| `macd_cross_up` | momentum.py | fast, slow, signal |
| `volume_spike` | volume.py | sma_period, threshold |
| `price_breakout` | price.py | lookback |
| `ema_above` | price.py | period |
| `bb_upper_break` | price.py | period, std |

### 1.3 핵심 인터페이스

**BaseFilter** (`base.py`):
- `compute(df)` — 인디케이터 컬럼 추가
- `check_entry(df)` — 진입 조건 (bool)
- `check_exit(df)` — 청산 조건 (bool)

**CombinedStrategy** (`combined.py`):
- 진입: 모든 entry 필터 AND
- 청산: exit 필터 중 하나라도 OR
- compute 중복 제거 (class + params 기준)

### 1.4 현재 combine-scan 템플릿 (15개, `cli.py`)

기존 필터 9개로 조합한 15개 템플릿. trend_up, rsi, macd, volume_spike, ema_above, bb_upper_break, price_breakout 위주.

---

## 2. 역할 기반 필터 태깅 시스템

각 필터를 **역할(role)**로 태깅하여 조합 로직을 명확하게 한다.

| 역할 | 의미 | 예시 |
|------|------|------|
| **Entry Signal** | 진입 시점 결정 | RSI 과매도 탈출, Stochastic 크로스, MACD 양전환 |
| **Trend Filter** | 추세 방향 확인 (필터) | ADX > 25, SMA 방향, Ichimoku 구름 위 |
| **Volatility Filter** | 변동성 조건 확인 (필터) | BB Squeeze, Keltner Break |
| **Volume Confirm** | 거래량으로 시그널 확인 | OBV 상승, Volume Spike, MFI 상승 |
| **Exit Signal** | 청산 시점 결정 | RSI 과매수, ATR Trailing, Stochastic 과매수 |

### 조합 규칙

```
Entry(진입 시그널) + Filter(추세 확인) + Confirm(거래량 확인) + Exit(청산 시그널)
```

예시: `Entry(RSI 과매도) + Filter(ADX > 25) + Confirm(MFI 상승) + Exit(ATR Trailing Stop)`

### 기존 필터의 역할 분류

| 필터 | 현재 역할 |
|------|----------|
| `rsi_oversold` | Entry Signal |
| `macd_cross_up` | Entry Signal |
| `bb_upper_break` | Entry Signal |
| `price_breakout` | Entry Signal |
| `trend_up` / `trend_down` | Trend Filter |
| `ema_above` | Trend Filter |
| `volume_spike` | Volume Confirm |
| `rsi_overbought` | Exit Signal |

**현재 부족한 역할**: Volatility Filter (0개), Volume Confirm (1개뿐), Exit Signal (1개뿐)

### 역할 태그와 CombinedStrategy 연동

현재 CombinedStrategy는 entry_filters / exit_filters로 나누는데, 역할 태그를 `BaseFilter.role` 필드로 추가한다.

```python
class BaseFilter(ABC):
    name: str = "base"
    role: str = "entry"  # "entry" | "trend" | "volatility" | "volume" | "exit"
```

**CombinedStrategy 분기 로직 변경**:
- `entry_filters` 목록에서 `role == "exit"`인 필터는 AND 진입 조건에서 **제외** (항상 True 취급)
- `exit_filters` 목록에서 `role != "exit"`인 필터는 OR 청산 조건에서 **제외**
- 이렇게 하면 Exit-only 필터(zscore_extreme, atr_trailing_exit 등)의 `check_entry()`가 False를 반환해도 조합이 깨지지 않음

---

## 3. 추가할 인디케이터 (`indicators.py`)

### ta 라이브러리 기반 (9개)

| 함수명 | ta 클래스 | 출력 컬럼 |
|--------|----------|-----------|
| `add_adx(period=14)` | `ta.trend.ADXIndicator` | `adx_{p}`, `adx_pos_{p}`, `adx_neg_{p}` |
| `add_cci(period=20)` | `ta.trend.CCIIndicator` | `cci_{p}` |
| `add_roc(period=12)` | `ta.momentum.ROCIndicator` | `roc_{p}` |
| `add_mfi(period=14)` | `ta.volume.MFIIndicator` | `mfi_{p}` |
| `add_obv()` | `ta.volume.OnBalanceVolumeIndicator` | `obv` |
| `add_keltner_channel(period=20, atr=10)` | `ta.volatility.KeltnerChannel` | `kc_upper_*`, `kc_middle_*`, `kc_lower_*` |
| `add_donchian_channel(period=20)` | `ta.volatility.DonchianChannel` | `dc_upper_*`, `dc_middle_*`, `dc_lower_*` |
| `add_ichimoku(w1=9, w2=26, w3=52)` | `ta.trend.IchimokuIndicator` | `ichi_conv`, `ichi_base`, `ichi_a`, `ichi_b` |
| `add_aroon(period=25)` | `ta.trend.AroonIndicator` | `aroon_up_{p}`, `aroon_down_{p}` |

### pandas 직접 구현 (2개)

| 함수명 | 설명 | 출력 컬럼 |
|--------|------|-----------|
| `add_zscore(period=20)` | (close - SMA) / std | `zscore_{p}` |
| `add_pct_from_ma(period=20)` | (close - SMA) / SMA × 100 | `pct_from_ma_{p}` |

**제외**: Williams %R (Stochastic과 수학적 중복), Supertrend (재귀적 방향 플립 로직으로 벡터화 어려움 — 2차 스코프로 이동)

---

## 4. 추가할 필터 — 역할별 분류

### 4.1 Entry Signal (진입 시그널) — 6개 추가

| 필터명 | 인디케이터 | Entry 조건 | Exit |
|--------|-----------|-----------|------|
| `stoch_oversold` | add_stochastic | %K가 threshold 아래서 위로 크로스 (iloc[-2] ≤ th, iloc[-1] > th) | False |
| `cci_oversold` | add_cci | CCI가 -threshold 아래서 위로 크로스 | False |
| `roc_positive` | add_roc | ROC가 0 아래서 위로 크로스 | False |
| `mfi_oversold` | add_mfi | MFI가 threshold 아래서 위로 크로스 | False |
| `ema_cross_up` | add_ema × 2 | fast EMA > slow EMA 크로스 (iloc[-2]: fast ≤ slow, iloc[-1]: fast > slow) | False |
| `donchian_break` | add_donchian_channel | close > 전봉 dc_upper (iloc[-2]의 upper) | False |

**제외**: williams_oversold (Stochastic과 중복), rsi_divergence (저점 감지 난이도 높음 — 대신 `rsi_rising` 단순화 버전 검토 가능, 2차 스코프)

### 4.2 Trend Filter (추세 필터) — 3개 추가

| 필터명 | 인디케이터 | Entry (추세 확인) | Exit (추세 이탈) |
|--------|-----------|-----------------|-----------------|
| `adx_strong` | add_adx | ADX > threshold (강한 추세 존재) | ADX < threshold |
| `ichimoku_above` | add_ichimoku | close > span_a AND close > span_b | close < min(span_a, span_b) |
| `aroon_up` | add_aroon | aroon_up > threshold AND aroon_up > aroon_down | aroon_down > aroon_up |

**제외**: supertrend_up (Supertrend 재귀 계산 이슈 — 2차 스코프)

### 4.3 Volatility Filter (변동성 필터) — 4개 추가

| 필터명 | 인디케이터 | Entry 조건 | Exit 조건 |
|--------|-----------|-----------|----------|
| `atr_breakout` | add_atr + add_ema | close > ema + atr × multiplier | close < ema - atr × multiplier |
| `keltner_break` | add_keltner_channel | close > kc_upper (entry), close < kc_middle (exit) | close < kc_middle |
| `bb_squeeze` | add_bb + add_kc | **상태 전환 감지**: iloc[-2]에서 bb < kc → iloc[-1]에서 bb >= kc | False (확인용) |
| `bb_bandwidth_low` | add_bollinger_bands | bandwidth < threshold (축소 상태, 폭발 전조) | False (확인용) |

**bb_squeeze 구현 참고**: 단순 현재 봉 조건이 아니라 전봉→현재봉 상태 전환을 감지해야 함. `df.iloc[-2]`와 `df.iloc[-1]`을 같이 비교. 이 패턴은 기존 RsiOversoldFilter (prev ≤ th, curr > th)와 동일하므로 BaseFilter 변경 불필요.

### 4.4 Volume Confirm (거래량 확인) — 2개 추가

| 필터명 | 인디케이터 | Entry 조건 | Exit 조건 |
|--------|-----------|-----------|----------|
| `obv_rising` | add_obv + SMA(obv) | OBV > OBV의 SMA (매집 진행) | OBV < OBV_SMA |

`obv_rising` 파라미터: `obv_sma_period` (기본값 20). OBV 자체는 파라미터 없음, OBV에 적용하는 SMA 기간만 지정. 레지스트리 파싱: `obv_rising:20` → `obv_sma_period=20`.

| `mfi_confirm` | add_mfi | MFI > threshold (자금 유입 중) | MFI < exit_threshold |

### 4.5 Exit Signal (청산 시그널) — 6개 추가

| 필터명 | 인디케이터 | Entry | Exit 조건 |
|--------|-----------|-------|----------|
| `stoch_overbought` | add_stochastic | False | %K >= threshold |
| `cci_overbought` | add_cci | False | CCI > +threshold |
| `mfi_overbought` | add_mfi | False | MFI >= threshold |
| `zscore_extreme` | add_zscore | False | zscore > +threshold (과매수) |
| `pct_from_ma_exit` | add_pct_from_ma | False | pct > +threshold (괴리 과다) |
| `atr_trailing_exit` | add_atr | False | close < highest_since_entry - atr × multiplier |

**atr_trailing_exit 인터페이스 설계**: 이 필터만 "포지션 진입 후 최고가"를 추적해야 하므로 기존 `check_exit(df)` 인터페이스만으로는 부족함. `check_exit(df, entry_index=None)` 형태로 진입 시점을 선택적으로 받도록 BaseFilter 시그니처를 확장. entry_index가 주어지면 `df["high"].iloc[entry_index:]`에서 최고가를 구함. 기존 필터는 entry_index를 무시하므로 하위 호환 유지됨. CombinedStrategy의 `should_exit()`에서 position.entry_time 기준으로 entry_index를 계산해서 전달.

**Exit 반전 패턴 정의**: keltner_break, donchian_break 같은 필터는 entry/exit 조건을 **둘 다 자체 구현**. 템플릿에서 같은 필터를 entry와 exit에 모두 사용하면 자동으로 각각의 check_entry/check_exit가 호출됨. 별도의 "반전 필터" 클래스 불필요.

---

## 5. 조합 스캔 성능 대응

필터 9개 → 30개로 증가하면 조합 공간이 폭발함. 대응 전략:

### 5.1 인디케이터 계산 캐싱

동일 인디케이터 중복 계산 방지. 현재 CombinedStrategy의 dedup 로직(class + params 기준)이 **단일 조합 내** 중복은 처리하지만, **조합 간** 중복(같은 심볼+타임프레임에서 여러 조합이 같은 인디케이터 사용)은 처리 안 됨.

**대응**: combine-scan에서 심볼+타임프레임별로 DataFrame을 한 번 로드한 뒤, 모든 인디케이터를 미리 계산해두고 각 조합에서 재사용. 현재는 매 조합마다 df.copy()로 시작하므로, 공통 인디케이터가 이미 붙어있는 "풍부한 DataFrame"을 캐시하면 됨.

### 5.2 역할 규칙 강제

조합 생성 시 최소 조건:
- Entry Signal 최소 1개
- Exit Signal 최소 1개
- 같은 역할 내에서 최대 2개

### 5.3 템플릿 기반 접근 유지

자동 조합 생성(combinatorial explosion)은 2차 스코프. 1차는 **수작업 큐레이팅된 템플릿**으로 접근. 기존 15개 + 신규 ~18개 = 약 33개 템플릿 × 8심볼 × 3타임프레임 = ~792회 백테스트 (관리 가능한 수준).

---

## 6. 역할 기반 조합 템플릿 (신규)

기존 15개 템플릿에 추가할 새 템플릿. `[역할]` 주석으로 구성 명시.

**참고**: `ema_above`는 기존 구현에서 `check_exit: close < ema`가 이미 정의되어 있음 (price.py:70-78). Trend Filter 역할이지만 Exit으로 사용해도 정상 동작 확인됨.

### 추세추종형 (Trend Following)

```
# [Entry] + [Trend] + [Volume] → [Exit]
ema_cross_up:12:26 + adx_strong:25 + volume_spike:2.0   → atr_trailing_exit:14:2.5
ema_cross_up:12:26 + adx_strong:25                       → rsi_overbought:70
stoch_oversold:20  + aroon_up:70                          → stoch_overbought:80
roc_positive:12    + ichimoku_above + obv_rising           → rsi_overbought:70
donchian_break:20  + adx_strong:25 + volume_spike:2.0     → donchian_break:20
macd_cross_up      + aroon_up:70   + mfi_confirm:50        → mfi_overbought:80
```

### 평균회귀형 (Mean Reversion)

```
# [Entry(과매도)] + [Trend(추세 건재)] + [Volume] → [Exit(과매수)]
rsi_oversold:30    + adx_strong:20 + obv_rising            → zscore_extreme:2.0
stoch_oversold:20  + ema_above:50  + mfi_confirm:40        → stoch_overbought:80
cci_oversold:100   + trend_up:4    + volume_spike:2.0      → cci_overbought:100
mfi_oversold:20    + trend_up:4                            → mfi_overbought:80
rsi_oversold:30    + ichimoku_above                        → pct_from_ma_exit:20:5.0
```

### 변동성 돌파형 (Volatility Breakout)

```
# [Entry(돌파)] + [Volatility(조건)] + [Volume] → [Exit]
bb_upper_break:20  + bb_squeeze     + volume_spike:2.0     → ema_above:20
atr_breakout:14:2.0 + adx_strong:25 + obv_rising          → atr_trailing_exit:14:2.5
keltner_break      + trend_up:4    + volume_spike:2.0      → keltner_break
price_breakout:20  + bb_bandwidth_low:0.05 + volume_spike:2.5 → pct_from_ma_exit:20:5.0
```

### 복합 확인형 (Multi-Confirm)

```
# 여러 역할의 필터를 겹쳐서 정확도 향상
rsi_oversold:30 + stoch_oversold:20 + adx_strong:25       → rsi_overbought:70
macd_cross_up + obv_rising + adx_strong:25                 → atr_trailing_exit:14:2.0
ema_cross_up:12:26 + mfi_confirm:50 + bb_bandwidth_low:0.04 → zscore_extreme:2.0
```

**총 신규 템플릿: 18개** (기존 15개 + 신규 18개 = 33개)

---

## 7. ta 라이브러리 API 레퍼런스

```python
# 추세
ta.trend.ADXIndicator(high, low, close, window=14)
  .adx(), .adx_pos(), .adx_neg()

ta.trend.IchimokuIndicator(high, low, window1=9, window2=26, window3=52)
  .ichimoku_conversion_line(), .ichimoku_base_line()
  .ichimoku_a(), .ichimoku_b()

ta.trend.AroonIndicator(high, low, window=25)
  .aroon_up(), .aroon_down()

# 모멘텀
ta.momentum.ROCIndicator(close, window=12)
  .roc()

ta.trend.CCIIndicator(high, low, close, window=20)  # 주의: ta.trend에 있음
  .cci()

ta.volume.MFIIndicator(high, low, close, volume, window=14)
  .money_flow_index()

# 거래량
ta.volume.OnBalanceVolumeIndicator(close, volume)
  .on_balance_volume()

# 변동성
ta.volatility.KeltnerChannel(high, low, close, window=20, window_atr=10)
  .keltner_channel_hband(), .keltner_channel_mband(), .keltner_channel_lband()

ta.volatility.DonchianChannel(high, low, close, window=20)
  .donchian_channel_hband(), .donchian_channel_mband(), .donchian_channel_lband()
```

---

## 8. 파일별 수정 계획

| 파일 | 작업 |
|------|------|
| `data/indicators.py` | 인디케이터 11개 추가 (ta 9개 + pandas 2개) |
| `strategy/filters/base.py` | `role` 필드 추가 ("entry" / "trend" / "volatility" / "volume" / "exit") |
| `strategy/filters/trend.py` | AdxStrongFilter, IchimokuAboveFilter, AroonUpFilter 추가 |
| `strategy/filters/momentum.py` | StochOversold/Overbought, CciOversold/Overbought, RocPositive, MfiOversold/Overbought 추가 |
| `strategy/filters/volume.py` | ObvRisingFilter, MfiConfirmFilter 추가 |
| `strategy/filters/price.py` | AtrBreakoutFilter, KeltnerBreakFilter, DonchianBreakFilter 추가 |
| `strategy/filters/volatility.py` (신규) | BbSqueezeFilter, BbBandwidthLowFilter 추가 |
| `strategy/filters/exit.py` (신규) | ZscoreExtremeFilter, PctFromMaExitFilter, AtrTrailingExitFilter 추가 |
| `strategy/filters/registry.py` | 새 필터 21개 등록 + 파라미터 파싱 |
| `strategy/combined.py` | 역할 태그 기반 분기 (exit role 필터는 entry AND에서 제외) |
| `cli.py` | COMBINE_TEMPLATES에 신규 18개 템플릿 추가 |
| `tests/test_indicators.py` | 새 인디케이터 테스트 |
| `tests/test_combine.py` | 레지스트리 개수 업데이트 (9→30), 새 필터 파싱 테스트 |

### 구현 순서

1. `indicators.py` — 인디케이터 함수 11개 추가 (필터가 의존)
2. `filters/base.py` — role 필드 추가
3. `filters/` — 역할별 새 필터 클래스 21개 구현
4. `registry.py` — 필터 등록 + 파라미터 파싱
5. `combined.py` — 역할 태그 기반 entry/exit 분기 로직
6. `cli.py` — 신규 18개 템플릿 추가
7. `tests/` — 테스트 작성
8. 검증 — `pytest tests/ -v` + `tradingbot combine-scan --top 15`

---

## 9. 필터 전체 목록 (기존 9 + 신규 21 = 총 30개)

| # | 필터명 | 역할 | 신규 |
|---|--------|------|------|
| 1 | `rsi_oversold` | Entry Signal | |
| 2 | `macd_cross_up` | Entry Signal | |
| 3 | `bb_upper_break` | Entry Signal | |
| 4 | `price_breakout` | Entry Signal | |
| 5 | `stoch_oversold` | Entry Signal | NEW |
| 6 | `cci_oversold` | Entry Signal | NEW |
| 7 | `roc_positive` | Entry Signal | NEW |
| 8 | `mfi_oversold` | Entry Signal | NEW |
| 9 | `ema_cross_up` | Entry Signal | NEW |
| 10 | `donchian_break` | Entry Signal | NEW |
| 11 | `trend_up` | Trend Filter | |
| 12 | `trend_down` | Trend Filter | |
| 13 | `ema_above` | Trend Filter | |
| 14 | `adx_strong` | Trend Filter | NEW |
| 15 | `ichimoku_above` | Trend Filter | NEW |
| 16 | `aroon_up` | Trend Filter | NEW |
| 17 | `atr_breakout` | Volatility Filter | NEW |
| 18 | `keltner_break` | Volatility Filter | NEW |
| 19 | `bb_squeeze` | Volatility Filter | NEW |
| 20 | `bb_bandwidth_low` | Volatility Filter | NEW |
| 21 | `volume_spike` | Volume Confirm | |
| 22 | `obv_rising` | Volume Confirm | NEW |
| 23 | `mfi_confirm` | Volume Confirm | NEW |
| 24 | `rsi_overbought` | Exit Signal | |
| 25 | `stoch_overbought` | Exit Signal | NEW |
| 26 | `cci_overbought` | Exit Signal | NEW |
| 27 | `mfi_overbought` | Exit Signal | NEW |
| 28 | `zscore_extreme` | Exit Signal | NEW |
| 29 | `pct_from_ma_exit` | Exit Signal | NEW |
| 30 | `atr_trailing_exit` | Exit Signal | NEW |

### 2차 스코프 (이번에 구현하지 않음)

| 필터명 | 사유 |
|--------|------|
| `williams_oversold` | Stochastic과 수학적 중복 (Williams %R = -(100 - %K)) |
| `rsi_divergence` | 저점 감지 로직 복잡, 엣지케이스 다수 — 단순화 버전(`rsi_rising`: RSI가 과매도 구간에서 N봉 연속 상승) 검토 후 추가 |
| `supertrend_up` | Supertrend 방향 플립이 재귀적 계산이라 벡터화 어려움 — pandas_ta 또는 numba 검토 후 추가 |
| 자동 조합 생성기 | 역할별 combinatorial 탐색은 조합 폭발 (수천~수만) — 캐싱/스크리닝 인프라 구축 후 추가 |
