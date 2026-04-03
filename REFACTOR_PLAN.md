# REFACTOR_PLAN.md

필터 확장 + 역할 태깅 시스템 구현 계획. REFACTOR_RESEARCH.md 기반.

---

## Phase 1: 인디케이터 추가 (`src/tradingbot/data/indicators.py`)

기존 8개에 11개 추가. 모든 필터의 기반이므로 가장 먼저 구현.

### 1-1. ta 라이브러리 기반 (9개)

각 함수는 기존 패턴(`add_rsi` 등)을 따름: 컬럼 존재 체크 → ta 클래스 생성 → 컬럼 추가 → df 반환.

```python
# 추세
def add_adx(df, period=14):
    # ta.trend.ADXIndicator → adx_{p}, adx_pos_{p}, adx_neg_{p}

def add_ichimoku(df, window1=9, window2=26, window3=52):
    # ta.trend.IchimokuIndicator → ichi_conv, ichi_base, ichi_a, ichi_b

def add_aroon(df, period=25):
    # ta.trend.AroonIndicator → aroon_up_{p}, aroon_down_{p}

# 모멘텀
def add_cci(df, period=20):
    # ta.trend.CCIIndicator → cci_{p}

def add_roc(df, period=12):
    # ta.momentum.ROCIndicator → roc_{p}

# 거래량
def add_mfi(df, period=14):
    # ta.volume.MFIIndicator(high, low, close, volume) → mfi_{p}

def add_obv(df):
    # ta.volume.OnBalanceVolumeIndicator(close, volume) → obv

# 변동성
def add_keltner_channel(df, period=20, atr_period=10):
    # ta.volatility.KeltnerChannel → kc_upper_{p}, kc_middle_{p}, kc_lower_{p}

def add_donchian_channel(df, period=20):
    # ta.volatility.DonchianChannel → dc_upper_{p}, dc_middle_{p}, dc_lower_{p}
```

### 1-2. pandas 직접 구현 (2개)

```python
def add_zscore(df, period=20):
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    df[f"zscore_{period}"] = (df["close"] - sma) / std

def add_pct_from_ma(df, period=20):
    sma = df["close"].rolling(period).mean()
    df[f"pct_from_ma_{period}"] = (df["close"] - sma) / sma * 100
```

### 1-3. 테스트 (`tests/test_indicators.py`)

각 인디케이터 함수에 대해 1개씩 테스트. 기존 패턴(TestSMA 등) 따름:
- 100봉 더미 데이터로 함수 호출
- 출력 컬럼 존재 확인
- NaN 아닌 값 존재 확인
- 기존 누락된 `add_stochastic` 테스트도 추가

---

## Phase 2: BaseFilter 역할 태깅 (`src/tradingbot/strategy/filters/base.py`)

### 2-1. role 필드 추가

```python
class BaseFilter(ABC):
    name: str = "base"
    role: str = "entry"  # "entry" | "trend" | "volatility" | "volume" | "exit"

    def __init__(self, **kwargs):
        self.params = kwargs
```

### 2-2. check_exit 시그니처 확장

atr_trailing_exit를 위해 entry_index 파라미터 추가 (선택적, 하위 호환):

```python
@abstractmethod
def check_exit(self, df: pd.DataFrame, entry_index: int | None = None) -> bool:
    ...
```

**기존 필터 9개 시그니처 업데이트 필수**: `@abstractmethod` 시그니처가 바뀌면 기존 구현체도 맞춰야 함. 기존 9개 필터의 `check_exit` 메서드에 `entry_index: int | None = None` 파라미터 추가 (로직은 변경 없이 무시). 한 줄씩이라 금방이지만 빠뜨리면 즉시 테스트가 깨짐.

대상 파일 (각 check_exit 메서드):
- `trend.py`: TrendUpFilter, TrendDownFilter
- `momentum.py`: RsiOversoldFilter, RsiOverboughtFilter, MacdCrossUpFilter
- `volume.py`: VolumeSpikeFilter
- `price.py`: PriceBreakoutFilter, EmaAboveFilter, BbUpperBreakFilter

### 2-3. 기존 필터에 role 설정

기존 9개 필터의 클래스 정의에 role 추가:

| 필터 | role |
|------|------|
| `trend_up`, `trend_down` | `"trend"` |
| `rsi_oversold`, `macd_cross_up` | `"entry"` |
| `rsi_overbought` | `"exit"` |
| `volume_spike` | `"volume"` |
| `price_breakout`, `bb_upper_break` | `"entry"` |
| `ema_above` | `"trend"` |

---

## Phase 3: 새 필터 구현 (21개)

### 3-1. Entry Signal 필터 6개

**파일**: `momentum.py`에 4개, `price.py`에 2개 추가

| 필터 | 파일 | 핵심 로직 |
|------|------|----------|
| `StochOversoldFilter` | momentum.py | stoch_k: prev ≤ th, curr > th |
| `CciOversoldFilter` | momentum.py | cci: prev ≤ -th, curr > -th |
| `RocPositiveFilter` | momentum.py | roc: prev ≤ 0, curr > 0 |
| `MfiOversoldFilter` | momentum.py | mfi: prev ≤ th, curr > th |
| `EmaCrossUpFilter` | price.py | ema_fast: prev ≤ ema_slow, curr > ema_slow |
| `DonchianBreakFilter` | price.py | close > prev dc_upper |

### 3-2. Trend Filter 필터 3개

**파일**: `trend.py`에 추가

| 필터 | 핵심 로직 |
|------|----------|
| `AdxStrongFilter` | adx > threshold |
| `IchimokuAboveFilter` | close > ichi_a AND close > ichi_b |
| `AroonUpFilter` | aroon_up > threshold AND aroon_up > aroon_down |

### 3-3. Volatility Filter 필터 4개

**파일**: `volatility.py` (신규 생성)

| 필터 | 핵심 로직 |
|------|----------|
| `AtrBreakoutFilter` | close > ema + atr × multiplier |
| `KeltnerBreakFilter` | close > kc_upper (entry), close < kc_middle (exit) |
| `BbSqueezeFilter` | 전봉: bb_upper < kc_upper → 현재봉: bb_upper >= kc_upper |
| `BbBandwidthLowFilter` | (bb_upper - bb_lower) / bb_middle < threshold |

### 3-4. Volume Confirm 필터 2개

**파일**: `volume.py`에 추가

| 필터 | 핵심 로직 |
|------|----------|
| `ObvRisingFilter` | OBV > SMA(OBV, obv_sma_period). 파라미터: `obv_sma_period=20` |
| `MfiConfirmFilter` | MFI > threshold. 파라미터: `period=14, threshold=50` |

### 3-5. Exit Signal 필터 6개

**파일**: `exit.py` (신규 생성)

| 필터 | 핵심 로직 |
|------|----------|
| `StochOverboughtFilter` | stoch_k >= threshold |
| `CciOverboughtFilter` | cci > +threshold |
| `MfiOverboughtFilter` | mfi >= threshold |
| `ZscoreExtremeFilter` | zscore > +threshold |
| `PctFromMaExitFilter` | pct_from_ma > +threshold |
| `AtrTrailingExitFilter` | close < max(high[entry_index:]) - atr × multiplier. entry_index로 진입 시점 이후 최고가 계산. **fallback**: entry_index가 None이면 최근 20봉 최고가 사용 |

---

## Phase 4: 레지스트리 업데이트 (`src/tradingbot/strategy/filters/registry.py`)

### 4-1. get_filter_map() 확장

기존 9개 + 신규 21개 = 30개 등록:

```python
# 신규 import 추가
from tradingbot.strategy.filters.volatility import (
    AtrBreakoutFilter, KeltnerBreakFilter, BbSqueezeFilter, BbBandwidthLowFilter,
)
from tradingbot.strategy.filters.exit import (
    StochOverboughtFilter, CciOverboughtFilter, MfiOverboughtFilter,
    ZscoreExtremeFilter, PctFromMaExitFilter, AtrTrailingExitFilter,
)

# 맵에 추가
"stoch_oversold": StochOversoldFilter,
"cci_oversold": CciOversoldFilter,
"roc_positive": RocPositiveFilter,
"mfi_oversold": MfiOversoldFilter,
"ema_cross_up": EmaCrossUpFilter,
"donchian_break": DonchianBreakFilter,
"adx_strong": AdxStrongFilter,
"ichimoku_above": IchimokuAboveFilter,
"aroon_up": AroonUpFilter,
"atr_breakout": AtrBreakoutFilter,
"keltner_break": KeltnerBreakFilter,
"bb_squeeze": BbSqueezeFilter,
"bb_bandwidth_low": BbBandwidthLowFilter,
"obv_rising": ObvRisingFilter,
"mfi_confirm": MfiConfirmFilter,
"stoch_overbought": StochOverboughtFilter,
"cci_overbought": CciOverboughtFilter,
"mfi_overbought": MfiOverboughtFilter,
"zscore_extreme": ZscoreExtremeFilter,
"pct_from_ma_exit": PctFromMaExitFilter,
"atr_trailing_exit": AtrTrailingExitFilter,
```

### 4-2. _parse_filter_params() 확장

각 필터의 파라미터 파싱 규칙 추가:

```
stoch_oversold:[threshold]:[k_period]:[d_period]     → threshold=20, k_period=14, d_period=3
stoch_overbought:[threshold]:[k_period]:[d_period]   → threshold=80, k_period=14, d_period=3
cci_oversold:[threshold]:[period]                     → threshold=100, period=20
cci_overbought:[threshold]:[period]                   → threshold=100, period=20
roc_positive:[period]                                 → period=12
mfi_oversold:[threshold]:[period]                     → threshold=20, period=14
mfi_overbought:[threshold]:[period]                   → threshold=80, period=14
mfi_confirm:[threshold]:[period]                      → threshold=50, period=14
ema_cross_up:[fast]:[slow]                            → fast=12, slow=26
donchian_break:[period]                               → period=20
adx_strong:[threshold]:[period]                       → threshold=25, period=14
ichimoku_above                                        → (기본값만, window1=9, window2=26, window3=52)
aroon_up:[threshold]:[period]                         → threshold=70, period=25
atr_breakout:[period]:[multiplier]:[ema_period]       → period=14, multiplier=2.0, ema_period=20
keltner_break:[period]:[atr_period]                   → period=20, atr_period=10
bb_squeeze:[bb_period]:[kc_period]                    → bb_period=20, kc_period=20
bb_bandwidth_low:[threshold]:[period]                 → threshold=0.05, period=20
obv_rising:[obv_sma_period]                           → obv_sma_period=20
zscore_extreme:[threshold]:[period]                   → threshold=2.0, period=20
pct_from_ma_exit:[period]:[threshold]                 → period=20, threshold=5.0
atr_trailing_exit:[period]:[multiplier]               → period=14, multiplier=2.5
```

### 4-3. `__init__.py` 업데이트

`strategy/filters/__init__.py`에 새 모듈 import 추가.

---

## Phase 5: CombinedStrategy 역할 분기 (`src/tradingbot/strategy/combined.py`)

### 5-1. should_entry 변경

```python
def should_entry(self, df, symbol):
    if not self.entry_filters:
        return None
    for f in self.entry_filters:
        if f.role == "exit":
            continue  # exit 역할은 entry AND에서 스킵
        if not f.check_entry(df):
            return None
    return Signal(...)
```

### 5-2. should_exit 변경

entry_index를 계산해서 atr_trailing_exit에 전달:

```python
def should_exit(self, df, symbol, position):
    if not self.exit_filters:
        return None

    # position.entry_time으로 entry_index 계산
    # Position 클래스에 entry_time: datetime 필드 확인됨 (core/models.py:148)
    # 백테스트 엔진과 라이브 엔진 모두에서 세팅됨
    entry_index = None
    if position and position.entry_time:
        try:
            entry_index = df.index.get_indexer([position.entry_time], method="ffill")[0]
            if entry_index < 0:
                entry_index = None
        except (KeyError, IndexError):
            entry_index = None

    for f in self.exit_filters:
        if f.check_exit(df, entry_index=entry_index):
            return Signal(...)
    return None
```

**entry_index fallback 동작**: `entry_index is None`일 때 (entry_time이 df 범위 밖이거나 매칭 실패) `AtrTrailingExitFilter`는 최근 20봉 최고가를 대신 사용. 다른 필터는 entry_index를 무시하므로 영향 없음.

---

## Phase 6: CLI 템플릿 추가 (`src/tradingbot/cli.py`)

### 6-1. COMBINE_TEMPLATES에 18개 신규 템플릿 추가

기존 15개 뒤에 추가. 4가지 유형으로 분류.

**ema_above Exit 확인 완료**: `ema_above`의 `check_exit`은 `close < ema`로 이미 구현됨 (price.py:70-78). role은 `"trend"`이지만 exit_filters 리스트에 넣으면 `check_exit`가 정상 호출됨. Phase 5의 should_exit는 role 기반 필터링을 하지 않으므로 문제 없음.

```python
# ── Trend Following (신규) ──
{"entry": "ema_cross_up:12:26 + adx_strong:25 + volume_spike:2.0", "exit": "atr_trailing_exit:14:2.5", "label": "EMACross+ADX+Vol→ATR"},
{"entry": "ema_cross_up:12:26 + adx_strong:25", "exit": "rsi_overbought:70", "label": "EMACross+ADX"},
{"entry": "stoch_oversold:20 + aroon_up:70", "exit": "stoch_overbought:80", "label": "Stoch+Aroon"},
{"entry": "roc_positive:12 + ichimoku_above + obv_rising", "exit": "rsi_overbought:70", "label": "ROC+Ichi+OBV"},
{"entry": "donchian_break:20 + adx_strong:25 + volume_spike:2.0", "exit": "donchian_break:20", "label": "Donchian+ADX+Vol"},
{"entry": "macd_cross_up + aroon_up:70 + mfi_confirm:50", "exit": "mfi_overbought:80", "label": "MACD+Aroon+MFI"},

# ── Mean Reversion (신규) ──
{"entry": "rsi_oversold:30 + adx_strong:20 + obv_rising", "exit": "zscore_extreme:2.0", "label": "RSI+ADX+OBV→Zscore"},
{"entry": "stoch_oversold:20 + ema_above:50 + mfi_confirm:40", "exit": "stoch_overbought:80", "label": "Stoch+EMA+MFI"},
{"entry": "cci_oversold:100 + trend_up:4 + volume_spike:2.0", "exit": "cci_overbought:100", "label": "CCI+Trend+Vol"},
{"entry": "mfi_oversold:20 + trend_up:4", "exit": "mfi_overbought:80", "label": "MFI+Trend"},
{"entry": "rsi_oversold:30 + ichimoku_above", "exit": "pct_from_ma_exit:20:5.0", "label": "RSI+Ichi→PctMA"},

# ── Volatility Breakout (신규) ──
{"entry": "bb_upper_break:20 + bb_squeeze + volume_spike:2.0", "exit": "ema_above:20", "label": "BB+Squeeze+Vol"},
{"entry": "atr_breakout:14:2.0 + adx_strong:25 + obv_rising", "exit": "atr_trailing_exit:14:2.5", "label": "ATR+ADX+OBV→ATR"},
{"entry": "keltner_break + trend_up:4 + volume_spike:2.0", "exit": "keltner_break", "label": "KC+Trend+Vol"},
{"entry": "price_breakout:20 + bb_bandwidth_low:0.05 + volume_spike:2.5", "exit": "pct_from_ma_exit:20:5.0", "label": "Breakout+BBW+Vol"},

# ── Multi-Confirm (신규) ──
{"entry": "rsi_oversold:30 + stoch_oversold:20 + adx_strong:25", "exit": "rsi_overbought:70", "label": "RSI+Stoch+ADX"},
{"entry": "macd_cross_up + obv_rising + adx_strong:25", "exit": "atr_trailing_exit:14:2.0", "label": "MACD+OBV+ADX→ATR"},
{"entry": "ema_cross_up:12:26 + mfi_confirm:50 + bb_bandwidth_low:0.04", "exit": "zscore_extreme:2.0", "label": "EMA+MFI+BBW→Zscore"},
```

---

## Phase 7: 테스트 업데이트

### 7-1. `tests/test_indicators.py`

기존 패턴 따라 11개 테스트 추가:
- TestADX, TestCCI, TestROC, TestMFI, TestOBV
- TestKeltnerChannel, TestDonchianChannel, TestIchimoku, TestAroon
- TestZscore, TestPctFromMa
- (누락된) TestStochastic 추가

### 7-2. `tests/test_combine.py`

- `test_all_filters_registered`: `len(fmap) == 9` → `len(fmap) == 30`
- 새 필터 파싱 테스트 추가:
  - `parse_filter_spec("adx_strong:25")` → AdxStrongFilter, threshold=25
  - `parse_filter_spec("ema_cross_up:12:26")` → EmaCrossUpFilter, fast=12, slow=26
  - `parse_filter_spec("atr_trailing_exit:14:2.5")` → AtrTrailingExitFilter
  - `parse_filter_spec("obv_rising:20")` → ObvRisingFilter, obv_sma_period=20
- role 태그 테스트: 각 역할별 필터의 role 값 검증
- CombinedStrategy role 분기 테스트: exit 역할 필터가 entry AND에서 스킵되는지 검증

### 7-3. `tests/test_filters.py` (신규)

개별 필터 단위 테스트. _make_data()로 더미 데이터 생성 후:
- compute()가 필요한 컬럼 추가하는지
- check_entry/check_exit가 bool 반환하는지
- edge case (NaN, 짧은 데이터) 처리 확인

---

## Phase 8: 검증

### 8-1. 단위 테스트

```bash
pytest tests/test_indicators.py tests/test_combine.py tests/test_filters.py -v
```

### 8-2. 전체 테스트

```bash
pytest tests/ -v
```

기존 ~119개 테스트 통과 + 신규 테스트 통과 확인.

### 8-3. 통합 테스트 (데이터 필요)

```bash
# 단일 조합 테스트
tradingbot combine --entry "adx_strong:25 + rsi_oversold:30" --exit "zscore_extreme:2.0" --symbol BTC/KRW

# 전체 스캔 (33개 템플릿 × 심볼 × 타임프레임)
tradingbot combine-scan --top 15
```

---

## 실행 순서 요약

| 순서 | Phase | 파일 | 작업량 |
|------|-------|------|--------|
| 1 | Phase 1 | `data/indicators.py` | 함수 11개 추가 |
| 2 | Phase 1 | `tests/test_indicators.py` | 테스트 12개 추가 |
| 3 | Phase 2 | `strategy/filters/base.py` | role 필드 + check_exit 시그니처 |
| 4 | Phase 2 | 기존 필터 5개 파일 | role 값 설정 (1줄씩) |
| 5 | Phase 3 | `strategy/filters/momentum.py` | 필터 4개 추가 |
| 6 | Phase 3 | `strategy/filters/price.py` | 필터 2개 추가 |
| 7 | Phase 3 | `strategy/filters/trend.py` | 필터 3개 추가 |
| 8 | Phase 3 | `strategy/filters/volatility.py` (신규) | 필터 4개 |
| 9 | Phase 3 | `strategy/filters/volume.py` | 필터 2개 추가 |
| 10 | Phase 3 | `strategy/filters/exit.py` (신규) | 필터 6개 |
| 11 | Phase 4 | `strategy/filters/registry.py` | 21개 등록 + 파싱 |
| 12 | Phase 4 | `strategy/filters/__init__.py` | import 추가 |
| 13 | Phase 5 | `strategy/combined.py` | role 분기 + entry_index 전달 |
| 14 | Phase 6 | `cli.py` | 템플릿 18개 추가 |
| 15 | Phase 7 | `tests/test_combine.py` | 테스트 업데이트 |
| 16 | Phase 7 | `tests/test_filters.py` (신규) | 필터 단위 테스트 |
| 17 | Phase 8 | — | pytest + combine-scan 검증 |

**총 수정 파일**: 12개 (신규 3개 + 기존 9개)
**총 신규 코드**: 인디케이터 11개, 필터 21개, 템플릿 18개, 테스트 ~30개
