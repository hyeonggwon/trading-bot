# Trading Bot

Upbit(한국 거래소) KRW 마켓 대상 알고리즘 트레이딩 봇.

Freqtrade의 전략 프레임워크, Jesse의 anti-lookahead 백테스트, NautilusTrader의 설계 철학을 참고하여 Python으로 구축.

## 주요 기능

- **Anti-lookahead 백테스트 엔진** — 전략은 과거 캔들만 접근, 체결은 다음 캔들 시가에 발생, 벡터화 스크리닝 엔진으로 combine-scan 1152조합 3분대
- **멀티 심볼 동시 매매** — 여러 종목을 하나의 포트폴리오로 동시 운영
- **7가지 내장 전략** — SMA, RSI, MACD, 볼린저, 멀티타임프레임, 거래량 돌파, LightGBM ML
- **전략 자동 스캔** — 전 전략 × 심볼 × 타임프레임 조합 자동 백테스트 + 랭킹
- **필터 조합 엔진** — 코드 없이 CLI로 필터 조합 (31종 필터, 48개 템플릿, 5가지 역할 태깅)
- **ML 전략 (LightGBM)** — 15개 피처 자동 생성, Walk-Forward 학습, Half-Kelly 포지션 사이징
- **파라미터 최적화** — 그리드 서치 + Walk-Forward 검증 (오버피팅 방지)
- **WebSocket 실시간 가격** — Upbit WebSocket으로 REST API 호출 최소화, 자동 재연결
- **페이퍼/실매매** — 모의 체결 및 Upbit API 연동 실매매 (주문 관리, 안전 장치)
- **웹 대시보드** — Streamlit 기반 실시간 모니터링 + 백테스트 시각화
- **텔레그램 알림** — 시그널, 체결, 에러 실시간 알림

## 아키텍처

### 전체 구조

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI (typer)                         │
│  download │ backtest │ scan │ optimize │ paper │ live │ ... │
└─────┬───────────┬───────────┬───────────┬───────────┬───────┘
      │           │           │           │           │
      ▼           ▼           ▼           ▼           ▼
┌──────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│   Data   │ │    Backtest Engine   │ │     Live Engine      │
│ Layer    │ │                      │ │                      │
│          │ │ ┌──────────────────┐ │ │ ┌──────────────────┐ │
│ Fetcher  │ │ │  Candle-by-     │ │ │ │  Async Polling   │ │
│ (CCXT)   │ │ │  Candle Loop    │ │ │ │  Loop            │ │
│          │ │ │  (anti-look-    │ │ │ │                  │ │
│ Storage  │ │ │   ahead)        │ │ │ │  Order Manager   │ │
│ (Parquet)│ │ ├──────────────────┤ │ │ │  State Persist   │ │
│          │ │ │  Vectorized     │ │ │ └──────────────────┘ │
│ Indica-  │ │ │  Engine (scan)  │ │ │                      │
│ tors(19) │ │ ├──────────────────┤ │ │ ┌──────────────────┐ │
│          │ │ │  Optimizer      │ │ │ │  Notifications   │ │
│          │ │ │  Walk-Forward   │ │ │ │  (Telegram)      │ │
│          │ │ │  Parallel       │ │ │ └──────────────────┘ │
└──────────┘ └──────────┬───────────┘ └──────────┬───────────┘
                        │                        │
              ┌─────────▼────────────────────────▼──────────┐
              │              Strategy Layer                  │
              │                                             │
              │  ┌───────────┐  ┌────────────┐  ┌────────┐ │
              │  │  7 Built- │  │  Combined   │  │  LGBM  │ │
              │  │  in       │  │  Strategy   │  │  ML    │ │
              │  │  Strats   │  │  (31 Fil-   │  │  Model │ │
              │  │           │  │   ters)     │  │        │ │
              │  └───────────┘  └────────────┘  └────────┘ │
              └─────────────────────┬───────────────────────┘
                                    │
              ┌─────────────────────▼───────────────────────┐
              │           Infrastructure Layer               │
              │                                             │
              │  ┌──────────┐  ┌──────────┐  ┌───────────┐ │
              │  │ Exchange  │  │   Risk   │  │ Dashboard │ │
              │  │ (CCXT /   │  │ Manager  │  │(Streamlit)│ │
              │  │  Paper /  │  │ + Valid- │  │           │ │
              │  │  WS)      │  │  ators   │  │           │ │
              │  └──────────┘  └──────────┘  └───────────┘ │
              └─────────────────────────────────────────────┘
```

### Anti-Lookahead 백테스트 루프 (핵심 원칙)

백테스트 엔진은 캔들을 하나씩 순회하며, 전략에는 **확정된(닫힌) 캔들만** 전달합니다. 현재 미완성 캔들이나 미래 데이터에 접근할 수 없으며, 이는 `backtest/engine.py`에서 구조적으로 강제됩니다.

```
사전 계산: strategy.indicators(full_df) per symbol  ← O(N), 1회

통합 타임라인 구성 (전 심볼 타임스탬프)

각 타임스탬프 ts에 대해:
  Phase 1 — 체결:    fill_candle = symbol_candles[idx]
                     손절 확인, 대기 주문 체결
  Phase 2 — 전략:    visible_df = indicator_df[0..idx-1]  ← 과거만 참조
                     should_exit → should_entry
                     리스크 매니저 검증 → fill_candle의 시가 + 슬리피지로 체결
  Phase 3 — 기록:    가격 갱신, 자산 스냅샷 저장
```

### 모듈 의존 관계

```
cli.py ─┬─→ data/{fetcher, storage, indicators}
        ├─→ backtest/{engine, vectorized, optimizer, walk_forward, parallel}
        ├─→ strategy/{base, registry, combined, lgbm_strategy, examples/*, filters/*}
        ├─→ live/{engine, state, order_manager}
        ├─→ exchange/{ccxt_exchange, paper, ws_client}
        ├─→ risk/{manager, validators}
        ├─→ ml/{trainer, features, targets, walk_forward, parallel}
        ├─→ notifications/telegram
        └─→ dashboard/app
```

## 설치

```bash
# Python 3.11+ 필요
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 빠른 시작

### 1. 데이터 다운로드

```bash
# 전체 대상 종목 일괄 다운로드
for sym in BTC/KRW ETH/KRW XRP/KRW SOL/KRW DOGE/KRW ADA/KRW AVAX/KRW LINK/KRW; do
  tradingbot download --symbol $sym --timeframe 1h --since 2024-01-01
done

# 또는 개별 다운로드
tradingbot download --symbol BTC/KRW --timeframe 1h --since 2024-01-01

# 다운로드한 데이터 확인
tradingbot data-list

# 거래 가능한 KRW 종목 조회
tradingbot symbols
```

### 2. 백테스트

```bash
# 멀티 심볼 백테스트 (config의 전체 심볼 사용)
tradingbot backtest --strategy sma_cross

# 단일 심볼 지정
tradingbot backtest --strategy sma_cross --symbol BTC/KRW

# 기간/잔고 지정
tradingbot backtest --strategy sma_cross --symbol BTC/KRW \
  --start 2024-06-01 --end 2024-12-31 --balance 5000000
```

### 3. 전략 최적화 & Walk-Forward 검증

```bash
# 그리드 서치 최적화
tradingbot optimize --strategy sma_cross --symbol BTC/KRW --top 10

# 커스텀 파라미터 그리드
tradingbot optimize --strategy sma_cross --symbol BTC/KRW \
  --param-grid '{"fast_period": [5, 10, 15, 20], "slow_period": [30, 40, 50, 60]}'

# Walk-Forward 검증 (3개월 훈련 → 1개월 테스트 롤링)
tradingbot walk-forward --strategy sma_cross --symbol BTC/KRW \
  --train-months 3 --test-months 1
```

| 지표 | 기준 | 의미 |
|------|------|------|
| WF Efficiency | > 50% | 파라미터가 미래 데이터에서도 유효 |
| Overfitting Ratio | < 50% | 과최적화되지 않음 |

### 4. 전략 자동 스캔

```bash
# 전 전략 × 심볼 × 타임프레임 조합 랭킹
tradingbot scan --top 15

# 수익률 기준 정렬
tradingbot scan --sort-by total_return --top 10
```

### 5. 필터 조합 (코드 없이 전략 만들기)

```bash
# 추세 상승 + RSI 과매도 진입 → RSI 과매수 청산
tradingbot combine \
  --entry "trend_up:4 + rsi_oversold:30" \
  --exit "rsi_overbought:70" \
  --symbol BTC/KRW

# ML + Rule 조합 (ML 모델이 veto 필터 역할)
tradingbot combine \
  --entry "trend_up:4 + rsi_oversold:30 + lgbm_prob:0.35" \
  --exit "rsi_overbought:70" \
  --symbol BTC/KRW

# 48개 사전정의 조합 자동 스캔
tradingbot combine-scan --top 10
```

**사용 가능한 필터 (31종, 역할별 분류):**

| 역할 | 필터 | 예시 |
|------|------|------|
| **Entry** | `rsi_oversold`, `macd_cross_up`, `stoch_oversold`, `cci_oversold`, `roc_positive`, `mfi_oversold`, `ema_cross_up`, `donchian_break`, `price_breakout`, `bb_upper_break`, `lgbm_prob` | `rsi_oversold:30` |
| **Trend** | `trend_up`, `trend_down`, `ema_above`, `adx_strong`, `ichimoku_above`, `aroon_up` | `adx_strong:25` |
| **Volatility** | `atr_breakout`, `keltner_break`, `bb_squeeze`, `bb_bandwidth_low` | `atr_breakout:14:2.0` |
| **Volume** | `volume_spike`, `obv_rising`, `mfi_confirm` | `volume_spike:2.5` |
| **Exit** | `rsi_overbought`, `stoch_overbought`, `cci_overbought`, `mfi_overbought`, `zscore_extreme`, `pct_from_ma_exit`, `atr_trailing_exit` | `atr_trailing_exit:14:2.5` |

조합 규칙: 진입은 모든 필터 AND 충족 시 매수 / 청산은 하나라도 OR 충족 시 매도

### 6. ML 전략 (LightGBM)

```bash
pip install -e ".[ml]"

# 모델 학습 (Walk-Forward 검증 포함)
tradingbot ml-train --symbol BTC/KRW --timeframe 1h --train-months 3 --test-months 1

# ML 전략으로 백테스트
tradingbot ml-backtest --symbol BTC/KRW --timeframe 1h

# 전체 심볼×타임프레임 일괄 병렬 학습
tradingbot ml-train-all --workers 4
```

### 7. 페이퍼 트레이딩

```bash
tradingbot paper --strategy sma_cross --symbol BTC/KRW --balance 1000000

# WebSocket 모드 (REST API 호출 최소화)
tradingbot paper --strategy sma_cross --symbol BTC/KRW --websocket

# 필터 조합으로 실행
tradingbot paper --entry "trend_up:4 + rsi_oversold:30" \
  --exit "rsi_overbought:70" --symbol BTC/KRW

# 상태 확인
tradingbot status
```

### 8. 실매매

```bash
# API 키 설정
cp .env.example .env
# .env 편집: UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY 입력

tradingbot balance  # 잔고 확인

tradingbot live --strategy sma_cross --symbol BTC/KRW \
  --max-order 100000 --daily-loss-limit 50000

# WebSocket + Combined 템플릿
tradingbot live --strategy ML+ADXTrend --symbol BTC/KRW \
  --max-order 100000 --daily-loss-limit 50000 --websocket
```

## 권장 워크플로우

```
데이터 다운로드 → 백테스트 → 최적화 → Walk-Forward 검증
→ 페이퍼 트레이딩 (1~2주) → 소액 실매매 (10만원) → 단계적 증액
```

**Sharpe > 1.5, Max Drawdown < 15%** 인 전략을 찾은 후 페이퍼로 넘어가는 것을 권장합니다.

## 내장 전략

| 전략 | 설명 | 주요 파라미터 |
|------|------|---------------|
| `sma_cross` | SMA 골든/데드크로스 | `fast_period`, `slow_period` |
| `rsi_mean_reversion` | RSI 과매도 진입 / 과매수 청산 | `rsi_period`, `oversold`, `overbought` |
| `macd_momentum` | MACD 히스토그램 제로크로스 | `fast`, `slow`, `signal` |
| `bollinger_breakout` | 볼린저밴드 상단 돌파 / 중간밴드 이탈 | `period`, `std` |
| `multi_tf` | 상위 TF 추세 필터 + 하위 TF RSI 진입 | `higher_tf_factor`, `trend_sma_period` |
| `volume_breakout` | 거래량 급등 + 최근 고점 돌파 | `volume_spike_threshold`, `price_lookback` |
| `lgbm` | LightGBM ML 메타 모델 (15 피처 → Half-Kelly) | `entry_threshold`, `exit_threshold` |

## 커스텀 전략 작성

`strategies/` 디렉토리에 새 전략 파일을 생성:

```python
from tradingbot.strategy.base import Strategy, StrategyParams
from tradingbot.core.enums import SignalType
from tradingbot.core.models import Position, Signal

class MyStrategy(Strategy):
    name = "my_strategy"
    timeframe = "1h"
    symbols = ["BTC/KRW"]

    def indicators(self, df):
        df["sma_20"] = df["close"].rolling(20).mean()
        return df

    def should_entry(self, df, symbol):
        if df["close"].iloc[-1] > df["sma_20"].iloc[-1]:
            return Signal(
                timestamp=df.index[-1].to_pydatetime(),
                symbol=symbol,
                signal_type=SignalType.LONG_ENTRY,
                price=df["close"].iloc[-1],
            )
        return None

    def should_exit(self, df, symbol, position):
        if df["close"].iloc[-1] < df["sma_20"].iloc[-1]:
            return Signal(
                timestamp=df.index[-1].to_pydatetime(),
                symbol=symbol,
                signal_type=SignalType.LONG_EXIT,
                price=df["close"].iloc[-1],
            )
        return None
```

## 프로젝트 구조

```
trading-bot/
├── pyproject.toml                    # 패키지 메타데이터 & 의존성
├── Dockerfile / docker-compose.yml   # 컨테이너 배포
├── config/default.yaml               # 기본 설정 (심볼, 리스크, 수수료)
├── strategies/                       # 사용자 정의 전략 디렉토리
│
├── src/tradingbot/                   # 메인 패키지 (66개 모듈, ~7,400 라인)
│   ├── cli.py                        # CLI 진입점 (typer)
│   ├── config.py                     # Pydantic 설정 로더
│   │
│   ├── core/                         # 도메인 모델
│   │   ├── models.py                 #   Candle, Signal, Order, Trade, Position
│   │   ├── enums.py                  #   OrderSide, SignalType, PositionSide
│   │   └── events.py                 #   이벤트 시스템
│   │
│   ├── data/                         # 데이터 레이어
│   │   ├── fetcher.py                #   CCXT 기반 OHLCV 다운로드
│   │   ├── storage.py                #   Parquet 저장/로드/병합
│   │   └── indicators.py             #   기술적 지표 19종
│   │
│   ├── strategy/                     # 전략 프레임워크
│   │   ├── base.py                   #   Strategy ABC (indicators, entry, exit)
│   │   ├── registry.py               #   전략 레지스트리 (동적 로딩)
│   │   ├── combined.py               #   필터 조합 전략 (AND 진입 / OR 청산)
│   │   ├── lgbm_strategy.py          #   LightGBM ML 전략
│   │   ├── examples/                 #   내장 전략 7종
│   │   │   ├── sma_cross.py
│   │   │   ├── rsi_mean_reversion.py
│   │   │   ├── macd_momentum.py
│   │   │   ├── bollinger_breakout.py
│   │   │   ├── volume_breakout.py
│   │   │   └── multi_timeframe.py
│   │   └── filters/                  #   재사용 필터 31종
│   │       ├── base.py               #     BaseFilter ABC + 역할 태깅
│   │       ├── registry.py           #     48개 조합 템플릿
│   │       ├── trend.py              #     TrendUp, AdxStrong, IchimokuAbove ...
│   │       ├── momentum.py           #     RsiOversold, MacdCrossUp, StochOversold ...
│   │       ├── price.py              #     PriceBreakout, EmaAbove, DonchianBreak ...
│   │       ├── volatility.py         #     AtrBreakout, BbSqueeze, KeltnerBreak ...
│   │       ├── volume.py             #     VolumeSpike, ObvRising, MfiConfirm
│   │       ├── exit.py               #     AtrTrailingExit, ZscoreExtreme ...
│   │       └── ml.py                 #     LgbmProbFilter (ML veto + Half-Kelly)
│   │
│   ├── backtest/                     # 백테스트 엔진
│   │   ├── engine.py                 #   캔들별 루프 (anti-lookahead 핵심)
│   │   ├── vectorized.py             #   벡터화 스크리닝 (~100x 빠름)
│   │   ├── simulator.py              #   주문 체결 시뮬레이션 (슬리피지, 수수료)
│   │   ├── report.py                 #   성과 지표 (Sharpe, Sortino, MDD, ...)
│   │   ├── optimizer.py              #   그리드 서치 최적화
│   │   ├── walk_forward.py           #   Walk-Forward 검증
│   │   └── parallel.py               #   병렬 백테스트 워커
│   │
│   ├── ml/                           # 머신러닝
│   │   ├── features.py               #   피처 엔지니어링 (15개 피처)
│   │   ├── targets.py                #   4h 순방향 수익률 이진 분류 타겟
│   │   ├── trainer.py                #   LightGBM 학습/평가/저장
│   │   ├── walk_forward.py           #   퍼지드 확장 윈도우 + 엠바고
│   │   └── parallel.py               #   병렬 학습 워커
│   │
│   ├── exchange/                     # 거래소 추상화
│   │   ├── base.py                   #   BaseExchange ABC
│   │   ├── ccxt_exchange.py          #   Upbit CCXT (재시도 + 레이트 리밋)
│   │   ├── paper.py                  #   페이퍼 트레이딩 (모의 체결)
│   │   └── ws_client.py              #   Upbit WebSocket (실시간, 자동 재연결)
│   │
│   ├── live/                         # 라이브 트레이딩
│   │   ├── engine.py                 #   비동기 폴링 루프
│   │   ├── order_manager.py          #   주문 수명주기 관리
│   │   └── state.py                  #   JSON 상태 영속화 (크래시 복구)
│   │
│   ├── risk/                         # 리스크 관리
│   │   ├── manager.py                #   포지션 사이징, 드로다운 서킷브레이커
│   │   └── validators.py             #   최대 주문, 일일 손실 한도
│   │
│   ├── notifications/telegram.py     # 텔레그램 알림
│   ├── dashboard/app.py              # Streamlit 웹 대시보드
│   └── utils/                        # 유틸리티
│       ├── logging.py                #   콘솔 + JSON 파일 로테이션
│       └── time.py                   #   타임존/날짜 파싱
│
└── tests/                            # 테스트 (212개, 17개 모듈)
```

## 설정

`config/default.yaml`:

```yaml
exchange:
  name: upbit
  rate_limit_per_sec: 10

trading:
  symbols:
    - BTC/KRW
    - ETH/KRW
    - XRP/KRW
    - SOL/KRW
    - DOGE/KRW
    - ADA/KRW
    - AVAX/KRW
    - LINK/KRW
  timeframe: "1h"
  initial_balance: 1000000

risk:
  max_position_size_pct: 0.1      # 포지션당 최대 10%
  max_open_positions: 5           # 최대 5개 동시
  max_drawdown_pct: 0.20          # 20% 드로다운 서킷브레이커
  default_stop_loss_pct: 0.02     # 2% 손절
  risk_per_trade_pct: 0.01        # 거래당 1% 리스크

backtest:
  fee_rate: 0.0005                # Upbit 0.05%
  slippage_pct: 0.001             # 0.1% 슬리피지
```

텔레그램 알림: `.env`에 `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` 추가.

## Docker 배포

```bash
docker build -t trading-bot .
docker-compose up -d        # 페이퍼 트레이딩 시작
docker-compose logs -f       # 로그 확인
docker-compose down          # 중지
```

`docker-compose.yml`에서 `command`를 변경하여 전략/모드 조정:
```yaml
command: ["tradingbot", "paper", "--strategy", "sma_cross", "--symbol", "BTC/KRW"]
command: ["tradingbot", "live", "--strategy", "sma_cross", "--symbol", "BTC/KRW"]
```

## 웹 대시보드

```bash
pip install -e ".[dashboard]"
tradingbot dashboard          # http://localhost:8501
```

- **Live Monitor**: state.json 기반 실시간 equity curve, 오픈 포지션, 자동 새로고침
- **Backtest Viewer**: 전략/심볼 선택 → 백테스트 실행 → equity chart + 드로다운 + 거래 내역

## 기술 스택

| 카테고리 | 선택 |
|----------|------|
| 언어 | Python 3.11+ |
| 거래소 | ccxt (Upbit) |
| 지표 | ta |
| 설정 | pydantic + pyyaml |
| 데이터 | pyarrow (Parquet) |
| CLI | typer + rich |
| 실시간 | websockets |
| ML | lightgbm + scikit-learn |
| 대시보드 | streamlit + plotly |
| 배포 | Docker + docker-compose |
| 테스트 | pytest (212개) |
| 린트 | ruff + mypy |

## 개발

```bash
pytest tests/ -v              # 테스트
ruff check src/ tests/        # 린트
ruff format src/ tests/       # 포맷팅
mypy src/                     # 타입 체크
```
