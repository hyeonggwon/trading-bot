# Trading Bot

Upbit(한국 거래소) KRW 마켓 대상 알고리즘 트레이딩 봇.

Freqtrade의 전략 프레임워크, Jesse의 anti-lookahead 백테스트, NautilusTrader의 설계 철학을 참고하여 Python으로 구축.

## 주요 기능

- **Anti-lookahead 백테스트 엔진** — 전략은 과거 캔들만 접근, 체결은 다음 캔들 시가에 발생
- **멀티 심볼 동시 매매** — 여러 종목을 하나의 포트폴리오로 동시 운영
- **7가지 내장 전략** — SMA, RSI, MACD, 볼린저, 멀티타임프레임, 거래량 돌파, LightGBM ML
- **전략 자동 스캔** — 전 전략 × 심볼 × 타임프레임 조합 자동 백테스트 + 랭킹
- **필터 조합 엔진** — 코드 없이 CLI로 필터 조합 (31종 필터, 5가지 역할 태깅, AND 진입 / OR 청산)
- **ML 전략 (LightGBM)** — 33개 피처 자동 생성, Walk-Forward 학습, Half-Kelly 포지션 사이징
- **파라미터 최적화** — 그리드 서치 + Walk-Forward 검증 (오버피팅 방지)
- **WebSocket 실시간 가격** — Upbit WebSocket으로 REST API 호출 최소화, 자동 재연결 + 쿨다운
- **페이퍼 트레이딩** — 실시간 데이터 + 모의 체결
- **실매매** — Upbit API 연동, 주문 관리, 안전 장치 (일일 손실 한도, 주문 크기 제한)
- **웹 대시보드** — Streamlit 기반 실시간 모니터링 + 백테스트 시각화
- **텔레그램 알림** — 시그널, 체결, 에러 실시간 알림

## 설치

```bash
# Python 3.11+ 필요
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

# 다른 전략들
tradingbot backtest --strategy rsi_mean_reversion
tradingbot backtest --strategy macd_momentum
tradingbot backtest --strategy bollinger_breakout
```

### 3. 전략 최적화

```bash
# 기본 파라미터 공간으로 최적화
tradingbot optimize --strategy sma_cross --symbol BTC/KRW --top 10

# 커스텀 파라미터 그리드
tradingbot optimize --strategy sma_cross --symbol BTC/KRW \
  --param-grid '{"fast_period": [5, 10, 15, 20], "slow_period": [30, 40, 50, 60]}'
```

### 4. Walk-Forward 검증

오버피팅 여부를 확인하는 가장 중요한 단계.

```bash
# 3개월 훈련 → 1개월 테스트 롤링
tradingbot walk-forward --strategy sma_cross --symbol BTC/KRW \
  --train-months 3 --test-months 1
```

| 지표 | 기준 | 의미 |
|------|------|------|
| WF Efficiency | > 50% | 파라미터가 미래 데이터에서도 유효 |
| Overfitting Ratio | < 50% | 과최적화되지 않음 |

### 4-1. 전략 자동 스캔

모든 전략 × 심볼 × 타임프레임 조합을 자동으로 백테스트하고 랭킹:

```bash
# Sharpe 기준 상위 15개 조합 찾기
tradingbot scan --top 15

# 수익률 기준 정렬
tradingbot scan --sort-by total_return --top 10
```

### 4-2. 필터 조합 (코드 없이 전략 만들기)

여러 필터를 레고처럼 조합하여 커스텀 전략을 만들 수 있습니다:

```bash
# 추세 상승 + RSI 과매도 진입 → RSI 과매수 청산
tradingbot combine \
  --entry "trend_up:4 + rsi_oversold:30" \
  --exit "rsi_overbought:70" \
  --symbol BTC/KRW

# 거래량 급등 + 가격 돌파 → EMA 이탈 청산
tradingbot combine \
  --entry "volume_spike:2.5 + price_breakout:10" \
  --exit "ema_above:20" \
  --symbol BTC/KRW

# 36개 사전정의 조합 자동 스캔
tradingbot combine-scan --top 10

# ML + Rule 조합 (ML 모델이 거부권 역할)
tradingbot combine \
  --entry "trend_up:4 + rsi_oversold:30 + lgbm_prob:0.55" \
  --exit "rsi_overbought:70" \
  --symbol BTC/KRW
```

**사용 가능한 필터 (31종, 역할별 분류):**

| 역할 | 필터 | 예시 |
|------|------|------|
| **Entry Signal** | `rsi_oversold`, `macd_cross_up`, `stoch_oversold`, `cci_oversold`, `roc_positive`, `mfi_oversold`, `ema_cross_up`, `donchian_break`, `price_breakout`, `bb_upper_break`, `lgbm_prob` | `rsi_oversold:30`, `lgbm_prob:0.55` |
| **Trend Filter** | `trend_up`, `trend_down`, `ema_above`, `adx_strong`, `ichimoku_above`, `aroon_up` | `adx_strong:25`, `ichimoku_above` |
| **Volatility Filter** | `atr_breakout`, `keltner_break`, `bb_squeeze`, `bb_bandwidth_low` | `atr_breakout:14:2.0`, `bb_squeeze` |
| **Volume Confirm** | `volume_spike`, `obv_rising`, `mfi_confirm` | `volume_spike:2.5`, `obv_rising:20` |
| **Exit Signal** | `rsi_overbought`, `stoch_overbought`, `cci_overbought`, `mfi_overbought`, `zscore_extreme`, `pct_from_ma_exit`, `atr_trailing_exit` | `atr_trailing_exit:14:2.5` |

조합 규칙: `Entry + Trend Filter + Volume Confirm → Exit`
진입: 모든 필터 AND 충족 시 매수 / 청산: 하나라도 OR 충족 시 매도
`lgbm_prob` 필터 사용 시 ML 확률 기반 Half-Kelly 포지션 사이징 자동 적용

### 4-3. ML 전략 (LightGBM)

기존 19개 인디케이터 값을 피처로 사용하는 LightGBM 메타 모델:

```bash
# ML 의존성 설치
pip install -e ".[ml]"

# 모델 학습 (Walk-Forward 검증 포함)
tradingbot ml-train --symbol BTC/KRW --timeframe 1h --train-months 3 --test-months 1

# ML 전략으로 백테스트
tradingbot ml-backtest --symbol BTC/KRW --timeframe 1h

# 모든 다운로드된 심볼×타임프레임 일괄 학습
tradingbot ml-train-all

# 특정 타임프레임만 학습
tradingbot ml-train-all --timeframe 1h

# 개별 심볼 학습
tradingbot ml-train --symbol ETH/KRW --timeframe 1h
```

기존 전략과 동일하게 `--strategy lgbm`으로도 실행 가능:
```bash
tradingbot backtest --strategy lgbm --symbol BTC/KRW
tradingbot paper --strategy lgbm --symbol BTC/KRW
```

### 5. 페이퍼 트레이딩

```bash
# 실시간 데이터로 모의 매매 (Ctrl+C로 중지)
tradingbot paper --strategy sma_cross --symbol BTC/KRW --balance 1000000

# WebSocket 모드 (Upbit 실시간 가격, REST API 호출 최소화)
tradingbot paper --strategy sma_cross --symbol BTC/KRW --websocket

# 다른 터미널에서 상태 확인
tradingbot status
```

### 6. 실매매

```bash
# 1) API 키 설정
cp .env.example .env
# .env 파일 편집: UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY 입력

# 2) 잔고 확인
tradingbot balance

# 3) 실매매 시작 (소액부터)
tradingbot live --strategy sma_cross --symbol BTC/KRW \
  --max-order 100000 --daily-loss-limit 50000

# WebSocket 모드로 실매매 (실시간 가격, REST 호출 최소화)
tradingbot live --strategy sma_cross --symbol BTC/KRW \
  --max-order 100000 --daily-loss-limit 50000 --websocket
```

## 권장 워크플로우

```
데이터 다운로드 → 백테스트 → 최적화 → Walk-Forward 검증
→ 페이퍼 트레이딩 (1~2주) → 소액 실매매 (10만원) → 단계적 증액
```

백테스트에서 **Sharpe > 1.5, Max Drawdown < 15%** 인 전략을 찾은 후 페이퍼로 넘어가는 것을 권장합니다.

## 내장 전략

| 전략 | 설명 | 주요 파라미터 |
|------|------|---------------|
| `sma_cross` | SMA 골든/데드크로스 | `fast_period`, `slow_period` |
| `rsi_mean_reversion` | RSI 과매도 진입 / 과매수 청산 | `rsi_period`, `oversold`, `overbought` |
| `macd_momentum` | MACD 히스토그램 제로크로스 | `fast`, `slow`, `signal` |
| `bollinger_breakout` | 볼린저밴드 상단 돌파 / 중간밴드 이탈 | `period`, `std` |
| `multi_tf` | 상위 TF 추세 필터 + 하위 TF RSI 진입 | `higher_tf_factor`, `trend_sma_period`, `rsi_period` |
| `volume_breakout` | 거래량 급등 + 최근 고점 돌파 | `volume_spike_threshold`, `price_lookback`, `exit_ema_period` |
| `lgbm` | LightGBM ML 메타 모델 (33 피처 → 확률 → Half-Kelly) | `entry_threshold`, `exit_threshold` |

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
        # 지표 컬럼 추가
        df["sma_20"] = df["close"].rolling(20).mean()
        return df

    def should_entry(self, df, symbol):
        # 진입 조건 → Signal 반환 또는 None
        if df["close"].iloc[-1] > df["sma_20"].iloc[-1]:
            return Signal(
                timestamp=df.index[-1].to_pydatetime(),
                symbol=symbol,
                signal_type=SignalType.LONG_ENTRY,
                price=df["close"].iloc[-1],
            )
        return None

    def should_exit(self, df, symbol, position):
        # 청산 조건 → Signal 반환 또는 None
        if df["close"].iloc[-1] < df["sma_20"].iloc[-1]:
            return Signal(
                timestamp=df.index[-1].to_pydatetime(),
                symbol=symbol,
                signal_type=SignalType.LONG_EXIT,
                price=df["close"].iloc[-1],
            )
        return None
```

## 텔레그램 알림 설정

`.env` 파일에 추가:

```
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

설정하면 paper/live 모드에서 시그널, 체결, 에러를 자동으로 알림받습니다.

## 설정

`config/default.yaml`에서 기본 설정을 변경할 수 있습니다:

```yaml
exchange:
  name: upbit
  rate_limit_per_sec: 10

trading:
  symbols:
    # 대형
    - BTC/KRW
    - ETH/KRW
    - XRP/KRW
    - SOL/KRW
    # 중형
    - DOGE/KRW
    - ADA/KRW
    - AVAX/KRW
    - LINK/KRW
  timeframe: "1h"
  initial_balance: 1000000

risk:
  max_position_size_pct: 0.1      # 포지션당 최대 10%
  max_open_positions: 5           # 8종목 중 최대 5개 동시
  max_drawdown_pct: 0.20           # 20% 드로다운 서킷브레이커
  default_stop_loss_pct: 0.02      # 2% 손절
  risk_per_trade_pct: 0.01         # 거래당 1% 리스크

backtest:
  fee_rate: 0.0005                 # Upbit 0.05%
  slippage_pct: 0.001              # 0.1% 슬리피지
```

## Docker 배포

```bash
# 빌드
docker build -t trading-bot .

# 페이퍼 트레이딩 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 실매매로 전환 (docker-compose.yml의 command 주석 해제 후)
docker-compose up -d

# 중지
docker-compose down

# 상태 확인
docker-compose ps
```

`docker-compose.yml`에서 `command`를 변경하여 전략/심볼/모드를 조정합니다:
```yaml
# 페이퍼 트레이딩 (기본)
command: ["tradingbot", "paper", "--strategy", "sma_cross", "--symbol", "BTC/KRW"]

# 실매매
command: ["tradingbot", "live", "--strategy", "sma_cross", "--symbol", "BTC/KRW"]
```

## 웹 대시보드

```bash
# 대시보드 의존성 설치
pip install -e ".[dashboard]"

# 대시보드 실행
tradingbot dashboard
```

브라우저에서 http://localhost:8501 접속. 두 가지 모드:

- **Live Monitor**: state.json 기반 실시간 equity curve, 오픈 포지션, 자동 새로고침
- **Backtest Viewer**: 전략/심볼 선택 → 백테스트 실행 → equity chart + 드로다운 + 거래 내역

## 개발

```bash
# 테스트 실행
pytest tests/ -v

# 린트
ruff check src/ tests/

# 포맷팅
ruff format src/ tests/

# 타입 체크
mypy src/
```

## 프로젝트 구조

```
trading-bot/
├── Dockerfile              # 컨테이너 빌드
├── docker-compose.yml      # 서비스 오케스트레이션
├── scripts/healthcheck.py  # Docker 헬스체크
├── config/                 # YAML 설정
├── src/tradingbot/
│   ├── core/           # 도메인 모델 (Candle, Order, Trade, Position)
│   ├── data/           # 데이터 다운로드, 저장(Parquet), 기술적 지표
│   ├── strategy/       # 전략 프레임워크 + 내장 전략 7종 + 31종 필터 조합 엔진
│   ├── backtest/       # 백테스트 엔진, 옵티마이저, Walk-Forward
│   ├── risk/           # 리스크 매니저, 사전 거래 검증
│   ├── exchange/       # 거래소 추상화 (Upbit CCXT, 페이퍼)
│   ├── live/           # 라이브 엔진, 주문 관리, 상태 영속화 (원자적 쓰기)
│   ├── notifications/  # 텔레그램 알림
│   ├── dashboard/      # Streamlit 웹 대시보드
│   └── utils/          # 로깅 (콘솔 + JSON 파일 로테이션), 시간 유틸리티
└── tests/              # 158개 테스트
```

## 기술 스택

| 카테고리 | 선택 |
|----------|------|
| 언어 | Python 3.11+ |
| 거래소 | ccxt (Upbit) |
| 지표 | ta |
| 설정 | pydantic + pyyaml |
| 데이터 | pyarrow (Parquet) |
| CLI | typer + rich |
| 실시간 | websockets (Upbit WebSocket) |
| 대시보드 | streamlit + plotly |
| 배포 | Docker + docker-compose |
| 테스트 | pytest (158개) |
