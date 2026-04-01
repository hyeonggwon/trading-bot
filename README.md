# Trading Bot

Upbit(한국 거래소) KRW 마켓 대상 알고리즘 트레이딩 봇.

Freqtrade의 전략 프레임워크, Jesse의 anti-lookahead 백테스트, NautilusTrader의 설계 철학을 참고하여 Python으로 구축.

## 주요 기능

- **Anti-lookahead 백테스트 엔진** — 전략은 과거 캔들만 접근, 체결은 다음 캔들 시가에 발생
- **멀티 심볼 동시 매매** — 여러 종목을 하나의 포트폴리오로 동시 운영
- **4가지 내장 전략** — SMA 크로스오버, RSI 역추세, MACD 모멘텀, 볼린저 밴드 브레이크아웃
- **파라미터 최적화** — 그리드 서치 + Walk-Forward 검증 (오버피팅 방지)
- **페이퍼 트레이딩** — 실시간 데이터 + 모의 체결
- **실매매** — Upbit API 연동, 주문 관리, 안전 장치 (일일 손실 한도, 주문 크기 제한)
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

### 5. 페이퍼 트레이딩

```bash
# 실시간 데이터로 모의 매매 (Ctrl+C로 중지)
tradingbot paper --strategy sma_cross --symbol BTC/KRW --balance 1000000

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
src/tradingbot/
├── core/           # 도메인 모델 (Candle, Order, Trade, Position)
├── data/           # 데이터 다운로드, 저장(Parquet), 기술적 지표
├── strategy/       # 전략 프레임워크 + 내장 전략 4종
├── backtest/       # 백테스트 엔진, 옵티마이저, Walk-Forward
├── risk/           # 리스크 매니저, 사전 거래 검증
├── exchange/       # 거래소 추상화 (Upbit CCXT, 페이퍼)
├── live/           # 라이브 엔진, 주문 관리, 상태 영속화
├── notifications/  # 텔레그램 알림
└── utils/          # 로깅, 시간 유틸리티
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
| 테스트 | pytest (97개) |
