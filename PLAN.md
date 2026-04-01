# Trading Bot 구현 계획

## 프로젝트 개요

Upbit(한국 최대 거래소) KRW 마켓 대상 알고리즘 트레이딩 봇.
Freqtrade의 전략 프레임워크, Jesse의 anti-lookahead 백테스트, NautilusTrader의 설계 철학을 참고하여 Python으로 처음부터 구축.

**핵심 선택:**
- 거래소: Upbit (KRW 마켓, 수수료 0.05%, API 레이트 리밋 10req/s)
- 전략: 방향성 매매 (기술적 지표 기반 추세추종/역추세)
- 마켓: 현물(Spot) 전용
- 언어: Python 3.11+
- 데이터: Parquet 파일 (시계열 최적)

---

## 현재 상태

| Phase | 상태 | 내용 |
|-------|------|------|
| Phase 1 | ✅ 완료 | 데이터 레이어 (다운로드, 저장, 지표) |
| Phase 2 | ✅ 완료 | 백테스트 엔진 + 버그 수정 12건 (43 tests passing) |
| Phase 3 | ⏳ 대기 | 전략 최적화 & 검증 |
| Phase 4 | ⏳ 대기 | 페이퍼 트레이딩 |
| Phase 5 | ⏳ 대기 | 실매매 |
| Phase 6 | ⏳ 대기 | 고도화 |

---

## Phase 1: 기반 구축 — 데이터 다운로드 & 지표 계산 ✅

### 목표
Upbit에서 과거 캔들 데이터를 다운로드하고, 기술적 지표를 계산할 수 있는 기반 구축.

### 구현 항목
1. **프로젝트 초기화**
   - `pyproject.toml` — hatchling 빌드, 의존성 정의
   - `.gitignore`, `.env.example`
   - `config/default.yaml`, `config/backtest.yaml`
   - 전체 디렉토리 구조 (`src/tradingbot/` 하위 모듈)

2. **도메인 모델** (`src/tradingbot/core/`)
   - `models.py` — `Candle`(frozen), `Signal`, `Order`, `Trade`, `Position`, `PortfolioState`
   - `enums.py` — `OrderSide`, `OrderType`, `OrderStatus`, `SignalType`, `PositionSide`
   - `events.py` — `CandleEvent`, `SignalEvent`, `OrderEvent`, `TradeEvent`
   - `candles_to_dataframe()` / `dataframe_to_candles()` 변환 유틸리티

3. **설정 시스템** (`src/tradingbot/config.py`)
   - Pydantic 모델: `ExchangeConfig`, `TradingConfig`, `RiskConfig`, `BacktestConfig`, `AppConfig`
   - YAML 파일 로딩 + deep merge + `.env` 오버라이드 (`EnvSettings`)
   - `load_config()`, `load_env()`

4. **데이터 페처** (`src/tradingbot/data/fetcher.py`)
   - CCXT 기반 OHLCV 다운로드 (`DataFetcher` 클래스)
   - 자동 페이지네이션 (거래소 API는 한 번에 200개 제한)
   - 레이트 리미팅 (Upbit 10req/s)
   - Naive datetime → UTC 강제 변환

5. **데이터 저장소** (`src/tradingbot/data/storage.py`)
   - Parquet 저장/로드 (`save_candles()`, `load_candles()`)
   - 기존 데이터와 자동 병합 (중복 제거, 시간순 정렬)
   - 파일 경로: `data/BTC_KRW/1h.parquet`
   - `list_available_data()` — 보유 데이터 목록

6. **기술적 지표** (`src/tradingbot/data/indicators.py`)
   - `ta` 라이브러리 래퍼 (pandas-ta는 Python 3.11+ 미지원)
   - SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic, Volume SMA
   - 모든 함수가 DataFrame을 받아서 컬럼 추가 후 반환 (체이닝 가능)

7. **유틸리티**
   - `utils/logging.py` — structlog 콘솔 출력 설정
   - `utils/time.py` — `parse_date()`, `now_utc()`

8. **CLI** (`src/tradingbot/cli.py`)
   - `tradingbot download --symbol BTC/KRW --timeframe 1h --since 2024-01-01`
   - `tradingbot data-list`
   - `tradingbot symbols --exchange upbit`

9. **테스트** (9개)
   - `test_models.py` — Candle/Signal/Trade/Position/Portfolio 테스트
   - `test_indicators.py` — SMA/EMA/RSI/MACD/BB/ATR/VolumeSMA 검증
   - `test_storage.py` — Parquet 저장/로드/병합/목록

### 마일스톤
BTC/KRW 1시간봉 다운로드 + 지표 계산 가능. CLI로 데이터 관리.

---

## Phase 2: 백테스팅 엔진 — 전략 실행 & 성과 리포트 ✅

### 목표
과거 데이터로 전략을 실행하고, 신뢰할 수 있는 성과 리포트를 출력하는 백테스트 엔진.

### 핵심 설계: Anti-Lookahead 엔진
```
For candle i (i >= 1):
  visible_df = candles[0..i-1]     ← 전략은 과거 캔들만 봄
  fill_candle = candle[i]           ← 체결은 다음 캔들에서 발생

  1. Stop loss 확인 (fill_candle의 OHLC로)
  2. 대기 주문 체결
  3. strategy.indicators(visible_df)
  4. strategy.should_exit(visible_df) → 보유 포지션 청산 판단
  5. strategy.should_entry(visible_df) → 신규 진입 판단 (손절 발동시 차단)
  6. 리스크 매니저 검증
  7. 승인된 시그널 → fill_candle.open + slippage에 체결
  8. peak_equity 업데이트 & equity 스냅샷 기록
```

### 구현 항목
1. **전략 인터페이스** (`src/tradingbot/strategy/base.py`)
   - `Strategy` 추상 클래스: `indicators()`, `should_entry()`, `should_exit()`
   - `StrategyParams` — 파라미터 컨테이너 (최적화용)
   - `param_space()` — 검색 공간 정의 (서브클래스 오버라이드)

2. **예제 전략** (`src/tradingbot/strategy/examples/`)
   - `sma_cross.py` — SMA 골든크로스/데드크로스 (fast/slow period 파라미터)
   - `rsi_mean_reversion.py` — RSI 과매도 진입 / 과매수 청산

3. **주문 시뮬레이터** (`src/tradingbot/backtest/simulator.py`)
   - `OrderSimulator` 클래스
   - Market order: 캔들 open + slippage에 체결
   - Limit order: 캔들 high/low가 지정가를 관통하면 체결
   - Stop loss: 캔들 low가 손절가 이하면 발동
   - 수수료 모델: Upbit 0.05% 기본

4. **백테스트 엔진** (`src/tradingbot/backtest/engine.py`) — **가장 중요한 파일**
   - `BacktestEngine` 클래스
   - Anti-lookahead: `visible_df = df_full.iloc[:i]` (전략은 캔들 i를 절대 못 봄)
   - `_entry_orders` dict로 심볼별 정확한 진입/청산 매칭
   - `stop_loss_fired` 플래그: 손절 발동 캔들에서 재진입 차단
   - 잔고 부족 시 수량+수수료 동시 재계산 (부동소수점 epsilon 방어)
   - 매 캔들 peak_equity 업데이트
   - 백테스트 종료 시 미청산 포지션 강제 청산

5. **성과 리포트** (`src/tradingbot/backtest/report.py`)
   - `BacktestReport` 데이터클래스
   - 지표: Sharpe ratio, Sortino ratio, max drawdown, win rate, profit factor, avg win/loss
   - Sharpe/Sortino: timeframe별 동적 연환산 (`PERIODS_PER_YEAR` 딕셔너리)
   - Sortino: 표준 downside deviation (전체 시리즈, 양수는 0으로 처리)
   - Rich 테이블로 콘솔 출력 (`print_summary()`)
   - 개별 거래 로그 (진입가, 청산가, PnL, 수익률, 기간)

6. **리스크 매니저** (`src/tradingbot/risk/manager.py`)
   - `RiskManager` 클래스
   - 포지션 사이징: 고정 비율법 (risk_per_trade_pct 기반, stop loss와 연동)
   - max_position_size_pct 캡핑
   - max_open_positions 제한
   - 드로다운 서킷브레이커 (max_drawdown_pct 초과 시 거래 중단)
   - 기본 손절가 계산 (`default_stop_loss_pct`)
   - price <= 0 방어

7. **CLI 확장**
   - `tradingbot backtest --strategy sma_cross --symbol BTC/KRW --start 2024-01-01 --end 2024-12-31 --balance 1000000`
   - 전략 맵: `sma_cross`, `rsi_mean_reversion`

8. **테스트** (총 43개)
   - `test_backtest_engine.py` — 기본 실행, 거래 생성, 수수료, equity curve, 빈 데이터
   - `TestAntiLookahead` — 결정론적 재현 검증
   - `TestBugFixes` — lookahead 방지, entry 매칭, Sharpe timeframe, zero price, peak equity, Sortino
   - `test_risk_manager.py` — 서킷브레이커, 포지션 사이징, 손절, 드로다운

### 버그 수정 이력 (12건)
| # | 심각도 | 수정 내용 |
|---|--------|-----------|
| 1 | CRITICAL | 전략이 캔들 i 종가를 보고 i 시가에 체결 → i-1까지만 보고 i 시가에 체결 |
| 2 | CRITICAL | final_balance가 미청산 포지션 무시 → equity snapshot 사용 |
| 3 | HIGH | 잔고 부족시 수량 줄인 후 수수료 미재계산 → 동시 재계산 |
| 4 | HIGH | _last_entry_order가 다른 거래와 매칭 → 심볼별 dict |
| 5 | HIGH | 손절+같은 캔들 재진입 가능 → stop_loss_fired 차단 |
| 6 | MEDIUM | Sharpe 연환산 1시간 하드코딩 → timeframe별 동적 |
| 8 | MEDIUM | price=0일 때 ZeroDivisionError → 방어 코드 |
| 9 | MEDIUM | peak_equity가 시그널 있을 때만 업데이트 → 매 캔들 업데이트 |
| 10 | MEDIUM | naive datetime이 로컬 시간으로 해석 → UTC 강제 |
| 12 | LOW | Sortino 비표준 (음수만 필터) → 표준 downside deviation |
| F6 | MEDIUM | force-close 후 final_balance가 종가 기준 → cash 기준 |
| F2 | LOW | 부동소수점 epsilon으로 cash 음수 가능 → max(0, cash) |

### 마일스톤
SMA 크로스오버 백테스트 실행, Sharpe/Sortino/드로다운 포함 성과 리포트 출력. 43개 테스트 통과.

---

## Phase 3: 전략 최적화 & 검증 ⏳

### 목표
전략 파라미터를 체계적으로 최적화하고, 오버피팅을 검증할 수 있는 도구 구축.

### 구현 항목

1. **그리드 서치 옵티마이저** (`src/tradingbot/backtest/optimizer.py`)
   - 전략의 `param_space()` 기반으로 모든 파라미터 조합 생성
   - 각 조합에 대해 백테스트 실행
   - 결과를 Sharpe, total return, max drawdown 등으로 정렬
   - `multiprocessing` 활용 병렬 실행 (CPU 코어 활용)
   - 결과 DataFrame 출력: 파라미터 + 주요 지표

2. **Walk-Forward 검증** (`src/tradingbot/backtest/walk_forward.py`)
   - 데이터를 N개 윈도우로 분할 (예: 3개월 훈련 + 1개월 테스트)
   - 각 윈도우에서 최적 파라미터 찾기 → 다음 윈도우에서 테스트
   - Out-of-sample 성과 집계
   - 오버피팅 비율 계산: (in-sample 성과 - out-of-sample 성과) / in-sample 성과
   - Walk-forward efficiency: out-of-sample Sharpe / in-sample Sharpe

3. **파라미터 안정성 분석**
   - 최적 파라미터 주변의 성과 변화 확인 (파라미터가 조금 바뀌어도 성과가 유지되는지)
   - 히트맵 시각화 (2D 파라미터 공간)

4. **추가 전략 구현**
   - MACD 모멘텀 전략 (MACD 히스토그램 방향 전환)
   - Bollinger Band 브레이크아웃 전략
   - 멀티 타임프레임 전략 (1h 지표 + 4h 추세 필터)

5. **CLI 확장**
   - `tradingbot optimize --strategy sma_cross --symbol BTC/KRW --param-grid '{"fast_period": [10,20,30], "slow_period": [40,50,60]}'`
   - `tradingbot walk-forward --strategy sma_cross --symbol BTC/KRW --train-months 3 --test-months 1`

6. **Jupyter 노트북 템플릿** (`notebooks/01_strategy_research.ipynb`)
   - 데이터 로드 → 지표 시각화 → 백테스트 실행 → 결과 분석 워크플로우

### 검증 방법
- 옵티마이저 결과 정렬 확인 (Sharpe 높은 순)
- Walk-forward: out-of-sample 성과가 in-sample의 50% 이상이면 합격
- 파라미터 안정성: 상위 10% 파라미터가 인접 값에서도 상위 30% 이내

### 마일스톤
파라미터 최적화 + walk-forward 검증으로 오버피팅 없는 전략 개발 환경 완성.

---

## Phase 4: 페이퍼 트레이딩 — 실시간 모의 매매 ⏳

### 목표
실시간 시장 데이터로 전략을 실행하되, 실제 주문 없이 모의 체결하는 시스템.

### 구현 항목

1. **거래소 추상 인터페이스** (`src/tradingbot/exchange/base.py`)
   ```python
   class BaseExchange(ABC):
       async def fetch_ohlcv(symbol, timeframe, since, limit) -> pd.DataFrame
       async def fetch_ticker(symbol) -> dict
       async def create_order(symbol, side, type, quantity, price) -> Order
       async def cancel_order(order_id, symbol) -> bool
       async def get_balance() -> dict[str, float]
       async def get_open_orders(symbol) -> list[Order]
   ```

2. **CCXT 거래소 구현** (`src/tradingbot/exchange/ccxt_exchange.py`)
   - `CcxtExchange(BaseExchange)` — CCXT async 클라이언트 래핑
   - Upbit API 키 연동 (`.env`의 `UPBIT_ACCESS_KEY`, `UPBIT_SECRET_KEY`)
   - 레이트 리미팅, 재시도 (지수 백오프), 에러 매핑
   - Phase 4에서는 읽기 전용 (fetch만): `fetch_ohlcv`, `fetch_ticker`

3. **페이퍼 거래소** (`src/tradingbot/exchange/paper.py`)
   - `PaperExchange(BaseExchange)` — 모의 체결 엔진
   - 실시간 가격 데이터 기반으로 주문 시뮬레이션
   - 잔고, 포지션, 거래 내역 메모리 관리
   - 슬리피지 + 수수료 적용 (백테스트 시뮬레이터와 동일 로직)

4. **라이브 트레이딩 엔진** (`src/tradingbot/live/engine.py`)
   - `LiveEngine` 클래스 (asyncio 기반)
   - 폴링 루프: 설정된 timeframe 간격으로 새 캔들 확인
   - 캔들 완성 감지 → 전략 실행 → 주문 생성
   - 전략 인터페이스는 백테스트와 동일 (코드 재사용)
   - Graceful shutdown: SIGINT/SIGTERM 처리

5. **상태 영속화** (`src/tradingbot/live/state.py`)
   - 현재 포지션, 대기 주문, equity 히스토리를 JSON 파일로 저장
   - 재시작 시 상태 복구 (크래시 안전)

6. **알림 시스템** (`src/tradingbot/notifications/`)
   - `telegram.py` — 텔레그램 봇 알림 (python-telegram-bot)
     - 시그널 발생, 주문 체결, 일일 리포트, 에러 알림
   - 설정: `.env`의 `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`

7. **CLI 확장**
   - `tradingbot paper --strategy sma_cross --symbol BTC/KRW` — 페이퍼 트레이딩 시작
   - `tradingbot status` — 현재 포지션/잔고/오늘 성과 확인

### 검증 방법
- 페이퍼 트레이딩 24시간 무중단 실행
- 시그널 발생 → 모의 체결 → 텔레그램 알림 전체 흐름 확인
- 프로세스 재시작 후 상태 복구 확인
- 백테스트 vs 페이퍼 결과 비교 (동일 기간, 유사한 결과여야 함)

### 마일스톤
24/7 실행, 실시간 캔들 처리, 모의 매매, 텔레그램 알림 발송. 재시작 안전.

---

## Phase 5: 실매매 ⏳

### 목표
Upbit에서 실제 KRW로 현물 매매를 수행하는 완전한 자동 트레이딩 시스템.

### 구현 항목

1. **CCXT 거래소 완성** (`src/tradingbot/exchange/ccxt_exchange.py`)
   - `create_order()` — Upbit 시장가/지정가 주문 생성
   - `cancel_order()` — 미체결 주문 취소
   - `get_open_orders()` — 미체결 주문 조회
   - `get_balance()` — KRW 및 코인 잔고 조회
   - 주문 상태 폴링 (체결 확인)

2. **주문 상태 관리** (`src/tradingbot/live/order_manager.py`)
   - 미체결 → 부분체결 → 완전체결 / 취소 상태 추적
   - 타임아웃: 일정 시간 미체결 시 자동 취소 후 시장가 재주문
   - 부분 체결 처리 로직

3. **안전 장치**
   - 최대 일일 손실 제한 (일일 드로다운 서킷브레이커)
   - 최대 주문 크기 하드 리밋 (설정 오류 방지)
   - API 에러 시 자동 재연결 (지수 백오프)
   - 네트워크 단절 감지 → 알림 + 보수적 모드 전환
   - 중복 주문 방지 (idempotency key)

4. **모니터링 강화**
   - 텔레그램: 실매매 시그널/체결 실시간 알림
   - 일일 성과 요약 (수익률, 거래 수, 최대 드로다운)
   - 에러/경고 즉시 알림 (API 에러, 서킷브레이커 발동)

5. **CLI 확장**
   - `tradingbot live --strategy sma_cross --symbol BTC/KRW --exchange upbit`
   - `tradingbot stop` — 안전하게 종료 (미체결 주문 취소, 포지션 선택적 청산)
   - `tradingbot balance` — 현재 잔고 조회

### 안전한 배포 절차
1. 최소 금액 (10만원)으로 1주일 실매매 테스트
2. 페이퍼 vs 실매매 성과 비교 (슬리피지, 체결률 차이 분석)
3. 안정적이면 금액 단계적 증가

### 검증 방법
- 소액 실매매로 주문→체결→PnL 전체 플로우 확인
- 의도적 네트워크 단절 후 복구 확인
- 서킷브레이커 강제 발동 테스트
- Upbit 점검 시간 대응 확인 (매일 04:00~04:10 KST)

### 마일스톤
Upbit에서 실제 현물 자동 매매. 리스크 관리 + 모니터링 + 알림 완비.

---

## Phase 6: 고도화 (Ongoing) ⏳

### 기능 확장
1. **멀티 심볼** — 여러 종목 동시 트레이딩 (포트폴리오 전략)
2. **WebSocket 실시간 데이터** — 폴링 대신 WebSocket으로 지연 최소화
3. **웹 대시보드** — Streamlit 또는 Dash로 실시간 성과 모니터링
   - Equity curve 차트
   - 현재 포지션/시그널 현황
   - 거래 히스토리
4. **Bybit 추가** — 두 번째 거래소 지원 (CCXT 추상화 덕분에 용이)
5. **선물/마진** — 숏 포지션, 레버리지 지원
6. **ML/AI 전략** — FreqAI 스타일 ML 지표 통합
   - scikit-learn/XGBoost로 방향 예측 피처 추가
   - Walk-forward로 오버피팅 검증 필수
7. **Docker 배포** — Docker Compose + health check + 자동 재시작
8. **데이터 갭 감지** — 캔들 누락 감지 및 자동 보정

### 성능 최적화
- 백테스트 속도: 대량 파라미터 최적화 시 Numba/Cython 가속 검토
- 메모리: 대용량 데이터셋 스트리밍 처리 (청크 단위)

---

## 기술 스택 요약

| 카테고리 | 선택 | 이유 |
|----------|------|------|
| 언어 | Python 3.11+ | 생태계, CCXT, 솔로 개발 생산성 |
| 거래소 추상화 | ccxt | Upbit 포함 100+ 거래소 지원 |
| 기술적 지표 | ta | pandas-ta가 Python 3.11+ 미지원 |
| 설정 | pydantic + pyyaml | 타입 안전 검증 + 읽기 쉬운 YAML |
| 데이터 저장 | pyarrow (Parquet) | 시계열 최적, pandas 직접 호환 |
| CLI | typer + rich | 자동 완성, 컬러 출력 |
| 로깅 | structlog | 구조화 로그, 디버깅 용이 |
| 테스트 | pytest | 표준, 커버리지, 병렬 실행 |
| 린팅 | ruff | 빠르고 올인원 |

## 프로젝트 구조

```
trading-bot/
├── pyproject.toml
├── CLAUDE.md                          # Claude Code 가이드
├── PLAN.md                            # 이 파일
├── .env.example
├── .gitignore
├── config/
│   ├── default.yaml                   # 기본 설정
│   └── backtest.yaml                  # 백테스트 설정
├── src/tradingbot/
│   ├── __init__.py
│   ├── cli.py                         # CLI 엔트리포인트
│   ├── config.py                      # Pydantic 설정
│   ├── core/
│   │   ├── models.py                  # 도메인 모델 (Candle, Order, Trade, ...)
│   │   ├── enums.py                   # 열거형
│   │   └── events.py                  # 이벤트 타입
│   ├── data/
│   │   ├── fetcher.py                 # CCXT OHLCV 다운로드
│   │   ├── storage.py                 # Parquet I/O
│   │   └── indicators.py             # 기술적 지표 래퍼
│   ├── exchange/                      # Phase 4~5
│   │   ├── base.py                    # 추상 인터페이스
│   │   ├── ccxt_exchange.py           # CCXT 구현
│   │   └── paper.py                   # 페이퍼 트레이딩
│   ├── strategy/
│   │   ├── base.py                    # 추상 전략 클래스
│   │   └── examples/
│   │       ├── sma_cross.py
│   │       └── rsi_mean_reversion.py
│   ├── risk/
│   │   ├── manager.py                 # 리스크 관리
│   │   └── validators.py              # 사전 검증 (Phase 5)
│   ├── backtest/
│   │   ├── engine.py                  # 핵심 백테스트 루프
│   │   ├── simulator.py               # 주문 체결 시뮬레이션
│   │   ├── report.py                  # 성과 리포트
│   │   ├── optimizer.py               # Phase 3: 파라미터 최적화
│   │   └── walk_forward.py            # Phase 3: Walk-forward 검증
│   ├── live/                          # Phase 4~5
│   │   ├── engine.py                  # 비동기 트레이딩 루프
│   │   ├── state.py                   # 상태 영속화
│   │   └── order_manager.py           # 주문 상태 관리
│   ├── notifications/                 # Phase 4
│   │   └── telegram.py
│   └── utils/
│       ├── logging.py
│       └── time.py
├── strategies/                        # 사용자 전략 디렉토리
├── data/                              # 다운로드 데이터 (gitignored)
├── tests/
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_indicators.py
│   ├── test_storage.py
│   ├── test_backtest_engine.py
│   ├── test_risk_manager.py
│   └── ... (Phase 3+ 추가)
└── notebooks/                         # Jupyter 리서치
```

## 인지된 한계 (Low Priority)

| # | 설명 | 대응 시점 |
|---|------|-----------|
| Bug #7 | Trade.pnl이 entry_order.quantity 사용 (현재는 정확하지만 fragile) | 부분 체결 구현 시 |
| Bug #11 | Anti-lookahead가 .copy()에 의존 (제거하면 위험) | 성능 최적화 시 주의 |
| Bug #13 | 캔들 데이터 갭 미감지 | Phase 3 데이터 검증 |
| F7 | Sharpe(ddof=1) vs Sortino(ddof=0) 미세 불일치 | 둘 다 개별적으로 올바름 |
