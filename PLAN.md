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
| Phase 2 | ✅ 완료 | 백테스트 엔진 + 버그 수정 (60+건 총 수정, 158 tests) |
| Phase 3 | ✅ 완료 | 전략 최적화 + Walk-Forward 검증 + 추가 전략 4종 |
| Phase 4 | ✅ 완료 | 페이퍼 트레이딩 (거래소 추상화, 모의 체결, 텔레그램) |
| Phase 5 | ✅ 완료 | 실매매 (주문 관리, 안전 장치, 일일 손실 한도) |
| Phase 6-1 | ✅ 완료 | 멀티 심볼 동시 매매 (백테스트 + 라이브 엔진) |
| Phase 6-7 | ✅ 완료 | Docker 배포 (Dockerfile, compose, healthcheck, 로그 관리) |
| Phase 6-3 | ✅ 완료 | 웹 대시보드 (Streamlit, Live Monitor + Backtest Viewer) |
| Phase 6-2 | ✅ 완료 | WebSocket 실시간 데이터 (Upbit ticker, 자동 재연결, 폴링 폴백) |
| 고급 전략 | ✅ 완료 | multi_tf, volume_breakout, scan CLI |
| 조합 엔진 | ✅ 완료 | 31종 필터 (역할 태깅 + ML), CombinedStrategy, combine/combine-scan CLI, 36 템플릿 |
| ML 전략 | ✅ 완료 | LightGBM 메타 모델 (33 피처, Walk-Forward 학습, Half-Kelly), ml-train/ml-backtest CLI |
| ML+Rule 조합 | ✅ 완료 | LgbmProbFilter (31번째 필터) — ML 확률을 기존 필터와 AND 조합, Half-Kelly strength, ml-train-all CLI |
| Phase 6-4~6,8 | ⏳ 대기 | Bybit, 선물/마진, 성능 최적화, 레짐 감지 (HMM) |

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

## Phase 6: 고도화 ⏳

### 6-1. 멀티 심볼 동시 트레이딩 ✅

**목표**: 여러 종목을 하나의 엔진에서 동시에 매매하는 포트폴리오 기반 트레이딩.

**현재 상태**: 엔진이 단일 심볼만 처리 (`strategy.symbols[0]`). 설정에는 8종목이 정의되어 있지만 개별 실행만 가능.

**구현 항목**:

1. **백테스트 엔진 멀티 심볼 확장** (`backtest/engine.py`)
   - `data` dict에 여러 심볼의 DataFrame 전달
   - 모든 심볼의 캔들을 시간순으로 통합 순회
   - 심볼별 독립 포지션 관리 (기존 `self.positions` dict 활용)
   - 포트폴리오 전체 equity 기반 리스크 관리
   - `max_open_positions` 제한이 포트폴리오 전체에 적용

2. **전략 인터페이스 확장** (`strategy/base.py`)
   - `should_entry(df, symbol)` / `should_exit(df, symbol, position)` 인터페이스 유지
   - 멀티 심볼 전략: 각 심볼별로 순회 호출 (기존 인터페이스 호환)
   - 교차 심볼 전략 (선택적): `should_entry_portfolio(dfs: dict, positions: dict)` 추가

3. **라이브 엔진 멀티 심볼** (`live/engine.py`)
   - 심볼별 독립 폴링 → asyncio.gather로 병렬 캔들 페치
   - 심볼별 마지막 확정 캔들 타임스탬프 추적 (`_last_candle_ts` → `dict[str, datetime]`)
   - 포지션/잔고를 포트폴리오 전체로 관리

4. **CLI 확장**
   - `tradingbot backtest --strategy sma_cross` → 설정의 모든 심볼에 대해 실행
   - `--symbol BTC/KRW` 옵션은 단일 심볼 오버라이드로 유지
   - `tradingbot paper` / `tradingbot live` → 멀티 심볼 동시 실행

5. **포트폴리오 리포트** (`backtest/report.py`)
   - 심볼별 성과 분해 (어떤 종목이 수익/손실 기여)
   - 포트폴리오 전체 Sharpe, 드로다운, 상관관계 분석

**대상 종목** (default.yaml 기준):
- 대형: BTC/KRW, ETH/KRW, XRP/KRW, SOL/KRW
- 중형: DOGE/KRW, ADA/KRW, AVAX/KRW, LINK/KRW

**검증 방법**:
- 멀티 심볼 백테스트: 8종목 동시 실행, 포지션 겹침 없이 `max_open_positions` 준수 확인
- 단일 심볼 백테스트와 결과 비교 (개별 심볼 성과 합 ≈ 포트폴리오 성과)
- 라이브 엔진: 8종목 동시 폴링, 각 심볼 독립 시그널 확인

---

### 6-2. WebSocket 실시간 데이터 ✅

**목표**: 폴링 대신 WebSocket으로 실시간 가격/캔들 수신, 지연 최소화.

**구현 항목**:

1. **WebSocket 클라이언트** (`src/tradingbot/exchange/ws_client.py`)
   - Upbit WebSocket API 연동 (wss://api.upbit.com/websocket/v1)
   - 실시간 체결가(trade), 현재가(ticker), 캔들(candle) 스트림
   - 자동 재연결 (연결 끊김 시 지수 백오프)
   - 하트비트/핑퐁 처리

2. **데이터 파이프라인**
   - WebSocket → 내부 이벤트 큐 (asyncio.Queue)
   - 캔들 완성 감지: 실시간 체결 데이터로 캔들 빌드 또는 캔들 스트림 구독
   - REST API 폴백: WebSocket 연결 실패 시 기존 폴링으로 자동 전환

3. **라이브 엔진 통합**
   - `LiveEngine`이 WebSocket 클라이언트를 선택적으로 사용
   - WebSocket 모드: 이벤트 드리븐 (캔들 완성 이벤트 → 전략 실행)
   - 폴링 모드: 기존 로직 유지 (폴백)

**의존성**: `websockets` 패키지 추가

**검증 방법**:
- WebSocket 연결 → 실시간 BTC/KRW 체결 데이터 수신 확인
- 의도적 연결 끊기 → 자동 재연결 확인
- WebSocket vs 폴링 모드 시그널 비교 (동일 시점에 동일 시그널)

---

### 6-3. 웹 대시보드 ✅

**목표**: 실시간 성과 모니터링 웹 UI.

**구현 항목**:

1. **Streamlit 대시보드** (`src/tradingbot/dashboard/app.py`)
   - 실시간 equity curve 차트 (state.json에서 로드)
   - 현재 오픈 포지션 테이블 (심볼, 진입가, 현재가, 미실현 손익)
   - 거래 히스토리 (최근 N개 거래, PnL 색상 표시)
   - 일일/주간/월간 성과 요약
   - 전략 파라미터 표시

2. **백테스트 결과 시각화**
   - Equity curve 인터랙티브 차트 (plotly)
   - 드로다운 구간 하이라이트
   - 거래 마커 (진입/청산 포인트를 캔들 차트에 표시)
   - 파라미터 최적화 히트맵

3. **CLI 통합**
   - `tradingbot dashboard` → Streamlit 서버 시작
   - `tradingbot dashboard --backtest-report report.json` → 백테스트 결과 시각화

**의존성**: `streamlit`, `plotly` 추가

---

### 6-4. Bybit 거래소 추가

**목표**: 두 번째 거래소 지원으로 거래소 다변화.

**구현 항목**:

1. **Bybit 설정** (`config/default.yaml`)
   - `exchange.name: bybit` 옵션
   - USDT 마켓 지원 (BTC/USDT, ETH/USDT 등)
   - Bybit 수수료: 0.1% maker/taker

2. **CcxtExchange 호환성 확인**
   - CCXT가 Bybit 지원하므로 기존 `CcxtExchange` 클래스 그대로 사용
   - Bybit 특화 레이트 리밋 조정
   - Bybit API 키 환경변수 추가 (`BYBIT_ACCESS_KEY`, `BYBIT_SECRET_KEY`)

3. **교차 거래소 전략** (선택적)
   - 동일 전략을 Upbit과 Bybit에서 동시 실행
   - 거래소 간 가격 차이 모니터링

**검증 방법**:
- Bybit USDT 마켓 데이터 다운로드 + 백테스트
- Upbit vs Bybit 동일 전략 성과 비교

---

### 6-5. 선물/마진 트레이딩

**목표**: 숏 포지션 + 레버리지 지원.

**구현 항목**:

1. **도메인 모델 확장**
   - `PositionSide`에 `SHORT` 추가 (이미 구조적으로 준비됨)
   - `Position`에 `leverage` 필드 추가
   - `Trade.pnl` 숏 포지션 계산 로직 추가

2. **전략 인터페이스 확장**
   - `SignalType`에 `SHORT_ENTRY`, `SHORT_EXIT` 추가
   - 기존 LONG 전략과 동일 인터페이스로 SHORT 전략 작성 가능

3. **리스크 매니저 확장**
   - 레버리지별 마진 요구량 계산
   - 청산 가격 계산 및 경고
   - 펀딩비 시뮬레이션 (백테스트)

4. **거래소 연동**
   - Bybit 선물 마켓 지원 (Upbit은 선물 미지원)
   - 레버리지 설정 API

**주의사항**: 레버리지 거래는 손실이 증폭되므로, 충분한 백테스트 + 페이퍼 트레이딩 후 진행.

---

### 6-6. ML/AI 전략

**목표**: FreqAI 스타일의 머신러닝 기반 시그널 생성.

**구현 항목**:

1. **ML 피처 엔진** (`src/tradingbot/ml/features.py`)
   - 기술적 지표 + 가격 패턴을 피처로 변환
   - 라벨링: 미래 N시간 수익률 기반 (상승/하락/횡보)
   - Walk-forward 방식 피처 생성 (미래 데이터 누수 방지)

2. **모델 학습** (`src/tradingbot/ml/trainer.py`)
   - scikit-learn: RandomForest, GradientBoosting
   - XGBoost/LightGBM: 빠른 학습, 피처 중요도 분석
   - Walk-forward 교차 검증 (시계열 분할)

3. **ML 전략** (`src/tradingbot/strategy/examples/ml_strategy.py`)
   - 학습된 모델로 방향 예측 → 확률 기반 시그널 생성
   - 기존 Strategy 인터페이스 준수
   - 모델 주기적 재학습 (라이브 모드)

4. **오버피팅 방지**
   - 순수 OOS (out-of-sample) 평가 필수
   - Walk-forward validation으로 일반화 성능 검증
   - 피처 중요도 분석으로 노이즈 피처 제거

**의존성**: `scikit-learn`, `xgboost` 또는 `lightgbm`

---

### 6-7. Docker 배포 ✅

**목표**: 안정적인 24/7 운영을 위한 컨테이너화.

**구현 항목**:

1. **Dockerfile**
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY pyproject.toml .
   RUN pip install .
   COPY src/ src/
   COPY config/ config/
   CMD ["tradingbot", "live", "--strategy", "sma_cross", "--symbol", "BTC/KRW"]
   ```

2. **docker-compose.yml**
   ```yaml
   services:
     bot:
       build: .
       env_file: .env
       volumes:
         - ./data:/app/data
         - ./state.json:/app/state.json
       restart: unless-stopped
       healthcheck:
         test: ["CMD", "python", "-c", "import tradingbot"]
         interval: 30s
         timeout: 10s
         retries: 3
   ```

3. **운영 기능**
   - Health check 엔드포인트
   - 로그 파일 로테이션 (structlog → JSON file output)
   - 자동 재시작 (crash recovery)
   - 상태 모니터링 (state.json 마운트)

---

### 6-8. 백테스트 성능 최적화

**목표**: 대량 파라미터 최적화 시 속도 개선.

**구현 항목**:

1. **인디케이터 캐싱**
   - 동일 데이터에 대한 인디케이터 재계산 방지
   - 파라미터가 다른 경우만 재계산

2. **Numba JIT 가속** (선택적)
   - 핵심 루프 (백테스트 엔진, 시뮬레이터)에 `@njit` 적용
   - 현재 `.copy()` 기반 anti-lookahead를 배열 인덱싱으로 대체

3. **병렬 최적화 개선**
   - `ProcessPoolExecutor` → `multiprocessing.Pool` (메모리 효율)
   - 대규모 그리드: 결과를 디스크에 스트리밍 저장 (메모리 제한 방지)

---

## Phase 6 우선순위

| 순위 | 항목 | 이유 |
|------|------|------|
| 1 | 6-1. 멀티 심볼 | 8종목 동시 운영의 핵심. 현재 가장 필요 |
| 2 | 6-7. Docker 배포 | 24/7 안정 운영 필수 |
| 3 | 6-2. WebSocket | 지연 최소화, 실시간성 향상 |
| 4 | 6-3. 웹 대시보드 | 모니터링 편의성 |
| 5 | 6-4. Bybit | 거래소 다변화 |
| 6 | 6-8. 성능 최적화 | 대규모 최적화 시 필요 |
| 7 | 6-6. ML/AI | 고급 전략, 충분한 데이터 확보 후 |
| 8 | 6-5. 선물/마진 | 리스크 높음, 마지막 단계 |

## TODO: 실사용 단계

개발은 Phase 6-1/2/3/7까지 완료. 이제 실제 사용하면서 필요에 따라 추가 개발.

### Step 1: 데이터 수집
```bash
for sym in BTC/KRW ETH/KRW XRP/KRW SOL/KRW DOGE/KRW ADA/KRW AVAX/KRW LINK/KRW; do
  tradingbot download --symbol $sym --timeframe 1h --since 2024-01-01
done
```

### Step 2: 전략 자동 스캔
```bash
tradingbot scan --top 15
```
→ 6전략 × 8심볼 × 3타임프레임 = 144 조합 자동 실행, Sharpe 기준 랭킹

### Step 3: 최적화 + Walk-Forward 검증
```bash
tradingbot optimize --strategy <best_strategy> --top 10
tradingbot walk-forward --strategy <best_strategy> --train-months 3 --test-months 1
```
→ WF Efficiency > 50%, Overfitting Ratio < 50% 확인

### Step 4: 페이퍼 트레이딩 (1~2주)
```bash
tradingbot paper --strategy <best_strategy> --websocket
tradingbot dashboard  # 대시보드로 모니터링
```

### Step 5: 소액 실매매 (10만원)
```bash
tradingbot live --strategy <best_strategy> --symbol BTC/KRW \
  --max-order 100000 --daily-loss-limit 50000 --websocket
```
→ 페이퍼 vs 실매매 결과 비교, 슬리피지/체결률 분석

### Step 6: 단계적 증액
→ 안정적이면 금액 증가, 종목 추가

## 추후 개발 (필요 시)

| 항목 | 시점 | 트리거 |
|------|------|--------|
| ~~조합 엔진~~ | ✅ 완료 | combine/combine-scan CLI, 15 템플릿, 9 필터 |
| **조합 생성기** | combine-scan 결과 부족 시 | `combine-generate` → 자동 조합 생성 → scan 파이프라인 |
| 6-8. 성능 최적화 | 백테스트 속도가 느릴 때 | 대규모 파라미터 그리드 실행 시 |
| 6-4. Bybit | Upbit에 불만 또는 USDT 마켓 필요 시 | 해외 거래소 접근 필요 |
| 6-6. ML/AI | 기본 전략 한계 체감 시 | 충분한 데이터(6개월+) 확보 후 |
| 6-5. 선물/마진 | 현물 경험 충분히 쌓은 후 | 숏/레버리지 전략 필요 시 |

## 인지된 한계 — 모두 해결됨

Phase 1~6에서 발견된 60+건의 버그가 14+ 리뷰 라운드를 거쳐 모두 수정됨. 현재 미해결 이슈 0건. 107개 테스트 통과.

잔여 설계 한계 (버그 아님, 의도적 설계):
- 인디케이터 함수가 DataFrame을 in-place 변경 (`.copy()` + assert로 이중 보호, 성능상 의도적)
- Walk-forward 검증은 단일 심볼 기준 윈도우 생성 (심볼별 데이터 기간 차이 시 복잡성 증가, 현재 설계가 합리적)
