# Trading Bot 구현 계획

## 프로젝트 개요

Upbit(한국 최대 거래소) KRW 마켓 대상 알고리즘 트레이딩 봇.
Freqtrade의 전략 프레임워크, Jesse의 anti-lookahead 백테스트, NautilusTrader의 설계 철학을 참고하여 Python으로 구축.

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
| Phase 1 | ✅ 완료 | 데이터 레이어 (다운로드, 저장, 지표 19종) |
| Phase 2 | ✅ 완료 | 백테스트 엔진 + 버그 수정 (60+건) |
| Phase 3 | ✅ 완료 | 전략 최적화 + Walk-Forward 검증 + 추가 전략 4종 |
| Phase 4 | ✅ 완료 | 페이퍼 트레이딩 (거래소 추상화, 모의 체결, 텔레그램) |
| Phase 5 | ✅ 완료 | 실매매 (주문 관리, 안전 장치, 일일 손실 한도) |
| Phase 6-1 | ✅ 완료 | 멀티 심볼 동시 매매 (백테스트 + 라이브 엔진) |
| Phase 6-2 | ✅ 완료 | WebSocket 실시간 데이터 (Upbit ticker, 자동 재연결, 폴링 폴백) |
| Phase 6-3 | ✅ 완료 | 웹 대시보드 (Streamlit, Live Monitor + Backtest Viewer) |
| Phase 6-7 | ✅ 완료 | Docker 배포 (Dockerfile, compose, healthcheck, 로그 관리) |
| 고급 전략 | ✅ 완료 | multi_tf, volume_breakout, scan CLI |
| 조합 엔진 | ✅ 완료 | 31종 필터 (역할 태깅 + ML), CombinedStrategy, combine/combine-scan CLI, 48 템플릿 |
| ML 전략 | ✅ 완료 | LightGBM 메타 모델 (15 피처, Walk-Forward 학습, Half-Kelly), ml-train/ml-backtest CLI |
| ML+Rule 조합 | ✅ 완료 | LgbmProbFilter — ML 확률을 veto 필터로 사용 (threshold 0.35), Half-Kelly strength |
| ML 파이프라인 개선 | ✅ 완료 | Feature 36→15 축소, median best_iteration, scale_pos_weight 1.0, LOG_LEVEL 지원 |
| Combined 전략 배포 | ✅ 완료 | paper/live/backtest에서 템플릿 이름으로 실행, --entry/--exit 커스텀 조합 |
| 성능 최적화 | ✅ 완료 | 인디케이터 사전 계산 (O(N²)→O(N)), 벡터화 스크리닝 엔진 (combine-scan 1.5h→3.5min) |
| CLI UX 개선 | ✅ 완료 | Rich Progress 바 (combine-scan, ml-train-all, scan, optimize, walk-forward) |
| ML 병렬 학습 | ✅ 완료 | ml-train-all --workers N (ProcessPoolExecutor + spawn) |
| 클라우드 배포 | ⏳ TODO | AWS/GCP에 Docker 배포 (24/7 무중단) |
| Phase 6-4 | ⏳ 대기 | Bybit 거래소 추가 |
| Phase 6-5 | ⏳ 대기 | 선물/마진 트레이딩 |
| Phase 6-8 | ⏳ 대기 | 레짐 감지 (HMM) |

**현재 테스트: 212개 (17개 모듈)**

---

## 완료된 Phase 요약

### Phase 1: 데이터 레이어 ✅
CCXT 기반 OHLCV 다운로드, Parquet 저장/로드/병합, 기술적 지표 19종, CLI (download, data-list, symbols).

### Phase 2: 백테스트 엔진 ✅

**핵심 설계: Anti-Lookahead 엔진**
```
For candle i (i >= 1):
  visible_df = candles[0..i-1]     ← 전략은 과거 캔들만 봄
  fill_candle = candle[i]           ← 체결은 다음 캔들에서 발생

  1. Stop loss 확인 (fill_candle의 OHLC로)
  2. 대기 주문 체결
  3. strategy.indicators(visible_df)
  4. strategy.should_exit(visible_df)
  5. strategy.should_entry(visible_df) — 손절 발동시 차단
  6. 리스크 매니저 검증 → fill_candle.open + slippage에 체결
  7. peak_equity 업데이트 & equity 스냅샷 기록
```

**주요 버그 수정 (12건):**

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

### Phase 3: 전략 최적화 & 검증 ✅
그리드 서치 옵티마이저 (병렬 실행), Walk-Forward 검증 (WF Efficiency, Overfitting Ratio), 추가 전략 4종 (MACD, 볼린저, 멀티타임프레임, 거래량 돌파).

### Phase 4: 페이퍼 트레이딩 ✅
거래소 추상 인터페이스 (BaseExchange), CCXT 구현 (Upbit), PaperExchange (모의 체결), LiveEngine (asyncio 폴링), 상태 영속화 (JSON, atomic write), 텔레그램 알림.

### Phase 5: 실매매 ✅
CCXT 주문 생성/취소/조회, OrderManager (미체결 → 체결 추적, 타임아웃 재주문), 안전 장치 (일일 손실 한도, 최대 주문 크기, 쿨다운).

### Phase 6 완료 항목 ✅
- **6-1. 멀티 심볼**: 백테스트/라이브 엔진이 여러 심볼 동시 처리, 포트폴리오 전체 equity 기반 리스크 관리
- **6-2. WebSocket**: Upbit WebSocket 실시간 ticker, 자동 재연결 (지수 백오프, 50회 실패 후 5분 쿨다운), 폴링 폴백
- **6-3. 웹 대시보드**: Streamlit (Live Monitor + Backtest Viewer), plotly 차트
- **6-7. Docker 배포**: python:3.11-slim, non-root user, healthcheck, volume 마운트 (data/logs/state/models)
- **조합 엔진**: 31종 필터 (5가지 역할 태깅), CombinedStrategy (AND 진입 / OR 청산), 48개 사전정의 템플릿, combine-scan CLI
- **ML 전략**: LightGBM 15 피처 (8 인디케이터에서 추출), Walk-Forward 학습 (purged expanding window + embargo), Half-Kelly 포지션 사이징
- **성능 최적화**: 인디케이터 사전 계산 O(N), 벡터화 스크리닝 엔진 (~100x), combine-scan 1152조합 3분대

---

## TODO: 실사용 워크플로우

```
데이터 다운로드 → 스캔 (scan / combine-scan) → 최적화 + Walk-Forward 검증
→ 페이퍼 트레이딩 (1~2주) → 소액 실매매 (10만원) → 단계적 증액
```

Sharpe > 1.5, Max Drawdown < 15% 인 전략을 찾은 후 페이퍼로 넘어가는 것을 권장.

---

## 추후 개발 (필요 시)

| 항목 | 트리거 |
|------|--------|
| 클라우드 배포 | 24/7 무중단 운영 필요 시 |
| 조합 생성기 | combine-scan 결과 부족 시 (`combine-generate` → 자동 조합 생성) |
| Bybit 추가 (6-4) | USDT 마켓 필요 시 (CCXT 추상화로 용이) |
| 선물/마진 (6-5) | 현물 경험 충분히 쌓은 후, 숏/레버리지 필요 시 |
| 레짐 감지 (6-8) | HMM 기반 시장 상태 분류, 기본 전략 한계 체감 시 |
| 성능 추가 최적화 | 대규모 파라미터 그리드 시 Numba/Cython 가속 검토 |

---

## 설계 한계 (버그 아님, 의도적)

- 인디케이터 함수가 DataFrame을 in-place 변경 (`.copy()` + assert로 이중 보호, 성능상 의도적)
- Walk-forward 검증은 단일 심볼 기준 윈도우 생성 (심볼별 데이터 기간 차이 시 복잡성 증가, 현재 설계가 합리적)
