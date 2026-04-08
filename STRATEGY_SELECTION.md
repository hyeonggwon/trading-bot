# 전략 선정 보고서 (2026-04-09)

## 1. 스크리닝 개요

6년간 전체 데이터(2020-01 ~ 2026-04)를 대상으로 두 가지 스캔 수행:

| 스캔 | 조합 수 | 소요 시간 | 설명 |
|------|--------|----------|------|
| scan | 168 | ~10초 | 등록 전략 7종 × 8심볼 × 3TF |
| combine-scan | 1,152 | ~3분 30초 | 48 필터 템플릿 × 8심볼 × 3TF (verify-top 50) |

---

## 2. 후보 선정 기준

### 제외 기준

1. **ML(lgbm) 포함 전략 제외**: 1d ML AUC 0.50~0.56으로 인샘플 과적합 가능성 높음
2. **1d 타임프레임 제외**: 샘플 수 부족 + 과적합 경향
3. **거래 수 < 50 제외**: 통계적 신뢰도 불충분

### 선정 기준

- Sharpe ratio > 1.2 (인샘플)
- Max Drawdown < 10%
- 거래 수 50건 이상
- 4h 타임프레임 (거래 빈도와 안정성의 균형)

---

## 3. 스크리닝 결과 (ML 제외, TOP 10)

scan + combine-scan 합산, ML 제외:

| # | Source | Strategy | Symbol | TF | Sharpe | Return | MaxDD | Win% | PF | Trades |
|---|--------|----------|--------|----|--------|--------|-------|------|----|--------|
| 1 | combine | KC+Trend+Vol | ETH/KRW | 4h | 1.77 | 41.52% | 2.22% | 40.3% | 3.06 | 139 |
| 2 | both | Vol+Breakout | ETH/KRW | 4h | 1.70 | 40.31% | 4.30% | 41.7% | 3.41 | 120 |
| 3 | combine | BB+Vol | ETH/KRW | 4h | 1.63 | 41.10% | 4.09% | 36.2% | 3.00 | 130 |
| 4 | scan | bollinger | BTC/KRW | 4h | 1.63 | 33.23% | 2.57% | 36.9% | 2.05 | 241 |
| 5 | scan | bollinger | ETH/KRW | 4h | 1.59 | 44.06% | 4.90% | 30.4% | 1.90 | 270 |
| 6 | scan | sma_cross | ETH/KRW | 4h | 1.52 | 47.23% | 4.51% | 24.8% | 2.49 | 157 |
| 7 | combine | Breakout+Trend | BTC/KRW | 4h | 1.50 | 49.42% | 7.28% | 25.6% | 4.24 | 78 |
| 8 | combine | KC+Trend+Vol | BTC/KRW | 4h | 1.44 | 21.72% | 2.00% | 42.1% | 2.35 | 140 |
| 9 | scan | sma_cross | BTC/KRW | 4h | 1.39 | 35.66% | 4.78% | 28.9% | 2.47 | 142 |
| 10 | scan | volume_breakout | BTC/KRW | 4h | 1.33 | 19.14% | 2.21% | 41.1% | 2.47 | 112 |

### 선정: 4개 전략

ETH 편중 분산을 위해 BTC 1개 포함:

| # | Strategy | Symbol | TF | 선정 이유 |
|---|----------|--------|----|----------|
| 1 | KC+Trend+Vol | ETH/KRW | 4h | Sharpe 1위, MaxDD 최저(2.22%) |
| 2 | Vol+Breakout | ETH/KRW | 4h | Sharpe 2위, PF 최고(3.41) |
| 3 | bollinger | BTC/KRW | 4h | BTC 분산, 거래 수 241건(최다), MaxDD 2.57% |
| 4 | BB+Vol | ETH/KRW | 4h | Sharpe 3위, 130 trades |

---

## 4. Walk-Forward 검증

### 설정

- Train: 3개월, Test: 1개월 (롤링)
- 24개 윈도우 (2020-01 ~ 2026-04)
- 인디케이터 워밍업 버퍼 300 bars 적용

### 결과

| # | Strategy | Symbol | TF | IS Sharpe | OOS Sharpe | WF Efficiency | Overfit | Cum Return | Trades |
|---|----------|--------|----|----------|-----------|---------------|---------|------------|--------|
| 1 | **KC+Trend+Vol** | ETH/KRW | 4h | 1.34 | **1.23** | **91.9%** | **8.1%** | **19.37%** | 52 |
| 2 | **bollinger** | BTC/KRW | 4h | 2.29 | **1.14** | 49.8% | 50.2% | 10.17% | 61 |
| 3 | Vol+Breakout | ETH/KRW | 4h | 0.91 | 0.79 | 86.3% | 13.7% | 10.39% | 43 |
| 4 | BB+Vol | ETH/KRW | 4h | 0.98 | 0.57 | 58.7% | 41.3% | 10.23% | 45 |

### 메트릭 설명

| 메트릭 | 설명 | 좋은 값 |
|--------|------|--------|
| IS Sharpe | In-Sample (Train) 평균 Sharpe | 높을수록 좋음 |
| OOS Sharpe | Out-of-Sample (Test) 평균 Sharpe | > 0.5 유의미, > 1.0 우수 |
| WF Efficiency | OOS/IS Sharpe 비율 | > 50% 양호, > 80% 우수 |
| Overfit | 1 - WF Efficiency, 과적합 정도 | < 30% 양호 |
| Cum Return | OOS 구간 누적 수익률 | 양수 |

### 분석

1. **KC+Trend+Vol** (ETH/KRW 4h)
   - OOS Sharpe **1.23** — 실전 배포 기준(1.0) 초과
   - 과적합률 **8.1%** — 4개 중 최저, 매우 안정적
   - 누적 OOS 수익 **19.37%** — 최고
   - 고정 필터 전략(파라미터 최적화 없음) → 과적합 구조적으로 낮음

2. **bollinger** (BTC/KRW 4h)
   - OOS Sharpe **1.14** — 실전 배포 기준 초과
   - 과적합률 **50.2%** — 파라미터 최적화(period, std) 때문에 높음
   - IS Sharpe 2.29 vs OOS 1.14: 최적화 과적합 전형적 패턴
   - 그럼에도 OOS > 1.0이므로 실전 유효

3. **Vol+Breakout** (ETH/KRW 4h)
   - OOS Sharpe 0.79 — 양호하지만 1.0 미달
   - 과적합률 13.7%로 낮음 — 고정 필터의 장점
   - 보조 전략으로 적합

4. **BB+Vol** (ETH/KRW 4h)
   - OOS Sharpe 0.57 — 유의미하지만 단독 배포에는 약함
   - Vol+Breakout과 수익률 비슷 (10.23% vs 10.39%)

---

## 5. 결론: 배포 후보

| 우선순위 | Strategy | Symbol | TF | OOS Sharpe | 비고 |
|---------|----------|--------|----|-----------|------|
| **1순위** | KC+Trend+Vol | ETH/KRW | 4h | 1.23 | 최고 OOS, 최저 과적합 |
| **2순위** | bollinger | BTC/KRW | 4h | 1.14 | BTC 분산, OOS > 1.0 |
| 보조 | Vol+Breakout | ETH/KRW | 4h | 0.79 | 포트폴리오 분산용 |

### 전략 상세

**KC+Trend+Vol** (`keltner_break + trend_up:4 + volume_spike:2.0 → keltner_break`):
- Entry: Keltner Channel 상단 돌파 + 상위 TF 상승 트렌드 + 거래량 스파이크
- Exit: Keltner Channel 중간선 하향 이탈

**bollinger** (`bollinger_breakout`, period/std 윈도우별 최적화):
- Entry: 가격이 BB 상단 돌파
- Exit: 가격이 BB 중간선 하향 이탈

### 다음 단계

- [ ] 1~2순위 전략으로 Paper Trading 1주 실행
- [ ] Paper 결과 확인 후 Live 배포 검토
