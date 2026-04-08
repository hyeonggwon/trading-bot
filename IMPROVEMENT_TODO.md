# ML Model Improvement TODO

## 배경

LightGBM 모델로 paper trading 5일간 거래 0건 발생.
원인: entry_threshold(0.60)이 모델 최대 출력(~0.54)보다 높아 시그널이 한 번도 발생하지 않음.

## 전략 결정

ML Engineer + Quant Analyst 분석 결과, **ML 단독 전략 대신 veto 필터로 전환**하기로 결정.

- 가격/거래량 feature만으로는 OOS AUC 0.58이 현실적 한계 → 단독 시그널로 부적합
- AUC 0.55 모델도 **하위 30% 나쁜 거래 필터링**에는 유효
- 룰 기반 시그널이 거래 빈도 담당, ML이 품질 필터 담당
- 레짐 변화 시에도 ML AUC가 0.50으로 떨어져도 룰 기반이 계속 작동 → 안정적
- 예상 Sharpe: 단독 ML 0.3~0.8 vs veto 필터 0.8~1.5

## PR 1: ML 학습 파이프라인 개선 ✅ (merged #13)

- [x] **scale_pos_weight 수정**: 2.0 → 1.0 (확률 왜곡 제거)
- [x] **fixed_rounds 제거**: 300 고정 → median best_iteration 사용
- [x] **Feature 축소**: 36개 → 15개 (gain importance 기반, 다중공선성 제거)
- [x] **Half-Kelly 주석 추가**: 백테스트 근거 (1h=1.52, 4h=2.07) → 1.5 유지
- [x] **LOG_LEVEL 환경변수 지원**: `setup_logging()`에서 자동 반영
- [x] **state/ git tracking 제거**: `.gitignore` 추가
- [x] 전 심볼 재학습 + walk-forward AUC 확인
  - BTC/KRW 1h: **0.6092** (목표 초과)
  - XRP/KRW 1h: 0.5877, DOGE/KRW 1h: 0.5734, ETH/KRW 1h: 0.5683
  - 4h/1d는 0.50~0.55 수준 (veto 필터로 충분)

## PR 2: Veto 필터 전환 + 룰 기반 조합 탐색 ✅ (merged #14)

- [x] **paper/live/backtest에서 COMBINE_TEMPLATES 이름 지원**: `--strategy ML+TrendEMA`
- [x] **paper/live에 --entry/--exit 옵션 추가**: 커스텀 조합 즉석 실행
- [x] **ML veto 템플릿 12개 추가** (Trend 3, MeanRev 3, Breakout 3, Volume 1, Multi 2, 총 48개)
- [x] **기존 ML 템플릿 threshold 0.55 → 0.35** (veto 모드)
- [x] **멀티 심볼 + ML 조합 방어**: ML 모델이 per-symbol이므로 멀티 심볼 사용 차단

### 타임프레임별 ML veto 적용 기준

| TF | 평균 AUC | ML veto 권장 | threshold |
|----|---------|-------------|-----------|
| **1h** | 0.53~0.61 | O | 0.35 |
| **4h** | 0.50~0.54 | △ (효과 미미) | 0.40 |
| **1d** | 0.50~0.56 | X (샘플 부족) | 사용 안 함 |

- AUC >= 0.55인 1h 심볼만 ML veto 적용: BTC, XRP, DOGE, ETH, ADA, SOL
- AVAX(0.54), LINK(0.53)은 ML veto 효과 낮음
- 4h/1d는 룰 기반만 사용 권장

## PR 3: 전략 검증 + 배포

### 스캔 결과 요약 (2026-04-08)

1d는 인샘플 과적합으로 제외. 1h/4h 기준:

| 전략 | 심볼 | TF | Sharpe | Return | Trades | 비고 |
|------|------|----|--------|--------|--------|------|
| Breakout+ATR | BTC/KRW | 1h | 2.38 | 8.49% | 43 | 룰 기반, 1순위 후보 |
| BB+Vol | BTC/KRW | 4h | 2.44 | 5.62% | 20 | 낮은 MaxDD(1.29%) |
| Trend+RSI | BTC/KRW | 4h | 2.26 | 6.44% | 77 | 거래 수 충분 |
| sma_cross | BTC/KRW | 4h | 2.58 | 7.94% | 21 | 단독 전략 중 최고 |

ML veto(threshold 0.35) 조합은 거래 多 + 수익 거의 0 → threshold 조정 필요

### Task A: 룰 기반 walk-forward 검증

- [ ] Breakout+ATR BTC/KRW 1h walk-forward (3개월 train / 1개월 test)
- [ ] BB+Vol BTC/KRW 4h walk-forward
- [ ] Trend+RSI BTC/KRW 4h walk-forward
- [ ] OOS Sharpe 1.2 이상인 전략 선정

### Task B: ML threshold 조정 + 재스캔

- [ ] ML veto threshold 0.35 → 0.55로 변경하여 combine-scan 재실행 (1h만)
- [ ] threshold 0.60으로도 테스트
- [ ] 룰 기반 대비 성능 비교
- [ ] 4h/1d ML 템플릿 비활성화 또는 제거 검토

### Task C: 배포

- [ ] Task A/B 결과에서 베스트 2~3개 전략 선정
- [ ] Docker 컨테이너 재배포
- [ ] Paper trading 1주 모니터링

### Task D: 벡터화 스크리닝 엔진 (scan/combine-scan 고속화)

- [ ] **scan/combine-scan 전용 간소화 엔진 추가**
  - 시그널만 벡터 계산 (pandas boolean mask), 거래 시뮬레이션은 간단히
  - 기존 엔진은 그대로 유지 (정밀 백테스트, live/paper용)
  - 상위 N개 결과만 기존 엔진으로 재검증
  - 예상: combine-scan 12~15분 → **1~2분**
- [ ] 벡터화 가능 필터: 27/31개 (AtrTrailingExit, LgbmProb 제외)
- [ ] 벡터화 불가 필터는 기존 엔진 fallback

### 보류

- [ ] **ML 포지션 사이즈 부스터 전환** (선택)
  - ML을 hard gate 대신 position size modifier로 전환
  - 룰 기반 base size + ML 확신도에 따라 1.0x~1.5x 조절
  - CombinedStrategy 수정 필요

- [ ] **Percentile 기반 동적 threshold 구현** (선택)
  - LGBMStrategy에 rolling probability buffer (최근 500개) 추가
  - 고정 threshold 대신 `percentile(buffer, 85~90)` 사용
  - 시장 레짐 변화에 자동 적응

## 보류 (현재 불필요)

- ~~Triple-Barrier 타겟~~: veto 필터 용도에서는 복잡성 대비 효과 낮음. 단독 전략 재시도 시 재검토
- ~~Prediction horizon 확대 (4→12~24)~~: 의견 분분. 샘플 감소 우려. veto 모드에서는 현행 유지
- ~~Cross-asset feature (BTC dominance)~~: 엔지니어링 오버헤드 대비 OOS AUC 개선 미미
- ~~Exit 로직 정합성~~: veto 필터 전환 시 exit은 룰 기반이 담당하므로 불필요

## 참고: AUC 해석

| AUC | 금융 데이터에서의 의미 |
|-----|----------------------|
| 0.50 | 랜덤 (쓸모없음) |
| 0.52~0.57 | 약한 시그널, **veto 필터로 유용** |
| 0.58~0.62 | 현 feature set의 현실적 한계 |
| 0.65+ | 가격/거래량만으로는 비현실적 (대체 데이터 필요) |
| 0.85+ | 데이터 누수/과적합 의심 |
