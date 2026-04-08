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

## 남은 작업

- [ ] **combine-scan으로 베스트 조합 찾기**
  - `tradingbot combine-scan --top 20`으로 전체 스캔 (실행 중)
  - ML 포함/미포함 성능 비교
  - 베스트 2~3개 조합 선정

- [ ] **Docker 컨테이너 재배포**: 베스트 조합으로 전환

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
