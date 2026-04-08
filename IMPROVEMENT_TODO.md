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

## PR 1: ML 학습 파이프라인 개선 + 버그 수정

- [ ] **scale_pos_weight 수정**: 2.0 → 1.0 (확률 왜곡 제거, veto 필터 보정에 필수)
  - `src/tradingbot/ml/trainer.py` line 33

- [ ] **fixed_rounds 제거**: 300 고정 → walk-forward에서 early stopping된 윈도우들의 median best_iteration 사용
  - `src/tradingbot/ml/walk_forward.py` line 131-138

- [ ] **Feature 축소**: 36개 → 상위 12~15개 (다중공선성 제거)
  - RSI 파생 3개(rsi_14, rsi_dist_from_50, rsi_roc_3) → 1개
  - Stochastic 파생 3개(stoch_k, stoch_d, stoch_kd_diff) → 1개
  - 핵심 feature: volume_ratio, atr_pct, close_roc_1/3/5, bb_pos
  - `src/tradingbot/ml/features.py`

- [ ] **Half-Kelly 수정**: `avg_win_loss_ratio` 1.5 하드코딩 → 백테스트 실제 승/패 비율로 산출
  - `src/tradingbot/strategy/lgbm_strategy.py` line 24

- [ ] **로그 레벨 반영**: `setup_logging()`이 `LOG_LEVEL` 환경변수를 읽도록 수정
  - `src/tradingbot/cli.py`의 `setup_logging()` 호출부

- [ ] 전 심볼 재학습 (`tradingbot ml-train-all`) + walk-forward AUC 확인 (0.56~0.58 목표)

## PR 2: Veto 필터 전환 + 룰 기반 조합 탐색

- [ ] **좋은 룰 기반 조합 찾기** (ML보다 우선)
  - `tradingbot combine-scan --top 15`로 walk-forward 생존 조합 탐색
  - 베스트 2~3개 조합 선정

- [ ] **combine 프레임워크에서 ML을 veto 필터로 사용**
  - `lgbm_prob` threshold: 0.30~0.35 (하위 30% 거래만 거부)
  - 베스트 조합에 ML veto 추가하여 성능 비교
  - 예시: `tradingbot combine --entry "trend_up:4 + rsi_oversold:30 + lgbm_prob:0.35" --exit "rsi_overbought:70"`

- [ ] **Docker 컨테이너 재배포**: 기존 lgbm 단독 → combine+veto로 전환

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
