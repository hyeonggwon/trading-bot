# Troubleshooting Log

## 2026-04-03: LightGBM 모델이 학습되지 않는 문제

### 증상

`ml-train-all --timeframe 1h` 실행 후 모든 심볼에서:
- Precision ≈ 0 (모델이 양성 예측을 거의 하지 않음)
- 확률 출력이 0.20~0.21에 몰림 (base rate와 동일)
- `best_iteration: 1` — 트리 1개만 생성하고 중단

```
BTC/KRW:  min=0.2018  max=0.2148  mean=0.2088
  >0.5: 0건  >0.4: 0건  >0.3: 0건
```

### 검증

simple 80/20 split으로 동일 데이터 학습 시:
- `best_iteration: 45`, AUC=0.66, proba range 0.10~0.73
- **모델은 학습 가능 → walk-forward 구조 문제**

### 근본 원인

1. **`is_unbalance=True` + 작은 val set → early stopping이 iteration 1에서 멈춤**
   - `is_unbalance`가 positive class를 ~4.2배 upweight
   - 첫 iteration의 base score(class prior ≈ 0.20)가 val loss 기준 "최선"으로 판정
   - iteration 2부터 val loss가 오히려 올라감 (작은 val set에서 불안정)
   - early stopping이 즉시 발동 → 트리 0개, 모든 예측 = base rate

2. **walk-forward 윈도우의 val set이 너무 작음**
   - 첫 윈도우: train ~2,160행, 80/20 split + 52 embargo → val ~380행
   - 380행에서 early stopping 신호가 노이즈 → iteration 1에서 멈춤

3. **최종 모델도 동일 문제**
   - 85/15 split + embargo로 early stopping → 같은 메커니즘으로 멈춤

4. **`learning_rate=0.02`가 너무 느림**
   - iteration 1에서 개선폭이 미미 → early stopping이 "개선 없음"으로 판단

5. **`min_data_in_leaf=200`이 너무 큼** (첫 시도)
   - 2,160행 train에서 리프당 200행 필요 → 트리가 분기를 거의 못 함

### 해결 (Round 1)

`trainer.py` DEFAULT_LGBM_PARAMS 변경:

| 파라미터 | Before | After | 이유 |
|----------|--------|-------|------|
| `metric` | `["binary_logloss", "auc"]` | `["auc", "binary_logloss"]` | AUC가 first metric → early stopping이 AUC 모니터링 (logloss보다 안정적) |
| `is_unbalance` | `True` | 삭제 | 작은 val set에서 loss curve 불안정 유발 |
| `scale_pos_weight` | — | `2.0` | `is_unbalance` 대체, 온건한 upweight |
| `min_data_in_leaf` | `200` → `50` | `20` | 작은 윈도우에서도 트리 분기 가능 |
| `min_sum_hessian_in_leaf` | `10.0` → `5.0` | `1.0` | 동일 이유 |
| `learning_rate` | `0.02` | `0.05` | 빠른 수렴, early stopping과 호환 |
| `n_estimators` | `2000` | `500` | lr=0.05에서 200-300 iteration이면 충분 |

`walk_forward.py` 변경:

| 변경 | 내용 |
|------|------|
| 윈도우 val set 최소 크기 | val < 1000행이면 early stopping 비활성화, `fixed_rounds=300` 사용 |
| 최종 모델 | early stopping 제거, 전체 데이터 + `fixed_rounds=300` |
| `trainer.train()` | `fixed_rounds` 파라미터 추가 (설정 시 early stopping 비활성화) |

### 결과 (Round 1)

```
BTC/KRW:  range 0.06~0.73, mean=0.31, std=0.11
  threshold=0.55: 19 predictions, precision=79%
  threshold=0.60: 9 predictions, precision=89%
```

### 추가 개선 (Round 2)

ML Engineer 에이전트 리뷰 후 추가 적용:

| 변경 | 파일 | 내용 |
|------|------|------|
| `obv_roc_10` 정규화 (버그 수정) | `features.py` | raw diff → rolling std로 나눔. OBV 스케일이 시간에 따라 달라지는 문제 |
| `hour`, `day_of_week` 피처 | `features.py` | 크립토 시간대별 패턴 (아시아/미국 세션 전환 등) |
| `body_to_range` 피처 | `features.py` | 캔들 body/range 비율 = 방향성 확신도 |
| 동적 `scale_pos_weight` | `walk_forward.py` | `min(2.0, max(1.0, 0.5 / pos_rate))` — 심볼별 positive rate 반영 |
| per-window best_iteration 로깅 | `walk_forward.py` | median을 final model의 `fixed_rounds`로 사용 (300 하드코딩 대체) |
| threshold coupling 문서화 | `trainer.py` | `scale_pos_weight` 변경 시 production threshold 재튜닝 필요 주석 |

피처 수: 33 → 36개

### Round 2 후퇴 — median fixed_rounds 로직 실패

Round 2에서 피처 추가 + 동적 scale_pos_weight + median fixed_rounds를 동시 적용 후 확률이 다시 좁아짐 (0.14~0.37).

**원인**: median best_iteration 로직.
- 대부분 윈도우가 val < 1000 → `fixed_rounds=300` 사용 → `best_iteration=-1`
- `> 0` 필터링 후 남는 건 val >= 1000인 소수 윈도우
- 그 윈도우에서 early stopping이 `best_iteration=1`로 멈춤 (is_unbalance→scale_pos_weight로 바꿨지만 여전히 작은 val에서 불안정)
- median(1, 1, 2, ...) → final_rounds=1~12 → 최종 모델 트리 12개 → 확률 범위 좁음

**해결**: median 로직 제거, 고정 `fixed_rounds=300` 복원. median은 모든 윈도우에서 early stopping이 안정적으로 작동할 때만 의미 있음.

### Round 2 최종 결과 (fixed 300 복원 + 피처 추가 + 동적 spw)

```
BTC/KRW:  range 0.05~0.75  @0.55: 23건 precision=61%  @0.60: 9건 precision=100%
ETH/KRW:  range 0.08~0.74  @0.55: 94건 precision=61%  @0.60: 38건 precision=71%
DOGE/KRW: range 0.05~0.86  @0.55: 184건 precision=75%  @0.60: 87건 precision=87%
XRP/KRW:  range 0.04~0.77  @0.55: 64건 precision=73%  @0.60: 32건 precision=88%
```

Round 1 대비 대부분 심볼에서 precision 개선 (피처 추가 + 동적 scale_pos_weight 효과).

### 교훈

1. **`is_unbalance=True`는 작은 val set에서 위험** — val loss를 불안정하게 만들어 early stopping을 즉시 트리거
2. **early stopping은 val set 크기에 의존** — 최소 1000행 이상 필요
3. **simple split으로 먼저 검증** — walk-forward 문제인지 모델 자체 문제인지 분리
4. **`best_iteration` 확인이 필수** — 1이면 모델이 학습하지 않은 것
5. **최종 배포 모델은 fixed rounds가 안전** — walk-forward로 일반화 검증 후, 전체 데이터로 고정 iteration 학습
