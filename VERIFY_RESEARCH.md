# --verify-top 구현 리서치

## 1. 개요

`combine-scan --verify-top N` 옵션: 벡터화 엔진으로 빠르게 스크리닝한 후, 상위 N개를 정밀한 candle-by-candle BacktestEngine으로 재검증.

**목적**: 벡터화 엔진은 스크리닝용 근사값. 최종 전략 선정 전에 정밀 수치 확인 필요.

---

## 2. 벡터화 엔진 vs 풀 엔진 차이점

### 2-1. 메트릭 차이

| 항목 | 벡터화 (`VectorizedResult`) | 풀 엔진 (`BacktestReport`) |
|------|---------------------------|---------------------------|
| **Sharpe** | ddof=0, 동일 annualization | ddof=0, 동일 |
| **Sortino** | ❌ 없음 | ✅ 있음 |
| **Max DD** | 동일 방식 (expanding max) | 동일 |
| **Win Rate** | 동일 | 동일 |
| **Profit Factor** | 동일 | 동일 |
| **Avg Duration** | ❌ 없음 | ✅ 있음 |
| **Trade Log** | tuple (idx, price, pnl) | Trade 객체 (timestamp, order 포함) |

→ 6가지 핵심 메트릭은 동일 수식. 차이는 Sortino, 평균 체류시간 등 부가 메트릭.

### 2-2. 로직 차이 (수치에 영향)

| 항목 | 벡터화 | 풀 엔진 |
|------|--------|---------|
| **포지션 사이징** | `cash × max_position_pct` 고정 | RiskManager: fixed-fractional + 서킷브레이커 |
| **복리 효과** | cash가 변동 (거래 후 잔고 반영) | 동일 (잔고 반영) |
| **슬리피지/수수료** | 동일 (fee_rate, slippage_pct) | 동일 |
| **스톱로스** | entry_price × (1 - sl_pct) 고정 | RiskManager에서 설정, 동일 수식 |
| **진입 체크** | 이전 바 시그널 → 다음 바 open 진입 | visible_df[:idx] → fill_candle.open 진입 |
| **같은 바 SL 방어** | `if i == entry_idx: continue` | 다음 캔들에서 fill 후 SL 체크 |
| **ATR 트레일링** | O(1) expanding max in loop | `entry_index` 기반 trailing |

**예상 수치 차이 원인**:
1. **포지션 사이징**: 벡터화는 `max_position_pct` 고정, 풀 엔진은 `fixed_fraction_pct` + 드로다운 서킷브레이커. 대부분 비슷하지만 연속 손실 시 서킷브레이커 차이 발생.
2. **Equity curve 정밀도**: 벡터화는 포지션 중 `closes[j]` 기반 mark-to-market, 풀 엔진은 매 캔들 정밀 추적. 근사값 수준 차이.
3. **안티 룩어헤드**: 벡터화는 `vectorized_entry(df)` → 전체 df 한 번에 계산. 대부분의 필터가 현재/이전 바만 참조하므로 안전하지만, 구조적으로 전체 df가 보임. 풀 엔진은 `visible_df[:idx]`로 구조적 차단.

→ **순위(ranking)**는 거의 동일할 것으로 예상. 절대 수치는 약간 차이 가능.

---

## 3. 현재 아키텍처 분석

### 3-1. combine-scan 플로우 (cli.py:985–1117)

```
1. list_available_data() → symbol/timeframe 목록
2. COMBINE_TEMPLATES × symbol × timeframe → batches (group by sym,tf)
3. ProcessPoolExecutor → _run_batch() per batch
4. _run_batch() 내부:
   - vectorizable_jobs: lgbm_prob 없는 combined → _run_vectorized_batch()
   - fallback_jobs: 등록 전략 + ML 포함 → _run_engine_batch()
5. 결과 수집 → Sharpe 내림차순 정렬 → 테이블 출력
```

### 3-2. 병렬 모듈 (parallel.py)

- `ScanResult` dataclass: strategy, symbol, timeframe, 6 metrics, entry, exit, error
- `_run_batch()`: 단일 (sym, tf) 배치 처리, vectorized/fallback 분기
- `_run_vectorized_batch()`: 필터 파싱 → union indicator → vectorized_backtest per job
- `_run_engine_batch()`: CombinedStrategy/등록전략 → BacktestEngine.run()

### 3-3. 재검증에 필요한 것

재검증 = 특정 (template, symbol, timeframe) 조합을 `_run_engine_batch()`로 재실행.

필요한 입력:
- `entry`, `exit` 문자열 (필터 파싱용)
- `symbol`, `timeframe` (데이터 로딩용)
- `balance`, `data_dir`, `config_dir`

이미 `_run_engine_batch()`가 이 로직을 전부 갖고 있음. 새 함수 불필요 — `_run_batch()`에서 모든 job을 fallback으로 강제하면 됨.

---

## 4. 구현 방식 비교

### Option A: CLI에서 재검증 (단순)

```python
# Phase 1: 기존 스캔 (변경 없음)
results = [...]  # 벡터화 + fallback 결과

# Phase 2: 상위 N개 추출 → _run_batch()로 재실행 (vectorized 경로 우회)
if verify_top > 0:
    to_verify = results[:verify_top]
    # Group by (sym, tf), _run_engine_batch() 호출
    # 결과 교체 → 재정렬
```

**장점**: 단순, parallel.py 변경 최소
**단점**: 데이터를 두 번 로드 (스캔 시 + 재검증 시)

### Option B: parallel.py에 verify 함수 추가

`_run_verify_batch()` — `_run_batch()`와 유사하되 모든 job을 `_run_engine_batch()`로 강제.

**장점**: 깔끔한 분리
**단점**: `_run_engine_batch()`와 거의 동일한 코드

### Option C: `_run_batch()`에 `force_engine=True` 파라미터 추가

```python
def _run_batch(symbol, timeframe, jobs, data_dir, balance, config_dir, force_engine=False):
    if force_engine:
        # 모든 job을 _run_engine_batch()로 전달
    else:
        # 기존 vectorized/fallback 분기
```

**장점**: 기존 함수 재활용, 데이터 로딩 로직 중복 없음
**단점**: 파라미터 하나 추가

### 결론: **Option C 추천**

- 코드 중복 최소
- `_run_batch()`의 데이터 로딩 + config 셋업 재사용
- CLI에서 `_run_batch(sym, tf, jobs, ..., force_engine=True)` 호출

---

## 5. CLI 통합 설계

### 5-1. 파라미터

```python
verify_top: int = typer.Option(0, "--verify-top", help="Re-verify top N with full engine")
```

- `--verify-top 0` (기본값): 재검증 안 함 (기존 동작)
- `--verify-top 10`: 상위 10개를 풀 엔진으로 재검증
- `--verify-top 50`: 상위 50개 재검증

### 5-2. 재검증 흐름

```
1. Phase 1: 기존 스캔 완료 → results 리스트
2. Sharpe 기준 내림차순 정렬
3. Phase 2: results[:verify_top] 추출
4. (sym, tf)별 그룹핑
5. ProcessPoolExecutor로 _run_batch(force_engine=True) 병렬 실행
6. 재검증 결과로 상위 N개 교체
7. 전체 results 재정렬
8. 테이블 출력 (verified 마크 표시)
```

### 5-3. 테이블 출력

재검증된 결과는 `✓` 마크로 구분:

```
# | Template       | Symbol   | TF  | Sharpe | Return  | MaxDD  | Win%  | PF   | Trades | V
1 | KC+Trend+Vol   | ETH/KRW  | 4h  | 1.78   | 41.50%  | 2.25%  | 52.5% | 1.85 | 135    | ✓
2 | BB+Vol         | ETH/KRW  | 4h  | 1.75   | 44.20%  | 4.10%  | 51.0% | 1.72 | 130    | ✓
3 | Vol+Breakout   | ETH/KRW  | 4h  | 1.70   | 40.31%  | 4.30%  | 50.0% | 1.65 | 120    |
```

---

## 6. 병렬화 전략

### 6-1. 재검증 작업 그룹핑

재검증 대상을 `(symbol, timeframe)`별로 그룹핑하면 데이터 로딩 1회로 여러 job 처리 가능.

예: `--verify-top 50`일 때
- 결과가 8 symbols × 3 timeframes에 분산 → 최대 24개 배치
- 각 배치에 1~5개 job → 병렬 처리

### 6-2. 예상 소요 시간

- 풀 엔진 1개 조합: ~5초 (6년 1h 데이터 기준)
- `--verify-top 10`: ~10~20초 (병렬 처리, 배치 수에 따라)
- `--verify-top 50`: ~30~60초 (병렬)
- 전체 스캔 (~3분) + verify-top 50 (~1분) = ~4분 총

### 6-3. Progress bar

재검증에도 Rich Progress 표시:
```
Phase 1: Scanning combinations  [████████████] 1152/1152  3:26
Phase 2: Verifying top 50       [████████    ] 35/50      0:42
```

---

## 7. 엣지 케이스

1. **verify_top > 실제 결과 수**: `min(verify_top, len(results))` 사용
2. **ML 템플릿 재검증**: ML 템플릿은 이미 풀 엔진 경유 → 재검증 불필요, 스킵
3. **등록 전략 재검증**: scan 결과에는 등록 전략 없음 (combine-scan 전용), 해당 없음
4. **재검증 후 순위 변동**: 벡터화 순위와 풀 엔진 순위가 다를 수 있음 → 전체 재정렬 필요
5. **재검증 실패**: error 발생 시 원래 벡터화 결과 유지

---

## 8. 테스트 계획

1. **CLI 테스트**: `--verify-top 0` (기본, 변경 없음), `--verify-top 5`
2. **_run_batch force_engine**: force_engine=True 시 모든 job이 engine batch로 가는지
3. **결과 교체**: 재검증 결과가 원래 결과를 올바르게 교체하는지
4. **ML 스킵**: ML 템플릿은 이미 풀 엔진이므로 재검증 스킵
5. **Progress bar**: Phase 1/Phase 2 표시 확인

---

## 9. 변경 파일 목록

| 파일 | 변경 내용 |
|------|----------|
| `src/tradingbot/cli.py` | `--verify-top` 파라미터 + Phase 2 재검증 로직 + 테이블에 V 컬럼 |
| `src/tradingbot/backtest/parallel.py` | `_run_batch()`에 `force_engine` 파라미터 추가 |
| `tests/test_vectorized.py` | `force_engine=True` 테스트 추가 |
