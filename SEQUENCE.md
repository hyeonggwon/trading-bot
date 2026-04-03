# ML 학습 + Docker 테스트 순서

## 현재 상태

- main 브랜치, 158 tests passing, 코드 커밋 완료
- 데이터: 8 심볼 × 3 타임프레임 = **24 조합** (data/ 디렉토리)
- 모델: **없음** (models/ 비어있음)
- Docker: ML deps(`lightgbm` 등) 미포함 (`pip install .`만 설치, `[ml]` 아님)

## Step 1: Docker에 ML 의존성 추가

Dockerfile 수정: `pip install --no-cache-dir .` → `pip install --no-cache-dir ".[ml]"`

이유: `ml-train-all`, `lgbm_prob` 필터, `ml-backtest` 모두 lightgbm 필요.

## Step 2: docker-compose에 볼륨 + 학습 명령 추가

현재 docker-compose.yml은 paper trading만 지원. 변경 필요:

1. **data 바인드 마운트**: 로컬 data/ → 컨테이너 /app/data (학습에 필요)
2. **models 바인드 마운트**: 로컬 models/ → 컨테이너 /app/models (모델 영속화)
3. **named volume(`bot-data` 등) → 바인드 마운트로 변경**: 로컬 파일과 직접 공유

```yaml
volumes:
  - ./data:/app/data:ro          # 로컬 데이터 읽기 전용
  - ./models:/app/models         # 모델 저장/로드
  - bot-logs:/app/logs
  - bot-state:/app/state
  - ./config:/app/config:ro
```

`docker-compose run --rm bot <명령>` 으로 일회성 명령 실행 가능 (매번 `docker run -v ...` 안 쳐도 됨).

## Step 3: 로컬에서 먼저 ml-train-all 테스트

Docker 빌드 전에 로컬에서 동작 확인:

```bash
# 1h만 먼저 (8 조합, 빠름)
tradingbot ml-train-all --timeframe 1h

# 결과 확인
ls models/
```

예상 소요: 심볼당 ~1-2분 × 8 = **약 10-15분**

## Step 4: 로컬에서 combine + lgbm_prob 테스트

학습된 모델로 ML+Rule 조합 백테스트:

```bash
# 단일 심볼 테스트
tradingbot combine \
  --entry "trend_up:4 + rsi_oversold:30 + lgbm_prob:0.55" \
  --exit "rsi_overbought:70" \
  --symbol BTC/KRW

# combine-scan (ML 포함 템플릿 3개 포함)
# 36 템플릿 × 8 심볼 × 1 타임프레임 = 288 백테스트 → 약 10-15분
tradingbot combine-scan --top 15 --timeframe 1h
```

## Step 5: Docker 빌드 + 학습

바인드 마운트(`-v ./data:/app/data`)로 호스트 디스크를 컨테이너와 **직접 공유**함. 복사가 아니라 같은 파일을 읽고 쓰는 것.

docker-compose에 이미 볼륨 설정이 반영되어 있으므로 `docker-compose run`으로 실행:

```bash
# 빌드 (ML deps 포함)
docker-compose build

# ML 학습
docker-compose run --rm bot tradingbot ml-train-all --timeframe 1h

# 모델 생성 확인
ls models/
```

## Step 6: Docker에서 combine-scan 테스트

```bash
docker-compose run --rm bot tradingbot combine-scan --top 15 --timeframe 1h
```

## Step 7: (선택) 데이터 추가 다운로드 후 재학습

기존 데이터로는 모델이 고정됨. 최신 데이터를 추가 다운로드한 후 재학습하면 모델 품질 향상 가능.

```bash
# 최신 데이터 추가 다운로드 (기존 parquet에 자동 병합됨)
tradingbot download --symbol BTC/KRW --timeframe 1h --since 2024-01-01
# 또는 전체 심볼 일괄 (config/default.yaml 기준)
tradingbot download --timeframe 1h --since 2024-01-01

# 재학습 (기존 모델 덮어쓰기)
tradingbot ml-train-all --timeframe 1h
```

데이터가 많을수록 walk-forward 윈도우가 늘어나 모델 안정성이 올라감.

## 판단 포인트

| 시점 | 확인 사항 |
|------|-----------|
| Step 3 후 | 8개 모델 파일 생성됨? AUC > 0.5? |
| Step 4 후 | lgbm_prob 포함 전략이 trades > 0 인가? |
| Step 5 후 | Docker 내에서도 동일 결과? |
| Step 6 후 | ML+Rule 템플릿이 combine-scan 결과에 포함? |

## 소요 시간 예상

- Step 1-2: 5분 (Dockerfile + compose 수정)
- Step 3: 10-15분 (1h × 8 심볼 학습)
- Step 4: 5분 (백테스트)
- Step 5: 5분 빌드 + 10-15분 학습
- Step 6: 5분
- **총: ~40-50분**
