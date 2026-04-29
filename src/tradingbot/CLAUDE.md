# src/tradingbot/

## What — 무엇을 하는가

`tradingbot` 파이썬 패키지의 루트. Upbit KRW 스팟 전용 백테스트·ML·라이브 엔진 + Typer 기반 CLI(`cli.py` 29 commands). 서브패키지가 책임을 나눠 갖고, `cli.py` 가 모든 진입점을 모은다.

## How — 일반적인 수정

- **새 CLI 명령**: `cli.py` 에 `@app.command()` 함수 추가 → 필요한 서브패키지 import만 lazy 로 (cold-start 무겁지 않게). README/루트 CLAUDE.md 의 CLI Reference 갱신.
- **새 전략**: `strategy/examples/<name>.py` 에 `Strategy` 상속 클래스 → `strategy/registry.py` 의 `get_strategy_map()` 에 등록. 테스트는 `tests/test_strategies.py`.
- **새 필터**: `strategy/filters/<role>.py` (trend/momentum/price/volatility/volume/exit/ml) 에 `BaseFilter` 상속 → `strategy/filters/registry.py` 에 추가. `combine`/`combine-scan` 에서 자동 사용.
- **백테스트 엔진/사이저 변경**: `backtest/engine.py` (anti-lookahead 핵심) 또는 `backtest/simulator.py`. 변경 시 `tests/test_backtest_engine.py`·`tests/test_multi_symbol.py` 동시 갱신.
- **ML 학습/튜닝 흐름**: `ml/trainer.py`(단일 fit) → `ml/walk_forward.py`(holdout 분할) → `ml/strategy_walk_forward.py`(time-honest WF) → `ml/tuner.py`(Optuna) → `ml/threshold_tuner.py`(threshold sweep). 메타 키(`holdout_start`, `avg_win_loss_ratio`, `best_params`) 는 `LGBMStrategy._load_model` 이 읽으므로 깨지면 inference 가 정렬을 잃는다.

## How not — 빌드를 깨뜨리는 비명백한 패턴

→ 누적 기록: [`anti-patterns.md`](./anti-patterns.md). 새 패턴 발견 시 반드시 append.

## Where — 의존성

- **Incoming**: `tradingbot` console_script(`pyproject.toml`) → `cli.py:app`. 외부 사용자는 CLI만 쓴다.
- **Outgoing**: `ccxt`(Upbit), `lightgbm`/`scikit-learn`/`optuna`(ML), `pandas`/`pyarrow`(Parquet), `typer`/`rich`(CLI), `streamlit`(`dashboard/`, optional), `httpx`/`websockets`(`exchange/ws_client.py`).
- **모듈 간**: `strategy` → `core/models`·`data/indicators`. `backtest/engine` → `strategy`·`risk/manager`·`backtest/simulator`. `live/engine` → `exchange/*`·`risk`·`notifications/telegram`. `ml/*` → `data/*`·`strategy/lgbm_strategy`. 직접 cross-call 보다 `core/models` 의 dataclass 경유.

## Why — 코드에 안 적힌 부족 지식

- **anti-lookahead 가 convention 이 아니라 구조적**: 엔진이 `visible_df = indicator_df[0..idx-1]` 슬라이스만 전략에 넘긴다. indicators pre-compute 단계에서 `shift(-1)` / `center=True` rolling 같은 미래 누수가 들어가면 잡히지 않으니 새 지표 추가 시 합성 데이터로 검증.
- **`supports_precompute` flag**: 일부 전략(multi_tf 등)은 per-iteration fallback 경로. flag 안 맞추면 1년 데이터에 30분+ 걸린다.
- **벡터화 엔진(`backtest/vectorized.py`) 은 screening 전용**: `combine-scan` 의 ~100x 속도용. 라이브/페이퍼에 절대 쓰지 않는다 — fill 시뮬·risk 검증을 안 함.
- **CLI lazy import 컨벤션**: `cli.py` 가 4159 줄이라 모듈 import 가 전부 top-level 이면 모든 명령이 무거워진다. 무거운 import(ML/ccxt/streamlit)는 함수 안에서.
