# src/tradingbot/

## What — 무엇을 하는가

`tradingbot` 파이썬 패키지의 루트. Upbit KRW 스팟 전용 백테스트·ML·라이브 엔진 + Typer 기반 CLI(`cli/` 패키지, 24 commands). 서브패키지가 책임을 나눠 갖고, `cli/` 가 모든 진입점을 모은다.

## How — 일반적인 수정

- **새 CLI 명령**: 도메인별 `cli/<domain>.py`(data/backtest/combine/trade/ml) 에 `@app.command()` 추가 — `app`/`console` 은 `cli/_shared.py` 에서 import. 새 도메인 모듈이면 `cli/__init__.py` 에 import 등록(등록 순서 = import 순서). 무거운 import는 lazy 로. README/루트 CLAUDE.md 의 CLI Reference 갱신. **GUI 파리티**: `dashboard/forms.py` 의 `PAGE_COMMANDS` 에 페이지 슬롯 배정 필수 — `tests/test_dashboard_forms.py` 래칫이 미배정 명령을 CI 에서 차단 (폼 자체는 click introspection 으로 자동 생성).
- **새 전략**: `strategy/examples/<name>.py` 에 `Strategy` 상속 클래스 → `strategy/registry.py` 의 `get_strategy_map()` 에 등록. 테스트는 `tests/test_strategies.py`.
- **새 필터**: `strategy/filters/<role>.py` (trend/momentum/price/volatility/session/volume/exit/ml) 에 `BaseFilter` 상속 → `strategy/filters/registry.py` 에 추가. `combine`/`combine-scan` 에서 자동 사용. lookback 긴 필터는 `min_history` 선언 — 라이브 워밍업 창(200캔들) 초과 시 `live/engine.py` 가 `filter_history_truncated` 경고 ([`anti-patterns.md`](./anti-patterns.md) 참조).
- **백테스트 엔진/사이저 변경**: `backtest/engine.py` (anti-lookahead 핵심) 또는 `backtest/simulator.py`. 변경 시 `tests/test_backtest_engine.py`·`tests/test_multi_symbol.py` 동시 갱신.
- **ML 학습/튜닝 흐름**: `ml/trainer.py`(단일 fit) → `ml/walk_forward.py`(holdout 분할) → `ml/strategy_walk_forward.py`(time-honest WF) → `ml/tuner.py`(Optuna) → `ml/threshold_tuner.py`(threshold sweep). 메타 키(`holdout_start`, `avg_win_loss_ratio`, `best_params`) 는 `LGBMStrategy._load_model` 이 읽으므로 깨지면 inference 가 정렬을 잃는다.
- **모든 수정 공통 — CI 차단 게이트**: ruff check/format, mypy strict 그린(`.mypy-baseline` 래칫이 0 도달 — 신규 오류 1건도 차단, pandas 는 `pandas-stubs` 실스텁 검사), pytest 커버리지 ≥ 60% (`.github/workflows/ci.yml`).

## How not — 빌드를 깨뜨리는 비명백한 패턴

→ 누적 기록: [`anti-patterns.md`](./anti-patterns.md). 새 패턴 발견 시 반드시 append.

## Where — 의존성

- **Incoming**: `tradingbot` console_script(`pyproject.toml`) → `cli:app`(`cli/_shared.py` 정의, `cli/__init__.py` re-export). 외부 사용자는 CLI만 쓴다.
- **Outgoing**: `ccxt`(Upbit), `lightgbm`/`scikit-learn`/`optuna`(ML), `pandas`/`pyarrow`(Parquet), `typer`/`rich`(CLI), `streamlit`(`dashboard/`, optional), `httpx`/`websockets`(`exchange/ws_client.py`).
- **모듈 간**: `strategy` → `core/models`·`data/indicators`. `backtest/engine` → `strategy`·`risk/manager`·`backtest/simulator`. `live/engine` → `exchange/*`·`risk`·`notifications/telegram`. `ml/*` → `data/*`·`strategy/lgbm_strategy`. 직접 cross-call 보다 `core/models` 의 dataclass 경유.
- **아키텍처 도식**: 서브패키지 의존 그래프 + layer 경계 규칙은 [`doc/architecture.md`](./doc/architecture.md) (자동 생성 mermaid, pre-push 훅이 drift 시 재생성).

## Why — 코드에 안 적힌 부족 지식

- **anti-lookahead 가 convention 이 아니라 구조적**: 엔진이 `visible_df = indicator_df[0..idx-1]` 슬라이스만 전략에 넘긴다. indicators pre-compute 단계에서 `shift(-1)` / `center=True` rolling 같은 미래 누수가 들어가면 잡히지 않으니 새 지표 추가 시 합성 데이터로 검증.
- **`supports_precompute` flag**: 일부 전략(multi_tf 등)은 per-iteration fallback 경로. flag 안 맞추면 1년 데이터에 30분+ 걸린다.
- **벡터화 엔진(`backtest/vectorized.py`) 은 screening 전용**: `combine-scan` 의 ~100x 속도용. 라이브/페이퍼에 절대 쓰지 않는다 — fill 시뮬·risk 검증을 안 함.
- **CLI 명령 등록은 import 부수효과**: `cli/__init__.py` 가 도메인 모듈을 import 해야 `@app.command()` 가 등록된다 — import 한 줄을 지우면 해당 명령들이 조용히 사라진다. 무거운 import(ML/ccxt/streamlit)는 명령 함수 안에서 (cold-start 컨벤션, 패키지 분할 후에도 유지).
- **ML 사이징 spine**: 확률 → `ml/utils.py` 의 `kelly_strength()`(Half-Kelly 를 구조적 상한 0.5 로 나눠 정규화) → `Signal.strength` ∈ [0,1] → 사이저 base size 배율. raw `half_kelly()` 는 strength 로 직접 쓰지 않는다 — 0.5 상한 탓에 ML 진입이 계통적으로 언더사이징된다.
- **라이브 안전레일은 이체 면역 원장 equity**: 드로다운 브레이커·피크 추적은 `live/engine.py` 의 `_ledger_equity`(baseline+누적 실현+미실현, `live/state.py` 에 영속) 기준 — 외부 입출금이 phantom drawdown 을 만들거나 실제 drawdown 을 가리지 못한다. 사이징 예산은 raw equity 이고, equity 계상은 관리 유니버스(전략 심볼+보유 포지션)만 포함한다.
