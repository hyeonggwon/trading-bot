# Architecture · src/tradingbot

> 자동 생성된 그래프는 별도 `.mmd` 파일에. 이 문서는 *왜 이 구조인지* 와 *경계 규칙* 만 보충한다.

## Overview

Upbit KRW 스팟 전용 트레이딩 봇 패키지. `core` 의 dataclass 를 공용 언어로 두고, 데이터 → 전략 → 백테스트/ML → 리스크 → 거래소/라이브 순으로 층을 쌓는다. 모든 진입점은 `cli.py` 한 곳에 모이고, 무거운 의존성(ML/ccxt/streamlit)은 명령 함수 안에서 lazy import 한다. layer 가 단방향으로만 의존하도록 유지해 백테스트에서 검증한 코드 경로가 그대로 라이브로 이어진다.

## Diagram

→ [`doc/mermaid/tradingbot-internal.mmd`](./mermaid/tradingbot-internal.mmd) (자동 생성, drift 시 풀 리젠)

## Layers

| Layer | 책임 (한 줄) | 의존 방향 (import 가능) |
|---|---|---|
| `core` | Candle/Signal/Order/Trade/Position/PortfolioState dataclass + enums + events | (leaf — 다른 tradingbot 패키지 import 안 함) |
| `config` | Pydantic 설정(YAML + .env) | `core` |
| `data` | OHLCV/외부데이터 fetch, Parquet I/O, 19개 지표 래퍼 | `config` |
| `strategy` | `Strategy` ABC + 31 filter + Combined/LGBM 전략 + registry | `core`, `data`, `ml` |
| `ml` | LightGBM 학습·walk-forward·Optuna·threshold·calibrator | `core`, `data`, `strategy.lgbm_strategy` |
| `risk` | 포지션 사이징, 드로다운 서킷브레이커, 사전 검증 | `core`, `config` |
| `backtest` | anti-lookahead 엔진 + 벡터화 screening + 최적화/walk-forward | `strategy`, `risk`, `core` |
| `exchange` | Upbit CCXT/paper/WebSocket 어댑터 | `core`, `config` |
| `live` | 비동기 라이브/페이퍼 루프, 상태 영속화, 주문 수명주기 | `exchange`, `risk`, `strategy`, `notifications`, `core` |
| `notifications` | Telegram 알림 | `config` |
| `dashboard` | Streamlit 대시보드 (optional) | `live`/`backtest` 산출물 |
| `cli` | Typer 진입점 — 위 layer 전부를 lazy import 로 묶음 | (모든 layer) |
| `utils` | 콘솔+JSON 로깅 단일 진입점 | (leaf) |

> 표는 자동 추출된 노드 + 사용자 보충 책임 설명. import 방향은 `extract.py` edge 와 일치해야 함.

## Cross-cutting concerns

코드만 보면 안 보이는 *의미* 를 적는다 (자동 추출 X — 사용자 보충).

- **anti-lookahead 는 구조적 불변식**: `backtest/engine.py` 가 전략에 `visible_df = indicator_df[0..idx-1]` 슬라이스만 넘긴다. convention 이 아니라 엔진이 강제 — 새 지표/전략의 미래 누수(`shift(-1)`, `center=True` rolling)는 합성 데이터 회귀로 검증.
- **ML 메타 계약**: `ml/*` 이 쓰는 meta.json 키(`holdout_start`, `avg_win_loss_ratio`, `best_params`, per-symbol threshold)를 `strategy/lgbm_strategy.py:_load_model` 이 읽는다. 키가 깨지면 inference 가 조용히 정렬을 잃는다.
- **CLI lazy import**: `cli.py` 가 4천 줄이라 무거운 import 를 top-level 에 두면 모든 명령이 느려진다 — ML/ccxt/streamlit 은 함수 안에서만.
- **로깅 단일 진입점**: `utils/logging.py` (콘솔 + JSON 파일, 일자 회전). 다른 layer 가 직접 핸들러를 구성하지 않는다.

## Boundaries (import 금지 규칙)

명시적 금지 규칙. 위반 시 review reject.

- `core/` 는 다른 tradingbot 패키지 import 금지 — 도메인 모델은 의존성 zero 여야 위/아래 layer 가 공용으로 쓴다.
- `backtest/vectorized.py` (screening 전용, ~100x) 는 라이브/페이퍼 경로에서 import 금지 — fill 시뮬·risk 검증을 하지 않는다. 실거래는 `backtest/engine.py` 또는 `live/engine.py` 만.
- 의존은 위 표의 단방향만 — 하위 layer(`core`/`config`/`data`)가 상위(`live`/`cli`)를 import 하면 순환. `extract.py` edge 로 감시.

## Generated

- mermaid: `python3 ~/.claude/skills/architecture-mapper/scripts/extract.py src --module tradingbot --output src/tradingbot/doc/mermaid/tradingbot-internal.mmd`
- drift 검증: `python3 ~/.claude/skills/architecture-mapper/scripts/drift.py src/tradingbot/doc/architecture.md src/tradingbot --quiet`
- pre-push 훅이 drift 감지 시 위 mermaid 를 자동 재생성한다 (`scripts/git-hooks/pre-push` 1단계).
