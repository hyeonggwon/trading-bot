# Anti-Patterns · src/tradingbot/

빌드를 깨뜨리거나 미묘한 회귀를 만드는 **비명백한** 패턴을 누적 기록한다.
새 패턴 발견 시 **append only** — 기존 entry는 사실이 틀렸을 때만 수정.

CLAUDE.md 의 "How not" 섹션은 이 파일을 1줄로만 가리킨다.

---

<!-- 새 entry 포맷:

## <짧은 패턴 이름> · YYYY-MM-DD

**증상:** 어떤 식으로 깨지는지 (에러 메시지·테스트 실패·런타임 이상)
**원인:** 왜 깨지는지 — 비명백한 이유. 코드만 보면 안 보이는 부분.
**처방:** 어떻게 피하는지. 가능하면 진입점 파일/함수 명시.
**참고:** 관련 PR / 이슈 / 커밋 (있으면).

-->

## raw `half_kelly` 를 Signal.strength 로 직접 사용 · 2026-07-02

**증상:** ML 전략/필터의 포지션이 계통적으로 언더사이징 — 현실적 calibrated 확률대에서 strength 가 0.04–0.20 에 머물고, 사이저의 [0,1] 클램프는 dead code 가 됨.
**원인:** `half_kelly()` 는 p=1.0 에서도 0.5 가 구조적 상한 (full Kelly 1.0 의 절반). [0,1] 스케일을 기대하는 사이저에 절반 스케일 값이 들어감.
**처방:** strength 매핑은 항상 `ml/utils.py` 의 `kelly_strength()` (HALF_KELLY_MAX=0.5 로 정규화) 사용. 호출처: `strategy/lgbm_strategy.py`, `strategy/filters/ml.py`.
**참고:** 83a3785 (사이징 spine 정합).

## 드로다운 브레이커를 raw 계좌 equity 에 연결 · 2026-07-02

**증상:** 외부 출금 직후 브레이커 오발동(전 포지션 강제 청산), 외부 입금은 실제 drawdown 을 마스킹.
**원인:** raw 계좌 잔고는 트레이딩 성과가 아닌 입출금에도 움직인다. 피크 갱신(`update_peak_equity`)을 틱 루프에서 raw equity 로 호출하면 브레이커가 읽는 ledger 시리즈와 갈라진다.
**처방:** 피크 추적·브레이커 판정은 `live/engine.py` 의 `_enforce_safety_rails` 안에서 `_ledger_equity`(baseline+누적 실현 PnL+미실현) 로만. `ledger_baseline`/`cum_realized_pnl` 은 `live/state.py` 가 영속.
**참고:** e7d57a8 (드로다운 브레이커 이체 면역).

## Upbit 마켓 매수를 reference price 없이 제출 · 2026-07-02

**증상:** `CcxtExchange.create_order` 가 ValueError("requires a positive reference price") — 진입/재주문 스킵.
**원인:** Upbit 마켓 BUY 는 quote 금액 주문(ord_type='price') — base 수량만으로는 KRW cost 를 계산할 수 없고, ccxt 는 전송 전에 InvalidOrder 를 던진다.
**처방:** 마켓 BUY 에도 slippage 반영 기준가를 `price` 로 전달해 quote cost 를 계산시킨다. limit→market 재주문(`live/order_manager.py`)도 limit price 를 그대로 넘긴다.
**참고:** ea237bb (라이브 마켓 매수 quote-cost).

## config.yaml 에 모델에 없는 키 추가 · 2026-07-02

**증상:** 부팅 시 pydantic ValidationError(extra_forbidden) 로 설정 로드 실패.
**원인:** 설정 모델이 `extra="forbid"` (`config.py` `_StrictModel`) — 오타 키(`max_drawdown_pcnt` 등)가 조용히 기본값으로 대체되는 실계좌 사고를 막는 의도적 전환. pydantic 기본(ignore)과 다르다.
**처방:** 새 설정 키는 `config.py` 의 해당 모델에 필드를 먼저 추가한 뒤 YAML 에 쓴다.
**참고:** 5571ae0 (설정 오타 거부).

## 레짐 필터 커스텀 파라미터가 라이브 200캔들 창을 초과 · 2026-07-06

**증상:** `realized_vol_low/high` 의 커스텀 `vol_period`/`rank_period` 가 크면(합+1 > 200) 라이브/페이퍼 신호가 백테스트와 조용히 어긋난다(에러·NaN 없음 — `min_periods=10` 때문에 값은 항상 나온다).
**원인:** 라이브 엔진은 캔들을 200개만 fetch 하는데, 백분위 랭크 창에 partial-window vol 값이 섞이면 풀 히스토리 계산과 순위가 달라진다. 풀 윈도우 패리티 경계는 `vol_period + rank_period + 1 ≤ 200` (기본 20/50 → 71, 안전).
**처방:** 필터는 `min_history`(BaseFilter 기본 0)를 선언하고 `CombinedStrategy.min_history` 가 최댓값을 집계 — 라이브 엔진이 워밍업에서 200 초과 시 `filter_history_truncated` 경고를 낸다. 새 장주기 필터를 만들면 `min_history` 를 같이 선언할 것. 회귀: `tests/test_combine.py::test_min_history_parity_bound`.

## standalone click 으로 Typer 명령 introspection · 2026-07-07

**증상:** `isinstance(typer.main.get_command(app), click.Group)` 이 False — 대시보드 auto-form 계층이 AssertionError 로 즉사. `isinstance(param, click.Option)` 도 전부 False 라 옵션이 조용히 0개가 될 수 있음.
**원인:** typer ≥0.16 은 click 을 `typer._click` 으로 벤더링 — 반환 객체가 PyPI click 클래스의 인스턴스가 아님. venv 에 click 이 별도로 깔려 있으면 import 는 성공해서 더 비명백해짐.
**처방:** duck-typing 으로만 introspection: `param.param_type_name == "option"`, `param.type.name`("integer"/"float"/"boolean"/"text"), `is_flag`/`secondary_opts` 속성 사용. 진입점: `dashboard/forms.py` `get_cli_commands()`/`command_param_specs()`.
**참고:** GUI 파리티 작업 (2026-07-07).

## ML 워크포워드 수익률을 fraction으로 가정 · 2026-07-08

**증상:** 파이프라인 통합 랭킹에서 ML 후보의 누적 수익률이 룰 후보의 100배로 계산돼 lgbm이 항상 압승 — 조용한 랭킹 오염 (예외 없음).
**원인:** `MLStrategyWalkForwardReport`는 `return_pct`/`cumulative_return_pct`/`max_dd_pct`가 **% 단위**(`total_return * 100`, `ml/strategy_walk_forward.py`)인데 `WalkForwardReport`는 fraction 단위. 이름의 `_pct` 접미사를 지나치면 두 보고서를 그대로 섞게 됨.
**처방:** 두 보고서를 한 척도로 비교할 땐 반드시 `backtest/pipeline.py`의 `serialize_ml_wf_report()` 경유(/100 정규화 + 단위 고정 테스트 `tests/test_pipeline.py::TestMlWfAdapter`). 새 지표 추가 시 `_pct` 접미사면 fraction 변환 후 공통 스키마에 편입.
**참고:** 파이프라인 ML 통합 (2026-07-08).
