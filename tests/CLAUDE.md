# tests/

## What — 무엇을 하는가

`tradingbot` 패키지의 pytest 스위트. 20 개 `test_*.py` 파일이 모듈별 단위·통합 테스트를 모은다. 외부 거래소·네트워크는 mock 으로 대체하고 candle/Parquet I/O 는 합성 데이터(fixture)로 검증.

## How — 일반적인 수정

- **새 모듈 테스트 추가**: `test_<module>.py` 네이밍. `conftest.py` 의 `sample_candles` / `sample_df` / `tmp_data_dir` fixture 활용 (10 candle, 2024-01-01T00:00Z UTC 시작, 1h timeframe). 추가로 OHLCV 가 필요한 케이스만 자체 fixture.
- **백테스트/엔진 회귀 테스트**: `test_backtest_engine.py`·`test_multi_symbol.py`·`test_vectorized.py`. anti-lookahead 가 깨졌는지 확인하는 테스트는 여기서 추가 — 지표 추가 시 in/out-of-sample 대비 검증 케이스 동봉.
- **ML 회귀 테스트**: `test_ml.py`(trainer/walk_forward), `test_tuner.py`(Optuna), `test_threshold_tuner.py`, `test_diagnostics.py`. fixture 가 작은 합성 시계열을 만들어 LightGBM 가 실제 fit 되도록 한다 — 2 분 안에 끝나야 CI 부담 없음.
- **라이브/페이퍼**: `test_live_engine.py`·`test_live_trading.py`·`test_paper_exchange.py`·`test_ws_client.py`. 실제 ccxt/WebSocket 연결 금지, `unittest.mock` 또는 `BaseExchange` 자식 stub.
- **실행**: `pytest tests/ -v` (전체) 또는 `pytest tests/test_<x>.py::test_<func> -v` (단건).

## How not — 빌드를 깨뜨리는 비명백한 패턴

→ 누적 기록: [`anti-patterns.md`](./anti-patterns.md). 새 패턴 발견 시 반드시 append.

## Where — 의존성

- **Incoming**: `pytest`(직접 실행), CI(있다면 `pytest tests/`), 개발자 수동 실행.
- **Outgoing**: `tradingbot.*` 전 서브패키지(테스트 대상), `pandas` / `numpy` (합성 데이터), `lightgbm` (ML 테스트), `pytest` fixtures, `tmp_path`(파일 I/O 격리).

## Why — 코드에 안 적힌 부족 지식

- **fixture 가 10 candle 로 작은 이유**: 단위 테스트 대량 실행 시 빠르게 끝나야 함. 지표 warmup 이 부족한 케이스(SMA50 등)는 해당 테스트가 자체적으로 더 긴 series 를 만든다.
- **외부 거래소·네트워크는 항상 mock**: 실제 ccxt 호출은 rate-limit·인증·시세 변동으로 flaky. `BaseExchange` ABC 가 stub 만들기 쉬워서 도입.
- **ML 테스트가 LightGBM 을 실제 fit 하는 이유**: mock 으로 대체하면 calibrator·threshold tuner 의 메타 키 직렬화 회귀를 잡지 못해 inference 단에서 silent 실패. 작은 데이터로라도 end-to-end 가 원칙.
- **anti-lookahead 회귀 케이스의 위치**: `test_backtest_engine.py` — 새 지표·새 전략 도입 시 future leak 검증 케이스 동봉이 컨벤션. 코드만 보면 발견 어려움.
