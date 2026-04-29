# scripts/

## What — 무엇을 하는가

운영·진단 보조 스크립트 모음. Docker healthcheck, Phase 6 ML 파이프라인 실행, 모델별 calibrator 분포 확인, pre-push 단계의 CLAUDE.md 동기화 훅 소스를 둔다. `tradingbot` CLI 외부에서 도는 것만 여기에 둔다 — CLI로 흡수 가능한 로직은 `src/tradingbot/cli.py` 쪽으로.

## How — 일반적인 수정

- **Phase 파이프라인 변경**: `run_phase6.sh` 의 5단계(`ml-train-all` → `ml-tune-all` → `ml-tune-thresholds-all` → `scan` → `combine-scan`) 인자 수정. train/tune-all 의 `--train-months 6 --test-months 2` 정렬은 깨지 않게 유지.
- **헬스체크 정책 변경**: `healthcheck.py` 의 `MAX_STALE_SECONDS` 기본값 또는 `STATE_FILE` 환경변수 해석 수정. exit 0/1 계약은 Dockerfile 의 HEALTHCHECK 기대값.
- **새 진단 스크립트**: `inspect_<topic>.py` 네이밍으로 추가. 모델·calibrator 로드는 `inspect_eth_calibrator.py` 패턴 차용.
- **pre-push 훅 변경**: `git-hooks/pre-push` 수정 후 반드시 `cp scripts/git-hooks/pre-push .git/hooks/pre-push && chmod +x .git/hooks/pre-push` (직접 복사, symlink 안 씀).

## How not — 빌드를 깨뜨리는 비명백한 패턴

→ 누적 기록: [`anti-patterns.md`](./anti-patterns.md). 새 패턴 발견 시 반드시 append.

## Where — 의존성

- **Incoming**: `Dockerfile`(healthcheck.py 복사), 사람이 수동 실행(`run_phase6.sh`, `inspect_*.py`), git push(`git-hooks/pre-push`).
- **Outgoing**: `tradingbot` CLI(`run_phase6.sh`가 호출), `tradingbot.data.external_fetcher` / `tradingbot.ml`(`inspect_eth_calibrator.py`), `state.json`(healthcheck), `claude` CLI(`git-hooks/pre-push`).

## Why — 코드에 안 적힌 부족 지식

- **`run_phase6.sh` 가 6/2 윈도우로 통일한 이유**: CLI 기본은 `ml-train-all` 3/1, `ml-tune-all` 6/2 라 그대로 이으면 step 1 진단이 step 2 재학습 모델과 호환되지 않는다. 스크립트 주석에도 명시.
- **Optuna trial 50 + 30분 cap 선택 근거**: Phase 3 sandbox 에서 best-value 가 trials 13/33/39 에 도달 → 50 이 수렴 horizon. 30분 cap 은 early-stop 안 되는 outlier 보호용 fuse.
- **healthcheck 가 `saved_at` 누락 시 healthy 반환**: 부팅 직후 state 가 채 안 쓰인 구간을 죽이지 않으려고. Docker `start_period` 와 함께 동작.
- **pre-push 훅이 worktree dirty 면 skip**: 자동 commit 이 사용자 변경과 충돌하는 사고를 막기 위함. 의도적 보수 동작.
