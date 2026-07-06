#!/usr/bin/env bash
# 새 PC 부트스트랩 원샷: venv + 고정 의존성 + .env + git hooks.
# 몇 번을 실행해도 안전(idempotent). 데이터 다운로드/모델 학습은 네트워크와
# 시간이 들어 자동화하지 않고 마지막에 다음 단계로 안내만 한다.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
[ -d .venv ] || "$PYTHON" -m venv .venv
.venv/bin/pip install -e ".[dev]" -c constraints.txt

# 실키가 든 기존 .env 는 절대 덮어쓰지 않는다.
[ -f .env ] || cp .env.example .env

# 버전 관리되는 pre-push 훅(문서 동기화 가드) 활성화
git config core.hooksPath scripts/git-hooks

echo ""
echo "부트스트랩 완료. 다음 단계:"
echo "  1) .env 에 Upbit/Telegram 키 입력 (live 거래 시에만 필요)"
echo "  2) source .venv/bin/activate"
echo "  3) tradingbot download --symbol BTC/KRW --timeframe 1h --since 2024-01-01"
echo "     tradingbot download-external --since 2024-01-01   # ML 외부 피처"
echo "     tradingbot ml-train-all                            # 모델 재생성"
