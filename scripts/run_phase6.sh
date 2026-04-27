#!/usr/bin/env bash
# Phase 6 full evaluation pipeline.
#
# Runs the five-step retrain + tune + threshold + scan + combine-scan pipeline
# end-to-end, with timestamps. Wall-clock estimate ~7h on a recent Mac with
# workers tuned for the machine. Total compute ≈ 24 LGBM models × (train +
# Optuna 50 trials @ 5min budget + threshold sweep) plus two scan passes.
#
# Logs to logs/phase6_<UTC-timestamp>.log; tee'd to stdout so the user can
# tail it. Stops on first failure.
#
# Usage: bash scripts/run_phase6.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -f ".venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
    else
        echo "[phase6] no .venv found and no VIRTUAL_ENV set — aborting" >&2
        exit 1
    fi
fi

mkdir -p logs
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="logs/phase6_${TS}.log"

step() {
    local name="$1"; shift
    local started; started="$(date -u +%H:%M:%SZ)"
    echo "" | tee -a "$LOG"
    echo "===== [phase6 ${started}] ${name} =====" | tee -a "$LOG"
    echo "+ $*" | tee -a "$LOG"
    "$@" 2>&1 | tee -a "$LOG"
    local ended; ended="$(date -u +%H:%M:%SZ)"
    echo "----- [phase6 ${ended}] ${name} done -----" | tee -a "$LOG"
}

echo "===== Phase 6 pipeline start $(date -u +%FT%TZ) =====" | tee -a "$LOG"
echo "Log: $LOG" | tee -a "$LOG"

# 1. Retrain all 24 (symbol × timeframe) models with triple-barrier labels.
step "ml-train-all (triple-barrier)" \
    tradingbot ml-train-all --target-kind triple-barrier --atr-mult 1.0 --workers 8

# 2. Optuna tune every model (30 trials × 5min budget × 24 models, workers=4).
step "ml-tune-all (Optuna)" \
    tradingbot ml-tune-all --workers 4 --time-budget 300 --trials 30

# 3. Per-model entry/exit threshold sweep on the freshly tuned models.
step "ml-tune-thresholds-all" \
    tradingbot ml-tune-thresholds-all --workers 4

# 4. Standalone lgbm scan — measures alpha vs rule-based on holdout.
step "scan --top 50" \
    tradingbot scan --top 50

# 5. ML+filter combine-scan — measures combined alpha; verify-top re-runs
#    the top 50 through the full backtest engine for a faithful Sharpe.
step "combine-scan --verify-top 50" \
    tradingbot combine-scan --verify-top 50

echo "" | tee -a "$LOG"
echo "===== Phase 6 pipeline done $(date -u +%FT%TZ) =====" | tee -a "$LOG"
