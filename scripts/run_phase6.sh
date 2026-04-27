#!/usr/bin/env bash
# Phase 6 full evaluation pipeline.
#
# Runs the five-step retrain + tune + threshold + scan + combine-scan pipeline
# end-to-end, with timestamps. Wall-clock estimate ~7.5h on a recent Mac with
# workers tuned for the machine. Total compute ≈ 24 LGBM models × (train +
# Optuna 50 trials @ 30min cap + threshold sweep) plus two scan passes.
#
# Trials and time-budget are sized from Phase 3 sandbox data: best-value was
# reached at trials 13 / 33 / 39 (ETH / BTC / XRP) with ~30s/trial, so 50
# trials @ ~25min is the convergence horizon we observed. The 30min cap
# (--time-budget 1800) is just an outlier fuse for degenerate models that
# never early-stop — under normal conditions Optuna terminates on trial
# count first. With workers=4 the step takes ~2.5h wall-clock.
#
# Both ml-train-all and ml-tune-all use --train-months 6 / --test-months 2
# so the baseline trained in step 1 stays comparable to the model retrained
# inside ml-tune-all's final-fit phase. (The CLI defaults differ — 3/1 for
# ml-train-all vs 6/2 for ml-tune-all — and unaligned windows would make
# step 1's diagnostics meaningless.)
#
# Logs to logs/phase6_<UTC-timestamp>.log; tee'd to stdout so the user can
# tail it. Stops on first failure. The deliverable scan / combine-scan
# tables are written directly via --output to personal/*.md.
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
    tradingbot ml-train-all \
    --target-kind triple-barrier --atr-mult 1.0 \
    --train-months 6 --test-months 2 \
    --workers 8

# 2. Optuna tune every model (50 trials × 30min cap × 24 models, workers=4).
#    Same windows as step 1 so the final-fit overwrite stays consistent.
#    50 trials matches Phase 3's observed convergence (BTC@33, XRP@39); the
#    30min cap is an outlier fuse, not the expected stop condition.
step "ml-tune-all (Optuna)" \
    tradingbot ml-tune-all \
    --target-kind triple-barrier --atr-mult 1.0 \
    --train-months 6 --test-months 2 \
    --workers 4 --time-budget 1800 --trials 50

# 3. Per-model entry/exit threshold sweep on the freshly tuned models.
step "ml-tune-thresholds-all" \
    tradingbot ml-tune-thresholds-all --workers 4

# 4. Standalone lgbm scan — measures alpha vs rule-based on holdout.
#    --output writes the Top-50 table directly into the deliverable md.
step "scan --top 50" \
    tradingbot scan --top 50 \
    --output personal/scan_holdout_result.md

# 5. ML+filter combine-scan — measures combined alpha; verify-top re-runs
#    the top 50 through the full backtest engine for a faithful Sharpe.
step "combine-scan --verify-top 50" \
    tradingbot combine-scan --verify-top 50 \
    --output personal/combine_scan_holdout_result.md

echo "" | tee -a "$LOG"
echo "===== Phase 6 pipeline done $(date -u +%FT%TZ) =====" | tee -a "$LOG"
