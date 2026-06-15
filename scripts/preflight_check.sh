#!/bin/bash
# preflight_check.sh
# Two-phase validation before every training run.
#
# Phase 1 — Config validation (~20 min)
#   All 42 bots, 1 worker, 1 iteration. Catches every model/config/import bug.
#
# Phase 2 — Throughput check (~10 min)
#   6 representative bots (pure_mlp × each algo), full workers, 5 iterations.
#   Measures seconds/iter and warns if any algo would overrun under parallel load.
#   Catches PPO-style slowness that Phase 1 misses (1 worker hides batch cost).
#
# Usage:
#   bash preflight_check.sh
#
# Exit code: 0 = all pass, 1 = config failures or throughput warnings found

set -eo pipefail
cd "$(dirname "$0")/.."
source "${FINRL_ENV:-/home/ubuntu/finrl_env}/bin/activate"

LOG="preflight.log"
> "$LOG"

# ── Phase 1: Config validation ────────────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== PHASE 1: Config Validation (42 bots × 1 iter) ====="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== PHASE 1: Config Validation (42 bots × 1 iter) =====" >> "$LOG"

python3 -u -m algo_v2.training.train_ensemble --start-bot 1 --end-bot 42 --preflight 2>&1 | tee -a "$LOG"

PASS=$(grep -c '\[SUCCESS\]' "$LOG" 2>/dev/null || true)
FAIL=$(grep -c '\[FATAL\]'   "$LOG" 2>/dev/null || true)

echo ""
echo "========================================"
echo "  Phase 1 — PASSED: $PASS / 42   FAILED: $FAIL / 42"
echo "========================================"

if [ "${FAIL:-0}" -gt 0 ]; then
    echo ""
    echo "FAILED BOTS:"
    grep '\[FATAL\]' "$LOG" || true
    echo ""
    echo "[PREFLIGHT] PHASE 1 FAILED — fix errors above before launching training."
    exit 1
fi

echo ""
echo "[PREFLIGHT] Phase 1 passed. Running throughput check..."
echo ""

# ── Phase 2: Throughput check ─────────────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== PHASE 2: Throughput Check (6 algos × 5 iters, full config) ====="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== PHASE 2: Throughput Check (6 algos × 5 iters, full config) =====" >> "$LOG"

python3 -u -m algo_v2.training.train_ensemble --start-bot 1 --end-bot 42 --timing 2>&1 | tee -a "$LOG"

TIMING_OK   =$(grep -c '\[TIMING_OK\]'   "$LOG" 2>/dev/null || true)
TIMING_WARN=$(grep -c '\[TIMING_WARN\]' "$LOG" 2>/dev/null || true)

echo ""
echo "========================================"
echo "  Phase 2 — OK: $TIMING_OK / 6   SLOW: $TIMING_WARN / 6"
echo "========================================"

if [ "${TIMING_WARN:-0}" -gt 0 ]; then
    echo ""
    echo "SLOW ALGOS:"
    grep '\[TIMING_WARN\]' "$LOG" || true
    echo ""
    echo "[PREFLIGHT] THROUGHPUT WARNING — training will run slowly under parallel load."
    echo "  Options:"
    echo "    1. Reduce PARALLEL_GROUPS (fewer concurrent Ray clusters = more CPU per group)"
    echo "    2. Reduce PPO train_batch_size in src/algo_v2/training/train_ensemble.py (4000 instead of 8000)"
    echo "    3. Proceed anyway — expect longer training time than estimated"
    echo ""
    echo "[PREFLIGHT] Phase 1 passed, Phase 2 warns. Proceed with caution."
    exit 1
fi

echo ""
echo "[PREFLIGHT] ALL CHECKS PASSED — safe to launch full training."
echo "  Next step: PARALLEL_GROUPS=21 nohup bash phase6_master.sh > /dev/null 2>&1 &"
