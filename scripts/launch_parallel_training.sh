#!/bin/bash
# launch_parallel_training.sh
# Splits 42-bot Phase 4 training across N parallel Ray instances.
#
# Each group runs in its own Ray cluster (unique temp dir + port) so there
# is zero cross-group interference. All groups share the same output dir
# (trained_models/ensemble/) — bot IDs are globally unique so no conflict.
#
# Usage:
#   bash launch_parallel_training.sh [GROUPS] [TOTAL_BOTS]
#
# Examples:
#   bash launch_parallel_training.sh 1  42   # 8-core instance  (sequential, same as before)
#   bash launch_parallel_training.sh 7  42   # 64-core instance (7x speedup, ~12 hr)
#
# Recommended instance for GROUPS>1:  c6i.16xlarge  (64 vCPU, 128 GB, ~$2.72/hr)
# Phases 1-3 checkpoints are copied from the GPU instance before running this.

set -eo pipefail
cd "$(dirname "$0")/.."
source "${FINRL_ENV:-/home/ubuntu/finrl_env}/bin/activate"

# N_GROUPS, not GROUPS: bash's special GROUPS array silently ignores assignments,
# so `GROUPS=...` expands to the user's primary GID (1000) downstream.
N_GROUPS=${1:-1}
TOTAL=${2:-42}
# Guard: a caller running the pre-fix phase6_master.sh passes the GID (e.g. 1000)
# instead of a group count. >21 groups can never be right for 42 bots.
if ! [[ "$N_GROUPS" =~ ^[0-9]+$ ]] || [ "$N_GROUPS" -lt 1 ] || [ "$N_GROUPS" -gt 42 ]; then
    echo "[WARN] groups arg '$N_GROUPS' is invalid — falling back to 1 (sequential)"
    N_GROUPS=1
fi
BOTS_PER_GROUP=$(( (TOTAL + N_GROUPS - 1) / N_GROUPS ))

LOG_DIR="parallel_logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Parallel training: $N_GROUPS groups × ~$BOTS_PER_GROUP bots each (total $TOTAL)"

PIDS=()
for (( g=0; g<N_GROUPS; g++ )); do
    S=$(( g * BOTS_PER_GROUP + 1 ))
    E=$(( (g + 1) * BOTS_PER_GROUP ))
    [ $E -gt $TOTAL ] && E=$TOTAL
    [ $S -gt $TOTAL ] && break

    GRP_LOG="$LOG_DIR/group${g}_bots${S}-${E}.log"
    log "[LAUNCH] Group $g: bots $S-$E → $GRP_LOG"

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2 RAY_memory_monitor_refresh_ms=0 nice -n 15 python3 -u -m algo_v2.training.train_ensemble \
        --start-bot "$S" --end-bot "$E" \
        > "$GRP_LOG" 2>&1 &
    PIDS+=($!)
    # Stagger launches: all groups hitting Ray init + data load simultaneously
    # caused a memory spike that OOM-killed first-step workers (bot_37 incident).
    # 30s gap spreads the peak across 10+ minutes instead of 1 second.
    [ $((g + 1)) -lt "$N_GROUPS" ] && sleep 30
done

log "All ${#PIDS[@]} groups launched (PIDs: ${PIDS[*]}). Waiting for completion..."

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        log "[WARN] Process $pid exited with error"
        FAILED=1
    fi
done

log "Merging group logs..."
for f in "$LOG_DIR"/group*.log; do
    echo "" >> phase6.log
    echo "=== $(basename $f) ===" >> phase6.log
    cat "$f" >> phase6.log
done

if [ $FAILED -eq 0 ]; then
    log "[+] All $N_GROUPS groups completed successfully."
else
    log "[WARN] One or more groups failed — check $LOG_DIR/ for details."
    exit 1
fi
