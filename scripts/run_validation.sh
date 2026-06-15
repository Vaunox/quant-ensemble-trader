#!/bin/bash
# Runs algo_v2.evaluation.validate once per bot — each bot gets its own Python/Ray process.
# Ray fully shuts down between bots so memory never accumulates across 39 bots.
# Crash-safe: already-scored bots are skipped on re-run (checks val_results.csv).
set -eo pipefail

cd "$(dirname "$0")/.."
source "${FINRL_ENV:-/home/ubuntu/finrl_env}/bin/activate"

LOG=val_perbot.log
VAL_CSV=checkpoints/phase5_validation/val_results.csv

# Kill any existing validation run cleanly
pkill -f algo_v2.evaluation.validate || true
sleep 2

> "$LOG"
echo "Per-bot validation started at $(date)" | tee -a "$LOG"

# Build ordered bot list (final dirs + _best_ckpt fallbacks, same logic as _resolve_checkpoints)
BOT_IDS=$(python3 - <<'PYEOF'
import os
d = 'trained_models/ensemble'
dirs = set(os.listdir(d))
final   = sorted(x for x in dirs if not x.endswith('_best_ckpt') and x.startswith('bot_'))
best    = sorted(x[:-len('_best_ckpt')] for x in dirs
                 if x.endswith('_best_ckpt') and x[:-len('_best_ckpt')] not in dirs)
for b in sorted(set(final) | set(best)):
    print(b)
PYEOF
)

TOTAL=$(echo "$BOT_IDS" | wc -l)
COUNT=0

for bot_id in $BOT_IDS; do
    COUNT=$((COUNT + 1))

    # Skip if already scored (crash recovery)
    if [ -f "$VAL_CSV" ] && grep -q "^${bot_id}," "$VAL_CSV" 2>/dev/null; then
        echo "[$COUNT/$TOTAL] SKIP $bot_id (already scored)" | tee -a "$LOG"
        continue
    fi

    echo "[$COUNT/$TOTAL] Scoring $bot_id at $(date)" | tee -a "$LOG"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m algo_v2.evaluation.validate --bots "$bot_id" >> "$LOG" 2>&1
    RET=$?
    if [ $RET -ne 0 ]; then
        echo "  [ERROR] $bot_id exited with code $RET" | tee -a "$LOG"
    else
        echo "  [OK] $bot_id done" | tee -a "$LOG"
    fi
done

echo "All bots complete at $(date)" | tee -a "$LOG"
