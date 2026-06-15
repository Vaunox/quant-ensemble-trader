#!/bin/bash
# Phase 6: Full Training Pipeline
# Run with:  nohup bash phase6_master.sh > /dev/null 2>&1 &
# Monitor:   tail -f phase6.log
#
# CHECKPOINT SYSTEM:
#   Each phase saves its output to checkpoints/ after completing.
#   On restart, completed phases are skipped automatically.
#   Checkpoint folders:
#     checkpoints/phase1_gan/       - market_generator.pth + gan_scaling_params.csv
#     checkpoints/phase2_synthetic/ - indian_stocks_synthetic.csv
#     checkpoints/phase3_expert/    - expert_data/ folder
#     checkpoints/phase4_ensemble/  - trained_models/ensemble/ folder
#
# ── STARTING A FRESH RUN ─────────────────────────────────────────────────────
#   To retrain everything from scratch:
#
#     rm -rf checkpoints/phase1_gan checkpoints/phase2_synthetic checkpoints/phase3_expert
#     rm -rf trained_models/ensemble/*
#     nohup bash phase6_master.sh > /dev/null 2>&1 &
#
#   This forces Phase 1 (GAN), Phase 2 (synthetic), Phase 3 (expert) to rerun
#   on the new data before Phase 4 (42-bot training) begins.
#   PRIMARY_DATA_PATH in src/algo_v2/config.py must be set to 'indian_stocks_15yr.csv'.
# ─────────────────────────────────────────────────────────────────────────────

set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
source "${FINRL_ENV:-/home/ubuntu/finrl_env}/bin/activate"

LOG=phase6.log
# Only clear log on a fresh run (no checkpoints exist yet)
if [ ! -d "checkpoints/phase1_gan" ] && [ ! -d "checkpoints/phase2_synthetic" ]; then
    > "$LOG"
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
# Print elapsed time in Xm Ys format given a $SECONDS snapshot
_elapsed() { local s=$(( SECONDS - $1 )); printf '%dm%02ds' $(( s / 60 )) $(( s % 60 )); }

# Create checkpoint directories
mkdir -p checkpoints/phase1_gan \
         checkpoints/phase2_synthetic \
         checkpoints/phase3_expert \
         checkpoints/phase4_ensemble \
         checkpoints/data \
         trained_models

log "=========================================="
log "PHASE 6: INSTITUTIONAL FACTORY BOOTUP"
log "=========================================="

# Kill any stale sidecar from a previous session before starting a fresh one.
# Without this, two sidecars tail the same log and write duplicate metrics.
pkill -f algo_v2.monitoring.cw_sidecar 2>/dev/null || true; sleep 1
nohup python3 -u -m algo_v2.monitoring.cw_sidecar > cw_sidecar.log 2>&1 &
SIDECAR_PID=$!
log "CloudWatch sidecar started (PID $SIDECAR_PID)"

# ── PHASE 1: WGAN-GP ────────────────────────────────────────────────────────
# algo_v2.data.gan.train_gan writes directly to checkpoints/phase1_gan/ — no intermediate copy.
_t1=$SECONDS
if [ -f "checkpoints/phase1_gan/market_generator.pth" ]; then
    log "[1/4] GAN checkpoint found — skipping training."
else
    log "[1/4] Training WGAN-GP..."
    python3 -u -m algo_v2.data.gan.train_gan 2>&1 | tee -a "$LOG"
    log "[+] Phase 1 complete ($(_elapsed _t1))."
fi

# ── PHASE 2: SYNTHETIC MARKET ────────────────────────────────────────────────
# algo_v2.data.gan.generate writes directly to checkpoints/phase2_synthetic/ — no copy.
_t2=$SECONDS
if [ -f "checkpoints/phase2_synthetic/indian_stocks_synthetic.csv" ]; then
    log "[2/4] Synthetic data checkpoint found — skipping generation."
else
    log "[2/4] Generating 60-year synthetic market data..."
    python3 -u -m algo_v2.data.gan.generate 2>&1 | tee -a "$LOG"
    log "[+] Phase 2 complete ($(_elapsed _t2))."
fi

# ── PHASE 3: EXPERT TRAJECTORIES ─────────────────────────────────────────────
_t3=$SECONDS
if [ -d "checkpoints/phase3_expert" ] && [ "$(ls -A checkpoints/phase3_expert)" ]; then
    log "[3/4] Expert trajectories checkpoint found — restoring and skipping."
    cp -r checkpoints/phase3_expert/. expert_data/
else
    log "[3/4] Running Hindsight Oracle (expert trajectory generator)..."
    python3 -u -m algo_v2.training.expert_trajectories 2>&1 | tee -a "$LOG"
    log "[+] Phase 3 complete ($(_elapsed _t3)) — saving checkpoint."
    mkdir -p checkpoints/phase3_expert
    cp -r expert_data/. checkpoints/phase3_expert/
fi

# ── PHASE 4: 42-BOT ENSEMBLE ─────────────────────────────────────────────────
# PARALLEL_GROUPS controls how many bots train simultaneously.
#   PARALLEL_GROUPS=1  → sequential (default, safe for 8-core g4dn.2xlarge)
#   PARALLEL_GROUPS=7  → parallel   (recommended for 64-core c6i.16xlarge)
# Override at launch:  PARALLEL_GROUPS=7 nohup bash phase6_master.sh ...
# N_GROUPS, not GROUPS: bash's special GROUPS array silently ignores assignments,
# so `GROUPS=...` expands to the user's primary GID (1000) downstream.
N_GROUPS=${PARALLEL_GROUPS:-21}
_t4=$SECONDS
log "[4/4] Starting 42-Bot Ensemble Factory (parallel groups: $N_GROUPS)..."
bash "$SCRIPT_DIR/launch_parallel_training.sh" "$N_GROUPS" 42 2>&1 | tee -a "$LOG"
log "[+] Phase 4 complete ($(_elapsed _t4)) — saving checkpoint."
cp -r trained_models/ensemble/. checkpoints/phase4_ensemble/

kill "$SIDECAR_PID" 2>/dev/null || true

# ── PHASE 5: WALK-FORWARD VALIDATION ─────────────────────────────────────────
_t5=$SECONDS
log "[5/5] Scoring all 42 bots on 2022-2023 validation data..."
python3 -u -m algo_v2.evaluation.validate 2>&1 | tee -a "$LOG"
log "[+] Phase 5 complete ($(_elapsed _t5)) — results saved to val_results.csv"

log "=========================================="
log "PHASE 6 COMPLETE."
log "Next step: python -m algo_v2.evaluation.validate --test   (run once, 2024 final score)"
log "=========================================="
