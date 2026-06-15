#!/bin/bash
# bootstrap_c6i.sh — one-shot environment setup for the c6i.32xlarge Phase-4 box.
# Run as ubuntu on a fresh Ubuntu 22.04 instance AFTER /tmp/algo_v2_migration.tgz
# has been copied over (MIGRATION_RUNBOOK.md step 3):
#
#   bash bootstrap_c6i.sh
#
# Mirrors the g4dn environment exactly: Python 3.14.4 + requirements_frozen.txt
# (143 pins incl. torch==2.12.0+cu130 — needs the PyTorch cu130 index; the CUDA
# wheels run fine on a CPU-only box, RLlib just won't use a GPU, which is the
# intended num_gpus=0 configuration anyway).

set -eo pipefail
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

[ -f /tmp/algo_v2_migration.tgz ] || {
    echo "ERROR: /tmp/algo_v2_migration.tgz not found — scp it from the g4dn first."
    exit 1
}

log "[1/4] Installing Python 3.14 (deadsnakes PPA)..."
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.14 python3.14-venv python3.14-dev build-essential

log "[2/4] Extracting project to /home/ubuntu/algo_v2..."
tar xzf /tmp/algo_v2_migration.tgz -C /home/ubuntu/

log "[3/4] Creating virtualenv at /home/ubuntu/finrl_env..."
python3.14 -m venv /home/ubuntu/finrl_env
source /home/ubuntu/finrl_env/bin/activate
pip install --upgrade pip

log "[4/4] Installing 143 pinned packages (torch cu130 wheels are ~3 GB — be patient)..."
pip install -r /home/ubuntu/algo_v2/requirements_frozen.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130

log "Bootstrap complete."
log "Launch:   cd /home/ubuntu/algo_v2 && PARALLEL_GROUPS=14 nohup bash scripts/phase6_master.sh > /dev/null 2>&1 &"
log "Monitor:  tail -f /home/ubuntu/algo_v2/phase6.log"
