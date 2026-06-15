#!/bin/bash
# One-shot AWS setup for algo_v2 on Ubuntu + NVIDIA GPU.
# Run: bash setup.sh
# Takes ~25–35 minutes on first run.

set -e
ALGO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${FINRL_ENV:-/home/ubuntu/finrl_env}"
IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "unknown")

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== algo_v2 AWS Setup (Ubuntu $(lsb_release -rs), IP: $IP) ==="

# ── 1. System packages ─────────────────────────────────────────────────────
log "[1/8] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-venv python3-pip python3-dev \
    build-essential wget curl git libgomp1

# ── 2. NVIDIA driver ───────────────────────────────────────────────────────
log "[2/8] Checking NVIDIA driver..."
if ! command -v nvidia-smi &>/dev/null; then
    log "    Installing NVIDIA driver 570 (T4 / data-centre GPU)..."
    sudo apt-get install -y -qq nvidia-driver-570-server
    log "    Driver installed."
    log ""
    log "  ┌─────────────────────────────────────────────────────────────────┐"
    log "  │  NVIDIA driver installed. Testing if kernel module loads...     │"
    log "  └─────────────────────────────────────────────────────────────────┘"
    sudo modprobe nvidia 2>/dev/null || true
    nvidia-smi || log "WARNING: nvidia-smi still not working — may need reboot after setup."
else
    log "    NVIDIA driver already installed:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
fi

# ── 3. Python venv ────────────────────────────────────────────────────────
log "[3/7] Creating Python venv at $VENV_DIR..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install -q --upgrade pip setuptools wheel
log "    venv ready (Python $(python3 --version))."

# ── 5. PyTorch (CUDA 13.0) ────────────────────────────────────────────────
log "[4/7] Installing PyTorch with CUDA 13.0..."
log "    (this is ~2–3 GB, takes a few minutes)"
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
python3 -c "import torch; print(f'    torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# ── 6. Project requirements ───────────────────────────────────────────────
log "[5/7] Installing the algo_v2 package + dependencies (editable)..."
cd "$ALGO_DIR"
pip install -q -e .
log "    All packages installed."

# ── 7. Telegram credentials ───────────────────────────────────────────────
log "[6/7] Telegram credentials setup..."
ENV_FILE="$ALGO_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    cp "$ALGO_DIR/.env.example" "$ENV_FILE"
fi

# Check if credentials are already set
source "$ENV_FILE" 2>/dev/null || true
if [ -z "$TELEGRAM_BOT_TOKEN" ] || [ "$TELEGRAM_BOT_TOKEN" = "your_bot_token_here" ]; then
    echo ""
    echo "  ┌─────────────────────────────────────────────────────────────────┐"
    echo "  │  Enter your Telegram credentials (press ENTER to skip for now)  │"
    echo "  └─────────────────────────────────────────────────────────────────┘"
    read -r -p "  TELEGRAM_BOT_TOKEN: " BOT_TOKEN
    read -r -p "  TELEGRAM_CHAT_ID:   " CHAT_ID
    if [ -n "$BOT_TOKEN" ]; then
        sed -i "s|TELEGRAM_BOT_TOKEN=.*|TELEGRAM_BOT_TOKEN=$BOT_TOKEN|" "$ENV_FILE"
        sed -i "s|TELEGRAM_CHAT_ID=.*|TELEGRAM_CHAT_ID=$CHAT_ID|" "$ENV_FILE"
        log "    Credentials saved to .env"
    else
        log "    Skipped — Telegram notifications will be disabled."
    fi
fi

set -a
source "$ENV_FILE"
set +a

# Persist env vars across SSH sessions
grep -qxF "set -a; source $ENV_FILE; set +a" "$HOME/.bashrc" 2>/dev/null || \
    echo "set -a; source $ENV_FILE; set +a" >> "$HOME/.bashrc"

# Also persist venv activation
grep -qxF "source $VENV_DIR/bin/activate" "$HOME/.bashrc" 2>/dev/null || \
    echo "source $VENV_DIR/bin/activate" >> "$HOME/.bashrc"

# ── 8. Data fetch ─────────────────────────────────────────────────────────
log "[7/7] Fetching market data (NIFTY 50, 2010–present)..."
log "    Downloading ~50 stocks × 16 years from Yahoo Finance..."
python3 -m algo_v2.data.fetch
log "    indian_stocks_15yr.csv (PRIMARY_DATA_PATH) ready."

# ── Done ──────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                    Setup complete!                                   ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║                                                                      ║"
echo "║  [A] Verify the full stack is wired correctly (do this first):       ║"
echo "║      bash scripts/preflight_check.sh                                 ║"
echo "║                                                                      ║"
echo "║  [B] Optional — tune hyperparameters (~2–4 hrs):                     ║"
echo "║      python3 -m algo_v2.training.tune                                ║"
echo "║                                                                      ║"
echo "║  [C] Start 42-bot training pipeline (detached):                      ║"
echo "║      nohup bash scripts/phase6_master.sh &                           ║"
echo "║      tail -f phase6.log                                              ║"
echo "║                                                                      ║"
echo "║  [D] Monitor training via CloudWatch (no tunnel needed):             ║"
echo "║      bash scripts/setup_cloudwatch.sh   ← verify IAM role first     ║"
echo "║      AWS Console → CloudWatch → Metrics → algo_v2/training          ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
