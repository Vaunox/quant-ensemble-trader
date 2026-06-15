# algo_v2 — 42-Bot RL Ensemble Trading System

A reinforcement-learning ensemble that trades the NIFTY 50 using 42 bots (6 algorithms × 7 neural architectures) trained on 16 years of Indian equity data augmented with 60 years of GAN-generated synthetic market data.

---

## Architecture Overview

The pipeline runs as five phases (orchestrated by `scripts/phase6_master.sh`), plus a daily live executor:

| Phase | Entry point | What it does |
|---|---|---|
| 1 | `python -m algo_v2.data.gan.train_gan` | WGAN-GP learns return distributions from 16yr real data |
| 2 | `python -m algo_v2.data.gan.generate` | Generates a 15,000-day (60yr) synthetic market from the GAN |
| 3 | `python -m algo_v2.training.expert_trajectories` | Hindsight Oracle → MARWIL imitation-learning data |
| 4 | `python -m algo_v2.training.train_ensemble` | Trains 42 bots in `CompetitiveTradingEnv` (5-agent CLOB) |
| 5 | `python -m algo_v2.evaluation.validate` | Walk-forward validation on 2022–2023 held-out data |
| Daily | `python -m algo_v2.live.execute` | Regime-aware weighted ensemble → broker orders |

**42 bots = 6 algorithms × 7 architectures**

| Algorithms | Architectures |
|---|---|
| PPO, APPO, SAC, IMPALA, TQC, MARWIL | pure_mlp, pure_lstm, cnn_micro, cnn_macro, cnn_dual, transformer_cnn, god_tier |

---

## Project Structure

```
src/algo_v2/
  config.py            Single source of truth: tickers, indicators, paths, ARCH_MAP
  platform/            Cross-cutting: logging_config, notifications, rllib_env wrapper
  core/                Business logic: env_single, env_competitive, models (+ register_models)
  data/                features (indicators/VIX), fetch; gan/ (train_gan, generate)
  training/            train_ensemble, tune, expert_trajectories
  evaluation/          validate (walk-forward + test scoring)
  live/                execute (daily), broker, regime, meta_controller
  monitoring/          cw_sidecar, cw_dashboard
scripts/               Shell entry points (setup, phase6_master, preflight, …) + smoke_test.sh
tests/smoke/           Fast import/config/env smoke checks
docs/exploratory/      Unintegrated v3 prototypes (not installed by default)
```

---

## Setup

### 1. PyTorch (install first, with the CUDA wheel for your box)

```bash
# GPU box (Phases 1–3 use the GPU). Pick the index matching your CUDA version.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# CPU-only box (Phase 4 training / validation run fine on CPU):
# pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Install the package + dependencies

```bash
pip install -e .            # runtime deps from pyproject.toml
pip install -e ".[dev]"     # + ruff, pytest (development)
```

`scripts/setup.sh` automates the full EC2 bootstrap (system packages, venv, torch, `pip install -e .`, data fetch).

### 3. Environment variables

```bash
cp .env.example .env
# Edit .env and fill in your Telegram credentials (never commit this file)
```

### 4. Data

The 16-year NIFTY 50 dataset (`indian_stocks_15yr.csv`, 49 tickers, 2010–2026) must be present in the project root. To regenerate it from scratch:

```bash
python -m algo_v2.data.fetch
```

---

## Running the Pipeline

### Full training run (EC2)

```bash
# STEP 0 (once): attach IAM role with CloudWatchFullAccess, then verify access
bash scripts/setup_cloudwatch.sh

# STEP 1: kill orphans + hard reset before a fresh run
pkill -9 -f 'algo_v2.training.train_ensemble|launch_parallel|ray::' 2>/dev/null || true; sleep 5
rm -rf /tmp/ray_bots_* checkpoints/ trained_models/
rm -f phase6.log

# STEP 2: preflight — validates all 42 bot configs (catches bugs with 1 iter/worker)
bash scripts/preflight_check.sh

# STEP 3: full training (~4.5 hours on c6a.48xlarge)
PARALLEL_GROUPS=21 nohup bash scripts/phase6_master.sh > /dev/null 2>&1 &
tail -f phase6.log
```

### Validation and test scoring

```bash
# Walk-forward validation on 2022–2023 data
python -m algo_v2.evaluation.validate

# Final test score on 2024 data — run ONCE only
python -m algo_v2.evaluation.validate --test
```

### Live / paper trading

```bash
# Dry run (compute orders, no execution)
python -m algo_v2.live.execute --dry-run

# Live run (executed by cron at 4:00 PM IST on weekdays)
python -m algo_v2.live.execute
```

Cron entry (server time = UTC):
```
30 10 * * 1-5  cd /home/ubuntu/algo_v2 && source /home/ubuntu/finrl_env/bin/activate && python -m algo_v2.live.execute >> live.log 2>&1
```

---

## Testing

A fast smoke suite (no training, no network) proves the wiring is intact — imports, the config contract, an env reset/step, logging/notifications, CLI entry points, and shell-script syntax:

```bash
bash scripts/smoke_test.sh
# or point at a specific interpreter:
PYTHON=/path/to/venv/bin/python bash scripts/smoke_test.sh
```

---

## Configuration

All tuneable constants live in `src/algo_v2/config.py`:

| Constant | Default | Purpose |
|---|---|---|
| `PRIMARY_DATA_PATH` | `indian_stocks_15yr.csv` | Training data source |
| `VAL_START_DATE` | `2022-01-01` | Validation period start |
| `TEST_START_DATE` | `2024-01-01` | Test period start (touch once) |
| `INITIAL_CAPITAL` | 1,000,000 | Starting portfolio value (INR) |
| `HMAX` | 5,000 | Max shares per order |
| `ACTION_THRESHOLD` | 0.3 | Min signal magnitude to place an order |
| `ARCH_MAP` | (7 entries) | Bot architecture definitions |
| `PROJECT_ROOT`, `CHECKPOINTS_DIR`, `TRAINED_MODELS_DIR`, `EXPERT_DATA_DIR`, `LIVE_STATES_DIR` | (derived) | Runtime-artifact locations, anchored at the repo root |
| `AWS_REGION` | `ap-south-1` (env `AWS_REGION`) | CloudWatch region |

Logging level is controlled by the `LOG_LEVEL` environment variable (default `INFO`); set `LOG_FILE` to also write to a file.

---

## Infrastructure

The pipeline targets two AWS EC2 instances:

- **GPU instance** (e.g. g4dn.2xlarge) — Phases 1–3 (GAN, synthetic data, expert trajectories)
- **High-core CPU instance** (e.g. c6a.48xlarge, 192 vCPU) — Phase 4 (parallel ensemble training)

```bash
ssh -i <your-key>.pem ubuntu@<gpu-instance-ip>
ssh -i <your-key>.pem ubuntu@<cpu-instance-ip>
source /home/ubuntu/finrl_env/bin/activate   # always activate first (override path with $FINRL_ENV)
```

---

## Key Invariants (Do Not Break)

- **Data contamination guard**: `train_ensemble` filters real data to `< VAL_START_DATE` before combining with synthetic data. Never remove this filter.
- **Old RLlib API stack**: all 42 bots use `.api_stack(enable_rl_module_and_learner=False, ...)`. TQC and custom ModelV2 models are not compatible with the new stack.
- **`num_gpus=0` in training**: intentional — GPU causes CUDA fragmentation across 42 sequential bots.
- **APPO/IMPALA `grad_clip=0.5`**: required; without it VTrace IS ratios grow past iter 300 → NaN explosion.
- **`GROUPS` shell variable**: never use in bash — it expands to the user's primary GID. Use `N_GROUPS` instead.
- **Best-checkpoint restore**: before the final `trainer.save()`, the best mid-run checkpoint is restored to prevent PPO policy forgetting.

---

## Broker Integration

`src/algo_v2/live/broker.py` defines `BaseBroker` (abstract) and `DummyBroker` (JSON-backed simulation).

To connect a real broker (Zerodha / Upstox / AngelOne), implement `BaseBroker` and pass an instance to `execute_live_pipeline()`. See `live/broker.py` for the required interface.

---

## Checkpoint Structure

```
checkpoints/
  phase1_gan/          market_generator.pth, gan_scaling_params.csv
  phase2_synthetic/    indian_stocks_synthetic.csv
  phase3_expert/       expert trajectory JSON files
  phase4_ensemble/     copy of trained_models/ensemble/
  phase5_validation/   val_results.csv, arch_summary.csv, algo_summary.csv
trained_models/
  ensemble/            bot_01_ppo_pure_mlp/ … bot_42_marwil_god_tier/
```

Phase 4 uses skip logic: if `trained_models/ensemble/bot_XX_.../` exists, that bot is not retrained. Delete specific bot directories to force a retrain. `checkpoints/` and `trained_models/` are runtime artifacts and are not tracked in git.

---

## Planned (v3)

See `UNIVERSAL_BOT_PLAN.md` — a per-stock transformer that trades any symbol without retraining. Do not start until v2 test Sharpe > 1.0.

Early prototypes for additional planned features (a mean-variance safety optimizer, SHAP trade explanations, and a FinBERT sentiment scraper) live unintegrated in [`docs/exploratory/`](docs/exploratory/README.md). They are not part of the production pipeline; their extra dependencies are available via the `exploratory` extra (`pip install -e ".[exploratory]"`).
