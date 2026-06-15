import argparse
import gc
import json
import math
import os
import shutil
import time
import traceback
from collections import deque

import numpy as np
import pandas as pd
import ray
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.algorithms.marwil import MARWILConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.tqc import TQCConfig
import boto3
from ray.tune.registry import register_env

from algo_v2.core.models import register_models
from algo_v2.config import (
    make_env_kwargs,
    PRIMARY_DATA_PATH,
    VAL_START_DATE,
    ARCH_MAP,
    SAC_TQC_ARCH_OVERRIDES,
    MARWIL_ARCH_OVERRIDES,
    PROJECT_ROOT,
    CHECKPOINTS_DIR,
    TRAINED_MODELS_DIR,
    EXPERT_DATA_DIR,
    AWS_REGION,
)
from algo_v2.core.env_competitive import CompetitiveTradingEnv
from algo_v2.data.features import grid_fill

os.environ.update(
    {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    }
)

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    _p = argparse.ArgumentParser()
    _p.add_argument("--start-bot", type=int, default=1, help="First bot index to train (inclusive)")
    _p.add_argument("--end-bot", type=int, default=42, help="Last bot index to train (inclusive)")
    _p.add_argument("--iterations", type=int, default=200, help="Training iterations per bot")
    _p.add_argument("--preflight", action="store_true", help="Config validation mode: 1 worker, 1 iter, no saves")
    _p.add_argument(
        "--timing", action="store_true", help="Throughput check: 5 iters × full workers × full batch, pure_mlp only"
    )
    _args, _ = _p.parse_known_args()
    START_BOT = _args.start_bot
    END_BOT = _args.end_bot
    ITERATIONS = _args.iterations
    PREFLIGHT = _args.preflight
    TIMING = _args.timing
    if PREFLIGHT:
        ITERATIONS = 1
        print("[PREFLIGHT MODE] 1 worker · 1 iteration · no saves — catching config/model bugs")
    if TIMING:
        ITERATIONS = 5
        print("[TIMING MODE] 5 iters × full workers × full batch on pure_mlp bots — measuring throughput")

    # Per-algo iteration counts. PPO/APPO/IMPALA need 2× more iterations to converge —
    # they showed mean Sharpe 0.3-0.4 at 200 iters vs SAC/MARWIL at 1.0+.
    # SAC/TQC/MARWIL already perform well; no benefit from extra iterations.
    ALGO_ITERATIONS = {
        "ppo": 200,
        "appo": 400,
        "impala": 400,
        "sac": 200,
        "tqc": 200,
        "marwil": 150,
    }

    # Seconds/iter threshold per algo. Exceeding this on a clean machine means the
    # full 200-iter run will overrun badly under parallel load.
    _TIMING_THRESHOLDS = {"ppo": 60, "appo": 45, "impala": 45, "sac": 60, "tqc": 60, "marwil": 45}

    register_models()

    print(f"Loading enriched data from {PRIMARY_DATA_PATH}...")
    df = pd.read_csv(PRIMARY_DATA_PATH)
    if "sentiment" not in df.columns:
        df["sentiment"] = 0.0
    if "vix" not in df.columns:
        df["vix"] = 0.0

    # Grid-fill real data so every date×tic pair exists
    real_tickers = sorted(df["tic"].unique().tolist())
    real_dates = sorted(df["date"].unique().tolist())
    df = grid_fill(df, real_dates, real_tickers)

    # Wire in GAN synthetic data (60yr) if available — benefits ALL 42 bots
    SYNTH_PATH = os.path.join(CHECKPOINTS_DIR, "phase2_synthetic", "indian_stocks_synthetic.csv")
    if os.path.exists(SYNTH_PATH):
        print("Loading GAN synthetic market data (60yr)...")
        synth_df = pd.read_csv(SYNTH_PATH)

        # Align columns to real data schema — fill any missing with 0
        ref_cols = df.columns.tolist()
        for col in ref_cols:
            if col not in synth_df.columns:
                synth_df[col] = 0.0
        synth_df = synth_df[ref_cols]

        # Only keep tickers that exist in both datasets
        common_tickers = set(df["tic"].unique()) & set(synth_df["tic"].unique())
        synth_df = synth_df[synth_df["tic"].isin(common_tickers)].reset_index(drop=True)

        # Only use real data before the validation period — prevents the bots from
        # training on 2022-2023 data that walk-forward validation tests them on.
        train_real_df = df[df["date"] < VAL_START_DATE]
        train_df = pd.concat([train_real_df, synth_df], ignore_index=True)
        train_df = train_df.sort_values(["tic", "date"]).reset_index(drop=True)
        # Clean NaN indicator values (synthetic data has NaN for first 30 days of each
        # stock's rolling indicators — ffill/bfill within each ticker, then fill 0 for
        # any that remain at the very start of the series)
        train_df = train_df.groupby("tic", group_keys=False).apply(lambda g: g.ffill().bfill()).reset_index(drop=True)
        train_df = train_df.fillna(0.0)
        train_df = train_df.sort_values(["date", "tic"]).reset_index(drop=True)
        nan_count = train_df.isnull().sum().sum()
        print(
            f"  Real: {len(df):,} rows | Synthetic: {len(synth_df):,} rows | Combined: {len(train_df):,} rows | NaNs: {nan_count}"
        )
    else:
        print("Synthetic data not found — training on real data only (GAN still running)")
        train_df = df[df["date"] < VAL_START_DATE].copy()

    stock_dimension = len(train_df.tic.unique())
    env_kwargs = make_env_kwargs(stock_dimension)

    dummy_env = CompetitiveTradingEnv({"num_agents": 5, "env_config": {"df": train_df, **env_kwargs}})
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space

    def env_creator(env_config):
        return CompetitiveTradingEnv(env_config)

    register_env("competitive_trading", env_creator)
    ray.init(
        ignore_reinit_error=True,
        # Each parallel group gets its own Ray cluster and plasma store so they
        # don't share ports or temp files when running on a multi-core instance.
        _temp_dir=f"/tmp/ray_bots_{START_BOT}_{END_BOT}",
        # No num_cpus override — Ray auto-detects the machine's vCPU count.
        # Works correctly on any EC2 instance type without code changes.
        runtime_env={
            "env_vars": {
                "CUDA_VISIBLE_DEVICES": "0",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
        },
    )

    # Put training data into Ray's shared object store once.
    # All 6 env workers read from this single plasma-store slot instead of each
    # receiving a serialised copy — saves ~5× the DataFrame's RAM per bot group.
    train_df_ref = ray.put(train_df)

    os.makedirs(os.path.join(TRAINED_MODELS_DIR, "ensemble"), exist_ok=True)

    # Load tuned hyperparameters if tune_model.py has been run.
    # Keys beginning with '_' are metadata (Sharpe score, trial counts) — skip them.
    def _load_best_params(algo_name: str) -> dict:
        path = os.path.join(PROJECT_ROOT, f"best_params_{algo_name}.json")
        if not os.path.exists(path):
            return {}
        try:
            with open(path) as f:
                raw = json.load(f)
            params = {k: v for k, v in raw.items() if not k.startswith("_")}
            print(f"  [TUNED] Loaded best_params_{algo_name}.json (Sharpe={raw.get('_objective_sharpe', '?'):.4f})")
            return params
        except Exception as e:
            print(f"  [WARN] Could not load best_params_{algo_name}.json: {e}")
            return {}

    _cw = None
    if not PREFLIGHT:
        try:
            _cw = boto3.client("cloudwatch", region_name=AWS_REGION)
            print("CloudWatch metrics enabled (namespace: algo_v2/training)")
        except Exception as _e:
            print(f"[WARN] CloudWatch unavailable — metrics disabled: {_e}")

    ALGORITHMS = {
        "ppo": PPOConfig,
        "appo": APPOConfig,
        "sac": SACConfig,
        "impala": ImpalaConfig,
        "tqc": TQCConfig,
        "marwil": MARWILConfig,
    }
    # ARCH_MAP, SAC_TQC_ARCH_OVERRIDES, MARWIL_ARCH_OVERRIDES imported from config

    print("\n=======================================================")
    print("Starting the 42-Bot Ensemble Factory (6 Algos x 7 Brains)")
    print("=======================================================\n")

    print(f"Bot range: {START_BOT} to {END_BOT}")

    # NaN-safe helper used in the sidecar feed — defined once here, not per-iteration
    safe = lambda v: v if not math.isnan(v) else 0.0

    # (bot_id, status, best_r_max, attempt_used) — printed as a table at the end
    _train_summary: list = []

    _factory_start = time.time()
    bot_id = 1
    for algo_name, ConfigClass in ALGORITHMS.items():
        for arch_name, model_settings in ARCH_MAP.items():
            bot_identifier = f"bot_{bot_id:02d}_{algo_name}_{arch_name}"

            if bot_id < START_BOT or bot_id > END_BOT:
                bot_id += 1
                continue

            # Timing mode: only run the pure_mlp representative for each algorithm
            if TIMING and arch_name != "pure_mlp":
                bot_id += 1
                continue

            print(f"\n---> Assembling {bot_identifier} ...")

            model_path = os.path.join(TRAINED_MODELS_DIR, f"ensemble/{bot_identifier}")
            if os.path.exists(model_path):
                print(f"[{bot_identifier}] Already trained — skipping.")
                _train_summary.append((bot_identifier, "SKIP", None, 0))
                bot_id += 1
                continue

            # SAC/TQC/MARWIL: swap LSTM slots for feedforward temporal-attention equivalents
            if algo_name in ("sac", "tqc"):
                effective_settings = SAC_TQC_ARCH_OVERRIDES.get(arch_name, model_settings)
            elif algo_name == "marwil":
                effective_settings = MARWIL_ARCH_OVERRIDES.get(arch_name, model_settings)
            else:
                effective_settings = model_settings

            _max_attempts = 5
            # Load tuned hyperparameters once per bot (same algo reuses the same file).
            _tuned = {} if PREFLIGHT else _load_best_params(algo_name)

            for attempt in range(_max_attempts):
                try:
                    # PPO: fixed 6 runners — learner-limited (synchronous barrier + heavy CNN).
                    # APPO/IMPALA/SAC/TQC: scale with machine CPUs (workers overlap learner async).
                    #   os.cpu_count()//10 → ~19 on c6a.48xlarge, ~12 on c6a.32xlarge, etc.
                    _async_runners = max(6, (os.cpu_count() or 20) // 10)
                    _num_runners = 1 if PREFLIGHT else (6 if algo_name == "ppo" else _async_runners)
                    config = (
                        ConfigClass()
                        .environment(
                            "competitive_trading",
                            env_config={"num_agents": 5, "env_config": {"df": train_df_ref, **env_kwargs}},
                            observation_space=obs_space,
                            action_space=act_space,
                        )
                        .framework("torch")
                        .resources(num_gpus=0)
                        .env_runners(
                            num_env_runners=_num_runners,
                            batch_mode="truncate_episodes",
                            rollout_fragment_length="auto",
                            sample_timeout_s=300,
                        )
                        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
                    )

                    # APPO / IMPALA use VTrace which calls make_time_major() — requires
                    # train_batch_size must be exactly num_env_runners * rollout_fragment_length.
                    # _async_runners × 400 steps — scales automatically with machine CPU count.
                    if algo_name in ("appo", "impala"):
                        config = config.training(
                            train_batch_size=400 if PREFLIGHT else _async_runners * 400, grad_clip=0.5
                        )
                        config = config.env_runners(rollout_fragment_length=400)
                    if algo_name == "appo":
                        config = config.training(
                            replay_proportion=0.0,
                            **{k: _tuned[k] for k in ("lr", "gamma", "clip_param", "entropy_coeff") if k in _tuned},
                        )
                    if algo_name == "impala":
                        config = config.training(
                            vtrace_clip_rho_threshold=1.0,
                            vtrace_clip_pg_rho_threshold=1.0,
                            grad_clip=0.5,
                            lr=5e-5,
                            **{k: _tuned[k] for k in ("gamma", "entropy_coeff") if k in _tuned},
                        )

                    # PPO: use a large batch (4000 steps = 6 workers × ~667 steps each).
                    # Each worker spans ~1.3 random market windows, so one PPO update sees
                    # ~8 different market eras — prevents locking onto a single regime
                    # (e.g., "always BUY" learned from a bull-market-only batch).
                    if algo_name == "ppo":
                        ppo_batch = _tuned.get("train_batch_size", 6000 if not PREFLIGHT else 1000)
                        config = config.training(
                            train_batch_size=1000 if PREFLIGHT else ppo_batch,
                            **{
                                k: _tuned[k]
                                for k in ("lr", "gamma", "clip_param", "lambda_", "sgd_minibatch_size", "num_sgd_iter")
                                if k in _tuned
                            },
                        )

                    if algo_name == "marwil":
                        expert_data_path = EXPERT_DATA_DIR
                        config = config.offline_data(input_=expert_data_path)
                        config = config.training(
                            beta=_tuned.get("beta", 1.0),
                            train_batch_size=_tuned.get("train_batch_size", 1024),
                            **{k: _tuned[k] for k in ("lr", "gamma") if k in _tuned},
                        )

                    if algo_name in ("sac", "tqc"):
                        # SAC/TQC require SACTorchModel as the outer model; custom models
                        # must go into policy_model_config (not model) or SAC's assertion
                        # isinstance(model, SACTorchModel) in build_sac_model fails.
                        # effective_settings already remaps LSTM slots to temporal-attn.
                        p_config = {}
                        if effective_settings.get("custom_model"):
                            p_config["custom_model"] = effective_settings["custom_model"]
                            if effective_settings.get("custom_model_config"):
                                p_config["custom_model_config"] = effective_settings["custom_model_config"]
                        sac_t_kwargs = {
                            k: _tuned[k]
                            for k in ("lr", "gamma", "tau", "initial_alpha", "n_step", "train_batch_size")
                            if k in _tuned
                        }
                        if algo_name == "tqc":
                            sac_t_kwargs.update(
                                {k: _tuned[k] for k in ("num_quantiles", "top_quantiles_to_drop") if k in _tuned}
                            )
                        if p_config:
                            config = config.training(policy_model_config=p_config, grad_clip=1.0, **sac_t_kwargs)
                        else:
                            config = config.training(grad_clip=1.0, **sac_t_kwargs)
                    else:
                        m_config = {
                            "use_lstm": effective_settings.get("use_lstm", False),
                            "lstm_cell_size": 256 if effective_settings.get("use_lstm") else 0,
                            "lstm_use_prev_action": bool(effective_settings.get("use_lstm")),
                            "lstm_use_prev_reward": bool(effective_settings.get("use_lstm")),
                        }
                        if effective_settings.get("custom_model"):
                            m_config["custom_model"] = effective_settings["custom_model"]
                            if effective_settings.get("custom_model_config"):
                                m_config["custom_model_config"] = effective_settings["custom_model_config"]
                        config = config.training(model=m_config, grad_clip=1.0)

                    # PPO: add entropy bonus to keep exploring past early local optima.
                    # entropy_coeff=0.01 gives a small reward for taking diverse actions,
                    # preventing the policy from locking onto the first decent strategy it finds.
                    # Only PPO — APPO/IMPALA handle exploration differently; SAC/TQC use
                    # their own temperature parameter; MARWIL is imitation so no exploration needed.
                    if algo_name == "ppo":
                        config = config.training(entropy_coeff=_tuned.get("entropy_coeff", 0.01))

                    if algo_name in ("sac", "tqc"):
                        config = config.training(
                            replay_buffer_config={
                                "type": "MultiAgentReplayBuffer",
                                "capacity": _tuned.get("replay_capacity", 100_000),
                            }
                        )

                    trainer = config.build_algo()

                    algo_iterations = ITERATIONS if PREFLIGHT or TIMING else ALGO_ITERATIONS.get(algo_name, ITERATIONS)
                    print(
                        f"[{bot_identifier}] {'Validating config' if PREFLIGHT else f'Training for {algo_iterations} iterations'}..."
                    )

                    best_r_max = -float("inf")
                    best_checkpoint = None
                    final_model_path = os.path.join(TRAINED_MODELS_DIR, f"ensemble/{bot_identifier}")
                    bot_start = time.time()
                    _return_history = deque(maxlen=50)  # rolling r_mean for in-training Sharpe

                    for i in range(algo_iterations):
                        res = trainer.train()
                        env_stats = res.get("env_runners", res)
                        r_mean = env_stats.get("episode_return_mean", env_stats.get("episode_reward_mean", 0))
                        r_max = env_stats.get("episode_return_max", env_stats.get("episode_reward_max", 0))

                        # Save best checkpoint whenever best game improves.
                        # Without this, PPO policy forgetting means we deploy the FINAL
                        # policy (which may have overwritten the best weights found mid-run)
                        # instead of the PEAK policy. bot_02 peaked at +6000 at iter 25
                        # but ended at -1843 at iter 149 — this fix captures the +6000 version.
                        if not PREFLIGHT and not math.isnan(r_max) and r_max > best_r_max:
                            best_r_max = r_max
                            ckpt_dir = os.path.join(TRAINED_MODELS_DIR, f"ensemble/{bot_identifier}_best_ckpt")
                            os.makedirs(ckpt_dir, exist_ok=True)
                            trainer.save(ckpt_dir)
                            best_checkpoint = ckpt_dir

                        # Rolling Sharpe from last 50 episode returns (training proxy).
                        # Uses r_mean (mean reward per episode) as a proxy for returns.
                        # Not a financial Sharpe — useful for spotting flat/degrading bots early.
                        _return_history.append(safe(r_mean))
                        if len(_return_history) >= 10:
                            arr = np.array(_return_history, dtype=np.float64)
                            train_sharpe = float(
                                np.clip(
                                    arr.mean() / (arr.std() + 1e-8) * np.sqrt(252),
                                    0.1,
                                    4.0,
                                )
                            )
                        else:
                            r_m = safe(r_mean)
                            train_sharpe = (
                                max(0.1, min(4.0, r_m / 10_000.0))
                                if r_m >= 0
                                else max(0.1, min(4.0, 1.0 / (1.0 + abs(r_m) / 10_000.0)))
                            )

                        if not PREFLIGHT and _cw and i % 10 == 0:
                            try:
                                _cw.put_metric_data(
                                    Namespace="algo_v2/training",
                                    MetricData=[
                                        {
                                            "MetricName": "Sharpe",
                                            "Value": train_sharpe,
                                            "Unit": "None",
                                            "Dimensions": [
                                                {"Name": "Bot", "Value": bot_identifier},
                                                {"Name": "Algorithm", "Value": algo_name},
                                            ],
                                        },
                                        {
                                            "MetricName": "RewardMean",
                                            "Value": 0.0 if math.isnan(r_mean) else safe(r_mean),
                                            "Unit": "None",
                                            "Dimensions": [
                                                {"Name": "Bot", "Value": bot_identifier},
                                                {"Name": "Algorithm", "Value": algo_name},
                                            ],
                                        },
                                        {
                                            "MetricName": "RewardBest",
                                            "Value": max(0.0, best_r_max) if best_r_max > -float("inf") else 0.0,
                                            "Unit": "None",
                                            "Dimensions": [
                                                {"Name": "Bot", "Value": bot_identifier},
                                                {"Name": "Algorithm", "Value": algo_name},
                                            ],
                                        },
                                    ],
                                )
                            except Exception:
                                pass  # CloudWatch failure never crashes training

                        if i % 10 == 0:
                            elapsed = time.time() - bot_start
                            remaining = (elapsed / (i + 1)) * (algo_iterations - i - 1) if i > 0 else 0.0
                            sharpe_str = f" | sharpe={train_sharpe:.2f}" if len(_return_history) >= 10 else ""
                            print(
                                f"  -> iter {i:3d}: r_mean={r_mean:.2f} | best={best_r_max:.2f}"
                                f"{sharpe_str} | {elapsed / 60:.1f}m elapsed | ~{remaining / 60:.1f}m left"
                            )

                        # Sidecar feed — total_reward and Sharpe only (total_trades/cost removed)
                        print(f"total_reward: {safe(r_max)}")
                        print(f"Sharpe: {train_sharpe}")

                    if not PREFLIGHT:
                        # Restore best checkpoint before saving final model.
                        # If training degraded after the peak, this recovers the best weights.
                        if best_checkpoint and best_r_max > -float("inf"):
                            trainer.restore(best_checkpoint)
                            print(f"  [BEST CKPT] Restored peak policy (best r_max={best_r_max:.2f}) before saving.")
                        trainer.save(final_model_path)
                        if best_checkpoint and os.path.exists(best_checkpoint):
                            shutil.rmtree(best_checkpoint, ignore_errors=True)

                    trainer.stop()
                    bot_elapsed = time.time() - bot_start
                    if TIMING:
                        sec_per_iter = bot_elapsed / ITERATIONS
                        threshold = _TIMING_THRESHOLDS.get(algo_name, 60)
                        est_full_hrs = sec_per_iter * algo_iterations / 3600
                        if sec_per_iter > threshold:
                            print(
                                f"[TIMING_WARN] {bot_identifier}: {sec_per_iter:.1f}s/iter "
                                f"(threshold {threshold}s) — full {algo_iterations}-iter run ~{est_full_hrs:.1f}hrs "
                                f"— consider reducing train_batch_size or PARALLEL_GROUPS"
                            )
                        else:
                            print(
                                f"[TIMING_OK]   {bot_identifier}: {sec_per_iter:.1f}s/iter "
                                f"(threshold {threshold}s) — full run ~{est_full_hrs:.1f}hrs"
                            )
                    else:
                        print(
                            f"[SUCCESS] {bot_identifier} {'config OK' if PREFLIGHT else f'complete. Peak r_max={best_r_max:.2f}'} | took {bot_elapsed / 60:.1f}m"
                        )
                    _train_summary.append((bot_identifier, "OK", best_r_max, attempt + 1))
                    break

                except Exception as e:
                    print(f"[FAILED] {bot_identifier} attempt {attempt + 1}/{_max_attempts}: {str(e)[:120]}")
                    traceback.print_exc()
                    if "trainer" in locals() and trainer:
                        try:
                            trainer.stop()
                        except Exception:
                            pass
                    if attempt < _max_attempts - 1:
                        # Exponential backoff: 30s, 60s, 120s, 240s
                        # Spreads retries so they don't all hit the same OOM pressure spike.
                        _wait = 30 * (2**attempt)
                        print(f"  -> waiting {_wait}s before retry {attempt + 2}...")
                        time.sleep(_wait)
                    else:
                        print(f"[FATAL] {bot_identifier} permanently failed after {_max_attempts} attempts.")
                        _train_summary.append((bot_identifier, "FAIL", None, _max_attempts))

            gc.collect()
            bot_id += 1

    factory_elapsed = time.time() - _factory_start
    print(f"\n{'=' * 70}")
    print(f"TRAINING SUMMARY  (total: {factory_elapsed / 3600:.1f}h)")
    print(f"{'=' * 70}")
    print(f"{'BOT':<38} {'STATUS':<7} {'PEAK r_max':>11} {'ATTEMPTS':>8}")
    print("-" * 70)
    for _bid, _status, _rmax, _att in _train_summary:
        _r = f"{_rmax:.1f}" if _rmax is not None else "—"
        print(f"{_bid:<38} {_status:<7} {_r:>11} {_att:>8}")
    _ok = sum(1 for _, s, _, _ in _train_summary if s == "OK")
    _fail = sum(1 for _, s, _, _ in _train_summary if s == "FAIL")
    _skip = sum(1 for _, s, _, _ in _train_summary if s == "SKIP")
    print(f"\n  {_ok} succeeded  |  {_fail} failed  |  {_skip} skipped")
    print(f"{'=' * 70}\n")

    ray.shutdown()


if __name__ == "__main__":
    main()
