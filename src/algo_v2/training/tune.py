"""
tune_model.py — Deep hyperparameter tuning for all 6 algorithms.

Uses Optuna TPE (Bayesian optimisation) to find the best hyperparameters
for each algorithm. Objective = mean episode reward on 2020-2021 held-out
data (stays within the pre-2022 training window — no val/test leakage).

Usage:
    python tune_model.py                       # tune all 6 algos sequentially
    python tune_model.py --algo ppo            # tune one algo
    python tune_model.py --algo sac --trials 50 --iters 40

Outputs:  best_params_{algo}.json  (one file per algo)
Usage:    train_ensemble.py auto-loads these files when present.
          Run a fresh full training run after tuning to apply them.
"""

import argparse
import json
import logging
import math
import os
import warnings
from collections import deque

import numpy as np
import optuna
import pandas as pd
import ray
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.algorithms.marwil import MARWILConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.tqc import TQCConfig
from ray.tune.registry import register_env

from algo_v2.core.models import register_models
from algo_v2.config import make_env_kwargs, PRIMARY_DATA_PATH, PROJECT_ROOT, CHECKPOINTS_DIR, EXPERT_DATA_DIR
from algo_v2.core.env_competitive import CompetitiveTradingEnv

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
os.environ.update(
    {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default=None, help="One of: ppo appo impala sac tqc marwil (default: all)")
    p.add_argument("--trials", type=int, default=30, help="Optuna trials per algorithm (default 30)")
    p.add_argument("--iters", type=int, default=40, help="Training iterations per trial (default 40)")
    return p.parse_args()


# Tuning depth — overwritten from CLI in __main__.
TUNE_ITERS = 40
TUNE_TRIALS = 30

# ── Data splits ───────────────────────────────────────────────────────────────
# Both splits stay below VAL_START_DATE (2022-01-01) — no leakage.
TUNE_TRAIN_END = "2020-01-01"  # tune train:  2010–2019 (+ synthetic)
TUNE_VAL_START = "2020-01-01"  # tune eval:   2020–2021
TUNE_VAL_END = "2022-01-01"  # matches VAL_START_DATE in config.py

ALGORITHMS = {
    "ppo": PPOConfig,
    "appo": APPOConfig,
    "impala": ImpalaConfig,
    "sac": SACConfig,
    "tqc": TQCConfig,
    "marwil": MARWILConfig,
}


# ── Model registration ────────────────────────────────────────────────────────
register_models()


# ── Data loading ──────────────────────────────────────────────────────────────


def _load_data():
    logger.info("Loading %s ...", PRIMARY_DATA_PATH)
    df = pd.read_csv(PRIMARY_DATA_PATH)
    for col in ("sentiment", "vix"):
        if col not in df.columns:
            df[col] = 0.0

    tickers = sorted(df["tic"].unique())
    dates = sorted(df["date"].unique())
    grid = pd.MultiIndex.from_product([dates, tickers], names=["date", "tic"]).to_frame(index=False)
    df = pd.merge(grid, df, on=["date", "tic"], how="left")
    df = df.sort_values(["tic", "date"]).reset_index(drop=True)
    df = df.groupby("tic").ffill().bfill().reset_index(drop=True)
    df["tic"] = grid.sort_values(["tic", "date"])["tic"].values
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)

    # Add synthetic data to tune-train window if available
    synth_path = os.path.join(CHECKPOINTS_DIR, "phase2_synthetic", "indian_stocks_synthetic.csv")
    real_train = df[df["date"] < TUNE_TRAIN_END].copy()
    if os.path.exists(synth_path):
        synth = pd.read_csv(synth_path)
        for col in df.columns:
            if col not in synth.columns:
                synth[col] = 0.0
        synth = synth[df.columns]
        common = set(df["tic"].unique()) & set(synth["tic"].unique())
        synth = synth[synth["tic"].isin(common)].reset_index(drop=True)
        tune_train = pd.concat([real_train, synth], ignore_index=True)
        tune_train = (
            tune_train.groupby("tic", group_keys=False)
            .apply(lambda g: g.ffill().bfill())
            .reset_index(drop=True)
            .fillna(0.0)
        )
        tune_train = tune_train.sort_values(["date", "tic"]).reset_index(drop=True)
        logger.info("  Tune-train: real %d + synth %d = %d rows", len(real_train), len(synth), len(tune_train))
    else:
        tune_train = real_train
        logger.info("  Tune-train: %d rows (no synthetic)", len(tune_train))

    tune_eval = df[(df["date"] >= TUNE_VAL_START) & (df["date"] < TUNE_VAL_END)].copy()
    logger.info("  Tune-eval:  %d rows (%s → %s)", len(tune_eval), TUNE_VAL_START, TUNE_VAL_END)
    return tune_train, tune_eval


# ── Hyperparameter search spaces ──────────────────────────────────────────────


def _sample_params(trial: optuna.Trial, algo_name: str) -> dict:
    if algo_name == "ppo":
        return {
            "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
            "train_batch_size": trial.suggest_categorical("train_batch_size", [4000, 8000, 16000]),
            "sgd_minibatch_size": trial.suggest_categorical("sgd_minibatch_size", [128, 256, 512]),
            "num_sgd_iter": trial.suggest_int("num_sgd_iter", 5, 20),
            "gamma": trial.suggest_float("gamma", 0.95, 0.999),
            "clip_param": trial.suggest_float("clip_param", 0.1, 0.4),
            "lambda_": trial.suggest_float("lambda_", 0.9, 1.0),
            "entropy_coeff": trial.suggest_float("entropy_coeff", 0.0, 0.03),
        }
    if algo_name == "appo":
        return {
            "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
            "gamma": trial.suggest_float("gamma", 0.95, 0.999),
            "clip_param": trial.suggest_float("clip_param", 0.1, 0.4),
            "entropy_coeff": trial.suggest_float("entropy_coeff", 0.0, 0.02),
        }
    if algo_name == "impala":
        return {
            "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
            "gamma": trial.suggest_float("gamma", 0.95, 0.999),
            "entropy_coeff": trial.suggest_float("entropy_coeff", 0.0, 0.02),
        }
    if algo_name in ("sac", "tqc"):
        params = {
            "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
            "gamma": trial.suggest_float("gamma", 0.95, 0.999),
            "tau": trial.suggest_float("tau", 0.001, 0.05),
            "initial_alpha": trial.suggest_float("initial_alpha", 0.1, 1.0),
            "replay_capacity": trial.suggest_categorical("replay_capacity", [50_000, 100_000, 200_000]),
            "n_step": trial.suggest_int("n_step", 1, 3),
            "train_batch_size": trial.suggest_categorical("train_batch_size", [256, 512, 1024]),
        }
        if algo_name == "tqc":
            params["num_quantiles"] = trial.suggest_categorical("num_quantiles", [25, 50])
            params["top_quantiles_to_drop"] = trial.suggest_int("top_quantiles_to_drop", 1, 5)
        return params
    if algo_name == "marwil":
        return {
            "beta": trial.suggest_float("beta", 0.0, 1.0),
            "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
            "train_batch_size": trial.suggest_categorical("train_batch_size", [512, 1024, 2048]),
            "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        }
    raise ValueError(f"Unknown algo: {algo_name}")


# ── Config builder ────────────────────────────────────────────────────────────


def _build_config(algo_name, ConfigClass, params, train_ref, env_kwargs, obs_space, act_space):
    cfg = (
        ConfigClass()
        .environment(
            "competitive_trading_tune",
            env_config={"num_agents": 5, "env_config": {"df": train_ref, **env_kwargs}},
            observation_space=obs_space,
            action_space=act_space,
        )
        .framework("torch")
        .resources(num_gpus=0)
        .env_runners(
            num_env_runners=2,  # fewer workers during tuning — faster iteration
            batch_mode="truncate_episodes",
            rollout_fragment_length="auto",
            sample_timeout_s=300,
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )

    # Algorithm-specific constraints
    if algo_name in ("appo", "impala"):
        cfg = cfg.training(
            train_batch_size=800, lr=params["lr"], gamma=params["gamma"], entropy_coeff=params["entropy_coeff"]
        )
        cfg = cfg.env_runners(rollout_fragment_length=400)
        if algo_name == "appo":
            cfg = cfg.training(replay_proportion=0.0, clip_param=params.get("clip_param", 0.3))

    elif algo_name == "ppo":
        cfg = cfg.training(
            lr=params["lr"],
            train_batch_size=params["train_batch_size"],
            sgd_minibatch_size=min(params["sgd_minibatch_size"], params["train_batch_size"]),
            num_sgd_iter=params["num_sgd_iter"],
            gamma=params["gamma"],
            clip_param=params["clip_param"],
            lambda_=params["lambda_"],
            entropy_coeff=params["entropy_coeff"],
            grad_clip=1.0,
            model={"use_lstm": False},  # pure_mlp for tuning speed
        )

    elif algo_name in ("sac", "tqc"):
        replay_cfg = {
            "type": "MultiAgentReplayBuffer",
            "capacity": params["replay_capacity"],
        }
        t_kwargs = dict(
            lr=params["lr"],
            gamma=params["gamma"],
            tau=params["tau"],
            initial_alpha=params["initial_alpha"],
            n_step=params["n_step"],
            train_batch_size=params["train_batch_size"],
            replay_buffer_config=replay_cfg,
            grad_clip=1.0,
        )
        if algo_name == "tqc":
            t_kwargs["num_quantiles"] = params["num_quantiles"]
            t_kwargs["top_quantiles_to_drop"] = params["top_quantiles_to_drop"]
        cfg = cfg.training(**t_kwargs)

    elif algo_name == "marwil":
        expert_path = EXPERT_DATA_DIR
        cfg = cfg.offline_data(input_=expert_path).training(
            beta=params["beta"],
            lr=params["lr"],
            train_batch_size=params["train_batch_size"],
            gamma=params["gamma"],
            grad_clip=1.0,
        )

    return cfg


# ── Evaluation helper ─────────────────────────────────────────────────────────


def _eval_on_val(trainer, eval_ref, env_kwargs, obs_space, act_space):
    """Run one episode on the eval data, return Sharpe proxy (mean/std of rewards)."""
    try:
        env = CompetitiveTradingEnv(
            {
                "num_agents": 5,
                "env_config": {
                    "df": ray.get(eval_ref) if isinstance(eval_ref, ray.ObjectRef) else eval_ref,
                    **env_kwargs,
                },
            }
        )
        obs, _ = env.reset()
        policy_map = {aid: "default_policy" for aid in obs}
        rewards_all = []
        done = False
        truncated = False
        step = 0
        while not (done or truncated) and step < 5000:
            actions = {}
            for aid, o in obs.items():
                policy = trainer.get_policy(policy_map.get(aid, "default_policy"))
                if policy is None:
                    actions[aid] = env.action_space.sample()
                else:
                    actions[aid], _, _ = policy.compute_single_action(o, explore=False)
            obs, rews, dones, truncateds, _ = env.step(actions)
            rewards_all.append(sum(rews.values()))
            done = dones.get("__all__", False)
            truncated = truncateds.get("__all__", False)
            step += 1
        if len(rewards_all) < 5:
            return -999.0
        arr = np.array(rewards_all, dtype=np.float64)
        # Annualised Sharpe proxy: mean/std × sqrt(252)
        return float(arr.mean() / (arr.std() + 1e-8) * math.sqrt(252))
    except Exception:
        return -999.0


# ── Optuna objective ──────────────────────────────────────────────────────────


def make_objective(algo_name, ConfigClass, train_ref, eval_ref, env_kwargs, obs_space, act_space):
    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, algo_name)
        trainer = None
        try:
            cfg = _build_config(algo_name, ConfigClass, params, train_ref, env_kwargs, obs_space, act_space)
            trainer = cfg.build_algo()

            reward_history = deque(maxlen=10)
            for i in range(TUNE_ITERS):
                result = trainer.train()
                env_s = result.get("env_runners", result)
                r_mean = env_s.get("episode_return_mean", env_s.get("episode_reward_mean", -999))
                r_mean = r_mean if not math.isnan(r_mean) else -999
                reward_history.append(r_mean)

                # Prune clearly losing trials at checkpoints
                if i in (9, 19, 29):
                    intermediate = float(np.mean(reward_history))
                    trial.report(intermediate, i)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            # Final objective: eval Sharpe on held-out 2020–2021 data
            sharpe = _eval_on_val(trainer, eval_ref, env_kwargs, obs_space, act_space)
            # Fallback to training proxy if eval failed
            if sharpe == -999.0 and reward_history:
                sharpe = float(np.mean(reward_history)) / 10_000.0
            return sharpe

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.warning("Trial %d failed: %s", trial.number, e)
            return -999.0
        finally:
            if trainer is not None:
                try:
                    trainer.stop()
                except Exception:
                    pass

    return objective


# ── Per-algo tuning ───────────────────────────────────────────────────────────


def tune_algo(algo_name, ConfigClass, tune_train, tune_eval, env_kwargs):
    print(f"\n{'=' * 60}")
    print(f"Tuning {algo_name.upper()}  ({TUNE_TRIALS} trials × {TUNE_ITERS} iters)")
    print(f"{'=' * 60}")

    if algo_name == "marwil":
        expert_path = EXPERT_DATA_DIR
        if not os.path.exists(expert_path) or not os.listdir(expert_path):
            logger.warning("[SKIP] expert_data/ not found — MARWIL tuning requires Phase 3 output.")
            return None

    # Build dummy env for obs/act spaces
    dummy = CompetitiveTradingEnv({"num_agents": 5, "env_config": {"df": tune_train.head(2000), **env_kwargs}})
    obs_space = dummy.observation_space
    act_space = dummy.action_space
    del dummy

    # Put data into Ray plasma store (shared across all trials in this algo's run)
    train_ref = ray.put(tune_train)
    eval_ref = ray.put(tune_eval)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=9)
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", pruner=pruner, sampler=sampler, study_name=f"tune_{algo_name}")
    study.optimize(
        make_objective(algo_name, ConfigClass, train_ref, eval_ref, env_kwargs, obs_space, act_space),
        n_trials=TUNE_TRIALS,
        show_progress_bar=False,
        gc_after_trial=True,
    )

    best = study.best_params
    best["_objective_sharpe"] = study.best_value
    best["_completed_trials"] = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    best["_pruned_trials"] = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

    out_path = os.path.join(PROJECT_ROOT, f"best_params_{algo_name}.json")
    with open(out_path, "w") as f:
        json.dump(best, f, indent=2)

    print(f"\n  Best Sharpe: {study.best_value:.4f}")
    print(f"  Best params: {best}")
    print(f"  Saved → {out_path}")
    print(f"  Completed: {best['_completed_trials']}  Pruned: {best['_pruned_trials']}")

    del train_ref, eval_ref
    return best


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from algo_v2.platform.logging_config import configure_logging

    configure_logging("tune_model")

    ARGS = parse_args()
    TUNE_ITERS = ARGS.iters
    TUNE_TRIALS = ARGS.trials

    tune_train, tune_eval = _load_data()

    stock_dim = len(tune_train["tic"].unique())
    env_kwargs = make_env_kwargs(stock_dim)

    register_env("competitive_trading_tune", lambda cfg: CompetitiveTradingEnv(cfg))

    ray.init(
        ignore_reinit_error=True,
        _temp_dir="/tmp/ray_tune",
        num_cpus=7,
        runtime_env={
            "env_vars": {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "CUDA_VISIBLE_DEVICES": "",
            }
        },
    )

    algos_to_tune = {ARGS.algo: ALGORITHMS[ARGS.algo]} if ARGS.algo else ALGORITHMS

    if ARGS.algo and ARGS.algo not in ALGORITHMS:
        print(f"Unknown algo '{ARGS.algo}'. Choose from: {list(ALGORITHMS)}")
        ray.shutdown()
        raise SystemExit(1)

    all_results = {}
    for algo_name, ConfigClass in algos_to_tune.items():
        result = tune_algo(algo_name, ConfigClass, tune_train, tune_eval, env_kwargs)
        if result:
            all_results[algo_name] = result

    ray.shutdown()

    print("\n" + "=" * 60)
    print("TUNING COMPLETE — Summary")
    print("=" * 60)
    for algo, params in all_results.items():
        print(
            f"  {algo:8s}  Sharpe={params['_objective_sharpe']:.4f}  "
            f"Trials={params['_completed_trials']}+{params['_pruned_trials']} pruned"
        )
    print("\nRetrain with best params:")
    print("  PARALLEL_GROUPS=21 nohup bash phase6_master.sh > /dev/null 2>&1 &")
    print("(train_ensemble.py auto-loads best_params_{algo}.json when present)")
