#!/usr/bin/env python3
"""
Walk-forward validation for the 42-bot ensemble.

Validation : 2022-01-01 -> 2024-01-01  (sequential 500-day windows, 30-day purge)
Test       : 2024-01-01 -> end of data  (run once, final score — touch only once)

Results saved to checkpoints/phase5_validation/ (consistent with phases 1-4).
Re-run is blocked unless --force is passed (same guard as test mode).

Usage:
    python -m algo_v2.evaluation.validate           # score all bots on 2022-2023 val data
    python -m algo_v2.evaluation.validate --test    # score all bots on 2024 test data (once)
    python -m algo_v2.evaluation.validate --force   # re-run validation even if results exist
"""

import argparse
import gc
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
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
from algo_v2.config import (
    PRIMARY_DATA_PATH,
    VAL_START_DATE,
    VAL_END_DATE,
    TEST_START_DATE,
    make_env_kwargs,
    ARCH_MAP,
    SAC_TQC_ARCH_OVERRIDES,
    MARWIL_ARCH_OVERRIDES,
    CHECKPOINTS_DIR,
    TRAINED_MODELS_DIR,
    EXPERT_DATA_DIR,
)
from algo_v2.core.env_single import StockTradingEnv, RiskAwareTradingEnv

PURGE_DAYS = 30  # skip first N days of each window (SMA30 warm-up period)
VAL_WINDOW = 490  # trading days per validation window (2022-2023 has ~493 trading days)
TOP_N = 10  # bots to carry through to the test period
ENSEMBLE_DIR = os.path.join(TRAINED_MODELS_DIR, "ensemble")
CKPT_DIR = os.path.join(CHECKPOINTS_DIR, "phase5_validation")
VAL_CSV = os.path.join(CKPT_DIR, "val_results.csv")
TEST_CSV = os.path.join(CKPT_DIR, "test_results.csv")
ARCH_CSV = os.path.join(CKPT_DIR, "arch_summary.csv")
ALGO_CSV = os.path.join(CKPT_DIR, "algo_summary.csv")

register_models()
register_env("risk_aware_trading", lambda cfg: RiskAwareTradingEnv(**cfg))

ALGO_MAP = {
    "ppo": PPOConfig,
    "appo": APPOConfig,
    "sac": SACConfig,
    "impala": ImpalaConfig,
    "tqc": TQCConfig,
    "marwil": MARWILConfig,
}
# ARCH_MAP, SAC_TQC_ARCH_OVERRIDES, MARWIL_ARCH_OVERRIDES imported from config


def _save_incremental(csv_path, result):
    """Append one bot result to csv_path, replacing any prior row for that bot_id."""
    row_df = pd.DataFrame([result])
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        existing = existing[existing["bot_id"] != result["bot_id"]]
        row_df = pd.concat([existing, row_df], ignore_index=True)
    row_df.to_csv(csv_path, index=False)


def _resolve_checkpoints(ensemble_dir):
    """Map each bot to its checkpoint path, using _best_ckpt as fallback.

    Some bots have only _best_ckpt because trainer.save() was interrupted
    (kill during final write, or disk I/O error). The _best_ckpt is the peak
    policy saved mid-training and is equally valid for validation.
    """
    all_dirs = {
        d for d in os.listdir(ensemble_dir) if os.path.isdir(os.path.join(ensemble_dir, d)) and d.startswith("bot_")
    }
    final_ids = {d for d in all_dirs if not d.endswith("_best_ckpt")}
    best_ckpt_ids = {
        d[: -len("_best_ckpt")]
        for d in all_dirs
        if d.endswith("_best_ckpt") and d[: -len("_best_ckpt")] not in final_ids
    }
    resolved = []
    for bot_id in sorted(final_ids):
        resolved.append((bot_id, os.path.join(ensemble_dir, bot_id)))
    for bot_id in sorted(best_ckpt_ids):
        path = os.path.join(ensemble_dir, bot_id + "_best_ckpt")
        print(f"  [FALLBACK] {bot_id} — final save was interrupted, using _best_ckpt")
        resolved.append((bot_id, path))
    return sorted(resolved, key=lambda x: x[0])


def _build_trainer(algo_name, arch_name, window_df, env_kwargs):
    ConfigClass = ALGO_MAP[algo_name]
    if algo_name in ("sac", "tqc"):
        model_settings = SAC_TQC_ARCH_OVERRIDES.get(arch_name, ARCH_MAP[arch_name])
    elif algo_name == "marwil":
        model_settings = MARWIL_ARCH_OVERRIDES.get(arch_name, ARCH_MAP[arch_name])
    else:
        model_settings = ARCH_MAP[arch_name]

    config = (
        ConfigClass()
        .environment("risk_aware_trading", env_config={"df": window_df, **env_kwargs})
        .framework("torch")
        .resources(num_gpus=0)
        .env_runners(num_env_runners=0)
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    )

    m_config = {
        "use_lstm": model_settings.get("use_lstm", False),
        "lstm_cell_size": 256 if model_settings.get("use_lstm") else 0,
        "lstm_use_prev_action": bool(model_settings.get("use_lstm")),
        "lstm_use_prev_reward": bool(model_settings.get("use_lstm")),
    }
    if model_settings.get("custom_model"):
        if algo_name in ("sac", "tqc"):
            # SAC/TQC: custom model must go in policy_model_config, not top-level model.
            # Putting it in model= makes SAC use the CNN as the outer SACTorchModel,
            # failing the isinstance check at policy build time.
            pmc = {"custom_model": model_settings["custom_model"]}
            if model_settings.get("custom_model_config"):
                pmc["custom_model_config"] = model_settings["custom_model_config"]
            config.training(policy_model_config=pmc)
        else:
            m_config["custom_model"] = model_settings["custom_model"]
            if model_settings.get("custom_model_config"):
                m_config["custom_model_config"] = model_settings["custom_model_config"]

    config.training(model=m_config)

    if algo_name in ("sac", "tqc"):
        config.training(replay_buffer_config={"type": "MultiAgentReplayBuffer", "capacity": 50_000})
    if algo_name == "marwil":
        config.offline_data(input_=EXPERT_DATA_DIR)
        config.training(beta=1.0)

    return config.build_algo()


def _run_episode(trainer, window_df, env_kwargs, use_lstm):
    env = StockTradingEnv(df=window_df, **env_kwargs)
    obs, _ = env.reset()
    state = trainer.get_policy().get_initial_state() if use_lstm else []
    prev_action = np.zeros(env.action_space.shape, dtype=np.float32)
    prev_reward = 0.0
    done = False

    while not done:
        if state:
            action, state, _ = trainer.compute_single_action(
                obs, state=state, prev_action=prev_action, prev_reward=prev_reward
            )
        else:
            action = trainer.compute_single_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        prev_action = action
        prev_reward = float(reward)
        done = terminated or truncated

    values = np.array(env.asset_memory, dtype=np.float64)
    final_val = float(values[-1])
    roi = (final_val - env.initial_amount) / env.initial_amount * 100.0

    peak = np.maximum.accumulate(values)
    dd = (peak - values) / np.maximum(peak, 1.0)
    max_dd = float(dd.max()) * 100.0

    rets = np.diff(values) / np.maximum(np.abs(values[:-1]), 1.0)
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252)) if len(rets) > 1 else 0.0

    return {"roi_pct": roi, "final_value": final_val, "max_drawdown_pct": max_dd, "sharpe": sharpe}


def score_bot(bot_id, checkpoint_path, windows, env_kwargs):
    # Parse algo and arch from bot identifier: bot_XX_<algo>_<arch_may_have_underscores>
    parts = bot_id.split("_")
    algo_name = parts[2]
    arch_name = "_".join(parts[3:])

    if algo_name not in ALGO_MAP:
        print(f"  [SKIP] {bot_id}: unknown algo '{algo_name}'")
        return None
    if arch_name not in ARCH_MAP:
        print(f"  [SKIP] {bot_id}: unknown arch '{arch_name}'")
        return None

    use_lstm = ARCH_MAP[arch_name].get("use_lstm", False)
    window_results = []

    for w_idx, window_df in enumerate(windows):
        try:
            trainer = _build_trainer(algo_name, arch_name, window_df, env_kwargs)
            trainer.restore(checkpoint_path)
            result = _run_episode(trainer, window_df, env_kwargs, use_lstm)
            window_results.append(result)
            trainer.stop()
            del trainer
            gc.collect()
        except Exception as e:
            print(f"    [WARN] window {w_idx} failed: {str(e)[:100]}")
            traceback.print_exc()

    if not window_results:
        return None

    return {
        "bot_id": bot_id,
        "algo": algo_name,
        "arch": arch_name,
        "mean_roi_pct": float(np.mean([r["roi_pct"] for r in window_results])),
        "mean_sharpe": float(np.mean([r["sharpe"] for r in window_results])),
        "mean_max_dd_pct": float(np.mean([r["max_drawdown_pct"] for r in window_results])),
        "n_windows": len(window_results),
    }


def _score_bot_subprocess(args):
    """Run one bot evaluation in an isolated subprocess with its own Ray instance.

    Using ProcessPoolExecutor so 40 bots run in parallel on the c6a.48xlarge,
    each claiming 1 CPU — no nested-actor conflicts, no shared ModelCatalog state.
    """
    bot_id, checkpoint_path, windows, env_kwargs = args
    import warnings

    warnings.filterwarnings("ignore")

    import ray as _ray
    from ray.tune.registry import register_env as _re

    _ray.init(
        ignore_reinit_error=True,
        num_cpus=1,
        _temp_dir=f"/tmp/ray_val_{bot_id}",
        runtime_env={
            "env_vars": {
                "CUDA_VISIBLE_DEVICES": "0",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "MALLOC_ARENA_MAX": "2",
            }
        },
    )
    register_models()
    _re("risk_aware_trading", lambda cfg: RiskAwareTradingEnv(**cfg))

    try:
        return score_bot(bot_id, checkpoint_path, windows, env_kwargs)
    finally:
        _ray.shutdown()


def build_windows(df, start_date, end_date):
    """
    Slice df into sequential VAL_WINDOW-day chunks between start_date and end_date.
    First PURGE_DAYS of each chunk are dropped (SMA30 warm-up contamination from
    the preceding training period bleeds into rolling indicators for ~30 bars).
    """
    period_df = df[(df["date"] >= start_date) & (df["date"] < end_date)].copy()
    all_dates = sorted(period_df["date"].unique())
    tickers = sorted(period_df["tic"].unique())
    windows = []

    idx = 0
    while idx + VAL_WINDOW <= len(all_dates):
        w_dates = all_dates[idx : idx + VAL_WINDOW]
        purged_dates = w_dates[PURGE_DAYS:]  # drop warm-up days
        window_df = period_df[period_df["date"].isin(set(purged_dates))].copy()

        # Ensure every date × ticker combination is present (forward-fill gaps)
        grid = pd.MultiIndex.from_product([purged_dates, tickers], names=["date", "tic"]).to_frame(index=False)
        window_df = pd.merge(grid, window_df, on=["date", "tic"], how="left")
        window_df = window_df.groupby("tic", group_keys=False).apply(lambda g: g.ffill().bfill()).reset_index(drop=True)
        window_df = window_df.sort_values(["date", "tic"]).reset_index(drop=True)
        window_df = window_df.fillna(0.0)

        windows.append(window_df)
        idx += VAL_WINDOW  # non-overlapping windows

    return windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run all bots on 2024 test data (once only, final score)")
    parser.add_argument(
        "--force", action="store_true", help="Allow overwriting an existing test_results.csv (use with caution)"
    )
    parser.add_argument(
        "--bots",
        type=str,
        default=None,
        help="Comma-separated bot IDs to re-score (e.g. bot_16,bot_17). "
        "Merges results into existing VAL_CSV; bypasses exists guard.",
    )
    args = parser.parse_args()
    bot_filter = set(b.strip() for b in args.bots.split(",")) if args.bots else None

    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "env_vars": {
                "CUDA_VISIBLE_DEVICES": "0",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "MALLOC_ARENA_MAX": "2",
            }
        },
    )

    df = pd.read_csv(PRIMARY_DATA_PATH)
    if "sentiment" not in df.columns:
        df["sentiment"] = 0.0
    if "vix" not in df.columns:
        df["vix"] = 0.0

    stock_dim = len(df["tic"].unique())
    env_kwargs = make_env_kwargs(stock_dim)

    # ── TEST MODE ─────────────────────────────────────────────────────────────
    if args.test and os.path.exists(TEST_CSV) and not args.force:
        print(f"\n[GUARD] {TEST_CSV} already exists.")
        print("The test set is a one-shot score — running again would contaminate your result.")
        print("Pass --force to override (only if you truly intend to re-run the final test).")
        ray.shutdown()
        return

    if args.test:
        print(f"\n{'=' * 60}")
        print("TEST MODE — 2024 data (touch once, final score)")
        print("Running all bots — MetaController handles selection in live trading")
        print(f"{'=' * 60}\n")

        test_windows = build_windows(df, TEST_START_DATE, "2099-01-01")
        if not test_windows:
            print(f"No data found after TEST_START_DATE ({TEST_START_DATE}).")
            ray.shutdown()
            return

        print(f"Test windows: {len(test_windows)} × {VAL_WINDOW - PURGE_DAYS} effective days\n")

        os.makedirs(CKPT_DIR, exist_ok=True)
        bots = _resolve_checkpoints(ENSEMBLE_DIR)

        n_parallel = min(len(bots), os.cpu_count() or 8)
        print(f"Parallel workers: {n_parallel}\n")
        test_rows = []
        sub_args = [(bid, cp, test_windows, env_kwargs) for bid, cp in bots]
        with ProcessPoolExecutor(
            max_workers=n_parallel, mp_context=__import__("multiprocessing").get_context("spawn")
        ) as pool:
            future_to_bot = {pool.submit(_score_bot_subprocess, a): a[0] for a in sub_args}
            for future in as_completed(future_to_bot):
                bot_id = future_to_bot[future]
                try:
                    result = future.result()
                    if result:
                        test_rows.append(result)
                        print(
                            f"  Done {bot_id}: ROI={result['mean_roi_pct']:.2f}%  "
                            f"Sharpe={result['mean_sharpe']:.2f}  "
                            f"MaxDD={result['mean_max_dd_pct']:.1f}%"
                        )
                except Exception as e:
                    print(f"  [FAILED] {bot_id}: {str(e)[:120]}")

        results_df = pd.DataFrame(test_rows).sort_values("mean_roi_pct", ascending=False)
        results_df.to_csv(TEST_CSV, index=False)

        print(f"\n{'=' * 60}")
        print("TEST RESULTS (2024)")
        print(f"{'=' * 60}")
        print(results_df[["bot_id", "mean_roi_pct", "mean_sharpe", "mean_max_dd_pct"]].to_string(index=False))
        print(f"\nSaved -> {TEST_CSV}")
        print(f"Checkpoint dir -> {CKPT_DIR}/")

    # ── VALIDATION MODE ───────────────────────────────────────────────────────
    else:
        if os.path.exists(VAL_CSV) and not args.force and bot_filter is None:
            print(f"\n[SKIP] {VAL_CSV} already exists — Phase 5 validation already complete.")
            print("Pass --force to re-run, or --bots bot_XX,... to re-score specific bots.")
            ray.shutdown()
            return

        val_windows = build_windows(df, VAL_START_DATE, VAL_END_DATE)
        eff_days = VAL_WINDOW - PURGE_DAYS

        print(f"\n{'=' * 60}")
        print(f"VALIDATION — {VAL_START_DATE} to {VAL_END_DATE}")
        if bot_filter:
            print(f"Re-scoring {len(bot_filter)} bots: {', '.join(sorted(bot_filter))}")
        print(
            f"Windows: {len(val_windows)} × {VAL_WINDOW} days ({PURGE_DAYS}-day purge = {eff_days} effective days each)"
        )
        print(f"{'=' * 60}\n")

        bots = _resolve_checkpoints(ENSEMBLE_DIR)
        if bot_filter:
            bots = [(bid, cp) for bid, cp in bots if bid in bot_filter]
        elif os.path.exists(VAL_CSV) and not args.force:
            # Crash recovery: skip bots already in partial CSV
            already_done = set(pd.read_csv(VAL_CSV)["bot_id"].tolist())
            skipped = [bid for bid, _ in bots if bid in already_done]
            bots = [(bid, cp) for bid, cp in bots if bid not in already_done]
            if skipped:
                print(f"  [RESUME] Skipping {len(skipped)} already-scored bots: {', '.join(skipped)}")
        print(f"Bots to score: {len(bots)}\n")

        os.makedirs(CKPT_DIR, exist_ok=True)

        # Run all bots in parallel — each subprocess owns 1 CPU + its own Ray instance.
        # On c6a.48xlarge (192 vCPU) this fires all bots simultaneously instead of sequentially.
        n_parallel = min(len(bots), os.cpu_count() or 8)
        print(f"Parallel workers: {n_parallel}\n")
        rows = []
        sub_args = [(bid, cp, val_windows, env_kwargs) for bid, cp in bots]
        with ProcessPoolExecutor(
            max_workers=n_parallel, mp_context=__import__("multiprocessing").get_context("spawn")
        ) as pool:
            future_to_bot = {pool.submit(_score_bot_subprocess, a): a[0] for a in sub_args}
            for future in as_completed(future_to_bot):
                bot_id = future_to_bot[future]
                try:
                    result = future.result()
                    if result:
                        rows.append(result)
                        print(
                            f"  Done {bot_id}: ROI={result['mean_roi_pct']:.2f}%  "
                            f"Sharpe={result['mean_sharpe']:.2f}  "
                            f"MaxDD={result['mean_max_dd_pct']:.1f}%"
                        )
                        _save_incremental(VAL_CSV, result)
                    else:
                        print(f"  [NO RESULT] {bot_id}")
                except Exception as e:
                    print(f"  [FAILED] {bot_id}: {str(e)[:120]}")
                    traceback.print_exc()

        if not rows:
            print("No bots scored — check that trained_models/ensemble/ contains checkpoints.")
            ray.shutdown()
            return

        # Final sorted CSV from the incrementally built file
        results_df = pd.read_csv(VAL_CSV).sort_values("mean_roi_pct", ascending=False).reset_index(drop=True)
        results_df.to_csv(VAL_CSV, index=False)

        # Architecture summary — key input for v3 architecture decision
        arch_summary = (
            results_df.groupby("arch")["mean_sharpe"]
            .agg(sharpe_mean="mean", sharpe_min="min", sharpe_max="max", n_bots="count")
            .sort_values("sharpe_mean", ascending=False)
            .reset_index()
        )
        arch_summary.to_csv(ARCH_CSV, index=False)

        algo_summary = (
            results_df.groupby("algo")["mean_sharpe"]
            .agg(sharpe_mean="mean", sharpe_min="min", sharpe_max="max", n_bots="count")
            .sort_values("sharpe_mean", ascending=False)
            .reset_index()
        )
        algo_summary.to_csv(ALGO_CSV, index=False)

        print(f"\n{'=' * 60}")
        print(f"VALIDATION RESULTS — {VAL_START_DATE} to {VAL_END_DATE}")
        print(f"{'=' * 60}")
        print(results_df[["bot_id", "mean_roi_pct", "mean_sharpe", "mean_max_dd_pct"]].to_string(index=False))

        print(f"\n{'=' * 40}")
        print("ARCHITECTURE SUMMARY (v3 decision gate)")
        print(f"{'=' * 40}")
        print(arch_summary.to_string(index=False))

        print(f"\n{'=' * 40}")
        print("ALGORITHM SUMMARY")
        print(f"{'=' * 40}")
        print(algo_summary.to_string(index=False))

        print(f"\nCheckpoint saved -> {CKPT_DIR}/")
        print("  val_results.csv  — per-bot rankings")
        print("  arch_summary.csv — Sharpe by architecture (v3 input)")
        print("  algo_summary.csv — Sharpe by algorithm")
        print("\nNext: python -m algo_v2.evaluation.validate --test")

    ray.shutdown()


if __name__ == "__main__":
    main()
