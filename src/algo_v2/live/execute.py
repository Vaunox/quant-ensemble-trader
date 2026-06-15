#!/usr/bin/env python3
"""
Live execution pipeline — 42-bot ensemble with bandit capital allocation.

Daily flow:
  1. Fetch today's live market data
  2. Simulate each bot's yesterday action at today's prices → per-bot returns
  3. Feed per-bot returns to MetaController (bandit learns who to trust)
  4. Get regime-aware Sharpe-weighted capital allocation from MetaController
  5. Each bot computes today's action
  6. Weighted ensemble action → execute orders via broker
  7. Save today's actions + prices for tomorrow's return calculation

Run once daily after market close:
    python live_executor.py
"""

import argparse
import gc
import os
import pickle
import warnings
from datetime import datetime, timedelta

_p = argparse.ArgumentParser()
_p.add_argument("--dry-run", action="store_true", help="Compute orders but do not execute them (safe for testing)")
_args, _ = _p.parse_known_args()
DRY_RUN = _args.dry_run

import numpy as np
import pandas as pd
import ray
import stockstats
import yfinance as yf
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.algorithms.marwil import MARWILConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.tqc import TQCConfig
from ray.tune.registry import register_env

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from algo_v2.platform.logging_config import configure_logging
from algo_v2.live.broker import DummyBroker
from algo_v2.core.models import register_models
from algo_v2.config import (
    INDICATORS,
    TECHNICAL_INDICATORS,
    PRIMARY_DATA_PATH,
    make_env_kwargs,
    HMAX,
    INITIAL_CAPITAL,
    ACTION_THRESHOLD,
    ARCH_MAP,
    SAC_TQC_ARCH_OVERRIDES,
    MARWIL_ARCH_OVERRIDES,
    PROJECT_ROOT,
    TRAINED_MODELS_DIR,
    EXPERT_DATA_DIR,
    LIVE_STATES_DIR,
)
from algo_v2.core.env_single import StockTradingEnv
from algo_v2.live.meta_controller import MetaController
from algo_v2.platform.notifications import send_telegram
from algo_v2.data.features import download_vix, grid_fill

logger = configure_logging("live_executor")

ENSEMBLE_DIR = os.path.join(TRAINED_MODELS_DIR, "ensemble")
STATE_DIR = LIVE_STATES_DIR
ACTIONS_FILE = os.path.join(PROJECT_ROOT, "last_actions.pkl")

os.makedirs(STATE_DIR, exist_ok=True)
register_env("stock_trading", lambda cfg: StockTradingEnv(**cfg))
register_models()

# SAC_TQC_ARCH_OVERRIDES, MARWIL_ARCH_OVERRIDES, ARCH_MAP imported from config.
# Only load bots that showed positive Sharpe in validation (Sharpe ≥ 0.1 → A or B grade).
# C-grade bots get zero weight from np.maximum(0, sharpe) anyway; excluding them saves memory.
PAPER_TRADE_WHITELIST = {
    # A-grade (Sharpe ≥ 0.7) — 14 bots (bot_40 excluded — corrupt checkpoint)
    "bot_14_appo_god_tier",
    "bot_36_marwil_pure_mlp",
    "bot_35_tqc_god_tier",
    "bot_18_sac_cnn_macro",
    "bot_32_tqc_cnn_macro",
    "bot_20_sac_transformer_cnn",
    "bot_27_impala_transformer_cnn",
    "bot_22_impala_pure_mlp",
    "bot_33_tqc_cnn_dual",
    "bot_19_sac_cnn_dual",
    "bot_42_marwil_god_tier",
    "bot_37_marwil_pure_lstm",
    "bot_09_appo_pure_lstm",
    "bot_29_tqc_pure_mlp",
    # B-grade (Sharpe 0.1–0.7) — 12 bots
    "bot_21_sac_god_tier",
    "bot_41_marwil_transformer_cnn",
    "bot_39_marwil_cnn_macro",
    "bot_38_marwil_cnn_micro",
    "bot_16_sac_pure_lstm",
    "bot_30_tqc_pure_lstm",
    "bot_12_appo_cnn_dual",
    "bot_11_appo_cnn_macro",
    "bot_07_ppo_god_tier",
    "bot_01_ppo_pure_mlp",
    "bot_26_impala_cnn_dual",
    "bot_28_impala_god_tier",
}


def _load_dotenv():
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


ALGO_MAP = {
    "ppo": PPOConfig,
    "appo": APPOConfig,
    "sac": SACConfig,
    "impala": ImpalaConfig,
    "tqc": TQCConfig,
    "marwil": MARWILConfig,
}
# ARCH_MAP, SAC_TQC_ARCH_OVERRIDES, MARWIL_ARCH_OVERRIDES imported from config


def fetch_live_data(tickers: list) -> pd.DataFrame:
    today = datetime.today()
    end_str = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    start_str = (today - timedelta(days=90)).strftime("%Y-%m-%d")

    df_list = []
    failed_tickers = []
    for tic in tickers:
        try:
            raw = yf.download(tic, start=start_str, end=end_str, progress=False)
            if raw.empty:
                continue
            raw = raw.reset_index()
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            raw.columns = [c.lower() for c in raw.columns]
            raw["tic"] = tic
            df_list.append(raw)
        except Exception as e:
            failed_tickers.append(tic)
            print(f"  [WARN] {tic} download failed: {str(e)[:80]}")

    if failed_tickers:
        print(f"  [WARN] {len(failed_tickers)}/{len(tickers)} tickers failed to download: {', '.join(failed_tickers)}")
    # More than 5 failures means the observation vector will be too short,
    # causing a shape mismatch inside Ray. Abort with a clear message.
    if len(failed_tickers) > 5:
        raise RuntimeError(
            f"{len(failed_tickers)} tickers failed to download — aborting to prevent "
            f"observation shape mismatch (model expects {len(tickers)} stocks)."
        )
    if not df_list:
        raise RuntimeError("No live data fetched — check network and ticker list.")

    df = pd.concat(df_list, ignore_index=True)
    df["date"] = df["date"].astype(str)
    if "adj close" in df.columns:
        df = df.drop(columns=["close"]).rename(columns={"adj close": "close"})
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)

    frames = []
    for tic, grp in df.groupby("tic"):
        grp = grp.sort_values("date").reset_index(drop=True)
        stock = stockstats.StockDataFrame.retype(grp.copy())
        for ind in TECHNICAL_INDICATORS:
            try:
                grp[ind] = stock[ind]
                grp[ind] = grp[ind].ffill().fillna(0.0)
            except Exception:
                grp[ind] = 0.0
        frames.append(grp)
    df = pd.concat(frames, ignore_index=True).sort_values(["date", "tic"]).reset_index(drop=True)

    vix_df = download_vix(start_str, end_str)
    if vix_df is not None:
        df = df.merge(vix_df, on="date", how="left")
        df["vix"] = df["vix"].ffill().bfill()
    else:
        df["vix"] = 0.0

    if os.path.exists("sentiment_db.csv"):
        sent_df = pd.read_csv("sentiment_db.csv")
        df = df.merge(sent_df, on=["date", "tic"], how="left")
        df["sentiment"] = df["sentiment"].ffill().fillna(0.0)
    else:
        df["sentiment"] = 0.0

    return grid_fill(df, df["date"].unique(), tickers)


def get_nifty_return(today_str: str, prev_date_str: str) -> float:
    try:
        raw = yf.download("^NSEI", start=prev_date_str, end=today_str, progress=False)
        if len(raw) >= 2:
            return float(raw["Close"].pct_change().iloc[-1])
    except Exception:
        pass
    return 0.0


def simulate_per_bot_returns(
    yesterday_actions: dict, entry_prices: np.ndarray, exit_prices: np.ndarray, initial_capital: float
) -> dict:
    """
    Simulate each bot's yesterday action on a clean portfolio at entry_prices,
    then value the result at exit_prices. Gives MetaController an independent
    per-bot signal quality score to learn from.

    All-cash start means only BUY signals are evaluated — correct, since the
    real env also cannot sell what it doesn't already hold.
    """
    returns = {}
    for bot_id, action in yesterday_actions.items():
        cash = float(initial_capital)
        holdings = np.zeros(len(action), dtype=np.float64)

        for i in np.argsort(action)[::-1]:
            if action[i] <= 0 or entry_prices[i] <= 0:
                continue
            cost = entry_prices[i] * 1.0002
            shares = min(int(action[i] * HMAX), int(cash / cost))
            if shares > 0:
                cash -= shares * cost
                holdings[i] = shares

        end_value = cash + float(np.dot(holdings, exit_prices))
        returns[bot_id] = (end_value - initial_capital) / initial_capital

    return returns


def _build_inference_config(algo_name: str, arch_name: str, window_df: pd.DataFrame, env_kwargs: dict):
    ConfigClass = ALGO_MAP[algo_name]
    if algo_name in ("sac", "tqc"):
        model_settings = SAC_TQC_ARCH_OVERRIDES.get(arch_name, ARCH_MAP[arch_name])
    elif algo_name == "marwil":
        model_settings = MARWIL_ARCH_OVERRIDES.get(arch_name, ARCH_MAP[arch_name])
    else:
        model_settings = ARCH_MAP[arch_name]

    config = (
        ConfigClass()
        .environment("stock_trading", env_config={"df": window_df, **env_kwargs})
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
            # Putting it in model= makes SAC wrap the CNN as the outer SACTorchModel,
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
        expert_dir = EXPERT_DATA_DIR
        if os.path.exists(expert_dir):
            config.offline_data(input_=expert_dir)
        config.training(beta=1.0)

    return config


def load_all_bots(window_df: pd.DataFrame, env_kwargs: dict) -> dict:
    """Load whitelisted bots (A/B-grade Sharpe ≥ 0.1) from ensemble directory."""
    os.makedirs(ENSEMBLE_DIR, exist_ok=True)
    bot_dirs = sorted(
        [
            d
            for d in os.listdir(ENSEMBLE_DIR)
            if os.path.isdir(os.path.join(ENSEMBLE_DIR, d))
            and d.startswith("bot_")
            and not d.endswith("_best_ckpt")
            and d in PAPER_TRADE_WHITELIST
        ]
    )

    bots = {}
    for bot_id in bot_dirs:
        parts = bot_id.split("_")
        if len(parts) < 4:
            continue
        algo_name = parts[2]
        arch_name = "_".join(parts[3:])

        if algo_name not in ALGO_MAP or arch_name not in ARCH_MAP:
            print(f"  [SKIP] {bot_id} — unknown algo/arch")
            continue

        try:
            config = _build_inference_config(algo_name, arch_name, window_df, env_kwargs)
            trainer = config.build_algo()
            trainer.restore(os.path.join(ENSEMBLE_DIR, bot_id))
            if algo_name in ("sac", "tqc") and arch_name in SAC_TQC_ARCH_OVERRIDES:
                _use_lstm = SAC_TQC_ARCH_OVERRIDES[arch_name].get("use_lstm", False)
            elif algo_name == "marwil" and arch_name in MARWIL_ARCH_OVERRIDES:
                _use_lstm = MARWIL_ARCH_OVERRIDES[arch_name].get("use_lstm", False)
            else:
                _use_lstm = ARCH_MAP[arch_name].get("use_lstm", False)
            bots[bot_id] = {
                "trainer": trainer,
                "use_lstm": _use_lstm,
            }
            print(f"  [OK]   {bot_id}")
        except Exception as e:
            print(f"  [FAIL] {bot_id}: {str(e)[:80]}")

    return bots


def build_live_state(df: pd.DataFrame, broker: DummyBroker, tickers: list, env_kwargs: dict) -> np.ndarray:
    latest_date = df["date"].max()
    today_df = df[df["date"] == latest_date].sort_values("tic").reset_index(drop=True)

    state = [broker.get_cash_balance()]
    holdings = broker.get_holdings()
    state.extend(holdings.get(tic, 0) for tic in tickers)

    for tic in tickers:
        row = today_df[today_df["tic"] == tic]
        if row.empty:
            state.extend([0.0] * (1 + len(INDICATORS)))
        else:
            r = row.iloc[0]
            close_val = float(r["close"])
            sma30 = float(r.get("close_30_sma", 0.0))
            state.append(close_val / sma30 if sma30 > 0 else 1.0)
            for ind in INDICATORS:
                val = r.get(ind, 0.0)
                state.append(0.0 if (val is None or (isinstance(val, float) and np.isnan(val))) else float(val))

    return np.array(state, dtype=np.float32)


def get_bot_actions(bots: dict, live_state: np.ndarray) -> dict:
    """Compute each bot's recommended action, persisting LSTM states between daily runs."""
    actions = {}
    for bot_id, bot in bots.items():
        try:
            state_file = os.path.join(STATE_DIR, f"{bot_id}.pkl")

            if bot["use_lstm"]:
                if os.path.exists(state_file):
                    with open(state_file, "rb") as f:
                        saved = pickle.load(f)
                    if isinstance(saved, dict):
                        state_in = saved.get("rnn_state")
                        prev_action = saved.get("prev_action", np.zeros(49, dtype=np.float32))
                    else:
                        state_in = saved
                        prev_action = np.zeros(49, dtype=np.float32)
                else:
                    state_in = bot["trainer"].get_policy().get_initial_state()
                    prev_action = np.zeros(49, dtype=np.float32)

                action, state_out, _ = bot["trainer"].compute_single_action(
                    live_state, state=state_in, prev_action=prev_action, prev_reward=0.0
                )
                with open(state_file, "wb") as f:
                    pickle.dump({"rnn_state": state_out, "prev_action": action}, f)
            else:
                action = bot["trainer"].compute_single_action(live_state)

            actions[bot_id] = np.array(action, dtype=np.float32)

        except Exception as e:
            print(f"  [WARN] {bot_id}: {str(e)[:120]}")

    return actions


def execute_orders(
    ensemble_action: np.ndarray, tickers: list, today_df: pd.DataFrame, broker: DummyBroker, dry_run: bool = False
) -> list:
    prices = {r["tic"]: float(r["close"]) for _, r in today_df.iterrows() if "close" in r}
    sort_idx = np.argsort(ensemble_action)

    prefix = "[DRY] " if dry_run else ""
    print(f"\n  --- {prefix}Orders (SELLs first) ---")
    orders = []
    for i in sort_idx:
        a = ensemble_action[i]
        tic = tickers[i]
        p = prices.get(tic, 0.0)
        if a == 0 or p <= 0:
            continue

        if a < 0:
            held = broker.get_holdings().get(tic, 0)
            qty = min(int(abs(a) * HMAX), held)
            if qty > 0:
                if dry_run or broker.place_market_order(tic, -qty, p):
                    print(f"  {prefix}SELL {qty:5d} x {tic:<20s} @ {p:>10.2f}")
                    orders.append({"side": "SELL", "tic": tic, "qty": qty, "price": p})
        else:
            affordable = int(broker.get_cash_balance() / (p * 1.0002))
            qty = min(int(a * HMAX), affordable)
            if qty > 0:
                if dry_run or broker.place_market_order(tic, qty, p):
                    print(f"  {prefix}BUY  {qty:5d} x {tic:<20s} @ {p:>10.2f}")
                    orders.append({"side": "BUY", "tic": tic, "qty": qty, "price": p})

    if not orders:
        print(f"  {prefix}No orders (all signals below threshold or insufficient funds)")
    return orders


def _build_voting_report(orders: list, bot_actions: dict, tickers: list, today_str: str) -> str:
    """Second Telegram message: per-stock bot consensus for each executed trade."""
    if not orders or not bot_actions:
        return ""

    n_bots = len(bot_actions)
    ALGOS = ["ppo", "appo", "sac", "impala", "tqc", "marwil"]
    SIGNAL_THRESHOLD = 0.05  # minimum action magnitude to count as a vote

    # Pre-group bots by algo
    algo_bots: dict[str, list] = {a: [] for a in ALGOS}
    for bot_id, action in bot_actions.items():
        algo = bot_id.split("_")[2]
        if algo in algo_bots:
            algo_bots[algo].append(action)

    sections = [f"<b>🤖 Bot Consensus — {today_str}</b>"]

    for order in orders:
        tic = order["tic"]
        side = order["side"]
        idx = tickers.index(tic)
        tic_short = tic.replace(".NS", "").replace(".BSE", "")

        signals = np.array([a[idx] for a in bot_actions.values()])
        buy_n = int((signals > SIGNAL_THRESHOLD).sum())
        sell_n = int((signals < -SIGNAL_THRESHOLD).sum())
        neut_n = n_bots - buy_n - sell_n
        avg_sig = float(signals.mean())

        # Algo-level consensus: ✓ = majority in same direction, ✗ = opposite, ~ = split
        algo_tags = []
        for alg in ALGOS:
            grp = algo_bots.get(alg, [])
            if not grp:
                continue
            alg_avg = float(np.mean([a[idx] for a in grp]))
            if alg_avg > SIGNAL_THRESHOLD:
                tag = f"{alg.upper()}✓"
            elif alg_avg < -SIGNAL_THRESHOLD:
                tag = f"{alg.upper()}✗"
            else:
                tag = f"{alg.upper()}~"
            algo_tags.append(tag)

        consensus_n = buy_n if side == "BUY" else sell_n
        em = "📈" if side == "BUY" else "📉"

        sections.append(
            f"{em} <b>{tic_short}</b> — {consensus_n}/{n_bots} bots {side} "
            f"(avg signal: {avg_sig:+.3f})\n"
            f"   Votes: {buy_n}▲ {sell_n}▼ {neut_n}⚪\n"
            f"   Algos: {' '.join(algo_tags)}"
        )

    # Top 3 highest-consensus stocks that were NOT traded (signal too weak to cross threshold)
    traded_tickers = {o["tic"] for o in orders}
    consensus_scores = []
    for i, tic in enumerate(tickers):
        if tic in traded_tickers:
            continue
        sigs = np.array([a[i] for a in bot_actions.values()])
        agree = int((np.abs(sigs) > SIGNAL_THRESHOLD).sum())
        avg = float(sigs.mean())
        consensus_scores.append((tic, agree, avg))

    top_watching = sorted(consensus_scores, key=lambda x: x[1], reverse=True)[:3]
    if top_watching:
        watch_lines = "\n".join(
            f"• {t.replace('.NS', '').replace('.BSE', '')}: {n}/{n_bots} bots (avg {a:+.3f})"
            for t, n, a in top_watching
        )
        sections.append(f"<b>👀 Watching (not traded):</b>\n{watch_lines}")

    return "\n\n".join(sections)


def execute_live_pipeline():
    _load_dotenv()
    today_str = datetime.today().strftime("%Y-%m-%d")
    yesterday_str = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"\n{'=' * 60}")
    print(f"LIVE EXECUTION — {today_str}")
    print(f"{'=' * 60}\n")

    if DRY_RUN:
        print("*** DRY RUN — orders will be computed but not executed ***\n")
    avg_bot_return = None
    nifty_ret = 0.0
    regime_str = "N/A"
    mc = MetaController()
    broker = DummyBroker(initial_capital=INITIAL_CAPITAL)
    ref_df = pd.read_csv(PRIMARY_DATA_PATH)
    tickers = sorted(ref_df["tic"].unique().tolist())
    stock_dim = len(tickers)
    env_kwargs = make_env_kwargs(stock_dim)

    # ── 1. Fetch live data ────────────────────────────────────────────────────
    print("Fetching live market data (90-day window for indicators)...")
    df = fetch_live_data(tickers)
    latest_date = df["date"].max()
    today_df = df[df["date"] == latest_date].sort_values("tic").reset_index(drop=True)
    today_prices = np.array(
        [
            float(today_df.loc[today_df["tic"] == t, "close"].values[0]) if t in today_df["tic"].values else 0.0
            for t in tickers
        ],
        dtype=np.float64,
    )
    print(f"  Latest data: {latest_date} | {len(tickers)} tickers")

    # ── 2. Update MetaController with yesterday's per-bot returns ─────────────
    if os.path.exists(ACTIONS_FILE):
        with open(ACTIONS_FILE, "rb") as f:
            saved = pickle.load(f)

        prev_actions = saved.get("actions", {})
        prev_prices = saved.get("prices", today_prices)
        prev_date = saved.get("date", yesterday_str)

        if prev_actions:
            print(f"\nSimulating per-bot returns from {prev_date}...")
            per_bot_returns = simulate_per_bot_returns(prev_actions, prev_prices, today_prices, INITIAL_CAPITAL)
            nifty_ret = get_nifty_return(today_str, prev_date)

            alerts = []
            for bot_id, ret in per_bot_returns.items():
                alert = mc.update_bot_performance(today_str, bot_id, ret, nifty_ret)
                if alert:
                    alerts.append(alert)

            pos = sum(1 for r in per_bot_returns.values() if r > 0)
            avg = np.mean(list(per_bot_returns.values())) * 100
            avg_bot_return = avg
            print(
                f"  {pos}/{len(per_bot_returns)} bots beat cash today | "
                f"avg return: {avg:+.3f}% | NIFTY: {nifty_ret * 100:+.3f}%"
            )
            for a in alerts:
                print(f"\n  [ALERT] {a}")
    else:
        print("First run — no previous actions to evaluate. MetaController starts fresh.")

    # ── 3. Load all 42 bots ───────────────────────────────────────────────────
    print(f"\nLoading ensemble from {ENSEMBLE_DIR}...")
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}},
    )
    bots = load_all_bots(df, env_kwargs)

    if not bots:
        print("FATAL: no trained bots found. Aborting.")
        ray.shutdown()
        return

    print(f"  {len(bots)}/{len(PAPER_TRADE_WHITELIST)} bots loaded\n")

    # ── 4. Get regime-aware capital weights ───────────────────────────────────
    weights = mc.get_capital_allocation()
    weights = {b: w for b, w in weights.items() if b in bots}

    if not weights:
        weights = {bot_id: 1.0 / len(bots) for bot_id in bots}
        print("No bandit history yet — equal weights.")
    else:
        regime = mc.regime_detector.detect_current_regime()
        regime_str = str(regime)
        active = sum(1 for w in weights.values() if w > 0.001)
        top3 = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"Regime: {regime} | Active bots: {active}/{len(bots)}")
        top3_str = ", ".join(f"{b.split('_', 3)[-1]} ({w:.1%})" for b, w in top3)
        print(f"Top 3: {top3_str}")

    # ── 5. Compute actions ────────────────────────────────────────────────────
    live_state = build_live_state(df, broker, tickers, env_kwargs)
    print(f"\nComputing {len(bots)} bot actions...")
    bot_actions = get_bot_actions(bots, live_state)

    # ── 6. Weighted ensemble action ───────────────────────────────────────────
    total_weight = sum(weights.get(bid, 0.0) for bid in bot_actions)
    if total_weight == 0:
        total_weight = len(bot_actions)
        weights = {bid: 1.0 for bid in bot_actions}

    ensemble_action = np.zeros(stock_dim, dtype=np.float32)
    for bot_id, action in bot_actions.items():
        ensemble_action += (weights.get(bot_id, 0.0) / total_weight) * action

    ensemble_action[np.abs(ensemble_action) < ACTION_THRESHOLD] = 0.0

    # ── 7. Execute orders ─────────────────────────────────────────────────────
    print(f"\nBroker — cash: {broker.get_cash_balance():,.0f} INR")
    orders = execute_orders(ensemble_action, tickers, today_df, broker, dry_run=DRY_RUN)
    n_orders = len(orders)

    cash = broker.get_cash_balance()
    holdings = broker.get_holdings()
    port_val = cash + sum(holdings[t] * today_prices[tickers.index(t)] for t in holdings if t in tickers)
    print(f"\n  Cash: {cash:,.0f} INR | Portfolio: {port_val:,.0f} INR | Positions: {len(holdings)}")

    # ── 8. Persist actions + prices for tomorrow ──────────────────────────────
    with open(ACTIONS_FILE, "wb") as f:
        pickle.dump({"date": today_str, "actions": bot_actions, "prices": today_prices}, f)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    for bot in bots.values():
        try:
            bot["trainer"].stop()
        except Exception:
            pass
    gc.collect()
    ray.shutdown()

    # ── Build enriched Telegram message ──────────────────────────────────────
    ret_line = (
        f"Bot avg: {avg_bot_return:+.2f}% | NIFTY: {nifty_ret * 100:+.2f}%"
        if avg_bot_return is not None
        else "First run — no return data yet"
    )

    side_em = {"BUY": "📈", "SELL": "📉"}
    if orders:
        trade_lines = "\n".join(
            f"{side_em[o['side']]} {o['side']} {o['qty']} "
            f"{o['tic'].replace('.NS', '').replace('.BSE', '')} @ ₹{o['price']:,.0f}"
            for o in orders[:6]
        )
        if len(orders) > 6:
            trade_lines += f"\n  …and {len(orders) - 6} more"
    else:
        trade_lines = "⚪ No orders (market neutral)"

    top3 = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
    bot_lines = "\n".join(f"• {b.split('_', 3)[-1]} ({w:.1%})" for b, w in top3)

    tg_msg = (
        f"<b>📊 Paper Trade — {today_str}</b>\n"
        f"Regime: <code>{regime_str}</code> | Bots: {len(bots)}/{len(PAPER_TRADE_WHITELIST)}\n\n"
        f"💰 Portfolio: ₹{port_val:,.0f}\n"
        f"Cash: ₹{cash:,.0f} | Positions: {len(holdings)}\n\n"
        f"<b>Trades ({n_orders}):</b>\n{trade_lines}\n\n"
        f"<b>Top bots:</b>\n{bot_lines}\n\n"
        f"{ret_line}"
    )
    if not DRY_RUN:
        send_telegram(tg_msg)
        send_telegram(_build_voting_report(orders, bot_actions, tickers, today_str))
    else:
        print(f"\n[DRY RUN] Telegram skipped (would send: {tg_msg[:120]}...)")

    print(f"\n{'=' * 60}")
    print(f"Done — {today_str}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    try:
        execute_live_pipeline()
    except Exception as exc:
        msg = f"<b>FATAL — live_executor crashed</b>\n<code>{type(exc).__name__}: {exc}</code>"
        send_telegram(msg)
        raise
