#!/usr/bin/env python3
"""
Smoke tests for algo_v2 — fast (<1 min), no training, no network, no real data.

Proves the core wiring still works after a refactor step:
  S2  every import-safe module imports cleanly
  S3  config.py exposes the expected contract (tickers, archs, thresholds)
  S4  the trading env builds, resets, and steps on a tiny synthetic frame
  S5  logging + notifications wire up and degrade gracefully without creds

Every package module is import-safe (no side effects at import) and is covered
by the S2 sweep; runnable entry points are additionally guarded by `__main__`.

Run:  python tests/smoke/run_smoke.py
Exit: 0 = all checks pass, 1 = any failure.
"""

import importlib
import os
import sys

# Make the repo root importable regardless of where this is invoked from.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_failures = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001 — smoke runner reports, never raises
        _failures.append(name)
        print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")


# ── S2: import-safe modules ────────────────────────────────────────────────
IMPORT_SAFE = [
    "algo_v2.config",
    "algo_v2.platform.logging_config",
    "algo_v2.platform.notifications",
    "algo_v2.platform.rllib_env",
    "algo_v2.live.broker",
    "algo_v2.live.regime",
    "algo_v2.live.meta_controller",
    "algo_v2.core.models",
    "algo_v2.core.env_single",
    "algo_v2.core.env_competitive",
    "algo_v2.data.features",
    "algo_v2.data.fetch",
    "algo_v2.data.gan.train_gan",
    "algo_v2.data.gan.generate",
    "algo_v2.training.expert_trajectories",
    "algo_v2.training.tune",
    "algo_v2.training.train_ensemble",
    "algo_v2.monitoring.cw_dashboard",
    "algo_v2.monitoring.cw_sidecar",
    "algo_v2.evaluation.validate",
    "algo_v2.live.execute",
]


def _s2():
    for mod in IMPORT_SAFE:
        importlib.import_module(mod)


# ── S3: config contract ────────────────────────────────────────────────────
def _s3():
    import algo_v2.config as c

    assert len(c.TICKER_LIST) == 49, f"TICKER_LIST={len(c.TICKER_LIST)}"
    assert len(c.ARCH_MAP) == 7, f"ARCH_MAP={len(c.ARCH_MAP)}"
    assert c.PRIMARY_DATA_PATH, "PRIMARY_DATA_PATH empty"
    assert os.path.isabs(c.DB_PATH), "DB_PATH not absolute"
    assert abs(c.ACTION_THRESHOLD - 0.3) < 1e-9, f"ACTION_THRESHOLD={c.ACTION_THRESHOLD}"
    assert len(c.SAC_TQC_ARCH_OVERRIDES) == 2
    assert len(c.MARWIL_ARCH_OVERRIDES) == 2


# ── S4: env build + reset + step ───────────────────────────────────────────
def _s4():
    import numpy as np
    import pandas as pd
    from algo_v2.config import INDICATORS, make_env_kwargs
    from algo_v2.core.env_single import StockTradingEnv

    n_stocks, days = 5, 30
    rng = np.random.default_rng(0)
    rows = []
    for d in range(days):
        for t in range(n_stocks):
            row = {"date": f"2020-01-{d + 1:02d}", "tic": f"STK{t}", "close": 100 + rng.normal()}
            for ind in INDICATORS:
                row[ind] = rng.normal()
            row["close_30_sma"] = 100.0  # used by _get_obs price normalisation
            rows.append(row)
    df = pd.DataFrame(rows)

    kwargs = make_env_kwargs(n_stocks)
    env = StockTradingEnv(df=df, **kwargs)
    obs, _ = env.reset()
    assert obs.shape == (kwargs["state_space"],), f"obs {obs.shape}"
    obs2, reward, _term, _trunc, _info = env.step(env.action_space.sample())
    assert obs2.shape == (kwargs["state_space"],)
    float(reward)  # reward must be numeric


# ── S5: logging + notifications ────────────────────────────────────────────
def _s5():
    os.environ.pop("TELEGRAM_BOT_TOKEN", None)
    os.environ.pop("TELEGRAM_CHAT_ID", None)
    from algo_v2.platform.logging_config import configure_logging

    configure_logging("smoke").info("smoke log line")
    from algo_v2.platform.notifications import send_telegram

    assert send_telegram("smoke msg") is False, "expected False without creds"


def main():
    print("algo_v2 smoke tests")
    check(f"S2 imports ({len(IMPORT_SAFE)} import-safe modules)", _s2)
    check("S3 config contract", _s3)
    check("S4 env reset+step", _s4)
    check("S5 logging + notifications", _s5)
    print()
    if _failures:
        print(f"SMOKE FAILED: {len(_failures)} check(s): {', '.join(_failures)}")
        return 1
    print("SMOKE OK: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
