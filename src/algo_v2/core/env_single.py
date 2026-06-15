import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# Each episode = 1000 trading days (~4 years).
# On every reset, a random start point is chosen so the bot trains on a
# different market era each episode — full use of the 60-year synthetic dataset.
# 1000 days captures a full bull→bear→recovery market cycle in every episode,
# forcing the bot to learn regime transitions rather than single-phase behaviour.
# PPO batch 8000 / 1000 = 8 episodes per update — same gradient diversity as before.
EPISODE_WINDOW = 1000


class StockTradingEnv(gym.Env):
    """
    Native gymnasium stock trading environment (no FinRL dependency).

    State layout:
        [cash(1)] [holdings(N)] [stock_0: price, ind1..ind14] [stock_1: price, ind1..ind14] ...
    Total length: 1 + N + 15*N = 1 + 16*N

    This layout lets the CNN treat obs[:, 1+N:].view(B, N, 15) as a per-stock feature matrix.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df,
        stock_dim,
        hmax,
        initial_amount,
        num_stock_shares,
        buy_cost_pct,
        sell_cost_pct,
        state_space,
        action_space,
        tech_indicator_list,
        reward_scaling=1e-4,
        **kwargs,
    ):
        super().__init__()

        try:
            import ray

            if isinstance(df, ray.ObjectRef):
                df = ray.get(df)
        except Exception:
            pass
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = float(initial_amount)
        self.num_stock_shares = list(num_stock_shares)
        self.buy_cost_pct = list(buy_cost_pct)
        self.sell_cost_pct = list(sell_cost_pct)
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = list(tech_indicator_list)
        self._fpstock = 1 + len(tech_indicator_list)  # features per stock (price + indicators)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(stock_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_space,), dtype=np.float32)

        # All dates in the full dataset — used to pick random windows in subclasses
        self._all_dates = sorted(df["date"].unique().tolist())
        self._tickers = sorted(df["tic"].unique().tolist())

        # Active window dates — rebuilt each reset from the current self.df window
        self._dates: list = list(self._all_dates)

        # Runtime state (populated in reset)
        self.day = 0
        self.terminal = False
        self.state: list = []
        self.asset_memory: list = []
        self.rewards_memory: list = []
        self.actions_memory: list = []
        self.date_memory: list = list(self._dates)

        # Day cache: date → pre-sliced 49-row DataFrame for O(1) per-step lookup.
        # Built in reset() for subclasses (which swap self.df to a window slice).
        # Built here in __init__ only for direct StockTradingEnv use (expert oracle).
        self._cached_df_id: int = -1
        self._day_cache: dict = {}
        if type(self) is StockTradingEnv:
            self._build_day_cache()

    # ── Internal helpers ───────────────────────────────────────────────────

    def _build_day_cache(self):
        """
        Pre-index self.df by date for O(1) per-step lookups.

        Also syncs self._dates to match whatever window is in self.df — this
        ensures the terminal condition (day >= len(_dates)) fires correctly
        at the end of the window, not the end of the full 40-year dataset.
        """
        grouped = self.df.groupby("date", sort=False)
        self._day_cache = {d: g.sort_values("tic").reset_index(drop=True) for d, g in grouped}
        self._cached_df_id = id(self.df)
        # Sync active dates to current window
        self._dates = sorted(self._day_cache.keys())
        self.date_memory = list(self._dates)

    def _get_day_data(self, day: int) -> pd.DataFrame:
        return self._day_cache[self._dates[day]]

    def _get_obs(self) -> np.ndarray:
        """
        Build the observation the neural network actually sees.

        The raw self.state stores ABSOLUTE close prices so that buy/sell/portfolio
        calculations remain correct.  This method replaces each stock's close price
        with  close / close_30_sma  before handing the array to the policy network.

        Why: a close of ₹150 (1986) vs ₹22,000 (2024) leaks the calendar era.
             close / SMA30 ≈ 1.0 everywhere — the bot sees *relative* price position,
             not absolute magnitude.  A value of 1.05 means "price is 5% above its
             30-day average" regardless of which decade the episode is drawn from.

        close_30_sma is already in self.state (it's indicator index 6), so this is
        O(N=49) arithmetic with no extra I/O.
        """
        N, F = self.stock_dim, self._fpstock
        obs = list(self.state)
        try:
            sma_offset = 1 + self.tech_indicator_list.index("close_30_sma")
            for i in range(N):
                base = 1 + N + i * F
                price = obs[base]
                sma30 = obs[base + sma_offset]
                obs[base] = price / sma30 if sma30 > 0 else 1.0
        except ValueError:
            pass  # close_30_sma not in indicator list — keep absolute price
        return np.array(obs, dtype=np.float32)

    def _build_state(self, cash: float, holdings: list, data: pd.DataFrame) -> list:
        state: list = [cash] + list(holdings)
        F = self._fpstock
        for i in range(self.stock_dim):
            row = data.iloc[i]
            state.append(float(row["close"]))
            for ind in self.tech_indicator_list:
                state.append(float(row.get(ind, 0.0)))
        return state

    def _portfolio_value(self) -> float:
        N, F = self.stock_dim, self._fpstock
        cash = self.state[0]
        holdings = np.array(self.state[1 : 1 + N], dtype=np.float64)
        prices = np.array([self.state[1 + N + i * F] for i in range(N)], dtype=np.float64)
        return float(cash + np.dot(holdings, prices))

    def _sell_stock(self, i: int, action: float):
        F = self._fpstock
        price = self.state[1 + self.stock_dim + i * F]
        shares_held = self.state[1 + i]
        shares_to_sell = min(int(abs(action) * self.hmax), int(shares_held))
        if shares_to_sell > 0 and price > 0:
            self.state[0] += price * shares_to_sell * (1.0 - self.sell_cost_pct[i])
            self.state[1 + i] -= shares_to_sell

    def _buy_stock(self, i: int, action: float):
        F = self._fpstock
        price = self.state[1 + self.stock_dim + i * F]
        if price <= 0:
            return
        cost_per_share = price * (1.0 + self.buy_cost_pct[i])
        max_by_cash = int(self.state[0] / cost_per_share)
        shares_to_buy = min(int(action * self.hmax), max_by_cash)
        if shares_to_buy > 0:
            self.state[0] -= price * shares_to_buy * (1.0 + self.buy_cost_pct[i])
            self.state[1 + i] += shares_to_buy

    def _update_market(self, data: pd.DataFrame):
        F = self._fpstock
        for i in range(self.stock_dim):
            row = data.iloc[i]
            base = 1 + self.stock_dim + i * F
            self.state[base] = float(row["close"])
            for j, ind in enumerate(self.tech_indicator_list):
                self.state[base + 1 + j] = float(row.get(ind, 0.0))

    # ── Gymnasium API ──────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Rebuild cache if self.df was swapped to a new window slice
        if id(self.df) != self._cached_df_id:
            self._build_day_cache()
        self.day = 0
        self.terminal = False

        data = self._get_day_data(0)
        self.state = self._build_state(self.initial_amount, self.num_stock_shares, data)

        self.asset_memory = [self._portfolio_value()]
        self.rewards_memory = []
        self.actions_memory = []

        return self._get_obs(), {}

    def step(self, actions):
        if self.terminal:
            return self._get_obs(), 0.0, True, False, {}

        actions = np.clip(actions, -1.0, 1.0)
        sort_idx = np.argsort(actions)

        for i in sort_idx:
            if actions[i] < 0:
                self._sell_stock(i, float(actions[i]))
        for i in sort_idx[::-1]:
            if actions[i] > 0:
                self._buy_stock(i, float(actions[i]))

        self.actions_memory.append(actions.tolist())
        prev_value = self.asset_memory[-1]

        self.day += 1
        self.terminal = self.day >= len(self._dates)

        if not self.terminal:
            self._update_market(self._get_day_data(self.day))

        curr_value = self._portfolio_value()
        self.asset_memory.append(curr_value)
        reward = (curr_value - prev_value) * self.reward_scaling
        self.rewards_memory.append(reward)

        return self._get_obs(), reward, self.terminal, False, {}

    def render(self):
        pass


class RiskAwareTradingEnv(StockTradingEnv):
    def __init__(self, df, **kwargs):
        try:
            import ray

            if isinstance(df, ray.ObjectRef):
                df = ray.get(df)
        except Exception:
            pass
        self.original_df = df.copy()
        self.downside_penalty = kwargs.pop("downside_penalty", 2.0)
        super().__init__(df, **kwargs)
        # Don't pre-build cache here — reset() will build it from the window slice,
        # which is much smaller (~49K rows) than the full 550K-row dataset.

    def _augment_data(self, df):
        df = df.reset_index(drop=True)
        for indicator in ["macd", "rsi_30", "vix", "sentiment"]:
            if indicator in df.columns:
                std_dev = df[indicator].std() * 0.05
                df.loc[:, indicator] = df[indicator] + np.random.normal(0, std_dev, size=len(df))

        num_crashes = max(1, int(len(df) * 0.005))
        crash_indices = np.random.choice(len(df), size=num_crashes, replace=False)
        severities = np.random.uniform(0.05, 0.25, size=num_crashes)
        for idx, severity in zip(crash_indices, severities):
            df.at[idx, "close"] = df.at[idx, "close"] * (1.0 - severity)
        return df

    def reset(self, *, seed=None, options=None, shared_df=None):
        if shared_df is not None:
            # Sub-agent: master already selected and augmented the window — reuse it
            self.df = shared_df
        else:
            # Master agent: randomly pick a 1000-day window from the full 40-year dataset,
            # then augment only that window (~49K rows, not 550K).
            # Each episode the bot sees a different market era: crashes, bubbles, sideways.
            n_all = len(self._all_dates)
            max_start = max(0, n_all - EPISODE_WINDOW)
            start_idx = int(np.random.randint(0, max_start + 1))
            window_dates = set(self._all_dates[start_idx : start_idx + EPISODE_WINDOW])
            window_df = self.original_df[self.original_df["date"].isin(window_dates)].reset_index(drop=True)
            self.df = self._augment_data(window_df)

        return super().reset(seed=seed, options=options)

    def step(self, actions):
        obs, reward, terminated, truncated, info = super().step(actions)

        N = self.stock_dim
        F = self._fpstock
        holdings = np.array(self.state[1 : 1 + N])
        prices = np.array([self.state[1 + N + i * F] for i in range(N)])
        asset_values = holdings * prices
        total_value = self.state[0] + asset_values.sum()

        # ── Concentration penalty ────────────────────────────────────────────
        # Penalise over-concentration: a bot that puts everything in one stock
        # and loses gets a worse reward; one that diversifies and profits keeps
        # more of the gain.  We ONLY scale — never add a second separate term —
        # to keep the reward bounded and the gradient signal clean.
        max_weight = (asset_values.max() / total_value) if total_value > 0 else 0.0
        concentration_factor = 1.0 + (max_weight**2) * 2.0  # max 3× (was 6×)
        if reward > 0:
            reward /= concentration_factor  # diversified profits kept in full
        elif reward < 0:
            reward *= concentration_factor  # concentrated losses amplified

        # ── Sortino scaling (multiplicative — NEVER additive) ────────────────────
        # Uses 100-day rolling window (31 was too short — noisy in early episodes).
        # Only activates once ≥31 days of history exist; defaults to scale=1.0 before.
        #
        # Formula:   scale ∈ [0.75, 1.25]
        #   Sortino +2 (excellent)  → reward × 1.25  (bonus for consistency)
        #   Sortino  0 (neutral)    → reward × 1.00  (no change)
        #   Sortino -2 (terrible)   → reward × 0.75  (soft reduction, not double-penalty)
        #
        # This is NOT a second penalty on losses — the base reward already captured
        # portfolio delta. This only rescales the already-clean signal, teaching
        # the bot that consistent returns are worth more than volatile ones.
        if len(self.asset_memory) >= 31:
            window = min(100, len(self.asset_memory) - 1)
            vals = np.array(self.asset_memory[-window - 1 :], dtype=np.float64)
            rets = np.diff(vals) / np.maximum(np.abs(vals[:-1]), 1.0)
            downside = rets[rets < 0]
            downside_std = float(np.std(downside)) if len(downside) >= 3 else 1e-6
            mean_ret = float(np.mean(rets))
            sortino = float(np.clip(mean_ret / (downside_std + 1e-8), -2.0, 2.0))
            sortino_scale = 1.0 + 0.125 * sortino  # maps [-2,+2] → [0.75, 1.25]
            # Sign-aware: for profits, good Sortino amplifies; for losses, bad Sortino
            # amplifies (more punishment for volatile losses) and good Sortino reduces
            # (losses are controlled — reduce gradient signal, not increase it).
            if reward >= 0:
                reward *= sortino_scale
            else:
                reward /= sortino_scale

        reward -= self.state[0] * 0.0001 * self.reward_scaling
        # Clamp to [-10, +10] per step: concentration_factor × Sortino can spike
        # edge-case steps to ±15+, creating noisy value-function targets.
        # Per-step rewards are normally ±2 so this clip only catches true outliers.
        reward = float(np.clip(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=-1.0), -10.0, 10.0))

        return obs, reward, terminated, truncated, info
