import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from algo_v2.core.env_single import RiskAwareTradingEnv


class CompetitiveTradingEnv(MultiAgentEnv):
    """
    Multi-Agent Competitive Training Environment.

    Wraps N isolated RiskAwareTradingEnvs into a single market simulation with a
    mock Central Limit Order Book: when multiple agents buy the same stock
    simultaneously, excess demand causes slippage proportional to oversubscription.
    """

    LIQUIDITY_THRESHOLD = 2.0
    SLIPPAGE_RATE = 0.05  # 5% price slip per unit of excess demand

    def __init__(self, config):
        super().__init__()
        self.n_agents = config.get("num_agents", 5)
        self.agents = [f"bot_{i}" for i in range(self.n_agents)]
        self.possible_agents = self.agents[:]
        self._agent_ids = set(self.agents)

        env_config = config.get("env_config", {})
        try:
            import ray

            if isinstance(env_config.get("df"), ray.ObjectRef):
                env_config = {**env_config, "df": ray.get(env_config["df"])}
        except Exception:
            pass
        self.agent_envs = {agent: RiskAwareTradingEnv(**env_config) for agent in self.agents}

        first_env = self.agent_envs[self.agents[0]]
        self.action_space = first_env.action_space
        self.observation_space = first_env.observation_space

    def reset(self, *, seed=None, options=None):
        # Agent 0 defines the augmented "master reality" (flash crashes, noise)
        master_env = self.agent_envs[self.agents[0]]
        master_obs, _ = master_env.reset(seed=seed)
        master_df = master_env.df.copy()

        obs_dict = {self.agents[0]: master_obs}

        # All other agents share the exact same augmented market — no double augmentation
        for agent in self.agents[1:]:
            agent_obs, _ = self.agent_envs[agent].reset(seed=seed, shared_df=master_df.copy())
            obs_dict[agent] = agent_obs

        return obs_dict, {}

    def step(self, action_dict):
        # Aggregate buy demand across all agents to compute CLOB slippage
        total_demand = np.zeros(self.action_space.shape[0])
        for action in action_dict.values():
            total_demand += np.maximum(0, action)

        excess_demand = np.maximum(0, total_demand - self.LIQUIDITY_THRESHOLD)
        slippage_multipliers = 1.0 + excess_demand * self.SLIPPAGE_RATE

        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = {}, {}, {}, {}, {}
        all_done = True

        for agent_id, action in action_dict.items():
            slipped_action = action.copy()
            buy_mask = slipped_action > 0
            slipped_action[buy_mask] /= slippage_multipliers[buy_mask]

            obs, reward, terminated, truncated, info = self.agent_envs[agent_id].step(slipped_action)
            obs_dict[agent_id] = obs
            reward_dict[agent_id] = reward
            terminated_dict[agent_id] = terminated
            truncated_dict[agent_id] = truncated
            info_dict[agent_id] = info

            if not terminated and not truncated:
                all_done = False

        terminated_dict["__all__"] = all_done
        truncated_dict["__all__"] = all_done
        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict
