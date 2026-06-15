"""RLlib environment shim wrapping the single-agent trading environment."""

from algo_v2.core.env_single import RiskAwareTradingEnv


class RLlibEnvWrapper(RiskAwareTradingEnv):
    """Thin RLlib shim — passes seed/options through and guarantees (obs, info) return."""

    def __init__(self, df, env_kwargs):
        super().__init__(df=df, **env_kwargs)

    def reset(self, *, seed=None, options=None):
        return super().reset(seed=seed, options=options)
