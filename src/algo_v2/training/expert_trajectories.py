import logging
import os

import numpy as np
import pandas as pd
from ray.rllib.offline.json_writer import JsonWriter
from ray.rllib.policy.sample_batch import SampleBatch

from algo_v2.config import make_env_kwargs, PRIMARY_DATA_PATH, VAL_START_DATE
from algo_v2.core.env_single import RiskAwareTradingEnv

logger = logging.getLogger(__name__)


class HindsightOracle:
    """
    Generates expert trading trajectories by looking ahead `lookahead_days` into the
    future and buying top performers / selling worst performers. These trajectories
    are fed to MARWIL for imitation learning (Task 13).
    """

    def __init__(self, df: pd.DataFrame, lookahead_days: int = 5):
        self.df = df
        self.lookahead_days = lookahead_days

        stock_dim = len(df["tic"].unique())
        env_kwargs = make_env_kwargs(stock_dim)
        self.env = RiskAwareTradingEnv(df=self.df, **env_kwargs)
        self.stock_dim = stock_dim

    def generate_trajectories(self, output_dir: str = "expert_data/"):
        logger.info("Generating hindsight trajectories → %s", output_dir)
        os.makedirs(output_dir, exist_ok=True)
        writer = JsonWriter(output_dir)

        obs, _ = self.env.reset()

        obs_list, action_list, reward_list, next_obs_list = [], [], [], []
        terminated_list, truncated_list = [], []
        done = False

        while not done:
            current_day = self.env.day
            future_day = min(current_day + self.lookahead_days, len(self.env.date_memory) - 1)

            current_data = self.df[self.df["date"] == self.env.date_memory[current_day]]
            future_data = self.df[self.df["date"] == self.env.date_memory[future_day]]

            if len(current_data) == self.stock_dim and len(future_data) == self.stock_dim:
                future_returns = (future_data["close"].values - current_data["close"].values) / current_data[
                    "close"
                ].values
                actions = np.zeros(self.stock_dim, dtype=np.float32)
                if len(future_returns) > 0:
                    top_thresh = np.percentile(future_returns, 80)
                    bot_thresh = np.percentile(future_returns, 20)
                    actions = np.where(
                        future_returns >= top_thresh, 1.0, np.where(future_returns <= bot_thresh, -1.0, 0.0)
                    ).astype(np.float32)
            else:
                actions = np.zeros(self.stock_dim, dtype=np.float32)

            next_obs, reward, terminated, truncated, _ = self.env.step(actions)
            is_done = terminated or truncated

            obs_list.append(np.array(obs, dtype=np.float32))
            action_list.append(actions)
            reward_list.append(float(reward))
            next_obs_list.append(np.array(next_obs, dtype=np.float32))
            terminated_list.append(bool(terminated))
            truncated_list.append(bool(truncated))

            obs = next_obs
            done = is_done

        batch = SampleBatch(
            {
                SampleBatch.OBS: obs_list,
                SampleBatch.ACTIONS: action_list,
                SampleBatch.REWARDS: reward_list,
                SampleBatch.NEXT_OBS: next_obs_list,
                SampleBatch.TERMINATEDS: terminated_list,
                SampleBatch.TRUNCATEDS: truncated_list,
            }
        )
        writer.write(batch)
        logger.info("Expert trajectories written successfully.")


if __name__ == "__main__":
    from algo_v2.platform.logging_config import configure_logging

    configure_logging("expert_trajectory_generator")

    df = pd.read_csv(PRIMARY_DATA_PATH)
    # Only use pre-validation data — same contamination boundary as train_ensemble.py
    df = df[df["date"] < VAL_START_DATE]

    for col in ("sentiment", "vix"):
        if col not in df.columns:
            df[col] = 0.0

    oracle = HindsightOracle(df)
    oracle.generate_trajectories()
