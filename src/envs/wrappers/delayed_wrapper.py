import copy
from typing import Any

import numpy as np

from envs.multiagentenv import MultiAgentEnv


class DelayedObservationWrapper(MultiAgentEnv):
    """Delay local observations while leaving state and action availability fresh."""

    def __init__(
        self,
        env: MultiAgentEnv,
        delay_type: str = "fixed",
        delay: int = 0,
        delay_mean: float = 0.0,
        delay_std: float = 0.0,
        max_delay: int = 0,
        per_agent: bool = True,
        seed: int | None = None,
    ):
        self.env = env
        self.delay_type = delay_type
        self.delay = max(0, int(delay))
        self.delay_mean = float(delay_mean)
        self.delay_std = max(0.0, float(delay_std))
        self.max_delay = max(0, int(max_delay))
        self.per_agent = bool(per_agent)
        self._rng = np.random.default_rng(seed)

        self.episode_limit = env.episode_limit
        env_info = env.get_env_info()
        self.n_agents = env_info["n_agents"]
        self._time = 0
        self._obs_history: list[list[np.ndarray]] = []
        self._current_obs: list[np.ndarray] = []
        self._current_delays = np.zeros(self.n_agents, dtype=np.int64)
        self._current_generation_times = np.zeros(self.n_agents, dtype=np.int64)

        if self.delay_type not in {"fixed", "uniform", "gaussian"}:
            raise ValueError(f"Unknown delay_type: {self.delay_type}")

    def reset(self, seed=None, options=None):
        fresh_obs, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._time = 0
        self._obs_history = [self._copy_obs(fresh_obs)]
        self._refresh_delayed_obs()
        return self.get_obs(), info

    def step(self, actions):
        _, reward, terminated, truncated, info = self.env.step(actions)
        self._time += 1
        self._obs_history.append(self._copy_obs(self.env.get_obs()))
        self._refresh_delayed_obs()
        return self.get_obs(), reward, terminated, truncated, info

    def get_obs(self):
        return self._copy_obs(self._current_obs)

    def get_obs_agent(self, agent_id):
        return np.array(self._current_obs[agent_id], copy=True)

    def get_obs_delay(self):
        return self._current_delays.copy()

    def get_obs_generation_time(self):
        return self._current_generation_times.copy()

    def get_obs_size(self):
        return self.env.get_obs_size()

    def get_state(self):
        return self.env.get_state()

    def get_state_size(self):
        return self.env.get_state_size()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_avail_agent_actions(self, agent_id):
        return self.env.get_avail_agent_actions(agent_id)

    def get_total_actions(self):
        return self.env.get_total_actions()

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)
        return self.env.seed(seed)

    def save_replay(self):
        return self.env.save_replay()

    def get_env_info(self):
        return self.env.get_env_info()

    def get_stats(self):
        return self.env.get_stats()

    def _refresh_delayed_obs(self):
        delays = self._sample_delays()
        generation_times = np.maximum(self._time - delays, 0)
        self._current_delays = self._time - generation_times
        self._current_generation_times = generation_times
        self._current_obs = [
            np.array(self._obs_history[generation_times[agent_id]][agent_id], copy=True)
            for agent_id in range(self.n_agents)
        ]

    def _sample_delays(self):
        sample_size = self.n_agents if self.per_agent else 1
        if self.delay_type == "fixed":
            sampled = np.full(sample_size, self.delay, dtype=np.int64)
        elif self.delay_type == "uniform":
            sampled = self._rng.integers(0, self.max_delay + 1, size=sample_size, dtype=np.int64)
        else:
            sampled = np.ceil(
                self._rng.normal(self.delay_mean, self.delay_std, size=sample_size)
            ).astype(np.int64)
            sampled = np.clip(sampled, 0, self.max_delay)

        if not self.per_agent:
            sampled = np.repeat(sampled, self.n_agents)
        return sampled

    @staticmethod
    def _copy_obs(obs: Any):
        return [np.array(agent_obs, copy=True) for agent_obs in copy.copy(obs)]
