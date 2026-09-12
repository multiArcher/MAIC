"""simple_spread_v3 (mpe2) → EPyMARL MultiAgentEnv.

First-wave MPE support: three cooperative agents, vector obs, discrete
actions. Gymnasium registration and other PettingZoo families are out of
scope. pygame's crash parachute is uninstalled after import so later CUDA
optimizer setup is not intercepted.
"""

from collections.abc import Mapping
import os
import signal
import warnings

import numpy as np
import torch
from gymnasium.spaces import Discrete

from .multiagentenv import MultiAgentEnv

_ALLOWED_SCENARIOS = {"simple_spread_v3"}
_CRASH_SIGNALS = ("SIGSEGV", "SIGBUS", "SIGFPE", "SIGABRT")


def _restore_default_crash_signals():
    for name in _CRASH_SIGNALS:
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, signal.SIG_DFL)
        except (ValueError, OSError):
            pass


def _disable_torch_dynamo():
    try:
        torch._dynamo.config.disable = True
    except Exception:
        pass


def _space_of(env, kind, agent):
    getter = getattr(env, f"{kind}_space", None)
    if callable(getter):
        return getter(agent)
    spaces = getattr(env, f"{kind}_spaces", None)
    if isinstance(spaces, Mapping) and agent in spaces:
        return spaces[agent]
    raise AttributeError(f"MPE env has no {kind} space for {agent!r}")


def _discrete_n(space):
    if not isinstance(space, Discrete):
        raise TypeError(f"MPEEnv expects Discrete actions, got {type(space)!r}")
    return int(space.n)


def _load_simple_spread(max_cycles):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    try:
        from mpe2 import simple_spread_v3
    except ImportError as exc:
        raise ImportError(
            "MPE simple_spread requires the mpe2 package. Install with `pip install mpe2`."
        ) from exc
    env = simple_spread_v3.parallel_env(max_cycles=int(max_cycles))
    _restore_default_crash_signals()
    _disable_torch_dynamo()
    return env


class MPEEnv(MultiAgentEnv):
    def __init__(
        self,
        scenario,
        time_limit,
        seed,
        common_reward,
        reward_scalarisation,
        **kwargs,
    ):
        kwargs.pop("args", None)
        del kwargs

        scenario = str(scenario)
        if scenario not in _ALLOWED_SCENARIOS:
            raise ValueError(
                f"Unsupported MPE scenario {scenario!r}. "
                f"This build only supports {sorted(_ALLOWED_SCENARIOS)}."
            )

        self.scenario = scenario
        self.episode_limit = int(time_limit)
        self._seed = seed
        self._env = _load_simple_spread(self.episode_limit)

        reset_kwargs = {}
        if seed is not None:
            reset_kwargs["seed"] = int(seed)
        boot_obs, _ = self._env.reset(**reset_kwargs)

        self.agent_ids = list(self._env.possible_agents)
        if not self.agent_ids:
            raise RuntimeError("simple_spread_v3 has no possible_agents")
        self.n_agents = len(self.agent_ids)

        self._action_n = {
            agent: _discrete_n(_space_of(self._env, "action", agent))
            for agent in self.agent_ids
        }
        self._n_actions = max(self._action_n.values())
        sample_obs = np.asarray(boot_obs[self.agent_ids[0]], dtype=np.float32).reshape(-1)
        self._obs_size = int(sample_obs.shape[0])

        self.common_reward = common_reward
        if self.common_reward:
            if reward_scalarisation == "sum":
                self._agg = lambda rewards: float(np.sum(rewards))
            elif reward_scalarisation == "mean":
                self._agg = lambda rewards: float(np.mean(rewards))
            else:
                raise ValueError(
                    f"Invalid reward_scalarisation: {reward_scalarisation} "
                    "(only support 'sum' or 'mean')"
                )
        else:
            self._agg = None

        self._episode_t = 0
        self._obs = []
        self._state = None
        self._cache(boot_obs)
        self._state_size = int(self._state.shape[0])

    def _obs_list(self, observations):
        out = []
        for agent in self.agent_ids:
            if agent in observations:
                vec = np.asarray(observations[agent], dtype=np.float32).reshape(-1)
            else:
                vec = np.zeros(self._obs_size, dtype=np.float32)
            if vec.shape[0] != self._obs_size:
                padded = np.zeros(self._obs_size, dtype=np.float32)
                padded[: min(self._obs_size, vec.shape[0])] = vec[: self._obs_size]
                vec = padded
            out.append(vec)
        return out

    def _extract_state(self, observations):
        state_fn = getattr(self._env, "state", None)
        if callable(state_fn):
            try:
                raw = state_fn()
                if raw is not None:
                    return np.asarray(raw, dtype=np.float32).reshape(-1)
            except Exception:
                pass
        return np.concatenate(self._obs_list(observations), axis=0).astype(np.float32)

    def _cache(self, observations):
        self._obs = self._obs_list(observations)
        self._state = self._extract_state(observations)

    def _to_int_actions(self, actions):
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        return [int(a) for a in np.asarray(actions).reshape(-1)]

    def step(self, actions):
        ints = self._to_int_actions(actions)
        if len(ints) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} actions, got {len(ints)}")
        action_dict = {}
        live = set(self._env.agents)
        for agent, action in zip(self.agent_ids, ints):
            if agent not in live:
                continue
            n = self._action_n[agent]
            action_dict[agent] = int(np.clip(action, 0, n - 1))
        if action_dict:
            observations, rewards, terminations, truncations, infos = self._env.step(
                action_dict
            )
        else:
            observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        self._episode_t += 1
        if observations:
            self._cache(observations)

        terminated = bool(terminations) and all(
            bool(terminations.get(agent, True)) for agent in self.agent_ids
        )
        env_truncated = bool(truncations) and all(
            bool(truncations.get(agent, True)) for agent in self.agent_ids
        )
        truncated = bool(env_truncated or self._episode_t >= self.episode_limit)
        info = {}
        # Do not set info["episode_limit"]. EpisodeRunner stores
        # terminated != episode_limit; SMAC default (continuing_episode=False)
        # and gymma TimeLimit both treat timeout as a true terminal so QMIX
        # does not bootstrap. Marking episode_limit here made every 25-step
        # MPE episode an infinite-horizon backup and the mixer Q exploded.
        if isinstance(infos, Mapping):
            for agent, payload in infos.items():
                if isinstance(payload, Mapping):
                    for key, value in payload.items():
                        info[f"{agent}_{key}"] = value

        agent_rewards = np.asarray(
            [float(rewards.get(agent, 0.0)) for agent in self.agent_ids],
            dtype=np.float32,
        )
        if self.common_reward:
            reward = self._agg(agent_rewards)
        else:
            if agent_rewards.size == 1:
                warnings.warn(
                    "common_reward is False but received a single agent reward, "
                    "returning reward as is"
                )
            reward = agent_rewards
        return self.get_obs(), reward, terminated, truncated, info

    def get_obs(self):
        return [np.array(obs, copy=True) for obs in self._obs]

    def get_obs_agent(self, agent_id):
        return np.array(self._obs[agent_id], copy=True)

    def get_obs_size(self):
        return self._obs_size

    def get_state(self):
        return np.array(self._state, copy=True)

    def get_state_size(self):
        return self._state_size

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        n = self._action_n[self.agent_ids[agent_id]]
        return [1] * n + [0] * (self._n_actions - n)

    def get_total_actions(self):
        return self._n_actions

    @property
    def episode_timestep(self):
        """Same clock DelayedObservationWrapper uses on SMAC (`_episode_steps`)."""
        return self._episode_t

    def reset(self, seed=None, options=None):
        del options
        # Match gymma/SMAC: only reseed when the caller passes seed.
        # Passing the constructor seed on every reset() made every episode
        # start from the same landmark layout (test_return_std stayed 0).
        reset_kwargs = {}
        if seed is not None:
            self._seed = seed
            reset_kwargs["seed"] = int(seed)
        observations, infos = self._env.reset(**reset_kwargs)
        self._episode_t = 0
        self._cache(observations)
        info = {}
        if isinstance(infos, Mapping):
            for agent, payload in infos.items():
                if isinstance(payload, Mapping):
                    for key, value in payload.items():
                        info[f"{agent}_{key}"] = value
        return self.get_obs(), info

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()

    def seed(self, seed=None):
        if seed is not None:
            self._seed = seed
            self.reset(seed=seed)
        return self._seed

    def save_replay(self):
        pass
