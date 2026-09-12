"""Bring-up check for delayed_mpe: train = fresh obs, eval = arrival-time delay."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np


def _make(**delay):
    from utils.maker import EnvMaker

    kwargs = dict(
        scenario="simple_spread_v3",
        time_limit=25,
        seed=0,
        common_reward=True,
        reward_scalarisation="sum",
        delay_type="gaussian",
        delay_mean=2.0,
        delay_std=0.0,
        max_delay=2,
        delay_per_agent=True,
        args=object(),
    )
    kwargs.update(delay)
    return EnvMaker.make_delayed_mpe(**kwargs)


def test_train_is_zero_delay():
    env = _make()
    env.training = True
    obs, _ = env.reset(seed=0)
    fresh = env.env.get_obs()
    assert all(np.allclose(a, b) for a, b in zip(obs, fresh))
    actions = np.zeros(3, dtype=np.int64)
    for _ in range(5):
        obs, _, terminated, truncated, _ = env.step(actions)
        fresh = env.env.get_obs()
        assert all(np.allclose(a, b) for a, b in zip(obs, fresh))
        assert np.all(env.get_obs_delay() == 0)
        if terminated or truncated:
            break
    env.close()


def test_eval_fixed_delay_two():
    env = _make()
    env.training = False
    delivered, _ = env.reset(seed=1)
    t0 = [np.array(o, copy=True) for o in env.env.get_obs()]
    assert all(np.allclose(o, 0.0) for o in delivered), "t=0 packet has not arrived yet"
    assert np.all(env.get_obs_generation_time() == -1)

    actions = np.ones(3, dtype=np.int64)  # move so consecutive obs actually differ
    history = [t0]
    for step in range(1, 5):
        delivered, _, terminated, truncated, _ = env.step(actions)
        history.append([np.array(o, copy=True) for o in env.env.get_obs()])
        if step < 2:
            assert all(np.allclose(o, 0.0) for o in delivered)
            assert np.all(env.get_obs_generation_time() == -1)
        else:
            expected = history[step - 2]
            assert all(
                np.allclose(a, b) for a, b in zip(delivered, expected)
            ), f"at t={step} expected obs from t={step-2}"
            assert np.all(env.get_obs_generation_time() == step - 2)
            assert np.all(env.get_obs_delay() == 2)
            fresh = env.env.get_obs()
            assert not all(np.allclose(a, b) for a, b in zip(delivered, fresh))
        if terminated or truncated:
            break
    env.close()


if __name__ == "__main__":
    test_train_is_zero_delay()
    test_eval_fixed_delay_two()
    print("delayed_mpe checks passed")
