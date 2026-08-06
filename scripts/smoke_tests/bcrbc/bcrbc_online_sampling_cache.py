"""Training rollout must retain transformer history between sampled actions."""

import sys

import torch

sys.path.insert(0, "src")

from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers.bcrbc_mac import BCRBCMAC
from scripts.smoke_tests.bcrbc._bcrbc_args import (
    delay_scheme,
    make_bcrbc_args,
    zero_delay_fill,
)


BATCH_SIZE = 1
TIME_SIZE = 4
N_AGENTS = 3
OBS_DIM = 6
N_ACTIONS = 4
STATE_DIM = 8
CONTEXT_WINDOW = 2

scheme = {
    "state": {"vshape": STATE_DIM, "dtype": torch.float32},
    "obs": {"vshape": OBS_DIM, "group": "agents", "dtype": torch.float32},
    "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
    "avail_actions": {"vshape": (N_ACTIONS,), "group": "agents", "dtype": torch.int},
    "reward": {"vshape": (1,), "dtype": torch.float32},
    "terminated": {"vshape": (1,), "dtype": torch.uint8},
    **delay_scheme(),
}
groups = {"agents": N_AGENTS}
preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=N_ACTIONS)])}

batch = EpisodeBatch(
    scheme,
    groups,
    BATCH_SIZE,
    TIME_SIZE,
    preprocess=preprocess,
    device="cpu",
)
batch.update(
    {
        "state": torch.randn(BATCH_SIZE, TIME_SIZE, STATE_DIM),
        "obs": torch.randn(BATCH_SIZE, TIME_SIZE, N_AGENTS, OBS_DIM),
        "actions": torch.randint(
            0, N_ACTIONS, (BATCH_SIZE, TIME_SIZE, N_AGENTS, 1)
        ),
        "avail_actions": torch.ones(
            BATCH_SIZE, TIME_SIZE, N_AGENTS, N_ACTIONS, dtype=torch.int
        ),
        "reward": torch.zeros(BATCH_SIZE, TIME_SIZE, 1),
        "terminated": torch.zeros(
            BATCH_SIZE, TIME_SIZE, 1, dtype=torch.uint8
        ),
        **zero_delay_fill(BATCH_SIZE, TIME_SIZE, N_AGENTS),
    },
    slice(None),
    slice(0, TIME_SIZE),
)

args = make_bcrbc_args(
    n_agents=N_AGENTS,
    n_actions=N_ACTIONS,
    state_shape=STATE_DIM,
    env_info={"episode_limit": TIME_SIZE - 1},
    bcrbc_depth=4,
    bcrbc_context_window=CONTEXT_WINDOW,
    bcrbc_use_comm=False,
)
mac = BCRBCMAC(batch.scheme, groups, args)
mac.agent.eval()
mac.init_hidden(BATCH_SIZE)

for t in range(TIME_SIZE):
    incremental = mac.forward(batch, t, incremental=True)["q_values"]
    dense = mac.forward(batch, slice(0, t + 1), test_mode=True)["q_values"][:, -1:]
    assert torch.allclose(incremental, dense, atol=1e-5), (
        f"incremental rollout differs from dense prefix at t={t}"
    )

    cache_lengths = [key.shape[-2] for key, _ in mac._online_kv_cache]
    assert cache_lengths == [min(t + 1, CONTEXT_WINDOW)]

mac.init_hidden(BATCH_SIZE)
assert mac._online_kv_cache is None
mac.select_actions(batch, t_ep=0, t_env=0, test_mode=False)
assert mac._online_kv_cache is not None

print("bcrbc online sampling cache ok")
