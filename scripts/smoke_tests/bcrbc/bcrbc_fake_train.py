import sys

import torch

sys.path.insert(0, "src")

from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers.bcrbc_mac import BCRBCMAC
from learners.bcrbc_learner import BCRBCLearner

from scripts.smoke_tests.bcrbc._bcrbc_args import make_bcrbc_args, delay_scheme, zero_delay_fill


class Logger:
    def info(self, *args, **kwargs):
        pass


BATCH_SIZE = 2
TIME_SIZE = 5
N_AGENTS = 3
OBS_DIM = 7
N_ACTIONS = 4
STATE_DIM = 9

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

batch = EpisodeBatch(scheme, groups, BATCH_SIZE, TIME_SIZE, preprocess=preprocess, device="cpu")
batch.update(
    {
        "state": torch.randn(BATCH_SIZE, TIME_SIZE, STATE_DIM),
        "obs": torch.randn(BATCH_SIZE, TIME_SIZE, N_AGENTS, OBS_DIM),
        "actions": torch.randint(0, N_ACTIONS, (BATCH_SIZE, TIME_SIZE, N_AGENTS, 1)),
        "avail_actions": torch.ones(BATCH_SIZE, TIME_SIZE, N_AGENTS, N_ACTIONS, dtype=torch.int),
        "reward": torch.randn(BATCH_SIZE, TIME_SIZE, 1),
        "terminated": torch.zeros(BATCH_SIZE, TIME_SIZE, 1, dtype=torch.uint8),
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
)

mac = BCRBCMAC(batch.scheme, groups, args)
learner = BCRBCLearner(mac, batch.scheme, Logger(), args)

out = mac.forward(batch, slice(0, TIME_SIZE))
assert out["q_values"].shape == (BATCH_SIZE, TIME_SIZE, N_AGENTS, 1, N_ACTIONS)
assert torch.isfinite(out["q_values"]).all()
learner.train(batch, 0, 0)
print("bcrbc fake train ok")

