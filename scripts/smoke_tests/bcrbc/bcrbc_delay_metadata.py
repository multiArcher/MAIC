"""Smoke: obs delay metadata must NOT affect BC-RBC Q values.

In the v2 design delay is never an input to the network — it is resolved in
controller code (latent imputation + correction-on-arrival). This test pins that
guarantee: two batches that differ ONLY in the obs delay metadata
(obs_delay / obs_gen_t / obs_fresh_mask), with identical obs/actions, must produce
identical Q values through the plain (non-generative) forward path.
"""

import sys

import torch

sys.path.insert(0, "src")

from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers.bcrbc_mac import BCRBCMAC

from scripts.smoke_tests.bcrbc._bcrbc_args import make_bcrbc_args


BATCH_SIZE = 1
TIME_SIZE = 4
N_AGENTS = 3
OBS_DIM = 5
N_ACTIONS = 4
STATE_DIM = 6

scheme = {
    "state": {"vshape": STATE_DIM, "dtype": torch.float32},
    "obs": {"vshape": OBS_DIM, "group": "agents", "dtype": torch.float32},
    "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
    "avail_actions": {"vshape": (N_ACTIONS,), "group": "agents", "dtype": torch.int},
    "obs_delay": {"vshape": (1,), "group": "agents", "dtype": torch.long},
    "obs_gen_t": {"vshape": (1,), "group": "agents", "dtype": torch.long},
    "obs_fresh_mask": {"vshape": (1,), "group": "agents", "dtype": torch.float32},
    "reward": {"vshape": (1,), "dtype": torch.float32},
    "terminated": {"vshape": (1,), "dtype": torch.uint8},
}
groups = {"agents": N_AGENTS}
preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=N_ACTIONS)])}

# Comm off: isolate the obs path so the only difference between batches is the
# (now-ignored) obs delay metadata.
args = make_bcrbc_args(
    n_agents=N_AGENTS,
    n_actions=N_ACTIONS,
    epsilon_start=0.0,
    epsilon_finish=0.0,
    epsilon_anneal_time=1,
    bcrbc_max_delay=8,
    bcrbc_use_comm=False,
    env_info={"episode_limit": TIME_SIZE - 1},
)


def make_batch(delay_value):
    batch = EpisodeBatch(scheme, groups, BATCH_SIZE, TIME_SIZE, preprocess=preprocess, device="cpu")
    obs = torch.ones(BATCH_SIZE, TIME_SIZE, N_AGENTS, OBS_DIM)
    actions = torch.zeros(BATCH_SIZE, TIME_SIZE, N_AGENTS, 1, dtype=torch.long)
    time_ids = torch.arange(TIME_SIZE).view(1, TIME_SIZE, 1, 1).expand(BATCH_SIZE, TIME_SIZE, N_AGENTS, 1)
    delay = torch.full((BATCH_SIZE, TIME_SIZE, N_AGENTS, 1), delay_value, dtype=torch.long)
    gen_t = (time_ids - delay).clamp(min=0)
    fresh = (delay == 0).float()
    batch.update(
        {
            "state": torch.zeros(BATCH_SIZE, TIME_SIZE, STATE_DIM),
            "obs": obs,
            "actions": actions,
            "avail_actions": torch.ones(BATCH_SIZE, TIME_SIZE, N_AGENTS, N_ACTIONS, dtype=torch.int),
            "obs_delay": delay,
            "obs_gen_t": gen_t,
            "obs_fresh_mask": fresh,
            "reward": torch.zeros(BATCH_SIZE, TIME_SIZE, 1),
            "terminated": torch.zeros(BATCH_SIZE, TIME_SIZE, 1, dtype=torch.uint8),
        },
        slice(None),
        slice(0, TIME_SIZE),
    )
    return batch


torch.manual_seed(7)
mac = BCRBCMAC(scheme | {"actions_onehot": {"vshape": (N_ACTIONS,), "group": "agents", "dtype": torch.float32}}, groups, args)
mac.eval()
with torch.no_grad():
    # Plain forward path (test_mode False -> no generative rollout); only the obs
    # delay metadata differs between the two batches.
    q_no_delay = mac.forward(make_batch(0), slice(0, TIME_SIZE))["q_values"]
    q_delayed = mac.forward(make_batch(3), slice(0, TIME_SIZE))["q_values"]

assert torch.allclose(q_no_delay, q_delayed), \
    "obs delay metadata must NOT affect Q values (delay is not a model input)"
print("delay metadata does not leak into the model ok")
