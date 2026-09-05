"""Phase E smoke: auxiliary losses active in a single train step.

Enables the auxiliary loss weights (rec, dyn, retro, msg_rec) together with the
communication pathway, runs one BCRBCLearner.train step, and asserts the model
outputs have the right shape and every loss term stays finite.
"""

import sys

import torch

sys.path.insert(0, "src")

from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers.bcrbc_mac import BCRBCMAC
from learners.bcrbc_learner import BCRBCLearner
from modules.bcrbc.losses import message_reconstruction_loss

from scripts.smoke_tests.bcrbc._bcrbc_args import make_bcrbc_args, delay_scheme


class Logger:
    def __init__(self):
        self.stats = {}

    def info(self, *args, **kwargs):
        pass

    def log_stat(self, key, value, step):
        self.stats[key] = value


BATCH_SIZE = 2
TIME_SIZE = 6
N_AGENTS = 3
OBS_DIM = 7
N_ACTIONS = 4
STATE_DIM = 9
MAX_DELAY = 4

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


def make_batch():
    batch = EpisodeBatch(scheme, groups, BATCH_SIZE, TIME_SIZE, preprocess=preprocess, device="cpu")
    time_ids = torch.arange(TIME_SIZE).view(1, TIME_SIZE, 1, 1).expand(BATCH_SIZE, TIME_SIZE, N_AGENTS, 1)
    delay = torch.ones(BATCH_SIZE, TIME_SIZE, N_AGENTS, 1, dtype=torch.long)
    delay[:, 0] = 0
    batch.update(
        {
            "state": torch.randn(BATCH_SIZE, TIME_SIZE, STATE_DIM),
            "obs": torch.randn(BATCH_SIZE, TIME_SIZE, N_AGENTS, OBS_DIM),
            "actions": torch.randint(0, N_ACTIONS, (BATCH_SIZE, TIME_SIZE, N_AGENTS, 1)),
            "avail_actions": torch.ones(BATCH_SIZE, TIME_SIZE, N_AGENTS, N_ACTIONS, dtype=torch.int),
            "obs_delay": delay,
            "obs_gen_t": (time_ids - delay).clamp(min=0),
            "obs_fresh_mask": (delay == 0).float(),
            "reward": torch.randn(BATCH_SIZE, TIME_SIZE, 1),
            "terminated": torch.zeros(BATCH_SIZE, TIME_SIZE, 1, dtype=torch.uint8),
        },
        slice(None),
        slice(0, TIME_SIZE),
    )
    return batch


args = make_bcrbc_args(
    n_agents=N_AGENTS,
    n_actions=N_ACTIONS,
    state_shape=STATE_DIM,
    bcrbc_use_comm=True,
    comm_gaussian_delay_mean=2.0,
    comm_gaussian_delay_std=1.5,
    bcrbc_max_delay=MAX_DELAY,
    env_info={"episode_limit": TIME_SIZE - 1},
    # auxiliary losses ON
    rec_loss_weight=0.5,
    flow_loss_weight=0.5,
    retro_loss_weight=0.1,
    bcrbc_retro_max_replay_len=TIME_SIZE,
    msg_rec_loss_weight=0.5,
)


torch.manual_seed(7)
batch = make_batch()
mac = BCRBCMAC(batch.scheme, groups, args)

# Model outputs reconstructed messages with the vector-axis shape [...,n,1,d].
out = mac.forward(batch, slice(0, TIME_SIZE))
assert out["reconstructed_messages"].shape == (
    BATCH_SIZE,
    TIME_SIZE,
    N_AGENTS,
    1,
    OBS_DIM,
)

# standalone loss-function sanity (finite, non-negative). Vector-axis convention:
# Reconstructed messages [b,t,n,1,d], targets [b,t,n,n-1,d].
mask = torch.ones(BATCH_SIZE, TIME_SIZE, N_AGENTS, 1, 1)
teacher_msgs = mac.comm_delay(batch["obs"][:, slice(0, TIME_SIZE)], start_t=0, training=True)
mrl = message_reconstruction_loss(out["reconstructed_messages"], teacher_msgs, mask)
assert torch.isfinite(mrl) and mrl.item() >= 0.0

# single-agent message rec must be zero (no senders)
zero_msg = torch.zeros(BATCH_SIZE, TIME_SIZE, 1, 0, OBS_DIM)
zr = message_reconstruction_loss(
    out["reconstructed_messages"][:, :, :1],
    zero_msg,
    mask[:, :, :1],
)
assert zr.item() == 0.0, "message rec with no senders must be exactly zero"

# full learner train step with every aux loss enabled; all params should get grad
logger = Logger()
learner = BCRBCLearner(mac, batch.scheme, logger, args)
learner.train(batch, 0, 0)
assert learner.logger is logger
for loss_name in ("td_loss", "rec_loss", "flow_loss", "msg_rec_loss", "retro_loss", "total_loss"):
    assert torch.isfinite(torch.tensor(logger.stats[f"loss/{loss_name}"]))
assert mac.agent.message_decoder[1].weight.grad is not None
assert torch.isfinite(mac.agent.message_decoder[1].weight.grad).all()

print("bcrbc aux losses ok")
