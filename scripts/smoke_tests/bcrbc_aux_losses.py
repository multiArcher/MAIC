"""Phase E smoke: all auxiliary losses active in a single train step.

Enables every auxiliary loss weight (rec, dyn, retro, msg_rec, arr) together with
the communication pathway and delay metadata, runs one BCRBCLearner.train step,
and asserts the new model outputs (delay_logits, recon_msg) have the right shape
and every loss term stays finite.
"""

import sys
from types import SimpleNamespace as SN

import torch

sys.path.insert(0, "src")

from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers.bcrbc_mac import BCRBCMAC
from learners.bcrbc_learner import BCRBCLearner
from modules.bcrbc.losses import arrival_loss, message_reconstruction_loss


class Logger:
    def info(self, *args, **kwargs):
        pass


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
    "obs_delay": {"vshape": (1,), "group": "agents", "dtype": torch.long},
    "obs_gen_t": {"vshape": (1,), "group": "agents", "dtype": torch.long},
    "obs_fresh_mask": {"vshape": (1,), "group": "agents", "dtype": torch.float32},
    "reward": {"vshape": (1,), "dtype": torch.float32},
    "terminated": {"vshape": (1,), "dtype": torch.uint8},
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


args = SN(
    device=torch.device("cpu"),
    use_cuda=False,
    n_agents=N_AGENTS,
    n_actions=N_ACTIONS,
    state_shape=STATE_DIM,
    common_reward=True,
    agent_output_type="q",
    action_selector="epsilon_greedy",
    epsilon_start=1.0,
    epsilon_finish=0.05,
    epsilon_anneal_time=100,
    evaluation_epsilon=0.0,
    obs_agent_id=True,
    obs_last_action=True,
    bcrbc_d_model=32,
    bcrbc_belief_dim=32,
    bcrbc_depth=1,
    bcrbc_heads=4,
    bcrbc_dropout=0.0,
    bcrbc_use_comm=True,
    comm_gaussian_delay_mean=2.0,
    comm_gaussian_delay_std=1.5,
    bcrbc_max_delay=MAX_DELAY,
    env_info={"episode_limit": TIME_SIZE - 1},
    mixer="new_qmix_mixer",
    mixing_embed_dim=8,
    hypernet_embed=16,
    optimizer="adamW",
    lr=0.001,
    standardise_returns=False,
    standardise_rewards=False,
    target_type="td",
    gamma=0.99,
    double_q=True,
    target_update_interval_or_tau=200,
    learner_log_interval=999999,
    grad_norm_clip=10,
    # all auxiliary losses ON
    td_loss_weight=1.0,
    rec_loss_weight=0.5,
    dyn_loss_weight=0.5,
    retro_loss_weight=0.1,
    bcrbc_retro_max_replay_len=TIME_SIZE,
    msg_rec_loss_weight=0.5,
    arr_loss_weight=0.5,
)


torch.manual_seed(7)
batch = make_batch()
mac = BCRBCMAC(batch.scheme, groups, args)

# model outputs the new heads with the right shapes (vector axis [...,n,1,*])
out = mac.forward(batch, slice(0, TIME_SIZE))
assert out["delay_logits"].shape == (BATCH_SIZE, TIME_SIZE, N_AGENTS, 1, MAX_DELAY + 1), \
    f"delay_logits shape {out['delay_logits'].shape}"
assert out["recon_msg"].shape == (BATCH_SIZE, TIME_SIZE, N_AGENTS, 1, OBS_DIM), \
    f"recon_msg shape {out['recon_msg'].shape}"

# standalone loss-function sanity (finite, non-negative). Vector-axis convention:
# recon_msg [b,t,n,1,d], teacher messages [b,t,n,n-1,d], delay target [b,t,n,1,1],
# mask [b,t,n,1,1].
mask = torch.ones(BATCH_SIZE, TIME_SIZE, N_AGENTS, 1, 1)
teacher_msgs = mac.comm_delay(batch["obs"][:, slice(0, TIME_SIZE)], start_t=0, training=True)["messages"]
mrl = message_reconstruction_loss(out["recon_msg"], teacher_msgs, mask)
al = arrival_loss(out["delay_logits"], batch["obs_delay"].unsqueeze(-2), mask)
assert torch.isfinite(mrl) and mrl.item() >= 0.0
assert torch.isfinite(al) and al.item() >= 0.0

# single-agent message rec must be zero (no senders)
zero_msg = torch.zeros(BATCH_SIZE, TIME_SIZE, 1, 0, OBS_DIM)
zr = message_reconstruction_loss(out["recon_msg"][:, :, :1], zero_msg, mask[:, :, :1])
assert zr.item() == 0.0, "message rec with no senders must be exactly zero"

# full learner train step with every aux loss enabled
learner = BCRBCLearner(mac, batch.scheme, Logger(), args)
learner.train(batch, 0, 0)

# grads flow into the new arrival head
assert mac.agent.arr_head.weight.grad is not None, "arr_head should receive gradient"
assert torch.isfinite(mac.agent.arr_head.weight.grad).all()

print("bcrbc aux losses ok")
