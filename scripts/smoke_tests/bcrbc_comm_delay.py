import sys
from types import SimpleNamespace as SN

import torch

sys.path.insert(0, "src")

from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers.bcrbc_mac import BCRBCMAC
from learners.bcrbc_learner import BCRBCLearner


class Logger:
    def info(self, *args, **kwargs):
        pass


BATCH_SIZE = 2
TIME_SIZE = 6
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
    },
    slice(None),
    slice(0, TIME_SIZE),
)

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
    bcrbc_max_delay=8,
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
    td_loss_weight=1.0,
    rec_loss_weight=0.0,
    dyn_loss_weight=0.0,
)

torch.manual_seed(0)
mac = BCRBCMAC(batch.scheme, groups, args)
assert mac.comm_delay is not None, "comm_delay should be built when bcrbc_use_comm=True"

# train forward: delay=0, all messages fresh
out_train = mac.forward(batch, slice(0, TIME_SIZE), test_mode=False)
assert out_train["q_values"].shape == (BATCH_SIZE, TIME_SIZE, N_AGENTS, 1, N_ACTIONS)
assert torch.isfinite(out_train["q_values"]).all()

# eval forward: Gaussian comm delay -> stale messages -> different q
torch.manual_seed(1)
out_eval = mac.forward(batch, slice(0, TIME_SIZE), test_mode=True)
assert torch.isfinite(out_eval["q_values"]).all()
assert not torch.allclose(out_train["q_values"], out_eval["q_values"]), \
    "comm delay at eval should change Q values vs no-delay train forward"

# message slot count: n_agents-1 senders per agent
comm = mac.comm_delay(batch["obs"][:, slice(0, TIME_SIZE)], start_t=0, training=True)
assert comm["messages"].shape == (BATCH_SIZE, TIME_SIZE, N_AGENTS, N_AGENTS - 1, OBS_DIM)
assert comm["msg_fresh_mask"].min().item() == 1.0, "training comm must be all-fresh"

comm_eval = mac.comm_delay(batch["obs"][:, slice(0, TIME_SIZE)], start_t=0, training=False)
assert comm_eval["msg_delay"].max().item() > 0, "eval comm should have nonzero delay"
# t-d<0 slots zero-filled
gen = comm_eval["msg_gen_t"]
arrive = comm_eval["msg_arrive_t"]
assert (gen <= arrive).all(), "generation time must not exceed arrival time"

# full learner train step with comm enabled
learner = BCRBCLearner(mac, batch.scheme, Logger(), args)
learner.train(batch, 0, 0)

print("bcrbc comm delay ok")
