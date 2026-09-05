import sys

import torch

sys.path.insert(0, "src")

from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers.bcrbc_mac import BCRBCMAC
from learners.bcrbc_learner import BCRBCLearner
from modules.bcrbc.comm_delay import CommDelay

from scripts.smoke_tests.bcrbc._bcrbc_args import make_bcrbc_args, delay_scheme, zero_delay_fill


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
    bcrbc_use_comm=True,
    comm_gaussian_delay_mean=2.0,
    comm_gaussian_delay_std=1.5,
    bcrbc_max_delay=8,
    env_info={"episode_limit": TIME_SIZE - 1},
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

# message slot count: n_agents-1 senders per agent. Training delivers each
# sender's current-step observation (delay 0), so messages[b,t,i,k] == obs of the
# k-th sender of agent i at step t.
comm = mac.comm_delay(batch["obs"][:, slice(0, TIME_SIZE)], start_t=0, training=True)
assert comm.shape == (BATCH_SIZE, TIME_SIZE, N_AGENTS, N_AGENTS - 1, OBS_DIM)
sender_idx = torch.tensor([[j for j in range(N_AGENTS) if j != i] for i in range(N_AGENTS)])
expected_fresh = batch["obs"][:, slice(0, TIME_SIZE)][:, :, sender_idx]  # [B,T,n,n-1,obs]
assert torch.allclose(comm, expected_fresh), "training comm must deliver current-step sender obs"

# eval: Gaussian delay -> stale content (taken from an earlier step), so at least
# some message slots differ from the fresh (current-step) delivery.
comm_eval = mac.comm_delay(batch["obs"][:, slice(0, TIME_SIZE)], start_t=0, training=False)
assert comm_eval.shape == comm.shape
assert not torch.allclose(comm_eval, expected_fresh), "eval comm should deliver stale content"

# full learner train step with comm enabled
learner = BCRBCLearner(mac, batch.scheme, Logger(), args)
learner.train(batch, 0, 0)

# A communication packet samples its delay once. With a deterministic one-step
# delay, the packet sent at k=0 is absent at k=0 and arrives at k=1 even though a
# newer packet is sent at k=1.
channel = CommDelay(
    N_AGENTS,
    delay_mean=1.0,
    delay_std=0.0,
    max_delay=1,
)
channel.reset()
obs_k0 = batch["obs"][:, 0]
obs_k1 = batch["obs"][:, 1]
messages_k0 = channel.push_and_query(
    obs_k0,
    t=0,
    training=False,
    max_t=TIME_SIZE,
)
messages_k1 = channel.push_and_query(
    obs_k1,
    t=1,
    training=False,
    max_t=TIME_SIZE,
)
expected_k1 = obs_k0[:, channel.sender_idx]
assert torch.count_nonzero(messages_k0) == 0
assert torch.allclose(messages_k1, expected_k1)

# The MAC keeps the delivered online messages used by generative evaluation.
online_args = make_bcrbc_args(**vars(args))
online_args.bcrbc_generative_eval = True
online_args.comm_gaussian_delay_mean = 1.0
online_args.comm_gaussian_delay_std = 0.0
online_args.bcrbc_max_delay = 1
online_mac = BCRBCMAC(batch.scheme, groups, online_args)
online_mac.init_hidden(BATCH_SIZE)
online_mac.forward(batch, 0, test_mode=True)
online_mac.forward(batch, 1, test_mode=True)
assert torch.allclose(online_mac._online_message_history[:, 1], expected_k1)

message_history = online_mac._online_message_history.clone()
online_mac.forward(batch, 1, test_mode=True)
assert torch.equal(online_mac._online_message_history, message_history)

online_mac.init_hidden(BATCH_SIZE)
assert online_mac._online_message_history is None
assert online_mac._online_message_t == -1
online_mac.forward(batch, 0, test_mode=True)
assert torch.count_nonzero(online_mac._online_message_history[:, 0]) == 0

# Overriding the messages delivered at one step must not drop that step's
# outgoing observation packet from the communication channel.
override_mac = BCRBCMAC(batch.scheme, groups, online_args)
override_mac.init_hidden(BATCH_SIZE)
override_mac.forward(batch, 0, test_mode=True)
override_messages = torch.zeros_like(expected_k1)
override_mac.forward(
    batch,
    1,
    test_mode=True,
    messages=override_messages.unsqueeze(1),
)
override_mac.forward(batch, 2, test_mode=True)
expected_k2 = batch["obs"][:, 1, override_mac.comm_delay.sender_idx]
assert torch.allclose(
    override_mac._online_message_history[:, 2],
    expected_k2,
)

# A slice query advances every requested environment step, not only its start.
slice_mac = BCRBCMAC(batch.scheme, groups, online_args)
slice_mac.init_hidden(BATCH_SIZE)
slice_mac.forward(batch, slice(0, 3), test_mode=True)
assert slice_mac._online_message_t == 2
assert torch.allclose(
    slice_mac._online_message_history[:, 2],
    expected_k2,
)

print("bcrbc comm delay ok")
