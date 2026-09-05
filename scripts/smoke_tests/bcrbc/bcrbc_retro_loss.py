import sys

import torch

sys.path.insert(0, "src")

from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers.bcrbc_mac import BCRBCMAC
from learners.bcrbc_learner import BCRBCLearner
from modules.bcrbc.retro_replay import RetroReplay

from scripts.smoke_tests.bcrbc._bcrbc_args import make_bcrbc_args


class Logger:
    def info(self, *args, **kwargs):
        pass

    def log_stat(self, key, value, step):
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


args = make_bcrbc_args(
    n_agents=N_AGENTS,
    n_actions=N_ACTIONS,
    state_shape=STATE_DIM,
    bcrbc_max_delay=4,
    env_info={"episode_limit": TIME_SIZE - 1},
    retro_loss_weight=0.1,
    bcrbc_retro_max_replay_len=TIME_SIZE,
)


torch.manual_seed(11)
batch = make_batch()
mac = BCRBCMAC(batch.scheme, groups, args)
learner = BCRBCLearner(mac, batch.scheme, Logger(), args)

source_obs = torch.arange(1 * 3 * 2 * 1, dtype=torch.float32).view(1, 3, 2, 1)
gen_t = torch.tensor([[[[0], [0]], [[0], [1]], [[1], [0]]]])
patched = RetroReplay.patch_observations(source_obs, gen_t, start_t=0)
# Zero-fill semantics: unarrived generation slots are zeroed (constraint #2),
# not left as a stale carry-over. Later arrivals overwrite earlier ones.
expected = torch.zeros_like(source_obs)
expected[0, 0, 0] = source_obs[0, 1, 0]  # agent0 gen0: t0 then overwritten by t1
expected[0, 1, 0] = source_obs[0, 2, 0]  # agent0 gen1: from t2
expected[0, 0, 1] = source_obs[0, 2, 1]  # agent1 gen0: t0 then overwritten by t2
expected[0, 1, 1] = source_obs[0, 1, 1]  # agent1 gen1: from t1
assert torch.equal(patched, expected), "retro patch moves arrivals to gen slots, zero-fills unarrived"

with torch.no_grad():
    delayed_out = mac.forward(batch, slice(0, TIME_SIZE))
    retro = RetroReplay(max_replay_len=TIME_SIZE, use_comm=False, n_agents=N_AGENTS).compute(mac, batch, slice(0, TIME_SIZE), delayed_out)
    assert retro["corrected_agent_outputs"].shape == delayed_out["agent_outputs"].shape
    assert retro["target_agent_outputs"].shape == delayed_out["agent_outputs"].shape
    assert retro["retro_mask"].shape == (BATCH_SIZE, TIME_SIZE, N_AGENTS, 1)
    assert retro["retro_mask"].sum() > 0

learner.train(batch, 0, 0)
print("bcrbc retro loss ok")
