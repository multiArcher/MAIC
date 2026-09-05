"""Phase 4 smoke: multi-token latent convention [b, t, n, num_token, d].

Runs the full DCRBC stack with bcrbc_num_z_tokens > 1 and asserts the token
axis is carried natively end-to-end: encoder emits num_token latents per agent,
forward returns z of shape [B, T, n, num_token, z_dim], a learner train step with
flow loss completes, and the generative eval rollout runs. Also re-confirms
num_token=1 still produces a size-1 token axis (degenerate case).
"""

import sys

import torch

sys.path.insert(0, "src")

from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers.bcrbc_mac import BCRBCMAC
from learners.bcrbc_learner import BCRBCLearner

from scripts.smoke_tests.bcrbc._bcrbc_args import make_bcrbc_args


class Logger:
    def info(self, *args, **kwargs):
        pass

    def log_stat(self, key, value, step):
        pass


BATCH, TIME, NA, OBS, NACT, STATE = 2, 6, 3, 7, 4, 9
Z_DIM = 8


def build_batch_and_args(num_z_tokens, generative_eval=False):
    scheme = {
        "state": {"vshape": STATE, "dtype": torch.float32},
        "obs": {"vshape": OBS, "group": "agents", "dtype": torch.float32},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {"vshape": (NACT,), "group": "agents", "dtype": torch.int},
        "obs_delay": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "obs_gen_t": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "obs_fresh_mask": {"vshape": (1,), "group": "agents", "dtype": torch.float32},
        "reward": {"vshape": (1,), "dtype": torch.float32},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
    }
    groups = {"agents": NA}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=NACT)])}
    batch = EpisodeBatch(scheme, groups, BATCH, TIME, preprocess=preprocess, device="cpu")
    time_ids = torch.arange(TIME).view(1, TIME, 1, 1).expand(BATCH, TIME, NA, 1)
    delay = torch.ones(BATCH, TIME, NA, 1, dtype=torch.long)
    delay[:, 0] = 0
    batch.update(
        {
            "state": torch.randn(BATCH, TIME, STATE),
            "obs": torch.randn(BATCH, TIME, NA, OBS),
            "actions": torch.randint(0, NACT, (BATCH, TIME, NA, 1)),
            "avail_actions": torch.ones(BATCH, TIME, NA, NACT, dtype=torch.int),
            "obs_delay": delay,
            "obs_gen_t": (time_ids - delay).clamp(min=0),
            "obs_fresh_mask": (delay == 0).float(),
            "reward": torch.randn(BATCH, TIME, 1),
            "terminated": torch.zeros(BATCH, TIME, 1, dtype=torch.uint8),
        },
        slice(None),
        slice(0, TIME),
    )
    args = make_bcrbc_args(
        n_agents=NA, n_actions=NACT, state_shape=STATE,
        bcrbc_z_dim=Z_DIM, bcrbc_num_z_tokens=num_z_tokens,
        env_info={"episode_limit": TIME - 1},
        rec_loss_weight=0.5, flow_loss_weight=0.5,
        bcrbc_generative_eval=generative_eval, bcrbc_flow_steps=4,
    )
    return batch, groups, args


# ---- num_token > 1: token axis carried natively end-to-end ------------------
NTOK = 4
torch.manual_seed(0)
batch, groups, args = build_batch_and_args(NTOK)
mac = BCRBCMAC(batch.scheme, groups, args)
out = mac.forward(batch, slice(0, TIME))
assert out["z"].shape == (BATCH, TIME, NA, NTOK, Z_DIM), f"z shape {out['z'].shape}"
assert out["q_values"].shape == (BATCH, TIME, NA, 1, NACT), f"q shape {out['q_values'].shape}"
assert out["reconstructed_observations"].shape == (BATCH, TIME, NA, OBS)
assert torch.isfinite(out["q_values"]).all()

learner = BCRBCLearner(mac, batch.scheme, Logger(), args)
learner.train(batch, 0, 0)
assert mac.agent.z_predictor.weight.grad is not None
assert mac.agent.dynamics_tokenizer.signal_projection.weight.grad is not None

# generative eval rollout with the token axis
torch.manual_seed(1)
batch_g, groups_g, args_g = build_batch_and_args(NTOK, generative_eval=True)
mac_g = BCRBCMAC(batch_g.scheme, groups_g, args_g)
for t in range(TIME):
    o = mac_g.forward(batch_g, t, test_mode=True)
    assert torch.isfinite(o["q_values"]).all()
    assert o["q_values"].shape == (BATCH, 1, NA, 1, NACT)

# ---- num_token == 1: degenerate size-1 token axis ---------------------------
torch.manual_seed(2)
batch1, groups1, args1 = build_batch_and_args(1)
mac1 = BCRBCMAC(batch1.scheme, groups1, args1)
out1 = mac1.forward(batch1, slice(0, TIME))
assert out1["z"].shape == (BATCH, TIME, NA, 1, Z_DIM), f"z1 shape {out1['z'].shape}"

print("bcrbc multi-token ok")
