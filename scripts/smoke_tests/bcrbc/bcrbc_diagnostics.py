"""Generated-z diagnostics smoke test.

Builds a delayed episode + the matching full (un-delayed) observations and checks
that BCRBCMAC.z_diagnostics returns finite, well-formed Tier-1 metrics:
latent reconstruction error over generated slots, correction-after-arrival
improvement, and (with comm on) delayed-message recovery error.
"""

import sys

import torch

sys.path.insert(0, "src")

from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers.bcrbc_mac import BCRBCMAC

from scripts.smoke_tests.bcrbc._bcrbc_args import make_bcrbc_args

BATCH, TIME, NA, OBS, NACT, STATE = 2, 6, 3, 6, 4, 8

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

args = make_bcrbc_args(
    n_agents=NA, n_actions=NACT, state_shape=STATE,
    env_info={"episode_limit": TIME - 1},
    bcrbc_generative_eval=True, bcrbc_flow_steps=8,
    bcrbc_use_comm=True, comm_gaussian_delay_mean=1.0, comm_gaussian_delay_std=1.0, bcrbc_max_delay=8,
)

torch.manual_seed(0)

# Full (un-delayed) obs, and a delayed delivery: each step shows obs from t-1
# (delay 1) except t0; later arrivals fill earlier generation slots.
full_obs = torch.randn(BATCH, TIME, NA, OBS) * 2.0
gen = torch.zeros(BATCH, TIME, NA, 1, dtype=torch.long)
delay = torch.zeros(BATCH, TIME, NA, 1, dtype=torch.long)
fresh = torch.zeros(BATCH, TIME, NA, 1)
delivered = torch.zeros(BATCH, TIME, NA, OBS)
for t in range(TIME):
    g = max(0, t - 1)  # latest received gen-time
    gen[:, t] = g
    delay[:, t] = t - g
    fresh[:, t] = float(t == g)
    delivered[:, t] = full_obs[:, g]

batch = EpisodeBatch(scheme, groups, BATCH, TIME, preprocess=preprocess, device="cpu")
batch.update(
    {
        "state": torch.randn(BATCH, TIME, STATE),
        "obs": delivered,
        "actions": torch.randint(0, NACT, (BATCH, TIME, NA, 1)),
        "avail_actions": torch.ones(BATCH, TIME, NA, NACT, dtype=torch.int),
        "obs_gen_t": gen,
        "obs_delay": delay,
        "obs_fresh_mask": fresh,
        "reward": torch.randn(BATCH, TIME, 1),
        "terminated": torch.zeros(BATCH, TIME, 1, dtype=torch.uint8),
    },
    slice(None),
    slice(0, TIME),
)

mac = BCRBCMAC(batch.scheme, groups, args)
diag = mac.z_diagnostics(batch, full_obs)

assert "diag/z_reconstruction_error" in diag
assert "diag/correction_improvement" in diag
assert "diag/msg_recovery_error" in diag, "comm enabled -> message recovery metric expected"
for k, v in diag.items():
    assert v == v and abs(v) < 1e6, f"{k} must be finite, got {v}"
assert diag["diag/z_reconstruction_error"] >= 0.0

print("diag:", {k: round(v, 4) for k, v in diag.items()})
print("bcrbc z diagnostics ok")
