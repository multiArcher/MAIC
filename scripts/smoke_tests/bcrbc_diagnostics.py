"""Phase 5 smoke: belief-quality diagnostics.

Builds a delayed episode + the matching full (un-delayed) observations and checks
that BCRBCMAC.belief_diagnostics returns finite, well-formed Tier-1 metrics:
latent reconstruction error over generated slots, correction-after-arrival
improvement, and (with comm on) delayed-message recovery error.
"""

import sys
from types import SimpleNamespace as SN

import torch

sys.path.insert(0, "src")

from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers.bcrbc_mac import BCRBCMAC

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

args = SN(
    device=torch.device("cpu"), use_cuda=False, n_agents=NA, n_actions=NACT, state_shape=STATE,
    common_reward=True, agent_output_type="q", action_selector="epsilon_greedy",
    epsilon_start=1.0, epsilon_finish=0.05, epsilon_anneal_time=100, evaluation_epsilon=0.0,
    obs_agent_id=True, obs_last_action=True, bcrbc_d_model=32, bcrbc_belief_dim=32, bcrbc_z_dim=16,
    bcrbc_depth=1, bcrbc_heads=4, bcrbc_dropout=0.0, env_info={"episode_limit": TIME - 1},
    mixer="new_qmix_mixer", mixing_embed_dim=8, hypernet_embed=16, optimizer="adamW", lr=0.001,
    standardise_returns=False, standardise_rewards=False, target_type="td", gamma=0.99,
    double_q=True, target_update_interval_or_tau=200, learner_log_interval=999999, grad_norm_clip=10,
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
diag = mac.belief_diagnostics(batch, full_obs)

assert "diag/latent_recon_error" in diag
assert "diag/correction_improvement" in diag
assert "diag/msg_recovery_error" in diag, "comm enabled -> message recovery metric expected"
for k, v in diag.items():
    assert v == v and abs(v) < 1e6, f"{k} must be finite, got {v}"
assert diag["diag/latent_recon_error"] >= 0.0, "recon error is a squared error, must be >= 0"

print("diag:", {k: round(v, 4) for k, v in diag.items()})
print("bcrbc belief diagnostics ok")
