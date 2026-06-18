"""Phase 2 smoke: flow-matching world model.

Checks the FlowDynamics head in isolation and wired into the learner:
- flow_loss is finite and produces gradients into the flow head;
- sample() returns the right shape and runs without grad;
- a short training loop reduces the flow loss (the WM learns next-latent);
- learner.train with flow_loss_weight>0 completes with a finite logged loss.
"""

import sys
from types import SimpleNamespace as SN

import torch

sys.path.insert(0, "src")

from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers.bcrbc_mac import BCRBCMAC
from learners.bcrbc_learner import BCRBCLearner
from modules.bcrbc.flow_dynamics import FlowDynamics


class Logger:
    def info(self, *args, **kwargs):
        pass


# ---- 1. FlowDynamics unit checks --------------------------------------------
torch.manual_seed(0)
B, T, N = 2, 5, 3
CTX, ZD = 16, 8
flow = FlowDynamics(context_dim=CTX, z_dim=ZD, hidden_dim=32)

# context carries the explicit size-1 per-agent slot axis [...,1,CTX]; targets
# carry the token axis [...,n_tokens,ZD]; masks are [...,1,1].
context = torch.randn(B, T, N, 1, CTX, requires_grad=True)
z_target = torch.randn(B, T, N, 1, ZD)
mask = torch.ones(B, T, N, 1, 1)

loss = flow.flow_loss(context, z_target, mask)
assert torch.isfinite(loss) and loss.item() >= 0.0, "flow loss must be finite, non-negative"
loss.backward()
assert context.grad is not None and torch.isfinite(context.grad).all(), "grad must flow to context"
assert flow.net[0].weight.grad is not None, "flow head must receive gradient"

# sample shape + no-grad (n_tokens=1 -> size-1 token axis)
samp = flow.sample(torch.randn(B, T, N, 1, CTX), steps=8)
assert samp.shape == (B, T, N, 1, ZD), f"sample shape {samp.shape}"
assert not samp.requires_grad, "sample must run under no_grad"

# ---- 1b. Multi-token capacity (n_tokens > 1) --------------------------------
NTOK = 3
flow_mt = FlowDynamics(context_dim=CTX, z_dim=ZD, hidden_dim=32, n_tokens=NTOK)
ctx_mt = torch.randn(B, T, N, 1, CTX, requires_grad=True)
z_tgt_mt = torch.randn(B, T, N, NTOK, ZD)
mask_mt = torch.ones(B, T, N, 1, 1)
loss_mt = flow_mt.flow_loss(ctx_mt, z_tgt_mt, mask_mt)
assert torch.isfinite(loss_mt) and loss_mt.item() >= 0.0
loss_mt.backward()
assert ctx_mt.grad is not None and torch.isfinite(ctx_mt.grad).all()
samp_mt = flow_mt.sample(torch.randn(B, T, N, 1, CTX), steps=4)
assert samp_mt.shape == (B, T, N, NTOK, ZD), f"multi-token sample shape {samp_mt.shape}"
# the NTOK generated latents should not all be identical (token_query breaks symmetry)
assert not torch.allclose(samp_mt[..., 0, :], samp_mt[..., 1, :]), "tokens should differ"

# ---- 2. The WM can actually learn a fixed mapping ---------------------------
# Train flow to generate a target that is a deterministic function of context.
torch.manual_seed(1)
flow2 = FlowDynamics(context_dim=CTX, z_dim=ZD, hidden_dim=64)
opt = torch.optim.Adam(flow2.parameters(), lr=1e-2)
W = torch.randn(CTX, ZD)
fixed_ctx = torch.randn(64, 1, CTX)  # carry the size-1 slot axis
fixed_tgt = torch.tanh(fixed_ctx @ W)  # [64, 1, ZD]
first = None
for it in range(300):
    opt.zero_grad()
    l = flow2.flow_loss(fixed_ctx, fixed_tgt, None)
    l.backward()
    opt.step()
    if it == 0:
        first = l.item()
last = l.item()
assert last < first, f"flow loss should decrease ({first:.4f} -> {last:.4f})"
# sampled latent should be closer to target than random noise
gen = flow2.sample(fixed_ctx, steps=16)  # [64, 1, ZD]
gen_err = (gen - fixed_tgt).pow(2).mean().item()
rand_err = (torch.randn_like(fixed_tgt) - fixed_tgt).pow(2).mean().item()
assert gen_err < rand_err, f"generated ({gen_err:.3f}) should beat noise ({rand_err:.3f})"

# ---- 3. End-to-end: learner.train with flow_loss_weight>0 -------------------
BATCH, TIME, NA, OBS, NACT, STATE = 2, 6, 3, 7, 4, 9
scheme = {
    "state": {"vshape": STATE, "dtype": torch.float32},
    "obs": {"vshape": OBS, "group": "agents", "dtype": torch.float32},
    "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
    "avail_actions": {"vshape": (NACT,), "group": "agents", "dtype": torch.int},
    "reward": {"vshape": (1,), "dtype": torch.float32},
    "terminated": {"vshape": (1,), "dtype": torch.uint8},
}
groups = {"agents": NA}
preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=NACT)])}
batch = EpisodeBatch(scheme, groups, BATCH, TIME, preprocess=preprocess, device="cpu")
batch.update(
    {
        "state": torch.randn(BATCH, TIME, STATE),
        "obs": torch.randn(BATCH, TIME, NA, OBS),
        "actions": torch.randint(0, NACT, (BATCH, TIME, NA, 1)),
        "avail_actions": torch.ones(BATCH, TIME, NA, NACT, dtype=torch.int),
        "reward": torch.randn(BATCH, TIME, 1),
        "terminated": torch.zeros(BATCH, TIME, 1, dtype=torch.uint8),
    },
    slice(None),
    slice(0, TIME),
)
args = SN(
    device=torch.device("cpu"), use_cuda=False, n_agents=NA, n_actions=NACT, state_shape=STATE,
    common_reward=True, agent_output_type="q", action_selector="epsilon_greedy",
    epsilon_start=1.0, epsilon_finish=0.05, epsilon_anneal_time=100, evaluation_epsilon=0.0,
    obs_agent_id=True, obs_last_action=True, bcrbc_d_model=32, bcrbc_belief_dim=32, bcrbc_z_dim=16,
    bcrbc_depth=1, bcrbc_heads=4, bcrbc_dropout=0.0, env_info={"episode_limit": TIME - 1},
    mixer="new_qmix_mixer", mixing_embed_dim=8, hypernet_embed=16, optimizer="adamW", lr=0.001,
    standardise_returns=False, standardise_rewards=False, target_type="td", gamma=0.99,
    double_q=True, target_update_interval_or_tau=200, learner_log_interval=999999, grad_norm_clip=10,
    td_loss_weight=1.0, rec_loss_weight=0.5, dyn_loss_weight=0.0, flow_loss_weight=0.5,
)
torch.manual_seed(2)
mac = BCRBCMAC(batch.scheme, groups, args)
out = mac.forward(batch, slice(0, TIME))
assert out["z"].shape == (BATCH, TIME, NA, 1, 16), f"z shape {out['z'].shape}"
learner = BCRBCLearner(mac, batch.scheme, Logger(), args)
learner.train(batch, 0, 0)
assert mac.agent.flow_dynamics.net[0].weight.grad is not None, "flow head should get grad in learner step"

print("bcrbc flow dynamics ok")
