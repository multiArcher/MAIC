"""DreamerV4-style BCRBC data-flow smoke test."""

import sys

import torch

sys.path.insert(0, "src")

from modules.bcrbc.bcrbc_model import BCRBCModel
from modules.bcrbc.losses import flow_matching_loss
from scripts.smoke_tests.bcrbc._bcrbc_args import make_bcrbc_args


torch.manual_seed(0)

BATCH, TIME, N_AGENTS = 2, 5, 3
OBS_DIM, N_ACTIONS = 7, 4
NUM_Z_TOKENS, Z_DIM = 3, 8

args = make_bcrbc_args(
    n_agents=N_AGENTS,
    n_actions=N_ACTIONS,
    bcrbc_z_dim=Z_DIM,
    bcrbc_num_z_tokens=NUM_Z_TOKENS,
    bcrbc_depth=4,
    env_info={"episode_limit": TIME - 1},
)
model = BCRBCModel(OBS_DIM, N_ACTIONS, N_AGENTS, args)

observations = torch.randn(BATCH, TIME, N_AGENTS, OBS_DIM)
previous_actions = torch.nn.functional.one_hot(
    torch.randint(N_ACTIONS, (BATCH, TIME, N_AGENTS)),
    N_ACTIONS,
).float()
messages = torch.randn(BATCH, TIME, N_AGENTS, N_AGENTS - 1, OBS_DIM)

z = model.encode_observations(observations)
assert z.shape == (BATCH, TIME, N_AGENTS, NUM_Z_TOKENS, Z_DIM)
assert z.abs().max() <= 1.0

reconstructed_observations = model.decode_observations(z)
assert reconstructed_observations.shape == observations.shape

signal_levels = torch.rand(BATCH, TIME, N_AGENTS, 1, 1)
noise = torch.randn_like(z)
noisy_z = (1.0 - signal_levels) * noise + signal_levels * z

output = model.predict_z(
    noisy_z,
    previous_actions,
    messages,
    signal_levels,
)
assert output["predicted_z"].shape == z.shape
assert output["agent_outputs"].shape == (
    BATCH,
    TIME,
    N_AGENTS,
    1,
    args.bcrbc_agent_output_dim,
)
assert output["q_values"].shape == (BATCH, TIME, N_AGENTS, 1, N_ACTIONS)
assert "beliefs" not in output
assert "latents" not in output
assert "pred_latents" not in output

mask = torch.ones(BATCH, TIME, N_AGENTS, 1, 1)
mask[:, 0] = 0
loss = flow_matching_loss(output["predicted_z"], z.detach(), mask)
assert torch.isfinite(loss)
changed_first_step = output["predicted_z"].detach().clone()
changed_first_step[:, 0] += 1000.0
assert torch.equal(
    flow_matching_loss(changed_first_step, z.detach(), mask),
    flow_matching_loss(output["predicted_z"].detach(), z.detach(), mask),
)
loss.backward()
assert model.z_predictor.weight.grad is not None
assert model.transformer.transformer.layers[0].attention.to_q.weight.grad is not None

with torch.no_grad():
    changed_actions = previous_actions.clone()
    changed_actions[:, 1] = changed_actions[:, 1].roll(1, dims=-1)
    changed = model.predict_z(noisy_z, changed_actions, messages, signal_levels)
assert not torch.allclose(output["predicted_z"][:, 1], changed["predicted_z"][:, 1])

with torch.no_grad():
    changed_future_actions = previous_actions.clone()
    changed_future_actions[:, 3:] = changed_future_actions[:, 3:].roll(1, dims=-1)
    changed_future = model.predict_z(
        noisy_z,
        changed_future_actions,
        messages,
        signal_levels,
    )
assert torch.allclose(
    output["predicted_z"][:, :3],
    changed_future["predicted_z"][:, :3],
    atol=1e-6,
)

sampled_z = model.sample_z(
    previous_actions[:, :1],
    messages[:, :1],
    steps=4,
)
assert sampled_z.shape == (BATCH, 1, N_AGENTS, NUM_Z_TOKENS, Z_DIM)
assert torch.isfinite(sampled_z).all()
assert not sampled_z.requires_grad

print("bcrbc flow dynamics ok")
