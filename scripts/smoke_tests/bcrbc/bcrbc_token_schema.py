"""Dynamics token layout and CTDE isolation smoke test."""

import sys
from pathlib import Path

import torch

repo_root = Path(__file__).parents[3]
sys.path.insert(0, str(repo_root / "src"))

from modules.bcrbc.agent_readout import AgentReadout
from modules.bcrbc.block_builder import BlockBuilder
from modules.bcrbc.block_causal_transformer import BlockCausalTransformer
from modules.bcrbc.dynamics_tokenizer import DynamicsTokenizer


batch_size, time_steps, n_agents = 2, 4, 3
num_z_tokens, z_dim = 2, 5
n_actions, model_hidden_dim = 4, 32
agent_output_dim = 24

tokenizer = DynamicsTokenizer(
    z_dim=z_dim,
    n_actions=n_actions,
    n_agents=n_agents,
    model_hidden_dim=model_hidden_dim,
    max_t=time_steps + 10,
    num_messages_per_agent=n_agents - 1,
    message_dim=16,
    num_z_tokens=num_z_tokens,
)
block_builder = BlockBuilder(tokenizer)

expected_type_ids = torch.tensor(
    [
        tokenizer.ACTION_TOKEN,
        tokenizer.SIGNAL_TOKEN,
        tokenizer.Z_TOKEN,
        tokenizer.Z_TOKEN,
        tokenizer.MSG_TOKEN,
        tokenizer.MSG_TOKEN,
        tokenizer.AGENT_TOKEN,
    ]
)
assert torch.equal(tokenizer.token_type_ids.cpu(), expected_type_ids)
assert block_builder.z_slice == slice(2, 4)
assert block_builder.query_slice == slice(6, 7)

noisy_z = torch.randn(
    batch_size,
    time_steps,
    n_agents,
    num_z_tokens,
    z_dim,
)
previous_actions = torch.nn.functional.one_hot(
    torch.randint(n_actions, (batch_size, time_steps, n_agents)),
    n_actions,
).float()
signal_levels = torch.rand(batch_size, time_steps, n_agents, 1, 1)
messages = torch.randn(batch_size, time_steps, n_agents, n_agents - 1, 16)

tokens = block_builder(
    noisy_z,
    previous_actions,
    signal_levels,
    messages=messages,
)
assert tokens.shape == (
    batch_size,
    time_steps,
    n_agents,
    len(expected_type_ids),
    model_hidden_dim,
)
assert torch.isfinite(tokens).all()

transformer = BlockCausalTransformer(
    model_hidden_dim=model_hidden_dim,
    num_transformer_layers=2,
    num_attention_heads=4,
    dropout=0.0,
    agent_slice=block_builder.query_slice,
)
readout = AgentReadout(model_hidden_dim, agent_output_dim)

with torch.no_grad():
    transformer_outputs = transformer(tokens)
    agent_outputs = readout(transformer_outputs, block_builder.query_slice)

assert transformer_outputs.shape == tokens.shape
assert agent_outputs.shape == (
    batch_size,
    time_steps,
    n_agents,
    1,
    agent_output_dim,
)

changed_tokens = tokens.clone()
changed_tokens[:, :, 1] += torch.randn_like(changed_tokens[:, :, 1])
with torch.no_grad():
    unchanged_outputs = transformer(tokens)
    changed_outputs = transformer(changed_tokens)
assert torch.equal(unchanged_outputs[:, :, 0], changed_outputs[:, :, 0])

changed_agent_token = tokens.clone()
changed_agent_token[..., block_builder.query_slice, :] += 10.0
with torch.no_grad():
    original_z_outputs = transformer(tokens)[..., block_builder.z_slice, :]
    changed_z_outputs = transformer(changed_agent_token)[..., block_builder.z_slice, :]
assert torch.equal(original_z_outputs, changed_z_outputs)

print("bcrbc token schema ok")
