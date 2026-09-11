import torch
import torch.nn as nn


class DynamicsTokenizer(nn.Module):
    """Build local action, signal, latent and agent-query token blocks."""

    ACTION_TOKEN = 0
    SIGNAL_TOKEN = 1
    Z_TOKEN = 2
    AGENT_TOKEN = 3
    NUM_MODALITIES = 4

    def __init__(
        self,
        z_dim: int,
        n_actions: int,
        n_agents: int,
        model_hidden_dim: int,
        max_t: int,
        num_z_tokens: int,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.num_z_tokens = num_z_tokens
        self.tokens_per_agent = 2 + self.num_z_tokens
        self.action_projection = nn.Linear(n_actions, model_hidden_dim)
        self.signal_projection = nn.Linear(1, model_hidden_dim)
        self.z_projection = nn.Linear(z_dim, model_hidden_dim)
        self.agent_tokens = nn.Parameter(torch.zeros(n_agents, model_hidden_dim))
        self.agent_embedding = nn.Embedding(n_agents, model_hidden_dim)
        self.type_embed = nn.Embedding(self.NUM_MODALITIES, model_hidden_dim)
        token_type_ids = torch.tensor(
            [self.ACTION_TOKEN]
            + [self.SIGNAL_TOKEN]
            + [self.Z_TOKEN] * self.num_z_tokens
            + [self.AGENT_TOKEN],
            dtype=torch.long,
        )
        self.register_buffer("token_type_ids", token_type_ids, persistent=False)
        self.time_embed = nn.Embedding(max_t, model_hidden_dim)
        self.max_t = max_t
        self.model_hidden_dim = model_hidden_dim
        self.norm = nn.LayerNorm(model_hidden_dim)
        nn.init.normal_(self.agent_tokens, mean=0.0, std=0.02)

    @property
    def z_slice(self) -> slice:
        return slice(2, 2 + self.num_z_tokens)

    @property
    def query_index(self) -> int:
        """Index of the query token inside every agent's local token block."""
        return self.tokens_per_agent

    @property
    def query_slice(self) -> slice:
        return slice(self.query_index, self.query_index + 1)

    def forward(self, noisy_z, previous_actions, signal_levels, start_t=0):
        batch_size, time_steps, n_agents = noisy_z.shape[:3]
        device = noisy_z.device
        action_tokens = self.action_projection(previous_actions.float()).unsqueeze(-2)
        signal_tokens = self.signal_projection(signal_levels[..., 0, :]).unsqueeze(-2)
        z_tokens = self.z_projection(noisy_z)
        agent_tokens = self.agent_tokens.view(1, 1, n_agents, 1, -1).expand(
            batch_size, time_steps, -1, -1, -1
        )
        tokens = torch.cat(
            [action_tokens, signal_tokens, z_tokens, agent_tokens],
            dim=-2,
        )

        time_ids = torch.arange(start_t, start_t + time_steps, device=device)
        time_embeddings = self.time_embed(time_ids)[None, :, None, None, :]  # [1, T, 1, 1, D]

        token_type_embeddings = self.type_embed(self.token_type_ids)  # [A, D]

        agent_embeddings = self.agent_embedding(torch.arange(n_agents, device=device))[:, None, :]

        tokens = tokens + token_type_embeddings + agent_embeddings + time_embeddings
        return self.norm(tokens)
