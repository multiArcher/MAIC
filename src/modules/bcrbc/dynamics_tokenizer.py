import torch
import torch.nn as nn


class DynamicsTokenizer(nn.Module):
    """Tokenize noisy z, previous actions, messages, and agent query tokens.

    Delay never enters the model as input: training is no-delay and at eval the
    generative rollout presents timely latents, so tokens carry only content +
    agent id + modality type + absolute time. Delayed observations/messages are
    resolved in controller code (latent imputation + correction-on-arrival), not
    by a delay/freshness embedding. Message payloads are sender observations whose
    staleness, when comm delay is active at eval, lives in the content itself.
    """

    ACTION_TOKEN = 0
    SIGNAL_TOKEN = 1
    Z_TOKEN = 2
    MSG_TOKEN = 3
    AGENT_TOKEN = 4
    NUM_MODALITIES = 5

    def __init__(
        self,
        z_dim: int,
        n_actions: int,
        n_agents: int,
        model_hidden_dim: int,
        max_t: int,
        num_messages_per_agent: int,
        message_dim: int,
        num_z_tokens: int,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.num_messages_per_agent = num_messages_per_agent
        self.num_z_tokens = num_z_tokens
        self.message_dim = message_dim
        self.tokens_per_agent = 2 + self.num_z_tokens + self.num_messages_per_agent
        self.action_projection = nn.Linear(n_actions, model_hidden_dim)
        self.signal_projection = nn.Linear(1, model_hidden_dim)
        self.z_projection = nn.Linear(z_dim, model_hidden_dim)
        # Always project messages into the model space (single code path). Message
        # payloads are raw sender observations of dim message_dim.
        self.message_projection = nn.Linear(message_dim, model_hidden_dim)
        self.agent_tokens = nn.Parameter(torch.zeros(n_agents, model_hidden_dim))
        self.agent_embedding = nn.Embedding(n_agents, model_hidden_dim)
        # The modality table has one row per token type. Message slots share
        # the MSG row and z slots share the Z row. Its size is independent of
        # n_agents, so there is no index collision or out-of-range lookup.
        self.type_embed = nn.Embedding(self.NUM_MODALITIES, model_hidden_dim)
        token_type_ids = torch.tensor(
            [self.ACTION_TOKEN]
            + [self.SIGNAL_TOKEN]
            + [self.Z_TOKEN] * self.num_z_tokens
            + [self.MSG_TOKEN] * self.num_messages_per_agent
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

    def forward(self, noisy_z, previous_actions, signal_levels, start_t=0, messages=None):
        batch_size, time_steps, n_agents = noisy_z.shape[:3]
        device = noisy_z.device
        action_tokens = self.action_projection(previous_actions.float()).unsqueeze(-2)
        signal_tokens = self.signal_projection(signal_levels[..., 0, :]).unsqueeze(-2)
        z_tokens = self.z_projection(noisy_z)
        # Message slots: filled when the comm pathway supplies sender observations,
        # otherwise zero (comm-off config). Staleness, if any, is in the content.
        if messages is not None:
            message_tokens = self.message_projection(messages)
        else:
            message_tokens = torch.zeros(
                batch_size, time_steps, n_agents, self.num_messages_per_agent, self.model_hidden_dim,
                device=device, dtype=z_tokens.dtype
            )
        agent_tokens = self.agent_tokens.view(1, 1, n_agents, 1, -1).expand(
            batch_size, time_steps, -1, -1, -1
        )
        tokens = torch.cat(
            [action_tokens, signal_tokens, z_tokens, message_tokens, agent_tokens],
            dim=-2,
        )

        time_ids = torch.arange(start_t, start_t + time_steps, device=device)
        time_embeddings = self.time_embed(time_ids)[None, :, None, None, :]  # [1, T, 1, 1, D]

        token_type_embeddings = self.type_embed(self.token_type_ids)  # [A, D]

        agent_embeddings = self.agent_embedding(torch.arange(n_agents, device=device))[:, None, :]

        tokens = tokens + token_type_embeddings + agent_embeddings + time_embeddings
        return self.norm(tokens)
