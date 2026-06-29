import torch
import torch.nn as nn


class DelayTokenizer(nn.Module):
    """Tokenize observation latents, actions, messages, and per-agent query tokens.

    Delay never enters the model as input: training is no-delay and at eval the
    generative rollout presents timely latents, so tokens carry only content +
    agent id + modality type + absolute time. Delayed observations/messages are
    resolved in controller code (latent imputation + correction-on-arrival), not
    by a delay/freshness embedding. Message payloads are sender observations whose
    staleness, when comm delay is active at eval, lives in the content itself.
    """

    OBS_TOKEN = 0
    ACTION_TOKEN = 1
    MSG_TOKEN = 2
    QUERY_TOKEN = 3
    NUM_MODALITIES = 4  # obs, action, message, query (fixed; independent of n_agents)

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_agents: int,
        model_hidden_dim: int,
        max_t: int,
        num_messages_per_agent: int,
        message_dim: int,
        num_latent_tokens: int,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.num_messages_per_agent = num_messages_per_agent
        self.num_latent_tokens = num_latent_tokens
        self.message_dim = message_dim
        # Per agent: num_latent_tokens obs-latents + 1 action + num_messages messages.
        self.tokens_per_agent = self.num_latent_tokens + 1 + self.num_messages_per_agent
        self.obs_proj = nn.Linear(obs_dim, model_hidden_dim)
        self.action_proj = nn.Linear(n_actions, model_hidden_dim)
        # Always project messages into the model space (single code path). Message
        # payloads are raw sender observations of dim message_dim.
        self.msg_proj = nn.Linear(message_dim, model_hidden_dim)
        self.query_token = nn.Parameter(torch.zeros(n_agents, model_hidden_dim))
        self.agent_embed = nn.Embedding(n_agents, model_hidden_dim)
        # Fixed 4-row modality table (obs/action/msg/query). All message slots
        # share the MSG row; sender identity is carried by agent_embed. All
        # num_latent_tokens obs-latents share the OBS row. Size is independent of
        # n_agents, so there is no index collision or out-of-range lookup.
        self.type_embed = nn.Embedding(self.NUM_MODALITIES, model_hidden_dim)
        self.time_embed = nn.Embedding(max_t, model_hidden_dim)
        self.max_t = max_t
        self.model_hidden_dim = model_hidden_dim
        self.norm = nn.LayerNorm(model_hidden_dim)
        nn.init.normal_(self.query_token, mean=0.0, std=0.02)

    @property
    def query_indices(self) -> torch.Tensor:
        sequence_length = self.n_agents * self.tokens_per_agent + self.n_agents
        return torch.arange(sequence_length - self.n_agents, sequence_length, dtype=torch.long)

    @property
    def agent_slice(self) -> slice:
        sequence_length = self.n_agents * self.tokens_per_agent + self.n_agents
        return slice(sequence_length - self.n_agents, sequence_length)

    def forward(self, obs, last_actions, start_t=0, messages=None):
        # obs is the per-agent bottleneck latents [B, T, n, num_latent_tokens, latent_dim].
        batch_size, time_steps, n_agents, num_latent_tokens, _ = obs.shape
        assert num_latent_tokens == self.num_latent_tokens, (
            f"obs has {num_latent_tokens} latent tokens, expected {self.num_latent_tokens}"
        )
        device = obs.device
        obs_tokens = self.obs_proj(obs)  # [B,T,n,num_latent_tokens,model_hidden_dim]
        action_tokens = self.action_proj(last_actions.float()).unsqueeze(3)  # [B,T,n,1,model_hidden_dim]
        # Message slots: filled when the comm pathway supplies sender observations,
        # otherwise zero (comm-off config). Staleness, if any, is in the content.
        if messages is not None:
            msg_flat = messages.reshape(
                batch_size * time_steps * n_agents * self.num_messages_per_agent, self.message_dim
            )
            msg_tokens_flat = self.msg_proj(msg_flat)
            msg_tokens = msg_tokens_flat.reshape(
                batch_size, time_steps, n_agents, self.num_messages_per_agent, self.model_hidden_dim
            )
        else:
            msg_tokens = torch.zeros(
                batch_size, time_steps, n_agents, self.num_messages_per_agent, self.model_hidden_dim,
                device=device, dtype=obs_tokens.dtype
            )
        query_tokens = self.query_token.view(1, 1, n_agents, -1).expand(batch_size, time_steps, -1, -1)
        # Per-agent content order: [obs_0..obs_{nt-1}, action, msg_0..msg_{nm-1}].
        content_tokens = torch.cat([obs_tokens, action_tokens, msg_tokens], dim=3)
        content_flat = content_tokens.reshape(
            batch_size, time_steps, n_agents * self.tokens_per_agent, self.model_hidden_dim
        )
        query_flat = query_tokens.reshape(batch_size, time_steps, n_agents, self.model_hidden_dim)
        tokens = torch.cat([content_flat, query_flat], dim=2)
        time_ids = torch.arange(start_t, start_t + time_steps, device=device).clamp(max=self.time_embed.num_embeddings - 1)
        time_emb = self.time_embed(time_ids).view(1, time_steps, -1)
        sequence_length = tokens.shape[2]
        msg_type_emb = self.type_embed(torch.tensor(self.MSG_TOKEN, device=device))
        # Add per-token identity embeddings (agent id + modality type + absolute time)
        # to every content/query slot, following the fixed per-agent token layout.
        for agent_idx in range(n_agents):
            agent_emb = self.agent_embed(torch.tensor(agent_idx, device=device))
            content_start = agent_idx * self.tokens_per_agent
            for latent_idx in range(num_latent_tokens):
                obs_idx = content_start + latent_idx
                tokens[:, :, obs_idx] = (tokens[:, :, obs_idx] + agent_emb
                    + self.type_embed(torch.tensor(self.OBS_TOKEN, device=device)) + time_emb)
            action_idx = content_start + num_latent_tokens
            tokens[:, :, action_idx] = (tokens[:, :, action_idx] + agent_emb
                + self.type_embed(torch.tensor(self.ACTION_TOKEN, device=device)) + time_emb)
            for msg_idx in range(self.num_messages_per_agent):
                token_pos = content_start + num_latent_tokens + 1 + msg_idx
                tokens[:, :, token_pos] = (tokens[:, :, token_pos] + agent_emb
                    + msg_type_emb + time_emb)
            query_idx = sequence_length - n_agents + agent_idx
            tokens[:, :, query_idx] = (tokens[:, :, query_idx] + agent_emb
                + self.type_embed(torch.tensor(self.QUERY_TOKEN, device=device)) + time_emb)
        return self.norm(tokens)
