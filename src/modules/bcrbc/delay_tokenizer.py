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

    @property
    def group_ids(self) -> torch.Tensor:
        """Per-token agent-group id over the space axis (length S).

        content of agent a -> a (tokens_per_agent entries), query of agent a -> a.
        Used to build the block-diagonal (CTDE) spatial attention mask: tokens with
        the same group id form one agent's block; cross-agent flow goes only through
        the message tokens (sender obs) that already live inside the receiver block.
        """
        agents = torch.arange(self.n_agents, dtype=torch.long)
        content = agents.repeat_interleave(self.tokens_per_agent)
        return torch.cat([content, agents])

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
        time_emb = self.time_embed(time_ids)  # [T, d]
        sequence_length = tokens.shape[2]

        # Build the per-position identity (agent id + modality type) as a structured
        # [S, model_hidden_dim] tensor, then add it (plus the absolute-time embedding)
        # to every slot with a single broadcast. No per-position scalar lookups, no
        # in-place per-agent writes: the per-agent token layout is encoded by the
        # repeat/concat structure instead of a Python loop.
        # Per-agent content type row: [OBS]*nt, ACTION, [MSG]*nm  -> [tokens_per_agent]
        content_type_ids = torch.tensor(
            [self.OBS_TOKEN] * num_latent_tokens + [self.ACTION_TOKEN]
            + [self.MSG_TOKEN] * self.num_messages_per_agent,
            device=device, dtype=torch.long,
        )
        content_type_emb = self.type_embed(content_type_ids)  # [tpa, d]
        agent_emb_all = self.agent_embed(torch.arange(n_agents, device=device))  # [n, d]
        # content identity [n, tpa, d] = agent id (broadcast over tpa) + modality type.
        content_identity = agent_emb_all.unsqueeze(1) + content_type_emb.unsqueeze(0)
        content_identity = content_identity.reshape(n_agents * self.tokens_per_agent, self.model_hidden_dim)
        query_identity = agent_emb_all + self.type_embed(
            torch.tensor(self.QUERY_TOKEN, device=device)
        )  # [n, d]
        identity = torch.cat([content_identity, query_identity], dim=0)  # [S, d]
        # tokens [B,T,S,d] + identity [1,1,S,d] + time [1,T,1,d] -> single broadcast add.
        tokens = tokens + identity.view(1, 1, sequence_length, self.model_hidden_dim) + time_emb.view(1, time_steps, 1, self.model_hidden_dim)
        return self.norm(tokens)
