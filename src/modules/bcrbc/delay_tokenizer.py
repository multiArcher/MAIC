import torch
import torch.nn as nn


class DelayTokenizer(nn.Module):
    """Tokenize observations, actions, messages, and per-agent query tokens."""

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
        d_model: int,
        max_t: int,
        max_delay: int = 16,
        n_msg_per_agent: int | None = None,
        d_msg: int = None,
        n_latent_tokens: int = 1,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.n_msg_per_agent = n_msg_per_agent if n_msg_per_agent is not None else (n_agents - 1)
        self.n_latent_tokens = n_latent_tokens
        self.d_msg = d_msg if d_msg is not None else d_model
        # Per agent: n_latent_tokens obs-latents + 1 action + n_msg messages.
        self.tokens_per_agent = self.n_latent_tokens + 1 + self.n_msg_per_agent
        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.action_proj = nn.Linear(n_actions, d_model)
        if self.d_msg != d_model:
            self.msg_proj = nn.Linear(self.d_msg, d_model)
        else:
            self.msg_proj = None
        self.query_token = nn.Parameter(torch.zeros(n_agents, d_model))
        self.agent_embed = nn.Embedding(n_agents, d_model)
        # Fixed 4-row modality table (obs/action/msg/query). All message slots
        # share the MSG row; sender identity is carried by agent_embed. All
        # n_latent_tokens obs-latents share the OBS row. Size is independent of
        # n_agents, so there is no index collision or out-of-range lookup.
        self.type_embed = nn.Embedding(self.NUM_MODALITIES, d_model)
        self.time_embed = nn.Embedding(max_t, d_model)
        self.generation_time_embed = nn.Embedding(max_t, d_model)
        self.arrival_time_embed = nn.Embedding(max_t, d_model)
        self.delay_embed = nn.Embedding(max_delay + 1, d_model)
        self.fresh_embed = nn.Embedding(2, d_model)
        self.max_delay = max_delay
        self.max_t = max_t
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.query_token, mean=0.0, std=0.02)

    @property
    def query_indices(self) -> torch.Tensor:
        S = self.n_agents * self.tokens_per_agent + self.n_agents
        return torch.arange(S - self.n_agents, S, dtype=torch.long)

    @property
    def agent_slice(self) -> slice:
        S = self.n_agents * self.tokens_per_agent + self.n_agents
        return slice(S - self.n_agents, S)

    def forward(self, obs, last_actions, start_t=0, obs_delay=None, obs_gen_t=None,
                obs_fresh_mask=None, messages=None, msg_gen_t=None, msg_arrive_t=None,
                msg_delay=None, msg_fresh_mask=None):
        # obs may be [B,T,n,z_dim] (num_token=1, legacy) or [B,T,n,num_token,z_dim].
        if obs.dim() == 4:
            obs = obs.unsqueeze(3)
        batch_size, time_size, n_agents, n_lat, _ = obs.shape
        assert n_lat == self.n_latent_tokens, (
            f"obs has {n_lat} latent tokens, expected {self.n_latent_tokens}"
        )
        device = obs.device
        obs_tokens = self.obs_proj(obs)  # [B,T,n,num_token,d_model]
        action_tokens = self.action_proj(last_actions.float()).unsqueeze(3)  # [B,T,n,1,d_model]
        if messages is not None:
            msg_flat = messages.reshape(batch_size * time_size * n_agents * self.n_msg_per_agent, self.d_msg)
            if self.msg_proj is not None:
                msg_tokens_flat = self.msg_proj(msg_flat)
            else:
                msg_tokens_flat = msg_flat
            msg_tokens = msg_tokens_flat.reshape(batch_size, time_size, n_agents, self.n_msg_per_agent, self.d_model)
        else:
            msg_tokens = torch.zeros(
                batch_size, time_size, n_agents, self.n_msg_per_agent, self.d_model,
                device=device, dtype=obs_tokens.dtype
            )
        query_tokens = self.query_token.view(1, 1, n_agents, -1).expand(batch_size, time_size, -1, -1)
        # Per-agent content order: [obs_0..obs_{nt-1}, action, msg_0..msg_{nm-1}].
        content_tokens = torch.cat([obs_tokens, action_tokens, msg_tokens], dim=3)
        content_flat = content_tokens.reshape(batch_size, time_size, n_agents * self.tokens_per_agent, self.d_model)
        query_flat = query_tokens.reshape(batch_size, time_size, n_agents, self.d_model)
        tokens = torch.cat([content_flat, query_flat], dim=2)
        time_ids = torch.arange(start_t, start_t + time_size, device=device).clamp(max=self.time_embed.num_embeddings - 1)
        time_emb = self.time_embed(time_ids).view(1, time_size, -1)
        S = tokens.shape[2]
        n_lat = self.n_latent_tokens
        msg_type_emb = self.type_embed(torch.tensor(self.MSG_TOKEN, device=device))
        for agent_idx in range(n_agents):
            agent_emb = self.agent_embed(torch.tensor(agent_idx, device=device))
            content_start = agent_idx * self.tokens_per_agent
            for lat_idx in range(n_lat):
                obs_idx = content_start + lat_idx
                tokens[:, :, obs_idx] = (tokens[:, :, obs_idx] + agent_emb
                    + self.type_embed(torch.tensor(self.OBS_TOKEN, device=device)) + time_emb)
            action_idx = content_start + n_lat
            tokens[:, :, action_idx] = (tokens[:, :, action_idx] + agent_emb
                + self.type_embed(torch.tensor(self.ACTION_TOKEN, device=device)) + time_emb)
            for msg_idx in range(self.n_msg_per_agent):
                token_pos = content_start + n_lat + 1 + msg_idx
                tokens[:, :, token_pos] = (tokens[:, :, token_pos] + agent_emb
                    + msg_type_emb + time_emb)
            query_idx = S - n_agents + agent_idx
            tokens[:, :, query_idx] = (tokens[:, :, query_idx] + agent_emb
                + self.type_embed(torch.tensor(self.QUERY_TOKEN, device=device)) + time_emb)
        if obs_delay is None:
            obs_delay = torch.zeros(batch_size, time_size, n_agents, 1, dtype=torch.long, device=device)
        if obs_gen_t is None:
            obs_gen_t = time_ids.view(1, time_size, 1, 1).expand(batch_size, -1, n_agents, -1)
        if obs_fresh_mask is None:
            obs_fresh_mask = torch.ones(batch_size, time_size, n_agents, 1, dtype=torch.float32, device=device)
        delay_ids = obs_delay.long().squeeze(-1).clamp(min=0, max=self.max_delay)
        gen_ids = obs_gen_t.long().squeeze(-1).clamp(min=0, max=self.time_embed.num_embeddings - 1)
        arrive_ids = (gen_ids + delay_ids).clamp(min=0, max=self.time_embed.num_embeddings - 1)
        fresh_ids = (obs_fresh_mask.squeeze(-1) > 0).long()
        for agent_idx in range(n_agents):
            # Same obs metadata applies to all n_latent_tokens obs slots of the agent.
            obs_meta = (self.delay_embed(delay_ids[:, :, agent_idx])
                + self.generation_time_embed(gen_ids[:, :, agent_idx]) + self.arrival_time_embed(arrive_ids[:, :, agent_idx])
                + self.fresh_embed(fresh_ids[:, :, agent_idx]))
            for lat_idx in range(n_lat):
                obs_idx = agent_idx * self.tokens_per_agent + lat_idx
                tokens[:, :, obs_idx] = tokens[:, :, obs_idx] + obs_meta
        if msg_gen_t is not None or msg_fresh_mask is not None:
            if msg_delay is None:
                msg_delay = torch.zeros(batch_size, time_size, n_agents, self.n_msg_per_agent, 1, dtype=torch.long, device=device)
            if msg_gen_t is None:
                msg_gen_t = time_ids.view(1, time_size, 1, 1, 1).expand(batch_size, -1, n_agents, self.n_msg_per_agent, -1)
            if msg_arrive_t is None:
                msg_arrive_t = msg_gen_t + msg_delay.clamp(min=0)
            if msg_fresh_mask is None:
                msg_fresh_mask = torch.ones(batch_size, time_size, n_agents, self.n_msg_per_agent, 1, dtype=torch.float32, device=device)
            msg_delay_ids = msg_delay.long().squeeze(-1).clamp(min=0, max=self.max_delay)
            msg_gen_ids = msg_gen_t.long().squeeze(-1).clamp(min=0, max=self.time_embed.num_embeddings - 1)
            msg_arrive_ids = msg_arrive_t.long().squeeze(-1).clamp(min=0, max=self.time_embed.num_embeddings - 1)
            msg_fresh_ids = (msg_fresh_mask.squeeze(-1) > 0).long()
            for agent_idx in range(n_agents):
                for msg_idx in range(self.n_msg_per_agent):
                    token_pos = agent_idx * self.tokens_per_agent + n_lat + 1 + msg_idx
                    tokens[:, :, token_pos] = (tokens[:, :, token_pos]
                        + self.delay_embed(msg_delay_ids[:, :, agent_idx, msg_idx])
                        + self.generation_time_embed(msg_gen_ids[:, :, agent_idx, msg_idx])
                        + self.arrival_time_embed(msg_arrive_ids[:, :, agent_idx, msg_idx])
                        + self.fresh_embed(msg_fresh_ids[:, :, agent_idx, msg_idx]))
        else:
            zero_fresh = self.fresh_embed(torch.tensor(0, device=device))
            for agent_idx in range(n_agents):
                for msg_idx in range(self.n_msg_per_agent):
                    token_pos = agent_idx * self.tokens_per_agent + n_lat + 1 + msg_idx
                    tokens[:, :, token_pos] = tokens[:, :, token_pos] + zero_fresh
        return self.norm(tokens)
