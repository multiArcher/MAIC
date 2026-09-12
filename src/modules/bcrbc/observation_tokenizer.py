import torch
import torch.nn as nn

from modules.bcrbc.layers import BlockCasualTransformer


class ObservationEncoder(nn.Module):
    def __init__(
        self,
        observation_dim,
        model_hidden_dim,
        z_dim,
        num_z_tokens,
        depth,
        heads,
        context_window=0,
    ):
        super().__init__()
        self.num_z_tokens = num_z_tokens
        self.observation_projection = nn.Linear(observation_dim, model_hidden_dim)
        self.latent_tokens = nn.Parameter(torch.randn(num_z_tokens, model_hidden_dim) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, model_hidden_dim) * 0.02)
        self.transformer = BlockCasualTransformer(
            dim=model_hidden_dim,
            depth=depth,
            heads=heads,
            kv_heads=heads,
            dim_head=model_hidden_dim // heads,
            time_block_every=4,
            context_window=context_window,
            num_special_tokens=num_z_tokens,
        )
        self.to_z = nn.Linear(model_hidden_dim, z_dim)

    def forward(self, observations, missing_mask=None, kv_cache=None,
                use_kv_cache=False, rope_offset=0):
        batch_size, time_steps, num_agents = observations.shape[:3]
        observation_tokens = self.observation_projection(observations).unsqueeze(-2)
        if missing_mask is not None:
            observation_tokens = torch.where(
                missing_mask, self.mask_token, observation_tokens
            )
        latent_tokens = self.latent_tokens.view(1, 1, 1, self.num_z_tokens, -1).expand(
            batch_size, time_steps, num_agents, -1, -1
        )
        tokens = torch.cat([observation_tokens, latent_tokens], dim=-2)
        encoder_outputs = self.transformer(
            tokens, kv_cache=kv_cache, use_kv_cache=use_kv_cache,
            rope_offset=rope_offset,
        )
        if use_kv_cache:
            encoder_outputs, new_cache = encoder_outputs
        latent_outputs = encoder_outputs[..., -self.num_z_tokens:, :]
        z = torch.tanh(self.to_z(latent_outputs))
        if use_kv_cache:
            return z, new_cache
        return z


class ObservationDecoder(nn.Module):
    def __init__(
        self,
        observation_dim,
        model_hidden_dim,
        z_dim,
        num_z_tokens,
        depth,
        heads,
        context_window=0,
    ):
        super().__init__()
        self.num_z_tokens = num_z_tokens
        self.z_projection = nn.Linear(z_dim, model_hidden_dim)
        self.observation_query = nn.Parameter(torch.randn(1, model_hidden_dim) * 0.02)
        self.transformer = BlockCasualTransformer(
            dim=model_hidden_dim,
            depth=depth,
            heads=heads,
            kv_heads=heads,
            dim_head=model_hidden_dim // heads,
            time_block_every=4,
            context_window=context_window,
            num_special_tokens=num_z_tokens,
            is_decoder=True,
        )
        self.to_observation = nn.Linear(model_hidden_dim, observation_dim)

    def forward(self, z, kv_cache=None, use_kv_cache=False, rope_offset=0,
                history_z=None):
        batch_size, time_steps, num_agents = z.shape[:3]
        observation_queries = self.observation_query.view(1, 1, 1, 1, -1).expand(
            batch_size, time_steps, num_agents, -1, -1
        )
        tokens = torch.cat([observation_queries, self.z_projection(z)], dim=-2)
        if history_z is not None:
            # Generated reconstruction uses the same fixed history definition.
            with torch.no_grad():
                history_tokens = torch.cat(
                    [observation_queries, self.z_projection(history_z)], dim=-2,
                )
                condition = self.transformer.prepare_condition(
                    history_tokens, rope_offset=rope_offset, detach=True,
                )
            decoder_outputs = self.transformer.query_condition(
                tokens, condition, rope_offset=rope_offset,
            )
        else:
            decoder_outputs = self.transformer(
                tokens, kv_cache=kv_cache, use_kv_cache=use_kv_cache,
                rope_offset=rope_offset,
            )
        if use_kv_cache:
            decoder_outputs, new_cache = decoder_outputs
        reconstructed_observations = self.to_observation(decoder_outputs[..., 0, :])
        if use_kv_cache:
            return reconstructed_observations, new_cache
        return reconstructed_observations
