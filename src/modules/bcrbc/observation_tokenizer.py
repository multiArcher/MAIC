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
    ):
        super().__init__()
        self.num_z_tokens = num_z_tokens
        self.observation_projection = nn.Linear(observation_dim, model_hidden_dim)
        self.latent_tokens = nn.Parameter(torch.randn(num_z_tokens, model_hidden_dim) * 0.02)
        self.transformer = BlockCasualTransformer(
            dim=model_hidden_dim,
            depth=depth,
            heads=heads,
            kv_heads=heads,
            dim_head=model_hidden_dim // heads,
            time_block_every=depth + 1,
            num_special_tokens=num_z_tokens,
        )
        self.to_z = nn.Linear(model_hidden_dim, z_dim)

    def forward(self, observations):
        batch_size, time_steps, num_agents = observations.shape[:3]
        observation_tokens = self.observation_projection(observations).unsqueeze(-2)
        latent_tokens = self.latent_tokens.view(1, 1, 1, self.num_z_tokens, -1).expand(
            batch_size, time_steps, num_agents, -1, -1
        )
        tokens = torch.cat([observation_tokens, latent_tokens], dim=-2)
        encoder_outputs = self.transformer(tokens)
        return torch.tanh(self.to_z(encoder_outputs[..., -self.num_z_tokens :, :]))


class ObservationDecoder(nn.Module):
    def __init__(
        self,
        observation_dim,
        model_hidden_dim,
        z_dim,
        num_z_tokens,
        depth,
        heads,
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
            time_block_every=depth + 1,
            num_special_tokens=num_z_tokens,
            is_decoder=True,
        )
        self.to_observation = nn.Linear(model_hidden_dim, observation_dim)

    def forward(self, z):
        batch_size, time_steps, num_agents = z.shape[:3]
        observation_queries = self.observation_query.view(1, 1, 1, 1, -1).expand(
            batch_size, time_steps, num_agents, -1, -1
        )
        tokens = torch.cat([observation_queries, self.z_projection(z)], dim=-2)
        decoder_outputs = self.transformer(tokens)
        return self.to_observation(decoder_outputs[..., 0, :])
